import csv
import hashlib
from html import escape
import json
import textwrap
import time
from io import StringIO

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="SIMVENT | EV Cooling Topology Solver",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_topology_contents(contents: str) -> tuple[np.ndarray, dict[int, str]]:
    if contents.lstrip().startswith(("{", "[")):
        data = json.loads(contents)
        if not isinstance(data, dict) or "adjacency_matrix" not in data:
            raise ValueError("JSON topology must contain an 'adjacency_matrix' field.")
        matrix_values = data["adjacency_matrix"]
        raw_labels = data.get("node_labels", {})
        if not isinstance(raw_labels, dict):
            raise ValueError("'node_labels' must be a JSON object.")
        labels = {int(key): str(value) for key, value in raw_labels.items()}
    else:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(StringIO(contents))
            if any(cell.strip() for cell in row)
        ]
        if not rows:
            raise ValueError("CSV file is empty.")

        header_keywords = {
            "source", "destination", "from", "to", "start", "end", "src",
            "dst", "node1", "node2", "u", "v", "origin", "target",
        }
        first_row_tokens = [cell.lower() for cell in rows[0]]
        is_edge_header = any(token in header_keywords for token in first_row_tokens)

        def is_binary(value: str) -> bool:
            try:
                return float(value) in (0, 1)
            except ValueError:
                return False

        edge_rows = rows[1:] if is_edge_header else rows
        is_edge_list = is_edge_header or (
            len(edge_rows) > 2
            and all(len(row) in (2, 3) for row in edge_rows)
            and any(not is_binary(cell) for row in edge_rows for cell in row[:2])
        )

        if is_edge_list:
            if not edge_rows or any(len(row) < 2 for row in edge_rows):
                raise ValueError("CSV edge lists must contain source and destination columns.")
            node_names = []
            edges = []
            for row in edge_rows:
                source, destination = row[:2]
                if not source or not destination:
                    raise ValueError("CSV edge-list source and destination values cannot be empty.")
                for name in (source, destination):
                    if name not in node_names:
                        node_names.append(name)
                edges.append((source, destination))

            node_map = {name: index for index, name in enumerate(node_names)}
            matrix_values = [[0] * len(node_names) for _ in node_names]
            for source, destination in edges:
                matrix_values[node_map[source]][node_map[destination]] = 1
            labels = {index: name for index, name in enumerate(node_names)}
        else:
            matrix_rows = rows
            column_labels = []
            if matrix_rows[0][0] == "":
                column_labels = matrix_rows[0][1:]
                matrix_rows = matrix_rows[1:]
            elif matrix_rows[0] and not any(is_binary(cell) for cell in matrix_rows[0]):
                column_labels = matrix_rows[0][1:]
                matrix_rows = matrix_rows[1:]

            row_labels = []
            matrix_data = []
            for row in matrix_rows:
                offset = 1 if len(row) > 1 and not is_binary(row[0]) and all(
                    is_binary(cell) for cell in row[1:]
                ) else 0
                if offset:
                    row_labels.append(row[0])
                values = row[offset:]
                if not values or not all(is_binary(cell) for cell in values):
                    raise ValueError("CSV adjacency matrices must contain only 0 and 1 values.")
                matrix_data.append([int(float(cell)) for cell in values])
            matrix_values = matrix_data
            if column_labels:
                labels = {index: name for index, name in enumerate(column_labels)}
            elif row_labels:
                labels = {index: name for index, name in enumerate(row_labels)}
            else:
                labels = {}

    try:
        matrix = np.asarray(matrix_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Adjacency matrix must contain numeric values.") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency matrix must be a non-empty square matrix.")
    if not np.isfinite(matrix).all() or not np.isin(matrix, (0, 1)).all():
        raise ValueError("Adjacency matrix values must be 0 or 1.")

    node_count = matrix.shape[0]
    if any(index < 0 or index >= node_count for index in labels):
        raise ValueError("Node labels must refer to valid adjacency-matrix indexes.")
    for index in range(node_count):
        labels.setdefault(index, f"N{index}")
    return matrix.astype(int), labels


class MathWorksThermalEngine:
    def __init__(self, adj_matrix: np.ndarray):
        self.matrix = adj_matrix.astype(int)
        self.n = len(self.matrix)

    def update_matrix(self, new_matrix: np.ndarray) -> None:
        self.matrix = new_matrix.astype(int)
        self.n = len(self.matrix)

    def compute_warshall(self) -> tuple[np.ndarray, list[np.ndarray]]:
        reachability = self.matrix.copy().astype(bool)
        history = [reachability.astype(int).copy()]
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    reachability[i][j] = reachability[i][j] or (
                        reachability[i][k] and reachability[k][j]
                    )
            history.append(reachability.astype(int).copy())
        return reachability.astype(int), history

    def compute_equivalence_classes(self, reachability: np.ndarray) -> list[list[int]]:
        equivalence_relation = (reachability == 1) & (reachability.T == 1)
        visited = [False] * self.n
        classes = []
        for i in range(self.n):
            if not visited[i]:
                equivalence_class = [
                    j for j in range(self.n) if equivalence_relation[i][j]
                ]
                for node in equivalence_class:
                    visited[node] = True
                classes.append(equivalence_class)
        return classes

    def check_planarity(self) -> bool:
        graph = nx.from_numpy_array(self.matrix, create_using=nx.Graph)
        is_planar, _ = nx.check_planarity(graph)
        return is_planar

    def find_flow_traversal(self) -> tuple[bool, list[tuple[int, int]], int]:
        graph = nx.from_numpy_array(self.matrix, create_using=nx.DiGraph)
        active_nodes = [node for node in graph.nodes() if graph.degree(node) > 0]
        if not active_nodes:
            return False, [], -1

        start_node = active_nodes[0]
        for node in active_nodes:
            if graph.out_degree(node) > graph.in_degree(node):
                start_node = node
                break

        blocked_nodes = [
            node for node in active_nodes
            if graph.in_degree(node) > 0 and graph.out_degree(node) == 0
        ]
        reachable_blocked_nodes = [
            node for node in blocked_nodes if nx.has_path(graph, start_node, node)
        ]
        if reachable_blocked_nodes:
            fault_node = min(
                reachable_blocked_nodes,
                key=lambda node: nx.shortest_path_length(graph, start_node, node),
            )
            path = nx.shortest_path(graph, start_node, fault_node)
            return False, list(zip(path, path[1:])), fault_node

        if nx.is_eulerian(graph):
            return True, list(nx.eulerian_circuit(graph, source=start_node)), -1
        if nx.has_eulerian_path(graph):
            return True, list(nx.eulerian_path(graph, source=start_node)), -1

        visited_edges = set()
        partial_path = []
        current_node = start_node
        while True:
            out_edges = [
                edge for edge in graph.out_edges(current_node)
                if edge not in visited_edges
            ]
            if not out_edges:
                break
            next_edge = out_edges[0]
            visited_edges.add(next_edge)
            partial_path.append(next_edge)
            current_node = next_edge[1]
            if len(partial_path) > graph.number_of_edges() * 2:
                break

        if len(visited_edges) == graph.number_of_edges() and visited_edges:
            return True, partial_path, -1
        return False, partial_path, current_node


def initialize_state() -> None:
    if "current_matrix" not in st.session_state:
        st.session_state.current_matrix = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
        ])
        st.session_state.original_matrix = st.session_state.current_matrix.copy()
        st.session_state.node_labels = {
            0: "Main_Coolant_Pump",
            1: "3Way_Thermostat",
            2: "HV_PTC_Heater",
            3: "ColdPlate_Front",
            4: "ColdPlate_Rear",
            5: "HVAC_Chiller",
        }
        st.session_state.positions = None
        st.session_state.selected_node = None
        st.session_state.flow_step = -1
        st.session_state.flow_running = False
        st.session_state.flow_started_at = None
        st.session_state.matrix_view_version = 0
        st.session_state.inspector_message = (
            "Click any node on the graph or run simulation to evaluate flow paths."
        )
        st.session_state.upload_digest = None


def stop_flow(reset_step: bool = True) -> None:
    st.session_state.flow_running = False
    st.session_state.flow_started_at = None
    if reset_step:
        st.session_state.flow_step = -1


def toggle_matrix_cell(row: int, column: int) -> None:
    st.session_state.current_matrix[row][column] = (
        1 - st.session_state.current_matrix[row][column]
    )
    st.session_state.matrix_view_version += 1
    stop_flow()


def reset_matrix() -> None:
    st.session_state.current_matrix = st.session_state.original_matrix.copy()
    st.session_state.selected_node = None
    st.session_state.inspector_message = (
        "Original topology restored from the loaded file. "
        "The matrix matches its source configuration."
    )
    st.session_state.matrix_view_version += 1
    stop_flow()


def clear_matrix() -> None:
    st.session_state.current_matrix = np.zeros_like(st.session_state.current_matrix)
    st.session_state.selected_node = None
    st.session_state.inspector_message = (
        "Matrix cleared: all coolant paths have been removed."
    )
    st.session_state.matrix_view_version += 1
    stop_flow()


def toggle_flow() -> None:
    if st.session_state.flow_running:
        stop_flow()
        return

    _, _, _, _, edges, fault_node = get_analysis()
    if not edges:
        st.session_state.inspector_message = (
            "FLOW ERROR: Completely disconnected manifold graph."
        )
        return
    st.session_state.flow_step = 0
    st.session_state.flow_running = len(edges) > 1
    st.session_state.flow_started_at = (
        time.monotonic() if st.session_state.flow_running else None
    )
    st.session_state.inspector_message = ""
    if len(edges) == 1:
        st.session_state.inspector_message = finished_flow_message(edges, fault_node)


@st.cache_data(show_spinner=False)
def analyze_matrix(matrix_values: tuple[tuple[int, ...], ...]):
    matrix = np.asarray(matrix_values, dtype=int)
    engine = MathWorksThermalEngine(matrix)
    reachability, _ = engine.compute_warshall()
    classes = engine.compute_equivalence_classes(reachability)
    planar = engine.check_planarity()
    eulerian, edges, fault_node = engine.find_flow_traversal()
    return reachability, classes, planar, eulerian, edges, fault_node


def get_analysis():
    matrix_values = tuple(
        tuple(int(value) for value in row)
        for row in st.session_state.current_matrix
    )
    return analyze_matrix(matrix_values)


def finished_flow_message(edges: list[tuple[int, int]], fault_node: int) -> str:
    labels = st.session_state.node_labels
    if fault_node != -1:
        fault_name = escape(labels.get(fault_node, f"N{fault_node}"))
        return (
            f"<b style='color:#C62828;'>PARTIAL FLOW STOPPED AT {fault_name}</b><br><br>"
            f"• <b>Reason:</b> Coolant flow reached junction {fault_name} but cannot continue.<br>"
            "• <b>Diagnostic:</b> Zero valid outlet conduits exist from this junction.<br>"
            f"• <b>Status:</b> Initial {len(edges)} flow segments verified working (Blue Arrows)."
        )
    return (
        "<b style='color:#248A3D;'>FULL EULERIAN FLOW CIRCUIT COMPLETE</b><br><br>"
        f"• <b>Status:</b> All {len(edges)} coolant conduits successfully traversed.<br>"
        "• <b>Equilibrium:</b> Uniform pressure and continuous thermal dissipation verified."
    )


def render_flow_control(flow_edges: list[tuple[int, int]]) -> None:
    if st.session_state.flow_running:
        label = "STOP SIMULATION"
    elif (
        st.session_state.flow_step >= 0
        and st.session_state.flow_step == len(flow_edges) - 1
    ):
        label = "RESTART SIMULATION"
    else:
        label = "SIMULATE COOLANT FLOW"
    st.button(
        label,
        key="flow_toggle",
        width="stretch",
        on_click=toggle_flow,
    )


def shortened_label(label: str, width: int = 11) -> str:
    return textwrap.fill(label.replace("_", " "), width=width, break_long_words=True)


def build_manifold_figure(
    matrix: np.ndarray,
    labels: dict[int, str],
    reachability: np.ndarray,
    edges: list[tuple[int, int]],
    active_step: int,
    fault_node: int,
    selected_node: int | None,
) -> go.Figure:
    graph = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    node_count = len(matrix)
    if "positions" not in st.session_state or st.session_state.positions is None:
        if node_count == 16:
            positions = {i: (i % 4, 3 - (i // 4)) for i in range(16)}
        else:
            positions = nx.spring_layout(graph, seed=42)
        st.session_state.positions = {
            node: [float(point[0]), float(point[1])]
            for node, point in positions.items()
        }
    elif len(st.session_state.positions) != node_count:
        positions = nx.spring_layout(graph, seed=42)
        st.session_state.positions = {
            node: [float(point[0]), float(point[1])]
            for node, point in positions.items()
        }
    positions = st.session_state.positions

    figure = go.Figure()
    active_edges = set(edges[:active_step + 1]) if active_step >= 0 else set()
    edge_lines = {
        "active": {"x": [], "y": [], "hover": []},
        "idle": {"x": [], "y": [], "hover": []},
    }
    arrow_x = []
    arrow_y = []
    arrow_angles = []
    arrow_colors = []
    arrow_hover = []
    for source, destination in graph.edges():
        start = np.asarray(positions[source], dtype=float)
        end = np.asarray(positions[destination], dtype=float)
        vector = end - start
        distance = np.linalg.norm(vector)
        if distance == 0:
            control = start + np.array([0.12, 0.12])
        else:
            normal = np.array([-vector[1], vector[0]]) / distance
            control = (start + end) / 2 + normal * distance * 0.08
        t = np.linspace(0, 1, 30)
        curve = (
            ((1 - t) ** 2)[:, None] * start
            + (2 * (1 - t) * t)[:, None] * control
            + (t ** 2)[:, None] * end
        )
        is_active = (source, destination) in active_edges
        source_name = labels.get(source, f"N{source}")
        destination_name = labels.get(destination, f"N{destination}")
        status = "ACTIVE BLUE COOLANT FLOW" if is_active else "IDLE CONDUIT"
        hover_text = (
            f"COOLANT CONDUIT: {source_name} → {destination_name}<br>"
            f"Source: {source_name}<br>Destination: {destination_name}<br>"
            f"Status: {status}<br>Direction: Unidirectional Micro-channel"
        )
        line_group = edge_lines["active" if is_active else "idle"]
        line_group["x"].extend(curve[:, 0].tolist() + [None])
        line_group["y"].extend(curve[:, 1].tolist() + [None])
        line_group["hover"].extend([hover_text] * len(curve) + [None])

        arrow_point = curve[25]
        arrow_x.append(float(arrow_point[0]))
        arrow_y.append(float(arrow_point[1]))
        arrow_angles.append(float(np.degrees(np.arctan2(vector[0], vector[1]))))
        arrow_colors.append("#007AFF" if is_active else "#AEB4BE")
        arrow_hover.append(hover_text)

    for group_name, group in edge_lines.items():
        if not group["x"]:
            continue
        is_active = group_name == "active"
        figure.add_trace(go.Scatter(
            x=group["x"],
            y=group["y"],
            text=group["hover"],
            mode="lines",
            line={
                "color": "#007AFF" if is_active else "#AEB4BE",
                "width": 3.5 if is_active else 2,
            },
            hovertemplate="%{text}<extra></extra>",
            connectgaps=False,
            showlegend=False,
        ))

    if arrow_x:
        figure.add_trace(go.Scatter(
            x=arrow_x,
            y=arrow_y,
            text=arrow_hover,
            mode="markers",
            marker={
                "symbol": "triangle-up",
                "size": 10,
                "angle": arrow_angles,
                "color": arrow_colors,
                "line": {"width": 0},
            },
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    node_colors = []
    border_colors = []
    for node in graph.nodes():
        if node == fault_node and active_step == len(edges) - 1 and fault_node != -1:
            node_colors.append("#FF3B30")
            border_colors.append("#C62828")
        elif node == selected_node:
            node_colors.append("#FF9F0A")
            border_colors.append("#C77700")
        else:
            node_colors.append("#007AFF" if graph.degree(node) > 0 else "#AEAEB2")
            border_colors.append("#0060DF" if graph.degree(node) > 0 else "#8E8E93")

    node_hover = []
    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        targets = [
            target for target in range(len(reachability))
            if reachability[node][target] == 1
        ]
        node_hover.append(
            f"COOLANT JUNCTION: {labels.get(node, f'N{node}')} (ID: {node})<br>"
            f"Inlet conduits: {in_degree}<br>Outlet conduits: {out_degree}<br>"
            f"Flow state: {'BALANCED' if in_degree == out_degree else 'UNBALANCED FAULT'}<br>"
            f"Reachability (Warshall): {targets}"
        )

    figure.add_trace(go.Scatter(
        x=[positions[node][0] for node in graph.nodes()],
        y=[positions[node][1] for node in graph.nodes()],
        mode="markers+text",
        text=[
            shortened_label(labels.get(node, f"N{node}"), width=13)
            for node in graph.nodes()
        ],
        textposition="middle center",
        textfont={"family": "Arial", "size": 9, "color": "#1D1D1F"},
        marker={
            "size": 35 if node_count == 16 else 43,
            "color": node_colors,
            "line": {"color": border_colors, "width": 2},
        },
        customdata=list(graph.nodes()),
        hovertext=node_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        selected={"marker": {"opacity": 1}},
        unselected={"marker": {"opacity": 0.95}},
        showlegend=False,
    ))

    if (
        fault_node != -1
        and active_step == len(edges) - 1
        and edges
    ):
        figure.add_annotation(
            x=positions[fault_node][0],
            y=positions[fault_node][1],
            text=(
                "STAGNATION FAULT AT "
                f"{shortened_label(labels.get(fault_node, f'N{fault_node}'), 22)}"
                "<br>(Circuit Broken / Zero Out-Flow)"
            ),
            showarrow=True,
            arrowhead=2,
            ax=60,
            ay=-65,
            bgcolor="#FFF2F2",
            bordercolor="#FF3B30",
            borderwidth=1.5,
            borderpad=7,
            font={"family": "Arial", "size": 11, "color": "#C62828"},
        )

    x_values = [point[0] for point in positions.values()]
    y_values = [point[1] for point in positions.values()]
    x_padding = max(0.55, (max(x_values) - min(x_values)) * 0.12)
    y_padding = max(0.55, (max(y_values) - min(y_values)) * 0.12)
    figure.update_layout(
        height=760,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="closest",
        clickmode="event+select",
        dragmode="pan",
        xaxis={
            "range": [min(x_values) - x_padding, max(x_values) + x_padding],
            "showgrid": True,
            "gridcolor": "#E5E5EA",
            "griddash": "dash",
            "zeroline": False,
            "visible": False,
            "scaleanchor": "y",
        },
        yaxis={
            "range": [min(y_values) - y_padding, max(y_values) + y_padding],
            "showgrid": True,
            "gridcolor": "#E5E5EA",
            "griddash": "dash",
            "zeroline": False,
            "visible": False,
        },
    )
    return figure


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: #F5F5F7;
            color: #1D1D1F;
            font-family: "Helvetica Neue", Arial, sans-serif;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"], footer {
            visibility: hidden;
            height: 0;
        }
        .block-container { max-width: 100%; padding: 16px 16px 20px; }
        [data-testid="stHorizontalBlock"] { gap: 12px; }
        .st-key-header-panel, .st-key-graph-panel, .st-key-matrix-panel {
            background: #FFFFFF;
            border: 1px solid #D2D2D7;
            border-radius: 12px;
            padding: 12px 16px;
        }
        .st-key-header-panel { border-left: 4px solid #007AFF; }
        .brand-title {
            font-size: 13px; font-weight: 700; color: #1D1D1F;
            white-space: nowrap; padding: 8px 0;
        }
        .metric-card {
            height: 60px; padding: 10px 16px;
            background: #FFFFFF; border: 1px solid #D2D2D7;
            border-top: 3px solid #007AFF; border-radius: 11px;
        }
        .metric-title {
            color: #6E6E73; font-size: 10px; font-weight: 600;
            text-transform: uppercase; margin-bottom: 4px;
        }
        .metric-value { color: #1D1D1F; font-size: 15px; font-weight: 700; }
        .section-title {
            color: #1D1D1F; font-size: 12px; font-weight: 700;
            margin: 0 0 8px 0;
        }
        .helper-text { color: #6E6E73; font-size: 10px; }
        .inspector-card {
            min-height: 170px; padding: 14px;
            background: #F5F5F7; border: 1px solid #D2D2D7;
            border-left: 3px solid #64A8FF; border-radius: 9px;
            color: #3A3A3C; font-size: 12px;
        }
        .st-key-matrix-grid {
            padding: 8px;
            border: 1px solid #E5E8EF;
            border-radius: 10px;
            background: #FFFFFF;
        }
        .st-key-matrix-grid [data-testid="stDataFrame"] {
            border: 1px solid #E7EBF2;
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stFileUploader"] {
            display: flex; align-items: center; gap: 12px; padding: 0;
        }
        div[data-testid="stFileUploader"] > label {
            flex: 1 1 auto; margin: 0; white-space: nowrap;
        }
        div[data-testid="stFileUploader"] section {
            flex: 0 0 auto; border: 0; padding: 0;
            min-height: 0; background: transparent;
        }
        div[data-testid="stFileUploader"] small { display: none; }
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none;
        }
        div[data-testid="stFileUploader"] label {
            margin: 0; font-size: 11px; font-weight: 600; color: #1D1D1F;
        }
        .stButton button {
            background: #FFFFFF !important; color: #1D1D1F !important;
            border: 1px solid #C7C7CC !important;
            border-radius: 7px; font-weight: 600; min-height: 36px;
        }
        .stButton button:hover {
            background: #F2F7FF !important; border-color: #64A8FF !important;
        }
        .st-key-matrix_reset button {
            background: #F2F7FF !important; border-color: #B8D7FF !important;
            color: #0060DF !important;
        }
        .st-key-matrix_clear button {
            background: #FFF5F4 !important; border-color: #FFC9C5 !important;
            color: #C62828 !important;
        }
        div[data-testid="stFileUploader"] button {
            background: #FFFFFF !important; color: #1D1D1F !important;
            border: 1px solid #C7C7CC !important; border-radius: 7px;
        }
        [data-baseweb="select"] > div,
        [data-testid="stWidgetLabel"] p {
            color: #1D1D1F !important;
        }
        @media (max-width: 850px) {
            .block-container { padding: 10px; }
            .brand-title { white-space: normal; font-size: 11px; }
            .metric-card { height: 72px; padding: 8px; }
            .metric-value { font-size: 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


initialize_state()
apply_styles()
reachability, equivalence_classes, is_planar, is_eulerian, flow_edges, fault_node = (
    get_analysis()
)
node_labels = st.session_state.node_labels
node_count = len(st.session_state.current_matrix)

with st.container(key="header-panel"):
    header_left, import_column, action_column = st.columns([1.5, 1.25, 1.1])
    with header_left:
        st.markdown(
            '<div class="brand-title">SIMVENT | EV COOLING TOPOLOGY SOLVER</div>',
            unsafe_allow_html=True,
        )
    with import_column:
        uploaded_file = st.file_uploader(
            "IMPORT EV TOPOLOGY (.JSON / .CSV)",
            type=["json", "csv"],
            key="topology_upload",
        )
    with action_column:
        render_flow_control(flow_edges)

if uploaded_file is not None:
    upload_bytes = uploaded_file.getvalue()
    upload_digest = (uploaded_file.name, hashlib.sha256(upload_bytes).hexdigest())
    if upload_digest != st.session_state.upload_digest:
        try:
            matrix, labels = load_topology_contents(
                upload_bytes.decode("utf-8-sig")
            )
            st.session_state.current_matrix = matrix
            st.session_state.original_matrix = matrix.copy()
            st.session_state.node_labels = labels
            st.session_state.positions = None
            st.session_state.selected_node = None
            st.session_state.matrix_view_version += 1
            st.session_state.inspector_message = (
                "Topology imported successfully. Run a flow simulation to inspect the route."
            )
            st.session_state.upload_digest = upload_digest
            stop_flow()
            st.rerun()
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            st.session_state.inspector_message = (
                f"Import failed: {escape(str(error))}"
            )
            st.session_state.upload_digest = upload_digest

reachability, equivalence_classes, is_planar, is_eulerian, flow_edges, fault_node = (
    get_analysis()
)
node_labels = st.session_state.node_labels
current_matrix = st.session_state.current_matrix
reachable_ratio = int(np.sum(reachability) / (node_count * node_count) * 100)

metric_columns = st.columns(4)
metric_values = [
    (
        "COLD PLATE LAYOUT",
        "SINGLE-LAYER PLANAR" if is_planar else "NON-PLANAR (3D PLATE)",
    ),
    (
        "MANIFOLD CIRCUIT PATH",
        "EULERIAN ROUTE VALID" if is_eulerian else "PARTIAL FLOW (FAULT)",
    ),
    ("THERMAL REACHABILITY RATIO", f"{reachable_ratio}% Reachable"),
    ("EQUIVALENCE CLASSES", f"{len(equivalence_classes)} Partition Groups"),
]
for column, (title, value) in zip(metric_columns, metric_values):
    with column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

left_panel, right_panel = st.columns([850, 590], gap="small")

with left_panel:
    with st.container(key="graph-panel"):
        st.markdown(
            '<div class="section-title">'
            "REALTIME FLOW MANIFOLD VIEWPORT (HOVER FOR TOOLTIPS)</div>",
            unsafe_allow_html=True,
        )

        @st.fragment(run_every="100ms")
        def render_manifold() -> None:
            if st.session_state.flow_running:
                started_at = st.session_state.flow_started_at
                if started_at is None:
                    st.session_state.flow_started_at = time.monotonic()
                    started_at = st.session_state.flow_started_at
                step = min(
                    int((time.monotonic() - started_at) / 0.45),
                    len(flow_edges) - 1,
                )
                st.session_state.flow_step = max(0, step)
                if step == len(flow_edges) - 1:
                    st.session_state.flow_running = False
                    st.session_state.flow_started_at = None
                    st.session_state.inspector_message = finished_flow_message(
                        flow_edges, fault_node
                    )
                    st.rerun()

            figure = build_manifold_figure(
                st.session_state.current_matrix,
                st.session_state.node_labels,
                reachability,
                flow_edges,
                st.session_state.flow_step,
                fault_node,
                st.session_state.selected_node,
            )
            event = st.plotly_chart(
                figure,
                key="manifold_chart",
                width="stretch",
                on_select="rerun",
                selection_mode="points",
                config={"scrollZoom": True, "displaylogo": False},
            )
            selected_points = event.get("selection", {}).get("points", [])
            if selected_points:
                selected_index = selected_points[-1].get("point_index")
                if selected_index is not None:
                    selected_index = int(selected_index)
                    if (
                        selected_index < node_count
                        and st.session_state.selected_node != selected_index
                    ):
                        st.session_state.selected_node = selected_index
                        st.session_state.inspector_message = ""
                        st.rerun()

        render_manifold()

with right_panel:
    with st.container(key="matrix-panel"):
        header_row = st.columns([1.2, 0.8, 0.8])
        with header_row[0]:
            st.markdown(
                '<div class="section-title" style="padding-top:10px">'
                "ADJACENCY MATRIX</div>",
                unsafe_allow_html=True,
            )
        with header_row[1]:
            st.button(
                "RESET", key="matrix_reset", width="stretch",
                on_click=reset_matrix,
            )
        with header_row[2]:
            st.button(
                "CLEAR", key="matrix_clear", width="stretch",
                on_click=clear_matrix,
            )

        st.markdown(
            '<div class="helper-text">Adjacency Matrix M_R '
            "(Click a cell to toggle a conduit)</div>",
            unsafe_allow_html=True,
        )

        with st.container(key="matrix-grid"):
            matrix_column_keys = [f"node_{column}" for column in range(node_count)]
            matrix_frame = pd.DataFrame(
                current_matrix,
                index=[
                    node_labels.get(row, f"N{row}")
                    for row in range(node_count)
                ],
                columns=matrix_column_keys,
            )
            matrix_frame.index.name = ""
            matrix_style = matrix_frame.style.map(
                lambda value: (
                    "background-color: #007AFF; color: #FFFFFF; font-weight: 600;"
                    if value == 1
                    else "background-color: #F8FAFD; color: #6E7480;"
                ),
                subset=matrix_column_keys,
            )
            matrix_columns_config = {
                key: st.column_config.NumberColumn(
                    label=node_labels.get(column, f"N{column}"),
                    format="%d",
                    width="small",
                )
                for column, key in enumerate(matrix_column_keys)
            }
            matrix_event = st.dataframe(
                matrix_style,
                key=f"matrix_view_{st.session_state.matrix_view_version}",
                column_config=matrix_columns_config,
                hide_index=False,
                height=min(600, max(240, 38 * (node_count + 1))),
                width="stretch",
                on_select="rerun",
                selection_mode="single-cell",
            )
            selected_cells = matrix_event.selection.cells
            if selected_cells:
                row, column_key = selected_cells[0]
                if column_key in matrix_column_keys and 0 <= row < node_count:
                    toggle_matrix_cell(row, int(column_key.removeprefix("node_")))
                    st.rerun()

        st.markdown(
            '<div class="section-title" style="margin-top:10px">'
            "THERMAL JUNCTION INSPECTOR</div>",
            unsafe_allow_html=True,
        )

        message = st.session_state.inspector_message
        selected_node = st.session_state.selected_node
        if selected_node is not None and not message:
            graph = nx.from_numpy_array(current_matrix, create_using=nx.DiGraph)
            in_degree = graph.in_degree(selected_node)
            out_degree = graph.out_degree(selected_node)
            reachable_targets = [
                node for node in range(node_count)
                if reachability[selected_node][node] == 1
            ]
            selected_name = escape(node_labels.get(selected_node, f"N{selected_node}"))
            message = (
                f"<b>ELEMENT INSPECTOR: {selected_name} "
                f"(ID: {selected_node})</b><br><br>"
                f"• <b>Inlet Coolant Conduit Count (In-Degree):</b> {in_degree}<br>"
                f"• <b>Outlet Coolant Conduit Count (Out-Degree):</b> {out_degree}<br>"
                f"• <b>Flow Equilibrium Status:</b> "
                f"{'BALANCED IN/OUT' if in_degree == out_degree else 'PRESSURE GRADIENT UNBALANCED'}"
                "<br>"
                f"• <b>Thermal Downstream Reachability (Warshall):</b> "
                f"{reachable_targets}"
            )
        elif not message and st.session_state.flow_step == -1:
            message = (
                "Click any node on the graph or run simulation to evaluate flow paths."
            )
        st.markdown(
            f'<div class="inspector-card">{message}</div>',
            unsafe_allow_html=True,
        )
