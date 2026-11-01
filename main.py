import sys
import os
import json
import csv
import textwrap
import numpy as np
import networkx as nx
os.environ["QT_API"] = "pyside6"
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSplitter, QHeaderView, QTableWidget,
    QTableWidgetItem, QToolTip, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QCursor


def load_topology_file(file_path: str) -> tuple[np.ndarray, dict[int, str]]:
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as topology_file:
        contents = topology_file.read()

    if contents.lstrip().startswith(('{', '[')):
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
            for row in csv.reader(contents.splitlines())
            if any(cell.strip() for cell in row)
        ]
        if not rows:
            raise ValueError("CSV file is empty.")

        header_keywords = {
            'source', 'destination', 'from', 'to', 'start', 'end', 'src',
            'dst', 'node1', 'node2', 'u', 'v', 'origin', 'target'
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
            if matrix_rows[0][0] == '':
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

    def update_matrix(self, new_matrix: np.ndarray):
        self.matrix = new_matrix.astype(int)
        self.n = len(self.matrix)

    def analyze_properties(self) -> dict:
        is_ref = all(self.matrix[i][i] == 1 for i in range(self.n))
        is_sym = np.array_equal(self.matrix, self.matrix.T)
        is_anti = True
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.matrix[i][j] == 1 and self.matrix[j][i] == 1:
                    is_anti = False
                    break
        is_trans = True
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i][j] == 1:
                    for k in range(self.n):
                        if self.matrix[j][k] == 1 and self.matrix[i][k] != 1:
                            is_trans = False
                            break
        return {
            "Reflexive": is_ref,
            "Symmetric": is_sym,
            "Anti-Symmetric": is_anti,
            "Transitive": is_trans
        }

    def compute_warshall(self) -> tuple[np.ndarray, list[np.ndarray]]:
        W = self.matrix.copy().astype(bool)
        history = [W.astype(int).copy()]
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    W[i][j] = W[i][j] or (W[i][k] and W[k][j])
            history.append(W.astype(int).copy())
        return W.astype(int), history

    def compute_equivalence_classes(self, W: np.ndarray) -> list[list[int]]:
        eq_rel = (W == 1) & (W.T == 1)
        visited = [False] * self.n
        classes = []
        for i in range(self.n):
            if not visited[i]:
                eq_class = [j for j in range(self.n) if eq_rel[i][j]]
                for node in eq_class:
                    visited[node] = True
                classes.append(eq_class)
        return classes

    def check_planarity(self) -> bool:
        G_undirected = nx.from_numpy_array(self.matrix, create_using=nx.Graph)
        is_planar, _ = nx.check_planarity(G_undirected)
        return is_planar

    def find_flow_traversal(self) -> tuple[bool, list[tuple[int, int]], int]:
        G_dir = nx.from_numpy_array(self.matrix, create_using=nx.DiGraph)
        active_nodes = [n for n in G_dir.nodes() if G_dir.degree(n) > 0]
        if not active_nodes:
            return False, [], -1

        start_node = active_nodes[0]
        for n in active_nodes:
            if G_dir.out_degree(n) > G_dir.in_degree(n):
                start_node = n
                break

        blocked_nodes = [
            node for node in active_nodes
            if G_dir.in_degree(node) > 0 and G_dir.out_degree(node) == 0
        ]
        reachable_blocked_nodes = [
            node for node in blocked_nodes if nx.has_path(G_dir, start_node, node)
        ]
        if reachable_blocked_nodes:
            fault_node = min(
                reachable_blocked_nodes,
                key=lambda node: nx.shortest_path_length(G_dir, start_node, node)
            )
            path = nx.shortest_path(G_dir, start_node, fault_node)
            return False, list(zip(path, path[1:])), fault_node

        try:
            if nx.is_eulerian(G_dir):
                return True, list(nx.eulerian_circuit(G_dir, source=start_node)), -1
        except Exception:
            pass

        try:
            if nx.has_eulerian_path(G_dir):
                return True, list(nx.eulerian_path(G_dir, source=start_node)), -1
        except Exception:
            pass

        visited_edges = set()
        partial_path = []
        curr = start_node
        
        while True:
            out_edges = [(u, v) for u, v in G_dir.out_edges(curr) if (u, v) not in visited_edges]
            if not out_edges:
                break
            nxt_edge = out_edges[0]
            visited_edges.add(nxt_edge)
            partial_path.append(nxt_edge)
            curr = nxt_edge[1]
            if len(partial_path) > G_dir.number_of_edges() * 2:
                break

        if len(visited_edges) == G_dir.number_of_edges() and len(visited_edges) > 0:
            return True, partial_path, -1

        fault_node = curr
        return False, partial_path, fault_node


class CADManifoldCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), facecolor='none')
        self.fig.patch.set_alpha(0)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('none')
        self.ax.patch.set_alpha(0)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.G = None
        self.pos = None
        self.selected_node = None
        self.current_W = None
        self.euler_path = []
        self.active_step = -1
        self.fault_node = -1
        self.node_labels = {}
        self.dragging_node = None
        self.drag_moved = False
        self.layout_node_count = None

        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_hover)
        self.mpl_connect('scroll_event', self._on_scroll)

    def set_on_node_selected_callback(self, callback):
        self.node_selected_callback = callback

    def render_manifold(self, matrix: np.ndarray, euler_path: list[tuple[int, int]] = None, 
                        active_step: int = -1, W: np.ndarray = None, fault_node: int = -1, node_labels: dict = None):
        reset_view = self.pos is None or self.layout_node_count != len(matrix)
        view_limits = None if reset_view else (self.ax.get_xlim(), self.ax.get_ylim())
        self.ax.clear()
        self.ax.set_facecolor('none')
        self.ax.patch.set_alpha(0)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        self.G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        self.current_W = W
        self.euler_path = euler_path or []
        self.active_step = active_step
        self.fault_node = fault_node
        self.node_labels = node_labels or {i: f"N{i}" for i in range(len(matrix))}
        n = len(matrix)

        if n == 16 and reset_view:
            self.pos = {i: (i % 4, 3 - (i // 4)) for i in range(16)}
        elif reset_view:
            self.pos = nx.spring_layout(self.G, seed=42)
        self.layout_node_count = n

        self.ax.grid(True, color='#E5E5EA', linestyle='--', linewidth=0.8, alpha=0.8)

        nx.draw_networkx_edges(
            self.G, self.pos, ax=self.ax, 
            edge_color='#AEB4BE', width=2, arrowsize=14, 
            arrowstyle='-|>', connectionstyle="arc3,rad=0.08"
        )

        if euler_path and active_step >= 0:
            highlighted_edges = euler_path[:active_step + 1]
            nx.draw_networkx_edges(
                self.G, self.pos, ax=self.ax, edgelist=highlighted_edges,
                edge_color='#007AFF', width=3.5, arrowsize=18,
                arrowstyle='-|>', connectionstyle="arc3,rad=0.08"
            )

        colors, border_colors = [], []
        for node in self.G.nodes():
            if node == self.fault_node and active_step == len(self.euler_path) - 1 and self.fault_node != -1:
                colors.append('#FF3B30')
                border_colors.append('#C62828')
            elif node == self.selected_node:
                colors.append('#FF9F0A')
                border_colors.append('#C77700')
            else:
                colors.append('#007AFF' if self.G.degree(node) > 0 else '#AEAEB2')
                border_colors.append('#0060DF' if self.G.degree(node) > 0 else '#8E8E93')

        node_size = 550 if n == 16 else 850
        nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.ax, 
            node_color=colors, node_size=node_size, 
            linewidths=2, edgecolors=border_colors
        )
        
        nx.draw_networkx_labels(
            self.G, self.pos, ax=self.ax,
            labels={
                node: textwrap.fill(
                    self.node_labels.get(node, f"N{node}").replace('_', ' '),
                    width=13, break_long_words=True
                )
                for node in self.G.nodes()
            },
            font_color='#1D1D1F', font_family='Arial', font_weight='bold',
            font_size=7,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFFFFF', edgecolor='none', alpha=0.92)
        )

        show_fault_badge = (
            self.fault_node != -1
            and active_step == len(self.euler_path) - 1
            and len(self.euler_path) > 0
        )
        if show_fault_badge:
            fx, fy = self.pos[self.fault_node]
            fault_label = textwrap.fill(
                self.node_labels.get(self.fault_node, f"N{self.fault_node}").replace('_', ' '),
                width=22, break_long_words=True
            )
            x_values = [coords[0] for coords in self.pos.values()]
            y_values = [coords[1] for coords in self.pos.values()]
            place_right = fx < (min(x_values) + max(x_values)) / 2
            self.ax.annotate(
                f"STAGNATION FAULT AT {fault_label}\n(Circuit Broken / Zero Out-Flow)",
                xy=(fx, fy), xytext=(fx + (0.4 if place_right else -0.4), fy + 0.45),
                ha='left' if place_right else 'right',
                bbox=dict(boxstyle="round,pad=0.5", fc="#FFF2F2", ec="#FF3B30", lw=1.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#FF3B30", lw=2),
                fontsize=9, fontfamily="Arial", fontweight="bold", color="#C62828"
            )

        x_values = [coords[0] for coords in self.pos.values()]
        y_values = [coords[1] for coords in self.pos.values()]
        x_padding = max(0.55, (max(x_values) - min(x_values)) * 0.12)
        y_padding = max(0.55, (max(y_values) - min(y_values)) * 0.12)
        if show_fault_badge:
            x_padding = max(x_padding, 3.0)
            y_padding = max(y_padding, 1.8)
        if reset_view:
            self.ax.set_xlim(min(x_values) - x_padding, max(x_values) + x_padding)
            self.ax.set_ylim(min(y_values) - y_padding, max(y_values) + y_padding)
        self.ax.set_aspect('auto')
        if view_limits is not None:
            self.ax.set_xlim(view_limits[0])
            self.ax.set_ylim(view_limits[1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.draw()

    def _on_press(self, event):
        if event.xdata is None or event.ydata is None or self.pos is None:
            return
        if event.button != 1:
            return

        click_coords = np.array([event.x, event.y])
        min_dist, closest_node = float('inf'), None
        for node, coords in self.pos.items():
            node_pixels = self.ax.transData.transform(coords)
            dist = np.linalg.norm(node_pixels - click_coords)
            if dist < 30 and dist < min_dist:
                min_dist = dist
                closest_node = node
        if closest_node is not None:
            self.selected_node = closest_node
            self.dragging_node = closest_node
            self.drag_moved = False
            if hasattr(self, 'node_selected_callback'):
                self.node_selected_callback(closest_node)

    def _on_release(self, event):
        self.dragging_node = None
        self.drag_moved = False

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == 'up' else 1.25
        x_limits = self.ax.get_xlim()
        y_limits = self.ax.get_ylim()
        self.ax.set_xlim(
            event.xdata - (event.xdata - x_limits[0]) * scale,
            event.xdata + (x_limits[1] - event.xdata) * scale
        )
        self.ax.set_ylim(
            event.ydata - (event.ydata - y_limits[0]) * scale,
            event.ydata + (y_limits[1] - event.ydata) * scale
        )
        self.draw_idle()

    def _on_hover(self, event):
        if self.dragging_node is not None:
            if event.xdata is not None and event.ydata is not None:
                self.pos[self.dragging_node] = np.array([event.xdata, event.ydata])
                self.drag_moved = True
                self.render_manifold(
                    nx.to_numpy_array(self.G, dtype=int),
                    self.euler_path, self.active_step, self.current_W,
                    self.fault_node, self.node_labels
                )
            return
        if event.xdata is None or event.ydata is None or self.pos is None or self.G is None:
            QToolTip.hideText()
            return

        p = np.array([event.xdata, event.ydata])
        node_radius = 0.22 if len(self.G) == 16 else 0.28
        
        for node, coords in self.pos.items():
            if np.linalg.norm(p - np.array(coords)) < node_radius:
                in_deg = self.G.in_degree(node)
                out_deg = self.G.out_degree(node)
                eq_status = "BALANCED" if in_deg == out_deg else "UNBALANCED FAULT"
                label_name = self.node_labels.get(node, f"N{node}")

                targets = []
                if self.current_W is not None:
                    targets = [j for j in range(len(self.current_W)) if self.current_W[node][j] == 1]

                tooltip_text = (
                    f"COOLANT JUNCTION: {label_name} (ID: {node})\n"
                    f"Inlet conduits: {in_deg}\n"
                    f"Outlet conduits: {out_deg}\n"
                    f"Flow state: {eq_status}\n"
                    f"Reachability (Warshall): {targets}"
                )
                QToolTip.showText(QCursor.pos(), tooltip_text, self)
                return

        edge_threshold = 0.08
        for u, v in self.G.edges():
            a = np.array(self.pos[u])
            b = np.array(self.pos[v])
            
            ba = b - a
            l2 = np.dot(ba, ba)
            dist = np.linalg.norm(p - a) if l2 == 0 else np.linalg.norm(p - (a + max(0.0, min(1.0, np.dot(p - a, ba) / l2)) * ba))

            if dist < edge_threshold:
                is_active = (u, v) in self.euler_path[:self.active_step + 1] if self.active_step >= 0 else False
                state_str = "ACTIVE BLUE COOLANT FLOW" if is_active else "IDLE CONDUIT"
                u_name = self.node_labels.get(u, f"N{u}")
                v_name = self.node_labels.get(v, f"N{v}")
                
                tooltip_text = (
                    f"COOLANT CONDUIT: {u_name} -> {v_name}\n"
                    f"Source: {u_name}\n"
                    f"Destination: {v_name}\n"
                    f"Status: {state_str}\n"
                    "Direction: Unidirectional Micro-channel"
                )
                QToolTip.showText(QCursor.pos(), tooltip_text, self)
                return

        QToolTip.hideText()


class KPICard(QFrame):
    def __init__(self, title: str, initial_value: str):
        super().__init__()
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)
        
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("MetricTitle")
        
        self.val_label = QLabel(initial_value)
        self.val_label.setObjectName("MetricValue")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.val_label)

    def update_card(self, value: str):
        self.val_label.setText(value)


class SimventThermalCAD(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIMVENT Thermal Studio v7.0 | EV Manifold Simulation Engine")
        self.resize(1440, 900)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(self._get_workspace_stylesheet())

        self.current_matrix = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0]
        ])
        self.original_matrix = self.current_matrix.copy()
        self.node_labels = {
            0: "Main_Coolant_Pump",
            1: "3Way_Thermostat",
            2: "HV_PTC_Heater",
            3: "ColdPlate_Front",
            4: "ColdPlate_Rear",
            5: "HVAC_Chiller"
        }
        
        self.engine = MathWorksThermalEngine(self.current_matrix)
        self.euler_edges = []
        self.fault_node = -1
        self.anim_index = -1
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._step_flow_animation)

        self._build_ui()
        self.recompute_all()

    def _build_ui(self):
        main_widget = QWidget()
        main_widget.setObjectName("MainWorkspace")
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        top_bar = QFrame()
        top_bar.setObjectName("HeaderPanel")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(18, 12, 18, 12)

        brand_title = QLabel("SIMVENT | EV COOLING TOPOLOGY SOLVER")

        import_btn = QPushButton("IMPORT EV TOPOLOGY (.JSON / .CSV)")
        import_btn.clicked.connect(self.import_file_data)

        self.anim_btn = QPushButton("SIMULATE COOLANT FLOW")
        self.anim_btn.clicked.connect(self.toggle_flow_simulation)

        top_layout.addWidget(brand_title)
        top_layout.addSpacing(16)
        top_layout.addWidget(import_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.anim_btn)
        root_layout.addWidget(top_bar)

        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(12)
        
        self.kpi_planarity = KPICard("Cold Plate Layout", "CHECKING")
        self.kpi_eulerian = KPICard("Manifold Circuit Path", "CHECKING")
        self.kpi_reachability = KPICard("Thermal Reachability Ratio", "0%")
        self.kpi_modules = KPICard("Equivalence Classes", "0 Submodules")

        kpi_layout.addWidget(self.kpi_planarity)
        kpi_layout.addWidget(self.kpi_eulerian)
        kpi_layout.addWidget(self.kpi_reachability)
        kpi_layout.addWidget(self.kpi_modules)
        root_layout.addLayout(kpi_layout)

        workbench_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        cad_container = QFrame()
        cad_container.setObjectName("WorkspacePanel")
        cad_layout = QVBoxLayout(cad_container)
        cad_layout.setContentsMargins(16, 14, 16, 16)
        cad_layout.setSpacing(10)
        
        vp_title = QLabel("REALTIME FLOW MANIFOLD VIEWPORT (HOVER FOR TOOLTIPS)")
        vp_title.setObjectName("SectionTitle")
        cad_layout.addWidget(vp_title)

        self.canvas = CADManifoldCanvas(self)
        self.canvas.set_on_node_selected_callback(self.on_node_clicked)
        cad_layout.addWidget(self.canvas, stretch=1)
        workbench_splitter.addWidget(cad_container)

        right_container = QFrame()
        right_container.setObjectName("WorkspacePanel")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(16, 14, 16, 16)
        right_layout.setSpacing(10)
        
        right_title = QLabel("ADJACENCY MATRIX")
        right_title.setObjectName("SectionTitle")

        matrix_header = QHBoxLayout()
        matrix_header.addWidget(right_title, stretch=1)
        reset_matrix_btn = QPushButton("RESET")
        reset_matrix_btn.setObjectName("ResetButton")
        reset_matrix_btn.setFixedWidth(104)
        reset_matrix_btn.clicked.connect(self.reset_matrix)
        clear_matrix_btn = QPushButton("CLEAR")
        clear_matrix_btn.setObjectName("ClearButton")
        clear_matrix_btn.setFixedWidth(104)
        clear_matrix_btn.clicked.connect(self.clear_matrix)
        matrix_header.addWidget(reset_matrix_btn)
        matrix_header.addWidget(clear_matrix_btn)
        right_layout.addLayout(matrix_header)

        self.matrix_table = QTableWidget()
        self.matrix_table.setObjectName("MatrixTable")
        self.matrix_table.cellClicked.connect(self.on_matrix_cell_clicked)

        self.inspector_card = QFrame()
        self.inspector_card.setObjectName("InspectorPanel")
        insp_layout = QVBoxLayout(self.inspector_card)
        insp_layout.setContentsMargins(14, 12, 14, 12)
        self.inspector_label = QLabel("Click any node on the graph or run simulation to evaluate flow paths.")
        self.inspector_label.setObjectName("InspectorText")
        self.inspector_label.setWordWrap(True)
        insp_layout.addWidget(self.inspector_label)

        matrix_help = QLabel("Adjacency Matrix M_R (Click a cell to toggle a conduit)")
        matrix_help.setObjectName("HelperText")
        right_layout.addWidget(matrix_help)
        right_layout.addWidget(self.matrix_table, stretch=2)
        inspector_title = QLabel("THERMAL JUNCTION INSPECTOR")
        inspector_title.setObjectName("SectionTitle")
        right_layout.addWidget(inspector_title)
        right_layout.addWidget(self.inspector_card, stretch=1)

        workbench_splitter.addWidget(right_container)
        workbench_splitter.setSizes([850, 590])
        root_layout.addWidget(workbench_splitter, stretch=1)

    def import_file_data(self):
        file_dialog = QFileDialog(self, "Import EV Manifold Data File")
        file_dialog.setNameFilter("EV Topology Files (*.json *.csv)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
        else:
            file_path = ""
        if not file_path:
            return

        try:
            matrix, labels = load_topology_file(file_path)
            self.anim_timer.stop()
            self.anim_btn.setText("SIMULATE COOLANT FLOW")
            self.anim_index = -1
            self.current_matrix = matrix
            self.original_matrix = matrix.copy()
            self.node_labels = labels
            self.recompute_all()
        except Exception as error:
            self.inspector_label.setText(f"Import failed: {error}")

    def recompute_all(self):
        self.engine.update_matrix(self.current_matrix)
        
        W, _ = self.engine.compute_warshall()
        eq_classes = self.engine.compute_equivalence_classes(W)
        is_planar = self.engine.check_planarity()
        is_eulerian, self.euler_edges, self.fault_node = self.engine.find_flow_traversal()

        if is_planar:
            self.kpi_planarity.update_card("SINGLE-LAYER PLANAR")
        else:
            self.kpi_planarity.update_card("NON-PLANAR (3D PLATE)")

        if is_eulerian:
            self.kpi_eulerian.update_card("EULERIAN ROUTE VALID")
        else:
            self.kpi_eulerian.update_card("PARTIAL FLOW (FAULT)")

        total_pairs = self.engine.n * self.engine.n
        reachable_pairs = np.sum(W)
        ratio = int((reachable_pairs / total_pairs) * 100)
        self.kpi_reachability.update_card(f"{ratio}% Reachable")

        self.kpi_modules.update_card(f"{len(eq_classes)} Partition Groups")

        self._populate_matrix_table(self.matrix_table, self.current_matrix)
        self.canvas.render_manifold(self.current_matrix, self.euler_edges, self.anim_index, W, self.fault_node, self.node_labels)

    def on_matrix_cell_clicked(self, row: int, col: int):
        self.current_matrix[row][col] = 1 - self.current_matrix[row][col]
        self.recompute_all()

    def reset_matrix(self):
        self.anim_timer.stop()
        self.anim_btn.setText("SIMULATE COOLANT FLOW")
        self.anim_index = -1
        self.canvas.selected_node = None
        self.current_matrix = self.original_matrix.copy()
        self.recompute_all()
        self.inspector_label.setText(
            "Original topology restored from the loaded file. The matrix matches its source configuration."
        )

    def clear_matrix(self):
        self.anim_timer.stop()
        self.anim_btn.setText("SIMULATE COOLANT FLOW")
        self.anim_index = -1
        self.canvas.selected_node = None
        self.current_matrix = np.zeros_like(self.current_matrix)
        self.recompute_all()
        self.inspector_label.setText(
            "Matrix cleared: all coolant paths have been removed."
        )

    def on_node_clicked(self, node_idx: int):
        W, _ = self.engine.compute_warshall()
        G = nx.from_numpy_array(self.current_matrix, create_using=nx.DiGraph)
        
        in_deg = G.in_degree(node_idx)
        out_deg = G.out_degree(node_idx)
        label_name = self.node_labels.get(node_idx, f"N{node_idx}")
        reachable_targets = [j for j in range(self.engine.n) if W[node_idx][j] == 1]
        
        info_text = (
            f"<b>ELEMENT INSPECTOR: {label_name} (ID: {node_idx})</b><br><br>"
            f"• <b>Inlet Coolant Conduit Count (In-Degree):</b> {in_deg}<br>"
            f"• <b>Outlet Coolant Conduit Count (Out-Degree):</b> {out_deg}<br>"
            f"• <b>Flow Equilibrium Status:</b> {'BALANCED IN/OUT' if in_deg == out_deg else 'PRESSURE GRADIENT UNBALANCED'}<br>"
            f"• <b>Thermal Downstream Reachability (Warshall):</b> {reachable_targets}"
        )
        self.inspector_label.setText(info_text)
        self.canvas.selected_node = node_idx
        self.canvas.render_manifold(self.current_matrix, self.euler_edges, self.anim_index, W, self.fault_node, self.node_labels)

    def toggle_flow_simulation(self):
        if self.anim_timer.isActive():
            self.anim_timer.stop()
            self.anim_btn.setText("SIMULATE COOLANT FLOW")
            self.anim_index = -1
            W, _ = self.engine.compute_warshall()
            self.canvas.render_manifold(self.current_matrix, self.euler_edges, -1, W, self.fault_node, self.node_labels)
        else:
            if not self.euler_edges:
                self.inspector_label.setText("<b style='color:#C62828;'>FLOW ERROR: Completely disconnected manifold graph.</b>")
                return

            self.anim_index = 0
            W, _ = self.engine.compute_warshall()
            self.canvas.render_manifold(self.current_matrix, self.euler_edges, 0, W, self.fault_node, self.node_labels)

            if len(self.euler_edges) == 1:
                self.anim_btn.setText("RESTART SIMULATION")
                self._update_inspector_on_finish()
            else:
                self.anim_btn.setText("STOP SIMULATION")
                self.anim_timer.start(450)

    def _step_flow_animation(self):
        if self.anim_index < len(self.euler_edges) - 1:
            self.anim_index += 1

        W, _ = self.engine.compute_warshall()
        self.canvas.render_manifold(self.current_matrix, self.euler_edges, self.anim_index, W, self.fault_node, self.node_labels)

        if self.anim_index == len(self.euler_edges) - 1:
            self.anim_timer.stop()
            self.anim_btn.setText("RESTART SIMULATION")
            self._update_inspector_on_finish()

    def _update_inspector_on_finish(self):
        if self.fault_node != -1:
            fault_name = self.node_labels.get(self.fault_node, f"N{self.fault_node}")
            self.inspector_label.setText(
                f"<b style='color:#C62828;'>PARTIAL FLOW STOPPED AT {fault_name}</b><br><br>"
                f"• <b>Reason:</b> Coolant flow reached junction {fault_name} but cannot continue.<br>"
                f"• <b>Diagnostic:</b> Zero valid outlet conduits exist from this junction.<br>"
                f"• <b>Status:</b> Initial {len(self.euler_edges)} flow segments verified working (Blue Arrows)."
            )
        else:
            self.inspector_label.setText(
                f"<b style='color:#248A3D;'>FULL EULERIAN FLOW CIRCUIT COMPLETE</b><br><br>"
                f"• <b>Status:</b> All {len(self.euler_edges)} coolant conduits successfully traversed.<br>"
                f"• <b>Equilibrium:</b> Uniform pressure and continuous thermal dissipation verified."
            )

    def _populate_matrix_table(self, table: QTableWidget, matrix: np.ndarray):
        rows, cols = matrix.shape
        table.setRowCount(rows)
        table.setColumnCount(cols)
        table.setHorizontalHeaderLabels([self.node_labels.get(i, f"N{i}") for i in range(cols)])
        table.setVerticalHeaderLabels([self.node_labels.get(i, f"N{i}") for i in range(rows)])

        for i in range(rows):
            for j in range(cols):
                val = str(matrix[i][j])
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if val == "1":
                    item.setBackground(QColor("#007AFF"))
                    item.setForeground(QColor("#FFFFFF"))
                else:
                    item.setBackground(QColor("#FFFFFF"))
                    item.setForeground(QColor("#6E6E73"))
                table.setItem(i, j, item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _get_workspace_stylesheet(self) -> str:
        return """
        QMainWindow {
            background-color: #F5F5F7;
        }
        QWidget {
            color: #1D1D1F;
            font-family: "Helvetica Neue", Arial;
            font-size: 12px;
        }
        QWidget#MainWorkspace {
            background-color: #F5F5F7;
        }
        QFrame#HeaderPanel, QFrame#WorkspacePanel {
            background-color: #FFFFFF;
            border: 1px solid #D2D2D7;
            border-radius: 12px;
        }
        QFrame#HeaderPanel {
            border-left: 4px solid #007AFF;
        }
        QFrame#MetricCard {
            background-color: #FFFFFF;
            border: 1px solid #D2D2D7;
            border-top: 3px solid #007AFF;
            border-radius: 11px;
        }
        QLabel#MetricTitle, QLabel#HelperText {
            color: #6E6E73;
            font-size: 10px;
            font-weight: 600;
        }
        QLabel#MetricValue {
            color: #1D1D1F;
            font-size: 15px;
            font-weight: 700;
        }
        QLabel#SectionTitle {
            color: #1D1D1F;
            font-size: 12px;
            font-weight: 700;
        }
        QFrame#InspectorPanel {
            background-color: #F5F5F7;
            border: 1px solid #D2D2D7;
            border-left: 3px solid #64A8FF;
            border-radius: 9px;
        }
        QLabel#InspectorText {
            color: #3A3A3C;
            font-size: 12px;
        }
        QPushButton {
            background-color: #FFFFFF;
            color: #1D1D1F;
            border: 1px solid #C7C7CC;
            border-radius: 7px;
            padding: 7px 12px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #F2F7FF;
            border-color: #64A8FF;
        }
        QPushButton:pressed {
            background-color: #E5F0FF;
        }
        QPushButton#ResetButton {
            background-color: #F2F7FF;
            border-color: #B8D7FF;
            color: #0060DF;
        }
        QPushButton#ClearButton {
            background-color: #FFF5F4;
            border-color: #FFC9C5;
            color: #C62828;
        }
        QPushButton#ClearButton:hover {
            background-color: #FFE8E6;
            border-color: #FF8A80;
        }
        QTableWidget#MatrixTable {
            background-color: #FFFFFF;
            alternate-background-color: #F7F7F9;
            color: #3A3A3C;
            gridline-color: #E5E5EA;
            border: 1px solid #D2D2D7;
            border-radius: 8px;
            selection-background-color: #DCEBFF;
            selection-color: #1D1D1F;
        }
        QTableWidget#MatrixTable QHeaderView::section {
            background-color: #F2F2F7;
            color: #3A3A3C;
            border: none;
            border-right: 1px solid #D2D2D7;
            border-bottom: 1px solid #D2D2D7;
            padding: 5px;
            font-weight: 600;
        }
        QTableWidget#MatrixTable::item {
            padding: 3px;
        }
        QToolTip {
            color: #1D1D1F;
            background-color: #FFFFFF;
            border: 1px solid #C7C7CC;
            padding: 7px;
        }
        QSplitter::handle {
            background-color: #D2D2D7;
        }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimventThermalCAD()
    window.show()
    sys.exit(app.exec())