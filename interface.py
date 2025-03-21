import pyqtgraph as pg
import networkx as nx
import pickle
import sys
import clickhouse_connect
from utils import page_rank_nibble
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QGridLayout,
    QLabel, QMessageBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QHeaderView, QAbstractItemView, QComboBox, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QMovie
from graph_utils import load_graph
from interface_utils import *
from functools import partial

class PageRankNibbleApp(QMainWindow):

    def setup_table_widget(self):
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)

    def set_top_left_group_box(self):
        self.top_left_group_box = CustomGroupBox("Product ID")
        self.top_left_group_box_layout = QVBoxLayout()
        self.top_left_group_box.setLayout(self.top_left_group_box_layout)

        self.product_id_entry = CustomLineEdit("Enter ID here...")
        self.top_left_group_box_layout.addWidget(self.product_id_entry, alignment=Qt.AlignCenter)

        self.submit_button = CustomPushButton("Submit")
        self.submit_button.clicked.connect(partial(self.loading_cluster_informations, True))

        self.top_left_group_box_layout.addWidget(self.submit_button, alignment=Qt.AlignCenter)

    def set_top_right_group_box_style(self):
        for label in [self.product_index_label, self.description_label, self.category_label]:
            label.setFont(self.label_font)
            label.setStyleSheet("color: rgb(255, 255, 153); padding: 5px;")

        for value in [self.product_index_value, self.description_value, self.category_value]:
            value.setFont(self.value_font)
            value.setStyleSheet("""
                color: #555555;
                background-color: #f0f0f0;
                padding: 5px;
                border-radius: 5px;
                """)

    def set_top_right_group_box(self):
        self.top_right_group_box = CustomGroupBox("Product Information")
        self.top_right_group_box_layout = QVBoxLayout()
        self.top_right_group_box.setLayout(self.top_right_group_box_layout)

        # Create a form layout
        self.info_form_layout = QFormLayout()
        self.info_form_layout.setLabelAlignment(Qt.AlignRight)
        self.info_form_layout.setFormAlignment(Qt.AlignLeft)

        # Set font and style for labels and values
        self.label_font = QFont("Arial", 12, QFont.Bold)
        self.value_font = QFont("Arial", 12)

        self.product_index_label = QLabel("Product Index:")
        self.product_index_value = QLabel("")

        self.description_label = QLabel("Description:")
        self.description_value = QLabel("")

        self.category_label = QLabel("Category:")
        self.category_value = QLabel("")

        self.set_top_right_group_box_style()
        # Add rows to the form layout
        self.info_form_layout.addRow(self.product_index_label, self.product_index_value)
        self.info_form_layout.addRow(self.description_label, self.description_value)
        self.info_form_layout.addRow(self.category_label, self.category_value)

        self.top_right_group_box_layout.addLayout(self.info_form_layout)

    def set_bottom_left_group_box_style(self):
        self.table_widget.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ddd;
                background-color: #ffffff;
                padding: 10px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                padding: 5px;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 5px;
            }
        """)

    def set_bottom_left_group_box(self):
        self.bottom_left_group_box = CustomGroupBox("Product Descriptions and Categories")
        self.bottom_left_group_box_layout = QVBoxLayout()
        self.bottom_left_group_box.setLayout(self.bottom_left_group_box_layout)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)  # Three columns: Index, Description and Category
        self.table_widget.setHorizontalHeaderLabels(["Id","Description", "Category"])

        self.set_bottom_left_group_box_style()
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Set to read-only mode
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_widget.setWordWrap(True)

        self.setup_table_widget()
        self.bottom_left_group_box_layout.addWidget(self.table_widget)

    def set_bottom_right_group_box(self):
        self.node_count_label = QLabel("Number of cluster nodes:")
        self.node_count_label.setFont(self.label_font)
        self.node_count_label.setStyleSheet("color: rgb(255, 255, 153); padding: 5px;")

        self.node_count_combo = QComboBox()
        self.node_count_combo.setStyleSheet("""
            QComboBox {
                min-width: 70px;  /* Minimum width */
                max-width: 70px;  /* Maximum width */
            }
            QComboBox::drop-down {
                width: 20px;  /* Width of the dropdown arrow */
            }
        """)
        self.node_count_combo.addItems(['5', '10', '15', '20', '25', '30'])
        self.node_count_combo.setCurrentText('30') 
        self.node_count_combo.currentIndexChanged.connect(partial(self.loading_cluster_informations, False))

        # Create a horizontal layout to hold the label and combo box
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.node_count_label, alignment=Qt.AlignCenter)
        h_layout.addWidget(self.node_count_combo, alignment=Qt.AlignCenter)

        # Create the group box and layout if not already created
        self.bottom_right_group_box = CustomGroupBox("Cluster Graph")
        self.bottom_right_group_box_layout = QVBoxLayout()
        # Add the horizontal layout to the group box layout with center alignment
        self.bottom_right_group_box_layout.addLayout(h_layout)

        # Create a placeholder for the cluster graph
        self.cluster_graph_placeholder = QWidget()  # A placeholder widget for the cluster graph
        self.bottom_right_group_box_layout.addWidget(self.cluster_graph_placeholder)

        # Create a PyQtGraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.bottom_right_group_box_layout.addWidget(self.plot_widget)
        
        self.plot_widget.setBackground(QColor('white'))
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.showGrid(x=False, y=False)  
        self.plot_widget.autoRange()

        self.bottom_right_group_box.setLayout(self.bottom_right_group_box_layout)
        
    def __init__(self, graph_path="graph.pkl", dict_path="products_dict.pkl", **client_params):
        super().__init__()
        # Load the graph
        self.graph = load_graph(graph_path)
        # Initialize data
        with open(dict_path, 'rb') as f:
            data = pickle.load(f)
            self.id_to_index = data['id_to_index']
            self.index_to_id = data['index_to_id']
        self.client = clickhouse_connect.get_client(
            host=client_params["host"], port=client_params["port"],
            username=client_params["user"], password=client_params["psw"],
            database=client_params["database"]
        )
        self.table_name = client_params["table_name"]
        self.setWindowTitle("PageRank Nibble Interface")
        # Set initial size of the window
        self.resize(1200, 800)
        # Set application-wide font and background color
        palette = QPalette()
        palette.setColor(QPalette.Background, QColor("#001f3f"))  # Deep blue color
        self.setPalette(palette)
        QApplication.setStyle("Fusion")

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # Text box and button in a group box at top left
        self.set_top_left_group_box()
        self.main_layout.addWidget(self.top_left_group_box, 0, 0)

        # Product Info Card at top right
        self.set_top_right_group_box()
        self.main_layout.addWidget(self.top_right_group_box, 0, 1)

        # Description and Categories Table at bottom left
        self.set_bottom_left_group_box()
        self.main_layout.addWidget(self.bottom_left_group_box, 1, 0)
        
        # Cluster Graph at bottom right
        self.set_bottom_right_group_box()
        self.main_layout.addWidget(self.bottom_right_group_box, 1, 1)

        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setColumnStretch(1, 1)

        self.loadingLabel = QLabel(self.main_widget)
        self.loadingLabel.setAlignment(Qt.AlignCenter)
        self.loadingLabel.setVisible(False)

        self.loadingMovie = QMovie("an.gif")
        self.loadingLabel.setMovie(self.loadingMovie)

    def block_plot_widget(self, block=True):
        if block:
            self.plot_widget.setMouseEnabled(False, False) # Block user interaction
            self.overlay = QFrame(self.plot_widget)
            self.overlay.setStyleSheet("background-color: rgba(255, 255, 255, 128);")
            self.overlay.setGeometry(self.plot_widget.rect())
            self.overlay.show()
        else:
            self.plot_widget.setMouseEnabled(True, True)
            if hasattr(self, 'overlay'):
                self.overlay.hide()
                self.overlay.deleteLater()

    def block_info_form_layout(self, block=True):
        for i in range(self.info_form_layout.rowCount()):
            value_widget = self.info_form_layout.itemAt(i, QFormLayout.FieldRole).widget()

            if block:
                value_widget.setEnabled(False)
                value_overlay = QFrame(value_widget)
                value_overlay.setStyleSheet("background-color: rgba(255, 255, 255, 128);")
                value_overlay.setGeometry(value_widget.rect())
                value_overlay.show()
                value_widget.overlay = value_overlay
            else:
                value_widget.setEnabled(True)
                if hasattr(value_widget, 'overlay'):
                    value_widget.overlay.hide()
                    value_widget.overlay.deleteLater()
                    del value_widget.overlay

    def loading_cluster_informations(self, calculation=True):
        self.table_widget.clearSelection()
        if calculation == False:
            if not hasattr(self, 'cluster') or len(self.cluster) == 0:
                return
        self.main_widget.setEnabled(False)
        self.block_plot_widget()
        if calculation == True:
            self.block_info_form_layout(True)
        
        self.show_loading_animation(self.loadingLabel, self.main_widget)
        self.workerThread = WorkerThread(self, calculation)
        self.workerThread.finished.connect(self.on_operation_finished)
        self.workerThread.start()

    def show_loading_animation(self, loadingLabel, referenceWidget):
        loadingLabel.setGeometry(referenceWidget.geometry())
        loadingLabel.setVisible(True)
        loadingLabel.movie().start()

    def on_operation_finished(self, desc, cluster_graph):
        self.hide_loading_animation(self.loadingLabel)
        if self.workerThread.calculation == True:
            self.block_info_form_layout(False)
        
        self.block_plot_widget(False)
        self.main_widget.setEnabled(True)

        product_id = self.product_id_entry.text()
        self.display_graph(cluster_graph)
        self.display_descriptions(desc)
        if self.workerThread.calculation == True:
            self.display_product_info(self.id_to_index[product_id], desc[product_id])
        
    def hide_loading_animation(self, loadingLabel):
        loadingLabel.movie().stop()
        loadingLabel.setVisible(False)

    def calculate_cluster_informations(self, cluster):
        num_nodes = int(self.node_count_combo.currentText())
        cluster = cluster[ : min(num_nodes, len(cluster))]
        desc = self.get_cluster_desc(cluster)
        cluster_graph = self.get_cluster_graph(cluster)
        #self.display_graph(cluster_graph)
        #self.display_descriptions(desc)
        return desc, cluster_graph   
 
    def run_page_rank_nibble(self):
        product_id = self.product_id_entry.text()
        if not product_id:
            QMessageBox.warning(self, "Input Error", "Please enter a product ID.")
            return

        try:
            product_index = self.id_to_index[product_id]
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"{str(e)} isn't present in the graph")
            return 
        n = self.graph.number_of_nodes()
        phi = 0.1
        beta = 0.9
        epsilon = 2e-05
        #epsilon = 0.0003
        try:
            seed, cluster = page_rank_nibble(self.graph, n, phi, beta, epsilon, "unweighted", product_index)
        except Exception as e:
            QMessageBox.critical(self, "Page Rank Nibble Error", "Error during the cluster algorithm")
            return

        self.cluster = cluster
        #desc = self.display_cluster_informations(self.cluster)
        #self.display_product_info(self.id_to_index[product_id], desc[product_id])

    def get_cluster_graph(self, cluster):
        subgraph = self.graph.subgraph(cluster)
        subgraph = nx.Graph(subgraph)
        return subgraph

    def get_cluster_desc(self, cluster):
        ids = [self.index_to_id[index] for index in cluster]
        #ids = ', '.join(map(str, ids))
        #query = f"SELECT DISTINCT descr_prod, descr_rep FROM {self.table_name} WHERE cod_prod IN ({ids});"
        #result = self.client.query(query)
        query_create_temp = """CREATE TEMPORARY TABLE temp_ids (cod_prod String, position INT);"""
        self.client.query(query_create_temp)
        query_insert_temp = f"""INSERT INTO temp_ids (cod_prod, position) VALUES{", ".join([f"({id}, {index})" for index, id in enumerate(ids)])};"""
        self.client.query(query_insert_temp)
        query_main = f"""SELECT DISTINCT t.descr_prod, t.descr_rep FROM {self.table_name} t JOIN temp_ids temp ON t.cod_prod = temp.cod_prod ORDER BY temp.position;"""
        result = self.client.query(query_main)
        query_drop_temp = "DROP TEMPORARY TABLE temp_ids;"
        self.client.query(query_drop_temp)
        desc = {}
        for row in result.result_rows:
            row_split = row[0].split()
            des = ' '.join(row_split[1:])
            words = des.split()
            if '.' not in words[0]:
                words[0] = '*' * len(words[0])
            des = ' '.join(words)
            desc[row_split[0]] = (des, row[1])
        return desc

    def display_product_info(self, product_id, product_info):
        self.product_index_value.setText(str(product_id))
        self.description_value.setText(product_info[0])
        self.category_value.setText(product_info[1])

    def display_descriptions(self, descriptions):
        self.table_widget.setRowCount(len(descriptions))
        for row, (product, (descr, cat)) in enumerate(descriptions.items()):
            index_item = QTableWidgetItem(str(product))
            descr_item = QTableWidgetItem(descr)
            cat_item = QTableWidgetItem(cat)

            # Center align the text in all columns
            index_item.setTextAlignment(Qt.AlignCenter)
            descr_item.setTextAlignment(Qt.AlignCenter)
            cat_item.setTextAlignment(Qt.AlignCenter)

            self.table_widget.setItem(row, 0, index_item)
            self.table_widget.setItem(row, 1, descr_item)
            self.table_widget.setItem(row, 2, cat_item)
            self.table_widget.resizeRowsToContents()

    def on_node_click(self, plot, points, nodes):
        self.clear_row_highlighting()
        for point in points:
            node_index = point.index()  # Get the index of the clicked node
            node_id = int(self.index_to_id[nodes[node_index]])  # Get the product ID from the nodes list
            # Find the row in the table that corresponds to the clicked node
            for row in range(self.table_widget.rowCount()):
                item = self.table_widget.item(row, 0)  # Get the item in the first column (product ID)
                if item is not None:  # Ensure the item exists
                    table_product_id = int(item.text())  # Get the product ID from the table
                    if table_product_id == node_id:
                        self.highlight_row(row)
                        break

    def highlight_row(self, row):
        for col in range(self.table_widget.columnCount()):
            item = self.table_widget.item(row, col)
            if item is not None:
                item.setBackground(QColor('#FFDD94'))  # Set the highlight color
                item.setForeground(QColor('black'))  # Set the text color

        # Scroll to the highlighted row
        item_to_scroll = self.table_widget.item(row, 0)
        if item_to_scroll is not None:
            self.table_widget.scrollToItem(item_to_scroll, QAbstractItemView.PositionAtCenter)

    def clear_row_highlighting(self):
        for row in range(self.table_widget.rowCount()):
            for col in range(self.table_widget.columnCount()):
                item = self.table_widget.item(row, col)
                if item is not None:
                    item.setBackground(QColor('white'))  # Reset background color
                    item.setForeground(QColor('black'))  # Reset text color
    
    def display_graph(self, graph):
        from pyqtgraph.Qt import QtGui
        import functools
        
        self.plot_widget.clear()
        
        pos = nx.spring_layout(graph, seed=42)
        edges = list(graph.edges())
        nodes = list(graph.nodes())
        
        # Extract node positions
        x = [pos[node][0] for node in nodes]
        y = [pos[node][1] for node in nodes]
        
        for edge in edges:
            x_start, y_start = pos[edge[0]]
            x_end, y_end = pos[edge[1]]
            self.plot_widget.plot([x_start, x_end], [y_start, y_end], pen=pg.mkPen('gray', width=1))
        
        scatter = pg.ScatterPlotItem(x=x, y=y, symbol='o', size=20, brush=pg.mkBrush('skyblue'))
        self.plot_widget.addItem(scatter)
        
        for node in nodes:
            x_pos, y_pos = pos[node]
            text_item = pg.TextItem(str(node), color='black', anchor=(0.5, 0.5))
            text_item.setPos(x_pos, y_pos)
            text_item.setFont(QtGui.QFont('Arial', 10))
            self.plot_widget.addItem(text_item)
        
        self.plot_widget.autoRange()
        scatter.sigClicked.connect(functools.partial(self.on_node_click, nodes=nodes))

if __name__ == "__main__":
    client_params = {
        "host": 'localhost',
        "port": 8123,
        "database": 'mydb',
        "user": 'evision',
        "psw": 'Evision!',
        "table_name": 'dati_scontrini'
    }
    app = QApplication(sys.argv)
    window = PageRankNibbleApp(**client_params)
    window.show()
    sys.exit(app.exec_())
