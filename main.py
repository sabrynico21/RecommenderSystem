from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QHeaderView
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os
import sys
import clickhouse_connect
from utils import page_rank_nibble

def get_cluster_graph(graph, cluster):
    subgraph = graph.subgraph(cluster)
    subgraph = nx.Graph(subgraph)
    print("Nodes in subgraph:", subgraph.nodes())
    print("Edges in subgraph:", subgraph.edges())
    return subgraph

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QHeaderView
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os
import sys
import clickhouse_connect
from utils import page_rank_nibble

def get_cluster_graph(graph, cluster):
    subgraph = graph.subgraph(cluster)
    subgraph = nx.Graph(subgraph)
    print("Nodes in subgraph:", subgraph.nodes())
    print("Edges in subgraph:", subgraph.edges())
    return subgraph

class PageRankNibbleApp(QMainWindow):
    def __init__(self, graph_path="graph.pkl", dict_path="products_dict.pkl", **client_params):
        super().__init__()

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
        self.graph_path = graph_path

        self.setWindowTitle("PageRank Nibble Interface")
        
        # Set initial size of the window
        self.resize(1200, 800)

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        # Top layout with text box, button, and product info table
        self.top_layout = QHBoxLayout()

        # Left side of the top layout
        self.controls_layout = QVBoxLayout()
        self.controls_layout.setAlignment(Qt.AlignTop)

        self.label = QLabel("Enter Product ID:")
        self.controls_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.product_id_entry = QLineEdit()
        self.controls_layout.addWidget(self.product_id_entry, alignment=Qt.AlignCenter)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.run_page_rank_nibble)
        self.controls_layout.addWidget(self.submit_button, alignment=Qt.AlignCenter)

        self.top_layout.addLayout(self.controls_layout, 1)

        # Product Info Table
        self.info_table_widget = QTableWidget()
        self.info_table_widget.setColumnCount(3)  # Three columns: Product Index, Description, Category
        self.info_table_widget.setHorizontalHeaderLabels(["Product Index", "Description", "Category"])
        header_info_table = self.info_table_widget.horizontalHeader()
        header_info_table.setSectionResizeMode(0, QHeaderView.Stretch)
        header_info_table.setSectionResizeMode(1, QHeaderView.Stretch)
        header_info_table.setSectionResizeMode(2, QHeaderView.Stretch)
        self.info_table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.info_table_widget.setWordWrap(True)
        self.top_layout.addWidget(self.info_table_widget, 1)

        self.main_layout.addLayout(self.top_layout)

        # Bottom layout with graph and description table
        self.bottom_layout = QHBoxLayout()

        # Cluster Graph
        self.graph_frame = QWidget()
        self.graph_layout = QVBoxLayout()
        self.graph_frame.setLayout(self.graph_layout)
        self.graph_title = QLabel("Cluster Graph")
        self.graph_title.setAlignment(Qt.AlignCenter)
        self.graph_layout.addWidget(self.graph_title)
        
        self.bottom_layout.addWidget(self.graph_frame, 1)

        # Description and Categories table
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)  # Two columns: Description and Category
        self.table_widget.setHorizontalHeaderLabels(["Description", "Category"])
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_widget.setWordWrap(True)

        self.bottom_layout.addWidget(self.table_widget, 1)

        self.main_layout.addLayout(self.bottom_layout)

        # Initially hide the graph and tables
        self.graph_frame.setVisible(False)
        self.info_table_widget.setVisible(False)
        self.table_widget.setVisible(False)

    def run_page_rank_nibble(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, "rb") as f:
                graph = pickle.load(f)
            print("Graph loaded successfully.")
        else:
            print(f"File {self.graph_path} does not exist.")
            return

        product_id = self.product_id_entry.text()

        if not product_id:
            QMessageBox.warning(self, "Input Error", "Please enter a product ID.")
            return

        n = graph.number_of_nodes()
        phi = 0.1
        beta = 0.9
        epsilon = 2e-05

        try:
            seed, cluster = page_rank_nibble(graph, n, phi, beta, epsilon, "unweighted", self.id_to_index[product_id])
            desc = self.get_cluster_desc(cluster)
            cluster_graph = get_cluster_graph(graph, cluster)
            self.display_graph(cluster_graph)
            self.display_descriptions(desc)
            self.display_product_info(self.id_to_index[product_id], desc[product_id])

            # Show the frames after button click
            self.graph_frame.setVisible(True)
            header_info_table = self.info_table_widget.horizontalHeader()
            header_info_table.setSectionResizeMode(0, QHeaderView.Stretch)
            header_info_table.setSectionResizeMode(1, QHeaderView.Stretch)
            header_info_table.setSectionResizeMode(2, QHeaderView.Stretch)
            self.info_table_widget.setVisible(True)
            header = self.table_widget.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            self.table_widget.setVisible(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def get_cluster_desc(self, cluster):
        ids = [self.index_to_id[index] for index in cluster]
        ids = ', '.join(map(str, ids))
        query = f"SELECT DISTINCT descr_prod, descr_rep FROM {self.table_name} WHERE cod_prod IN ({ids});"
        result = self.client.query(query)
        desc = {}
        for row in result.result_rows:
            print(row)
            row_split = row[0].split()
            des = ' '.join(row_split[1:])
            desc[row_split[0]] = {}
            desc[row_split[0]]["description"] = des
            desc[row_split[0]]["category"] = row[1]
    
        return desc

    def display_graph(self, graph):
        # Clear the existing graph layout but keep the title
        for i in reversed(range(self.graph_layout.count())):
            widget = self.graph_layout.itemAt(i).widget()
            if widget is not None and widget != self.graph_title:
                widget.deleteLater()

        fig, ax = plt.subplots(figsize=(5, 4))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, ax=ax)

        canvas = FigureCanvas(fig)
        self.graph_layout.addWidget(canvas)
        canvas.draw()

    def display_descriptions(self, descriptions):
        self.table_widget.setRowCount(len(descriptions))
        for row, (node, desc) in enumerate(descriptions.items()):
            self.table_widget.setItem(row, 0, QTableWidgetItem(desc["description"]))
            self.table_widget.setItem(row, 1, QTableWidgetItem(desc["category"]))
        
        self.table_widget.resizeRowsToContents()  # Adjust row heights to fit content
        self.table_widget.resizeColumnsToContents()

    def display_product_info(self, product_id, product_desc):
        self.info_table_widget.setRowCount(1)  # Ensure the table has only one row
        item = QTableWidgetItem(str(product_id))
        item.setTextAlignment(Qt.AlignCenter)
        self.info_table_widget.setItem(0, 0, item)
        self.info_table_widget.setItem(0, 1, QTableWidgetItem(product_desc["description"]))
        self.info_table_widget.setItem(0, 2, QTableWidgetItem(product_desc["category"]))

        # Ensure all contents are shown properly
        self.info_table_widget.resizeRowsToContents()
        self.info_table_widget.resizeColumnsToContents()
        
        # Manually adjust the row height if needed
        row_height = self.info_table_widget.rowHeight(0)
        self.info_table_widget.setRowHeight(0, max(row_height, 30))

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
