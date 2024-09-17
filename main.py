from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QHeaderView, QGroupBox, QAbstractItemView
)
from PyQt5.QtCore import Qt, QSize 
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os
import sys
import clickhouse_connect
from utils import page_rank_nibble


class CustomLineEdit(QLineEdit):
    def __init__(self, placeholder_text="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPlaceholderText(placeholder_text)
        self.setFont(QFont("Arial", 12))
        self.setStyleSheet("""
            padding: 10px; 
            border: 2px solid #007bff; 
            border-radius: 8px;
            background-color: #ffffff;
            font-size: 14px;
        """)

class CustomPushButton(QPushButton):
    def __init__(self, text, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.setFont(QFont("Arial", 12))
        self.setStyleSheet("""
            background-color: #007bff; 
            color: white; 
            padding: 12px 20px; 
            border-radius: 8px; 
            border: none;
            font-weight: bold;
            font-size: 14px;
        """)
        self.setCursor(Qt.PointingHandCursor)
        self.setIconSize(QSize(20, 20))
        self.setMinimumWidth(100)

class CustomGroupBox(QGroupBox):
    def __init__(self, title, *args, **kwargs):
        super().__init__(title, *args, **kwargs)
        self.setFont(QFont("Arial", 14, QFont.Bold))
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid gray;
                border-radius: 5px;
                margin-top: 4ex;
                background-color: #001f3f;
                color: #FFBF00;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
        """)

class PageRankNibbleApp(QMainWindow):
    def __init__(self, graph_path="graph.pkl", dict_path="products_dict.pkl", **client_params):
        super().__init__()

        # Load the graph
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
            print("Graph loaded successfully.")
        else:
            print(f"File {graph_path} does not exist.")
            return

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

        # Text box, button in a group box at top left
        self.top_left_group_box = CustomGroupBox("Product ID")
        self.top_left_group_box_layout = QVBoxLayout()
        self.top_left_group_box.setLayout(self.top_left_group_box_layout)

        # self.label = QLabel("Enter Product ID:")
        # self.label.setFont(QFont("Arial", 14, QFont.Bold))
        # self.label.setStyleSheet("color: #ffffff;")  # White color for contrast
        # self.top_left_group_box_layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self.product_id_entry = CustomLineEdit("Enter ID here...")
        self.top_left_group_box_layout.addWidget(self.product_id_entry, alignment=Qt.AlignCenter)

        self.submit_button = CustomPushButton("Submit")
        self.submit_button.clicked.connect(self.run_page_rank_nibble)
        self.top_left_group_box_layout.addWidget(self.submit_button, alignment=Qt.AlignCenter)

        self.main_layout.addWidget(self.top_left_group_box, 0, 0)

        # # Product Info Table at top right
        # self.top_right_group_box = CustomGroupBox("Product Information")
        # self.top_right_group_box_layout = QVBoxLayout()
        # self.top_right_group_box.setLayout(self.top_right_group_box_layout)

        # self.info_table_widget = QTableWidget()
        # self.info_table_widget.setColumnCount(3)  # Three columns: Product Index, Description, Category
        # self.info_table_widget.setHorizontalHeaderLabels(["Product Index", "Description", "Category"])
        # self.info_table_widget.setStyleSheet("""
        #     QTableWidget {
        #         border: 1px solid #ddd;
        #         background-color: #ffffff;
        #         padding: 10px;
        #     }
        #     QHeaderView::section {
        #         background-color: #f0f0f0;
        #         border: 1px solid #ddd;
        #         padding: 5px;
        #         font-weight: bold;
        #     }
        #     QTableWidget::item {
        #         padding: 5px;
        #     }
        # """)
        # self.info_table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Set to read-only mode
        # header_info_table = self.info_table_widget.horizontalHeader()
        # header_info_table.setSectionResizeMode(QHeaderView.Stretch)
        # self.info_table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # self.info_table_widget.setWordWrap(True)
        # self.top_right_group_box_layout.addWidget(self.info_table_widget)

        # self.main_layout.addWidget(self.top_right_group_box, 0, 1)

        # Product Info Card at top right
        self.top_right_group_box = CustomGroupBox("Product Information")
        self.top_right_group_box_layout = QVBoxLayout()
        self.top_right_group_box.setLayout(self.top_right_group_box_layout)

        # Create a form layout
        self.info_form_layout = QFormLayout()
        self.info_form_layout.setLabelAlignment(Qt.AlignRight)
        self.info_form_layout.setFormAlignment(Qt.AlignLeft)

        # Set font and style for labels and values
        label_font = QFont("Arial", 12, QFont.Bold)
        value_font = QFont("Arial", 12)

        self.product_index_label = QLabel("Product Index:")
        self.product_index_value = QLabel("")

        self.description_label = QLabel("Description:")
        self.description_value = QLabel("")

        self.category_label = QLabel("Category:")
        self.category_value = QLabel("")

        # Apply font to labels and values
        for label in [self.product_index_label, self.description_label, self.category_label]:
            label.setFont(label_font)
            label.setStyleSheet("color: rgb(255, 255, 153); padding: 5px;")

        for value in [self.product_index_value, self.description_value, self.category_value]:
            value.setFont(value_font)
            value.setStyleSheet("""
                color: #555555;
                background-color: #f0f0f0;
                padding: 5px;
                border-radius: 5px;
                """)

        # Add rows to the form layout
        self.info_form_layout.addRow(self.product_index_label, self.product_index_value)
        self.info_form_layout.addRow(self.description_label, self.description_value)
        self.info_form_layout.addRow(self.category_label, self.category_value)

        self.top_right_group_box_layout.addLayout(self.info_form_layout)
        self.main_layout.addWidget(self.top_right_group_box, 0, 1)

        # Description and Categories Table at bottom left
        self.bottom_left_group_box = CustomGroupBox("Product Descriptions and Categories")
        self.bottom_left_group_box_layout = QVBoxLayout()
        self.bottom_left_group_box.setLayout(self.bottom_left_group_box_layout)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)  # Two columns: Description and Category
        self.table_widget.setHorizontalHeaderLabels(["Description", "Category"])
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
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Set to read-only mode
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_widget.setWordWrap(True)
        self.bottom_left_group_box_layout.addWidget(self.table_widget)

        self.main_layout.addWidget(self.bottom_left_group_box, 1, 0)

        # Cluster Graph at bottom right
        self.bottom_right_group_box = CustomGroupBox("Cluster Graph")
        self.bottom_right_group_box_layout = QVBoxLayout()
        self.bottom_right_group_box.setLayout(self.bottom_right_group_box_layout)

        self.main_layout.addWidget(self.bottom_right_group_box, 1, 1)

        # Set equal stretch for rows and columns
        self.main_layout.setColumnStretch(0, 1)
        self.main_layout.setColumnStretch(1, 1)
        #self.main_layout.setRowStretch(0, 1)
        #self.main_layout.setRowStretch(1, 1)

    def run_page_rank_nibble(self):
        product_id = self.product_id_entry.text()

        if not product_id:
            QMessageBox.warning(self, "Input Error", "Please enter a product ID.")
            return

        n = self.graph.number_of_nodes()
        phi = 0.1
        beta = 0.9
        epsilon = 2e-05

        try:
            seed, cluster = page_rank_nibble(self.graph, n, phi, beta, epsilon, "unweighted", self.id_to_index[product_id])
            desc = self.get_cluster_desc(cluster)
            cluster_graph = self.get_cluster_graph(cluster)
            self.display_graph(cluster_graph)
            self.display_descriptions(desc)
            self.display_product_info(self.id_to_index[product_id], desc[product_id])

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def get_cluster_graph(self, cluster):
        subgraph = self.graph.subgraph(cluster)
        subgraph = nx.Graph(subgraph)
        return subgraph

    def get_cluster_desc(self, cluster):
        ids = [self.index_to_id[index] for index in cluster]
        ids = ', '.join(map(str, ids))
        query = f"SELECT DISTINCT descr_prod, descr_rep FROM {self.table_name} WHERE cod_prod IN ({ids});"
        result = self.client.query(query)
        desc = {}
        for row in result.result_rows:
            row_split = row[0].split()
            des = ' '.join(row_split[1:])
            desc[row_split[0]] = (des, row[1])
        return desc

    def display_product_info(self, product_id, product_info):
        # self.info_table_widget.setRowCount(1)
        # item = QTableWidgetItem(str(product_id))
        # item.setTextAlignment(Qt.AlignCenter)
        # self.info_table_widget.setItem(0, 0, item)
        # self.info_table_widget.setItem(0, 1, QTableWidgetItem(product_info[0]))
        # self.info_table_widget.setItem(0, 2, QTableWidgetItem(product_info[1]))
        self.product_index_value.setText(str(product_id))
        self.description_value.setText(product_info[0])
        self.category_value.setText(product_info[1])

    def display_descriptions(self, descriptions):
        self.table_widget.setRowCount(len(descriptions))
        for row, (product, (descr, cat)) in enumerate(descriptions.items()):
            self.table_widget.setItem(row, 0, QTableWidgetItem(descr))
            self.table_widget.setItem(row, 1, QTableWidgetItem(cat))
        self.table_widget.resizeRowsToContents()

    # def display_graph(self, graph):
    #     # Clear the existing graph layout but keep the title
    #     for i in reversed(range(self.bottom_right_group_box.layout().count())):
    #         widget = self.bottom_right_group_box.layout().itemAt(i).widget()
    #         if widget is not None: #and widget != self.graph_title:
    #             widget.deleteLater()

    #     fig, ax = plt.subplots(figsize=(7, 5))
    #     pos = nx.spring_layout(graph)
    #     nx.draw(graph, pos, with_labels=True, ax=ax, node_color='skyblue', node_size=500, edge_color='gray', linewidths=1, font_size=10)

    #     canvas = FigureCanvas(fig)
    #     self.bottom_right_group_box.layout().addWidget(canvas)
    #     canvas.draw()

    def display_graph(self, graph):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui
    # Clear the existing graph layout but keep the title
        for i in reversed(range(self.bottom_right_group_box.layout().count())):
            widget = self.bottom_right_group_box.layout().itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Create a PyQtGraph PlotWidget
        plot_widget = pg.PlotWidget()
        self.bottom_right_group_box.layout().addWidget(plot_widget)

        # Set the background color of the PlotWidget
        plot_widget.setBackground(QColor('white'))  # Change this color as needed

        # Convert the networkx graph to PyQtGraph data
        pos = nx.spring_layout(graph, seed=42)  # Optional: Seed for reproducibility
        edges = list(graph.edges())
        nodes = list(graph.nodes())

        # Extract node positions
        x = [pos[node][0] for node in nodes]
        y = [pos[node][1] for node in nodes]

        # Plot edges
        for edge in edges:
            x_start, y_start = pos[edge[0]]
            x_end, y_end = pos[edge[1]]
            plot_widget.plot([x_start, x_end], [y_start, y_end], pen=pg.mkPen('gray', width=1))  # Gray edges with width 1

        # Plot nodes
        plot_widget.plot(x, y, pen=None, symbol='o', symbolSize=40, symbolBrush=pg.mkBrush('skyblue'))  # Larger nodes with skyblue color

        # Add node indices as text labels
        for node in nodes:
            x_pos, y_pos = pos[node]
            text_item = pg.TextItem(str(node), color='black', anchor=(0.5, 0.5))
            text_item.setPos(x_pos, y_pos)
            text_item.setFont(QtGui.QFont('Arial', 10))  # Increase the font size for the text labels
            plot_widget.addItem(text_item)

        # Remove axis labels and grid
        plot_widget.hideAxis('left')
        plot_widget.hideAxis('bottom')
        plot_widget.showGrid(x=False, y=False)

        # Optionally, adjust the view to fit the graph
        plot_widget.autoRange()

    # def display_graph(self, graph):
    #     from pyvis.network import Network
    #     import tempfile
    #     from PyQt5.QtWebEngineWidgets import QWebEngineView
    #     from PyQt5.QtCore import QUrl
    #     # Clear the existing graph layout but keep the title
    #     for i in reversed(range(self.bottom_right_group_box_layout.count())):
    #         widget = self.bottom_right_group_box_layout.itemAt(i).widget()
    #         if widget is not None:
    #             widget.deleteLater()

    #     # Create a pyvis Network object
    #     net = Network(height='600px', width='100%', notebook=False)
        
    #     # Add nodes and edges to the Network
    #     for node in graph.nodes():
    #         net.add_node(node)
    #     for edge in graph.edges():
    #         net.add_edge(edge[0], edge[1])
        
    #     # Generate a temporary HTML file to display the graph
    #     html_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    #     html_file_path = html_file.name
    #     net.save_graph(html_file_path)
    #     html_file.close()  # Ensure the file is closed

    #     # Create a QWebEngineView to display the HTML
    #     graph_view = QWebEngineView()
    #     graph_view.setUrl(QUrl.fromLocalFile(html_file_path))

    #     # Add the QWebEngineView widget to the layout
    #     self.bottom_right_group_box_layout.addWidget(graph_view)

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
