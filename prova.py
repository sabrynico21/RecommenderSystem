import sys
import time
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QVBoxLayout, QLabel, QPushButton, QWidget
from PyQt5.QtGui import QMovie

# Worker thread for running the long function
class WorkerThread(QThread):
    finished = pyqtSignal()

    def run(self):
        # Simulate a long-running function
        time.sleep(5)  # Replace with your actual logic
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create group boxes
        self.groupBox1 = QGroupBox("Group Box 1")
        self.groupBox2 = QGroupBox("Group Box 2")
        self.groupBox3 = QGroupBox("Group Box 3")
        self.groupBox4 = QGroupBox("Group Box 4")

        # Add content to group boxes
        layout1 = QVBoxLayout()
        layout1.addWidget(QLabel("Content in Group Box 1"))
        self.groupBox1.setLayout(layout1)

        layout2 = QVBoxLayout()
        self.labelGroupBox2 = QLabel("Content in Group Box 2")
        layout2.addWidget(self.labelGroupBox2)
        self.groupBox2.setLayout(layout2)

        layout3 = QVBoxLayout()
        self.labelGroupBox3 = QLabel("Content in Group Box 3")
        layout3.addWidget(self.labelGroupBox3)
        self.groupBox3.setLayout(layout3)

        layout4 = QVBoxLayout()
        self.pushButton = QPushButton("Start Operation")
        self.pushButton.clicked.connect(self.on_start_operation)
        layout4.addWidget(self.pushButton)
        self.groupBox4.setLayout(layout4)

        # Main layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.groupBox1)
        mainLayout.addWidget(self.groupBox2)
        mainLayout.addWidget(self.groupBox3)
        mainLayout.addWidget(self.groupBox4)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

        # Create a QLabel for displaying the GIF in GroupBox2
        self.loadingLabelGroupBox2 = QLabel(self.groupBox2)
        self.loadingLabelGroupBox2.setAlignment(Qt.AlignCenter)
        self.loadingLabelGroupBox2.setVisible(False)

        # Create a QLabel for displaying the GIF in GroupBox3
        self.loadingLabelGroupBox3 = QLabel(self.groupBox3)
        self.loadingLabelGroupBox3.setAlignment(Qt.AlignCenter)
        self.loadingLabelGroupBox3.setVisible(False)

        # Load the GIF
        self.loadingMovieGroupBox2 = QMovie("loading.gif")
        self.loadingLabelGroupBox2.setMovie(self.loadingMovieGroupBox2)

        self.loadingMovieGroupBox3 = QMovie("loading.gif")
        self.loadingLabelGroupBox3.setMovie(self.loadingMovieGroupBox3)

    def on_start_operation(self):
        # Disable Group Box 2 and 3
        self.groupBox2.setEnabled(False)
        self.groupBox3.setEnabled(False)

        # Show the loading GIF in Group Box 2 and 3
        self.show_loading_animation(self.loadingLabelGroupBox2, self.labelGroupBox2)
        self.show_loading_animation(self.loadingLabelGroupBox3, self.labelGroupBox3)

        # Run the long function in a separate thread
        self.workerThread = WorkerThread()
        self.workerThread.finished.connect(self.on_operation_finished)
        self.workerThread.start()

    def show_loading_animation(self, loadingLabel, referenceWidget):
        # Adjust size of the GIF to match the size of the reference widget
        loadingLabel.setGeometry(referenceWidget.geometry())
        loadingLabel.setVisible(True)
        loadingLabel.movie().start()

    def on_operation_finished(self):
        # Stop the loading GIF and hide it for both group boxes
        self.hide_loading_animation(self.loadingLabelGroupBox2)
        self.hide_loading_animation(self.loadingLabelGroupBox3)

        # Re-enable Group Box 2 and 3
        self.groupBox2.setEnabled(True)
        self.groupBox3.setEnabled(True)

        # Update the content of the group boxes if needed
        self.groupBox2.setTitle("Updated Group Box 2")
        self.groupBox3.setTitle("Updated Group Box 3")

    def hide_loading_animation(self, loadingLabel):
        loadingLabel.movie().stop()
        loadingLabel.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
