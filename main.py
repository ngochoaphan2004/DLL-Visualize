import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QStackedWidget, QScrollArea, QMainWindow, QToolBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from screen.PlotScreen import PlotScreen
from screen.SearchScreen import SearchScreen

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 Multi-Screen Demo")
        self.resize(800, 600)

        # Create stacked widget
        self.stacked = QStackedWidget()
        self.search_screen = SearchScreen()
        self.plot_screen = PlotScreen()

        self.stacked.addWidget(self.plot_screen)
        self.stacked.addWidget(self.search_screen)

        self.setCentralWidget(self.stacked)

        # Toolbar to switch screens
        toolbar = QToolBar("Navigation")
        self.addToolBar(toolbar)

        plot_action = QAction("Plot", self)
        plot_action.triggered.connect(lambda: self.stacked.setCurrentWidget(self.plot_screen))

        search_action = QAction("Search", self)
        search_action.triggered.connect(lambda: self.stacked.setCurrentWidget(self.search_screen))


        toolbar.addAction(plot_action)
        toolbar.addAction(search_action)


# ========== RUN ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
