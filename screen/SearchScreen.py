import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QStackedWidget, QScrollArea, QMainWindow, QToolBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SearchScreen(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Search bar ---
        search_bar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search term...")
        self.search_input.setFixedHeight(35)
        self.search_input.setStyleSheet("QLineEdit { border: 1px solid #aaa; border-radius: 10px; padding: 0 10px; }")
        search_button = QPushButton("Search")
        search_button.setFixedHeight(35)
        search_button.setStyleSheet("QPushButton { background: #1A73E8; color: white; border-radius: 10px; padding: 0 10px; }")
        search_bar.addWidget(self.search_input)
        search_bar.addWidget(search_button)

        layout.addLayout(search_bar)

        # --- Scroll area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        layout.addWidget(scroll)

        # --- Results list ---
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(container)

        # Mock results
        mock_data = [
            ("Python", "Python is an interpreted, high-level programming language."),
            ("Qt Framework", "Qt is a cross-platform application development framework."),
            ("Matplotlib", "Matplotlib is a library for creating static, animated, and interactive plots in Python."),
        ]
        for title, desc in mock_data:
            title_label = QLabel(f"<b style='color:#1A0DAB'>{title}</b>")
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            container_layout.addWidget(title_label)
            container_layout.addWidget(desc_label)

        container_layout.addStretch()