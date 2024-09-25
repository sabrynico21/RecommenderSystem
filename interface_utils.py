from PyQt5.QtWidgets import (QLineEdit, QPushButton, QGroupBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QSize 

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