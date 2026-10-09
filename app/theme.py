"""Dark ground-control theme."""

DARK_QSS = """
* { font-family: "Segoe UI"; font-size: 12px; color: #d7e0ea; }
QMainWindow, QWidget#Root { background: #0c1218; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
QFrame#Header {
    background: #101820;
    border-bottom: 1px solid #243140;
}
QFrame#VideoFrame {
    background: #070b10;
    border: 1px solid #243140;
    border-radius: 10px;
}
QFrame#BottomBar {
    background: #101820;
    border-top: 1px solid #243140;
}
QLabel#Title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #f2f6fb;
}
QLabel#Subtitle { color: #8ea0b5; font-size: 11px; }
QLabel#BadgeVirtual {
    background: #143024;
    color: #7dffa8;
    border: 1px solid #1f6b45;
    border-radius: 8px;
    padding: 4px 8px;
    font-weight: 700;
}
QLabel#BadgeModel {
    background: #1a2430;
    color: #d5e4f5;
    border: 1px solid #314255;
    border-radius: 8px;
    padding: 4px 8px;
}
QLabel#RecBadge {
    color: #ff5d5d;
    font-weight: 700;
    font-size: 14px;
}
QLabel#ReadoutKey { color: #8ea0b5; }
QLabel#ReadoutVal {
    color: #f4f7fb;
    font-family: "Cascadia Mono", "Consolas";
    font-size: 13px;
}
QGroupBox {
    background: #141c26;
    border: 1px solid #2a394a;
    border-radius: 10px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #9eb4c9;
}
QPushButton {
    background: #1c2836;
    border: 1px solid #33485f;
    border-radius: 7px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton:hover { background: #243447; }
QPushButton:pressed { background: #182230; }
QPushButton:disabled { color: #66788a; background: #151c24; }
QPushButton#Primary {
    background: #1f6b45;
    border: 1px solid #2f8f5e;
    color: #f3fff7;
}
QPushButton#Primary:hover { background: #258055; }
QPushButton#Record {
    background: #7a2430;
    border: 1px solid #a13a48;
    color: #fff5f5;
}
QTabWidget::pane { border: 1px solid #243140; background: #101820; }
QTabBar::tab {
    background: #182230;
    color: #c5d4e4;
    padding: 8px 10px;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #1f6b45; color: #f3fff7; }
QTableWidget {
    background: #101820;
    gridline-color: #243140;
    selection-background-color: #1f3d32;
}
QHeaderView::section { background: #182230; color: #c5d4e4; border: none; padding: 4px; }
QLabel#BadgeDemo {
    background: #5a1d28;
    color: #ffe1e1;
    border: 1px solid #a13a48;
    border-radius: 8px;
    padding: 4px 8px;
    font-weight: 700;
}
QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #0e1620;
    border: 1px solid #314255;
    border-radius: 6px;
    padding: 4px;
}
QPushButton#Record:hover { background: #922c3b; }
QPushButton#Record:disabled { background: #3a2228; color: #b9a0a4; }
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QPlainTextEdit {
    background: #0f1620;
    border: 1px solid #314255;
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: #1f6b45;
}
QComboBox::drop-down { border: none; width: 18px; }
QSlider::groove:horizontal {
    height: 6px;
    background: #243140;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    background: #3ecf8e;
    border-radius: 7px;
}
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #3d5166;
    background: #0f1620;
}
QCheckBox::indicator:checked { background: #1f8f5a; border: 1px solid #3ecf8e; }
QStatusBar { background: #0c1218; color: #8ea0b5; }
QScrollBar:vertical {
    background: #0c1218;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical { background: #314255; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""
