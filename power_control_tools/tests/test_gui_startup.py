import os

import pytest


def test_control_tools_window_initializes_without_early_tab_signal():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    from power_control_tools.gui.main_window import ControlToolsMainWindow

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    window = ControlToolsMainWindow()

    assert window.filter_impl.currentText() == "IIR"
    assert window.centralWidget() is not None

    window.close()
    app.processEvents()
