"""Application launcher separating LLC and PFC into independent workspaces."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from llc_design.core.spec import LLCDesignSpec
from llc_design.gui import theme
from llc_design.gui.main_window import LLCMainWindow
from pfc_design.gui.main_window import PFCMainWindow


class WorkspaceSelectionDialog(QDialog):
    """Initial function selector shown before either engineering workspace."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.selected_workspace: str | None = None
        self.setWindowTitle("电源设计工具箱 — 选择设计功能")
        self.setMinimumSize(760, 420)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setStyleSheet(theme.launcher_stylesheet(theme.active_theme()))

        root = QVBoxLayout(self)
        title = QLabel("请选择进入的设计工作区")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 25px; font-weight: 650; padding: 18px;")
        root.addWidget(title)

        subtitle = QLabel(
            "LLC 与 PFC 使用独立参数区、Bode、波形和控制设计页面，"
            "进入工作区后可通过工具栏随时切换。"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 14px; padding: 4px 30px 20px 30px;")
        root.addWidget(subtitle)

        choices = QHBoxLayout()
        choices.setSpacing(24)
        llc_button = self._choice_button(
            "进入 LLC 设计",
            "谐振腔、磁性器件、损耗、开关波形、小信号与数字电压环",
        )
        pfc_button = self._choice_button(
            "进入 PFC 设计",
            "单相 TTPL + 三相 Vienna：双环控制、采样链、Bode、AC 周期、开关波形与 PF/THD",
        )
        llc_button.clicked.connect(lambda: self._select("llc"))
        pfc_button.clicked.connect(lambda: self._select("pfc"))
        choices.addWidget(llc_button)
        choices.addWidget(pfc_button)
        root.addLayout(choices, 1)

        cancel = QPushButton("退出")
        cancel.clicked.connect(self.reject)
        root.addWidget(cancel, alignment=Qt.AlignmentFlag.AlignCenter)

    @staticmethod
    def _choice_button(title: str, description: str) -> QPushButton:
        t = theme.active_theme()
        button = QPushButton(f"{title}\n\n{description}")
        button.setMinimumSize(320, 180)
        button.setStyleSheet(
            "QPushButton {"
            f"font-size: 16px; font-weight: 600; text-align: center;"
            f"padding: 24px; border: 2px solid {t.border_input}; border-radius: 10px;"
            f"background: {t.surface_alt}; color: {t.text_strong};"
            "}"
            f"QPushButton:hover {{background: {t.hover}; border-color: {t.accent};}}"
            f"QPushButton:pressed {{background: {t.pressed};}}"
        )
        return button

    def _select(self, workspace: str) -> None:
        self.selected_workspace = workspace
        self.accept()


class WorkspaceApplicationController:
    """Own both top-level windows and switch without destroying user state."""

    def __init__(self, initial_spec: LLCDesignSpec) -> None:
        self.llc_window = LLCMainWindow(initial_spec)
        self.pfc_window = PFCMainWindow()
        self.active_workspace: str | None = None
        self.llc_window.workspace_switch_requested.connect(self._handle_request)
        self.pfc_window.workspace_switch_requested.connect(self._handle_request)

    def start(self) -> bool:
        dialog = WorkspaceSelectionDialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        if dialog.selected_workspace is None:
            return False
        self.show_workspace(dialog.selected_workspace)
        return True

    def show_workspace(self, workspace: str) -> None:
        if workspace not in {"llc", "pfc"}:
            raise ValueError(f"unsupported workspace: {workspace}")
        self.llc_window.hide()
        self.pfc_window.hide()
        target = self.llc_window if workspace == "llc" else self.pfc_window
        self.active_workspace = workspace
        target.showMaximized()
        target.raise_()
        target.activateWindow()

    def _handle_request(self, workspace: str) -> None:
        if workspace == "home":
            self._show_selector_again()
        else:
            self.show_workspace(workspace)

    def _show_selector_again(self) -> None:
        previous = self.active_workspace
        self.llc_window.hide()
        self.pfc_window.hide()
        dialog = WorkspaceSelectionDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_workspace:
            self.show_workspace(dialog.selected_workspace)
        elif previous is not None:
            self.show_workspace(previous)


__all__ = ["WorkspaceApplicationController", "WorkspaceSelectionDialog"]
