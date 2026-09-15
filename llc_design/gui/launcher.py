"""Application launcher for the independent power-design workspaces."""

from __future__ import annotations

import textwrap

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from llc_design.core.spec import LLCDesignSpec
from llc_design.gui import theme
from llc_design.gui.help import show_help
from llc_design.i18n import t
from llc_design.gui.main_window import LLCMainWindow
from llc_design.gui.closed_loop_install import install_closed_loop_verification
from pfc_design.gui.main_window import PFCMainWindow
from power_control_tools.gui.fra_advanced import install_advanced_fra_actions
from power_control_tools.gui.fra_loop_designer import FRALoopDesignerWindow
from power_control_tools.gui.main_window import ControlToolsMainWindow


class WorkspaceSelectionDialog(QDialog):
    """Initial function selector shown before an engineering workspace."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.selected_workspace: str | None = None
        self.setWindowTitle(t("电源设计工具箱 — 选择设计功能"))
        self.setMinimumSize(1180, 650)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setStyleSheet(theme.launcher_stylesheet(theme.active_theme()))

        root = QVBoxLayout(self)
        title = QLabel(t("请选择进入的设计工作区"))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 25px; font-weight: 650; padding: 18px;")
        root.addWidget(title)

        subtitle = QLabel(
            t("LLC、PFC、数字控制工具与 FRA Loop Designer 使用独立工作区；")
            + t("FRA 工作区支持控制器剥离、实时整定、目标 Fc/PM 自动设计与低阶模型辨识。")
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 14px; padding: 4px 30px 20px 30px;")
        root.addWidget(subtitle)

        choices = QGridLayout()
        choices.setHorizontalSpacing(24)
        choices.setVerticalSpacing(20)
        llc_button = self._choice_button(
            t("进入 LLC 设计"),
            t("谐振腔、磁性器件、损耗、开关波形、小信号与数字电压环"),
        )
        pfc_button = self._choice_button(
            t("进入 PFC 设计"),
            t("单相 TTPL + 三相 Vienna：控制、采样链、Bode、AC 周期、开关波形与 PF/THD"),
        )
        control_button = self._choice_button(
            t("进入 Control Tools"),
            t("S2Z、数字滤波器、Bode、Step/Impulse、P/Z、SOS 与 C99 float32_t 导出"),
        )
        fra_button = self._choice_button(
            t("进入 FRA Loop Designer"),
            t("Bode100 / SIMPLIS / Generic：Equivalent Plant、Auto Design、Model ID、稳定性与 C99"),
        )
        llc_button.clicked.connect(lambda: self._select("llc"))
        pfc_button.clicked.connect(lambda: self._select("pfc"))
        control_button.clicked.connect(lambda: self._select("control"))
        fra_button.clicked.connect(lambda: self._select("fra"))
        choices.addWidget(llc_button, 0, 0)
        choices.addWidget(pfc_button, 0, 1)
        choices.addWidget(control_button, 1, 0)
        choices.addWidget(fra_button, 1, 1)
        root.addLayout(choices, 1)

        cancel = QPushButton(t("退出"))
        cancel.clicked.connect(self.reject)
        help_button = QPushButton(t("使用说明 / 帮助 (F1)"))
        help_button.setToolTip(t("四个工作区分别做什么、如何选择、通用操作与快捷键"))
        help_button.clicked.connect(lambda: show_help(self, "selector"))
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(help_button)
        footer.addWidget(cancel)
        footer.addStretch(1)
        root.addLayout(footer)
        shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.HelpContents), self)
        shortcut.activated.connect(lambda: show_help(self, "selector"))

    @staticmethod
    def _choice_button(title: str, description: str) -> QPushButton:
        """Create a launcher card whose localized description cannot overflow.

        ``QPushButton`` does not word-wrap automatically.  English/Japanese/
        Korean launcher text can therefore be wider than the 500 px card even
        when the Chinese source fits.  Insert explicit line breaks at a bounded
        character width; ``break_long_words`` also gives CJK text a safe path
        because those scripts do not necessarily contain spaces.
        """

        t = theme.active_theme()
        wrapped_description = "\n".join(
            textwrap.wrap(
                description,
                width=38,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
        button = QPushButton(f"{title}\n\n{wrapped_description}")
        button.setMinimumSize(500, 170)
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
    """Own top-level windows and switch without destroying user state."""

    def __init__(self, initial_spec: LLCDesignSpec) -> None:
        self.llc_window = LLCMainWindow(initial_spec)
        # Closed-loop verification is an LLC design stage, not another top-level
        # workspace: power design -> exact digital H(z) -> shared-ngspice verify.
        install_closed_loop_verification(self.llc_window)
        self.pfc_window = PFCMainWindow()
        self.control_window = ControlToolsMainWindow()
        self.fra_window = FRALoopDesignerWindow()
        install_advanced_fra_actions(self.fra_window)
        self.active_workspace: str | None = None
        self.llc_window.workspace_switch_requested.connect(self._handle_request)
        self.pfc_window.workspace_switch_requested.connect(self._handle_request)
        self.control_window.workspace_switch_requested.connect(self._handle_request)
        self.fra_window.workspace_switch_requested.connect(self._handle_request)
        self.control_window.digital_design_updated.connect(self.llc_window.set_external_control_design)
        self.control_window.digital_design_updated.connect(
            lambda digital, label="": self.llc_window.refresh_closed_loop_controller()
        )

    def start(self) -> bool:
        dialog = WorkspaceSelectionDialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        if dialog.selected_workspace is None:
            return False
        self.show_workspace(dialog.selected_workspace)
        return True

    def _hide_all(self) -> None:
        self.llc_window.hide()
        self.pfc_window.hide()
        self.control_window.hide()
        self.fra_window.hide()

    def show_workspace(self, workspace: str) -> None:
        if workspace not in {"llc", "pfc", "control", "fra"}:
            raise ValueError(f"unsupported workspace: {workspace}")
        self._hide_all()
        target = {
            "llc": self.llc_window,
            "pfc": self.pfc_window,
            "control": self.control_window,
            "fra": self.fra_window,
        }[workspace]
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
        self._hide_all()
        dialog = WorkspaceSelectionDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_workspace:
            self.show_workspace(dialog.selected_workspace)
        elif previous is not None:
            self.show_workspace(previous)


__all__ = ["WorkspaceApplicationController", "WorkspaceSelectionDialog"]
