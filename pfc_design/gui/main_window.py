"""Dedicated top-level PFC Control Lab window."""

from __future__ import annotations

from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QStatusBar,
    QWidget,
)

from llc_design.gui.workers import FunctionWorker
from pfc_design.control import (
    PFCControlLabConfig,
    build_pfc_control_lab_analysis,
    build_pfc_switching_waveforms,
    simulate_pfc_line_cycle,
)

from .control_lab_view import PFCControlLabView


class PFCMainWindow(QMainWindow):
    """Independent PFC workspace, separated from all LLC design pages."""

    workspace_switch_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("电源设计工具箱 — PFC Control Lab")
        self.resize(1920, 1080)
        self.thread_pool = QThreadPool.globalInstance()
        self._active_workers: list[FunctionWorker] = []
        self.result = None

        self.control_lab_view = PFCControlLabView()
        self.control_lab_view.analysis_requested.connect(self.run_analysis)
        self.setCentralWidget(self.control_lab_view)
        self._build_toolbar()
        self._build_statusbar()

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("PFC 工作区")
        toolbar.setMovable(False)

        switch_action = QAction("切换到 LLC", self)
        switch_action.triggered.connect(
            lambda: self.workspace_switch_requested.emit("llc"))
        toolbar.addAction(switch_action)

        home_action = QAction("功能选择", self)
        home_action.triggered.connect(
            lambda: self.workspace_switch_requested.emit("home"))
        toolbar.addAction(home_action)

        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        toolbar.addWidget(spacer)

        run_action = QAction("运行 PFC 完整分析", self)
        run_action.triggered.connect(self.control_lab_view._request)
        toolbar.addAction(run_action)

        about_action = QAction("关于 PFC", self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)

    def _build_statusbar(self) -> None:
        status = QStatusBar()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        status.addPermanentWidget(self.progress)
        self.setStatusBar(status)
        self.statusBar().showMessage("PFC 工作区就绪")

    def set_busy(self, busy: bool, message: str = "") -> None:
        self.progress.setVisible(busy)
        self.control_lab_view.set_busy(busy)
        self.statusBar().showMessage(
            message if busy else "PFC 工作区就绪")

    def _run_worker(self, label: str, function, callback) -> None:
        self.set_busy(True, label)
        worker = FunctionWorker(function)
        self._active_workers.append(worker)
        worker.signals.result.connect(callback)
        worker.signals.error.connect(self._worker_error)
        worker.signals.finished.connect(lambda: self.set_busy(False))
        worker.signals.finished.connect(
            lambda: self._active_workers.remove(worker))
        self.thread_pool.start(worker)

    def _worker_error(self, error: str) -> None:
        QMessageBox.critical(self, "PFC 计算失败", error.splitlines()[-1])

    def run_analysis(self, config: PFCControlLabConfig) -> None:
        """Run Bode, one-AC-period and local switching-cycle calculations."""

        def calculate():
            analysis = build_pfc_control_lab_analysis(config)
            line_cycle = simulate_pfc_line_cycle(config)
            switching = build_pfc_switching_waveforms(config)
            return analysis, line_cycle, switching

        self._run_worker(
            "正在建立 PFC 电流环、电压环、采样链与完整波形…",
            calculate,
            self._analysis_ready,
        )

    def _analysis_ready(self, result) -> None:
        self.result = result
        self.control_lab_view.set_result(result)
        analysis, line_cycle, _ = result
        current_margin = analysis.current_loop.margins
        voltage_margin = analysis.voltage_loop.margins
        self.statusBar().showMessage(
            "PFC 分析完成："
            f"Li fc={current_margin.critical_gain_crossover_hz}, "
            f"PM={current_margin.phase_margin_deg}; "
            f"Lv fc={voltage_margin.critical_gain_crossover_hz}, "
            f"PM={voltage_margin.phase_margin_deg}; "
            f"PF={line_cycle.metrics.power_factor:.6g}, "
            f"THD={line_cycle.metrics.current_thd_percent:.6g}%"
        )

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 PFC Control Lab",
            "<h3>PFC Control Lab</h3>"
            "<p>单相 TTPL PFC 电流内环、母线电压外环、采样链、"
            "完整 AC 周期与局部开关周期分析。</p>"
            "<p>Bode 默认仅显示系统开环，其他传递函数可按需启用。</p>",
        )


__all__ = ["PFCMainWindow"]
