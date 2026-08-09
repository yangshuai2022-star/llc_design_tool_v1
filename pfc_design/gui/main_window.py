"""Dedicated PFC top-level workspace with TTPL and Vienna sub-workspaces."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow,QMessageBox,QProgressBar,QSizePolicy,QStatusBar,QTabWidget,QWidget

from llc_design.gui.workers import FunctionWorker
from pfc_design.control import PFCControlLabConfig,build_pfc_control_lab_analysis,build_pfc_switching_waveforms,simulate_pfc_line_cycle
from pfc_design.vienna import (
    ViennaControlLabConfig,
    build_vienna_control_lab_analysis,
    build_vienna_switching_waveforms,
    simulate_vienna_line_cycle,
    validate_vienna_analysis,
    validate_vienna_line_cycle,
    validate_vienna_switching,
)
from .control_lab_view import PFCControlLabView
from .vienna_control_view import ViennaControlLabView


class PFCMainWindow(QMainWindow):
    """Independent PFC workspace: single-phase TTPL + three-phase Vienna."""
    workspace_switch_requested=Signal(str)

    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle("电源设计工具箱 — PFC Design: TTPL / Vienna");self.resize(1920,1080)
        self.thread_pool=QThreadPool.globalInstance();self._active_workers=[];self.result=None
        self.subtabs=QTabWidget();self.control_lab_view=PFCControlLabView();self.vienna_view=ViennaControlLabView()
        self.control_lab_view.analysis_requested.connect(self.run_ttpl_analysis);self.vienna_view.analysis_requested.connect(self.run_vienna_analysis)
        self.subtabs.addTab(self.control_lab_view,"Single-Phase TTPL PFC");self.subtabs.addTab(self.vienna_view,"Three-Phase Vienna PFC")
        self.setCentralWidget(self.subtabs);self._build_toolbar();self._build_statusbar()

    def _build_toolbar(self):
        tb=self.addToolBar("PFC 工作区");tb.setMovable(False)
        a=QAction("切换到 LLC",self);a.triggered.connect(lambda:self.workspace_switch_requested.emit("llc"));tb.addAction(a)
        a=QAction("功能选择",self);a.triggered.connect(lambda:self.workspace_switch_requested.emit("home"));tb.addAction(a);tb.addSeparator()
        a=QAction("TTPL",self);a.triggered.connect(lambda:self.subtabs.setCurrentIndex(0));tb.addAction(a)
        a=QAction("Vienna",self);a.triggered.connect(lambda:self.subtabs.setCurrentIndex(1));tb.addAction(a)
        spacer=QWidget();spacer.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Preferred);tb.addWidget(spacer)
        a=QAction("运行当前 PFC 分析",self);a.triggered.connect(self._run_current);tb.addAction(a)
        a=QAction("关于 PFC",self);a.triggered.connect(self.show_about);tb.addAction(a)

    def _run_current(self):
        w=self.subtabs.currentWidget()
        if hasattr(w,"_request"):w._request()

    def _build_statusbar(self):
        st=QStatusBar();self.progress=QProgressBar();self.progress.setRange(0,0);self.progress.setVisible(False);st.addPermanentWidget(self.progress);self.setStatusBar(st);st.showMessage("PFC 工作区就绪")

    def set_busy(self,busy,message=""):
        self.progress.setVisible(busy);self.control_lab_view.set_busy(busy);self.vienna_view.set_busy(busy);self.statusBar().showMessage(message if busy else "PFC 工作区就绪")

    def _run_worker(self,label,function,callback):
        self.set_busy(True,label);w=FunctionWorker(function);self._active_workers.append(w);w.signals.result.connect(callback);w.signals.error.connect(self._worker_error);w.signals.finished.connect(lambda:self.set_busy(False));w.signals.finished.connect(lambda:self._active_workers.remove(w));self.thread_pool.start(w)

    def _worker_error(self,error):
        lines=[line for line in str(error).splitlines() if line.strip()]
        last=lines[-1] if lines else "Unknown PFC calculation error"
        box=QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("PFC 计算失败")
        box.setText(last)
        box.setInformativeText("已保留完整 traceback。点击“显示详细信息”可直接复制给开发者定位。")
        box.setDetailedText(str(error))
        box.exec()

    def run_ttpl_analysis(self,config:PFCControlLabConfig):
        def calc():
            a=build_pfc_control_lab_analysis(config);line=simulate_pfc_line_cycle(config);sw=build_pfc_switching_waveforms(config,line_cycle=line,line_angle_deg=config.power_stage.line_angle_deg);return a,line,sw
        self._run_worker("正在建立 TTPL 双环、AC 周期、过零与开关工作点…",calc,self._ttpl_ready)

    def _ttpl_ready(self,result):
        try:
            self.result=result
            self.control_lab_view.set_result(result)
            a,line,_=result;ci=a.current_loop.margins;cv=a.voltage_loop.margins
            self.statusBar().showMessage(f"TTPL 完成: Li fc={ci.critical_gain_crossover_hz}, PM={ci.phase_margin_deg}; Lv fc={cv.critical_gain_crossover_hz}, PM={cv.phase_margin_deg}; PF={line.metrics.power_factor:.6g}, THD={line.metrics.current_thd_percent:.5g}%")
        except Exception:
            self._worker_error("TTPL 结果绘图/GUI 更新失败\n" + traceback.format_exc())

    def run_vienna_analysis(self,config:ViennaControlLabConfig):
        def calc():
            # Keep every numerical stage isolated.  A Vienna failure should say
            # exactly whether it came from Bode, AC-cycle or local switching
            # reconstruction instead of surfacing later as a plotting KeyError.
            try:
                a=build_vienna_control_lab_analysis(config)
                validate_vienna_analysis(a)
            except Exception as exc:
                raise RuntimeError(f"Vienna 小信号/Bode 建模失败: {exc}") from exc
            try:
                line=simulate_vienna_line_cycle(config)
                validate_vienna_line_cycle(line)
            except Exception as exc:
                raise RuntimeError(f"Vienna 三相 AC 周期求解失败: {exc}") from exc
            try:
                sw=build_vienna_switching_waveforms(
                    config,line,line_angle_deg=config.switching_line_angle_deg)
                validate_vienna_switching(sw)
            except Exception as exc:
                raise RuntimeError(f"Vienna 开关工作点重建失败: {exc}") from exc
            return a,line,sw
        self._run_worker("正在建立 Vienna ABC 双环、中点平衡、三相 AC 周期与 Sector…",calc,self._vienna_ready)

    def _vienna_ready(self,result):
        try:
            self.result=result
            self.vienna_view.set_result(result)
            a,line,_=result;ci=a.current_loop.margins;cv=a.voltage_loop.margins;cb=a.balance_loop.margins
            self.statusBar().showMessage(f"Vienna 完成: Li PM={ci.phase_margin_deg}; Lv PM={cv.phase_margin_deg}; Balance PM={cb.phase_margin_deg}; PF={line.metrics.overall_power_factor:.6g}")
        except Exception:
            # Result callbacks execute in the GUI thread, outside FunctionWorker's
            # try/except.  Previously a plotting/key error here could look like
            # a Vienna calculation crash and the useful traceback was lost.
            self._worker_error("Vienna 结果绘图/GUI 更新失败\n" + traceback.format_exc())

    def show_about(self):
        QMessageBox.about(self,"关于 PFC Design","<h3>PFC Design Workspace</h3><p>Single-phase TTPL + Three-phase Vienna PFC.</p><p>双环数字控制、模拟采样链、开环 Bode、完整 AC 周期、开关工作点、PF/THD；Vienna 增加 Split DC Bus / Midpoint Balance / Sector Analyzer。</p>")


__all__=["PFCMainWindow"]
