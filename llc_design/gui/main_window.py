"""PySide6 main window for LLC engineering design, waveform and control work."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback

from PySide6.QtCore import QThreadPool, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from ..control.analysis import SmallSignalAnalysis, build_small_signal_analysis
from ..control.digital_loop import DigitalLoopAnalysis, build_digital_loop_analysis
from ..control.linearize import ControlInputKind
from ..core.config import load_spec, save_spec
from ..core.spec import LLCDesignSpec, PrimaryTopology
from ..core.tank import equivalent_ac_load_ohm, gain, target_gain
from ..core.q_zvs import LLCQZVSAnalysis, build_q_zvs_analysis
from ..dynamics.plant import DynamicPhasorModel
from ..dynamics.switched import SwitchedSimulationConfig, simulate_switched_steady_state
from ..dynamics.waveforms import WaveformBundle, reconstruct_dynamic_phasor_waveforms
from ..models.system import LLCSystemAnalyzer, SystemAnalysis
from ..magnetics.transformer_designer import (FerriteCoreInput, TransformerSynthesisResult,
                                               TransformerSynthesisSettings, synthesize_transformer,
                                               export_transformer_synthesis)
from ..models.devices import DeviceDatabase
from .workers import FunctionWorker

from .widgets.small_signal_view import SmallSignalView
from .widgets.digital_loop_view import DigitalLoopView
from .widgets.q_zvs_view import LLCQZVSView
from .widgets.transformer_design_view import TransformerDesignView
from .widgets.topology import TopologyView
from .widgets.waveform_view import WaveformView


class LLCMainWindow(QMainWindow):
    workspace_switch_requested = Signal(str)

    def __init__(self, initial_spec: LLCDesignSpec | None = None):
        super().__init__()
        self.setWindowTitle("电源设计工具箱 — LLC Design / Waveform / Control")
        self.resize(1920, 1080)
        self.spec = initial_spec or LLCDesignSpec()
        self.system_analysis: SystemAnalysis | None = None
        self.small_signal_analysis: SmallSignalAnalysis | None = None
        self.digital_loop_analysis: DigitalLoopAnalysis | None = None
        self.q_zvs_analysis: LLCQZVSAnalysis | None = None
        self.transformer_synthesis: TransformerSynthesisResult | None = None
        self.thread_pool = QThreadPool.globalInstance()
        self._active_workers: list[FunctionWorker] = []
        self.output_directory = Path("output/gui_session")
        self._build_actions()
        self._build_ui()
        self._load_spec_to_widgets(self.spec)

    def _build_actions(self) -> None:
        """Build a deliberately small toolbar.

        V7.1 keeps the global toolbar for project/navigation commands only.
        Analysis commands live in the page that owns them, while a single
        ``运行当前`` action remains available for keyboard-driven work.
        """
        toolbar = self.addToolBar("工程")
        toolbar.setObjectName("llc_main_toolbar")
        toolbar.setMovable(False)

        for label, callback in (
            ("加载 JSON", self.load_json),
            ("保存 JSON", self.save_json),
            ("输出目录", self.choose_output_directory),
        ):
            action = QAction(label, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)

        toolbar.addSeparator()
        switch_action = QAction("切换到 PFC", self)
        switch_action.triggered.connect(
            lambda: self.workspace_switch_requested.emit("pfc"))
        toolbar.addAction(switch_action)
        home_action = QAction("功能选择", self)
        home_action.triggered.connect(
            lambda: self.workspace_switch_requested.emit("home"))
        toolbar.addAction(home_action)

        toolbar.addSeparator()
        # Global LLC design parameters use the toolbar action itself as a true
        # show/hide toggle.  The dock deliberately has no close (X) button: the
        # same toolbar control (or F4) is the single, predictable way to hide it.
        self.toggle_params_action = QAction("隐藏设计参数", self)
        self.toggle_params_action.setCheckable(True)
        self.toggle_params_action.setChecked(True)
        self.toggle_params_action.setShortcut("F4")
        self.toggle_params_action.setToolTip("显示/隐藏 LLC 全局设计参数（F4）")
        toolbar.addAction(self.toggle_params_action)

        self.toggle_log_action = QAction("运行日志", self)
        self.toggle_log_action.setCheckable(True)
        self.toggle_log_action.setChecked(False)
        self.toggle_log_action.setShortcut("F8")
        toolbar.addAction(self.toggle_log_action)

        self.focus_action = QAction("专注模式", self)
        self.focus_action.setCheckable(True)
        self.focus_action.setShortcut("F9")
        self.focus_action.toggled.connect(self._toggle_focus_mode)
        toolbar.addAction(self.focus_action)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        run_button = QToolButton()
        run_button.setText("运行当前")
        run_button.setToolTip("运行当前页面对应的分析（Ctrl+R）")
        run_button.clicked.connect(self._run_current_page)
        toolbar.addWidget(run_button)
        run_action = QAction(self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self._run_current_page)
        self.addAction(run_action)

        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)

    def _build_ui(self) -> None:
        """Use dockable global inputs instead of a permanent nested sidebar."""
        self.setDockNestingEnabled(True)
        self.setCentralWidget(self._build_workspace())

        self.parameter_dock = QDockWidget("LLC 设计参数", self)
        self.parameter_dock.setObjectName("llc_parameter_dock")
        self.parameter_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.parameter_dock.setFeatures(
            # No DockWidgetClosable: users hide/show this panel with the same
            # “设计参数” toolbar action instead of hunting for a tiny X button.
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.parameter_dock.setWidget(self._build_parameter_panel())
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.parameter_dock)
        self.resizeDocks([self.parameter_dock], [300], Qt.Orientation.Horizontal)
        self.toggle_params_action.toggled.connect(self.parameter_dock.setVisible)
        self.parameter_dock.visibilityChanged.connect(self._sync_parameter_toggle)
        self._sync_parameter_toggle(self.parameter_dock.isVisible())

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(3000)
        self.log_dock = QDockWidget("运行日志", self)
        self.log_dock.setObjectName("llc_log_dock")
        self.log_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea |
            Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.log_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.resizeDocks([self.log_dock], [180], Qt.Orientation.Vertical)
        self.log_dock.hide()
        self.toggle_log_action.toggled.connect(self.log_dock.setVisible)
        self.log_dock.visibilityChanged.connect(self.toggle_log_action.setChecked)

        status = QStatusBar()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        status.addPermanentWidget(self.progress)
        self.setStatusBar(status)
        self.statusBar().showMessage("就绪")
        self._apply_llc_style()

    def _sync_parameter_toggle(self, visible: bool) -> None:
        """Keep the design-parameter toolbar toggle and dock visibility in sync."""
        self.toggle_params_action.blockSignals(True)
        self.toggle_params_action.setChecked(bool(visible))
        self.toggle_params_action.setText(
            "隐藏设计参数" if visible else "显示设计参数"
        )
        self.toggle_params_action.blockSignals(False)

    def _toggle_focus_mode(self, enabled: bool) -> None:
        if enabled:
            self._params_was_visible = self.parameter_dock.isVisible() if hasattr(self, "parameter_dock") else True
            self._log_was_visible = self.log_dock.isVisible() if hasattr(self, "log_dock") else False
            if hasattr(self, "parameter_dock"):
                self.parameter_dock.hide()
            if hasattr(self, "log_dock"):
                self.log_dock.hide()
            self.statusBar().showMessage("专注模式：已隐藏全局参数与运行日志")
        else:
            if hasattr(self, "parameter_dock") and getattr(self, "_params_was_visible", True):
                self.parameter_dock.show()
            if hasattr(self, "log_dock") and getattr(self, "_log_was_visible", False):
                self.log_dock.show()
            self.statusBar().showMessage("就绪")

    def _run_current_page(self) -> None:
        if not hasattr(self, "tabs"):
            return
        current = self.tabs.currentWidget()
        if current is self.q_zvs_view:
            self.run_q_zvs()
        elif current is self.transformer_design_view:
            self.run_transformer_design(None, None)
        elif current is self.waveform_view:
            self.run_waveforms(False)
        elif current is self.small_signal_view:
            self.run_small_signal({})
        elif current is self.digital_loop_view:
            self.run_digital_loop({})
        else:
            self.run_design()

    def _apply_llc_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f4f6f8; }
            QToolBar { background: #ffffff; border-bottom: 1px solid #d9dee7; spacing: 4px; padding: 3px 6px; }
            QToolButton { padding: 4px 10px; border-radius: 5px; }
            QToolButton:hover { background: #eef4ff; }
            QDockWidget { font-weight: 600; color: #344054; }
            QDockWidget::title { background: #f8fafc; border-bottom: 1px solid #d9dee7; padding: 7px 10px; }
            QTabWidget::pane { border: 1px solid #d9dee7; background: #ffffff; top: -1px; }
            QTabBar::tab { background: #f2f4f7; border: 1px solid #d9dee7; padding: 7px 14px; margin-right: 2px; }
            QTabBar::tab:selected { background: #ffffff; color: #175cd3; border-bottom-color: #ffffff; font-weight: 600; }
            QGroupBox { background: #ffffff; border: 1px solid #d9dee7; border-radius: 7px; margin-top: 12px; padding-top: 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #344054; }
            QDoubleSpinBox, QSpinBox, QComboBox { min-height: 25px; padding: 1px 4px; }
            QPushButton { min-height: 29px; padding: 3px 10px; border: 1px solid #b9c2cf; border-radius: 5px; background: #ffffff; }
            QPushButton:hover { background: #eef4ff; border-color: #84adff; }
            QPushButton:pressed { background: #dbe8ff; }
            QPlainTextEdit { background: #ffffff; border: 1px solid #d9dee7; }
            QScrollArea { border: none; background: transparent; }
            """
        )

    def _spin(self, minimum, maximum, decimals, suffix="") -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSuffix(suffix)
        widget.setKeyboardTracking(False)
        widget.setMinimumWidth(112)
        return widget

    def _build_parameter_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 8, 10, 12)
        layout.setSpacing(8)
        hint = QLabel("全局 LLC 设计输入。需要最大化图形区域时，可按 F4 隐藏此面板，F9 进入专注模式。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#667085;padding:2px 2px 6px 2px;")
        layout.addWidget(hint)
        self.fields: dict[str, object] = {}

        electrical = QGroupBox("输入与输出")
        form = QFormLayout(electrical)
        field_specs = [
            ("vbus_nom_v", "母线额定", self._spin(20, 2000, 2, " V")),
            ("vbus_min_normal_v", "母线最低正常", self._spin(20, 2000, 2, " V")),
            ("vbus_max_v", "母线最高", self._spin(20, 2000, 2, " V")),
            ("vbus_hold_end_v", "Hold-up 末端", self._spin(20, 2000, 2, " V")),
            ("vout_v", "输出电压", self._spin(0.1, 1000, 3, " V")),
            ("pout_w", "输出功率", self._spin(1, 200000, 1, " W")),
        ]
        for key, label, widget in field_specs:
            self.fields[key] = widget
            form.addRow(label, widget)
        layout.addWidget(electrical)

        tank = QGroupBox("谐振腔与变压器")
        form = QFormLayout(tank)
        tank_specs = [
            ("resonant_frequency_hz", "谐振频率", self._spin(1, 2000, 3, " kHz")),
            ("minimum_frequency_hz", "最低频率", self._spin(1, 2000, 3, " kHz")),
            ("maximum_frequency_hz", "最高频率", self._spin(1, 3000, 3, " kHz")),
            ("ln_ratio", "Ln=Lm/Lr", self._spin(1.01, 30, 4)),
            ("q_full_load", "满载 Qe", self._spin(0.01, 5, 4)),
        ]
        for key, label, widget in tank_specs:
            self.fields[key] = widget
            form.addRow(label, widget)
        self.primary_turns = QSpinBox(); self.primary_turns.setRange(1, 500)
        self.secondary_turns = QSpinBox(); self.secondary_turns.setRange(1, 100)
        self.fields["primary_turns"] = self.primary_turns
        self.fields["secondary_turns"] = self.secondary_turns
        form.addRow("原边匝数", self.primary_turns)
        form.addRow("副边匝数", self.secondary_turns)
        self.topology = QComboBox()
        self.topology.addItem("全桥 LLC", PrimaryTopology.FULL_BRIDGE)
        self.topology.addItem("半桥 LLC", PrimaryTopology.HALF_BRIDGE)
        form.addRow("一次侧拓扑", self.topology)
        layout.addWidget(tank)

        capacitors = QGroupBox("电容与控制建模")
        form = QFormLayout(capacitors)
        cap_specs = [
            ("bus_capacitance_f", "母线电容", self._spin(1, 100000, 1, " µF")),
            ("requested_hold_time_s", "Hold-up 时间", self._spin(0.1, 1000, 2, " ms")),
            ("output_capacitance_f", "输出电容", self._spin(1, 100000, 1, " µF")),
            ("output_cap_esr_ohm", "输出电容 ESR", self._spin(0, 1000, 4, " mΩ")),
            ("primary_deadtime_s", "一次死区", self._spin(0, 5000, 2, " ns")),
            ("primary_zvs_margin_required", "ZVS 首选裕量", self._spin(0.1, 20, 3)),
        ]
        for key, label, widget in cap_specs:
            self.fields[key] = widget
            form.addRow(label, widget)
        self.primary_device_combo = QComboBox()
        for device in DeviceDatabase().primary:
            self.primary_device_combo.addItem(
                f"{device.part_number} | Coss={device.coss_er_f*1e12:.0f} pF | Qoss={device.qoss_c*1e9:.1f} nC",
                device.part_number,
            )
        form.addRow("ZVS 主功率器件", self.primary_device_combo)
        layout.addWidget(capacitors)

        actions = QGroupBox("快速运行")
        action_layout = QVBoxLayout(actions)
        action_layout.setSpacing(6)
        self.run_button = QPushButton("运行 LLC 完整计算")
        self.fast_wave_button = QPushButton("快速波形")
        self.detail_wave_button = QPushButton("详细开关波形")
        self.small_signal_button = QPushButton("小信号 G(s) / G(z)")
        self.run_button.clicked.connect(self.run_design)
        self.fast_wave_button.clicked.connect(lambda: self.run_waveforms(False))
        self.detail_wave_button.clicked.connect(lambda: self.run_waveforms(True))
        self.small_signal_button.clicked.connect(lambda: self.run_small_signal({}))
        action_layout.addWidget(self.run_button)
        row = QHBoxLayout()
        row.addWidget(self.fast_wave_button)
        row.addWidget(self.detail_wave_button)
        action_layout.addLayout(row)
        action_layout.addWidget(self.small_signal_button)
        layout.addWidget(actions)
        layout.addStretch(1)
        scroll.setWidget(content)
        scroll.setMinimumWidth(265)
        scroll.setMaximumWidth(420)
        return scroll

    def _build_workspace(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)

        summary = QWidget()
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(10, 10, 10, 10)
        self.topology_view = TopologyView()
        self.topology_view.component_selected.connect(self._component_selected)
        summary_layout.addWidget(self.topology_view)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text, 1)
        self.tabs.addTab(summary, "设计总览")

        self.gain_figure = Figure(figsize=(9, 6))
        self.gain_canvas = FigureCanvasQTAgg(self.gain_figure)
        self.tabs.addTab(self.gain_canvas, "增益 / 工作区")

        self.q_zvs_view = LLCQZVSView()
        self.q_zvs_view.analysis_requested.connect(self.run_q_zvs)
        self.tabs.addTab(self.q_zvs_view, "Q / ZVS")

        self.transformer_design_view = TransformerDesignView()
        self.transformer_design_view.analysis_requested.connect(self.run_transformer_design)
        self.transformer_design_view.apply_turns_requested.connect(self._apply_transformer_turns)
        self.transformer_design_view.export_requested.connect(self._export_transformer_design)
        self.tabs.addTab(self.transformer_design_view, "变压器")

        self.waveform_view = WaveformView()
        self.waveform_view.fast_requested.connect(lambda options: self.run_waveforms(False, options))
        self.waveform_view.detailed_requested.connect(lambda options: self.run_waveforms(True, options))
        self.tabs.addTab(self.waveform_view, "波形")

        self.small_signal_view = SmallSignalView()
        self.small_signal_view.analysis_requested.connect(self.run_small_signal)
        self.tabs.addTab(self.small_signal_view, "小信号")

        self.digital_loop_view = DigitalLoopView()
        self.digital_loop_view.analysis_requested.connect(self.run_digital_loop)
        self.tabs.addTab(self.digital_loop_view, "数字控制")
        return self.tabs

    def _component_selected(self, key: str) -> None:
        self.statusBar().showMessage(f"已选择功率级组件：{key}")
        mapping = {
            "bridge": ("v_bridge", "i_resonant"),
            "lr": ("i_resonant", "v_resonant_inductor", "energy_lr"),
            "cr": ("v_resonant_cap", "i_resonant"),
            "transformer": ("v_transformer_primary", "i_transformer_primary", "i_magnetizing", "v_transformer_secondary"),
            "sr": ("i_transformer_secondary", "i_rectified"),
            "output": ("i_output_cap", "v_output", "v_output_ripple"),
        }
        if key == "transformer":
            self.tabs.setCurrentWidget(self.transformer_design_view)
            return
        if self.waveform_view.bundle is not None and key in mapping:
            self.waveform_view.select_component(key)
            self.tabs.setCurrentWidget(self.waveform_view)

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.progress.setVisible(busy)
        for button in (self.run_button, self.fast_wave_button, self.detail_wave_button, self.small_signal_button):
            button.setEnabled(not busy)
        self.waveform_view.set_busy(busy)
        self.small_signal_view.set_busy(busy)
        self.digital_loop_view.set_busy(busy)
        self.transformer_design_view.set_busy(busy)
        self.statusBar().showMessage(message if busy else "就绪")

    def _append_log(self, message: str) -> None:
        self.log_text.appendPlainText(message.rstrip())

    def _spec_from_widgets(self) -> LLCDesignSpec:
        changes = {}
        frequency_keys = {"resonant_frequency_hz", "minimum_frequency_hz", "maximum_frequency_hz"}
        microfarad_keys = {"bus_capacitance_f", "output_capacitance_f"}
        for key, widget in self.fields.items():
            value = widget.value()
            if key in frequency_keys:
                value *= 1e3
            elif key in microfarad_keys:
                value *= 1e-6
            elif key == "requested_hold_time_s":
                value *= 1e-3
            elif key == "output_cap_esr_ohm":
                value *= 1e-3
            elif key == "primary_deadtime_s":
                value *= 1e-9
            elif key in {"primary_turns", "secondary_turns"}:
                value = int(value)
            changes[key] = value
        changes["primary_topology"] = PrimaryTopology(self.topology.currentData())
        changes["primary_device"] = self.primary_device_combo.currentData()
        spec = self.spec.clone(**changes)
        spec.validate()
        return spec

    def _load_spec_to_widgets(self, spec: LLCDesignSpec) -> None:
        for key, widget in self.fields.items():
            value = getattr(spec, key)
            if key in {"resonant_frequency_hz", "minimum_frequency_hz", "maximum_frequency_hz"}:
                value /= 1e3
            elif key in {"bus_capacitance_f", "output_capacitance_f"}:
                value *= 1e6
            elif key == "requested_hold_time_s":
                value *= 1e3
            elif key == "output_cap_esr_ohm":
                value *= 1e3
            elif key == "primary_deadtime_s":
                value *= 1e9
            widget.setValue(value)
        self.topology.setCurrentIndex(0 if PrimaryTopology(spec.primary_topology) == PrimaryTopology.FULL_BRIDGE else 1)
        idx = self.primary_device_combo.findData(spec.primary_device)
        if idx >= 0:
            self.primary_device_combo.setCurrentIndex(idx)
        if hasattr(self, "waveform_view"):
            self.waveform_view.set_nominal_work_point(spec.vbus_nom_v)
        if hasattr(self, "small_signal_view"):
            self.small_signal_view.set_nominal_work_point(spec.vbus_nom_v)
        if hasattr(self, "digital_loop_view"):
            self.digital_loop_view.set_nominal_work_point(spec.vbus_nom_v)
        if hasattr(self, "transformer_design_view"):
            self.transformer_design_view.set_nominal_spec(spec)

    def load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载 LLC JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.spec = load_spec(path)
            self._load_spec_to_widgets(self.spec)
            self._append_log(f"Loaded: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def save_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存 LLC JSON", "llc_project.json", "JSON (*.json)")
        if not path:
            return
        try:
            self.spec = self._spec_from_widgets()
            save_spec(self.spec, path)
            self._append_log(f"Saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))

    def choose_output_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", str(self.output_directory))
        if path:
            self.output_directory = Path(path)
            self.statusBar().showMessage(f"输出目录：{path}")

    def show_about(self) -> None:
        QMessageBox.about(
            self, "关于",
            "<h3>电源设计工具箱 LLC Design / Waveform / Control</h3>"
            "<p>工具设计人：<b>杨帅锅</b></p>"
            "<p>开关电源仿真与实用设计</p>"
            "<p>LLC 谐振变换器设计、波形仿真与数字控制小信号建模工具。</p>",
        )

    def _run_worker(self, label: str, function, callback) -> None:
        self._set_busy(True, label)
        worker = FunctionWorker(function)
        # Keep a strong reference until the worker finishes; the thread pool
        # would otherwise drop it and its signals QObject could be
        # garbage-collected before the result callback fires.
        self._active_workers.append(worker)
        worker.signals.result.connect(callback)
        worker.signals.error.connect(self._worker_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        worker.signals.finished.connect(
            lambda: self._active_workers.remove(worker))
        self.thread_pool.start(worker)

    def _worker_error(self, error: str) -> None:
        self._append_log(error)
        QMessageBox.critical(self, "计算失败", error.splitlines()[-1])

    def run_design(self) -> None:
        try:
            self.spec = self._spec_from_widgets()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc)); return
        self._run_worker("正在运行 LLC 完整计算…", lambda: LLCSystemAnalyzer().analyze(self.spec), self._design_ready)

    def _design_ready(self, analysis: SystemAnalysis) -> None:
        self.system_analysis = analysis
        nominal = analysis.nominal
        op = nominal.operating_point
        text = [
            "LLC 设计结果摘要",
            "=" * 72,
            f"拓扑: {PrimaryTopology(analysis.spec.primary_topology).value} + {analysis.spec.secondary_topology.value}",
            f"输入/输出: {analysis.spec.vbus_nom_v:.1f} Vdc -> {analysis.spec.vout_v:.3f} V / {analysis.spec.pout_w/1000:.3f} kW",
            "",
            f"Lr={analysis.tank.lr_h*1e6:.6f} µH",
            f"Cr={analysis.tank.cr_f*1e9:.6f} nF",
            f"Lm={analysis.tank.lm_h*1e6:.6f} µH",
            f"标称 fs={op.switching_frequency_hz/1e3:.6f} kHz",
            "",
            f"变压器: {analysis.transformer.core.part_number}, {analysis.spec.primary_turns}:{analysis.spec.secondary_turns}, fill={analysis.transformer.fill_factor*100:.3f}%",
            f"谐振电感: {analysis.resonant_inductor.core.part_number}, {analysis.resonant_inductor.turns} T, {analysis.resonant_inductor.layers} layers",
            "",
            f"标称损耗={nominal.total_loss_w:.5f} W",
            f"标称效率={nominal.efficiency*100:.5f}%",
            f"状态={'PASS' if analysis.feasible else 'FAIL'}",
        ]
        if analysis.feasibility_reasons:
            text.extend(["", "不可行原因:"] + [f"- {reason}" for reason in analysis.feasibility_reasons])
        self.summary_text.setPlainText("\n".join(text))
        self._plot_gain(analysis)
        try:
            self.q_zvs_analysis = build_q_zvs_analysis(analysis.spec)
            self.q_zvs_view.set_analysis(self.q_zvs_analysis)
        except Exception as exc:
            self._append_log(f"Q/ZVS map warning: {exc}")
        self._append_log("Design analysis complete")
        self.tabs.setCurrentIndex(0)

    def _plot_gain(self, analysis: SystemAnalysis) -> None:
        self.gain_figure.clear()
        ax = self.gain_figure.add_subplot(111)
        spec, tank = analysis.spec, analysis.tank
        frequencies = np.linspace(spec.minimum_frequency_hz, spec.maximum_frequency_hz, 800)
        for load in (0.1, 0.25, 0.5, 0.75, 1.0):
            pout = spec.pout_w * max(load, spec.minimum_modeled_load_fraction)
            rac = equivalent_ac_load_ohm(spec.turns_ratio, spec.vout_v + spec.rectifier_equivalent_drop_v,
                                         pout * (1.0 + spec.rectifier_equivalent_drop_v / spec.vout_v))
            ax.plot(frequencies / 1e3, [gain(tank, f, rac) for f in frequencies], label=f"{load*100:.0f}%")
        for bus in (spec.vbus_hold_end_v, spec.vbus_min_normal_v, spec.vbus_nom_v, spec.vbus_max_v):
            ax.axhline(target_gain(spec, bus), linestyle="--", linewidth=0.9, label=f"{bus:.0f} V")
        for point in analysis.operating_points:
            op = point.operating_point
            ax.scatter(op.switching_frequency_hz / 1e3, op.achieved_gain, s=28)
        ax.axvline(tank.fr_hz / 1e3, linestyle=":", label="fr")
        ax.set_xlabel("Switching frequency (kHz)")
        ax.set_ylabel("Normalized gain")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        self.gain_figure.tight_layout()
        self.gain_canvas.draw_idle()


    def run_transformer_design(self, core_input=None, settings=None) -> None:
        try:
            self.spec = self._spec_from_widgets()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc)); return
        if core_input is None:
            core_input = self.transformer_design_view._core_input()
        if settings is None:
            settings = self.transformer_design_view._settings()
        self._run_worker(
            "正在根据磁芯规格书自动设计 LLC 变压器…",
            lambda: synthesize_transformer(self.spec, core_input, settings),
            self._transformer_design_ready,
        )

    def _transformer_design_ready(self, result: TransformerSynthesisResult) -> None:
        self.transformer_synthesis = result
        self.transformer_design_view.set_result(result)
        self.tabs.setCurrentWidget(self.transformer_design_view)
        self._append_log(
            f"Transformer synthesis: {result.core.shape}/{result.core.material_grade}, "
            f"Np:Ns={result.primary_turns}:{result.secondary_turns}, "
            f"P={result.primary_litz.strand_count}x{result.primary_litz.strand_diameter_mm:.3f} mm, "
            f"S={result.secondary_litz.strand_count}x{result.secondary_litz.strand_diameter_mm:.3f} mm, "
            f"loss={result.total_nominal_loss_w:.3f} W, feasible={result.feasible}"
        )

    def _apply_transformer_turns(self, primary_turns: int, secondary_turns: int) -> None:
        self.primary_turns.setValue(int(primary_turns))
        self.secondary_turns.setValue(int(secondary_turns))
        self.spec = self._spec_from_widgets()
        self.statusBar().showMessage(
            f"已将变压器推荐匝数应用到主设计：{primary_turns}:{secondary_turns}"
        )

    def _export_transformer_design(self) -> None:
        if self.transformer_synthesis is None:
            return
        out = self.output_directory / "transformer_design"
        paths = export_transformer_synthesis(self.transformer_synthesis, out)
        self._append_log(f"Transformer design exported: {paths['json']}")
        self.statusBar().showMessage(f"变压器设计已导出到：{out}")



    def run_q_zvs(self) -> None:
        try:
            self.spec = self._spec_from_widgets()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc)); return
        self._run_worker(
            "正在计算 LLC 多负载 Q / ZVS 工作区域…",
            lambda: build_q_zvs_analysis(self.spec),
            self._q_zvs_ready,
        )

    def _q_zvs_ready(self, result: LLCQZVSAnalysis) -> None:
        self.q_zvs_analysis = result
        self.q_zvs_view.set_analysis(result)
        self.tabs.setCurrentWidget(self.q_zvs_view)
        self._append_log(
            f"Q/ZVS map ready: {len(result.map.load_fractions)} load levels, "
            f"{len(result.workpoints)} operating points, warnings={len(result.warnings)}"
        )

    def _ensure_analysis(self) -> SystemAnalysis:
        if self.system_analysis is None or self.system_analysis.spec != self.spec:
            self.system_analysis = LLCSystemAnalyzer().analyze(self.spec)
        return self.system_analysis

    def _build_small_signal(self, options: dict) -> SmallSignalAnalysis:
        analysis = self._ensure_analysis()
        defaults = {
            "vbus_v": self.spec.vbus_nom_v,
            "load_fraction": 1.0,
            "sample_time_s": 20e-6,
            "control_input_kind": ControlInputKind.FREQUENCY_HZ,
            "timer_clock_hz": 120e6,
            "input_delay_samples": 0,
        }
        defaults.update(options)
        return build_small_signal_analysis(
            self.spec,
            system_analysis=analysis,
            **defaults,
        )

    def run_waveforms(self, detailed: bool, options: dict | None = None) -> None:
        try:
            self.spec = self._spec_from_widgets()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc)); return
        options = options or {}
        def calculate():
            small = self._build_small_signal(options)
            model = DynamicPhasorModel(small.parameters)
            if detailed:
                bundle = simulate_switched_steady_state(
                    model, small.steady_state,
                    SwitchedSimulationConfig(samples_per_cycle=512, output_cycles=2))
            else:
                bundle = reconstruct_dynamic_phasor_waveforms(
                    model, small.steady_state, cycles=2, samples_per_cycle=1024)
            return small, bundle
        self._run_worker(
            "正在计算详细开关波形…" if detailed else "正在重构关键波形…",
            calculate, self._waveforms_ready)

    def _waveforms_ready(self, result: tuple[SmallSignalAnalysis, WaveformBundle]) -> None:
        small, bundle = result
        self.small_signal_analysis = small
        self.waveform_view.set_bundle(bundle)
        self.tabs.setCurrentWidget(self.waveform_view)
        self._append_log(f"Waveforms ready: {bundle.model_name}; warnings={len(bundle.warnings)}")

    def run_small_signal(self, options: dict) -> None:
        try:
            self.spec = self._spec_from_widgets()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc)); return
        self._run_worker("正在建立 LLC 小信号与 ZOH 对象…",
                         lambda: self._build_small_signal(options), self._small_signal_ready)

    def _small_signal_ready(self, result: SmallSignalAnalysis) -> None:
        self.small_signal_analysis = result
        self.small_signal_view.set_analysis(result)
        self.tabs.setCurrentWidget(self.small_signal_view)
        self._append_log(
            f"Small signal ready: G(0)={result.continuous_transfer.dc_gain:.9g}, "
            f"stable={result.stable}")

    def run_digital_loop(self, options: dict) -> None:
        try:
            self.spec = self._spec_from_widgets()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc)); return

        def calculate():
            small_options = options.get("small_signal", {}) if options else {}
            loop_options = options.get("loop", {}) if options else {}
            small = self._build_small_signal(small_options)
            return build_digital_loop_analysis(small, **loop_options)

        self._run_worker(
            "正在建立完整 LLC 数字电压环…",
            calculate,
            self._digital_loop_ready,
        )

    def _digital_loop_ready(self, result: DigitalLoopAnalysis) -> None:
        self.digital_loop_analysis = result
        self.small_signal_analysis = result.small_signal
        self.digital_loop_view.set_analysis(result)
        self.tabs.setCurrentWidget(self.digital_loop_view)
        margin = result.margins_nominal_delay
        self._append_log(
            "Digital loop ready: "
            f"PCMD={result.fm_operating_point.command_pu:.6g}, "
            f"Kfm={result.fm_operating_point.gain_hz_per_pu:.7g} Hz/pu, "
            f"fc={margin.critical_gain_crossover_hz}, "
            f"PM={margin.phase_margin_deg}, stable={result.likely_stable}"
        )
