# Power Design Toolkit V7.3

Integrated engineering design and control-analysis toolkit for:

- **LLC resonant converter**: resonant-tank design, multi-load Q/gain maps, theoretical + engineering ZVS maps, operating-point trajectory, switching/dynamic waveforms, magnetics/loss analysis and digital voltage-loop design.
- **Single-phase TTPL PFC**: firmware-shaped dual-loop control, three sensing chains, open-loop Bode analysis, full settled AC-cycle solver, zero-crossing analyzer, workpoint-derived switching waveforms, PF/THD/harmonics.
- **Three-phase Vienna PFC**: DC-voltage outer loop + three ABC stationary-frame current loops, split-bus midpoint balance, three-phase sensing, common-mode/third-harmonic modulation support, full line-cycle solver, sector analysis, workpoint-derived three-level switching waveforms and per-phase PF/THD.

V7 reorganizes the GUI into separate **LLC** and **PFC** workspaces. The PFC workspace contains independent **TTPL** and **Vienna** sub-workspaces.

## 开发时间线 / Development Timeline

功能演进与缺陷修复同步记录，向所有使用者公开。
Legend: 🚀 Feature · 🐛 Bugfix · 🎨 GUI/UX · 🧪 Test/CI/Build

| 版本 / 日期 | 类型 | 内容 |
| --- | --- | --- |
| V4.1 | 🚀 | 可拖动 Bode 光标（功率级小信号页 + 完整数字环页），幅频/相频同步、对数频率插值 |
| V4.2 | 🐛 | Bode 光标中文标注方块修复：移除强制 monospace，新增 Win/macOS/Linux CJK 字体解析器 |
| V6 | 🚀 | LLC 与 PFC 拆分为独立顶层工作区；PFC Bode 默认仅开环、逐传递函数独立显隐；新增完整 AC 周期与开关周期波形 |
| 2026-08-02 | 🚀 | Windows/macOS 启动脚本（中文文件名、venv 兜底） |
| 2026-08-02 | 🧪 | PyInstaller 打包入口 + GitHub Actions 构建（macOS/Windows，tag 触发发布） |
| 2026-08-02 | 🐛 | CI 修复：bash 续行符、Windows pwsh `ls -la` 兼容 |
| 2026-08-03 | 🐛 | 修复 Windows 启动脚本：UTF-8/CRLF 解析损坏 + errorlevel 展开错误 |
| V7 | 🚀 | 结构性重构：LLC/PFC/Vienna 三工作区；新增三相 Vienna PFC 拓扑（ABC 电流环 + 中点平衡 + 三电平调制） |
| V7.1 | 🎨 | LLC GUI 信息架构整理：可停靠参数面板（F4）、专注模式（F9）、运行日志（F8）、Ctrl+R 按页运行 |
| V7.1.1 | 🎨 | “设计参数”统一为单一显隐切换；移除 dock 关闭按钮避免不一致 |
| V7.1.2 | 🎨 | 信号链框图可读性：放大块/标签/箭头，正交路由，角色配色与文字层级 |
| V7.1.3 | 🚀 | LLC 数字控制信号链改为可缩放/平移矢量画布；全屏检视；选中高亮 + 联动参数/Bode |
| V7.1.4 | 🚀 | LLC 变压器规格书驱动设计：TDK PQ35/35 预设、整数匝数搜索、Litz 选型、iGSE/分层铜损、一键回填主工程 |
| V7.1.5 | 🚀 | TTPL 电流环 PI 一键自动整定 + indu_comp 包络校验；Vienna 保守外环 PI（~53° 相位裕度） |
| V7.1.5 | 🐛 | 修复 TTPL 开关电流重构：跨 PWM 周期连续积分，不再每周期重置为三角纹波模板 |
| V7.1.5 | 🐛 | 修复 TTPL/Vienna 采样链延迟双重计数：ADC/SOC 延迟归采样块，计算/PWM 延迟归固件块 |
| V7.2 | 🚀 | C99 Float32 控制代码生成器（LLC / TTPL / Vienna），Init/Reset/ControlStep API + ISR 集成模板 + 稳定性报告 |
| V7.2 | 🐛 | 修复 Vienna GUI `self.fsw` spinbox 与 Figure 属性冲突（重命名为 `self.fsw_fig`） |
| V7.2.1 | 🎨 | PFC UI 密度优化：TTPL 紧凑控制头、可收起框图/参数、Bode 复选框默认折叠；Windows 11 样式统一 |
| V7.3 | 🚀 | PFC 电感设计页（Magnetics High Flux 254，DC-bias 跌落 + 铜损 + 一键回填）；Vienna UI 与 TTPL/LLC 统一 |
| V7.3 | 🐛 | 修复 LLC/PFC 共享模拟采样示意图在窄面板下被裁剪（高度自适应） |
| V7.4 | 🚀 | 工具栏“检查更新”：后台线程查 GitHub Releases API；启动 2.5s 静默自检；联系邮箱/微信 |
| 2026-08-17 | 🚀 | GUI 自动适配 Windows 11 深色/浅色主题（注册表检测 + Fusion 调色板 + 两套 token） |
| 2026-08-17 | 🐛 | 修复 Win11 深色主题下整屏不可见：浅色背景 + 跟随系统的浅色文字 → 统一从主题模块取色 |

## Install

Core/CLI:

```bash
python -m pip install -e .
```

GUI:

```bash
python -m pip install -e ".[gui]"
```

Development tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Start the GUI

```bash
python -m llc_design gui
```

or:

```bash
power-design-gui
```

The launcher first asks whether to enter **LLC Design** or **PFC Design**. Both top-level windows preserve their state while switching.

## LLC V7 highlights

### Q / gain / ZVS map

The LLC page calculates Q at multiple load fractions (default 10/25/50/75/100/120%), overlays gain curves, and shows real bus/load workpoints. The ZVS view separates:

1. **Theoretical inductive region**: positive tank input phase / `Im{Zin} > 0`.
2. **Engineering ZVS margin**: commutation current and deadtime versus device Qoss/Coss charge and energy demand.

The map therefore distinguishes capacitive operation, inductive-but-insufficient commutation, warning margin and safe ZVS margin.

### Digital control diagram

The LLC digital-control page has a clickable signal-chain diagram:

`Vref -> controller -> PCMD clamp -> PCMD/TBPRD FM LUT -> PWM/ZOH/delay -> Gvf -> Vout -> divider/op-amp/RC/ADC -> feedback`

Selecting a block moves to the corresponding parameter group and focuses the associated Bode trace. The Bode cursor reports gain/phase and the phase-budget view decomposes the total response by controller, FM, plant, sensing and delay.

## Single-phase TTPL PFC V7

The main control structure remains **two feedback loops**:

- 50 kHz inductor-current inner loop.
- 10 kHz DC-bus voltage outer loop.

The 25 kHz AMC layer is **not** a third feedback loop; it generates the current reference and feed-forward terms.

Firmware-shaped relations used by the model include:

```text
i_ref      = gcmd * |Vac_meas|
duty_ff    = 1 - |Vac_meas| / Vbus_set
indu_comp  = clamp(0.085 * i_ref, 0.7, 1.0)
duty_total = clamp(duty_ff + duty_pi * indu_comp)
```

Three external sensing paths are modeled independently:

- inductor current `iL`;
- AC input voltage `Vac`;
- DC bus voltage `Vbus`.

Each path can include front-end gain/divider, op-amp bandwidth, external RC, ADC aperture/sample timing, digital filtering and delay.

### PFC Bode policy

The current-loop and voltage-loop pages default to **open loop only**. Individual plant/controller/sensing/PWM/closed-loop traces can be shown with checkboxes. Clicking a control-diagram block focuses the relevant trace. The cursor provides gain/phase and phase-budget data.

### PFC waveforms

The line-cycle solver is multi-rate (50/25/10 kHz control updates) and uses a bus-capacitor energy state. It explicitly distinguishes:

- non-negative boost-inductor current magnitude;
- signed AC input current;
- real bus-voltage dynamics.

The local switching waveform is reconstructed from a selected point in the **final settled AC cycle**, rather than from an unrelated independent workpoint.

A dedicated zero-crossing analyzer shows the firmware-like half-cycle/ZC states, current reference/error, duty FF/PI/total, minimum-pulse state, LF bridge intent and PI reset events.

PF/THD is calculated on a complete settled line period with explicit integer harmonics.

## Three-phase Vienna PFC V7

Vienna reuses the PFC-common controller/sensing/Bode infrastructure, but has topology-specific plant and modulation models.

### Main control

The V7 baseline uses stationary-frame ABC control:

```text
DC Vbus voltage outer loop
        -> Gcmd
        -> ia*=Gcmd*va, ib*=Gcmd*vb, ic*=Gcmd*vc
        -> three phase-current controllers
        -> Vienna modulator
        -> Vienna power stage
```

The split DC bus adds an **auxiliary midpoint-balance controller** based on `Vdc+ - Vdc-`; it is intentionally presented separately from the two main PFC feedback loops.

### Vienna averaged model

Dynamic states include:

```text
ia, ib, ic, Vdc+, Vdc-
```

The model includes:

- three-wire floating-neutral current dynamics;
- optional common-mode/third-harmonic injection;
- optional `R*i_ref + L*di_ref/dt` inductor-voltage-drop feed-forward;
- Vienna signed active-state modulation;
- center-switch zero-state duty `D0 = 1 - |m|`;
- minimum-pulse handling;
- split-bus capacitor energy dynamics;
- averaged midpoint current from zero-state occupancy.

### Vienna sensing

Core sensing channels are:

- `Ia/Ib/Ic`;
- `Va/Vb/Vc`;
- `Vdc+/Vdc-`.

The GUI can lock phase channels to common hardware values or unlock gain/offset mismatch diagnostics.

### Vienna Bode and waveforms

Three return-ratio views are kept separate and clean:

- Phase-A current open loop (ABC loops are symmetric in the nominal model).
- Total DC-voltage open loop.
- Midpoint-balance open loop.

All default to open loop only; component traces are opt-in. A separate sampling Bode page can focus current, phase-voltage or split-bus sensing.

The final settled 3-phase AC-cycle view includes Vabc, Iabc and references, modulation/zero-state duty, split-bus voltages, midpoint current, control outputs and power. The sector analyzer tracks sector, phase polarity, modulation and midpoint current. Switching waveforms are derived from the selected AC phase and include center-switch gates, three-level converter voltage, phase-current ripple, upper/lower diode current and split-bus/midpoint currents.

## CLI

Single-phase PFC:

```bash
pfc-control-lab --output output/ttpl
```

Vienna:

```bash
vienna-control-lab --vll 400 --vdc 700 --power 10000 --output output/vienna
```

LLC CLI remains available through:

```bash
llc-design --help
```

## Validation status

The source tree includes unit/regression tests covering LLC tank/magnetics/digital control/Q-ZVS, TTPL sensing/control/waveforms/PF-THD, and Vienna nested-loop/midpoint/switching behavior.

GUI source is import/compile checked by the project; a real window run still requires PySide6 on the target machine.

## Modeling boundary

Bode pages are local linear models. Saturation, minimum pulse, TTPL zero-crossing state transitions, Vienna switching-state decisions and protection behavior are assessed in the time-domain solvers rather than being folded into a misleading single LTI transfer function.

## LLC transformer design (V7.1.4)

The LLC GUI now includes a dedicated **变压器** page.  Enter ferrite/core/bobbin datasheet values (Ae/Amin/le/Ve/AL/µe/AN/lN etc.) or load the bundled TDK PQ35/35 example, then run automatic turns and Litz synthesis.  The default conductor is 0.10 mm Litz and strand count is rounded in multiples of 50.  The page reports winding feasibility, gap/AL, Bpk, DCR, harmonic copper loss, iGSE core loss, hotspot estimate and per-workpoint loss.

CLI example:

```bash
python -m llc_design transformer-design --preset TDK_PQ35_35_B65881A_N87 --output output/transformer_design
```

### PFC stability tools (V7.1.5)

The single-phase TTPL control page includes **一键稳定整定并应用**.  It designs a conservative
50 kHz current-loop PI from the current L/R, current-sense filter, ADC/ZOH and computation/PWM
delay, then checks the scheduled `indu_comp` gain over a line/load/phase envelope.  The legacy
firmware PI remains available as an explicit comparison preset.

The Vienna page now validates the Bode, three-phase AC-cycle and local switching stages
independently.  If a calculation or plot fails, the GUI reports the exact failing stage and exposes
the complete traceback through the error dialog's detailed-information section.

## V7.2 C99 / Float32 control-code generator

The GUI and CLI can generate portable real-time control cores for:

- LLC frequency control;
- single-phase TTPL PFC double-loop control;
- three-phase Vienna PFC ABC current control + DC-voltage loop + midpoint balance.

The generated code deliberately stops before the BSP boundary. ADC/PWM/GPIO/interrupt-controller setup is not emitted. The BSP supplies engineering-unit inputs and consumes semantic duty/frequency outputs.

CLI examples:

```bash
python -m power_codegen ttpl --output output/generated_ttpl
python -m power_codegen vienna --output output/generated_vienna
python -m power_codegen llc --output output/generated_llc
```

Every generated folder contains C99/float32 sources, a compile-ready ISR integration template, `design_snapshot.json`, and `stability_report.txt`.

### V7.2.1 PFC workspace UI

The TTPL PFC page now mirrors the LLC digital-control workspace: a compact interactive overview, collapsible parameter inspector, collapsible transfer-function selector and a larger default Bode/waveform area. The PFC main window also uses the same cross-platform Qt stylesheet as LLC so Windows 11 no longer falls back to inconsistent native-looking tabs/buttons/spin boxes.


## V7.3 PFC inductor design

TTPL and Vienna workspaces include a dedicated **Inductor Design** page.

### Default core

Built-in default is the Magnetics **High Flux** powder-core **Core Data 254** (60 µ grade, AL = 81 nH/T²), with editable geometry/AL:

- Le = 98.4 mm, Ae = 110.6 mm², Ve = 10880 mm³
- OD = 40.77 mm, ID = 23.32 mm, HT = 15.37 mm

Only permeability grades with both a Core Data 254 AL value and complete published DC-bias/core-loss coefficients are selectable in the automatic model.

### Manufacturer fits

- DC-bias permeability droop: `%ui = 1 / (a + b·H^c)`, H in Oe.
- Core-loss density: `Pv = a·B^b·f^c`, B in T, f in kHz, Pv in mW/cm³.
- Full-load L(I) droop curve and line-cycle switching-ripple/core-loss calculation.

### Copper model

Default winding is enamelled round copper; the page sizes the wire from current density, computes hot DCR and DC I²R copper loss, and reports the full-load operating point. Skin/proximity losses are intentionally excluded in V7.3 per the design requirement.

### Apply back

One click applies the calculated full-load L and hot DCR back into the TTPL/Vienna power-stage parameters.

### V7.3 Vienna UI unification

The Vienna workspace now uses the same compact toolbar/diagram/hidden-parameter interaction model as TTPL/LLC (oversized title row and oversized diagram removed). The shared LLC/PFC analog-sense schematic wraps responsively in narrow inspectors instead of being clipped.

## V7.4 Update check and contact info

The GUI toolbar (top-right, LLC and PFC workspaces) shows the author's contact info and an **检查更新** button. The button queries the GitHub Releases API in a background thread: when a newer release exists it shows the release notes with a link to the download page. A silent auto-check runs 2.5 s after startup and only prompts when an update is available.

- 邮箱: maileyang@qq.com(点击可直接写信)
- 微信: maileyang

## License

MIT License — see [LICENSE](LICENSE).

