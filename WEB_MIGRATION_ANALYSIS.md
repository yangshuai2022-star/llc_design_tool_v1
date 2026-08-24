# LLC Design Tool → Web 迁移分析报告（第一阶段交付）

> 分析基于 `llc_design_tool_v1-main`（含先前 webapp/ 迭代）。纯只读结论，未修改任何内核文件。

## 0. 现状结论

仓库已存在一个可运行的 Web 后端（`webapp/`，FastAPI + 静态前端），架构正是
**Browser → FastAPI → Python Engineering Engine**：`webapp/service.py` 直接 import
`llc_design` 内核，浏览器只提交参数、渲染结果，**核心计算零复制、零改写**。
本报告确认该架构符合迁移目标，后续工作在此基础上增量完成，不推倒重来。

## 1. GUI 模块列表（保留不动）

| 模块 | 内容 |
|------|------|
| `llc_design/gui/` | PySide6 桌面端：`app.py`(run_gui) / `main_window.py` / `launcher.py` / `workers.py` / `theme.py` / `updater.py` |
| `llc_design/gui/widgets/` | bode_cursor / control_block_diagram / digital_loop_view / q_zvs_view / sense_schematic / small_signal_view / topology / transformer_design_view / waveform_view |
| `pfc_design/gui/` | PFC/Vienna 控制实验面板（同族，保留） |

## 2. Core 算法模块列表（只读复用，禁止修改）

| 模块 | 职责 | 核心函数 |
|------|------|----------|
| `core/spec.py` | 60+ 参数规格 dataclass | `LLCDesignSpec` / `PrimaryTopology` / `SecondaryTopology` |
| `core/tank.py` | FHA 谐振腔设计 | `design_tank` / `gain_vector` / `target_gain` / `equivalent_ac_load_ohm` / `solve_frequency` |
| `core/operating_point.py` | 工作点频域求解 | `solve_operating_point` |
| `core/q_zvs.py` | ZVS 判定 | （GUI/CLI 用） |
| `core/waveform.py` / `core/bus_capacitor.py` / `core/config.py` | 波形/母线电容/规格持久化 | `load_spec` / `save_spec` |
| `magnetics/transformer_designer.py` | 变压器综合（含候选摘要：热点温度/成本） | `design_transformer` → `TransformerDesign` |
| `magnetics/resonant_inductor.py` | 谐振电感综合 | `design_resonant_inductor` |
| `magnetics/litz.py` | 利兹线精确损耗（集肤/邻近/环流/端接） | `layered_litz_stack_loss` 等 |
| `magnetics/core.py` / `material.py` | 磁芯/材料与 Steinmetz 温度插值 | `CoreDatabase` / `MaterialDatabase` |
| `models/system.py` | 整机分析（效率/ZVS/损耗/可行性） | `LLCSystemAnalyzer.analyze(spec)` → `SystemAnalysis` |
| `models/devices.py` / `primary_bridge.py` / `synchronous_rectifier.py` / `capacitors.py` | 器件/桥臂/整流/电容损耗模型 | `DeviceDatabase` |
| `optimization/sweep.py` | **内核级优化器（Pareto 前沿）** | `LLCOptimizer.run` → `OptimizationResult` |
| `optimization/pareto.py` | Pareto 前沿 | `pareto_front` |
| `report/export.py` | 内核报告（Markdown/图表/计算书） | `write_markdown_report` / `create_all_plots` / `operating_point_rows` |
| `control/` `dynamics/` | 小信号/数字环路/时域开关模型（Web 后续 CodeGen 候选） | — |

## 3. 数据库模块列表

| 文件 | 内容 |
|------|------|
| `llc_design/data/cores.json` | 14 个磁芯（PQ/EE/EC/EER/ETD），含 Ae/Ve/le/质量/cost/热阻 |
| `llc_design/data/materials.json` | 4 种材料（N87 等）Steinmetz 温度点 |
| `llc_design/data/devices.json` | 原边 3 + 副边 2 器件参考记录（REF_*，非真实 datasheet） |
| `llc_design/data/transformer_core_presets.json` | 2 个变压器磁芯预设 |

> 注意：器件/磁芯均为**工程参考记录**，UI 只展示内核返回内容，不硬编码。

## 4. 可以直接复用的函数

已接入 `webapp/service.py`：`LLCDesignSpec.clone/validate`、`design_tank`、`gain_vector`、
`target_gain`、`equivalent_ac_load_ohm`、`LLCSystemAnalyzer.analyze`、`DeviceDatabase`。
待接入（下一步）：`LLCOptimizer.run`（优化器）、`pareto_front`、`operating_point_rows`（报告数据源）、
`load_spec/save_spec`（项目保存）。

## 5. 需要新增的 API 接口（对照设计稿 V1.0）

已有：`GET /api/health`、`GET /api/llc/defaults`、`POST /api/llc/analyze`。
待新增：
- `GET  /api/llc/cores` — 磁芯库浏览（数据已就绪）
- `POST /api/llc/optimize` — 基于内核 `LLCOptimizer` + 显式工程约束门控（ZVS/频率/电流/温升）
- `POST /api/llc/report` — PDF 报告（9 章节，fpdf2 + matplotlib）
- `POST /api/llc/report.xlsx` — Excel 导出（pandas + openpyxl）
- SPA 回退路由：`/llc`、`/llc/optimizer`、`/report`、`/pfc`（占位）
- 未来：`/api/pfc/*`、`/api/vienna/*`、`/api/codegen/*`（control 环路 C 导出 → CodeGen Tab）

## 6. 工程规则（本阶段起强制）

1. 不为 Web 展示修改核心计算模型；内核文件只读。
2. 所有新增功能经 `webapp/service.py` 调用内核。
3. 新增字段建立 pydantic schema（`webapp/schemas.py`），前端只消费 schema 化 JSON。
4. 优化算法保留工程约束：ZVS 裕量、频率范围、电流、温升，门控结果显式返回。
5. 所有新功能增加测试。
