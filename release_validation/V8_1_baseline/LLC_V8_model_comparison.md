# LLC V8 多保真模型对比计算书

> 模型层级：FHA 快速设计、多谐波谐波平衡、理想开关分段时域。

## 工作点

- 母线电压：400 V
- 目标输出：53 V
- 负载：100.000% / 3000 W
- 比较参考：`switched_time_domain`

## 模型结果

| 模型 | 收敛 | Fsw/kHz | Vo/V | Gain | Ir RMS/A | Ir PK/A | Im RMS/A | Is RMS/A | VCr PK/V | ΔIr RMS/% | ΔVCr PK/% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fha | PASS | 99.688762 | 53.000000 | 1.00125000 | 9.680411 | 13.689913 | 4.841391 | 62.870985 | 205.194532 | -6.5825 | -6.6269 |
| harmonic_balance | PASS | 99.258907 | 53.000000 | 1.00125000 | 10.257447 | 14.524717 | 4.897150 | 63.705007 | 218.265411 | -1.0141 | -0.6790 |
| switched_time_domain **REF** | PASS | 99.665617 | 52.955203 | 1.00041005 | 10.362528 | 14.674003 | 4.869039 | 63.826410 | 219.757540 | 0.0000 | 0.0000 |

## 收敛与模型边界

- `fha`：converged=True，residual=0.000000e+00，method=`closed_form_fha`
  - 谐波阶次：1
  - FHA retains only the fundamental resonant-state components.
  - Deadtime is displayed in bridge/gate traces but is not included in the FHA phasor solution.
- `harmonic_balance`：converged=True，residual=3.107417e-14，method=`exact_zero_crossing_multi_harmonic_balance`
  - 谐波阶次：1, 3, 5, 7
  - Multi-harmonic HB uses an ideal transformer and ideal full-bridge rectifier clamp.
  - Output-capacitor ripple is reconstructed after the HB DC balance; it is not a retained harmonic state.
  - Nonlinear MOSFET Coss, winding capacitance and leakage parasitics are reserved for the V8 parasitic layer.
- `switched_time_domain`：converged=True，residual=3.237970e-12，method=`periodic_shooting_rk4`
  - Switched model uses an ideal transformer and ideal full-bridge SR clamp.
  - MOSFET nonlinear Coss/Qoss, leakage inductance and winding capacitance are not yet included.
  - This is the V8.1 ideal switched reference; nonlinear Coss and magnetic parasitics are not included yet.

## 文件结构

- `fha/`：该模型的同步波形 CSV、统计、图表和单模型报告。
- `harmonic_balance/`：该模型的同步波形 CSV、统计、图表和单模型报告。
- `switched_time_domain/`：该模型的同步波形 CSV、统计、图表和单模型报告。