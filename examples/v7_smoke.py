"""V7 non-GUI smoke example for LLC, TTPL PFC and Vienna PFC."""
from dataclasses import replace
from pathlib import Path
import sys

# Allow direct execution from an unpacked source tree before editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llc_design.core.q_zvs import build_q_zvs_analysis
from llc_design.core.spec import LLCDesignSpec
from pfc_design.control import PFCControlLabConfig, build_pfc_control_lab_analysis
from pfc_design.control.waveforms import simulate_pfc_line_cycle
from pfc_design.vienna import (
    ViennaControlLabConfig,
    build_vienna_control_lab_analysis,
    simulate_vienna_line_cycle,
)


def main() -> None:
    llc = build_q_zvs_analysis(LLCDesignSpec(), frequency_points=120)
    print("LLC Q/ZVS workpoints:", len(llc.workpoints))

    ttpl_cfg = replace(PFCControlLabConfig(), waveform_line_cycles=3, waveform_integration_rate_hz=250e3)
    ttpl_a = build_pfc_control_lab_analysis(ttpl_cfg)
    ttpl_w = simulate_pfc_line_cycle(ttpl_cfg)
    print("TTPL current PM:", ttpl_a.current_loop.margins.phase_margin_deg)
    print("TTPL PF / THD:", ttpl_w.metrics.power_factor, ttpl_w.metrics.current_thd_percent)

    vienna_cfg = replace(ViennaControlLabConfig(), waveform_line_cycles=4, waveform_integration_rate_hz=250e3, frequency_points=240)
    vienna_a = build_vienna_control_lab_analysis(vienna_cfg)
    vienna_w = simulate_vienna_line_cycle(vienna_cfg)
    print("Vienna current/voltage/balance PM:", vienna_a.current_loop.margins.phase_margin_deg, vienna_a.voltage_loop.margins.phase_margin_deg, vienna_a.balance_loop.margins.phase_margin_deg)
    print("Vienna PF / THD ABC:", vienna_w.metrics.overall_power_factor, vienna_w.metrics.phase_current_thd_percent)


if __name__ == "__main__":
    main()
