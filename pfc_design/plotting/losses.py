"""Loss breakdown visualization: pie chart and stacked bar."""

import matplotlib.pyplot as plt
import numpy as np


def pie_chart(breakdown: dict[str, float], efficiency: float, total_loss: float,
              title: str = "PFC Loss Breakdown", save_path: str | None = None):
    """Generate a donut-style pie chart of loss breakdown.

    Args:
        breakdown: dict of {component_name: loss_W}
        efficiency: system efficiency (0-1)
        total_loss: total power loss in W
        title: chart title
        save_path: if provided, save to file
    """
    # Filter out negligible losses
    items = {k: v for k, v in breakdown.items() if v > total_loss * 0.005}
    labels = list(items.keys())
    sizes = list(items.values())

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%',
        startangle=90, pctdistance=0.82,
        colors=colors, wedgeprops=dict(width=0.35, edgecolor='w')
    )

    # Legend
    legend_labels = [f"{l}: {s:.1f}W" for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Components",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Center text
    ax.text(0, 0, f"η = {efficiency*100:.1f}%\nTotal Loss\n{total_loss:.1f}W",
            ha='center', va='center', fontsize=14, fontweight='bold')

    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def stacked_bar(breakdown: dict[str, float],
                title: str = "Loss Breakdown by Category",
                save_path: str | None = None):
    """Stacked bar chart grouped by category."""
    # Group components
    categories = {
        "Magnetics": ["Inductor Core", "Inductor Copper"],
        "Semiconductors": ["MOSFET Conduction", "MOSFET Switching",
                           "MOSFET Coss", "MOSFET Gate Drive",
                           "Diode Forward", "Diode RR", "Bridge Rectifier"],
        "Capacitors": ["Capacitor ESR"],
    }

    cat_totals = {}
    for cat, keys in categories.items():
        cat_totals[cat] = sum(breakdown.get(k, 0) for k in keys)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cat_totals.keys(), cat_totals.values(),
                  color=['#E74C3C', '#3498DB', '#2ECC71'], edgecolor='white')

    # Add value labels
    for bar, val in zip(bars, cat_totals.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}W', ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel("Power Loss (W)")
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
