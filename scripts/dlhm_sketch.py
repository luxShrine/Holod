"""Draw the DLHM point-source geometry sketch used in the slides.

Outputs:
fig_geometry_sketch.png   -- DLHM point-source geometry + z_eff formula.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def geometry_sketch(out: Path) -> None:
    """Write the schematic of the source/sample/sensor geometry to ``out``."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    # positions along the optical axis (schematic, not to scale)
    src, sample, sensor = 0.0, 3.0, 9.0
    # diverging rays from the point source to the sensor edges
    for y_end in (1.6, -1.6):
        ax.plot([src, sensor], [0, y_end], color="tab:blue", lw=1.2, alpha=0.8)
        # ray height at the sample plane, to show the illuminated patch
        y_s = y_end * (sample - src) / (sensor - src)
        ax.plot([sample, sample], [-abs(y_s), abs(y_s)], color="tab:orange", lw=4)
    ax.plot([sensor, sensor], [-1.7, 1.7], color="k", lw=4)
    ax.scatter([src], [0], s=60, color="tab:red", zorder=5)
    ax.annotate("point source", (src, 0), textcoords="offset points", xytext=(-8, 12), ha="left")
    ax.annotate(
        "sample", (sample, 1.0), textcoords="offset points", xytext=(-10, 8), color="tab:orange"
    )
    ax.annotate("sensor", (sensor, 1.75), textcoords="offset points", xytext=(-12, 6))
    # distance markers
    for x0, x1, y, label in [(src, sample, -1.95, r"$z$"), (src, sensor, -2.45, r"$L$")]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops={"arrowstyle": "<->", "lw": 1})
        ax.text((x0 + x1) / 2, y - 0.12, label, ha="center", va="top", fontsize=13)
    ax.text(
        4.9,
        2.9,
        r"$M = L/z, \quad z_{\mathrm{eff}} = M\,(L - z) = \dfrac{L^2}{z} - L$"
        "\n"
        r"$\left|\dfrac{dz_{\mathrm{eff}}}{dz}\right| = \dfrac{L^2}{z^2}$"
        r"  $\Rightarrow$ near-field errors amplify",
        fontsize=12,
        ha="center",
        va="top",
        bbox={"boxstyle": "round", "fc": "whitesmoke", "ec": "0.6"},
    )
    ax.set_xlim(-0.8, 10.2)
    ax.set_ylim(-3.0, 3.1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
