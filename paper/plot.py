"""Regenerate the manuscript figure from committed numerical CSV outputs."""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def values(rows, key):
    if not rows:
        raise ValueError(f"No recorded rows selected for {key}")
    return np.array([float(row[key]) for row in rows])


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.35, 2.35), layout="constrained")
rows = read("audit.csv")
for group in ("held_out", "shifted"):
    ranks = sorted({int(r["rank"]) for r in rows})
    errors = [
        max(
            float(r["relative_l2_error"])
            for r in rows
            if r["group"] == group and int(r["rank"]) == rank
        )
        for rank in ranks
    ]
    axes[0].semilogy(ranks, errors, "o-", label=group.replace("_", " "))
axes[0].set(
    title="Worst prediction error", xlabel="Reduced dimension", ylabel="Relative L2 error"
)
axes[0].legend(fontsize=8)
data = [r for r in rows if int(r["rank"]) == 20]
axes[1].loglog(values(data, "l2_error"), values(data, "l2_bound"), "o", markersize=3)
lo, hi = min(values(data, "l2_error")), max(values(data, "l2_bound"))
axes[1].plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
axes[1].set(
    title="Rank-20 bound coverage", xlabel="Measured L2 error", ylabel="Residual L2 bound"
)

for ax in axes:
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    if ax.get_xscale() == "log":
        points = np.unique(ax.lines[0].get_xdata())
        if 2 <= len(points) <= 6:
            ax.set_xticks(points, labels=[f"{x:.3g}" for x in points])
    ax.grid(alpha=0.2, which="both")
fig.savefig(ROOT / "paper" / "figure.pdf")
plt.close(fig)
