"""Reproducible synthetic heat-surrogate audit and analytical mesh validation."""

import csv
import json
import platform
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from .model import Parameters, ThermalPlate
from .reduction import ReducedModel, fit_pod


def training_parameters() -> list[Parameters]:
    """54 deterministic samples; source amplitudes are endpoint mixtures."""
    return [
        Parameters(kx, ky, beta, a, 1 - a)
        for kx, ky, beta, a in product([0.4, 1, 2.5], [0.5, 1.5, 3], [0, 6, 18], [0, 1])
    ]


def evaluation_parameters() -> list[tuple[str, Parameters]]:
    """36 interpolation samples and 12 deliberately shifted samples, seed 20260928."""
    rng = np.random.default_rng(20260928)
    result = []
    for _ in range(36):
        kx, ky, beta, a = rng.uniform([0.4, 0.5, 0, 0], [2.5, 3, 18, 1])
        result.append(("held_out", Parameters(kx, ky, beta, a, 1 - a)))
    for _ in range(12):
        kx, ky, beta, a = rng.uniform([0.04, 4, 22, 0], [0.15, 8, 40, 1])
        result.append(("shifted", Parameters(kx, ky, beta, a, 1 - a)))
    return result


def manufactured_convergence() -> list[dict]:
    """Compare to sin(pi*x)*sin(2*pi*y), independently of the discrete stencil."""
    p = Parameters(0.7, 1.6, 2)
    rows = []
    for n in (7, 15, 31, 63):
        grid = ThermalPlate(n)
        exact = (np.sin(np.pi * grid.x) * np.sin(2 * np.pi * grid.y)).ravel()
        rhs = (np.pi**2 * (p.kx + 4 * p.ky) + p.reaction) * exact
        plate = ThermalPlate(n, np.column_stack([rhs, np.zeros(n * n)]))
        error = plate.norm(plate.solve(p) - exact)
        order = None if not rows else float(np.log2(rows[-1]["l2_error"] / error))
        rows.append(dict(n=n, dofs=n * n, h=plate.h, l2_error=error, order=order))
    return rows


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_study(output: str | Path, n: int = 39, rank: int = 20) -> dict:
    """Write CSVs, PNG, and JSON. Timing is descriptive and never a pass criterion."""
    output = Path(output)
    plate = ThermalPlate(n)
    training = training_parameters()
    started = perf_counter()
    fit = fit_pod(plate, training, rank)
    offline_seconds = perf_counter() - started
    ranks = sorted({r for r in [2, 4, 8, 12, rank] if r <= rank})
    cases = evaluation_parameters()
    references = [plate.solve(p) for _, p in cases]
    rows = []
    direct_discrepancies = []
    for r in ranks:
        model = fit.model if r == rank else ReducedModel(plate, fit.model.basis[:, :r])
        for i, ((group, p), reference) in enumerate(zip(cases, references, strict=True)):
            result = model.predict(p)
            field = model.reconstruct(result)
            error = reference - field
            l2_error = plate.norm(error)
            direct = plate.norm(plate.load(p) - plate.operator(p) @ field)
            direct_discrepancies.append(abs(direct - result.residual_norm))
            rows.append(
                dict(
                    case=i,
                    group=group,
                    rank=r,
                    **asdict(p),
                    l2_error=l2_error,
                    relative_l2_error=l2_error / plate.norm(reference),
                    l2_bound=result.l2_bound,
                    energy_error=plate.energy_norm(error, p),
                    energy_bound=result.energy_bound,
                    mean_error=abs(plate.mean(error)),
                    mean_bound=result.mean_bound,
                    residual_norm=result.residual_norm,
                    roundoff_scale=result.roundoff_scale,
                )
            )
    convergence = manufactured_convergence()
    selected = [row for row in rows if row["rank"] == rank]
    held_out = [row for row in selected if row["group"] == "held_out"]
    shifted = [row for row in selected if row["group"] == "shifted"]
    checks = {
        "manufactured_second_order": all(1.99 < row["order"] < 2.05 for row in convergence[1:]),
        "l2_bounds_cover": all(row["l2_error"] <= row["l2_bound"] + 1e-11 for row in rows),
        "energy_bounds_cover": all(
            row["energy_error"] <= row["energy_bound"] + 1e-11 for row in rows
        ),
        "mean_bounds_cover": all(
            row["mean_error"] <= row["mean_bound"] + 1e-11 for row in rows
        ),
        "qr_matches_direct_residual": max(direct_discrepancies) < 1e-9,
        "nested_energy_error_decreases": all(
            rows[j * len(cases) + i]["energy_error"]
            <= rows[(j - 1) * len(cases) + i]["energy_error"] + 1e-11
            for j in range(1, len(ranks))
            for i in range(len(cases))
        ),
    }
    # Warm both paths; then time equal parameter sets without CSV, error evaluation,
    # field reconstruction, or plotting. The full solve includes fresh assembly/LU.
    plate.solve(cases[0][1])
    fit.model.predict(cases[0][1])
    full_times, online_times = [], []
    for _ in range(5):
        t = perf_counter()
        for _, p in cases:
            plate.solve(p)
        full_times.append((perf_counter() - t) / len(cases))
        t = perf_counter()
        for _, p in cases:
            fit.model.predict(p)
        online_times.append((perf_counter() - t) / len(cases))
    summary = dict(
        n=n,
        full_dofs=plate.size,
        reduced_dofs=rank,
        training_samples=len(training),
        held_out_samples=len(held_out),
        shifted_samples=len(shifted),
        numerical_snapshot_rank=fit.numerical_rank,
        discarded_snapshot_energy_fraction=fit.discarded_energy_fraction,
        max_held_out_relative_l2_error=max(row["relative_l2_error"] for row in held_out),
        max_shifted_relative_l2_error=max(row["relative_l2_error"] for row in shifted),
        min_l2_bound_to_error=min(row["l2_bound"] / row["l2_error"] for row in rows),
        max_qr_direct_residual_difference=max(direct_discrepancies),
        finest_manufactured_order=convergence[-1]["order"],
        offline_seconds=offline_seconds,
        median_full_seconds=float(np.median(full_times)),
        median_online_seconds=float(np.median(online_times)),
        timing_repeats=5,
        timing_scope="Full: assembly+fresh sparse LU; online: reduced solve+QR bounds+mean. "
        "Excludes reconstruction and offline work; not an optimized full-order baseline.",
        bound_scope="Reduction error versus this grid's full-order solution. Exact-arithmetic "
        "bound; no interval rounding guarantee or continuum-error coverage.",
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            matplotlib=matplotlib.__version__,
        ),
        checks=checks,
    )
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "audit.csv", rows)
    _csv(output / "convergence.csv", convergence)
    _csv(output / "training.csv", [asdict(p) for p in training])
    _csv(
        output / "spectrum.csv",
        [dict(mode=i + 1, singular_value=float(s)) for i, s in enumerate(fit.singular_values)],
    )
    p = cases[-1][1]
    prediction = fit.model.predict(p)
    reference = references[-1]
    field = fit.model.reconstruct(prediction)
    _csv(
        output / "field.csv",
        [
            dict(x=float(x), y=float(y), full=float(u), reduced=float(v))
            for x, y, u, v in zip(
                plate.x.ravel(), plate.y.ravel(), reference, field, strict=True
            )
        ],
    )
    (output / "validation.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _plot(output, plate, fit.singular_values, rows, convergence, reference, field, rank)
    return summary


def _plot(output, plate, singular_values, rows, convergence, reference, field, rank):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
    fig.suptitle(
        "PODGuard | a heat surrogate that checks its own error",
        fontsize=19,
        fontweight="bold",
        color="#12324b",
    )
    common = dict(
        origin="lower",
        extent=[plate.h / 2, 1 - plate.h / 2, plate.h / 2, 1 - plate.h / 2],
        interpolation="nearest",
        aspect="equal",
    )
    for ax, data, title, cmap in [
        (axes[0, 0], reference, "Shifted case 47: full-order temperature", "magma"),
        (axes[0, 1], abs(reference - field), f"Absolute error | {rank} POD modes", "viridis"),
    ]:
        im = ax.imshow(data.reshape(plate.n, plate.n), cmap=cmap, **common)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set(xlabel="x", ylabel="y", title=title)
    axes[0, 2].semilogy(
        np.arange(1, len(singular_values) + 1),
        singular_values / singular_values[0],
        color="#007f83",
    )
    axes[0, 2].axvline(rank, color="#e07a24", linestyle="--", label="retained modes")
    axes[0, 2].set(
        xlabel="POD mode",
        ylabel="Singular value / largest",
        title="POD spectrum | 54 synthetic snapshots",
    )
    axes[0, 2].legend()
    for group, label, color in [
        ("held_out", "Held-out", "#007f83"),
        ("shifted", "Shifted", "#ca5d22"),
    ]:
        ranks = sorted({row["rank"] for row in rows})
        worst = [
            max(
                row["relative_l2_error"]
                for row in rows
                if row["group"] == group and row["rank"] == r
            )
            for r in ranks
        ]
        axes[1, 0].semilogy(ranks, worst, "o-", color=color, label=label)
        subset = [row for row in rows if row["group"] == group and row["rank"] == rank]
        axes[1, 1].loglog(
            [row["l2_error"] for row in subset],
            [row["l2_bound"] for row in subset],
            "o",
            color=color,
            label=label,
        )
    axes[1, 0].set(
        xlabel="Reduced dimension",
        ylabel="Worst relative discrete L2 error",
        title="Unseen parameters: worst-case error",
    )
    axes[1, 0].legend()
    selected = [row for row in rows if row["rank"] == rank]
    low = min(row["l2_error"] for row in selected) * 0.6
    high = max(row["l2_bound"] for row in selected) * 1.6
    axes[1, 1].loglog([low, high], [low, high], "--", color="0.5", label="bound = error")
    axes[1, 1].set(
        xlabel="Measured reduction error",
        ylabel="Residual / coercivity",
        title="Residual bounds cover measured errors",
    )
    axes[1, 1].legend()
    h = np.array([row["h"] for row in convergence])
    e = np.array([row["l2_error"] for row in convergence])
    axes[1, 2].loglog(h, e, "o-", color="#12324b", label="Manufactured sine field")
    axes[1, 2].loglog(h, e[-1] * (h / h[-1]) ** 2, "--", color="#ca5d22", label="Second order")
    axes[1, 2].set(
        xlabel="Grid spacing h",
        ylabel="Discrete L2 error vs exact PDE field",
        title="Mesh error is validated separately",
    )
    axes[1, 2].legend()
    for ax in axes[1]:
        ax.grid(alpha=0.18, which="both")
    fig.savefig(output / "podguard_audit.png", dpi=160)
    plt.close(fig)
