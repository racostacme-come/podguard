"""Snapshot POD, affine Galerkin projection, and stable online residual norms."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve

from .model import FloatArray, Parameters, ThermalPlate, _array, _integer


@dataclass(frozen=True)
class Prediction:
    """Online result. Bounds refer to the discrete full-order solution.

    Bounds are analytical in exact arithmetic; floating-point calculations are
    not interval certified. ``roundoff_scale`` flags the residual noise scale.
    """

    coefficients: FloatArray
    mean: float
    residual_norm: float
    l2_bound: float
    energy_bound: float
    mean_bound: float
    roundoff_scale: float


class ReducedModel:
    """A fixed basis with an online solve independent of full-order dimension.

    Construct through ``fit_pod`` or supply an h-weighted orthonormal basis.
    Reconstructing the field remains O(n^2 r) and is explicitly separate.
    """

    def __init__(self, plate: ThermalPlate, basis: ArrayLike) -> None:
        raw = np.asarray(basis)
        if raw.ndim != 2 or raw.shape[1] < 1 or raw.shape[1] > plate.size:
            raise ValueError("basis must have between 1 and n*n columns")
        self._basis = _array(basis, (plate.size, raw.shape[1]), "basis")
        v = self._basis
        self.rank = v.shape[1]
        if not np.allclose(plate.weight * v.T @ v, np.eye(self.rank), atol=1e-10, rtol=0):
            raise ValueError("basis must be orthonormal in the h^2 weighted inner product")
        axv, ayv = plate._ax @ v, plate._ay @ v
        self._ax = plate.weight * v.T @ axv
        self._ay = plate.weight * v.T @ ayv
        # Keep the computed mass instead of assuming an exactly orthonormal basis.
        self._mass = plate.weight * v.T @ v
        self._loads = plate.weight * v.T @ plate._sources
        self._mean = plate.weight * v.sum(axis=0)
        self._output_norm = plate.h * np.sqrt(plate.size)
        self._lambda = plate.lambda_1d
        # QR avoids forming B.T B, whose subtraction can destroy small residuals.
        blocks = plate.h * np.column_stack([plate._sources, axv, ayv, v])
        _, self._residual_r = np.linalg.qr(blocks, mode="reduced")
        self._r_norm = float(np.linalg.norm(self._residual_r, ord="fro"))

    @property
    def basis(self) -> FloatArray:
        return self._basis.copy()

    def predict(self, parameters: Parameters) -> Prediction:
        """Solve r-by-r equations and evaluate a QR residual error certificate."""
        p = parameters
        alpha = (p.kx + p.ky) * self._lambda + p.reaction
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("coercivity underflow or overflow: rescale parameters")
        matrix = p.kx * self._ax + p.ky * self._ay + p.reaction * self._mass
        rhs = self._loads @ p.weights
        coefficients = _array(solve(matrix, rhs, assume_a="pos"), (self.rank,), "coefficients")
        c = np.concatenate(
            [p.weights, -p.kx * coefficients, -p.ky * coefficients, -p.reaction * coefficients]
        )
        residual = float(np.linalg.norm(self._residual_r @ c))
        # Diagnostic scale, NOT a mathematically verified floating-point enclosure.
        noise = float(32 * np.finfo(float).eps * self._r_norm * np.linalg.norm(c))
        if not np.isfinite(residual) or not np.isfinite(noise):
            raise ValueError("residual overflow: rescale parameters")
        coefficients.setflags(write=False)
        return Prediction(
            coefficients,
            float(self._mean @ coefficients),
            residual,
            residual / alpha,
            residual / np.sqrt(alpha),
            self._output_norm * residual / alpha,
            noise,
        )

    def reconstruct(self, prediction: Prediction) -> FloatArray:
        """Recover the full nodal field (the prediction must come from this model)."""
        c = _array(prediction.coefficients, (self.rank,), "coefficients")
        return self._basis @ c


@dataclass(frozen=True)
class PODResult:
    """Model and singular values of the h-weighted snapshot matrix."""

    model: ReducedModel
    singular_values: FloatArray
    discarded_energy_fraction: float
    numerical_rank: int


def fit_pod(plate: ThermalPlate, training: Sequence[Parameters], rank: int) -> PODResult:
    """Solve independent snapshots, then retain ``rank`` left singular vectors.

    Samples have equal weight and are not centered: zero is a physically meaningful
    reference. The numerical-rank cutoff is max(snapshot_shape)*eps*sigma_max.
    """
    rank = _integer(rank, "rank")
    if len(training) == 0:
        raise ValueError("training must contain at least one parameter sample")
    if rank > min(plate.size, len(training)):
        raise ValueError("rank exceeds snapshot matrix dimensions")
    snapshots = plate.h * np.column_stack([plate.solve(p) for p in training])
    u, s, _ = np.linalg.svd(snapshots, full_matrices=False)
    cutoff = max(snapshots.shape) * np.finfo(float).eps * s[0]
    numerical_rank = int(np.count_nonzero(s > cutoff))
    if rank > numerical_rank:
        raise ValueError(f"rank exceeds numerical snapshot rank ({numerical_rank})")
    model = ReducedModel(plate, u[:, :rank] / plate.h)
    discarded = float(np.sum((s[rank:] / s[0]) ** 2) / np.sum((s / s[0]) ** 2))
    s.setflags(write=False)
    return PODResult(model, s, discarded, numerical_rank)
