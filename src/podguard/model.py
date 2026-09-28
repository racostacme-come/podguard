"""Five-point anisotropic diffusion on a unit square, with zero boundary values."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve

FloatArray = NDArray[np.float64]


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class Parameters:
    """Dimensionless positive conductivities, nonnegative loss, and two source weights."""

    kx: float = 1.0
    ky: float = 1.0
    reaction: float = 0.0
    source_a: float = 1.0
    source_b: float = 0.0

    def __post_init__(self) -> None:
        for name in ("kx", "ky", "reaction", "source_a", "source_b"):
            value = getattr(self, name)
            if isinstance(value, (bool, complex)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite real scalar")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite real scalar") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.kx <= 0 or self.ky <= 0 or self.reaction < 0:
            raise ValueError("kx and ky must be positive; reaction must be nonnegative")

    @property
    def weights(self) -> FloatArray:
        return np.array([self.source_a, self.source_b])


class ThermalPlate:
    """Uniform interior grid. Arrays flatten in C order (x varies fastest).

    ``sources`` optionally supplies two nodal load columns of shape (n*n, 2).
    The default synthetic Gaussian loads each have unit discrete integral.
    """

    def __init__(self, n: int = 39, sources: ArrayLike | None = None) -> None:
        self.n = _integer(n, "n", 2)
        self.size = self.n**2
        self.h = 1.0 / (self.n + 1)
        self.weight = self.h**2
        axis = np.arange(1, self.n + 1) * self.h
        self.x, self.y = np.meshgrid(axis, axis)
        t = (
            diags(
                [-np.ones(self.n - 1), 2 * np.ones(self.n), -np.ones(self.n - 1)],
                [-1, 0, 1],
                format="csr",
            )
            / self.weight
        )
        identity = eye(self.n, format="csr")
        self._ax = kron(identity, t, format="csr")
        self._ay = kron(t, identity, format="csr")
        self._identity = eye(self.size, format="csr")
        self.lambda_1d = 4 * np.sin(np.pi * self.h / 2) ** 2 / self.weight
        if sources is None:
            a = np.exp(-((self.x - 0.3) ** 2 + (self.y - 0.4) ** 2) / (2 * 0.085**2))
            b = np.exp(-((self.x - 0.72) ** 2 + (self.y - 0.65) ** 2) / (2 * 0.12**2))
            sources = np.column_stack([a.ravel(), b.ravel()])
            sources /= self.weight * sources.sum(axis=0)
        self._sources = _array(sources, (self.size, 2), "sources")

    @property
    def sources(self) -> FloatArray:
        """Return a copy of the two nodal source columns."""
        return self._sources.copy()

    def operator(self, parameters: Parameters) -> csr_matrix:
        """Assemble the symmetric positive definite full-order operator."""
        p = parameters
        matrix = p.kx * self._ax + p.ky * self._ay + p.reaction * self._identity
        if not np.isfinite(matrix.data).all():
            raise ValueError("operator overflow: rescale parameters")
        return matrix

    def load(self, parameters: Parameters) -> FloatArray:
        return _array(self._sources @ parameters.weights, (self.size,), "load")

    def coercivity(self, parameters: Parameters) -> float:
        """Exact smallest eigenvalue in the discrete L2 inner product."""
        value = (parameters.kx + parameters.ky) * self.lambda_1d + parameters.reaction
        if not np.isfinite(value) or value <= 0:
            raise ValueError("coercivity underflow or overflow: rescale parameters")
        return float(value)

    def solve(self, parameters: Parameters) -> FloatArray:
        """Solve a fresh sparse system; no factorization is cached."""
        self.coercivity(parameters)
        return _array(
            spsolve(self.operator(parameters), self.load(parameters)), (self.size,), "solution"
        )

    def norm(self, vector: ArrayLike) -> float:
        """Discrete L2 norm sqrt(h^2 sum(u_i^2))."""
        return float(self.h * np.linalg.norm(_array(vector, (self.size,), "vector")))

    def mean(self, vector: ArrayLike) -> float:
        """Interior quadrature for the integral over the unit-area square."""
        return float(self.weight * _array(vector, (self.size,), "vector").sum())

    def energy_norm(self, vector: ArrayLike, parameters: Parameters) -> float:
        v = _array(vector, (self.size,), "vector")
        return float(np.sqrt(max(0.0, self.weight * v @ (self.operator(parameters) @ v))))
