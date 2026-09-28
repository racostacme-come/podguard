import numpy as np
import pytest
from numpy.testing import assert_allclose

from podguard import Parameters, ThermalPlate


@pytest.mark.parametrize("n", [2, 7, 16])
def test_operator_matches_independent_stencil_and_eigenvalue(n):
    plate = ThermalPlate(n)
    p = Parameters(0.4, 2.3, 1.2)
    values = np.random.default_rng(7).normal(size=(n, n))
    pad = np.pad(values, 1)
    stencil = (
        p.kx * (2 * values - pad[1:-1, :-2] - pad[1:-1, 2:])
        + p.ky * (2 * values - pad[:-2, 1:-1] - pad[2:, 1:-1])
    ) / plate.h**2
    stencil += p.reaction * values
    matrix = plate.operator(p)
    assert_allclose(matrix @ values.ravel(), stencil.ravel(), atol=1e-11)
    assert_allclose(matrix.toarray(), matrix.toarray().T)
    eigenvalues = np.linalg.eigvalsh(matrix.toarray())
    assert_allclose(eigenvalues[0], plate.coercivity(p), rtol=1e-12)


def test_sine_manufactured_solution_is_second_order():
    errors = []
    p = Parameters(0.7, 1.6, 2.0)
    for n in (7, 15, 31, 63):
        grid = ThermalPlate(n)
        exact = (np.sin(np.pi * grid.x) * np.sin(2 * np.pi * grid.y)).ravel()
        rhs = (np.pi**2 * (p.kx + 4 * p.ky) + p.reaction) * exact
        plate = ThermalPlate(n, np.column_stack([rhs, np.zeros(n * n)]))
        errors.append(plate.norm(plate.solve(p) - exact))
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders > 1.99) & (orders < 2.05))


def test_source_normalization_linearity_and_discrete_norm():
    plate = ThermalPlate(11)
    assert_allclose(plate.weight * plate.sources.sum(axis=0), 1)
    a = plate.solve(Parameters(source_a=1, source_b=0))
    b = plate.solve(Parameters(source_a=0, source_b=1))
    both = plate.solve(Parameters(source_a=2, source_b=-0.3))
    assert_allclose(both, 2 * a - 0.3 * b, atol=1e-15)
    assert plate.norm(np.ones(plate.size)) == pytest.approx(11 / 12)
    assert plate.mean(np.ones(plate.size)) == pytest.approx((11 / 12) ** 2)
    assert plate.energy_norm(a, Parameters()) > 0
    sources = plate.sources
    sources[:] = 0
    assert plate.sources.sum() > 0


@pytest.mark.parametrize("n", [True, 0, 1, 2.2, -1])
def test_invalid_grid(n):
    with pytest.raises(ValueError):
        ThermalPlate(n)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kx=0),
        dict(ky=-1),
        dict(reaction=-1),
        dict(source_a=np.nan),
        dict(source_b=np.inf),
        dict(kx=1j),
        dict(ky=[1]),
        dict(kx=True),
        dict(kx="bad"),
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Parameters(**kwargs)


@pytest.mark.parametrize(
    "sources", [np.zeros((9, 1)), np.full((9, 2), np.nan), np.ones((9, 2), dtype=complex)]
)
def test_invalid_sources(sources):
    with pytest.raises(ValueError):
        ThermalPlate(3, sources)


def test_invalid_vector():
    with pytest.raises(ValueError):
        ThermalPlate(3).norm([1])


def test_zero_load():
    plate = ThermalPlate(7)
    assert_allclose(plate.solve(Parameters(source_a=0, source_b=0)), 0)
