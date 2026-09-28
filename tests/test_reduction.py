from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose

from podguard import Parameters, ReducedModel, ThermalPlate, fit_pod


@pytest.fixture(scope="module")
def trained():
    plate = ThermalPlate(13)
    training = [
        Parameters(kx, ky, beta, a, 1 - a)
        for kx, ky, beta, a in product([0.5, 2], [0.5, 2], [0, 8], [0, 1])
    ]
    return plate, fit_pod(plate, training, 10)


def test_pod_orthogonality_and_energy(trained):
    plate, fit = trained
    v = fit.model.basis
    assert_allclose(plate.weight * v.T @ v, np.eye(10), atol=1e-14)
    assert np.all(np.diff(fit.singular_values) <= 0)
    assert 0 <= fit.discarded_energy_fraction < 1e-5
    v[:] = 0
    assert np.linalg.norm(fit.model.basis) > 1


@pytest.mark.parametrize("seed", range(12))
def test_held_out_and_extrapolated_bounds(trained, seed):
    plate, fit = trained
    rng = np.random.default_rng(seed)
    p = Parameters(
        *np.exp(rng.uniform(-2.5, 2.5, 2)), rng.uniform(0, 15), *rng.uniform(-1, 2, 2)
    )
    model = fit.model
    prediction = model.predict(p)
    field = model.reconstruct(prediction)
    error = plate.solve(p) - field
    direct = plate.norm(plate.load(p) - plate.operator(p) @ field)
    assert_allclose(prediction.residual_norm, direct, rtol=2e-9, atol=1e-12)
    assert plate.norm(error) <= prediction.l2_bound + 1e-12
    assert plate.energy_norm(error, p) <= prediction.energy_bound + 1e-12
    assert abs(plate.mean(error)) <= prediction.mean_bound + 1e-12
    assert_allclose(prediction.mean, plate.mean(field), atol=1e-15)
    residual = plate.load(p) - plate.operator(p) @ field
    assert_allclose(plate.weight * model.basis.T @ residual, 0, atol=1e-12)


def test_full_basis_exact_and_small_grid_wide_qr():
    plate = ThermalPlate(3)
    model = ReducedModel(plate, np.eye(plate.size) / plate.h)
    p = Parameters(1, 2, 3, 0.4, 0.9)
    result = model.predict(p)
    assert_allclose(model.reconstruct(result), plate.solve(p), atol=1e-15)
    assert result.residual_norm < 1e-12


def test_zero_load_prediction(trained):
    _, fit = trained
    result = fit.model.predict(Parameters(source_a=0, source_b=0))
    assert result.l2_bound == result.mean == result.roundoff_scale == 0


def test_pod_rank_rejection():
    plate = ThermalPlate(4)
    for rank in [0, True, 1.5, 3]:
        with pytest.raises(ValueError):
            fit_pod(plate, [Parameters(), Parameters()], rank)
    with pytest.raises(ValueError, match="numerical snapshot rank"):
        fit_pod(plate, [Parameters(), Parameters()], 2)
    with pytest.raises(ValueError, match="numerical snapshot rank"):
        fit_pod(plate, [Parameters(source_a=0, source_b=0)], 1)
    with pytest.raises(ValueError):
        fit_pod(plate, [], 1)


@pytest.mark.parametrize(
    "basis",
    [np.zeros((9, 0)), np.ones((9, 10)), np.eye(9), np.ones(9), np.full((9, 1), np.inf)],
)
def test_invalid_basis(basis):
    with pytest.raises(ValueError):
        ReducedModel(ThermalPlate(3), basis)


def test_pod_tail_matches_optimal_projection():
    plate = ThermalPlate(9)
    training = [
        Parameters(k, 1, b, a, 1 - a) for k, b, a in product([0.2, 1, 5], [0, 10], [0, 1])
    ]
    fit = fit_pod(plate, training, 3)
    x = plate.h * np.column_stack([plate.solve(p) for p in training])
    q = plate.h * fit.model.basis
    observed = np.linalg.norm(x - q @ (q.T @ x)) ** 2 / np.linalg.norm(x) ** 2
    assert_allclose(observed, fit.discarded_energy_fraction, rtol=1e-12)
