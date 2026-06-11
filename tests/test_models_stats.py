import numpy as np
import pytest

from pydartlab import Lorenz63, Lorenz96, advance_oned, get_ens_rank, kurt, product_of_gaussians
from pydartlab.algorithms import comp_cov_factor, cyclic_distance, get_state_increments
from pydartlab.models import LinearGrowth
from pydartlab.stats import RankHistogram, error_and_spread

RNG = np.random.default_rng(11)


def test_lorenz96_fixed_point():
    # x_j = F for all j is a fixed point of the dynamics
    model = Lorenz96(forcing=8.0)
    x = np.full(40, 8.0)
    assert np.allclose(model.comp_dt(x), 0.0)
    assert np.allclose(model.step(x), x)


def test_lorenz96_chaotic_growth():
    model = Lorenz96()
    x = np.full(40, 8.0)
    x[19] += 0.001  # tiny perturbation
    y = x.copy()
    for _ in range(500):
        y = model.step(y)
    # Perturbation has grown and spread; trajectory stays bounded
    assert np.abs(y - 8.0).max() > 1.0
    assert np.abs(y).max() < 50.0


def test_lorenz63_step_matches_hand_rk2():
    model = Lorenz63()
    x = np.array([1.0, 2.0, 3.0])
    dt = model.delta_t
    x1 = x + dt * model.comp_dt(x)
    x2 = x1 + dt * model.comp_dt(x1)
    assert np.allclose(model.step(x), (x + x2) / 2)


def test_lorenz63_attractor_bounded():
    model = Lorenz63()
    x = np.array([1.0, 1.0, 1.0])
    for _ in range(5000):
        x = model.step(x)
    assert np.all(np.abs(x) < 100.0)


def test_advance_oned():
    # x_new = 2x + bias + alpha * x * |x|
    assert advance_oned(1.0) == pytest.approx(2.0)
    assert advance_oned(1.0, model_bias=0.5) == pytest.approx(2.5)
    assert advance_oned(-2.0, alpha=0.1) == pytest.approx(-4.0 - 0.4)
    assert np.allclose(advance_oned(np.array([1.0, -1.0])), [2.0, -2.0])


def test_linear_growth():
    model = LinearGrowth(growth_rate=1.5)
    assert model.step(2.0) == pytest.approx(3.0)


def test_product_of_gaussians_equal_uncertainty():
    mean, sd, weight = product_of_gaussians(0.0, 1.0, 2.0, 1.0)
    assert mean == pytest.approx(1.0)
    assert sd == pytest.approx(np.sqrt(0.5))
    assert weight > 0


def test_kurt_gaussian_near_three():
    vals = RNG.normal(0, 1, 200_000)
    assert kurt(vals) == pytest.approx(3.0, abs=0.1)


def test_get_ens_rank():
    ens = np.array([1.0, 3.0, 2.0])
    assert get_ens_rank(ens, 0.0) == 1
    assert get_ens_rank(ens, 1.5) == 2
    assert get_ens_rank(ens, 2.5) == 3
    assert get_ens_rank(ens, 10.0) == 4


def test_rank_histogram_uniform_for_consistent_ensemble():
    ens_size = 9
    hist = RankHistogram(ens_size)
    n = 20_000
    for _ in range(n):
        ens = RNG.normal(0, 1, ens_size)
        hist.add(ens, RNG.normal(0, 1))
    expected = n / (ens_size + 1)
    # Chi-squared-ish check: all bins within 10% of uniform
    assert np.all(np.abs(hist.counts - expected) < 0.1 * expected)


def test_error_and_spread_multivariate():
    truth = np.zeros(40)
    ens = RNG.normal(0, 1, (20, 40))
    error, spread = error_and_spread(ens, truth)
    assert 0 < error < 1.0
    assert spread == pytest.approx(1.0, abs=0.15)


def test_comp_cov_factor_shape():
    c = 0.2
    assert comp_cov_factor(0.0, c) == pytest.approx(1.0)
    assert comp_cov_factor(2 * c, c) == pytest.approx(0.0)
    assert comp_cov_factor(5.0, c) == 0.0
    dists = np.linspace(0, 2 * c, 50)
    factors = np.asarray(comp_cov_factor(dists, c))
    assert np.all(np.diff(factors) <= 1e-12)  # monotone decreasing
    assert np.all((factors >= 0) & (factors <= 1))


def test_comp_cov_factor_half_width_value():
    # At distance c the Gaspari-Cohn value is 5/24... compute from the polynomial
    c = 1.0
    r = 1.0
    expected = (((-0.25 * r + 0.5) * r + 0.625) * r - 5.0 / 3.0) * r**2 + 1.0
    assert comp_cov_factor(c, c) == pytest.approx(expected)


def test_cyclic_distance_wraps():
    assert cyclic_distance(0, 39, 40) == pytest.approx(1 / 40)
    assert cyclic_distance(0, 20, 40) == pytest.approx(0.5)
    assert cyclic_distance(5, 5, 40) == 0.0


def test_get_state_increments_regression():
    obs_ens = RNG.normal(0, 1, 30)
    state_ens = 2.0 * obs_ens + RNG.normal(0, 0.01, 30)
    obs_incs = RNG.normal(0, 0.5, 30)
    state_incs, rxy = get_state_increments(state_ens, obs_ens, obs_incs)
    # Nearly perfectly correlated with slope 2 -> state increments ~ 2x obs increments
    assert np.allclose(state_incs, 2.0 * obs_incs, atol=0.05)
    assert rxy > 0
