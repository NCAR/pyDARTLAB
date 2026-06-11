import numpy as np
import pytest

from pydartlab import (
    InvalidVarianceError,
    obs_increment,
    obs_increment_bnrhf,
    obs_increment_eakf,
    obs_increment_enkf,
    obs_increment_gamma,
    obs_increment_rhf,
    product_of_gaussians,
)

RNG = np.random.default_rng(1234)


def test_eakf_posterior_matches_product_of_gaussians():
    ens = RNG.normal(2.0, 1.5, 20)
    obs, obs_var = 1.0, 0.5
    post = ens + obs_increment_eakf(ens, obs, obs_var)

    expected_mean, expected_sd, _ = product_of_gaussians(
        ens.mean(), ens.std(ddof=1), obs, np.sqrt(obs_var))
    assert post.mean() == pytest.approx(expected_mean)
    assert post.std(ddof=1) == pytest.approx(expected_sd)


def test_eakf_preserves_member_order_and_shape():
    ens = RNG.normal(0.0, 1.0, 10)
    post = ens + obs_increment_eakf(ens, 0.5, 1.0)
    # EAKF is a linear shift+contraction: order preserved
    assert np.array_equal(np.argsort(ens), np.argsort(post))


def test_eakf_zero_obs_error_collapses_to_observation():
    ens = RNG.normal(0.0, 1.0, 5)
    post = ens + obs_increment_eakf(ens, 3.0, 0.0)
    assert np.allclose(post, 3.0)


def test_eakf_zero_prior_variance_keeps_mean():
    ens = np.full(5, 2.0)
    post = ens + obs_increment_eakf(ens, 3.0, 1.0)
    assert np.allclose(post, 2.0)


def test_both_variances_zero_raises():
    ens = np.full(5, 2.0)
    with pytest.raises(InvalidVarianceError):
        obs_increment_eakf(ens, 3.0, 0.0)
    with pytest.raises(InvalidVarianceError):
        obs_increment_enkf(ens, 3.0, 0.0)


def test_enkf_mean_matches_eakf_mean():
    # With the mean correction, the EnKF posterior mean is deterministic and
    # equals the EAKF posterior mean
    ens = RNG.normal(2.0, 1.5, 40)
    obs, obs_var = 1.0, 0.5
    post_enkf = ens + obs_increment_enkf(ens, obs, obs_var, rng=np.random.default_rng(7))
    post_eakf = ens + obs_increment_eakf(ens, obs, obs_var)
    assert post_enkf.mean() == pytest.approx(post_eakf.mean())


def test_enkf_reproducible_with_seeded_rng():
    ens = RNG.normal(0.0, 1.0, 10)
    inc1 = obs_increment_enkf(ens, 0.5, 1.0, rng=np.random.default_rng(99))
    inc2 = obs_increment_enkf(ens, 0.5, 1.0, rng=np.random.default_rng(99))
    assert np.array_equal(inc1, inc2)


def test_rhf_reduces_spread_and_moves_toward_obs():
    ens = RNG.normal(2.0, 1.0, 15)
    obs, obs_var = 0.0, 0.5
    result = obs_increment_rhf(ens, obs, obs_var)
    post = ens + result.increments
    assert post.std(ddof=1) < ens.std(ddof=1)
    assert abs(post.mean() - obs) < abs(ens.mean() - obs)


def test_rhf_pdf_curves_returned():
    ens = RNG.normal(0.0, 1.0, 8)
    result = obs_increment_rhf(ens, 0.5, 1.0)
    xp, yp = result.prior_pdf
    assert xp.shape == yp.shape and xp.size > 100
    # Prior pdf integrates to ~1
    assert np.trapezoid(yp, xp) == pytest.approx(1.0, abs=0.05)
    xq, yq = result.post_pdf
    assert np.trapezoid(yq, xq) == pytest.approx(1.0, abs=0.05)


def test_rhf_quantile_conservation_when_likelihood_flat():
    # A very uninformative observation should leave members nearly unchanged
    ens = np.sort(RNG.normal(0.0, 1.0, 10))
    result = obs_increment_rhf(ens, 0.0, 1e8)
    assert np.allclose(result.increments, 0.0, atol=1e-3)


def test_bnrhf_respects_lower_bound():
    for _ in range(20):
        ens = np.abs(RNG.normal(0.3, 0.5, 10)) + 1e-3
        result = obs_increment_bnrhf(ens, 0.05, 0.01)
        post = ens + result.increments
        assert np.all(post >= 0.0)


def test_gamma_filter_respects_lower_bound_and_reduces_spread():
    for _ in range(10):
        ens = RNG.gamma(3.0, 1.0, 20) + 1e-3
        result = obs_increment_gamma(ens, 1.0, 0.5)
        post = ens + result.increments
        assert np.all(post >= 0.0)
        assert post.std(ddof=1) < ens.std(ddof=1)


def test_gamma_filter_preserves_member_order():
    ens = RNG.gamma(3.0, 1.0, 12) + 1e-3
    result = obs_increment_gamma(ens, 1.0, 0.5)
    post = ens + result.increments
    assert np.array_equal(np.argsort(ens), np.argsort(post))


def test_obs_increment_dispatch():
    ens = RNG.normal(0.0, 1.0, 10)
    assert np.array_equal(obs_increment(ens, 0.5, 1.0, "EAKF"),
                          obs_increment_eakf(ens, 0.5, 1.0))
    rhf = obs_increment(ens, 0.5, 1.0, "rhf")
    assert rhf.shape == ens.shape
    enkf = obs_increment(ens, 0.5, 1.0, "EnKF", rng=np.random.default_rng(3))
    assert enkf.shape == ens.shape
    with pytest.raises(ValueError):
        obs_increment(ens, 0.5, 1.0, "bogus")
