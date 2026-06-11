import numpy as np
import pytest

from pydartlab.algorithms.distributions import (
    bnrh_fit,
    ens_quantiles,
    fit_gamma,
    inv_bnrh_cdf,
    weighted_norm_inv,
)

RNG = np.random.default_rng(42)


def test_ens_quantiles_distinct_members_are_uniform():
    ens = np.sort(RNG.normal(0, 1, 9))
    q = ens_quantiles(ens)
    assert np.allclose(q, np.arange(1, 10) / 10.0)


def test_ens_quantiles_lower_bound_duplicates():
    ens = np.array([0.0, 0.0, 1.0, 2.0])
    q = ens_quantiles(ens, bounded_below=True, lower_bound=0.0)
    # Two members at the bound share quantile 2/(2*(n+1))
    assert q[0] == q[1] == pytest.approx(2 / (2 * 5.0))
    assert q[2] == pytest.approx(3 / 5.0)
    assert q[3] == pytest.approx(4 / 5.0)


def test_ens_quantiles_interior_duplicates_share_mean_quantile():
    ens = np.array([1.0, 2.0, 2.0, 3.0])
    q = ens_quantiles(ens)
    # Run of 2 starting at 1-based index 2: 2/5 + 1/10
    assert q[1] == q[2] == pytest.approx(2 / 5.0 + 1 / 10.0)


def test_bnrh_quantiles_unsorted_order_matches():
    ens = RNG.normal(0, 1, 10)
    dist = bnrh_fit(ens)
    order = np.argsort(ens)
    assert np.allclose(dist.quantiles[order], dist.sorted_quantiles)


def test_bnrh_round_trip_recovers_ensemble():
    # ppf of the fitted quantiles must return the original sorted ensemble
    ens = RNG.normal(5, 2, 15)
    dist = bnrh_fit(ens)
    x = inv_bnrh_cdf(dist.sorted_quantiles, dist)
    assert np.allclose(x, dist.sort_x)


def test_bnrh_cdf_ppf_round_trip_off_members():
    ens = np.sort(RNG.normal(0, 1, 10))
    dist = bnrh_fit(ens)
    for v in [-2.5, -0.3, 0.4, 1.7]:
        q = dist.cdf(v)
        assert dist.ppf([q])[0] == pytest.approx(v, abs=1e-8)


def test_bnrh_bounded_below_round_trip():
    ens = np.abs(RNG.normal(1.0, 0.5, 12)) + 1e-6
    dist = bnrh_fit(ens, bounded_below=True, lower_bound=0.0)
    x = inv_bnrh_cdf(dist.sorted_quantiles, dist)
    assert np.allclose(x, dist.sort_x)
    # Quantiles near 0 map to values above the bound
    small = dist.ppf([1e-4])[0]
    assert 0.0 <= small <= dist.sort_x[0]


def test_bnrh_rejects_out_of_bounds_ensemble():
    with pytest.raises(ValueError):
        bnrh_fit(np.array([-1.0, 0.5, 1.0]), bounded_below=True, lower_bound=0.0)


def test_bnrh_rejects_zero_spread():
    with pytest.raises(ValueError):
        bnrh_fit(np.full(5, 1.0))


def test_weighted_norm_inv_matches_scaled_normal():
    from scipy.stats import norm
    # cdf of 2*N(1, 0.5) at x equals p  <=>  x = weighted_norm_inv(2, 1, 0.5, p)
    x = weighted_norm_inv(2.0, 1.0, 0.5, 0.6)
    assert 2.0 * norm.cdf(x, 1.0, 0.5) == pytest.approx(0.6)


def test_fit_gamma_recovers_parameters():
    ens = RNG.gamma(4.0, 2.0, 40_000)
    shape, scale = fit_gamma(ens)
    assert shape == pytest.approx(4.0, rel=0.05)
    assert scale == pytest.approx(2.0, rel=0.05)
