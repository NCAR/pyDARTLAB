import numpy as np
import pytest

from pydartlab import inflate_bnrh, inflate_ensemble, inflate_gamma, update_inflate

RNG = np.random.default_rng(7)


def test_inflate_ensemble_scales_variance_preserves_mean():
    ens = RNG.normal(3.0, 1.5, 20)
    inflated = inflate_ensemble(ens, 2.5)
    assert inflated.mean() == pytest.approx(ens.mean())
    assert inflated.var(ddof=1) == pytest.approx(2.5 * ens.var(ddof=1))


def test_update_inflate_grows_with_large_innovation():
    # Innovation much larger than expected separation -> inflation increases
    lambda_mean, lambda_sd = 1.0, 0.6
    new_mean, _ = update_inflate(
        x_p=0.0, r_var=1.0, y_o=6.0, sigma_o_2=1.0, ss_inflate_base=lambda_mean,
        lambda_mean=lambda_mean, lambda_sd=lambda_sd,
        inf_lower_bound=1.0, inf_upper_bound=100.0, gamma_corr=1.0,
        sd_lower_bound=0.1, ens_size=20, flavor="Gaussian")
    assert new_mean > lambda_mean


def test_update_inflate_shrinks_with_small_innovation():
    # Innovation much smaller than expected separation -> inflation decreases,
    # but never below the lower bound
    new_mean, _ = update_inflate(
        x_p=0.0, r_var=4.0, y_o=0.01, sigma_o_2=1.0, ss_inflate_base=1.5,
        lambda_mean=1.5, lambda_sd=0.6,
        inf_lower_bound=1.0, inf_upper_bound=100.0, gamma_corr=1.0,
        sd_lower_bound=0.1, ens_size=20, flavor="Gaussian")
    assert 1.0 <= new_mean < 1.5


@pytest.mark.parametrize("flavor", ["Gaussian", "I-Gamma"])
def test_update_inflate_sd_never_grows_much(flavor):
    lambda_sd = 0.6
    for y_o in [0.5, 2.0, 4.0, 8.0]:
        _, new_sd = update_inflate(
            x_p=0.0, r_var=1.0, y_o=y_o, sigma_o_2=1.0, ss_inflate_base=1.2,
            lambda_mean=1.2, lambda_sd=lambda_sd,
            inf_lower_bound=1.0, inf_upper_bound=100.0, gamma_corr=1.0,
            sd_lower_bound=0.1, ens_size=20, flavor=flavor)
        assert new_sd <= 1.05 * lambda_sd + 1e-12


def test_update_inflate_respects_sd_lower_bound():
    _, new_sd = update_inflate(
        x_p=0.0, r_var=1.0, y_o=3.0, sigma_o_2=1.0, ss_inflate_base=1.2,
        lambda_mean=1.2, lambda_sd=0.05,
        inf_lower_bound=1.0, inf_upper_bound=100.0, gamma_corr=1.0,
        sd_lower_bound=0.05, ens_size=20, flavor="Gaussian")
    assert new_sd == pytest.approx(0.05)


def test_update_inflate_bad_flavor_raises():
    with pytest.raises(ValueError):
        update_inflate(0, 1, 1, 1, 1, 1, 0.6, 1, 100, 1, 0.1, 20, flavor="bogus")


def test_inflate_gamma_preserves_bound_and_grows_spread():
    ens = RNG.gamma(2.0, 1.0, 20) + 1e-3
    inflated = inflate_gamma(ens, 3.0)
    assert np.all(inflated >= 0.0)
    assert inflated.std(ddof=1) > ens.std(ddof=1)


def test_inflate_bnrh_preserves_bound_and_grows_spread():
    ens = RNG.gamma(2.0, 1.0, 20) + 1e-3
    inflated = inflate_bnrh(ens, 3.0, bounded_below=True, lower_bound=0.0)
    assert np.all(inflated >= 0.0)
    assert inflated.std(ddof=1) > ens.std(ddof=1)


def test_inflate_bnrh_unbounded_close_to_linear():
    # With no bounds and a near-Gaussian ensemble, BNRH-space inflation should
    # roughly match linear inflation
    ens = RNG.normal(0.0, 1.0, 40)
    a = inflate_bnrh(ens, 2.0)
    b = inflate_ensemble(ens, 2.0)
    assert np.corrcoef(a, b)[0, 1] > 0.98
