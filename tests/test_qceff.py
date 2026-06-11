import numpy as np
import pytest

from pydartlab import gamma_ppi_update, obs_increment_eakf, ppi_update
from pydartlab.algorithms import get_state_increments

RNG = np.random.default_rng(21)


def _correlated_pair(n=20, rho=0.8):
    obs = RNG.normal(5.0, 1.0, n)
    state = rho * (obs - 5.0) + np.sqrt(1 - rho**2) * RNG.normal(0, 1, n) + 3.0
    return obs, state


def test_ppi_normal_normal_matches_linear_regression():
    # With Normal distributions the PPI transform is linear, so the QCEFF
    # update must reproduce plain regression of increments
    prior_obs, prior_state = _correlated_pair()
    obs_incs = obs_increment_eakf(prior_obs, 4.0, 0.5)
    post_obs = prior_obs + obs_incs

    result = ppi_update(prior_obs, prior_state, post_obs, "Normal", "Normal")
    state_incs, _ = get_state_increments(prior_state, prior_obs, obs_incs)
    assert np.allclose(result.post_state, prior_state + state_incs, atol=1e-8)


@pytest.mark.parametrize("state_dist", ["Gamma", "BNRH"])
def test_ppi_bounded_state_stays_nonnegative(state_dist):
    for _ in range(10):
        prior_obs = RNG.normal(0.0, 1.0, 20)
        prior_state = RNG.gamma(2.0, 0.5, 20) + 1e-3
        obs_incs = obs_increment_eakf(prior_obs, -2.0, 0.5)
        result = ppi_update(prior_obs, prior_state, prior_obs + obs_incs,
                            state_dist, "Normal")
        assert np.all(result.post_state >= 0.0)


def test_ppi_rhf_obs_dist_runs():
    prior_obs, prior_state = _correlated_pair()
    obs_incs = obs_increment_eakf(prior_obs, 4.0, 0.5)
    result = ppi_update(prior_obs, prior_state, prior_obs + obs_incs, "RHF", "RHF")
    assert result.post_state.shape == prior_state.shape
    assert np.all(np.isfinite(result.post_state))


def test_ppi_uncorrelated_gives_small_update():
    prior_obs = RNG.normal(0.0, 1.0, 1000)
    prior_state = RNG.normal(0.0, 1.0, 1000)  # independent
    obs_incs = obs_increment_eakf(prior_obs, 1.0, 0.5)
    result = ppi_update(prior_obs, prior_state, prior_obs + obs_incs, "Normal", "Normal")
    assert np.abs(result.post_state - prior_state).max() < 0.2


def test_ppi_invalid_dist_raises():
    prior_obs, prior_state = _correlated_pair()
    with pytest.raises(ValueError):
        ppi_update(prior_obs, prior_state, prior_obs, "bogus", "Normal")
    with pytest.raises(ValueError):
        ppi_update(prior_obs, prior_state, prior_obs, "Normal", "bogus")


def test_gamma_ppi_update_matches_ppi_gamma():
    # gamma_ppi_update is the special case of ppi_update with a Gamma state
    # and Normal obs; posteriors must agree
    prior_obs = RNG.normal(5.0, 1.0, 20)
    prior_state = RNG.gamma(2.0, 1.0, 20)
    obs_incs = obs_increment_eakf(prior_obs, 4.0, 0.5)
    post_obs = prior_obs + obs_incs

    a = gamma_ppi_update(prior_obs, prior_state, post_obs)
    b = ppi_update(prior_obs, prior_state, post_obs, "Gamma", "Normal")
    assert np.allclose(a.post_state, b.post_state, atol=1e-8)
