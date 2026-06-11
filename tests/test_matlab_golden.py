"""Golden-file tests against the MATLAB DART_LAB private functions.

The reference CSVs are produced by ``matlab_reference/generate_reference.m``
(one MATLAB run; the CSVs are then checked in so CI does not need MATLAB).
Tests are skipped for any reference file that has not been generated yet.

The fixed inputs here must match generate_reference.m exactly.
"""

from pathlib import Path

import numpy as np
import pytest

import pydartlab as dl
from pydartlab.algorithms.distributions import bnrh_fit, ens_quantiles
from pydartlab.algorithms.inflation import change_ga_ig

REF = Path(__file__).parent / "matlab_reference"

ENS5 = np.array([0.7, 2.3, -0.4, 1.1, 3.0])
ENS10 = np.array([-1.2, 0.3, 1.9, 0.6, -0.8, 2.4, 1.1, -0.1, 0.9, 1.6])
POS10 = np.array([0.4, 1.3, 2.6, 0.9, 0.2, 3.1, 1.8, 0.5, 1.1, 2.2])
STATE10 = np.array([2.1, 0.8, 3.0, 1.5, 0.9, 3.6, 2.2, 1.0, 1.7, 2.9])
OBSERVATION = 0.5
OBS_ERROR_VAR = 1.0


def load(name):
    path = REF / name
    if not path.exists():
        pytest.skip(f"MATLAB reference {name} not generated yet "
                    "(run tests/matlab_reference/generate_reference.m)")
    return np.loadtxt(path, delimiter=",", ndmin=2)


def test_eakf_increments():
    expected = load("eakf_increments.csv").ravel()
    inc = dl.obs_increment_eakf(ENS10, OBSERVATION, OBS_ERROR_VAR)
    np.testing.assert_allclose(inc, expected, atol=1e-10)


def test_rhf_increments():
    expected = load("rhf_increments.csv").ravel()
    inc = dl.obs_increment_rhf(ENS10, OBSERVATION, OBS_ERROR_VAR).increments
    np.testing.assert_allclose(inc, expected, atol=1e-8)


def test_rhf_bounded_increments():
    expected = load("rhf_bounded_increments.csv").ravel()
    inc = dl.obs_increment_rhf(POS10, OBSERVATION, OBS_ERROR_VAR,
                               bounded_left=True).increments
    np.testing.assert_allclose(inc, expected, atol=1e-8)


def test_product_of_gaussians():
    expected = load("product_of_gaussians.csv").ravel()
    result = dl.product_of_gaussians(1.0, 2.0, 0.5, 1.0)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_comp_cov_factor():
    data = load("comp_cov_factor.csv")
    dists, expected = data[:, 0], data[:, 1]
    np.testing.assert_allclose(dl.comp_cov_factor(dists, 0.2), expected, atol=1e-12)


def test_state_increments():
    expected = load("state_increments.csv").ravel()
    obs_inc = dl.obs_increment_eakf(ENS10, OBSERVATION, OBS_ERROR_VAR)
    state_incs, _ = dl.get_state_increments(STATE10, ENS10, obs_inc)
    np.testing.assert_allclose(state_incs, expected, atol=1e-10)


@pytest.mark.parametrize("flavor,fname", [
    ("Gaussian", "update_inflate_gaussian.csv"),
    ("I-Gamma", "update_inflate_igamma.csv"),
])
def test_update_inflate(flavor, fname):
    data = load(fname)
    for row in data:
        (x_p, r_var, y_o, sigma_o_2, ss_base, lam_mean, lam_sd,
         lower, upper, gamma_corr, sd_lower, ens_size) = row[:12]
        expected_mean, expected_sd = row[12], row[13]
        new_mean, new_sd = dl.update_inflate(
            x_p, r_var, y_o, sigma_o_2, ss_base, lam_mean, lam_sd,
            lower, upper, gamma_corr, sd_lower, int(ens_size), flavor)
        assert new_mean == pytest.approx(expected_mean, abs=1e-8)
        assert new_sd == pytest.approx(expected_sd, abs=1e-8)


def test_change_ga_ig():
    expected = float(load("change_ga_ig.csv").ravel()[0])
    assert change_ga_ig(1.2, 0.36) == pytest.approx(expected, rel=1e-10)


def test_bnrh_quantiles():
    expected = load("bnrh_quantiles.csv").ravel()
    dist = bnrh_fit(POS10, bounded_below=True, lower_bound=0.0)
    np.testing.assert_allclose(dist.quantiles, expected, atol=1e-12)


def test_ens_quantiles_duplicates():
    expected = load("ens_quantiles_dups.csv").ravel()
    dup_ens = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 3.5])
    q = ens_quantiles(dup_ens, bounded_below=True, lower_bound=0.0)
    np.testing.assert_allclose(q, expected, atol=1e-12)


@pytest.mark.parametrize("state_dist,obs_dist,fname,tol", [
    ("Normal", "Normal", "ppi_normal_normal.csv", 1e-10),
    ("Gamma", "Normal", "ppi_gamma_normal.csv", 1e-5),
    # BNRH tail parameters differ from MATLAB (del_q typo there), so interior
    # members agree tightly but tail members only loosely
    ("BNRH", "RHF", "ppi_bnrh_rhf.csv", 5e-2),
])
def test_ppi_update(state_dist, obs_dist, fname, tol):
    expected = load(fname).ravel()
    obs_inc = dl.obs_increment_eakf(ENS10, OBSERVATION, OBS_ERROR_VAR)
    post_obs = ENS10 + obs_inc
    result = dl.ppi_update(ENS10, STATE10, post_obs, state_dist, obs_dist)
    np.testing.assert_allclose(result.post_state, expected, atol=tol)


def test_inflate_gamma():
    expected = load("inflate_gamma.csv").ravel()
    # gamfit/scipy MLE agree to ~1e-6; allow slightly looser tolerance
    np.testing.assert_allclose(dl.inflate_gamma(POS10, 2.0), expected, atol=1e-4)


def test_inflate_bnrh():
    expected = load("inflate_bnrh.csv").ravel()
    np.testing.assert_allclose(
        dl.inflate_bnrh(POS10, 2.0, bounded_below=True, lower_bound=0.0),
        expected, atol=5e-2)


def test_lorenz63_step():
    expected = load("lorenz63_step.csv").ravel()
    model = dl.Lorenz63()
    np.testing.assert_allclose(model.step([1.0, 2.0, 3.0]), expected, atol=1e-12)


def test_lorenz96_step():
    expected = load("lorenz96_step.csv").ravel()
    model = dl.Lorenz96()
    x0 = np.full(40, 8.0)
    x0[19] = 8.01
    np.testing.assert_allclose(model.step(x0), expected, atol=1e-12)


def test_advance_oned():
    expected = load("advance_oned.csv").ravel()
    vals = dl.advance_oned(np.array([0.5, -1.0, 2.0]), 0.1, 0.25)
    np.testing.assert_allclose(vals, expected, atol=1e-12)


def test_kurt():
    expected = float(load("kurt.csv").ravel()[0])
    assert dl.kurt(ENS10) == pytest.approx(expected, abs=1e-12)


def test_ens_rank():
    expected = load("ens_rank.csv").ravel().astype(int)
    ranks = [dl.get_ens_rank(ENS10, 0.0), dl.get_ens_rank(ENS10, 5.0),
             dl.get_ens_rank(ENS10, -5.0)]
    assert list(ranks) == list(expected)
