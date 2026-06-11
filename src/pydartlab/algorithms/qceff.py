"""Quantile-conserving ensemble filter framework (QCEFF) updates.

Ports of ``ppi_update.m`` and ``gamma_ppi_update.m`` from
``DART_LAB/matlab/private``.

The probit probability integral (PPI) transform maps ensemble members to a
standard normal space via the quantiles of a chosen continuous prior
distribution. Linear regression of observation increments is performed in
the transformed space, then the result is mapped back. This respects bounds
and handles non-Gaussian priors and nonlinear relationships.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from pydartlab.algorithms.distributions import bnrh_fit, fit_gamma

#: Distribution choices for the PPI transform, matching the MATLAB tools.
OBS_DIST_TYPES = ("Normal", "RHF")
STATE_DIST_TYPES = ("Normal", "Gamma", "RHF", "BNRH")


@dataclass
class PPIResult:
    """Result of a PPI/QCEFF update of an unobserved state variable.

    The ``*_ppi`` arrays are the ensembles in the probit-transformed space,
    used by ``twod_ppi_ensemble`` to display the transformed update.
    """

    post_state: NDArray[np.float64]
    prior_obs_ppi: NDArray[np.float64]
    post_obs_ppi: NDArray[np.float64]
    prior_state_ppi: NDArray[np.float64]
    post_state_ppi: NDArray[np.float64]


def ppi_update(prior_obs: ArrayLike, prior_state: ArrayLike, post_obs: ArrayLike,
               state_dist_type: str = "Normal",
               obs_dist_type: str = "Normal") -> PPIResult:
    """Update an unobserved variable using probit-transformed regression.

    Parameters
    ----------
    prior_obs : array_like
        Prior ensemble of the observed variable.
    prior_state : array_like
        Prior ensemble of the unobserved state variable.
    post_obs : array_like
        Posterior ensemble of the observed variable (prior plus increments
        from any observation-space filter).
    state_dist_type : str
        Continuous distribution for the state PPI transform: ``"Normal"``,
        ``"Gamma"``, ``"RHF"`` (unbounded) or ``"BNRH"`` (bounded below at 0).
    obs_dist_type : str
        Continuous distribution for the observed-variable transform:
        ``"Normal"`` or ``"RHF"``.

    Returns
    -------
    PPIResult
        Posterior state ensemble plus all four PPI-space ensembles.
    """
    prior_obs = np.asarray(prior_obs, dtype=float)
    prior_state = np.asarray(prior_state, dtype=float)
    post_obs = np.asarray(post_obs, dtype=float)
    ens_size = prior_state.size

    # Transform the observed prior and posterior to PPI space. Both use the
    # prior statistics, as per the QCEFF algorithm.
    if obs_dist_type == "Normal":
        prior_obs_mean = prior_obs.mean()
        prior_obs_sd = prior_obs.std(ddof=1)
        prior_obs_q = norm.cdf(prior_obs, prior_obs_mean, prior_obs_sd)
        post_obs_q = norm.cdf(post_obs, prior_obs_mean, prior_obs_sd)
    elif obs_dist_type == "RHF":
        obs_dist = bnrh_fit(prior_obs)
        prior_obs_q = obs_dist.quantiles
        post_obs_q = np.array([obs_dist.cdf(v) for v in post_obs])
    else:
        raise ValueError(f"Unknown obs_dist_type {obs_dist_type!r}; expected {OBS_DIST_TYPES}")

    prior_obs_ppi = norm.ppf(prior_obs_q)
    post_obs_ppi = norm.ppf(post_obs_q)

    # Observation increments in the transformed space
    obs_increments = post_obs_ppi - prior_obs_ppi

    # Transform the state prior to PPI space with the chosen distribution
    state_dist = None
    pmean = psd = pshape = pscale = None
    if state_dist_type == "Normal":
        pmean = prior_state.mean()
        psd = prior_state.std(ddof=1)
        prior_state_q = norm.cdf(prior_state, pmean, psd)
    elif state_dist_type == "Gamma":
        pshape, pscale = fit_gamma(prior_state)
        prior_state_q = gamma_dist.cdf(prior_state, pshape, scale=pscale)
    elif state_dist_type == "RHF":
        state_dist = bnrh_fit(prior_state)
        prior_state_q = state_dist.quantiles
    elif state_dist_type == "BNRH":
        state_dist = bnrh_fit(prior_state, bounded_below=True, lower_bound=0.0)
        prior_state_q = state_dist.quantiles
    else:
        raise ValueError(
            f"Unknown state_dist_type {state_dist_type!r}; expected {STATE_DIST_TYPES}")

    prior_state_ppi = norm.ppf(prior_state_q)

    # Linear regression of the increments in the transformed space
    covar = np.cov(prior_obs_ppi, prior_state_ppi, ddof=1)
    state_inc = obs_increments * covar[0, 1] / covar[0, 0]
    post_state_ppi = prior_state_ppi + state_inc

    # Back to quantile space, then to physical space
    post_state_q = norm.cdf(post_state_ppi)

    if state_dist_type == "Normal":
        post_state = norm.ppf(post_state_q, pmean, psd)
    elif state_dist_type == "Gamma":
        post_state = gamma_dist.ppf(post_state_q, pshape, scale=pscale)
    else:  # RHF or BNRH
        post_state = state_dist.ppf(post_state_q)

    assert ens_size == post_state.size
    return PPIResult(
        post_state=post_state,
        prior_obs_ppi=prior_obs_ppi, post_obs_ppi=post_obs_ppi,
        prior_state_ppi=prior_state_ppi, post_state_ppi=post_state_ppi,
    )


def gamma_ppi_update(prior_obs: ArrayLike, prior_state: ArrayLike,
                     post_obs: ArrayLike) -> PPIResult:
    """Gamma-transform update of a non-negative unobserved variable.

    Port of ``gamma_ppi_update.m``. The observed variable is normally
    distributed (its transform is just a normalization); the state variable
    is transformed through a fitted gamma distribution and probit.
    """
    prior_obs = np.asarray(prior_obs, dtype=float)
    prior_state = np.asarray(prior_state, dtype=float)
    post_obs = np.asarray(post_obs, dtype=float)

    # The observed transform is a normalization, keeping the prior mean
    prior_obs_mean = prior_obs.mean()
    prior_obs_sd = prior_obs.std(ddof=1)
    prior_obs_ppi = (prior_obs - prior_obs_mean) / prior_obs_sd + prior_obs_mean
    post_obs_ppi = (post_obs - prior_obs_mean) / prior_obs_sd + prior_obs_mean

    obs_increments = post_obs_ppi - prior_obs_ppi

    pshape, pscale = fit_gamma(prior_state)
    prior_state_q = gamma_dist.cdf(prior_state, pshape, scale=pscale)
    prior_state_ppi = norm.ppf(prior_state_q)

    covar = np.cov(prior_obs_ppi, prior_state_ppi, ddof=1)
    state_inc = obs_increments * covar[0, 1] / covar[0, 0]
    post_state_ppi = prior_state_ppi + state_inc

    post_state_q = norm.cdf(post_state_ppi)
    post_state = gamma_dist.ppf(post_state_q, pshape, scale=pscale)

    return PPIResult(
        post_state=post_state,
        prior_obs_ppi=prior_obs_ppi, post_obs_ppi=post_obs_ppi,
        prior_state_ppi=prior_state_ppi, post_state_ppi=post_state_ppi,
    )
