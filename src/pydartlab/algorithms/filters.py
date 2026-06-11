"""Bounded-variable observation-space filters.

The gamma filter and the bounded normal rank histogram filter (BNRHF) used
by ``bounded_oned_ensemble.m``. Unlike the EAKF these respect a lower bound
at zero, so non-negative quantities stay non-negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gamma as gamma_dist

from pydartlab.algorithms.distributions import fit_gamma
from pydartlab.algorithms.increments import RHFResult, obs_increment_rhf


@dataclass
class GammaFilterResult:
    """Result of a gamma-filter update.

    The prior and posterior gamma parameters are kept for plotting the
    continuous distributions.
    """

    increments: NDArray[np.float64]
    prior_shape: float
    prior_scale: float
    post_shape: float
    post_scale: float


def obs_increment_gamma(ensemble: ArrayLike, observation: float,
                        obs_error_sd: float) -> GammaFilterResult:
    """Increments for a gamma filter for non-negative variables.

    The prior ensemble is fit with a gamma distribution and the likelihood
    is a gamma with mean ``observation`` and sd ``obs_error_sd``. The
    product of the two gammas has known shape and scale; new members are
    placed by quantile conservation (members keep their prior quantiles in
    the posterior distribution).
    """
    ensemble = np.asarray(ensemble, dtype=float)

    # Shape and scale of the likelihood from its mean and sd
    lgamma_var = obs_error_sd**2
    lgamma_shape = observation**2 / lgamma_var
    lgamma_scale = lgamma_var / observation

    # Shape and scale of the prior
    pgamma_shape, pgamma_scale = fit_gamma(ensemble)

    # Quantiles of the prior ensemble members
    prior_q = gamma_dist.cdf(ensemble, pgamma_shape, scale=pgamma_scale)

    # Product of two gammas is a gamma
    agamma_shape = pgamma_shape + lgamma_shape - 1
    agamma_scale = pgamma_scale * lgamma_scale / (pgamma_scale + lgamma_scale)

    # Posterior ensemble conserves the prior quantiles
    new_ensemble = gamma_dist.ppf(prior_q, agamma_shape, scale=agamma_scale)

    return GammaFilterResult(
        increments=new_ensemble - ensemble,
        prior_shape=pgamma_shape, prior_scale=pgamma_scale,
        post_shape=agamma_shape, post_scale=agamma_scale,
    )


def obs_increment_bnrhf(ensemble: ArrayLike, observation: float,
                        obs_error_var: float) -> RHFResult:
    """Increments for a bounded (at zero) normal rank histogram filter."""
    return obs_increment_rhf(ensemble, observation, obs_error_var, bounded_left=True)
