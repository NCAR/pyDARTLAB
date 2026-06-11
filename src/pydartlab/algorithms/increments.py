"""Observation-space increments for ensemble filters.

Ports of ``obs_increment_eakf.m``, ``obs_increment_enkf.m`` and
``obs_increment_rhf.m`` from ``DART_LAB/matlab/private``.

Each function takes a 1-D prior ensemble, a scalar observation and the
observation error variance and returns the increments that move the prior
ensemble members to the posterior. State variables are then updated by
regressing these increments (see :mod:`pydartlab.algorithms.regression`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from pydartlab.algorithms.distributions import rhf_pdf_points, weighted_norm_inv


class InvalidVarianceError(Exception):
    """Raised when both the prior and observation error variance are non-positive."""


def obs_increment_eakf(ensemble: ArrayLike, observation: float,
                       obs_error_var: float) -> NDArray[np.float64]:
    """Increments for an ensemble adjustment Kalman filter (EAKF).

    The prior ensemble is shifted to the posterior mean and linearly
    contracted to the posterior variance, preserving its shape.

    Parameters
    ----------
    ensemble : array_like
        Prior ensemble (1-D).
    observation : float
        Observed value.
    obs_error_var : float
        Observation error variance.

    Returns
    -------
    obs_increments : ndarray
        Increments to add to the prior ensemble members.
    """
    ensemble = np.asarray(ensemble, dtype=float)
    prior_mean = ensemble.mean()
    prior_var = ensemble.var(ddof=1)

    if prior_var <= 0 and obs_error_var <= 0:
        raise InvalidVarianceError(
            "Both prior variance and observation error variance are non-positive.")

    if prior_var == 0:
        post_mean, post_var = prior_mean, 0.0
    elif obs_error_var == 0:
        post_mean, post_var = observation, 0.0
    else:
        # Product of Gaussians
        post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_error_var)
        post_mean = post_var * (prior_mean / prior_var + observation / obs_error_var)

    # Shift the prior ensemble to have the posterior mean
    updated_ensemble = ensemble - prior_mean + post_mean

    # Contract the ensemble to have the posterior variance
    if prior_var > 0:
        var_ratio = post_var / prior_var
        updated_ensemble = np.sqrt(var_ratio) * (updated_ensemble - post_mean) + post_mean

    return updated_ensemble - ensemble


def obs_increment_enkf(ensemble: ArrayLike, observation: float, obs_error_var: float,
                       rng: np.random.Generator | None = None) -> NDArray[np.float64]:
    """Increments for a perturbed-observations ensemble Kalman filter (EnKF).

    Each member is paired with an observation perturbed by a draw from
    ``Normal(0, obs_error_var)``; the perturbed observations are recentered
    on the observation (the "mean correction" enhancement). The algorithm is
    stochastic: repeated calls with the same inputs give different results
    unless ``rng`` is seeded.

    Parameters
    ----------
    ensemble : array_like
        Prior ensemble (1-D).
    observation : float
        Observed value.
    obs_error_var : float
        Observation error variance.
    rng : numpy.random.Generator, optional
        Source of randomness; defaults to a fresh default generator.

    Returns
    -------
    obs_increments : ndarray
        Increments to add to the prior ensemble members.
    """
    ensemble = np.asarray(ensemble, dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    prior_var = ensemble.var(ddof=1)

    if prior_var <= 0 and obs_error_var <= 0:
        raise InvalidVarianceError(
            "Both prior variance and observation error variance are non-positive.")

    if prior_var == 0 or obs_error_var == 0:
        # Degenerate cases reduce to the deterministic update
        return obs_increment_eakf(ensemble, observation, obs_error_var)

    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_error_var)

    # Perturbed observations, recentered so their mean equals the observation
    temp_obs = observation + np.sqrt(obs_error_var) * rng.standard_normal(ensemble.shape)
    temp_obs = temp_obs - temp_obs.mean() + observation

    # Product of each prior member with its perturbed observation
    updated_ens = post_var * (ensemble / prior_var + temp_obs / obs_error_var)

    return updated_ens - ensemble


@dataclass
class RHFResult:
    """Result of a rank histogram filter update.

    Attributes
    ----------
    increments : ndarray
        Increments to add to the (unsorted) prior ensemble members.
    prior_pdf : tuple of ndarray
        ``(x, y)`` points tracing the continuous prior PDF, for plotting.
    post_pdf : tuple of ndarray
        ``(x, y)`` points tracing the continuous posterior PDF.
    """

    increments: NDArray[np.float64]
    prior_pdf: tuple[NDArray[np.float64], NDArray[np.float64]]
    post_pdf: tuple[NDArray[np.float64], NDArray[np.float64]]


def obs_increment_rhf(ensemble: ArrayLike, observation: float, obs_error_var: float,
                      bounded_left: bool = False) -> RHFResult:
    """Increments for a rank histogram filter (RHF).

    The prior is represented by a rank histogram: ``1/(n+1)`` probability
    mass between each pair of adjacent sorted members, with Gaussian tails.
    The likelihood multiplies the mass in each bin and new members are placed
    so each has ``1/(n+1)`` posterior mass below it (quantile conservation).

    Parameters
    ----------
    ensemble : array_like
        Prior ensemble (1-D).
    observation : float
        Observed value.
    obs_error_var : float
        Observation error variance.
    bounded_left : bool
        If True the variable is bounded below at zero (the BNRHF of the
        bounded_oned_ensemble tool) and the left tail respects the bound.

    Returns
    -------
    RHFResult
        Increments plus ``(x, y)`` curves for the continuous prior and
        posterior PDFs.
    """
    ensemble = np.asarray(ensemble, dtype=float)
    ens_size = ensemble.size
    prior_sd = ensemble.std(ddof=1)

    # Sort the ensemble members and keep the indices
    e_ind = np.argsort(ensemble)
    x = ensemble[e_ind]

    # Likelihood of each member given the observation (unnormalized)
    like = np.exp(-((x - observation) ** 2) / (2 * obs_error_var))

    # Mean likelihood density in each interior bin; like_dense[i] is for the
    # bin between x[i-1] and x[i]
    like_dense = np.zeros(ens_size)
    like_dense[1:] = (like[:-1] + like[1:]) / 2

    # For a unit normal, distance from the mean to where the cdf is 1/(n+1)
    dist_for_unit_sd = -weighted_norm_inv(1, 0, 1, 1.0 / (ens_size + 1))

    # Tail normals have the sample prior sd; means are adjusted so that
    # 1/(n+1) of the mass is outside the outermost members
    left_mean = x[0] + dist_for_unit_sd * prior_sd
    left_sd = prior_sd
    right_mean = x[-1] - dist_for_unit_sd * prior_sd
    right_sd = prior_sd

    # If bounded on the left, the left tail amplitude is greater than one so
    # that 1/(n+1) mass still lies between the bound and the smallest member
    if bounded_left:
        cdf_at_ens = 1.0 / (ens_size + 1)
        cdf_at_bound = norm.cdf(0, left_mean, left_sd)
        left_prior_amp = cdf_at_ens / (cdf_at_ens - cdf_at_bound)
    else:
        left_prior_amp = 1.0

    # Prior mass is uniform over the n+1 bins
    prior_mass = np.full(ens_size + 1, 1.0 / (ens_size + 1))
    prior_pdf = rhf_pdf_points(x, prior_mass, left_mean, left_sd, left_prior_amp,
                               right_mean, right_sd, 1.0, bounded_left)

    # Flat tails for the likelihood: posterior tail keeps the prior tail shape
    # weighted by the likelihood at the outermost member
    mass = np.zeros(ens_size + 1)
    mass[0] = like[0] / (ens_size + 1)
    mass[-1] = like[-1] / (ens_size + 1)
    # Interior bins: prior mass 1/(n+1) times the mean likelihood density
    mass[1:ens_size] = like_dense[1:ens_size] / (ens_size + 1)

    # Normalize the mass to get a posterior pdf
    mass_sum = mass.sum()
    nmass = mass / mass_sum

    # Weights for the final normalized tail Gaussians
    left_amp = left_prior_amp * like[0] / mass_sum
    right_amp = like[-1] / mass_sum

    # Cumulative mass at each bin boundary
    cumul_mass = np.zeros(ens_size + 2)
    cumul_mass[1:] = np.cumsum(nmass)

    new_ens = np.zeros(ens_size)
    lowest_box = 0

    for i in range(ens_size):
        # Each updated member has (i+1)/(n+1) mass below it
        umass = (i + 1.0) / (ens_size + 1)

        if umass < cumul_mass[1]:
            # Left tail
            if bounded_left:
                # How much tail-normal mass lies to the left of the bound
                lower_mass = left_amp * norm.cdf(0, left_mean, left_sd)
                new_ens[i] = weighted_norm_inv(left_amp, left_mean, left_sd,
                                               umass + lower_mass)
            else:
                new_ens[i] = weighted_norm_inv(left_amp, left_mean, left_sd, umass)
        elif umass > cumul_mass[ens_size]:
            # Right tail: use symmetry after pretending it's on the left
            new_ens[i] = weighted_norm_inv(right_amp, right_mean, right_sd, 1 - umass)
            new_ens[i] = right_mean + (right_mean - new_ens[i])
        else:
            # In one of the interior boxes; linearly interpolate in mass
            for j in range(lowest_box, ens_size - 1):
                if cumul_mass[j + 1] <= umass <= cumul_mass[j + 2]:
                    new_ens[i] = x[j] + ((umass - cumul_mass[j + 1])
                                         / (cumul_mass[j + 2] - cumul_mass[j + 1])
                                         ) * (x[j + 1] - x[j])
                    lowest_box = j
                    break

    # Convert to increments for the unsorted ensemble
    obs_increments = np.zeros(ens_size)
    obs_increments[e_ind] = new_ens - x

    post_pdf = rhf_pdf_points(x, nmass, left_mean, left_sd, left_amp,
                              right_mean, right_sd, right_amp, bounded_left)

    return RHFResult(increments=obs_increments, prior_pdf=prior_pdf, post_pdf=post_pdf)


def obs_increment(ensemble: ArrayLike, observation: float, obs_error_var: float,
                  filter_type: str = "EAKF",
                  rng: np.random.Generator | None = None) -> NDArray[np.float64]:
    """Dispatch to one of the filters by name and return just the increments.

    ``filter_type`` is one of ``"EAKF"``, ``"EnKF"`` or ``"RHF"`` (case
    insensitive), matching the radio buttons of the MATLAB tools.
    """
    kind = filter_type.upper()
    if kind == "EAKF":
        return obs_increment_eakf(ensemble, observation, obs_error_var)
    if kind == "ENKF":
        return obs_increment_enkf(ensemble, observation, obs_error_var, rng=rng)
    if kind == "RHF":
        return obs_increment_rhf(ensemble, observation, obs_error_var).increments
    raise ValueError(f"Unknown filter type {filter_type!r}; expected EAKF, EnKF or RHF")
