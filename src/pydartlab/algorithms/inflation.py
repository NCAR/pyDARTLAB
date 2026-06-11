"""Ensemble inflation: fixed multiplicative and adaptive algorithms.

Ports of ``update_inflate.m`` (Anderson 2009 Gaussian flavor and El Gharamti
2018 inverse-gamma flavor), ``compute_new_density.m``,
``enh_compute_new_density.m``, ``change_GA_IG.m``, ``inflate_gamma.m`` and
``inflate_bnrh.m`` from ``DART_LAB/matlab/private``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gamma as gamma_function
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from pydartlab.algorithms.distributions import bnrh_fit, fit_gamma


def inflate_ensemble(ens: ArrayLike, inflation: float) -> NDArray[np.float64]:
    """Multiplicative covariance inflation about the ensemble mean.

    The ensemble variance is multiplied by ``inflation``; each member moves
    away from the mean by a factor ``sqrt(inflation)``.
    """
    ens = np.asarray(ens, dtype=float)
    mean = ens.mean()
    return (ens - mean) * np.sqrt(inflation) + mean


def compute_new_density(dist_2: float, sigma_p_2: float, sigma_o_2: float,
                        lambda_mean: float, lambda_sd: float, gamma_corr: float,
                        lam: float) -> float:
    """Posterior density of inflation ``lam`` for the Gaussian flavor."""
    exponent_prior = -0.5 * (lam - lambda_mean) ** 2 / lambda_sd**2

    # Probability that the observation would be seen given this lambda
    theta_2 = (1.0 + gamma_corr * (np.sqrt(lam) - 1.0)) ** 2 * sigma_p_2 + sigma_o_2
    theta = np.sqrt(theta_2)
    exponent_likelihood = dist_2 / (-2.0 * theta_2)

    return float(np.exp(exponent_likelihood + exponent_prior)
                 / (2.0 * np.pi * lambda_sd * theta))


def enh_compute_new_density(dist_2: float, sigma_p_2: float, sigma_o_2: float,
                            alpha: float, beta: float, gamma_corr: float,
                            lam: float, ens_size: int) -> float:
    """Posterior density of inflation ``lam`` for the inverse-gamma flavor."""
    exp_prior = -beta / lam

    fac1 = (1 + gamma_corr * (np.sqrt(lam) - 1.0)) ** 2
    fac2 = -1.0 / ens_size
    if fac1 < abs(fac2):
        fac2 = 0.0

    theta = np.sqrt((fac1 + fac2) * sigma_p_2 + sigma_o_2)
    exp_like = -0.5 * dist_2 / theta**2

    return float(beta**alpha / gamma_function(alpha) * lam ** (-alpha - 1)
                 / (np.sqrt(2.0 * np.pi) * theta) * np.exp(exp_like + exp_prior))


def change_ga_ig(mode: float, var: float) -> float:
    """Rate parameter of the inverse-gamma matching a Gaussian's mode and variance.

    Port of ``change_GA_IG.m``: the rate is the real solution of a cubic.
    """
    var_p = [var, var**2, var**3]
    mode_p = [mode**i for i in range(1, 10)]  # mode_p[i] = mode**(i+1)

    aa = mode_p[3] * np.sqrt((var_p[1] + 47 * var * mode_p[1] + 3 * mode_p[3]) / var_p[2])
    bb = 75 * var_p[1] * mode_p[4]
    cc = 21 * var * mode_p[6]
    dd = var_p[2] * mode_p[2]
    ee = (cc + bb + dd + mode_p[8] + 6 * np.sqrt(3) * aa * var_p[2]) / var_p[2]

    beta = ((7 * var * mode + mode_p[2]) / (3 * var)
            + ee ** (1 / 3) / 3
            + mode_p[1] * (var_p[1] + 14 * var * mode_p[1] + mode_p[3])
            / (3 * var_p[1] * ee ** (1 / 3)))
    return float(beta)


def update_inflate(x_p: float, r_var: float, y_o: float, sigma_o_2: float,
                   ss_inflate_base: float, lambda_mean: float, lambda_sd: float,
                   inf_lower_bound: float, inf_upper_bound: float, gamma_corr: float,
                   sd_lower_bound: float, ens_size: int,
                   flavor: str = "Gaussian") -> tuple[float, float]:
    """Bayesian update of the adaptive inflation mean and standard deviation.

    Port of ``update_inflate.m``. Supports two flavors:

    * ``"Gaussian"`` -- Anderson (2009), Tellus A 61, 72-83.
    * ``"I-Gamma"`` -- El Gharamti (2018), MWR 146, 623-640.

    Parameters
    ----------
    x_p : float
        Prior ensemble mean of the observed quantity.
    r_var : float
        Prior ensemble variance of the observed quantity (inflated by
        ``ss_inflate_base``).
    y_o : float
        Observed value.
    sigma_o_2 : float
        Observation error variance.
    ss_inflate_base : float
        Inflation already applied to the sample whose variance is ``r_var``.
    lambda_mean, lambda_sd : float
        Mean and standard deviation of the prior inflation distribution.
    inf_lower_bound, inf_upper_bound : float
        Bounds on the updated inflation mean.
    gamma_corr : float
        Correlation factor for spatially varying inflation (1 for the
        observed variable itself).
    sd_lower_bound : float
        Lower bound for the updated inflation standard deviation.
    ens_size : int
        Ensemble size (used by the inverse-gamma flavor).
    flavor : str
        ``"Gaussian"`` or ``"I-Gamma"``.

    Returns
    -------
    new_cov_inflate, new_cov_inflate_sd : float
        Updated inflation mean and standard deviation.
    """
    if flavor not in ("Gaussian", "I-Gamma"):
        raise ValueError(f"Unknown inflation flavor {flavor!r}; expected 'Gaussian' or 'I-Gamma'")

    # FIRST, update the inflation mean.
    # "Non-inflated" variance of the sample
    sigma_p_2 = r_var / (1 + gamma_corr * (np.sqrt(ss_inflate_base) - 1)) ** 2

    # Squared innovation
    dist_2 = (x_p - y_o) ** 2

    # d is drawn from a Gaussian with variance lambda * sigma_p^2 + sigma_o^2
    beta = None
    if flavor == "Gaussian":
        theta_bar_2 = (1 + gamma_corr * (np.sqrt(lambda_mean) - 1)) ** 2 * sigma_p_2 + sigma_o_2
    else:  # I-Gamma
        fac1 = (1 + gamma_corr * (np.sqrt(lambda_mean) - 1)) ** 2
        fac2 = -1.0 / ens_size
        if fac1 < abs(fac2):
            fac2 = 0.0
        theta_bar_2 = (fac1 + fac2) * sigma_p_2 + sigma_o_2

    theta_bar = np.sqrt(theta_bar_2)
    u_bar = 1.0 / (np.sqrt(2.0 * np.pi) * theta_bar)
    like_exp_bar = -0.5 * dist_2 / theta_bar_2
    v_bar = np.exp(like_exp_bar)

    # The likelihood p(d | lambda)
    like_bar = u_bar * v_bar
    if like_bar <= 0:
        return float(inf_lower_bound), float(lambda_sd)

    gamma_terms = 1 - gamma_corr + gamma_corr * np.sqrt(lambda_mean)
    dtheta_dlambda = (0.5 * sigma_p_2 * gamma_corr * gamma_terms
                      / (theta_bar * np.sqrt(lambda_mean)))

    # Derivative of the likelihood at the current inflation mean
    like_prime = (like_bar * dtheta_dlambda / theta_bar) * (dist_2 / theta_bar_2 - 1)
    if like_prime == 0:
        return float(inf_lower_bound), float(lambda_sd)

    like_ratio = like_bar / like_prime

    # Solve a quadratic for the mode of the posterior
    if flavor == "Gaussian":
        a = 1.0
        b = like_ratio - 2 * lambda_mean
        c = lambda_mean**2 - lambda_sd**2 - like_ratio * lambda_mean
    else:  # I-Gamma: switch from Gaussian prior to inverse-gamma
        beta = change_ga_ig(lambda_mean, lambda_sd**2)
        a = 1 - lambda_mean / beta
        b = like_ratio - 2 * lambda_mean
        c = lambda_mean**2 - like_ratio * lambda_mean

    scale = max(abs(a), abs(b), abs(c))
    a, b, c = a / scale, b / scale, c / scale
    disc = b**2 - 4 * a * c

    if b < 0:
        s1 = 0.5 * (-b + np.sqrt(disc)) / a
    else:
        s1 = 0.5 * (-b - np.sqrt(disc)) / a
    s2 = (c / a) / s1

    # Select the updated mean closest to the prior
    if abs(s2 - lambda_mean) < abs(s1 - lambda_mean):
        new_cov_inflate = s2
    else:
        new_cov_inflate = s1

    if (new_cov_inflate < inf_lower_bound or new_cov_inflate > inf_upper_bound
            or np.isnan(new_cov_inflate)):
        return float(inf_lower_bound), float(lambda_sd)

    # SECOND, update the inflation variance
    if lambda_sd <= sd_lower_bound:
        return float(new_cov_inflate), float(lambda_sd)

    tiny = np.finfo(float).tiny
    new_cov_inflate_sd = lambda_sd

    if flavor == "Gaussian":
        # Compare the density at the new mean with one OLD sd above it
        new_max = compute_new_density(dist_2, sigma_p_2, sigma_o_2, lambda_mean,
                                      lambda_sd, gamma_corr, new_cov_inflate)
        new_1_sd = compute_new_density(dist_2, sigma_p_2, sigma_o_2, lambda_mean,
                                       lambda_sd, gamma_corr, new_cov_inflate + lambda_sd)
        if abs(new_max) <= tiny or abs(new_1_sd) <= tiny:
            return float(new_cov_inflate), float(lambda_sd)

        ratio = new_1_sd / new_max
        if ratio > 0.99:
            return float(new_cov_inflate), float(lambda_sd)

        # sd consistent with the density dropping to `ratio` one sd away
        new_cov_inflate_sd = np.sqrt(-0.5 * lambda_sd**2 / np.log(ratio))
        # Prevent an increase in the sd of lambda
        new_cov_inflate_sd = min(new_cov_inflate_sd, lambda_sd)
    else:  # I-Gamma
        # Shape of the prior IG, from matching its mode to the Gaussian's
        shape_old = beta / lambda_mean - 1
        if shape_old <= 2:
            return float(new_cov_inflate), float(lambda_sd)

        density_1 = enh_compute_new_density(dist_2, sigma_p_2, sigma_o_2, shape_old,
                                            beta, gamma_corr,
                                            new_cov_inflate + lambda_sd, ens_size)
        density_2 = enh_compute_new_density(dist_2, sigma_p_2, sigma_o_2, shape_old,
                                            beta, gamma_corr, new_cov_inflate, ens_size)
        if (abs(density_1) <= tiny or abs(density_2) <= tiny
                or np.isnan(density_1) or np.isnan(density_2)):
            return float(new_cov_inflate), float(lambda_sd)

        # Fit the posterior IG through the two density evaluations
        ratio = density_1 / density_2
        omega = (np.log(new_cov_inflate) / new_cov_inflate + 1 / new_cov_inflate
                 - np.log(new_cov_inflate + lambda_sd) / new_cov_inflate
                 - 1 / (new_cov_inflate + lambda_sd))
        rate_new = np.log(ratio) / omega
        shape_new = rate_new / new_cov_inflate - 1
        if shape_new <= 2:
            return float(new_cov_inflate), float(lambda_sd)

        new_cov_inflate_sd = np.sqrt(rate_new**2 / ((shape_new - 1) ** 2 * (shape_new - 2)))
        # Keep the prior sd if the update grew more than 5% (stability) or is NaN
        if new_cov_inflate_sd > 1.05 * lambda_sd or np.isnan(new_cov_inflate_sd):
            new_cov_inflate_sd = lambda_sd

    new_cov_inflate_sd = max(new_cov_inflate_sd, sd_lower_bound)
    return float(new_cov_inflate), float(new_cov_inflate_sd)


def _inflate_in_probit(probit_ens: NDArray[np.float64], var_inf: float
                       ) -> NDArray[np.float64]:
    """Inflate (by sd factor sqrt(var_inf)) about the mean in probit space."""
    inf = np.sqrt(var_inf)
    probit_mean = probit_ens.mean()
    return (probit_ens - probit_mean) * inf + probit_mean


def inflate_gamma(ens: ArrayLike, var_inf: float) -> NDArray[np.float64]:
    """Inflate an ensemble in a gamma/probit transformed space.

    Fits a gamma distribution to the ensemble, transforms members to probit
    space via their gamma quantiles, inflates there, and transforms back.
    Preserves the non-negative support of the ensemble.
    """
    ens = np.asarray(ens, dtype=float)
    shape, scale = fit_gamma(ens)

    prior_q = gamma_dist.cdf(ens, shape, scale=scale)
    probit_ens = norm.ppf(prior_q)
    probit_inf_ens = _inflate_in_probit(probit_ens, var_inf)
    inf_prior_q = norm.cdf(probit_inf_ens)
    return gamma_dist.ppf(inf_prior_q, shape, scale=scale)


def inflate_bnrh(ens: ArrayLike, var_inf: float, bounded_below: bool = False,
                 bounded_above: bool = False, lower_bound: float = 0.0,
                 upper_bound: float = 0.0) -> NDArray[np.float64]:
    """Inflate an ensemble in a BNRH/probit transformed space.

    Like :func:`inflate_gamma` but with a bounded normal rank histogram
    distribution, supporting arbitrary bounds.
    """
    ens = np.asarray(ens, dtype=float)
    dist = bnrh_fit(ens, bounded_below, bounded_above, lower_bound, upper_bound)

    probit_ens = norm.ppf(dist.quantiles)
    probit_inf_ens = _inflate_in_probit(probit_ens, var_inf)
    inf_prior_q = norm.cdf(probit_inf_ens)
    return dist.ppf(inf_prior_q)
