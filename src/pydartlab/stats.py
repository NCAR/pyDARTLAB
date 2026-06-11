"""Statistical helpers shared across DART_LAB tools.

Ports of ``product_of_gaussians.m``, ``kurt.m``, and ``get_ens_rank.m``,
plus a rank-histogram accumulator used by the cycling apps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def product_of_gaussians(prior_mean: float, prior_sd: float,
                         obs: float, obs_err_sd: float) -> tuple[float, float, float]:
    """Mean, standard deviation, and weight of the product of two Gaussians.

    Returns
    -------
    post_mean, post_sd, weight : float
        Mean and standard deviation of the (renormalized) product, and the
        integral of the raw product (the "weight"), which is the likelihood
        of the observation given the prior.
    """
    prior_var = prior_sd**2
    obs_err_var = obs_err_sd**2

    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_err_var)
    post_sd = np.sqrt(post_var)
    post_mean = post_var * (prior_mean / prior_var + obs / obs_err_var)

    weight = (1.0 / (np.sqrt(2.0 * np.pi) * np.sqrt(prior_var + obs_err_var))) * np.exp(
        -0.5 * (obs - prior_mean) ** 2 / (prior_var + obs_err_var)
    )
    return float(post_mean), float(post_sd), float(weight)


def kurt(vals: ArrayLike) -> float:
    """Kurtosis of ``vals`` (biased moment estimator, not excess: Gaussian -> 3)."""
    vals = np.asarray(vals, dtype=float)
    delta = vals - vals.mean()
    m2 = np.mean(delta**2)
    m4 = np.mean(delta**4)
    return float(m4 / m2**2)


def get_ens_rank(ens: ArrayLike, x: float) -> int:
    """Rank of the value ``x`` within the sorted ensemble ``ens``.

    Returns a 1-based rank in ``1 .. ens_size + 1`` as in the MATLAB
    original, suitable for accumulating rank histograms.
    """
    s_ens = np.sort(np.asarray(ens, dtype=float))
    return int(np.searchsorted(s_ens, x, side="left")) + 1


class RankHistogram:
    """Accumulates a rank histogram for an ensemble of size ``ens_size``.

    The histogram has ``ens_size + 1`` bins. ``add(ens, truth)`` computes the
    rank of the truth in the ensemble and increments the matching bin.
    """

    def __init__(self, ens_size: int):
        self.ens_size = ens_size
        self.counts = np.zeros(ens_size + 1, dtype=int)
        self.last_rank: int | None = None

    def add(self, ens: ArrayLike, truth: float) -> int:
        rank = get_ens_rank(ens, truth)
        self.counts[rank - 1] += 1
        self.last_rank = rank
        return rank

    def reset(self) -> None:
        self.counts[:] = 0
        self.last_rank = None


def error_and_spread(ens: ArrayLike, truth: float | ArrayLike) -> tuple[float, float]:
    """RMS error of the ensemble mean and ensemble spread (standard deviation).

    For a multivariate ensemble pass ``ens`` with shape ``(ens_size, model_size)``;
    the error is the RMS over variables of (ensemble mean - truth) and the
    spread is the RMS over variables of the ensemble standard deviation.
    """
    ens = np.asarray(ens, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if ens.ndim == 1:
        error = abs(float(ens.mean()) - float(truth))
        spread = float(ens.std(ddof=1))
    else:
        error = float(np.sqrt(np.mean((ens.mean(axis=0) - truth) ** 2)))
        spread = float(np.sqrt(np.mean(ens.std(axis=0, ddof=1) ** 2)))
    return error, spread
