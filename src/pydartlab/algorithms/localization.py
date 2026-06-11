"""Covariance localization (port of ``comp_cov_factor.m``)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def comp_cov_factor(z_in: ArrayLike, c: float) -> NDArray[np.float64] | float:
    """Gaspari-Cohn localization factor.

    A compactly supported 5th-order polynomial correlation function that
    decreases from 1 at distance 0 to exactly 0 at distance ``2c``.

    Parameters
    ----------
    z_in : array_like
        Distance(s) between the observation and the state variable.
    c : float
        Half-width: the factor reaches zero at ``2c``.

    Returns
    -------
    cov_factor : ndarray or float
        Regression/covariance reduction factor in [0, 1].
    """
    z = np.abs(np.asarray(z_in, dtype=float))
    r = z / c

    inner = (((-0.25 * r + 0.5) * r + 0.625) * r - 5.0 / 3.0) * r**2 + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        outer = ((((r / 12 - 0.5) * r + 0.625) * r + 5.0 / 3.0) * r - 5.0) * r \
            + 4.0 - 2.0 / (3.0 * r)

    cov_factor = np.where(z >= 2.0 * c, 0.0, np.where(z <= c, inner, outer))
    return cov_factor if cov_factor.ndim else float(cov_factor)


def cyclic_distance(i: int | ArrayLike, j: int | ArrayLike, domain_size: int
                    ) -> NDArray[np.float64] | float:
    """Distance between grid indices ``i`` and ``j`` on a cyclic domain.

    Returned as a fraction of the domain (in [0, 0.5]) as used by the
    Lorenz 96 tools: ``dist = |i - j| / N``, wrapped around the cycle.
    """
    dist = np.abs(np.asarray(i, dtype=float) - np.asarray(j, dtype=float)) / domain_size
    dist = np.where(dist > 0.5, 1.0 - dist, dist)
    return dist if dist.ndim else float(dist)
