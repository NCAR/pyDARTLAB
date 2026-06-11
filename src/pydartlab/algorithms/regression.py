"""Regression of observation increments onto state variables.

Port of ``get_state_increments.m``: the heart of multivariate ensemble
assimilation. Increments computed for an observed quantity are mapped to
any state variable with linear regression on the joint prior ensemble.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def get_state_increments(state_ens: ArrayLike, obs_ens: ArrayLike,
                         obs_incs: ArrayLike) -> tuple[NDArray[np.float64], float]:
    """State increments from observation increments by linear regression.

    Parameters
    ----------
    state_ens : array_like
        Prior ensemble of the (unobserved) state variable.
    obs_ens : array_like
        Prior ensemble of the observed variable.
    obs_incs : array_like
        Observation-space increments for each member.

    Returns
    -------
    state_incs : ndarray
        Increments for the state variable: ``obs_incs * cov(x, y) / var(y)``.
    r_xy : float
        Sample covariance between the state and observed ensembles.
    """
    state_ens = np.asarray(state_ens, dtype=float)
    obs_ens = np.asarray(obs_ens, dtype=float)
    obs_incs = np.asarray(obs_incs, dtype=float)

    covar = np.cov(state_ens, obs_ens, ddof=1)
    state_incs = obs_incs * covar[0, 1] / covar[1, 1]
    return state_incs, float(covar[0, 1])
