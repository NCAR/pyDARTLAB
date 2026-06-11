"""Cycled assimilation with a continuous Kalman filter and an ensemble filter.

The experiment behind the ``oned_cycle`` tool: a scalar state with a linear
error-growth model ``x(t+1) = G x(t)``, truth fixed at zero, observations
drawn from ``Normal(0, obs_error_sd^2)``. A continuous (Gaussian) Kalman
filter and an ensemble filter assimilate the same observations side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pydartlab.algorithms.increments import obs_increment
from pydartlab.stats import product_of_gaussians


@dataclass
class CycleRecord:
    """Diagnostics from one assimilation cycle of :class:`KalmanCycle`."""

    observation: float
    kf_prior_mean: float
    kf_prior_sd: float
    kf_post_mean: float
    kf_post_sd: float
    prior_ens: NDArray[np.float64]
    post_ens: NDArray[np.float64]


@dataclass
class KalmanCycle:
    """Continuous Kalman filter and ensemble filter cycling side by side.

    Parameters
    ----------
    growth_rate : float
        Linear model growth rate ``G``; values > 1 grow forecast errors.
    obs_error_sd : float
        Observation error standard deviation.
    filter_type : str
        Ensemble filter: ``"EAKF"``, ``"EnKF"`` or ``"RHF"``.
    kf_mean, kf_sd : float
        Initial mean and sd of the continuous Kalman filter prior.
    seed : int or None
        Seed for the observation draws and the EnKF.
    """

    growth_rate: float = 1.0
    obs_error_sd: float = 1.0
    filter_type: str = "EAKF"
    kf_mean: float = 1.0
    kf_sd: float = 2.0
    seed: int | None = None
    ensemble: NDArray[np.float64] | None = None
    history: list[CycleRecord] = field(default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def set_ensemble(self, ensemble: ArrayLike) -> None:
        """Set the ensemble (e.g. created interactively or programmatically)."""
        self.ensemble = np.asarray(ensemble, dtype=float)
        self.history.clear()

    def cycle(self) -> CycleRecord:
        """Assimilate one observation, then advance both filters with the model."""
        if self.ensemble is None:
            raise RuntimeError("Set an ensemble with set_ensemble() before cycling")

        observation = float(self.rng.standard_normal() * self.obs_error_sd)

        # Continuous Kalman filter: product of Gaussians
        kf_prior_mean, kf_prior_sd = self.kf_mean, self.kf_sd
        kf_post_mean, kf_post_sd, _ = product_of_gaussians(
            kf_prior_mean, kf_prior_sd, observation, self.obs_error_sd)

        # Ensemble filter
        prior_ens = self.ensemble.copy()
        incs = obs_increment(prior_ens, observation, self.obs_error_sd**2,
                             self.filter_type, rng=self.rng)
        post_ens = prior_ens + incs

        record = CycleRecord(observation=observation,
                             kf_prior_mean=kf_prior_mean, kf_prior_sd=kf_prior_sd,
                             kf_post_mean=kf_post_mean, kf_post_sd=kf_post_sd,
                             prior_ens=prior_ens, post_ens=post_ens)
        self.history.append(record)

        # Advance with the linear growth model
        self.kf_mean = self.growth_rate * kf_post_mean
        self.kf_sd = self.growth_rate * kf_post_sd
        self.ensemble = self.growth_rate * post_ens

        return record
