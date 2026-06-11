"""Ensemble DA on the 40-variable Lorenz 96 model.

The experiment behind ``run_lorenz_96`` and ``run_lorenz_96_inf``:
sequential assimilation of noisy observations with Gaspari-Cohn
localization, fixed or spatially-varying adaptive inflation, configurable
observing networks, and an optional imperfect model (different forcing for
the ensemble than for the truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pydartlab.algorithms.increments import obs_increment
from pydartlab.algorithms.inflation import update_inflate
from pydartlab.algorithms.localization import comp_cov_factor, cyclic_distance
from pydartlab.models.lorenz96 import Lorenz96
from pydartlab.stats import RankHistogram, error_and_spread

#: Observing networks of run_lorenz_96_inf (1-based MATLAB ranges -> indices)
OBS_NETWORKS = {
    "1:40:1": np.arange(0, 40),
    "1:40:2": np.arange(1, 40, 2),
    "1:40:4": np.arange(1, 40, 4),
    "1:20": np.arange(1, 21),
    "10:30": np.arange(10, 31),
    "1:10; 30:40": np.concatenate([np.arange(0, 11), np.arange(30, 40)]),
}


@dataclass
class Lorenz96Experiment:
    """Cycling DA on the Lorenz 96 model.

    Parameters
    ----------
    ens_size : int
        Number of ensemble members.
    filter_type : str
        ``"No Assimilation"``, ``"EAKF"``, ``"EnKF"`` or ``"RHF"``.
    localization : float
        Gaspari-Cohn half-width as a fraction of the domain (factor reaches
        zero at twice this distance). Large values (e.g. 1e6) disable
        localization.
    inflation : float
        Fixed inflation value, or initial inflation when adaptive.
    adaptive_inflation : bool
        Use spatially-varying adaptive inflation (run_lorenz_96_inf).
    inflation_flavor : str
        ``"Gaussian"`` or ``"I-Gamma"``.
    inflation_sd : float
        Inflation standard deviation (also its lower bound, as in the tool).
    inflation_min, inflation_max : float
        Bounds for the inflation mean.
    inflation_damping : float
        Damping applied to the inflation field at each model advance.
    forcing : float
        Forcing used to advance the ensemble (truth always uses 8.0);
        a different value gives an imperfect model.
    obs_network : str
        One of the keys of :data:`OBS_NETWORKS`.
    obs_error_sd : float
        Observation error standard deviation (4 in the MATLAB tools).
    seed : int or None
        Seed for reproducibility.
    """

    ens_size: int = 20
    filter_type: str = "No Assimilation"
    localization: float = 1.0
    inflation: float = 1.0
    adaptive_inflation: bool = False
    inflation_flavor: str = "Gaussian"
    inflation_sd: float = 0.6
    inflation_min: float = 1.0
    inflation_max: float = 5.0
    inflation_damping: float = 0.9
    forcing: float = 8.0
    true_forcing: float = 8.0
    obs_network: str = "1:40:1"
    obs_error_sd: float = 4.0
    seed: int | None = None

    time: int = 0
    history: dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.true_model = Lorenz96(forcing=self.true_forcing)
        self.assim_model = Lorenz96(forcing=self.forcing)
        self.model_size = self.true_model.model_size

        # Truth starts near the unstable fixed point x_j = F with one
        # variable perturbed, as in the MATLAB tool
        self.truth = np.full(self.model_size, self.true_forcing)
        self.truth[0] = 1.001 * self.true_forcing

        self.posterior = self.truth + 0.001 * self.rng.standard_normal(
            (self.ens_size, self.model_size))
        self.prior = self.posterior.copy()

        #: Spatially varying inflation field (scalar replicated when fixed)
        self.inflate = np.full(self.model_size, float(self.inflation))

        self.prior_rank_hist = RankHistogram(self.ens_size)
        self.post_rank_hist = RankHistogram(self.ens_size)
        self.ready_to_advance = True
        self.history = {k: [] for k in [
            "time", "prior_error", "prior_spread", "post_error", "post_spread",
            "inflate_mean", "inflate"]}

    def set_obs_locations(self) -> NDArray[np.intp]:
        try:
            return OBS_NETWORKS[self.obs_network]
        except KeyError:
            raise ValueError(f"Unknown obs network {self.obs_network!r}; "
                             f"expected one of {list(OBS_NETWORKS)}") from None

    def advance(self) -> None:
        """Advance truth and ensemble one step; damp and apply inflation."""
        if not self.ready_to_advance:
            raise RuntimeError("Expected assimilate() next; advance/assimilate alternate")
        self.truth = self.true_model.step(self.truth)
        self.prior = np.array([self.assim_model.step(m) for m in self.posterior])
        self.time += 1
        self.ready_to_advance = False

        if self.adaptive_inflation:
            # Damp the inflation field toward 1, then inflate the prior
            self.inflate = 1.0 + self.inflation_damping * (self.inflate - 1.0)
        else:
            self.inflate = np.full(self.model_size, float(self.inflation))
        ens_mean = self.prior.mean(axis=0)
        self.prior = ens_mean + np.sqrt(self.inflate) * (self.prior - ens_mean)

        if self.filter_type == "No Assimilation":
            self.posterior = self.prior.copy()

        error, spread = error_and_spread(self.prior, self.truth)
        for i in range(self.model_size):
            self.prior_rank_hist.add(self.prior[:, i], self.truth[i])
        self.history["time"].append(self.time)
        self.history["prior_error"].append(error)
        self.history["prior_spread"].append(spread)

    def assimilate(self) -> None:
        """Sequentially assimilate observations from the chosen network."""
        if self.ready_to_advance:
            raise RuntimeError("Expected advance() next; advance/assimilate alternate")
        self.ready_to_advance = True

        obs_error_var = self.obs_error_sd**2
        obs = np.full(self.model_size, np.nan)
        self.last_obs = obs

        if self.filter_type != "No Assimilation":
            temp_ens = self.prior.copy()
            ss_inflate_base = self.inflate.copy()

            all_vars = np.arange(self.model_size)
            for i in self.set_obs_locations():
                obs_prior = temp_ens[:, i].copy()
                obs[i] = self.truth[i] + self.obs_error_sd * self.rng.standard_normal()
                incs = obs_increment(obs_prior, obs[i], obs_error_var,
                                     self.filter_type, rng=self.rng)

                # Regress the increments onto every state variable at once;
                # equivalent to get_state_increments per variable
                obs_anom = obs_prior - obs_prior.mean()
                obs_var = obs_anom @ obs_anom / (self.ens_size - 1)
                state_anom = temp_ens - temp_ens.mean(axis=0)
                covars = obs_anom @ state_anom / (self.ens_size - 1)
                cov_factors = comp_cov_factor(
                    cyclic_distance(i, all_vars, self.model_size), self.localization)
                temp_ens += np.outer(incs, covars / obs_var * cov_factors)

                if self.adaptive_inflation:
                    for j in range(self.model_size):
                        # As in run_lorenz_96_inf.m: gamma is the localization
                        # factor times |r_xy| from get_state_increments
                        gamma = cov_factors[j] * abs(covars[j])
                        self.inflate[j], _ = update_inflate(
                            float(obs_prior.mean()), float(obs_prior.var(ddof=1)),
                            obs[i], obs_error_var, ss_inflate_base[j],
                            self.inflate[j], self.inflation_sd,
                            self.inflation_min, self.inflation_max, gamma,
                            self.inflation_sd, self.ens_size, self.inflation_flavor)

            self.posterior = temp_ens

        error, spread = error_and_spread(self.posterior, self.truth)
        for i in range(self.model_size):
            self.post_rank_hist.add(self.posterior[:, i], self.truth[i])
        self.history["post_error"].append(error)
        self.history["post_spread"].append(spread)
        self.history["inflate_mean"].append(float(self.inflate.mean()))
        self.history["inflate"].append(self.inflate.copy())

    def step(self) -> None:
        """One full advance + assimilate cycle."""
        self.advance()
        self.assimilate()

    def reset_histograms(self) -> None:
        self.prior_rank_hist.reset()
        self.post_rank_hist.reset()
