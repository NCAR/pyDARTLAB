"""Ensemble DA on the Lorenz 63 attractor (run_lorenz_63).

Truth and a 20-member ensemble evolve on the chaotic attractor; at each
assimilation time all three state variables are observed with independent
``Normal(0, 1)`` errors and assimilated sequentially, with increments
regressed onto every state variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pydartlab.algorithms.increments import obs_increment
from pydartlab.algorithms.regression import get_state_increments
from pydartlab.models.lorenz63 import Lorenz63


@dataclass
class Lorenz63Experiment:
    """Cycling DA on the Lorenz 63 model.

    Parameters
    ----------
    ens_size : int
        Number of ensemble members (the MATLAB tool uses 20).
    filter_type : str
        ``"No Assimilation"``, ``"EAKF"``, ``"EnKF"`` or ``"RHF"``.
    obs_error_sd : float
        Observation error standard deviation (1 in the MATLAB tool).
    spin_up : int
        Steps to advance the initial truth onto the attractor.
    seed : int or None
        Seed for reproducibility.
    """

    ens_size: int = 20
    filter_type: str = "No Assimilation"
    obs_error_sd: float = 1.0
    spin_up: int = 1500
    seed: int | None = None

    time: int = 0
    history: dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.model = Lorenz63()

        # Spin the truth up onto the attractor
        truth = np.array([1.0, 1.0, 1.0])
        for _ in range(self.spin_up):
            truth = self.model.step(truth)
        self.truth = truth

        # Ensemble is the truth plus small perturbations
        self.posterior = truth + 0.1 * self.rng.standard_normal((self.ens_size, 3))
        self.prior = self.posterior.copy()
        self.ready_to_advance = True
        self.history = {k: [] for k in [
            "time", "truth", "prior", "posterior", "observation", "increments"]}

    def advance(self) -> None:
        """Advance the truth and all ensemble members one model step."""
        if not self.ready_to_advance:
            raise RuntimeError("Expected assimilate() next; advance/assimilate alternate")
        self.truth = self.model.step(self.truth)
        self.prior = np.array([self.model.step(m) for m in self.posterior])
        self.posterior = self.prior.copy()
        self.time += 1
        self.ready_to_advance = False

        self.history["time"].append(self.time)
        self.history["truth"].append(self.truth.copy())
        self.history["prior"].append(self.prior.copy())

    def assimilate(self) -> None:
        """Observe all three variables (sequentially) and update the ensemble."""
        if self.ready_to_advance:
            raise RuntimeError("Expected advance() next; advance/assimilate alternate")
        self.ready_to_advance = True

        if self.filter_type == "No Assimilation":
            self.history["observation"].append(None)
            self.history["increments"].append(np.zeros_like(self.prior))
            self.history["posterior"].append(self.posterior.copy())
            return

        temp_ens = self.prior.copy()
        obs = self.truth + self.obs_error_sd * self.rng.standard_normal(3)

        for i in range(3):
            obs_prior = temp_ens[:, i].copy()
            incs = obs_increment(obs_prior, obs[i], self.obs_error_sd**2,
                                 self.filter_type, rng=self.rng)
            for j in range(3):
                state_incs, _ = get_state_increments(temp_ens[:, j], obs_prior, incs)
                temp_ens[:, j] += state_incs

        increments = temp_ens - self.prior
        self.posterior = temp_ens

        self.history["observation"].append(obs)
        self.history["increments"].append(increments)
        self.history["posterior"].append(self.posterior.copy())

    def step(self) -> None:
        """One full advance + assimilate cycle."""
        self.advance()
        self.assimilate()
