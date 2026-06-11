"""Cycling DA for the one-variable nonlinear model (oned_model / oned_model_inf).

The truth is always zero; observations are drawn from
``Normal(0, obs_error_sd^2)``. The model ``x_{t+1} = 2x + bias + a x|x|``
grows errors, optionally with systematic bias and nonlinearity, so the
filter must work to track the truth. Supports fixed and adaptive inflation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pydartlab.algorithms.increments import obs_increment
from pydartlab.algorithms.inflation import update_inflate
from pydartlab.models.oned import advance_oned
from pydartlab.stats import RankHistogram, kurt


@dataclass
class OneDExperiment:
    """The cycling experiment behind ``oned_model`` and ``oned_model_inf``.

    Call :meth:`advance` and :meth:`assimilate` alternately (as the MATLAB
    button does), or :meth:`step` for one full cycle.

    Parameters
    ----------
    ens_size : int
        Ensemble size (>= 2).
    filter_type : str
        ``"EAKF"``, ``"EnKF"`` or ``"RHF"``.
    model_bias : float
        Systematic model error added each advance (>= 0 in the GUI).
    alpha : float
        Nonlinearity parameter ``a`` in ``x + bias + a x |x|``.
    inflation : float
        Fixed inflation, or the initial inflation mean when adaptive.
    adaptive_inflation : bool
        Use the adaptive algorithm of ``oned_model_inf`` instead of fixed.
    inflation_flavor : str
        ``"Gaussian"`` (Anderson 2009) or ``"I-Gamma"`` (El Gharamti 2018).
    inflation_sd : float
        Initial standard deviation of the inflation distribution.
    inflation_sd_min : float
        Lower bound for the inflation standard deviation.
    inflation_min, inflation_max : float
        Bounds for the inflation mean.
    inflation_damping : float
        Damping factor: ``inflation = 1 + damping * (lambda - 1)``.
    obs_error_sd : float
        Observation error standard deviation.
    seed : int or None
        Seed for reproducibility.
    """

    ens_size: int = 4
    filter_type: str = "EAKF"
    model_bias: float = 0.0
    alpha: float = 0.0
    inflation: float = 1.0
    adaptive_inflation: bool = False
    inflation_flavor: str = "Gaussian"
    inflation_sd: float = 0.6
    inflation_sd_min: float = 0.6
    inflation_min: float = 1.0
    inflation_max: float = 5.0
    inflation_damping: float = 1.0
    obs_error_sd: float = 1.0
    seed: int | None = None

    truth: float = 0.0
    time: int = 0
    history: dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.ens: NDArray[np.float64] = self.rng.standard_normal(self.ens_size)
        self.prior_rank_hist = RankHistogram(self.ens_size)
        self.post_rank_hist = RankHistogram(self.ens_size)
        self.ready_to_advance = True
        self.history = {k: [] for k in [
            "time", "observation", "prior_ens", "post_ens",
            "prior_error", "prior_spread", "post_error", "post_spread",
            "prior_kurtosis", "post_kurtosis", "inflation", "inflation_sd"]}

    def set_ens_size(self, ens_size: int) -> None:
        """Change the ensemble size, keeping or padding existing members."""
        if ens_size < self.ens_size:
            self.ens = self.ens[:ens_size]
        else:
            extra = self.rng.standard_normal(ens_size - self.ens_size)
            self.ens = np.concatenate([self.ens, extra])
        self.ens_size = ens_size
        self.prior_rank_hist = RankHistogram(ens_size)
        self.post_rank_hist = RankHistogram(ens_size)

    def advance(self) -> NDArray[np.float64]:
        """Advance the ensemble one step, then apply (current) inflation."""
        if not self.ready_to_advance:
            raise RuntimeError("Expected assimilate() next; advance/assimilate alternate")
        ens_new = advance_oned(self.ens, self.alpha, self.model_bias)
        ens_mean = ens_new.mean()
        self.ens = (ens_new - ens_mean) * np.sqrt(self.inflation) + ens_mean
        self.time += 1
        self.ready_to_advance = False

        self.prior_rank_hist.add(self.ens, self.truth)
        self.history["time"].append(self.time)
        self.history["prior_ens"].append(self.ens.copy())
        self.history["prior_error"].append(abs(self.ens.mean() - self.truth))
        self.history["prior_spread"].append(self.ens.std(ddof=1))
        self.history["prior_kurtosis"].append(kurt(self.ens))
        return self.ens

    def assimilate(self) -> NDArray[np.float64]:
        """Draw an observation of the (zero) truth and assimilate it."""
        if self.ready_to_advance:
            raise RuntimeError("Expected advance() next; advance/assimilate alternate")
        observation = float(self.obs_error_sd * self.rng.standard_normal())

        if self.adaptive_inflation:
            # The ensemble was inflated at advance time with the current
            # inflation; update the inflation distribution from this obs.
            # The new value takes effect at the next advance.
            inf_prior = self.inflation
            lam, self.inflation_sd = update_inflate(
                float(self.ens.mean()), float(self.ens.var(ddof=1)), observation,
                self.obs_error_sd**2, inf_prior, self.inflation, self.inflation_sd,
                self.inflation_min, self.inflation_max, 1.0, self.inflation_sd_min,
                self.ens_size, self.inflation_flavor)
            self.inflation = 1.0 + self.inflation_damping * (lam - 1.0)

        incs = obs_increment(self.ens, observation, self.obs_error_sd**2,
                             self.filter_type, rng=self.rng)
        self.ens = self.ens + incs
        self.ready_to_advance = True

        self.post_rank_hist.add(self.ens, self.truth)
        self.history["observation"].append(observation)
        self.history["post_ens"].append(self.ens.copy())
        self.history["post_error"].append(abs(self.ens.mean() - self.truth))
        self.history["post_spread"].append(self.ens.std(ddof=1))
        self.history["post_kurtosis"].append(kurt(self.ens))
        self.history["inflation"].append(self.inflation)
        self.history["inflation_sd"].append(self.inflation_sd)
        return self.ens

    def step(self) -> NDArray[np.float64]:
        """One full advance + assimilate cycle."""
        self.advance()
        return self.assimilate()

    def reset_histograms(self) -> None:
        """Clear the accumulated rank histograms (the "Clear Hist" button)."""
        self.prior_rank_hist.reset()
        self.post_rank_hist.reset()
