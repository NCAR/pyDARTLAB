"""One-dimensional toy models used by ``oned_model`` and ``oned_cycle``.

Port of ``DART_LAB/matlab/private/advance_oned.m`` plus the linear growth
model used by ``oned_cycle.m``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def advance_oned(x: ArrayLike, alpha: float = 0.0, model_bias: float = 0.0
                 ) -> NDArray[np.float64] | float:
    """Advance the oned_model state by one step.

    The time tendency is ``dx = (x + model_bias) + alpha * x * |x|`` so

    .. math:: x_{t+1} = 2 x_t + \\text{bias} + \\alpha\\, x_t |x_t|

    ``alpha`` controls nonlinearity and ``model_bias`` a systematic shift in
    the model dynamics. The truth in ``oned_model`` is always 0, so any bias
    or nonlinearity acts as model error. Works on scalars or ensembles.
    """
    x = np.asarray(x, dtype=float)
    dx = (x + model_bias) + alpha * x * np.abs(x)
    result = x + dx
    return result if result.ndim else float(result)


class LinearGrowth:
    """The linear growth model of ``oned_cycle``: ``x_{t+1} = G x_t``."""

    def __init__(self, growth_rate: float = 1.0):
        self.growth_rate = growth_rate

    def step(self, x: ArrayLike) -> NDArray[np.float64] | float:
        x = np.asarray(x, dtype=float)
        result = self.growth_rate * x
        return result if result.ndim else float(result)
