"""Curves for continuous distributions (ports of plot_gaussian.m / plot_gamma.m).

These return ``(x, y)`` arrays rather than drawing, so they can be used by
apps, notebooks and tests alike.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


def gaussian_curve(mean: float, sd: float, weight: float = 1.0, num_points: int = 1001
                   ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Points tracing ``weight * N(mean, sd)`` over +/- 5 standard deviations."""
    x = np.linspace(mean - 5 * sd, mean + 5 * sd, num_points)
    return x, weight * norm.pdf(x, mean, sd)


def gamma_curve(shape: float, scale: float, num_points: int = 1001
                ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Points tracing a gamma PDF from 0 to the mean plus 5 standard deviations."""
    gmean = shape * scale
    gsd = np.sqrt(shape) * scale
    x = np.linspace(0.0, gmean + 5 * gsd, num_points)
    return x, gamma_dist.pdf(x, shape, scale=scale)
