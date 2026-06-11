"""The Lorenz (1963) three-variable convection model.

Port of ``DART_LAB/matlab/private/lorenz_63_adv_1step.m`` and
``lorenz_63_static_init_model.m``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Lorenz63:
    """Lorenz 63 model advanced with the two-step Runge-Kutta scheme of DART_LAB.

    The classical chaotic "butterfly" system:

    .. math::

        \\dot x = \\sigma (y - x), \\quad
        \\dot y = -xz + rx - y, \\quad
        \\dot z = xy - bz

    Parameters
    ----------
    sigma, r, b : float
        Model parameters; defaults are the classical chaotic values.
    delta_t : float
        Non-dimensional time step (DART_LAB uses 0.01).
    """

    model_size = 3

    def __init__(self, sigma: float = 10.0, r: float = 28.0, b: float = 8.0 / 3.0,
                 delta_t: float = 0.01):
        self.sigma = sigma
        self.r = r
        self.b = b
        self.delta_t = delta_t

    def comp_dt(self, x: ArrayLike) -> NDArray[np.float64]:
        """Time tendency of the state vector ``x`` (length 3)."""
        x = np.asarray(x, dtype=float)
        return np.array([
            self.sigma * (x[1] - x[0]),
            -x[0] * x[2] + self.r * x[0] - x[1],
            x[0] * x[1] - self.b * x[2],
        ])

    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        """Advance ``x`` by one time step with the DART_LAB two-step RK2 scheme."""
        x = np.asarray(x, dtype=float)
        x1 = x + self.delta_t * self.comp_dt(x)
        x2 = x1 + self.delta_t * self.comp_dt(x1)
        return (x + x2) / 2.0
