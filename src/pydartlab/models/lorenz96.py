"""The Lorenz (1996) model on a cyclic domain.

Port of ``DART_LAB/matlab/private/lorenz_96_adv_1step.m`` and
``lorenz_96_static_init_model.m`` (originally ported in pyDARTLAB 0.0.1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Lorenz96:
    """Lorenz 96 model advanced with a four-step Runge-Kutta scheme.

    The model simulates an atmospheric quantity at ``model_size`` equally
    spaced points on a cyclic (circular) domain:

    .. math::

        \\dot x_j = (x_{j+1} - x_{j-2}) x_{j-1} - x_j + F

    Parameters
    ----------
    model_size : int
        Number of state variables (DART_LAB uses 40).
    delta_t : float
        Non-dimensional time step (DART_LAB uses 0.05).
    forcing : float
        Forcing term ``F``; 8.0 gives chaotic dynamics.
    """

    def __init__(self, model_size: int = 40, delta_t: float = 0.05, forcing: float = 8.0):
        self.model_size = model_size
        self.delta_t = delta_t
        self.forcing = forcing
        #: Locations of the state variables on the [0, 1) cyclic domain.
        self.state_loc = np.arange(model_size) / model_size

    def comp_dt(self, x: ArrayLike) -> NDArray[np.float64]:
        """Time tendency of the state vector ``x``."""
        x = np.asarray(x, dtype=float)
        xp1 = np.roll(x, -1)
        xm2 = np.roll(x, 2)
        xm1 = np.roll(x, 1)
        return (xp1 - xm2) * xm1 - x + self.forcing

    def step(self, x: ArrayLike) -> NDArray[np.float64]:
        """Advance ``x`` by one time step with the DART_LAB RK4 scheme."""
        x = np.asarray(x, dtype=float)
        x1 = self.delta_t * self.comp_dt(x)
        x2 = self.delta_t * self.comp_dt(x + x1 / 2)
        x3 = self.delta_t * self.comp_dt(x + x2 / 2)
        x4 = self.delta_t * self.comp_dt(x + x3)
        return x + x1 / 6 + x2 / 3 + x3 / 3 + x4 / 6
