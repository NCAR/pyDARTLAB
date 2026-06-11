"""Ensemble data assimilation algorithms (GUI-free).

These functions mirror ``DART_LAB/matlab/private`` and are the computational
core of pyDARTLAB.
"""

from pydartlab.algorithms.filters import obs_increment_bnrhf, obs_increment_gamma
from pydartlab.algorithms.increments import (
    InvalidVarianceError,
    RHFResult,
    obs_increment,
    obs_increment_eakf,
    obs_increment_enkf,
    obs_increment_rhf,
)
from pydartlab.algorithms.inflation import (
    inflate_bnrh,
    inflate_ensemble,
    inflate_gamma,
    update_inflate,
)
from pydartlab.algorithms.localization import comp_cov_factor, cyclic_distance
from pydartlab.algorithms.qceff import gamma_ppi_update, ppi_update
from pydartlab.algorithms.regression import get_state_increments

__all__ = [
    "InvalidVarianceError",
    "RHFResult",
    "obs_increment",
    "obs_increment_eakf",
    "obs_increment_enkf",
    "obs_increment_rhf",
    "obs_increment_gamma",
    "obs_increment_bnrhf",
    "inflate_ensemble",
    "update_inflate",
    "inflate_gamma",
    "inflate_bnrh",
    "comp_cov_factor",
    "cyclic_distance",
    "get_state_increments",
    "ppi_update",
    "gamma_ppi_update",
]
