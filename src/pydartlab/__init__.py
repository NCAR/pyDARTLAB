"""pyDARTLAB: interactive ensemble data assimilation teaching tools.

A Python version of the MATLAB DART_LAB tools distributed with DART
(https://github.com/NCAR/DART), designed for use in Jupyter notebooks.

The package has three layers:

* :mod:`pydartlab.algorithms`, :mod:`pydartlab.models`, :mod:`pydartlab.stats`
  -- pure computational routines mirroring ``DART_LAB/matlab/private``.
* :mod:`pydartlab.experiments` -- scriptable, GUI-free cycling data
  assimilation experiments.
* :mod:`pydartlab.apps` -- the interactive tools (one per MATLAB app),
  for use with ``%matplotlib widget`` in Jupyter.
"""

from pydartlab import style
from pydartlab.algorithms import (
    InvalidVarianceError,
    comp_cov_factor,
    cyclic_distance,
    gamma_ppi_update,
    get_state_increments,
    inflate_bnrh,
    inflate_ensemble,
    inflate_gamma,
    obs_increment,
    obs_increment_bnrhf,
    obs_increment_eakf,
    obs_increment_enkf,
    obs_increment_gamma,
    obs_increment_rhf,
    ppi_update,
    update_inflate,
)
from pydartlab.models import Lorenz63, Lorenz96, advance_oned
from pydartlab.stats import get_ens_rank, kurt, product_of_gaussians

__version__ = "0.1.0.dev0"

__all__ = [
    "style",
    "InvalidVarianceError",
    "obs_increment",
    "obs_increment_eakf",
    "obs_increment_enkf",
    "obs_increment_rhf",
    "obs_increment_gamma",
    "obs_increment_bnrhf",
    "inflate_ensemble",
    "inflate_gamma",
    "inflate_bnrh",
    "update_inflate",
    "comp_cov_factor",
    "cyclic_distance",
    "get_state_increments",
    "ppi_update",
    "gamma_ppi_update",
    "Lorenz63",
    "Lorenz96",
    "advance_oned",
    "product_of_gaussians",
    "kurt",
    "get_ens_rank",
]
