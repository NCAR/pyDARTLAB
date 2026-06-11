"""Interactive DART_LAB apps for Jupyter (use ``%matplotlib widget``).

Each function launches the Python version of the MATLAB tool of the same
name and returns the app object. Apps display themselves when they are the
last expression in a notebook cell::

    import pydartlab.apps as apps
    apps.oned_ensemble()

Every click-driven app also has a ``set_ensemble`` method for environments
without interactive matplotlib support.
"""

from pydartlab.apps.bounded_oned_ensemble import (
    BoundedOnedEnsembleApp,
    bounded_oned_ensemble,
)
from pydartlab.apps.gaussian_product import GaussianProductApp, gaussian_product
from pydartlab.apps.oned_cycle import OnedCycleApp, oned_cycle
from pydartlab.apps.oned_ensemble import OnedEnsembleApp, oned_ensemble
from pydartlab.apps.oned_model import OnedModelApp, oned_model, oned_model_inf
from pydartlab.apps.run_lorenz_63 import RunLorenz63App, run_lorenz_63
from pydartlab.apps.run_lorenz_96 import (
    RunLorenz96App,
    run_lorenz_96,
    run_lorenz_96_inf,
)
from pydartlab.apps.twod_ensemble import TwodEnsembleApp, twod_ensemble
from pydartlab.apps.twod_ppi_ensemble import TwodPPIEnsembleApp, twod_ppi_ensemble

__all__ = [
    "gaussian_product", "GaussianProductApp",
    "oned_ensemble", "OnedEnsembleApp",
    "oned_cycle", "OnedCycleApp",
    "oned_model", "oned_model_inf", "OnedModelApp",
    "twod_ensemble", "TwodEnsembleApp",
    "twod_ppi_ensemble", "TwodPPIEnsembleApp",
    "bounded_oned_ensemble", "BoundedOnedEnsembleApp",
    "run_lorenz_63", "RunLorenz63App",
    "run_lorenz_96", "run_lorenz_96_inf", "RunLorenz96App",
]
