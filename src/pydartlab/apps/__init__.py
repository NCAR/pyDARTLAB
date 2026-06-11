"""Interactive DART_LAB apps for Jupyter (use ``%matplotlib widget``).

Each function launches the Python version of the MATLAB tool of the same
name and returns the app object. Apps display themselves when they are the
last expression in a notebook cell::

    import pydartlab.apps as apps
    apps.oned_ensemble()

Every click-driven app also has a ``set_ensemble`` method for environments
without interactive matplotlib support.
"""

from pydartlab.apps.gaussian_product import GaussianProductApp, gaussian_product
from pydartlab.apps.oned_cycle import OnedCycleApp, oned_cycle
from pydartlab.apps.oned_ensemble import OnedEnsembleApp, oned_ensemble
from pydartlab.apps.twod_ensemble import TwodEnsembleApp, twod_ensemble

__all__ = [
    "gaussian_product", "GaussianProductApp",
    "oned_ensemble", "OnedEnsembleApp",
    "oned_cycle", "OnedCycleApp",
    "twod_ensemble", "TwodEnsembleApp",
]
