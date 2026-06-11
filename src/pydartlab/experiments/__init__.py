"""Scriptable, GUI-free cycling data assimilation experiments.

Each class encapsulates the model advance / assimilate cycle of one of the
DART_LAB cycling tools, keeping a history of diagnostics. The interactive
apps drive these classes, but they can equally be run in a loop from a
notebook or a test.
"""

from pydartlab.experiments.kalman_cycle import KalmanCycle
from pydartlab.experiments.lorenz63_da import Lorenz63Experiment
from pydartlab.experiments.lorenz96_da import Lorenz96Experiment
from pydartlab.experiments.oned_da import OneDExperiment

__all__ = ["KalmanCycle", "OneDExperiment", "Lorenz63Experiment", "Lorenz96Experiment"]
