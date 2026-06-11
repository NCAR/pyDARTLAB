Quickstart
==========

Installation
------------

pyDARTLAB can be installed through pip. We recommend installing pyDARTLAB
in a virtual environment:

.. code-block:: text

    python3 -m venv dartlab
    source dartlab/bin/activate
    pip install -e ".[tutorial]"

Running the tutorial
--------------------

.. code-block:: text

    jupyter lab notebooks/

Start with ``00_getting_started.ipynb``. Each notebook launches the
interactive tools inline; make sure the first cell runs
``%matplotlib widget``.

Using the package directly
--------------------------

All the algorithms are plain functions::

    import numpy as np
    import pydartlab as dl

    ens = np.array([-1.2, 0.3, 1.9, 0.6, -0.8])
    increments = dl.obs_increment_eakf(ens, observation=0.5, obs_error_var=1.0)
    posterior = ens + increments

and the cycling experiments are scriptable::

    from pydartlab.experiments import Lorenz96Experiment

    exp = Lorenz96Experiment(filter_type="EAKF", localization=0.2, seed=1)
    for _ in range(100):
        exp.step()
    print(exp.history["post_error"][-1])
