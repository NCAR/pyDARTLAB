pyDARTLAB
=========

.. toctree::
   :maxdepth: 3
   :hidden:

   self
   quickstart
   tutorial
   api

pyDARTLAB is a Python version of DART_LAB, the interactive ensemble data
assimilation tutorial distributed with the Data Assimilation Research
Testbed (`DART <https://github.com/NCAR/DART>`_). The MATLAB tools and
slide decks become a Python package plus Jupyter notebooks, so the whole
tutorial can be done in a notebook.

The package has three layers:

* :mod:`pydartlab.algorithms`, :mod:`pydartlab.models`,
  :mod:`pydartlab.stats` -- GUI-free computational routines ported from
  ``DART/guide/DART_LAB/matlab/private``.
* :mod:`pydartlab.experiments` -- scriptable cycling DA experiments.
* :mod:`pydartlab.apps` -- the interactive tools, one per MATLAB DART_LAB
  app, for use in Jupyter with ``%matplotlib widget``.

.. note::

    The MATLAB version of
    `DART_LAB <https://dart.ucar.edu/tutorials/dart-lab/>`_
    remains available through `DART <https://github.com/NCAR/DART>`_.

Contributing
============

Contributions are welcome! If you have a feature request, bug report, or a suggestion,
please open an issue on our `GitHub repository <https://github.com/NCAR/pyDARTLAB>`_.

License
=======

pyDARTLAB is released under the Apache License 2.0. For more details, see the LICENSE file in the root directory of this source
tree or visit `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
