[![notebook-link](https://img.shields.io/badge/notebook-link-e2d610?logo=jupyter&logoColor=white)](https://notebook.link/github/NCAR/pyDARTLAB/?path=notebooks)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NCAR/pyDARTLAB/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2F00_getting_started.ipynb)

# pyDARTLAB

pyDARTLAB is a Python version of [DART_LAB](https://docs.dart.ucar.edu/en/latest/guide/DART_LAB/DART_LAB.html),
the interactive ensemble data assimilation tutorial distributed with NCAR's
[DART](https://github.com/NCAR/DART) (Data Assimilation Research Testbed).
The DART_LAB MATLAB tools and slide decks become a Python package plus a set
of Jupyter notebooks, so the whole tutorial can be done in a notebook.

![Example twod](https://github.com/NCAR/pyDARTLAB/blob/main/docs/images/two_example.png?raw=true "twod_ensemble")

Documentation is online at https://ncar.github.io/pyDARTLAB

You can try the notebooks online with Notebook.link or Binder:  
[![notebook-link](https://img.shields.io/badge/notebook-link-e2d610?logo=jupyter&logoColor=white)](https://notebook.link/github/NCAR/pyDARTLAB/?path=notebooks)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/NCAR/pyDARTLAB/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2F00_getting_started.ipynb)


Disclaimer: This is a project to explore using Claude - the AI tool. 

## Installation

```bash
pip install -e ".[tutorial]"
```

## The tutorial

```bash
jupyter lab notebooks/
```

Start with `00_getting_started.ipynb`. The six numbered notebooks replace the
six DART_LAB slide sections:

1. Ensemble data assimilation concepts in 1D
2. Multivariate assimilation
3. Inflation and localization
4. Non-Gaussian and bounded filters (QCEFF)
5. Adaptive inflation
6. Using the real DART system

## The package

Three layers, mirroring the design of the MATLAB DART_LAB:

* **`pydartlab`** — GUI-free algorithms ported from
  `DART/guide/DART_LAB/matlab/private`: EAKF/EnKF/RHF observation
  increments, gamma and bounded-RHF filters, the QCEFF/probit (PPI)
  transforms, fixed and adaptive inflation (Gaussian and inverse-gamma),
  Gaspari-Cohn localization, increment regression, and the Lorenz 63/96
  models.
* **`pydartlab.experiments`** — scriptable cycling DA experiments
  (`OneDExperiment`, `KalmanCycle`, `Lorenz63Experiment`,
  `Lorenz96Experiment`).
* **`pydartlab.apps`** — the interactive tools, one per MATLAB app
  (`gaussian_product`, `oned_ensemble`, `oned_cycle`, `oned_model`,
  `oned_model_inf`, `twod_ensemble`, `twod_ppi_ensemble`,
  `bounded_oned_ensemble`, `run_lorenz_63`, `run_lorenz_96`,
  `run_lorenz_96_inf`). Use `%matplotlib widget` in the notebook; every
  click-driven app also has a `set_ensemble()` method for environments
  without mouse support.

Colors follow the DART_LAB convention (green = prior, red = observation,
blue = posterior) and are settable — `pydartlab.style.use_colorblind_palette()`
switches to an Okabe-Ito palette.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
ruff check src tests
```

Notebook outputs are kept out of version control with
[nbstripout](https://github.com/kynan/nbstripout). After cloning, enable the
filter once:

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
git config filter.nbstripout.extrakeys metadata.language_info
```

Running the tutorial notebooks then never shows up as a change in git.

The test suite includes golden-file comparisons against the MATLAB
DART_LAB private functions. To (re)generate the reference data, run
`tests/matlab_reference/generate_reference.m` in MATLAB once; the resulting
CSVs are read by `tests/test_matlab_golden.py` (tests skip when the CSVs
are absent).

## Contributing

Contributions are welcome! If you have a feature request, bug report, or a
suggestion, please open an issue on our GitHub repository.

## License

pyDARTLAB is released under the Apache License 2.0. For more details, see the
LICENSE file in the root directory of this source tree or visit
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
