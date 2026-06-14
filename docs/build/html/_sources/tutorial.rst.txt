Tutorial notebooks
==================

The tutorial lives in the ``notebooks/`` directory of the repository. The
six numbered notebooks replace the six DART_LAB slide sections:

1. ``01_da_concepts_in_1d`` -- Bayes' rule, the product of Gaussians, the
   Kalman filter, and ensemble filters (EAKF, EnKF, RHF) in one dimension.
2. ``02_multivariate_assimilation`` -- regression of observation
   increments onto unobserved variables; Lorenz 63 and Lorenz 96.
3. ``03_inflation_and_localization`` -- rank histograms, variance
   inflation, regression sampling error, and Gaspari-Cohn localization.
4. ``04_nongaussian_qceff`` -- bounded variables, the rank histogram
   filter, quantile conservation, and probit (PPI) transforms.
5. ``05_adaptive_inflation`` -- Bayesian inflation updates, spatially
   varying adaptive inflation, observing networks.
6. ``06_the_real_dart_system`` -- running OSSEs with the Fortran DART
   system, namelist control, QCEFF tables and diagnostics.

``00_getting_started`` checks your environment and introduces the
interactive tools and the color conventions.

Apps
----

Each interactive tool matches a MATLAB DART_LAB app of the same name:

================================  ===================================================
App                               Teaches
================================  ===================================================
``gaussian_product``              product of two Gaussians
``oned_ensemble``                 EAKF / EnKF / RHF updates of a 1-D ensemble
``oned_cycle``                    continuous Kalman filter vs ensemble filters
``oned_model``                    cycling DA with model error
``oned_model_inf``                adaptive inflation in 1-D
``twod_ensemble``                 updating an unobserved variable by regression
``twod_ppi_ensemble``             QCEFF / probit-transformed regression
``bounded_oned_ensemble``         filters for non-negative variables
``run_lorenz_63``                 ensemble DA on the Lorenz 63 attractor
``run_lorenz_96``                 localization and inflation in 40 variables
``run_lorenz_96_inf``             spatially varying adaptive inflation
================================  ===================================================
