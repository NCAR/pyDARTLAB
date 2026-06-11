"""Smoke tests: instantiate every app headless and drive its callbacks."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pydartlab.apps as apps  # noqa: E402


class FakeEvent:
    """Stand-in for a matplotlib mouse event."""

    def __init__(self, ax, x, y):
        self.inaxes = ax
        self.xdata = x
        self.ydata = y


def test_gaussian_product_app():
    app = apps.gaussian_product()
    app.prior_mean.value = 0.0
    app.prior_sd.value = 2.0
    app.obs_mean.value = 2.0
    app.obs_sd.value = 1.0
    app.update()
    mean, sd, weight = app.last_result
    # Posterior mean between prior and obs; spread smaller than both
    assert 0.0 < mean < 2.0
    assert sd < 1.0
    assert weight > 0


@pytest.mark.parametrize("filter_type", ["EAKF", "EnKF", "RHF"])
def test_oned_ensemble_app_update(filter_type):
    app = apps.oned_ensemble()
    app.filter_radio.value = filter_type
    app.set_ensemble([-1.0, 0.2, 1.5, 2.8, 4.0])
    app.update_ensemble()
    assert app.last_posterior.shape == (5,)
    assert np.std(app.last_posterior, ddof=1) < np.std([-1.0, 0.2, 1.5, 2.8, 4.0], ddof=1)


def test_oned_ensemble_app_inflation():
    app = apps.oned_ensemble()
    app.set_ensemble([0.0, 1.0, 2.0, 3.0])
    app.inf_toggle.value = True
    app.inf_slider.value = 3.0
    app.update_ensemble()
    # Inflated posterior has more spread than the plain posterior
    assert (np.std(app.last_inflated_posterior, ddof=1)
            > np.std(app.last_posterior, ddof=1))


def test_oned_ensemble_click_creation():
    app = apps.oned_ensemble()
    app.start_ensemble_creation()
    for x in [0.5, 1.5, 2.5]:
        app._handle_click(FakeEvent(app.ax, x, 0.3))
    # Click outside the axes finishes creation
    app._handle_click(FakeEvent(None, None, None))
    assert app.ens.size == 3
    assert not app.creating


def test_twod_ensemble_app():
    app = apps.twod_ensemble()
    # Correlated joint prior
    members = [(2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 8.5), (5.0, 6.0)]
    app.set_ensemble(members)
    app.update_ensemble()
    post = app.last_posterior
    assert post.shape == (5, 2)
    obs_ens = np.array([m[0] for m in members])
    # Observed-variable posterior pulled toward the observation (5.0)
    assert abs(post[:, 0].mean() - 5.0) <= abs(obs_ens.mean() - 5.0) + 1e-12


def test_twod_ensemble_click_creation():
    app = apps.twod_ensemble()
    app.start_ensemble_creation()
    app._handle_click(FakeEvent(app.ax_joint, 3.0, 4.0))
    app._handle_click(FakeEvent(app.ax_joint, 5.0, 6.0))
    app._handle_click(FakeEvent(None, None, None))
    assert len(app.members) == 2


@pytest.mark.parametrize("filter_type", ["EAKF", "EnKF", "RHF"])
def test_oned_cycle_app(filter_type):
    app = apps.oned_cycle()
    app.filter_radio.value = filter_type
    app.set_ensemble([-1.0, 0.5, 1.0, 2.0, 3.5])
    for _ in range(5):
        app.cycle()
    assert len(app.experiment.history) == 5
    # With G=1 cycling shrinks the KF sd
    assert app.experiment.kf_sd < 2.0


def test_oned_cycle_needs_ensemble():
    app = apps.oned_cycle()
    app.cycle()
    assert "Create an ensemble" in app.status.value
