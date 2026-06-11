"""Smoke tests: instantiate every app headless and drive its callbacks."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import pydartlab.apps as apps  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


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


def test_oned_model_app_cycles():
    app = apps.oned_model(seed=1)
    for _ in range(10):
        app.single_step()
    assert app.experiment.time == 5
    assert app.experiment.prior_rank_hist.counts.sum() == 5


def test_oned_model_inf_app_adaptive():
    app = apps.oned_model_inf(seed=1)
    app.inf_mode.value = "Adaptive Inflation"
    app.model_bias.value = 1.0
    for _ in range(40):
        app.single_step()
    assert app.experiment.adaptive_inflation
    assert app.experiment.inflation >= 1.0


def test_oned_model_ens_size_change():
    app = apps.oned_model(seed=1)
    app.ens_size.value = 10
    assert app.experiment.ens.size == 10


def test_run_lorenz_63_app():
    app = apps.run_lorenz_63(seed=1)
    app.filter_radio.value = "EAKF"
    for _ in range(6):
        app.single_step()
    assert app.experiment.time == 3


def test_run_lorenz_96_app():
    app = apps.run_lorenz_96(seed=1)
    app.filter_dd.value = "EAKF"
    app.localization.value = 0.2
    for _ in range(6):
        app.single_step()
    assert app.experiment.time == 3
    assert len(app.experiment.history["post_error"]) == 3


def test_run_lorenz_96_inf_app():
    app = apps.run_lorenz_96_inf(seed=1)
    app.filter_dd.value = "EAKF"
    app.inf_mode.value = "Adaptive Inflation"
    app.obs_network.value = "1:40:2"
    for _ in range(4):
        app.single_step()
    assert app.experiment.adaptive_inflation
    obs = app.experiment.last_obs
    assert np.isnan(obs[0]) and np.isfinite(obs[1])


@pytest.mark.parametrize("filter_type", ["EAKF", "Gamma", "BNRHF"])
def test_bounded_oned_ensemble_app(filter_type):
    app = apps.bounded_oned_ensemble()
    app.filter_radio.value = filter_type
    app.set_ensemble([0.4, 0.9, 1.4, 2.1, 3.0])
    app.update_ensemble()
    assert app.last_posterior.shape == (5,)
    if filter_type in ("Gamma", "BNRHF"):
        assert np.all(app.last_posterior >= 0.0)


def test_bounded_oned_ensemble_inflation_respects_bound():
    app = apps.bounded_oned_ensemble()
    app.filter_radio.value = "BNRHF"
    app.set_ensemble([0.1, 0.4, 0.9, 1.6, 2.5])
    app.inf_toggle.value = True
    app.inf_slider.value = 3.0
    app.update_ensemble()
    assert np.all(app.last_inflated_prior >= 0.0)
    assert np.all(app.last_inflated_posterior >= 0.0)


def test_bounded_oned_ensemble_rejects_negative():
    app = apps.bounded_oned_ensemble()
    with pytest.raises(ValueError):
        app.set_ensemble([-0.5, 1.0])


@pytest.mark.parametrize("state_dist", ["Normal", "Gamma", "RHF (unbounded)",
                                        "BNRH (bounded)"])
@pytest.mark.parametrize("obs_dist", ["Normal", "RHF"])
def test_twod_ppi_ensemble_app(state_dist, obs_dist):
    app = apps.twod_ppi_ensemble()
    app.state_dist.value = state_dist
    app.obs_dist.value = obs_dist
    members = [(2.0, 0.5), (4.0, 1.5), (6.0, 3.0), (8.0, 5.0), (5.0, 2.0),
               (3.0, 1.0), (7.0, 4.0)]
    app.set_ensemble(members)
    app.update_ensemble()
    assert app.last_posterior.shape == (7, 2)
    if state_dist in ("Gamma", "BNRH (bounded)"):
        assert np.all(app.last_posterior[:, 1] >= 0.0)
