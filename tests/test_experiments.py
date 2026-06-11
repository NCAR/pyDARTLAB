import numpy as np
import pytest

from pydartlab.experiments import (
    KalmanCycle,
    Lorenz63Experiment,
    Lorenz96Experiment,
    OneDExperiment,
)


def test_kalman_cycle_eakf_tracks_continuous_kf():
    # With G=1 and an EAKF, the ensemble mean/sd follow the continuous KF
    cycle = KalmanCycle(growth_rate=1.0, obs_error_sd=1.0, filter_type="EAKF", seed=5)
    ens = np.random.default_rng(0).normal(1.0, 2.0, 100)
    # Match the KF prior to the ensemble prior
    cycle.kf_mean, cycle.kf_sd = ens.mean(), ens.std(ddof=1)
    cycle.set_ensemble(ens)
    for _ in range(10):
        rec = cycle.cycle()
        assert rec.post_ens.mean() == pytest.approx(rec.kf_post_mean, abs=1e-10)
        assert rec.post_ens.std(ddof=1) == pytest.approx(rec.kf_post_sd, abs=1e-10)


def test_kalman_cycle_sd_shrinks_without_growth():
    cycle = KalmanCycle(growth_rate=1.0, obs_error_sd=1.0, seed=1)
    cycle.set_ensemble(np.array([0.0, 1.0, 2.0, 3.0]))
    sds = [cycle.kf_sd]
    for _ in range(20):
        cycle.cycle()
        sds.append(cycle.kf_sd)
    assert all(b < a for a, b in zip(sds, sds[1:]))
    assert sds[-1] < 0.3


def test_kalman_cycle_requires_ensemble():
    cycle = KalmanCycle()
    with pytest.raises(RuntimeError):
        cycle.cycle()


def test_oned_experiment_filter_keeps_error_bounded():
    exp = OneDExperiment(ens_size=10, filter_type="EAKF", seed=3)
    for _ in range(100):
        exp.step()
    # Model doubles state each step; without assimilation the ensemble mean
    # would blow up. With assimilation errors stay O(1).
    assert np.max(np.abs(exp.history["post_error"])) < 5.0
    assert len(exp.history["time"]) == 100
    # Both rank histograms accumulated one entry per cycle
    assert exp.prior_rank_hist.counts.sum() == 100
    assert exp.post_rank_hist.counts.sum() == 100


def test_oned_experiment_alternation_enforced():
    exp = OneDExperiment(seed=0)
    exp.advance()
    with pytest.raises(RuntimeError):
        exp.advance()
    exp.assimilate()
    with pytest.raises(RuntimeError):
        exp.assimilate()


def test_oned_experiment_bias_underdisperses_without_inflation():
    # With model bias, the truth falls outside the ensemble most of the time:
    # the rank histogram piles up in the extreme bins
    exp = OneDExperiment(ens_size=10, filter_type="EAKF", model_bias=1.0, seed=4)
    for _ in range(300):
        exp.step()
    counts = exp.prior_rank_hist.counts
    edge_fraction = (counts[0] + counts[-1]) / counts.sum()
    assert edge_fraction > 0.3


def test_oned_experiment_adaptive_inflation_rises_with_bias():
    exp = OneDExperiment(ens_size=10, filter_type="EAKF", model_bias=1.0,
                         adaptive_inflation=True, inflation_sd=0.6,
                         inflation_sd_min=0.6, inflation_max=10.0, seed=8)
    for _ in range(200):
        exp.step()
    assert exp.inflation > 1.2
    # Inflation reduces the edge pile-up relative to the no-inflation case
    counts = exp.prior_rank_hist.counts
    edge_fraction = (counts[0] + counts[-1]) / counts.sum()
    assert edge_fraction < 0.3


def test_oned_experiment_ens_size_change():
    exp = OneDExperiment(ens_size=4, seed=0)
    exp.set_ens_size(8)
    assert exp.ens.size == 8
    exp.set_ens_size(3)
    assert exp.ens.size == 3


def test_lorenz63_assimilation_beats_free_run():
    free = Lorenz63Experiment(filter_type="No Assimilation", seed=9)
    assim = Lorenz63Experiment(filter_type="EAKF", seed=9)
    for _ in range(200):
        free.step()
        assim.step()

    def mean_error(exp):
        errs = [np.sqrt(np.mean((p.mean(axis=0) - t) ** 2))
                for p, t in zip(exp.history["posterior"][-100:],
                                exp.history["truth"][-100:])]
        return np.mean(errs)

    assert mean_error(assim) < mean_error(free)


def test_lorenz96_free_run_error_saturates():
    exp = Lorenz96Experiment(filter_type="No Assimilation", seed=2)
    for _ in range(200):
        exp.step()
    early = np.mean(exp.history["prior_error"][:20])
    late = np.mean(exp.history["prior_error"][-50:])
    assert late > early  # perturbations grow from tiny initial values
    assert late < 20.0   # but stay bounded on the attractor


def test_lorenz96_eakf_with_localization_beats_free_run():
    free = Lorenz96Experiment(filter_type="No Assimilation", seed=6)
    assim = Lorenz96Experiment(filter_type="EAKF", localization=0.2,
                               inflation=1.0, seed=6)
    for _ in range(120):
        free.step()
        assim.step()
    assert (np.mean(assim.history["post_error"][-40:])
            < 0.5 * np.mean(free.history["post_error"][-40:]))


def test_lorenz96_adaptive_inflation_responds_to_model_error():
    exp = Lorenz96Experiment(filter_type="EAKF", localization=0.2,
                             adaptive_inflation=True, inflation_sd=0.6,
                             inflation_damping=1.0, forcing=12.0,  # truth uses 8
                             obs_error_sd=1.0, seed=12)
    for _ in range(100):
        exp.step()
    # Model error should drive inflation above 1 once spun up
    assert np.mean(exp.history["inflate_mean"][-30:]) > 1.01
    assert exp.inflate.max() > 1.1
    assert np.all(exp.inflate >= exp.inflation_min)
    # Inflation is spatially varying
    assert np.std(exp.history["inflate"][-1]) > 0.0


def test_lorenz96_obs_networks():
    exp = Lorenz96Experiment(obs_network="1:40:4", seed=0)
    assert exp.set_obs_locations().size == 10
    exp.obs_network = "bogus"
    with pytest.raises(ValueError):
        exp.set_obs_locations()


def test_lorenz96_partial_network_runs():
    exp = Lorenz96Experiment(filter_type="EAKF", localization=0.3,
                             obs_network="1:20", seed=1)
    for _ in range(10):
        exp.step()
    obs = exp.last_obs
    assert np.isfinite(obs[1:21]).all()
    assert np.isnan(obs[25])
