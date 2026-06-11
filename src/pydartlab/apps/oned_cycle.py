"""The oned_cycle app: continuous Kalman filter vs ensemble filters.

Python version of ``DART_LAB/matlab/oned_cycle.m``. A scalar state with a
linear growth model is cycled: each press of "Cycle DA" assimilates an
observation of the (zero) truth and advances both a continuous Kalman filter
(top panel) and the ensemble filter (bottom panel).
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from pydartlab.apps._base import ClickEnsembleMixin, DartLabApp, fmt_stats, make_figure
from pydartlab.experiments.kalman_cycle import KalmanCycle
from pydartlab.plotting.dists import gaussian_curve
from pydartlab.style import colors

PRIOR_Y, POST_Y = 0.10, 0.04


class OnedCycleApp(DartLabApp, ClickEnsembleMixin):
    def __init__(self):
        super().__init__()
        self.fig = make_figure(figsize=(8, 6))
        self.ax_kf, self.ax_ens = self.fig.subplots(2, 1, sharex=True)

        self.growth_rate = self.float_field(1.0, "Growth G:", step=0.1)
        self.obs_sd = self.float_field(1.0, "Obs SD:", minimum=1e-6)
        self.filter_radio = self.filter_selector()
        create_btn = self.button("Create New Ensemble", self.start_ensemble_creation)
        cycle_btn = self.button("Cycle DA", self.cycle, button_style="primary")
        self.stats_html = widgets.HTML()

        controls = widgets.VBox([create_btn, cycle_btn, self.growth_rate, self.obs_sd,
                                 self.filter_radio, self.stats_html, self.status])
        canvas = self.fig.canvas if isinstance(self.fig.canvas, widgets.Widget) else None
        self.widget = widgets.HBox([canvas, controls] if canvas else [controls])

        self.experiment = KalmanCycle()
        self.init_click_capture(self.fig, self.ax_ens,
                                value_from_event=lambda ev: ev.xdata,
                                on_member_added=self._draw_pending,
                                on_finished=self._creation_finished)
        self.draw_base()

    def _make_experiment(self):
        return KalmanCycle(growth_rate=self.growth_rate.value,
                           obs_error_sd=self.obs_sd.value,
                           filter_type=self.filter_radio.value)

    # ---- ensemble creation -------------------------------------------

    def start_ensemble_creation(self):
        self.experiment = self._make_experiment()
        self.draw_base()
        self.status.value = ("Click in the lower plot to place members; "
                             "click outside it to finish.")
        ClickEnsembleMixin.start_ensemble_creation(self)

    def _draw_pending(self, value):
        self.ax_ens.plot(value, PRIOR_Y, "*", color=colors.prior, markersize=12)
        self.fig.canvas.draw_idle()

    def _creation_finished(self, members):
        self.set_ensemble(members)

    def set_ensemble(self, members):
        """Set the ensemble programmatically (mouse-free path)."""
        self.creating = False
        self.experiment = self._make_experiment()
        self.experiment.set_ensemble(np.asarray(members, dtype=float))
        self.draw_base()
        ens = self.experiment.ensemble
        if ens is not None and ens.size:
            self.ax_ens.plot(ens, np.full(ens.size, PRIOR_Y), "*",
                             color=colors.prior, markersize=12)
            self.status.value = f"Ensemble of {ens.size} members; press Cycle DA."
        self.fig.canvas.draw_idle()

    # ---- drawing ------------------------------------------------------

    def draw_base(self):
        for ax, title in [(self.ax_kf, "Continuous Kalman filter"),
                          (self.ax_ens, f"Ensemble filter ({self.filter_radio.value})")]:
            ax.clear()
            ax.set_title(title, fontsize=10)
            ax.set_ylim(-0.05, 1.0)
        self.ax_kf.set_xlim(-4, 4)
        self.fig.canvas.draw_idle()

    def cycle(self):
        if self.experiment.ensemble is None:
            self.status.value = "Create an ensemble first."
            return
        # Push current control values into the experiment
        self.experiment.growth_rate = self.growth_rate.value
        self.experiment.obs_error_sd = self.obs_sd.value
        self.experiment.filter_type = self.filter_radio.value

        rec = self.experiment.cycle()
        self.draw_base()

        # Top: continuous KF prior (green), likelihood (red), posterior (blue)
        for mean, sd, color, style, label in [
                (rec.kf_prior_mean, rec.kf_prior_sd, colors.prior, "-", "Prior"),
                (rec.observation, self.experiment.obs_error_sd, colors.observation,
                 "--", "Likelihood"),
                (rec.kf_post_mean, rec.kf_post_sd, colors.posterior, "-", "Posterior")]:
            x, y = gaussian_curve(mean, sd)
            self.ax_kf.plot(x, y, style, color=color, lw=2, label=label)
        self.ax_kf.plot(rec.observation, 0, "*", color=colors.observation,
                        markersize=14)
        self.ax_kf.legend(loc="upper right", fontsize=8)

        # Bottom: ensemble members, fitted prior/posterior, likelihood
        self.ax_ens.plot(rec.prior_ens, np.full(rec.prior_ens.size, PRIOR_Y), "*",
                         color=colors.prior, markersize=12)
        self.ax_ens.plot(rec.post_ens, np.full(rec.post_ens.size, POST_Y), "*",
                         color=colors.posterior, markersize=12)
        for p, q in zip(rec.prior_ens, rec.post_ens):
            self.ax_ens.plot([p, q], [PRIOR_Y, POST_Y], color="0.6", lw=0.8)
        x, y = gaussian_curve(rec.prior_ens.mean(), rec.prior_ens.std(ddof=1))
        self.ax_ens.plot(x, y, color=colors.prior, lw=2)
        x, y = gaussian_curve(rec.post_ens.mean(), rec.post_ens.std(ddof=1))
        self.ax_ens.plot(x, y, color=colors.posterior, lw=2)
        x, y = gaussian_curve(rec.observation, self.experiment.obs_error_sd)
        self.ax_ens.plot(x, y, "--", color=colors.observation, lw=2)

        lims = (min(rec.kf_prior_mean - 4 * rec.kf_prior_sd, rec.prior_ens.min() - 1),
                max(rec.kf_prior_mean + 4 * rec.kf_prior_sd, rec.prior_ens.max() + 1))
        self.ax_kf.set_xlim(lims)

        self.stats_html.value = "<br>".join([
            f'<span style="color:{colors.prior}">KF prior: mean = '
            f"{rec.kf_prior_mean:.3f}, sd = {rec.kf_prior_sd:.3f}</span>",
            f'<span style="color:{colors.posterior}">KF posterior: mean = '
            f"{rec.kf_post_mean:.3f}, sd = {rec.kf_post_sd:.3f}</span>",
            fmt_stats("Ens prior", rec.prior_ens, colors.prior),
            fmt_stats("Ens posterior", rec.post_ens, colors.posterior),
        ])
        self.fig.canvas.draw_idle()


def oned_cycle() -> OnedCycleApp:
    """Launch the oned_cycle app."""
    return OnedCycleApp()
