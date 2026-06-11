"""The oned_ensemble app: how EAKF, EnKF and RHF update a 1-D ensemble.

Python version of ``DART_LAB/matlab/oned_ensemble.m``. Create a prior
ensemble by clicking in the plot (or with ``set_ensemble``), choose a
filter, and update. Optionally inflate the prior before the update; the
inflated ensemble and its posterior are shown on a second row.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from pydartlab.algorithms.increments import (
    obs_increment_eakf,
    obs_increment_enkf,
    obs_increment_rhf,
)
from pydartlab.algorithms.inflation import inflate_ensemble
from pydartlab.apps._base import ClickEnsembleMixin, DartLabApp, fmt_stats, make_figure
from pydartlab.plotting.dists import gaussian_curve
from pydartlab.style import colors

PRIOR_Y, POST_Y = 0.4, 0.25
INF_PRIOR_Y, INF_POST_Y = 0.1, -0.05


class OnedEnsembleApp(DartLabApp, ClickEnsembleMixin):
    def __init__(self, obs_mean: float = 1.0, obs_sd: float = 1.0):
        super().__init__()
        self.fig = make_figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(111)
        self.ens = np.array([])
        self.rng = np.random.default_rng()

        self.obs_mean = self.float_field(obs_mean, "Obs:")
        self.obs_sd = self.float_field(obs_sd, "Obs SD:", minimum=1e-6)
        self.filter_radio = self.filter_selector()
        self.inf_toggle, self.inf_slider = self.inflation_controls()
        create_btn = self.button("Create New Ensemble", self.start_ensemble_creation)
        update_btn = self.button("Update Ensemble", self.update_ensemble,
                                 button_style="primary")
        self.stats_html = widgets.HTML()

        controls = widgets.VBox([create_btn, update_btn, self.obs_mean, self.obs_sd,
                                 self.filter_radio, self.inf_toggle, self.inf_slider,
                                 self.stats_html, self.status])
        canvas = self.fig.canvas if isinstance(self.fig.canvas, widgets.Widget) else None
        self.widget = widgets.HBox([canvas, controls] if canvas else [controls])

        self.init_click_capture(self.fig, self.ax,
                                value_from_event=lambda ev: ev.xdata,
                                on_member_added=self._draw_pending,
                                on_finished=self._creation_finished)
        self.draw_base()

    # ---- ensemble creation -------------------------------------------

    def start_ensemble_creation(self):
        self.ens = np.array([])
        self.draw_base()
        self.status.value = ("Click in the plot to place ensemble members; "
                             "click outside the axes to finish.")
        ClickEnsembleMixin.start_ensemble_creation(self)

    def _draw_pending(self, value):
        self.ax.plot(value, PRIOR_Y, "*", color=colors.prior, markersize=12)
        self.fig.canvas.draw_idle()

    def _creation_finished(self, members):
        self.set_ensemble(members)

    def set_ensemble(self, members):
        """Set the prior ensemble programmatically (mouse-free path)."""
        self.creating = False
        self.ens = np.asarray(members, dtype=float)
        self.status.value = f"Prior ensemble of {self.ens.size} members."
        self.draw_base()
        if self.ens.size:
            self.ax.plot(self.ens, np.full(self.ens.size, PRIOR_Y), "*",
                         color=colors.prior, markersize=12)
            self.stats_html.value = fmt_stats("Prior", self.ens, colors.prior)
        self.fig.canvas.draw_idle()

    # ---- drawing ------------------------------------------------------

    def draw_base(self):
        self.ax.clear()
        x, y = gaussian_curve(self.obs_mean.value, self.obs_sd.value)
        self.ax.plot(x, y, "--", color=colors.observation, lw=2, label="Likelihood")
        self.ax.plot(self.obs_mean.value, 0, "*", color=colors.observation,
                     markersize=14)
        self.ax.axhline(PRIOR_Y, color="0.85", lw=0.5)
        self.ax.set_ylim(-0.2, max(1.0, y.max() * 1.1))
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()

    def _increments(self, ens):
        obs, var = self.obs_mean.value, self.obs_sd.value**2
        kind = self.filter_radio.value
        if kind == "EAKF":
            return obs_increment_eakf(ens, obs, var), None
        if kind == "EnKF":
            return obs_increment_enkf(ens, obs, var, rng=self.rng), None
        result = obs_increment_rhf(ens, obs, var)
        return result.increments, result

    def update_ensemble(self):
        if self.ens.size < 2:
            self.status.value = "Create an ensemble (at least 2 members) first."
            return
        ens = self.ens
        incs, rhf = self._increments(ens)
        post = ens + incs

        self.set_ensemble(ens)  # redraw base + prior
        # Posterior members and connecting segments
        self.ax.plot(post, np.full(post.size, POST_Y), "*",
                     color=colors.posterior, markersize=12)
        for p, q in zip(ens, post):
            self.ax.plot([p, q], [PRIOR_Y, POST_Y], color="0.6", lw=0.8)

        # Continuous prior/posterior distributions
        if rhf is None:
            x, y = gaussian_curve(ens.mean(), ens.std(ddof=1))
            self.ax.plot(x, y, color=colors.prior, lw=2)
            x, y = gaussian_curve(post.mean(), post.std(ddof=1))
            self.ax.plot(x, y, color=colors.posterior, lw=2)
        else:
            self.ax.plot(*rhf.prior_pdf, color=colors.prior, lw=2)
            self.ax.plot(*rhf.post_pdf, color=colors.posterior, lw=2)

        stats = [fmt_stats("Prior", ens, colors.prior),
                 fmt_stats("Posterior", post, colors.posterior)]

        # Inflation branch: inflate the prior, update the inflated ensemble
        if self.inf_toggle.value and self.inf_slider.value > 1.0:
            inf_ens = inflate_ensemble(ens, self.inf_slider.value)
            inf_incs, _ = self._increments(inf_ens)
            inf_post = inf_ens + inf_incs
            self.ax.plot(inf_ens, np.full(inf_ens.size, INF_PRIOR_Y), "*",
                         color=colors.inflated, markersize=12)
            self.ax.plot(inf_post, np.full(inf_post.size, INF_POST_Y), "*",
                         color=colors.posterior, markersize=8)
            for p, q in zip(inf_ens, inf_post):
                self.ax.plot([p, q], [INF_PRIOR_Y, INF_POST_Y], color="0.8", lw=0.8)
            stats += [fmt_stats("Inflated prior", inf_ens, colors.inflated),
                      fmt_stats("Inflated posterior", inf_post, colors.posterior)]
            self.last_inflated_posterior = inf_post

        self.stats_html.value = "<br>".join(stats)
        self.fig.canvas.draw_idle()
        self.last_posterior = post


def oned_ensemble(**kwargs) -> OnedEnsembleApp:
    """Launch the oned_ensemble app."""
    return OnedEnsembleApp(**kwargs)
