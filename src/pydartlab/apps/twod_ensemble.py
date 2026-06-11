"""The twod_ensemble app: observing one variable updates another.

Python version of ``DART_LAB/matlab/twod_ensemble.m``. Create a bivariate
ensemble by clicking in the joint plot: the horizontal axis is the observed
variable, the vertical axis an unobserved state variable. Updating shows how
observation increments are regressed onto the unobserved variable through
the joint prior.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from pydartlab.algorithms.increments import obs_increment
from pydartlab.algorithms.regression import get_state_increments
from pydartlab.apps._base import ClickEnsembleMixin, DartLabApp, make_figure
from pydartlab.plotting.dists import gaussian_curve
from pydartlab.style import colors


class TwodEnsembleApp(DartLabApp, ClickEnsembleMixin):
    def __init__(self, obs_mean: float = 5.0, obs_sd: float = 2.0):
        super().__init__()
        self.fig = make_figure(figsize=(8, 7))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                                   hspace=0.08, wspace=0.08)
        self.ax_top = self.fig.add_subplot(gs[0, 0])     # marginal: observed
        self.ax_joint = self.fig.add_subplot(gs[1, 0])   # joint distribution
        self.ax_right = self.fig.add_subplot(gs[1, 1])   # marginal: unobserved
        self.ax_top.set_xticks([])
        self.ax_right.set_yticks([])

        self.members: list[tuple[float, float]] = []
        self.rng = np.random.default_rng()

        self.obs_mean = self.float_field(obs_mean, "Obs:")
        self.obs_sd = self.float_field(obs_sd, "Obs SD:", minimum=1e-6)
        self.filter_radio = self.filter_selector()
        create_btn = self.button("Create New Ensemble", self.start_ensemble_creation)
        update_btn = self.button("Update Ensemble", self.update_ensemble,
                                 button_style="primary")
        self.stats_html = widgets.HTML()

        controls = widgets.VBox([create_btn, update_btn, self.obs_mean, self.obs_sd,
                                 self.filter_radio, self.stats_html, self.status])
        canvas = self.fig.canvas if isinstance(self.fig.canvas, widgets.Widget) else None
        self.widget = widgets.HBox([canvas, controls] if canvas else [controls])

        self.init_click_capture(self.fig, self.ax_joint,
                                value_from_event=lambda ev: (ev.xdata, ev.ydata),
                                on_member_added=self._draw_pending,
                                on_finished=self._creation_finished)
        self.draw_base()

    # ---- ensemble creation -------------------------------------------

    def start_ensemble_creation(self):
        self.members = []
        self.draw_base()
        self.status.value = ("Click in the joint plot to place members; "
                             "click outside it to finish.")
        ClickEnsembleMixin.start_ensemble_creation(self)

    def _draw_pending(self, value):
        x, y = value
        self.ax_joint.plot(x, y, "*", color=colors.prior, markersize=12)
        self.fig.canvas.draw_idle()

    def _creation_finished(self, members):
        self.set_ensemble(members)

    def set_ensemble(self, members):
        """Set the ensemble as a sequence of (observed, unobserved) pairs."""
        self.creating = False
        self.members = [tuple(m) for m in members]
        self.draw_base()
        if not self.members:
            return
        obs_ens, state_ens = self._ensembles()
        self.ax_joint.plot(obs_ens, state_ens, "*", color=colors.prior, markersize=12)
        self._draw_marginals(obs_ens, state_ens, colors.prior)
        if obs_ens.size >= 2:
            corr = np.corrcoef(obs_ens, state_ens)[0, 1]
            self.stats_html.value = f"Sample correlation = {corr:.3f}"
            # Least squares best-fit line through the joint prior
            slope = (np.cov(obs_ens, state_ens, ddof=1)[0, 1]
                     / np.var(obs_ens, ddof=1))
            xfit = np.array([obs_ens.min(), obs_ens.max()])
            self.ax_joint.plot(xfit, state_ens.mean() + slope * (xfit - obs_ens.mean()),
                               color="0.6", lw=1)
        self.fig.canvas.draw_idle()

    def _ensembles(self):
        arr = np.asarray(self.members, dtype=float)
        return arr[:, 0], arr[:, 1]

    # ---- drawing ------------------------------------------------------

    def draw_base(self):
        for ax in (self.ax_top, self.ax_joint, self.ax_right):
            ax.clear()
        self.ax_top.set_xticks([])
        self.ax_right.set_yticks([])
        self.ax_joint.set_xlabel("Observed variable")
        self.ax_joint.set_ylabel("Unobserved state variable")

        # Observation likelihood on the top marginal
        x, y = gaussian_curve(self.obs_mean.value, self.obs_sd.value)
        self.ax_top.plot(x, y, "--", color=colors.observation, lw=2)
        self.ax_top.plot(self.obs_mean.value, 0, "*", color=colors.observation,
                         markersize=14, clip_on=False)
        self.ax_joint.set_xlim(self.obs_mean.value - 5 * self.obs_sd.value,
                               self.obs_mean.value + 5 * self.obs_sd.value)
        self.ax_joint.set_ylim(0, 10)
        self._sync_marginal_limits()
        self.fig.canvas.draw_idle()

    def _sync_marginal_limits(self):
        self.ax_top.set_xlim(self.ax_joint.get_xlim())
        self.ax_right.set_ylim(self.ax_joint.get_ylim())

    def _draw_marginals(self, obs_ens, state_ens, color):
        y_obs = 0.0
        self.ax_top.plot(obs_ens, np.full(obs_ens.size, y_obs), "*",
                         color=color, markersize=9, clip_on=False)
        self.ax_right.plot(np.zeros(state_ens.size), state_ens, "*",
                           color=color, markersize=9, clip_on=False)
        if obs_ens.size >= 2:
            x, y = gaussian_curve(obs_ens.mean(), obs_ens.std(ddof=1))
            self.ax_top.plot(x, y, color=color, lw=1.5)
            x, y = gaussian_curve(state_ens.mean(), state_ens.std(ddof=1))
            self.ax_right.plot(y, x, color=color, lw=1.5)
        self._sync_marginal_limits()

    def update_ensemble(self):
        if len(self.members) < 2:
            self.status.value = "Create an ensemble (at least 2 members) first."
            return
        obs_ens, state_ens = self._ensembles()
        obs_incs = obs_increment(obs_ens, self.obs_mean.value, self.obs_sd.value**2,
                                 self.filter_radio.value, rng=self.rng)
        state_incs, _ = get_state_increments(state_ens, obs_ens, obs_incs)
        post_obs = obs_ens + obs_incs
        post_state = state_ens + state_incs

        self.set_ensemble(self.members)  # redraw prior
        self.ax_joint.plot(post_obs, post_state, "*", color=colors.posterior,
                           markersize=12)
        for x0, y0, x1, y1 in zip(obs_ens, state_ens, post_obs, post_state):
            self.ax_joint.plot([x0, x1], [y0, y1], color="c", lw=0.8)
        self._draw_marginals(post_obs, post_state, colors.posterior)
        self.fig.canvas.draw_idle()

        self.last_posterior = np.column_stack([post_obs, post_state])
        corr = np.corrcoef(obs_ens, state_ens)[0, 1]
        self.stats_html.value = f"Sample correlation = {corr:.3f}"


def twod_ensemble(**kwargs) -> TwodEnsembleApp:
    """Launch the twod_ensemble app."""
    return TwodEnsembleApp(**kwargs)
