"""The twod_ppi_ensemble app: QCEFF updates with probit (PPI) transforms.

Python version of ``DART_LAB/matlab/twod_ppi_ensemble.m``. The observed
variable (horizontal) is updated with an EAKF or RHF; the increments are
regressed onto a non-negative unobserved variable (vertical) in a
probit-transformed space defined by a selectable continuous distribution
(Normal, Gamma, unbounded RHF, or BNRH bounded at zero). A second panel
shows both ensembles in the transformed space.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from pydartlab.algorithms.increments import obs_increment_eakf, obs_increment_rhf
from pydartlab.algorithms.qceff import ppi_update
from pydartlab.apps._base import ClickEnsembleMixin, DartLabApp, make_figure
from pydartlab.plotting.dists import gaussian_curve
from pydartlab.style import colors

STATE_DIST_LABELS = {
    "Normal": "Normal",
    "Gamma": "Gamma",
    "RHF (unbounded)": "RHF",
    "BNRH (bounded)": "BNRH",
}


class TwodPPIEnsembleApp(DartLabApp, ClickEnsembleMixin):
    def __init__(self, obs_mean: float = 5.0, obs_sd: float = 2.0):
        super().__init__()
        self.fig = make_figure(figsize=(9, 5))
        self.ax_joint = self.fig.add_subplot(121)
        self.ax_ppi = self.fig.add_subplot(122)

        self.members: list[tuple[float, float]] = []

        self.obs_mean = self.float_field(obs_mean, "Obs:")
        self.obs_sd = self.float_field(obs_sd, "Obs SD:", minimum=1e-6)
        self.obs_dist = widgets.RadioButtons(options=("Normal", "RHF"),
                                             description="Observed:")
        self.state_dist = widgets.RadioButtons(options=list(STATE_DIST_LABELS),
                                               description="Unobserved:")
        create_btn = self.button("Create New Ensemble", self.start_ensemble_creation)
        update_btn = self.button("Update Ensemble", self.update_ensemble,
                                 button_style="primary")
        self.stats_html = widgets.HTML()

        controls = widgets.VBox([create_btn, update_btn, self.obs_mean, self.obs_sd,
                                 self.obs_dist, self.state_dist,
                                 self.stats_html, self.status])
        canvas = self.fig.canvas if isinstance(self.fig.canvas, widgets.Widget) else None
        self.widget = widgets.HBox([canvas, controls] if canvas else [controls])

        self.init_click_capture(self.fig, self.ax_joint,
                                value_from_event=self._value_from_click,
                                on_member_added=self._draw_pending,
                                on_finished=self._creation_finished)
        self.draw_base()

    def _value_from_click(self, event):
        if event.ydata is None or event.ydata < 0:
            self.status.value = "Unobserved variable must be non-negative (y >= 0)."
            return None
        return (event.xdata, event.ydata)

    # ---- ensemble creation -------------------------------------------

    def start_ensemble_creation(self):
        self.members = []
        self.draw_base()
        self.status.value = ("Click in the left plot (y >= 0) to place members; "
                             "click outside it to finish.")
        ClickEnsembleMixin.start_ensemble_creation(self)

    def _draw_pending(self, value):
        self.ax_joint.plot(*value, "*", color=colors.prior, markersize=12)
        self.fig.canvas.draw_idle()

    def _creation_finished(self, members):
        self.set_ensemble(members)

    def set_ensemble(self, members):
        """Set the ensemble as (observed, unobserved >= 0) pairs."""
        members = [tuple(m) for m in members]
        if any(m[1] < 0 for m in members):
            raise ValueError("Unobserved variable must be non-negative")
        self.creating = False
        self.members = members
        self.draw_base()
        if members:
            obs_ens, state_ens = self._ensembles()
            self.ax_joint.plot(obs_ens, state_ens, "*", color=colors.prior,
                               markersize=12)
        self.fig.canvas.draw_idle()

    def _ensembles(self):
        arr = np.asarray(self.members, dtype=float)
        return arr[:, 0], arr[:, 1]

    # ---- drawing ------------------------------------------------------

    def draw_base(self):
        self.ax_joint.clear()
        self.ax_ppi.clear()
        self.ax_joint.set_title("Original space", fontsize=10)
        self.ax_joint.set_xlabel("Observed variable")
        self.ax_joint.set_ylabel("Unobserved variable (>= 0)")
        self.ax_joint.axhline(0.0, color="k", lw=1)
        x, y = gaussian_curve(self.obs_mean.value, self.obs_sd.value)
        # Likelihood drawn along the bottom of the joint panel
        self.ax_joint.plot(x, -1.5 + 2.0 * y / y.max(), "--",
                           color=colors.observation, lw=1.5)
        self.ax_joint.set_xlim(self.obs_mean.value - 5 * self.obs_sd.value,
                               self.obs_mean.value + 5 * self.obs_sd.value)
        self.ax_joint.set_ylim(-1.6, 10)
        self.ax_ppi.set_title("Probit (PPI) space", fontsize=10)
        self.ax_ppi.set_xlabel("Observed (probit)")
        self.ax_ppi.set_ylabel("Unobserved (probit)")
        self.fig.canvas.draw_idle()

    def update_ensemble(self):
        if len(self.members) < 2:
            self.status.value = "Create an ensemble (at least 2 members) first."
            return
        obs_ens, state_ens = self._ensembles()
        obs, var = self.obs_mean.value, self.obs_sd.value**2

        if self.obs_dist.value == "Normal":
            obs_incs = obs_increment_eakf(obs_ens, obs, var)
        else:
            obs_incs = obs_increment_rhf(obs_ens, obs, var).increments
        post_obs = obs_ens + obs_incs

        state_dist_type = STATE_DIST_LABELS[self.state_dist.value]
        result = ppi_update(obs_ens, state_ens, post_obs,
                            state_dist_type=state_dist_type,
                            obs_dist_type=self.obs_dist.value)

        self.set_ensemble(self.members)
        self.ax_joint.plot(post_obs, result.post_state, "*",
                           color=colors.posterior, markersize=12)
        for x0, y0, x1, y1 in zip(obs_ens, state_ens, post_obs, result.post_state):
            self.ax_joint.plot([x0, x1], [y0, y1], color="c", lw=0.8)

        self.ax_ppi.plot(result.prior_obs_ppi, result.prior_state_ppi, "*",
                         color=colors.prior, markersize=10, label="Prior")
        self.ax_ppi.plot(result.post_obs_ppi, result.post_state_ppi, "*",
                         color=colors.posterior, markersize=10, label="Posterior")
        self.ax_ppi.legend(fontsize=8)

        bound_note = ""
        if np.any(result.post_state < 0):
            bound_note = (f'<br><b style="color:{colors.observation}">'
                          "Posterior violates the bound at zero!</b>")
        corr = np.corrcoef(obs_ens, state_ens)[0, 1]
        self.stats_html.value = f"Sample correlation = {corr:.3f}{bound_note}"
        self.fig.canvas.draw_idle()
        self.last_result = result
        self.last_posterior = np.column_stack([post_obs, result.post_state])


def twod_ppi_ensemble(**kwargs) -> TwodPPIEnsembleApp:
    """Launch the twod_ppi_ensemble app."""
    return TwodPPIEnsembleApp(**kwargs)
