"""The run_lorenz_96 / run_lorenz_96_inf apps: DA on a 40-variable model.

Python versions of ``DART_LAB/matlab/run_lorenz_96.m`` and
``run_lorenz_96_inf.m``. The state panel shows the truth and all ensemble
members against variable index (the MATLAB tools draw the same data on a
polar ring). The error/spread panel and rank histograms track filter
performance; the adaptive app adds the spatially varying inflation field.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from pydartlab.apps._base import DartLabApp, make_figure
from pydartlab.experiments.lorenz96_da import OBS_NETWORKS, Lorenz96Experiment
from pydartlab.plotting.ensembles import plot_rank_histogram
from pydartlab.style import colors


class RunLorenz96App(DartLabApp):
    def __init__(self, adaptive: bool = False, seed: int | None = None):
        super().__init__()
        self.adaptive = adaptive
        self.seed = seed

        nrows = 3 if adaptive else 2
        self.fig = make_figure(figsize=(9, 2.6 * nrows))
        gs = self.fig.add_gridspec(nrows, 2, hspace=0.5, wspace=0.25)
        self.ax_state = self.fig.add_subplot(gs[0, 0])
        self.ax_err = self.fig.add_subplot(gs[0, 1])
        self.ax_prior_rank = self.fig.add_subplot(gs[1, 0])
        self.ax_post_rank = self.fig.add_subplot(gs[1, 1])
        self.ax_inflation = self.fig.add_subplot(gs[2, :]) if adaptive else None

        self.step_button = self.button("Advance Model", self.single_step,
                                       button_style="primary")
        self.play = widgets.Play(value=0, min=0, max=100_000, interval=200,
                                 description="Auto run")
        self.play.observe(lambda ch: self.single_step(), names="value")
        self.filter_dd = widgets.Dropdown(
            options=("No Assimilation", "EAKF", "EnKF", "RHF"),
            description="Filter:")
        self.ens_size = widgets.BoundedIntText(value=20, min=4, max=200,
                                               description="Ens size:",
                                               layout=widgets.Layout(width="180px"))
        self.localization = self.float_field(1.0, "Localization:", minimum=1e-6)
        self.inflation = self.float_field(1.0, "Inflation:", minimum=0.0)
        self.forcing = self.float_field(8.0, "Forcing:", step=0.5, minimum=4.0)

        items = [self.step_button, self.play, self.filter_dd, self.ens_size,
                 self.localization, self.inflation, self.forcing]
        if adaptive:
            self.inf_mode = widgets.RadioButtons(
                options=["Fixed Inflation", "Adaptive Inflation"],
                description="Inflation:")
            self.inf_flavor = widgets.RadioButtons(
                options=["Gaussian", "I-Gamma"], description="Flavor:")
            self.inf_damping = self.float_field(0.9, "Damping:", step=0.05, minimum=0.0)
            self.inf_sd = self.float_field(0.6, "Inf SD:", minimum=1e-6)
            self.inf_min = self.float_field(1.0, "Inf min:", minimum=0.0)
            self.obs_network = widgets.Dropdown(options=list(OBS_NETWORKS),
                                                description="Obs network:")
            items += [self.inf_mode, self.inf_flavor, self.inf_damping,
                      self.inf_sd, self.inf_min, self.obs_network]

        reset_btn = self.button("Reset", self.reset)
        self.time_html = widgets.HTML()
        items += [reset_btn, self.time_html, self.status]
        controls = widgets.VBox(items)
        canvas = self.fig.canvas if isinstance(self.fig.canvas, widgets.Widget) else None
        self.widget = widgets.HBox([canvas, controls] if canvas else [controls])
        self.reset()

    # ---- experiment wiring --------------------------------------------

    def reset(self):
        self.experiment = Lorenz96Experiment(
            ens_size=self.ens_size.value,
            filter_type=self.filter_dd.value,
            localization=self.localization.value,
            inflation=self.inflation.value,
            forcing=self.forcing.value,
            obs_network=self.obs_network.value if self.adaptive else "1:40:1",
            seed=self.seed)
        for ax in self.fig.axes:
            ax.clear()
        self.step_button.description = "Advance Model"
        self.time_html.value = "Time = 0"
        self.status.value = ("It takes ~20 steps for the tiny initial "
                             "perturbations to grow (spin-up).")
        self.fig.canvas.draw_idle()

    def _push_config(self):
        exp = self.experiment
        exp.filter_type = self.filter_dd.value
        exp.localization = self.localization.value
        exp.inflation = self.inflation.value
        exp.assim_model.forcing = self.forcing.value
        if self.adaptive:
            exp.adaptive_inflation = self.inf_mode.value == "Adaptive Inflation"
            exp.inflation_flavor = self.inf_flavor.value
            exp.inflation_damping = self.inf_damping.value
            exp.inflation_sd = self.inf_sd.value
            exp.inflation_min = self.inf_min.value
            exp.obs_network = self.obs_network.value

    def single_step(self):
        self._push_config()
        exp = self.experiment
        if exp.ready_to_advance:
            exp.advance()
            self.step_button.description = "Assimilate Obs"
        else:
            exp.assimilate()
            self.step_button.description = "Advance Model"
        self.redraw()

    # ---- drawing ------------------------------------------------------

    def redraw(self):
        exp = self.experiment
        h = exp.history
        idx = np.arange(exp.model_size)

        ax = self.ax_state
        ax.clear()
        ax.set_title("State (truth black, ensemble green)", fontsize=10)
        for member in exp.prior:
            ax.plot(idx, member, color=colors.prior, lw=0.5)
        ax.plot(idx, exp.truth, color=colors.truth, lw=2)
        if getattr(exp, "last_obs", None) is not None and not exp.ready_to_advance:
            pass  # obs drawn only right after an assimilation
        if not np.all(np.isnan(getattr(exp, "last_obs", np.full(1, np.nan)))):
            ax.plot(idx, exp.last_obs, "*", color=colors.observation, markersize=6)
        ax.set_xlabel("State variable")

        ax = self.ax_err
        ax.clear()
        ax.set_title("Prior error (blue) and spread (red)", fontsize=10)
        t = np.asarray(h["time"])
        ax.plot(t, h["prior_error"], color=colors.posterior, lw=1.5)
        ax.plot(t, h["prior_spread"], color=colors.observation, lw=1.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time")

        self.ax_prior_rank.clear()
        plot_rank_histogram(self.ax_prior_rank, exp.prior_rank_hist.counts,
                            None, colors.prior, "Prior rank histogram")
        self.ax_post_rank.clear()
        plot_rank_histogram(self.ax_post_rank, exp.post_rank_hist.counts,
                            None, colors.posterior, "Posterior rank histogram")

        if self.ax_inflation is not None and h["inflate"]:
            ax = self.ax_inflation
            ax.clear()
            ax.set_title(f"Inflation field (mean = {h['inflate_mean'][-1]:.4f})",
                         fontsize=10)
            ax.plot(idx, h["inflate"][-1], "-x", color=colors.posterior, lw=1)
            ax.axhline(1.0, color="0.7", lw=0.5)
            ax.set_xlabel("State variable")

        self.time_html.value = f"Time = {exp.time}"
        if h["prior_error"]:
            self.status.value = (f"Prior error = {h['prior_error'][-1]:.3f}, "
                                 f"spread = {h['prior_spread'][-1]:.3f}")
        self.fig.canvas.draw_idle()


def run_lorenz_96(seed: int | None = None) -> RunLorenz96App:
    """Launch the run_lorenz_96 app (fixed inflation)."""
    return RunLorenz96App(adaptive=False, seed=seed)


def run_lorenz_96_inf(seed: int | None = None) -> RunLorenz96App:
    """Launch the run_lorenz_96_inf app (adaptive inflation, obs networks)."""
    return RunLorenz96App(adaptive=True, seed=seed)
