"""The oned_model / oned_model_inf apps: cycling DA on a 1-D model.

Python versions of ``DART_LAB/matlab/oned_model.m`` and
``oned_model_inf.m`` (the latter adds adaptive inflation). The truth is
always zero; the model doubles the state each step (plus optional bias and
nonlinearity), so the filter must continually pull the ensemble back.

Panels: latest update, state evolution, error vs spread, kurtosis, and
prior/posterior rank histograms (plus inflation evolution when adaptive).
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from pydartlab.apps._base import DartLabApp, make_figure
from pydartlab.experiments.oned_da import OneDExperiment
from pydartlab.plotting.dists import gaussian_curve
from pydartlab.plotting.ensembles import plot_rank_histogram
from pydartlab.style import colors

WINDOW = 10  # time steps shown in the evolution panels


class OnedModelApp(DartLabApp):
    def __init__(self, adaptive: bool = False, seed: int | None = None):
        super().__init__()
        self.adaptive = adaptive
        self.seed = seed

        nrows = 4 if adaptive else 3
        self.fig = make_figure(figsize=(9, 2.3 * nrows))
        gs = self.fig.add_gridspec(nrows, 2, hspace=0.55, wspace=0.25)
        self.ax_update = self.fig.add_subplot(gs[0, 0])
        self.ax_evolution = self.fig.add_subplot(gs[0, 1])
        self.ax_err = self.fig.add_subplot(gs[1, 0])
        self.ax_kurt = self.fig.add_subplot(gs[1, 1])
        self.ax_prior_rank = self.fig.add_subplot(gs[2, 0])
        self.ax_post_rank = self.fig.add_subplot(gs[2, 1])
        self.ax_inflation = self.fig.add_subplot(gs[3, :]) if adaptive else None

        # Controls
        self.step_button = self.button("Advance Model", self.single_step,
                                       button_style="primary")
        self.play = widgets.Play(value=0, min=0, max=10_000, interval=400,
                                 description="Auto run")
        self.play.observe(self._play_tick, names="value")
        self.ens_size = widgets.BoundedIntText(value=4, min=2, max=100,
                                               description="Ens size:",
                                               layout=widgets.Layout(width="180px"))
        self.ens_size.observe(lambda ch: self._set_ens_size(ch["new"]), names="value")
        self.model_bias = self.float_field(0.0, "Bias:", minimum=0.0)
        self.alpha = self.float_field(0.0, "Nonlin a:", step=0.05, minimum=0.0)
        self.filter_radio = self.filter_selector(on_change=self._set_filter)

        if adaptive:
            self.inf_mode = widgets.RadioButtons(
                options=["Fixed Inflation", "Adaptive Inflation"],
                description="Inflation:")
            self.inf_mode.observe(lambda ch: self._push_config(), names="value")
        self.fixed_inflation = self.float_field(1.0, "Inflation:", minimum=1.0)
        self.inf_damping = self.float_field(1.0, "Damping:", step=0.05, minimum=0.0)
        self.inf_sd = self.float_field(0.6, "Inf SD:", minimum=1e-6)
        self.inf_sd_min = self.float_field(0.6, "Min SD:", minimum=0.0)
        self.inf_min = self.float_field(1.0, "Inf min:", minimum=0.0)
        self.inf_max = self.float_field(5.0, "Inf max:", minimum=0.0)

        reset_btn = self.button("Reset", self.reset)
        clear_btn = self.button("Clear Hist", self.clear_histograms)

        items = [self.step_button, self.play, self.ens_size, self.model_bias,
                 self.alpha, self.filter_radio, self.fixed_inflation]
        if adaptive:
            items += [self.inf_mode, self.inf_damping, self.inf_sd,
                      self.inf_sd_min, self.inf_min, self.inf_max]
        items += [widgets.HBox([reset_btn, clear_btn]), self.status]
        controls = widgets.VBox(items)
        canvas = self.fig.canvas if isinstance(self.fig.canvas, widgets.Widget) else None
        self.widget = widgets.HBox([canvas, controls] if canvas else [controls])

        self.reset()

    # ---- experiment wiring --------------------------------------------

    def _new_experiment(self) -> OneDExperiment:
        return OneDExperiment(ens_size=self.ens_size.value,
                              filter_type=self.filter_radio.value,
                              seed=self.seed)

    def _push_config(self):
        """Copy current control values into the experiment."""
        exp = self.experiment
        exp.model_bias = self.model_bias.value
        exp.alpha = self.alpha.value
        exp.filter_type = self.filter_radio.value
        adaptive_on = self.adaptive and self.inf_mode.value == "Adaptive Inflation"
        exp.adaptive_inflation = adaptive_on
        if adaptive_on:
            exp.inflation_damping = self.inf_damping.value
            exp.inflation_sd = max(exp.inflation_sd, 1e-6)
            exp.inflation_sd_min = self.inf_sd_min.value
            exp.inflation_min = self.inf_min.value
            exp.inflation_max = self.inf_max.value
        else:
            exp.inflation = self.fixed_inflation.value

    def _set_filter(self, value):
        self.experiment.filter_type = value

    def _set_ens_size(self, value):
        self.experiment.set_ens_size(value)
        self.clear_histograms()

    def reset(self):
        self.experiment = self._new_experiment()
        if self.adaptive:
            self.experiment.inflation_sd = self.inf_sd.value
        for ax in self.fig.axes:
            ax.clear()
        self.status.value = "Press 'Advance Model' to begin."
        self.step_button.description = "Advance Model"
        self.fig.canvas.draw_idle()

    def clear_histograms(self):
        self.experiment.reset_histograms()
        self._draw_histograms()
        self.fig.canvas.draw_idle()

    # ---- stepping ------------------------------------------------------

    def _play_tick(self, _change):
        self.single_step()

    def single_step(self):
        """One button press: advance or assimilate, whichever is next."""
        self._push_config()
        exp = self.experiment
        if exp.ready_to_advance:
            if self.adaptive and exp.adaptive_inflation:
                # New sd field value only takes effect via the experiment state
                pass
            exp.advance()
            self.step_button.description = "Assimilate Obs"
        else:
            exp.assimilate()
            self.step_button.description = "Advance Model"
        self.redraw()

    # ---- drawing ---------------------------------------------------------

    def _draw_histograms(self):
        exp = self.experiment
        self.ax_prior_rank.clear()
        plot_rank_histogram(self.ax_prior_rank, exp.prior_rank_hist.counts,
                            exp.prior_rank_hist.last_rank, colors.prior,
                            "Prior rank histogram")
        self.ax_post_rank.clear()
        plot_rank_histogram(self.ax_post_rank, exp.post_rank_hist.counts,
                            exp.post_rank_hist.last_rank, colors.posterior,
                            "Posterior rank histogram")

    def redraw(self):
        exp = self.experiment
        h = exp.history
        t = np.asarray(h["time"])
        if t.size == 0:
            return
        t0 = max(0, t[-1] - WINDOW)

        # Latest update panel
        ax = self.ax_update
        ax.clear()
        ax.set_title("Latest ensemble", fontsize=10)
        prior = h["prior_ens"][-1]
        ax.plot(prior, np.zeros(prior.size), "*", color=colors.prior, markersize=10,
                label="Prior")
        if len(h["post_ens"]) == len(h["prior_ens"]):
            post = h["post_ens"][-1]
            obs = h["observation"][-1]
            ax.plot(post, np.full(post.size, -0.1), "*", color=colors.posterior,
                    markersize=10, label="Posterior")
            ax.plot(obs, 0.05, "*", color=colors.observation, markersize=14)
            x, y = gaussian_curve(obs, exp.obs_error_sd)
            ax.plot(x, y, "--", color=colors.observation, lw=1.5)
        ax.set_ylim(-0.2, 0.6)
        ax.legend(fontsize=7, loc="upper right")

        # State evolution
        ax = self.ax_evolution
        ax.clear()
        ax.set_title("State evolution", fontsize=10)
        ax.axhline(0, color=colors.truth, ls="--", lw=1)  # truth
        for i, (ti, pens) in enumerate(zip(t, h["prior_ens"])):
            if ti < t0:
                continue
            ax.plot(np.full(pens.size, ti - 0.1), pens, "*", color=colors.prior,
                    markersize=5)
        for i, pens in enumerate(h["post_ens"]):
            ti = t[i]
            if ti < t0:
                continue
            ax.plot(np.full(pens.size, ti + 0.1), pens, "*", color=colors.posterior,
                    markersize=5)
            ax.plot(ti, h["observation"][i], "*", color=colors.observation,
                    markersize=9)
        ax.set_xlim(t0, t[-1] + 1)
        ax.set_xlabel("Time")

        # Error and spread
        ax = self.ax_err
        ax.clear()
        ax.set_title("Error (blue) and spread (red)", fontsize=10)
        n = min(len(h["prior_error"]), len(h["post_error"]))
        times, errs, spreads = [], [], []
        for i in range(n):
            times += [t[i] - 0.1, t[i] + 0.1]
            errs += [h["prior_error"][i], h["post_error"][i]]
            spreads += [h["prior_spread"][i], h["post_spread"][i]]
        ax.plot(times, errs, color=colors.posterior, lw=1.5)
        ax.plot(times, spreads, color=colors.observation, lw=1.5)
        ax.set_xlim(t0, t[-1] + 1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Time")

        # Kurtosis
        ax = self.ax_kurt
        ax.clear()
        ax.set_title("Prior kurtosis", fontsize=10)
        ax.plot(t, h["prior_kurtosis"], color=colors.observation, lw=1.5)
        ax.set_xlim(t0, t[-1] + 1)
        ax.set_xlabel("Time")

        self._draw_histograms()

        # Inflation evolution (adaptive app)
        if self.ax_inflation is not None and h["inflation"]:
            ax = self.ax_inflation
            ax.clear()
            ax.set_title("Inflation", fontsize=10)
            ti = t[:len(h["inflation"])]
            lam = np.asarray(h["inflation"])
            sd = np.asarray(h["inflation_sd"])
            ax.plot(ti, lam, ".-", color=colors.posterior, lw=1)
            ax.fill_between(ti, lam - sd, lam + sd, color=colors.posterior, alpha=0.15)
            ax.set_xlim(t0, t[-1] + 1)
            ax.set_xlabel("Time")
            self.status.value = (f"Inflation = {lam[-1]:.3f} "
                                 f"(sd = {sd[-1]:.3f})")

        self.fig.canvas.draw_idle()


def oned_model(seed: int | None = None) -> OnedModelApp:
    """Launch the oned_model app (fixed inflation)."""
    return OnedModelApp(adaptive=False, seed=seed)


def oned_model_inf(seed: int | None = None) -> OnedModelApp:
    """Launch the oned_model_inf app (fixed or adaptive inflation)."""
    return OnedModelApp(adaptive=True, seed=seed)
