"""The run_lorenz_63 app: ensemble DA on the Lorenz 63 attractor.

Python version of ``DART_LAB/matlab/run_lorenz_63.m``. A local 3-D view
follows the truth and the 20-member ensemble; a global view shows the whole
attractor for context. With ipympl the 3-D axes can be rotated when paused.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from pydartlab.apps._base import DartLabApp, make_figure
from pydartlab.experiments.lorenz63_da import Lorenz63Experiment
from pydartlab.style import colors

TRAIL = 8  # how many recent truth/ensemble segments the local view shows


class RunLorenz63App(DartLabApp):
    def __init__(self, seed: int | None = None):
        super().__init__()
        self.seed = seed
        self.fig = make_figure(figsize=(9, 5))
        self.ax_local = self.fig.add_subplot(121, projection="3d")
        self.ax_global = self.fig.add_subplot(122, projection="3d")

        self.step_button = self.button("Advance Model", self.single_step,
                                       button_style="primary")
        self.play = widgets.Play(value=0, min=0, max=100_000, interval=250,
                                 description="Auto run")
        self.play.observe(lambda ch: self.single_step(), names="value")
        self.filter_radio = self.filter_selector(
            options=("No Assimilation", "EAKF", "EnKF", "RHF"))
        reset_btn = self.button("Reset", self.reset)
        self.time_html = widgets.HTML()

        controls = widgets.VBox([self.step_button, self.play, self.filter_radio,
                                 reset_btn, self.time_html, self.status])
        canvas = self.fig.canvas if isinstance(self.fig.canvas, widgets.Widget) else None
        self.widget = widgets.HBox([canvas, controls] if canvas else [controls])
        self.reset()

    def reset(self):
        self.experiment = Lorenz63Experiment(filter_type=self.filter_radio.value,
                                             seed=self.seed)
        self._draw_global_attractor()
        self.ax_local.clear()
        self.step_button.description = "Advance Model"
        self.time_html.value = "Time = 0"
        self.fig.canvas.draw_idle()

    def _draw_global_attractor(self):
        """Trace the attractor once for the context view."""
        ax = self.ax_global
        ax.clear()
        ax.set_title("Global view", fontsize=10)
        model = self.experiment.model
        x = self.experiment.truth.copy()
        traj = np.empty((1500, 3))
        for i in range(traj.shape[0]):
            x = model.step(x)
            traj[i] = x
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="0.8", lw=0.4)
        self._global_truth, = ax.plot([], [], [], "*", color=colors.truth,
                                      markersize=10)
        ax.set_axis_off()

    def single_step(self):
        exp = self.experiment
        exp.filter_type = self.filter_radio.value
        if exp.ready_to_advance:
            exp.advance()
            self.step_button.description = "Assimilate Obs"
        else:
            exp.assimilate()
            self.step_button.description = "Advance Model"
        self.redraw()

    def redraw(self):
        exp = self.experiment
        h = exp.history
        ax = self.ax_local
        ax.clear()
        ax.set_title(f"Local view ({exp.filter_type})", fontsize=10)

        # Recent truth trajectory in black
        truth = np.asarray(h["truth"][-TRAIL:])
        if truth.shape[0] > 1:
            ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], color=colors.truth, lw=2)
        ax.plot([truth[-1, 0]], [truth[-1, 1]], [truth[-1, 2]], "*",
                color=colors.truth, markersize=12)

        # Ensemble member trajectories in green
        priors = h["prior"][-TRAIL:]
        if len(priors) > 1:
            arr = np.asarray(priors)  # (time, member, 3)
            for m in range(arr.shape[1]):
                ax.plot(arr[:, m, 0], arr[:, m, 1], arr[:, m, 2],
                        color=colors.prior, lw=0.5)

        # Observation and increments at assimilation times
        if h["observation"] and h["observation"][-1] is not None \
                and len(h["posterior"]) == len(h["prior"]):
            obs = h["observation"][-1]
            ax.plot([obs[0]], [obs[1]], [obs[2]], "*", color=colors.observation,
                    markersize=12)
            prior = h["prior"][-1]
            post = h["posterior"][-1]
            for m in range(prior.shape[0]):
                ax.plot([prior[m, 0], post[m, 0]], [prior[m, 1], post[m, 1]],
                        [prior[m, 2], post[m, 2]], color=colors.observation, lw=0.7)

        # Update the truth marker on the global view
        self._global_truth.set_data_3d([truth[-1, 0]], [truth[-1, 1]], [truth[-1, 2]])
        self.time_html.value = f"Time = {exp.time}"
        self.fig.canvas.draw_idle()


def run_lorenz_63(seed: int | None = None) -> RunLorenz63App:
    """Launch the run_lorenz_63 app."""
    return RunLorenz63App(seed=seed)
