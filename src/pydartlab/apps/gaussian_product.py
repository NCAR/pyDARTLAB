"""The gaussian_product app: the product of two Gaussians is Gaussian.

Python version of ``DART_LAB/matlab/gaussian_product.m``. Set the prior
(green) and the observation likelihood (red); the posterior (blue) is their
normalized product, with the weight (the likelihood integral) reported.
"""

from __future__ import annotations

import ipywidgets as widgets

from pydartlab.apps._base import DartLabApp, make_figure
from pydartlab.plotting.dists import gaussian_curve
from pydartlab.stats import product_of_gaussians
from pydartlab.style import colors


class GaussianProductApp(DartLabApp):
    def __init__(self):
        super().__init__()
        self.fig = make_figure(figsize=(7, 5))
        self.ax = self.fig.add_subplot(111)

        self.prior_mean = self.float_field(1.0, "Mean:")
        self.prior_sd = self.float_field(1.0, "SD:", minimum=1e-6)
        self.obs_mean = self.float_field(1.0, "Mean:")
        self.obs_sd = self.float_field(1.0, "SD:", minimum=1e-6)
        self.posterior_html = widgets.HTML()
        plot_btn = self.button("Plot Posterior", self.update, button_style="primary")

        controls = widgets.VBox([
            widgets.HTML(f'<b style="color:{colors.prior}">Prior</b>'),
            self.prior_mean, self.prior_sd,
            widgets.HTML(f'<b style="color:{colors.observation}">Observation likelihood</b>'),
            self.obs_mean, self.obs_sd,
            plot_btn,
            widgets.HTML(f'<b style="color:{colors.posterior}">Posterior</b>'),
            self.posterior_html,
        ])
        canvas = self.fig.canvas if isinstance(self.fig.canvas, widgets.Widget) else None
        self.widget = widgets.HBox([canvas, controls] if canvas else [controls])
        self.draw_priors()

    def draw_priors(self):
        """Draw the prior and likelihood only (before Plot Posterior)."""
        self.ax.clear()
        x, y = gaussian_curve(self.prior_mean.value, self.prior_sd.value)
        self.ax.plot(x, y, color=colors.prior, lw=2, label="Prior")
        x, y = gaussian_curve(self.obs_mean.value, self.obs_sd.value)
        self.ax.plot(x, y, "--", color=colors.observation, lw=2, label="Obs. likelihood")
        self.ax.set_ylim(bottom=0)
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()

    def update(self):
        post_mean, post_sd, weight = product_of_gaussians(
            self.prior_mean.value, self.prior_sd.value,
            self.obs_mean.value, self.obs_sd.value)

        self.draw_priors()
        x, y = gaussian_curve(post_mean, post_sd)
        self.ax.plot(x, y, color=colors.posterior, lw=2, label="Posterior")
        x, y = gaussian_curve(post_mean, post_sd, weight=weight)
        self.ax.plot(x, y, "--", color=colors.posterior, lw=1.5,
                     label="Weighted posterior")
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()

        self.posterior_html.value = (
            f"mean = {post_mean:.4f}<br>sd = {post_sd:.4f}<br>weight = {weight:.4f}")
        self.last_result = (post_mean, post_sd, weight)


def gaussian_product() -> GaussianProductApp:
    """Launch the gaussian_product app."""
    return GaussianProductApp()
