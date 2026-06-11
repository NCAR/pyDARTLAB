"""The bounded_oned_ensemble app: filters for non-negative variables.

Python version of ``DART_LAB/matlab/bounded_oned_ensemble.m``. Compare a
standard EAKF (which can produce negative posterior members), a gamma
filter and a bounded normal rank histogram filter (BNRHF), both of which
respect the bound at zero. Inflation is applied in a probit-transformed
space so the inflated ensemble also respects the bound.
"""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np

from pydartlab.algorithms.filters import obs_increment_bnrhf, obs_increment_gamma
from pydartlab.algorithms.increments import obs_increment_eakf
from pydartlab.algorithms.inflation import inflate_bnrh, inflate_ensemble, inflate_gamma
from pydartlab.apps._base import ClickEnsembleMixin, DartLabApp, fmt_stats, make_figure
from pydartlab.plotting.dists import gamma_curve, gaussian_curve
from pydartlab.style import colors

PRIOR_Y, POST_Y = 0.4, 0.25
INF_PRIOR_Y, INF_POST_Y = 0.1, -0.05


class BoundedOnedEnsembleApp(DartLabApp, ClickEnsembleMixin):
    def __init__(self, obs_mean: float = 1.0, obs_sd: float = 1.0):
        super().__init__()
        self.fig = make_figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(111)
        self.ens = np.array([])

        self.obs_mean = self.float_field(obs_mean, "Obs:", minimum=1e-6)
        self.obs_sd = self.float_field(obs_sd, "Obs SD:", minimum=1e-6)
        self.filter_radio = self.filter_selector(options=("EAKF", "Gamma", "BNRHF"))
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
                                value_from_event=self._value_from_click,
                                on_member_added=self._draw_pending,
                                on_finished=self._creation_finished)
        self.draw_base()

    def _value_from_click(self, event):
        if event.xdata is None or event.xdata < 0:
            self.status.value = "Members must be non-negative; click at x >= 0."
            return None
        return event.xdata

    # ---- ensemble creation -------------------------------------------

    def start_ensemble_creation(self):
        self.ens = np.array([])
        self.draw_base()
        self.status.value = ("Click in the plot (x >= 0) to place members; "
                             "click outside the axes to finish.")
        ClickEnsembleMixin.start_ensemble_creation(self)

    def _draw_pending(self, value):
        self.ax.plot(value, PRIOR_Y, "*", color=colors.prior, markersize=12)
        self.fig.canvas.draw_idle()

    def _creation_finished(self, members):
        self.set_ensemble(members)

    def set_ensemble(self, members):
        """Set the (non-negative) prior ensemble programmatically."""
        members = np.asarray(members, dtype=float)
        if np.any(members < 0):
            raise ValueError("bounded_oned_ensemble members must be non-negative")
        self.creating = False
        self.ens = members
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
        keep = x >= 0
        self.ax.plot(x[keep], y[keep], "--", color=colors.observation, lw=2,
                     label="Likelihood")
        self.ax.axvline(0.0, color="k", lw=1)
        self.ax.set_xlim(left=-0.3)
        self.ax.set_ylim(-0.2, max(1.0, y.max() * 1.1))
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()

    def _update(self, ens):
        """Return (posterior, prior_curve, post_curve) for the chosen filter."""
        obs, sd = self.obs_mean.value, self.obs_sd.value
        kind = self.filter_radio.value
        if kind == "EAKF":
            post = ens + obs_increment_eakf(ens, obs, sd**2)
            prior_curve = gaussian_curve(ens.mean(), ens.std(ddof=1))
            post_curve = gaussian_curve(post.mean(), post.std(ddof=1))
        elif kind == "Gamma":
            result = obs_increment_gamma(ens, obs, sd)
            post = ens + result.increments
            prior_curve = gamma_curve(result.prior_shape, result.prior_scale)
            post_curve = gamma_curve(result.post_shape, result.post_scale)
        elif kind == "BNRHF":
            result = obs_increment_bnrhf(ens, obs, sd**2)
            post = ens + result.increments
            prior_curve = result.prior_pdf
            post_curve = result.post_pdf
        else:  # pragma: no cover
            raise ValueError(kind)
        return post, prior_curve, post_curve

    def _inflate(self, ens):
        """Inflate in the space appropriate to the chosen filter."""
        var_inf = self.inf_slider.value
        kind = self.filter_radio.value
        if kind == "Gamma":
            return inflate_gamma(ens, var_inf)
        if kind == "BNRHF":
            return inflate_bnrh(ens, var_inf, bounded_below=True, lower_bound=0.0)
        return inflate_ensemble(ens, var_inf)

    def update_ensemble(self):
        if self.ens.size < 2:
            self.status.value = "Create an ensemble (at least 2 members) first."
            return
        ens = self.ens
        post, prior_curve, post_curve = self._update(ens)

        self.set_ensemble(ens)
        self.ax.plot(post, np.full(post.size, POST_Y), "*",
                     color=colors.posterior, markersize=12)
        for p, q in zip(ens, post):
            self.ax.plot([p, q], [PRIOR_Y, POST_Y], color="0.6", lw=0.8)
        self.ax.plot(*prior_curve, color=colors.prior, lw=2)
        self.ax.plot(*post_curve, color=colors.posterior, lw=2)

        stats = [fmt_stats("Prior", ens, colors.prior),
                 fmt_stats("Posterior", post, colors.posterior)]
        if np.any(post < 0):
            stats.append(f'<b style="color:{colors.observation}">'
                         "Posterior violates the bound at zero!</b>")

        if self.inf_toggle.value and self.inf_slider.value > 1.0:
            inf_ens = self._inflate(ens)
            inf_post, _, _ = self._update(inf_ens)
            self.ax.plot(inf_ens, np.full(inf_ens.size, INF_PRIOR_Y), "*",
                         color=colors.inflated, markersize=12)
            self.ax.plot(inf_post, np.full(inf_post.size, INF_POST_Y), "*",
                         color=colors.posterior, markersize=8)
            for p, q in zip(inf_ens, inf_post):
                self.ax.plot([p, q], [INF_PRIOR_Y, INF_POST_Y], color="0.8", lw=0.8)
            stats += [fmt_stats("Inflated prior", inf_ens, colors.inflated)]
            self.last_inflated_prior = inf_ens
            self.last_inflated_posterior = inf_post

        self.stats_html.value = "<br>".join(stats)
        self.fig.canvas.draw_idle()
        self.last_posterior = post


def bounded_oned_ensemble(**kwargs) -> BoundedOnedEnsembleApp:
    """Launch the bounded_oned_ensemble app."""
    return BoundedOnedEnsembleApp(**kwargs)
