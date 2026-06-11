"""Shared scaffolding for the interactive pyDARTLAB apps.

Apps combine a matplotlib figure (rendered live by ipympl when the notebook
uses ``%matplotlib widget``) with ipywidgets controls laid out beside it.
All computation is delegated to :mod:`pydartlab.algorithms` and
:mod:`pydartlab.experiments`; app classes contain only widget wiring and
drawing.

Every click-driven app also works without a mouse: ``set_ensemble()`` takes
a list of values, so all tutorial exercises can be completed in environments
where ipympl events are unavailable.
"""

from __future__ import annotations

from collections.abc import Callable

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

from pydartlab.style import colors

FILTER_NAMES = ("EAKF", "EnKF", "RHF")


def make_figure(**kwargs):
    """Create a figure for an app without triggering duplicate display.

    With ipympl, the canvas itself is a widget that the app embeds in its
    layout, so the figure must not also be shown by the normal pyplot
    mechanism.
    """
    with plt.ioff():
        fig = plt.figure(**kwargs)
    # ipympl: hide the default toolbar/header for a cleaner app look
    canvas = fig.canvas
    for attr, value in [("header_visible", False), ("footer_visible", False),
                        ("toolbar_visible", False)]:
        try:
            setattr(canvas, attr, value)
        except AttributeError:
            pass
    return fig


class DartLabApp:
    """Base class for the interactive apps.

    Subclasses build their figure and controls in ``__init__`` and assign
    the ipywidgets layout to ``self.widget``.
    """

    def __init__(self):
        self.widget: widgets.Widget | None = None
        self.status = widgets.HTML()

    def _ipython_display_(self):
        from IPython.display import display

        if self.widget is not None:
            display(self.widget)

    def show(self):
        """Display the app (alternative to letting it auto-display)."""
        self._ipython_display_()

    # ---- common control builders -------------------------------------

    def filter_selector(self, options=FILTER_NAMES, description="Filter:",
                        on_change: Callable | None = None) -> widgets.RadioButtons:
        radio = widgets.RadioButtons(options=options, description=description,
                                     layout=widgets.Layout(width="auto"))
        if on_change is not None:
            radio.observe(lambda change: on_change(change["new"]), names="value")
        return radio

    def float_field(self, value: float, description: str, step: float = 0.1,
                    minimum: float | None = None) -> widgets.BoundedFloatText | widgets.FloatText:
        if minimum is not None:
            return widgets.BoundedFloatText(value=value, min=minimum, max=1e12,
                                            step=step, description=description,
                                            layout=widgets.Layout(width="180px"))
        return widgets.FloatText(value=value, step=step, description=description,
                                 layout=widgets.Layout(width="180px"))

    def button(self, description: str, callback: Callable,
               **kwargs) -> widgets.Button:
        btn = widgets.Button(description=description,
                             layout=widgets.Layout(width="auto"), **kwargs)
        btn.on_click(lambda _b: callback())
        return btn

    def inflation_controls(self, on_change: Callable | None = None
                           ) -> tuple[widgets.Checkbox, widgets.FloatSlider]:
        """The inflation toggle + slider used by several tools (1 to 5)."""
        toggle = widgets.Checkbox(value=False, description="Apply inflation",
                                  indent=False)
        slider = widgets.FloatSlider(value=1.0, min=1.0, max=5.0, step=0.1,
                                     description="Inflation:", readout_format=".1f",
                                     continuous_update=False)
        if on_change is not None:
            toggle.observe(lambda change: on_change(), names="value")
            slider.observe(lambda change: on_change(), names="value")
        return toggle, slider


class ClickEnsembleMixin:
    """Click-to-create-ensemble behavior shared by several apps.

    After ``start_ensemble_creation()`` (wired to the "Create New Ensemble"
    button), clicks inside ``self.click_axes`` append members; a click
    outside the axes finishes creation, exactly like the MATLAB tools.
    ``set_ensemble()`` is the programmatic equivalent.
    """

    #: Maximum members before creation auto-finishes (safety)
    max_members = 100

    def init_click_capture(self, fig, click_axes, value_from_event: Callable,
                           on_member_added: Callable, on_finished: Callable):
        self._click_fig = fig
        self.click_axes = click_axes
        self._value_from_event = value_from_event
        self._on_member_added = on_member_added
        self._on_finished = on_finished
        self.creating = False
        self.pending_members: list = []
        fig.canvas.mpl_connect("button_press_event", self._handle_click)

    def start_ensemble_creation(self):
        self.creating = True
        self.pending_members = []

    def _handle_click(self, event):
        if not self.creating:
            return
        if event.inaxes is not self.click_axes:
            # Click outside the axes ends ensemble creation
            self.finish_ensemble_creation()
            return
        value = self._value_from_event(event)
        if value is None:
            return
        self.pending_members.append(value)
        self._on_member_added(value)
        if len(self.pending_members) >= self.max_members:
            self.finish_ensemble_creation()

    def finish_ensemble_creation(self):
        if not self.creating:
            return
        self.creating = False
        self._on_finished(self.pending_members)


def likelihood_label() -> str:
    return f'<span style="color:{colors.observation}">Likelihood</span>'


def fmt_stats(name: str, ens: np.ndarray, color: str) -> str:
    """Small HTML line with the mean and sd of an ensemble."""
    return (f'<span style="color:{color}">{name}: '
            f"mean = {np.mean(ens):.3f}, sd = {np.std(ens, ddof=1):.3f}</span>")
