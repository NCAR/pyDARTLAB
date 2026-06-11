"""Color and style conventions for pyDARTLAB.

The defaults reproduce the MATLAB DART_LAB color semantics used throughout
the tutorial slides and apps:

* green  = prior
* red    = observation / likelihood
* blue   = posterior

Colors are settable at runtime so the palette can be swapped for a
colorblind-friendly one::

    import pydartlab
    pydartlab.style.use_colorblind_palette()
    # or change individual roles
    pydartlab.style.set_colors(prior="#009E73")
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Palette:
    """Role-based colors used by all pyDARTLAB plots and apps."""

    prior: str = "#008000"        # DART_LAB green
    observation: str = "#D70A53"  # DART_LAB red
    posterior: str = "#0000FF"    # DART_LAB blue
    inflated: str = "#FF9933"     # DART_LAB orange
    truth: str = "#000000"        # black
    highlight: str = "#FFFF00"    # yellow (e.g. newest rank histogram entry)
    background: str = "#E1E1E1"   # light gray figure background
    lightblue: str = "#ADEBFF"    # inflation panel background in MATLAB
    member: str = "#008000"       # individual ensemble members default to prior color
    localization: list[str] = field(
        default_factory=lambda: ["#969696", "#1E90FF", "#FF3333", "#009900"]
    )


# DART_LAB defaults (matlab/private/stylesheet.m)
DARTLAB = Palette()

# Okabe-Ito based palette, distinguishable under common color vision deficiencies
COLORBLIND = Palette(
    prior="#009E73",        # bluish green
    observation="#D55E00",  # vermillion
    posterior="#0072B2",    # blue
    inflated="#E69F00",     # orange
    truth="#000000",
    highlight="#F0E442",    # yellow
    background="#E1E1E1",
    lightblue="#56B4E9",
    member="#009E73",
    localization=["#999999", "#56B4E9", "#D55E00", "#009E73"],
)

#: The active palette. Import this and read attributes at draw time;
#: do not copy values at import time or set_colors() will not take effect.
colors = Palette()


def set_colors(**kwargs: str) -> None:
    """Change individual role colors on the active palette.

    Parameters are any of the :class:`Palette` field names, e.g.
    ``set_colors(prior="#009E73", posterior="#0072B2")``.
    """
    for name, value in kwargs.items():
        if not hasattr(colors, name):
            valid = ", ".join(f.name for f in colors.__dataclass_fields__.values())
            raise ValueError(f"Unknown color role {name!r}. Valid roles: {valid}")
        setattr(colors, name, value)


def use_palette(palette: Palette) -> None:
    """Make ``palette`` the active palette (copies values in place)."""
    for name in palette.__dataclass_fields__:
        setattr(colors, name, getattr(palette, name))


def use_colorblind_palette() -> None:
    """Switch the active palette to a colorblind-friendly (Okabe-Ito) palette."""
    use_palette(COLORBLIND)


def use_dartlab_palette() -> None:
    """Restore the default MATLAB DART_LAB palette."""
    use_palette(DARTLAB)
