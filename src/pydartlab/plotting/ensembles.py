"""Drawing helpers for ensembles and rank histograms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pydartlab.style import colors


def plot_ensemble_ticks(ax, ens: ArrayLike, y: float = 0.0, color: str | None = None,
                        marker: str = "*", markersize: float = 10, **kwargs):
    """Plot ensemble members as markers along a horizontal line at ``y``."""
    ens = np.asarray(ens, dtype=float)
    color = color if color is not None else colors.prior
    return ax.plot(ens, np.full(ens.size, y), marker, color=color,
                   markersize=markersize, linestyle="none", **kwargs)


def plot_rank_histogram(ax, counts: ArrayLike, highlight_last: int | None = None,
                        color: str | None = None, title: str | None = None):
    """Bar plot of a rank histogram; optionally highlight the newest entry.

    ``highlight_last`` is the 1-based rank of the most recent addition, drawn
    in the highlight color as in the MATLAB tools.
    """
    counts = np.asarray(counts)
    bins = np.arange(1, counts.size + 1)
    color = color if color is not None else colors.prior
    bars = ax.bar(bins, counts, color=color, edgecolor="black", linewidth=0.5)
    if highlight_last is not None and 1 <= highlight_last <= counts.size:
        bars[highlight_last - 1].set_facecolor(colors.highlight)
    ax.set_xlim(0.5, counts.size + 0.5)
    ax.set_xlabel("Rank")
    if title:
        ax.set_title(title)
    return bars
