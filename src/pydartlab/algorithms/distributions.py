"""Distribution machinery for rank-histogram and bounded filters.

Ports of ``weighted_norm_inv.m``, ``ens_quantiles.m``, ``bnrh_cdf.m``,
``bnrh_cdf_initialized.m``, ``inv_bnrh_cdf.m`` and ``plot_rhf_pdf.m`` from
``DART_LAB/matlab/private``, plus a gamma-distribution fit matching MATLAB's
``gamfit``.

The bounded normal rank histogram (BNRH) distribution places ``1/(n+1)``
probability mass between adjacent sorted ensemble members with (possibly
bounded) Gaussian tails outside the outermost members. It is the
non-parametric workhorse of the QCEFF framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


def weighted_norm_inv(alpha: float, mean: float, sd: float, p: float) -> float:
    """Value of x where the cdf of ``alpha * N(mean, sd)`` equals ``p``."""
    return float(mean + norm.ppf(p / alpha) * sd)


def fit_gamma(ens: ArrayLike) -> tuple[float, float]:
    """Maximum-likelihood gamma fit, returning ``(shape, scale)``.

    Matches MATLAB's ``gamfit`` (two-parameter gamma, location fixed at 0).
    """
    ens = np.asarray(ens, dtype=float)
    shape, _loc, scale = gamma_dist.fit(ens, floc=0)
    return float(shape), float(scale)


def ens_quantiles(sorted_ens: ArrayLike, bounded_below: bool = False,
                  bounded_above: bool = False, lower_bound: float = 0.0,
                  upper_bound: float = 0.0) -> NDArray[np.float64]:
    """Quantiles of a sorted ensemble for a BNRH distribution.

    Handles ensemble members identical to the bounds and duplicate interior
    members (duplicates share the mean quantile of their run).
    """
    sorted_ens = np.asarray(sorted_ens, dtype=float)
    ens_size = sorted_ens.size
    q = np.zeros(ens_size)

    # Number of ensemble members that duplicate the lower bound
    lower_dups = 0
    if bounded_below:
        while lower_dups < ens_size and sorted_ens[lower_dups] == lower_bound:
            lower_dups += 1

    # Number of ensemble members that duplicate the upper bound
    upper_dups = 0
    if bounded_above:
        while upper_dups < ens_size and sorted_ens[ens_size - 1 - upper_dups] == upper_bound:
            upper_dups += 1

    # Quantiles for the bound duplicates
    q[:lower_dups] = lower_dups / (2.0 * (ens_size + 1.0))
    if upper_dups:
        q[ens_size - upper_dups:] = 1.0 - upper_dups / (2.0 * (ens_size + 1.0))

    # Interior members: runs of duplicates share a quantile. series_start is
    # the 1-based index of the first member of the run, as in the MATLAB.
    i = lower_dups
    end = ens_size - upper_dups
    while i < end:
        j = i
        while j + 1 < end and sorted_ens[j + 1] == sorted_ens[i]:
            j += 1
        series_start = i + 1          # 1-based
        series_length = j - i + 1
        q[i:j + 1] = (series_start / (ens_size + 1.0)
                      + (series_length - 1.0) / (2.0 * (ens_size + 1.0)))
        i = j + 1

    return q


@dataclass
class BNRH:
    """A fitted bounded normal rank histogram distribution.

    Build with :func:`bnrh_fit`; ``cdf`` and ``ppf`` evaluate the fitted
    distribution. ``quantiles`` holds the quantiles of the fitting ensemble
    in its original (unsorted) order.
    """

    sort_x: NDArray[np.float64]
    quantiles: NDArray[np.float64]      # unsorted order, matching the input ensemble
    sorted_quantiles: NDArray[np.float64]
    tail_amp_left: float
    tail_mean_left: float
    tail_sd_left: float
    tail_amp_right: float
    tail_mean_right: float
    tail_sd_right: float
    do_uniform_tail_left: bool
    do_uniform_tail_right: bool
    bounded_below: bool
    bounded_above: bool
    lower_bound: float
    upper_bound: float

    @property
    def ens_size(self) -> int:
        return self.sort_x.size

    def cdf(self, x: float) -> float:
        """Quantile of a value ``x`` in the fitted distribution."""
        return _bnrh_cdf_initialized(x, self)

    def ppf(self, quantiles: ArrayLike) -> NDArray[np.float64]:
        """Inverse cdf for an array of quantiles."""
        return inv_bnrh_cdf(quantiles, self)


def bnrh_fit(x: ArrayLike, bounded_below: bool = False, bounded_above: bool = False,
             lower_bound: float = 0.0, upper_bound: float = 0.0) -> BNRH:
    """Fit a BNRH distribution to the ensemble ``x`` (port of ``bnrh_cdf.m``)."""
    x = np.asarray(x, dtype=float)
    ens_size = x.size

    tail_sd = float(x.std(ddof=1))
    if tail_sd <= 0.0:
        raise ValueError("Cannot fit a BNRH distribution to an ensemble with zero spread")

    sort_index = np.argsort(x)
    sort_x = x[sort_index]

    if bounded_below and sort_x[0] < lower_bound:
        raise ValueError(
            f"Smallest ensemble member {sort_x[0]} less than lower bound {lower_bound}")
    if bounded_above and sort_x[-1] > upper_bound:
        raise ValueError(
            f"Largest ensemble member {sort_x[-1]} greater than upper bound {upper_bound}")

    q = ens_quantiles(sort_x, bounded_below, bounded_above, lower_bound, upper_bound)
    quantiles = np.zeros(ens_size)
    quantiles[sort_index] = q

    # NOTE: the MATLAB original has del_q = 1/(ens_size + 1.8), a typo; the
    # DART Fortran (bnrh_distribution_mod.f90) uses 1/(ens_size + 1), which
    # is used here.
    del_q = 1.0 / (ens_size + 1.0)

    # Distance from the mean of a unit normal to where the cdf is del_q
    dist_for_unit_sd = -float(norm.ppf(del_q))

    # Means put 1/(n+1) probability outside the outermost ensemble members
    tail_mean_left = sort_x[0] + dist_for_unit_sd * tail_sd
    tail_mean_right = sort_x[-1] - dist_for_unit_sd * tail_sd

    # For bounded tails, an amplitude > 1 keeps del_q mass between the bound
    # and the outermost member. If the bound and outermost member are very
    # close in quantile, fall back to a uniform tail.
    uniform_threshold = 0.01
    tail_amp_left = 1.0
    do_uniform_tail_left = False
    if bounded_below:
        bound_quantile = norm.cdf(lower_bound, tail_mean_left, tail_sd)
        if (del_q - bound_quantile) / del_q < uniform_threshold:
            do_uniform_tail_left = True
        else:
            tail_amp_left = del_q / (del_q - bound_quantile)

    tail_amp_right = 1.0
    do_uniform_tail_right = False
    if bounded_above:
        bound_quantile = norm.cdf(upper_bound, tail_mean_right, tail_sd)
        if (bound_quantile - (1.0 - del_q)) / del_q < uniform_threshold:
            do_uniform_tail_right = True
        else:
            tail_amp_right = del_q / (del_q - (1.0 - bound_quantile))

    return BNRH(
        sort_x=sort_x, quantiles=quantiles, sorted_quantiles=q,
        tail_amp_left=tail_amp_left, tail_mean_left=tail_mean_left, tail_sd_left=tail_sd,
        tail_amp_right=tail_amp_right, tail_mean_right=tail_mean_right, tail_sd_right=tail_sd,
        do_uniform_tail_left=do_uniform_tail_left, do_uniform_tail_right=do_uniform_tail_right,
        bounded_below=bounded_below, bounded_above=bounded_above,
        lower_bound=lower_bound, upper_bound=upper_bound,
    )


def _bnrh_cdf_initialized(x: float, dist: BNRH) -> float:
    """Quantile of ``x`` in a previously fitted BNRH (port of ``bnrh_cdf_initialized.m``)."""
    sort_ens = dist.sort_x
    ens_size = dist.ens_size
    q = dist.sorted_quantiles
    del_q = 1.0 / (ens_size + 1.0)

    if x < sort_ens[0]:
        # Left tail
        if dist.bounded_below and x < dist.lower_bound:
            raise ValueError(f"Value {x} less than lower bound {dist.lower_bound}")
        if dist.do_uniform_tail_left:
            quantile = (x - dist.lower_bound) / (sort_ens[0] - dist.lower_bound) * del_q
        elif dist.bounded_below:
            quantile = dist.tail_amp_left * (
                norm.cdf(x, dist.tail_mean_left, dist.tail_sd_left)
                - norm.cdf(dist.lower_bound, dist.tail_mean_left, dist.tail_sd_left))
            quantile = min(quantile, q[0])
        else:
            # Unbounded: normal tail reaches quantile 0 with amplitude 1
            quantile = (norm.cdf(x, dist.tail_mean_left, dist.tail_sd_left)
                        / norm.cdf(sort_ens[0], dist.tail_mean_left, dist.tail_sd_left)
                        ) * del_q
            quantile = min(quantile, q[0])
    elif x == sort_ens[0]:
        quantile = q[0]
    elif x > sort_ens[-1]:
        # Right tail
        if dist.bounded_above and x > dist.upper_bound:
            raise ValueError(f"Value {x} greater than upper bound {dist.upper_bound}")
        if dist.do_uniform_tail_right:
            quantile = (ens_size * del_q
                        + (x - sort_ens[-1]) / (dist.upper_bound - sort_ens[-1]) * del_q)
        else:
            q_at_largest = norm.cdf(sort_ens[-1], dist.tail_mean_right, dist.tail_sd_right)
            if dist.bounded_above:
                upper_q = dist.tail_amp_right * norm.cdf(
                    dist.upper_bound, dist.tail_mean_right, dist.tail_sd_right)
                fract = (dist.tail_amp_right
                         * (norm.cdf(x, dist.tail_mean_right, dist.tail_sd_right)
                            - q_at_largest)) / (upper_q - dist.tail_amp_right * q_at_largest)
            else:
                fract = (norm.cdf(x, dist.tail_mean_right, dist.tail_sd_right)
                         - q_at_largest) / (1.0 - q_at_largest)
            quantile = min(ens_size * del_q + fract * del_q, 1.0)
    else:
        # Interior bin: linear interpolation in the gap containing x
        quantile = q[-1]
        for j in range(ens_size - 1):
            if x < sort_ens[j + 1]:
                quantile = ((j + 1) * del_q
                            + ((x - sort_ens[j]) / (sort_ens[j + 1] - sort_ens[j])) * del_q)
                break
            elif x == sort_ens[j + 1]:
                quantile = q[j + 1]
                break

    return float(quantile)


def bnrh_cdf_initialized(x: float, dist: BNRH) -> float:
    """Quantile of a single value in a fitted BNRH distribution."""
    return _bnrh_cdf_initialized(float(x), dist)


def inv_bnrh_cdf(quantiles: ArrayLike, dist: BNRH) -> NDArray[np.float64]:
    """Inverse cdf of a fitted BNRH distribution (port of ``inv_bnrh_cdf.m``)."""
    quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))
    sort_ens = dist.sort_x
    ens_size = dist.ens_size
    del_q = 1.0 / (ens_size + 1.0)
    # Quantiles at the ensemble members, uniform for BNRH
    q = np.arange(1, ens_size + 1) * del_q

    x = np.zeros(quantiles.size)
    for i, curr_q in enumerate(quantiles):
        # BNRH quantiles are uniform so the region for this quantile is trivial
        region = int(np.floor(curr_q * (ens_size + 1.0)))
        region = min(max(region, 0), ens_size)

        if region == 0:
            # Lower tail
            if dist.bounded_below and dist.do_uniform_tail_left:
                x[i] = dist.lower_bound + (curr_q / q[0]) * (sort_ens[0] - dist.lower_bound)
            else:
                if dist.bounded_below:
                    lower_mass = dist.tail_amp_left * norm.cdf(
                        dist.lower_bound, dist.tail_mean_left, dist.tail_sd_left)
                else:
                    lower_mass = 0.0
                upper_mass = dist.tail_amp_left * norm.cdf(
                    sort_ens[0], dist.tail_mean_left, dist.tail_sd_left)
                fract = curr_q / q[0]
                target_mass = lower_mass + fract * (upper_mass - lower_mass)
                x[i] = weighted_norm_inv(dist.tail_amp_left, dist.tail_mean_left,
                                         dist.tail_sd_left, target_mass)
        elif region == ens_size:
            # Upper tail
            if dist.bounded_above and dist.do_uniform_tail_right:
                x[i] = sort_ens[-1] + ((curr_q - q[-1]) * (dist.upper_bound - sort_ens[-1])
                                       / (1.0 - q[-1]))
            else:
                if dist.bounded_above:
                    upper_mass = dist.tail_amp_right * norm.cdf(
                        dist.upper_bound, dist.tail_mean_right, dist.tail_sd_right)
                else:
                    upper_mass = 1.0
                lower_mass = dist.tail_amp_right * norm.cdf(
                    sort_ens[-1], dist.tail_mean_right, dist.tail_sd_right)
                fract = (curr_q - q[-1]) / (1.0 - q[-1])
                target_mass = lower_mass + fract * (upper_mass - lower_mass)
                x[i] = weighted_norm_inv(dist.tail_amp_right, dist.tail_mean_right,
                                         dist.tail_sd_right, target_mass)
        else:
            # Interior region between members region-1 and region (0-based)
            lower_q = q[region - 1]
            upper_q = q[region]
            x[i] = sort_ens[region - 1] + ((curr_q - lower_q) / (upper_q - lower_q)
                                           ) * (sort_ens[region] - sort_ens[region - 1])

    return x


def rhf_pdf_points(x: ArrayLike, mass: ArrayLike, left_mean: float, left_sd: float,
                   left_amp: float, right_mean: float, right_sd: float, right_amp: float,
                   bounded_left: bool = False
                   ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(x, y)`` points tracing a rank histogram filter PDF, for plotting.

    Port of ``plot_rhf_pdf.m``. ``x`` is the sorted ensemble and ``mass`` the
    probability mass in each of the ``n+1`` bins.
    """
    x = np.asarray(x, dtype=float)
    mass = np.asarray(mass, dtype=float)
    ens_size = x.size

    # Plot 5 standard deviations of the tail distributions
    xlow = 0.0 if bounded_left else left_mean - 5.0 * left_sd
    xhigh = right_mean + 5.0 * right_sd
    point_del = (xhigh - xlow) / 1000

    # Left tail curve
    xptsb = np.arange(xlow, x[0], point_del)
    yptsb = left_amp * norm.pdf(xptsb, left_mean, left_sd)

    # Interior bins are flat boxes; h[k] is the height of the bin between
    # x[k-1] and x[k]
    h = np.zeros(ens_size)
    for k in range(1, ens_size):
        h[k] = mass[k] / (x[k] - x[k - 1])

    xpts: list[float] = []
    ypts: list[float] = []
    for k in range(1, ens_size):
        xpts += [x[k - 1], x[k - 1], x[k]]
        if k == 1:
            ypts.append(left_amp * norm.pdf(x[0], left_mean, left_sd))
        else:
            ypts.append(h[k - 1])
        ypts += [h[k], h[k]]

    # Vertical on the right of the last interior bin
    xptsv = [x[-1], x[-1]]
    yptsv = [h[ens_size - 1], right_amp * norm.pdf(x[-1], right_mean, right_sd)]

    # Right tail curve
    xptsa = np.arange(x[-1], xhigh, point_del)
    yptsa = right_amp * norm.pdf(xptsa, right_mean, right_sd)

    xp = np.concatenate([xptsb, xpts, xptsv, xptsa])
    yp = np.concatenate([yptsb, ypts, yptsv, yptsa])
    return xp, yp
