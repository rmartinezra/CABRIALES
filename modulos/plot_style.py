#!/usr/bin/env python3
"""Shared scientific plotting style for CABRIALES figures."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


COUNTS_CMAP = "cividis"
DIVERGING_CMAP = "coolwarm"
CONTOUR_COLOR = "#00bcd4"


def apply_scientific_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "regular",
        "axes.linewidth": 0.85,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.86",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.70,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "image.cmap": COUNTS_CMAP,
    })


def finite_percentile(values, percentile: float, positive_only: bool = False, fallback: float | None = None):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if positive_only:
        arr = arr[arr > 0.0]
    if arr.size == 0:
        return fallback
    out = float(np.nanpercentile(arr, percentile))
    if not np.isfinite(out) or out <= 0.0:
        return fallback
    return out


def format_angular_axes(ax, theta_min: float, theta_max: float, phi_min: float, phi_max: float) -> None:
    ax.set_xlim(phi_min, phi_max)
    ax.set_ylim(theta_max, theta_min)
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_on()


def add_inside_contour(ax, phi, theta, mask, label: str | None = None) -> None:
    try:
        cs = ax.contour(
            phi,
            theta,
            np.asarray(mask, dtype=float),
            levels=[0.5],
            colors=CONTOUR_COLOR,
            linewidths=0.75,
        )
        if label:
            cs.collections[0].set_label(label)
    except Exception:
        return


def style_colorbar(cbar, label: str) -> None:
    cbar.set_label(label)
    cbar.outline.set_linewidth(0.7)
    cbar.ax.tick_params(direction="in", length=3.0, width=0.7)
