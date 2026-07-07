#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Angular analysis of ARTI/CORSIKA .shw muons.

This updated version preserves the original normalized outputs and also adds
optional absolute-flux figures when an SHW acquisition time is provided.

New feature:
    --shw-time-s <seconds>
If this argument is given, the script assumes an effective area of 1 m^2 and
creates extra absolute-flux tables/figures in units of m^-2 s^-1 sr^-1.

Additional summary figure:
    - Left panel: Reyna/Bugaev model, one color per energy band.
    - Right panel: ARTI SHW, same color convention per energy band.

Plotting convention requested by user:
    - Simulation points are never connected with straight lines.
    - The simulation is shown only as a step histogram, without marker layer.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


E_MU_GEV = 0.1056583755
MUON_IDS = {"0005", "0006"}
CM2_TO_M2 = 1.0e4
AREA_M2_DEFAULT = 1.0


# ----------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------
def set_article_style() -> None:
    """Matplotlib style with clean article-like output."""
    plt.rcParams.update({
        "figure.figsize": (7.2, 4.8),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
    })


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def _step_xy_from_edges(edges: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(edges, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(edges) != len(values) + 1:
        raise ValueError("edges length must be values length + 1")
    ystep = np.r_[values, values[-1]] if len(values) else values
    return edges, ystep


def plot_step_with_markers(
    ax,
    edges: np.ndarray,
    centers: np.ndarray,
    values: np.ndarray,
    yerr: np.ndarray | None = None,
    *,
    label: str,
    color=None,
    marker: str | None = None,
    alpha_step: float = 0.95,
    alpha_markers: float = 1.0,
):
    """
    Draw ARTI/simulation histograms as a single step curve.

    The centers, yerr, marker, and alpha_markers arguments are kept only for
    backward compatibility with older calls. They are intentionally ignored so
    ARTI is not drawn twice and no marker layer is added.
    """
    values = np.asarray(values, dtype=float)
    xstep, ystep = _step_xy_from_edges(edges, values)
    ax.step(xstep, ystep, where="post", color=color, alpha=alpha_step, label=label)


# ----------------------------------------------------------------------
# SHW reader
# ----------------------------------------------------------------------
def read_shw_muons(shw_path: Path) -> pd.DataFrame:
    """Read muons from a 12-column ARTI .shw file."""
    rows = []

    with shw_path.open("r", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 12:
                continue

            pid = parts[0]
            if pid not in MUON_IDS:
                continue

            try:
                px = float(parts[1])
                py = float(parts[2])
                pz = float(parts[3])
                x = float(parts[4])
                y = float(parts[5])
                z = float(parts[6])
            except ValueError:
                continue

            p = math.sqrt(px*px + py*py + pz*pz)
            if p <= 0:
                continue

            theta = math.acos(max(-1.0, min(1.0, pz / p)))
            phi = (math.atan2(py, px) + 2.0*np.pi) % (2.0*np.pi)
            energy_total = math.sqrt(p*p + E_MU_GEV*E_MU_GEV)
            energy_kinetic = energy_total - E_MU_GEV

            rows.append({
                "pid": pid,
                "px_GeV": px,
                "py_GeV": py,
                "pz_GeV": pz,
                "p_GeV": p,
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "theta_rad": theta,
                "theta_deg": math.degrees(theta),
                "phi_rad": phi,
                "phi_deg": math.degrees(phi),
                "E_total_GeV": energy_total,
                "E_kinetic_GeV": energy_kinetic,
            })

    if not rows:
        raise RuntimeError(f"No muons with IDs {sorted(MUON_IDS)} were found in {shw_path}")

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Geometry / histograms
# ----------------------------------------------------------------------
def theta_edges_from_bin_width(bin_width_deg: float, theta_max_deg: float = 90.0) -> np.ndarray:
    n = int(round(theta_max_deg / bin_width_deg))
    return np.deg2rad(np.linspace(0.0, theta_max_deg, n + 1))


def delta_omega_from_theta_edges(theta_edges: np.ndarray, phi_span_rad: float = 2.0*np.pi) -> np.ndarray:
    """Exact solid angle for each theta bin integrated over phi."""
    return phi_span_rad * (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))


def corrected_theta_histogram(theta_rad: np.ndarray, theta_edges: np.ndarray) -> pd.DataFrame:
    """Return raw counts and dN/dOmega in theta bins."""
    counts, _ = np.histogram(theta_rad, bins=theta_edges)
    domega = delta_omega_from_theta_edges(theta_edges)
    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = counts / domega
        intensity_err = np.sqrt(counts) / domega

    return pd.DataFrame({
        "theta_low_deg": np.rad2deg(theta_edges[:-1]),
        "theta_high_deg": np.rad2deg(theta_edges[1:]),
        "theta_center_deg": np.rad2deg(centers),
        "theta_center_rad": centers,
        "count": counts,
        "delta_omega_sr": domega,
        "dN_dOmega": intensity,
        "dN_dOmega_err": intensity_err,
    })


def corrected_mu_histogram(theta_rad: np.ndarray, n_bins: int = 40) -> pd.DataFrame:
    """Return dN/dOmega histogram in mu = cos(theta)."""
    mu = np.cos(theta_rad)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(mu, bins=edges)
    delta_mu = np.diff(edges)
    domega = 2.0*np.pi*delta_mu
    centers = 0.5*(edges[:-1] + edges[1:])

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = counts / domega
        intensity_err = np.sqrt(counts) / domega

    return pd.DataFrame({
        "mu_low": edges[:-1],
        "mu_high": edges[1:],
        "mu_center": centers,
        "theta_center_deg": np.rad2deg(np.arccos(np.clip(centers, 0.0, 1.0))),
        "count": counts,
        "delta_omega_sr": domega,
        "dN_dOmega": intensity,
        "dN_dOmega_err": intensity_err,
    })


def normalize_to_max(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return y*np.nan
    ymax = np.nanmax(y[finite])
    if ymax <= 0:
        return y*np.nan
    return y / ymax


def normalized_hist_columns(df: pd.DataFrame, value_col: str = "dN_dOmega") -> pd.DataFrame:
    out = df.copy()
    y = out[value_col].to_numpy(dtype=float)
    yn = normalize_to_max(y)
    out[f"{value_col}_norm"] = yn

    err_col = f"{value_col}_err"
    if err_col in out.columns:
        ymax = np.nanmax(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else np.nan
        out[f"{err_col}_norm"] = out[err_col] / ymax if np.isfinite(ymax) and ymax > 0 else np.nan

    return out


# ----------------------------------------------------------------------
# Parametric flux models
# ----------------------------------------------------------------------
def cos_corr(theta):
    RE = 6370.0
    h_atm = 15.0
    theta = np.asarray(theta)
    return np.sqrt(1.0 - ((1.0 - np.cos(theta)**2) / (1.0 + h_atm/RE)**2))


def E_hat(E, theta):
    c = cos_corr(theta)
    dE0 = 0.0026 * (1030.0 / c - 120.0)
    return E + dE0


def safe_cos(theta):
    return np.clip(np.cos(theta), 1.0e-8, 1.0)


def Gaisser(E, theta, h):
    """Differential flux Gaisser model with angle and altitude correction."""
    Ag = 0.1258
    Bg = 0.0588
    gamma = 2.56
    Ecr_pi = 100.0
    Ecr_K = 650.0
    rc = 0.0

    c = cos_corr(theta)
    Ehat = E_hat(E, theta)

    Phi_G = Ag * E**(-gamma) * (
        1.0 / (1.0 + Ehat * c / Ecr_pi)
        + Bg / (1.0 + Ehat * c / Ecr_K)
        + rc
    )

    p = np.sqrt(np.maximum(E*E - E_MU_GEV*E_MU_GEV, 0.0))
    h0 = 4900.0 + 750.0 * p
    h_corr = np.exp(-h / h0)

    return Phi_G * h_corr


def ReynaBugaev(E, theta, h):
    """Differential flux Reyna/Bugaev model with altitude correction."""
    p = np.sqrt(np.maximum(E*E - E_MU_GEV*E_MU_GEV, 0.0))
    Ab = 0.00253
    a0 = 0.2455
    a1 = 1.288
    a2 = -0.2555
    a3 = 0.0209

    c = safe_cos(theta)
    pcos = np.maximum(p*c, 1.0e-12)
    y = np.log10(pcos)

    phi_B = Ab * (pcos ** (-(a3*y**3 + a2*y**2 + a1*y + a0)))
    Phi_RB = (c**3) * phi_B

    h0 = 4900.0 + 750.0 * p
    h_corr = np.exp(-h / h0)

    return Phi_RB * h_corr


def ReynaHebbeker(E, theta, h):
    """Differential flux Reyna/Hebbeker model with altitude correction."""
    p = np.sqrt(np.maximum(E*E - E_MU_GEV*E_MU_GEV, 0.0))
    c = safe_cos(theta)
    pcos = np.maximum(p*c, 1.0e-12)
    y = np.log10(pcos)

    Ah = 0.00253
    h1 = 0.144
    h2 = -2.51
    h3 = -5.76
    s2 = -2.22

    H = (
        h1*(y**3 - 5*y**2 + 6*y)/2.0
        + h2*(-2*y**3 + 9*y**2 - 10*y + 3)/3.0
        + h3*(y**3 - 3*y**2 + 2*y)/6.0
        + s2*(y**3 - 6*y**2 + 11*y - 6)/3.0
    )

    Phi_H = Ah * 10**H
    Phi_RH = c**3 * Phi_H

    h0 = 4900.0 + 750.0 * p
    h_corr = np.exp(-h / h0)

    return Phi_RH * h_corr


def GaisserTang(E, theta, h):
    """Differential flux Gaisser/Tang model."""
    p1 = 0.102573
    p2 = -0.068287
    p3 = 0.958633
    p4 = 0.0407253
    p5 = 0.817285

    x = safe_cos(theta)
    num = (x**2 + p1**2 + p2*x**p3 + p4*x**p5)
    denom = (1.0 + p1**2 + p2 + p4)
    ctt = np.sqrt(np.maximum(num / denom, 1.0e-12))

    gamma = 2.7
    E0 = np.asarray(E, dtype=float)

    cond1 = E0 >= 100.0/ctt
    cond2 = (E0 < 100.0/ctt) & (E0 > 1.0/ctt)
    cond3 = E0 <= 1.0/ctt

    Ehat_arr = np.zeros_like(E0)
    AT = np.zeros_like(E0)
    rc = np.zeros_like(E0)

    Ehat_arr = np.where(cond1, E0, Ehat_arr)
    AT = np.where(cond1, 1.0, AT)
    rc = np.where(cond1, 0.0, rc)

    AT_val = 1.1 * (
        (90.0*np.sqrt(np.maximum(ctt - 0.001, 1.0e-12))) / 1030.0
    ) ** (4.5/(E0*ctt))
    Delta = 2.06e-3 * (950.0/ctt - 90.0)

    Ehat_arr = np.where(cond2, E0 + Delta, Ehat_arr)
    AT = np.where(cond2, AT_val, AT)
    rc = np.where(cond2, 1.0e-4, rc)

    Ehat_low = (3.0*E0 + 7.0*(1.0/ctt))/10.0

    Ehat_arr = np.where(cond3, Ehat_low, Ehat_arr)
    AT = np.where(cond3, AT_val, AT)
    rc = np.where(cond3, 1.0e-4, rc)

    Phi = AT * 0.14 * E0**(-gamma) * (
        1.0/(1.0 + (1.1*(Ehat_arr*ctt)/115.0))
        + 0.054/(1.0 + (1.1*(Ehat_arr*ctt)/810.0))
        + rc
    )

    return Phi


MODEL_FUNCS = {
    "Reyna/Bugaev": ReynaBugaev,
    "Reyna/Hebbeker": ReynaHebbeker,
    "Gaisser/Tang": GaisserTang,
}


# ----------------------------------------------------------------------
# Model utilities
# ----------------------------------------------------------------------
def integrate_model_over_energy(model_func, theta_array, h, emin, emax, n_grid):
    """Energy integral of model_func(E, theta, h)."""
    egrid = np.logspace(np.log10(emin), np.log10(emax), n_grid)
    values = []
    for th in theta_array:
        y = model_func(egrid, th, h)
        y = np.where(np.isfinite(y), y, 0.0)
        values.append(np.trapezoid(y, egrid))
    return np.asarray(values)


def integrate_model_over_energy_absolute(model_func, theta_array, h, emin, emax, n_grid):
    """Integrated model flux in m^-2 s^-1 sr^-1."""
    emin_eff = max(float(emin), E_MU_GEV * 1.000001)
    emax_eff = float(emax)
    if emax_eff <= emin_eff:
        return np.full_like(theta_array, np.nan, dtype=float)
    egrid = np.logspace(np.log10(emin_eff), np.log10(emax_eff), n_grid)
    values = []
    for th in theta_array:
        y = model_func(egrid, th, h)
        y = np.where(np.isfinite(y), y, 0.0)
        values.append(np.trapezoid(y, egrid) * CM2_TO_M2)
    return np.asarray(values)


def model_shapes_at_fixed_energy(theta_array, energy, h):
    """Return normalized model shapes per solid angle at fixed total energy."""
    out = {}
    Earr = np.full_like(theta_array, energy, dtype=float)
    for name, func in MODEL_FUNCS.items():
        y = func(Earr, theta_array, h)
        y = np.where(np.isfinite(y), y, np.nan)
        out[name] = normalize_to_max(y)
    return out


def model_raw_theta_shapes_at_fixed_energy(theta_edges, energy, h):
    """
    Return normalized expected raw theta-bin shapes:
        counts_i ∝ Phi(E, theta_center) * DeltaOmega_i.
    This is the correct object to compare with uncorrected theta histograms.
    """
    centers = 0.5*(theta_edges[:-1] + theta_edges[1:])
    domega = delta_omega_from_theta_edges(theta_edges)
    out = {}
    Earr = np.full_like(centers, energy, dtype=float)

    for name, func in MODEL_FUNCS.items():
        y = func(Earr, centers, h) * domega
        y = np.where(np.isfinite(y), y, np.nan)
        out[name] = normalize_to_max(y)

    return out


# ----------------------------------------------------------------------
# Absolute-flux utilities
# ----------------------------------------------------------------------
def flux_from_counts(counts: np.ndarray, domega: np.ndarray, time_s: float, area_m2: float = AREA_M2_DEFAULT):
    counts = np.asarray(counts, dtype=float)
    domega = np.asarray(domega, dtype=float)
    denom = area_m2 * time_s * domega
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = counts / denom
        err = np.sqrt(counts) / denom
    return flux, err


def build_absolute_overall_table(df, theta_edges, time_s, area_m2, altitude_m, emin, emax, n_grid):
    domega = delta_omega_from_theta_edges(theta_edges)
    centers = 0.5*(theta_edges[:-1] + theta_edges[1:])

    h_all = corrected_theta_histogram(df["theta_rad"].to_numpy(), theta_edges)
    sel_match = df[(df["E_total_GeV"] >= emin) & (df["E_total_GeV"] < emax)]
    h_match = corrected_theta_histogram(sel_match["theta_rad"].to_numpy(), theta_edges)

    flux_all, err_all = flux_from_counts(h_all["count"].to_numpy(), domega, time_s, area_m2)
    flux_match, err_match = flux_from_counts(h_match["count"].to_numpy(), domega, time_s, area_m2)

    out = pd.DataFrame({
        "theta_low_deg": h_all["theta_low_deg"],
        "theta_high_deg": h_all["theta_high_deg"],
        "theta_center_deg": h_all["theta_center_deg"],
        "theta_center_rad": centers,
        "delta_omega_sr": domega,
        "count_allE": h_all["count"],
        "sim_flux_allE_m2_s_sr": flux_all,
        "sim_flux_allE_err_m2_s_sr": err_all,
        "count_match_model_range": h_match["count"],
        "sim_flux_match_model_range_m2_s_sr": flux_match,
        "sim_flux_match_model_range_err_m2_s_sr": err_match,
    })

    for name, func in {"Gaisser": Gaisser, **MODEL_FUNCS}.items():
        clean = name.replace("/", "_").replace(" ", "_")
        out[f"model_{clean}_m2_s_sr"] = integrate_model_over_energy_absolute(
            func, centers, altitude_m, emin, emax, n_grid
        )

    return out


def build_absolute_energy_band_table(df, energy_bins, theta_edges, time_s, area_m2, altitude_m, n_grid):
    centers = 0.5*(theta_edges[:-1] + theta_edges[1:])
    domega = delta_omega_from_theta_edges(theta_edges)
    rows = []

    for lo, hi in energy_bins:
        sub = df[(df["E_total_GeV"] >= lo) & (df["E_total_GeV"] < hi)]
        hdf = corrected_theta_histogram(sub["theta_rad"].to_numpy(), theta_edges)
        flux, err = flux_from_counts(hdf["count"].to_numpy(), domega, time_s, area_m2)

        row = pd.DataFrame({
            "E_low_GeV": lo,
            "E_high_GeV": hi,
            "N_band": len(sub),
            "E_median_GeV": float(np.median(sub["E_total_GeV"])) if len(sub) else np.nan,
            "theta_low_deg": hdf["theta_low_deg"],
            "theta_high_deg": hdf["theta_high_deg"],
            "theta_center_deg": hdf["theta_center_deg"],
            "theta_center_rad": centers,
            "delta_omega_sr": domega,
            "count": hdf["count"],
            "sim_flux_m2_s_sr": flux,
            "sim_flux_err_m2_s_sr": err,
        })

        for name, func in {"Gaisser": Gaisser, **MODEL_FUNCS}.items():
            clean = name.replace("/", "_").replace(" ", "_")
            row[f"model_{clean}_m2_s_sr"] = integrate_model_over_energy_absolute(
                func, centers, altitude_m, lo, hi, n_grid
            )

        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------
# Plotting functions (normalized/original outputs)
# ----------------------------------------------------------------------
def plot_overall_theta_corrected(hist_df, h, emin, emax, n_grid, outdir):
    theta = hist_df["theta_center_rad"].to_numpy()
    hist_df = normalized_hist_columns(hist_df)

    model_columns = {}
    for name, func in MODEL_FUNCS.items():
        vals = integrate_model_over_energy(func, theta, h, emin, emax, n_grid)
        model_columns[name] = normalize_to_max(vals)
        hist_df[f"model_{name}_integrated_norm"] = model_columns[name]

    fig, ax = plt.subplots()
    ok = hist_df["count"].to_numpy() > 0

    plot_step_with_markers(
        ax,
        edges=np.r_[hist_df["theta_low_deg"].to_numpy(), hist_df["theta_high_deg"].iloc[-1]],
        centers=hist_df["theta_center_deg"].to_numpy(),
        values=hist_df["dN_dOmega_norm"].to_numpy(),
        yerr=hist_df["dN_dOmega_err_norm"].to_numpy(),
        label="ARTI SHW, corrected by ΔΩ",
    )

    for name, vals in model_columns.items():
        ax.plot(hist_df["theta_center_deg"], vals, label=name)

    ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
    ax.set_ylabel(r"Normalized $dN/d\Omega$")
    ax.set_title("Solid-angle-corrected angular distribution")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()

    png = outdir / "overall_theta_corrected_vs_energy_integrated_models.png"
    csv = outdir / "overall_theta_corrected_vs_energy_integrated_models.csv"
    fig.savefig(png)
    plt.close(fig)
    hist_df.to_csv(csv, index=False)
    return png, csv


def plot_overall_mu_corrected(mu_df, h, emin, emax, n_grid, outdir):
    mu_df = normalized_hist_columns(mu_df)
    theta = np.arccos(np.clip(mu_df["mu_center"].to_numpy(), 0.0, 1.0))

    model_columns = {}
    for name, func in MODEL_FUNCS.items():
        vals = integrate_model_over_energy(func, theta, h, emin, emax, n_grid)
        model_columns[name] = normalize_to_max(vals)
        mu_df[f"model_{name}_integrated_norm"] = model_columns[name]

    fig, ax = plt.subplots()
    ok = mu_df["count"].to_numpy() > 0

    plot_step_with_markers(
        ax,
        edges=np.r_[mu_df["mu_low"].to_numpy(), mu_df["mu_high"].iloc[-1]],
        centers=mu_df["mu_center"].to_numpy(),
        values=mu_df["dN_dOmega_norm"].to_numpy(),
        yerr=mu_df["dN_dOmega_err_norm"].to_numpy(),
        label=r"ARTI SHW, corrected by $\Delta\Omega$",
    )

    for name, vals in model_columns.items():
        ax.plot(mu_df["mu_center"], vals, label=name)

    ax.set_xlabel(r"$\mu = \cos\theta$")
    ax.set_ylabel(r"Normalized $dN/d\Omega$")
    ax.set_title(r"Corrected angular distribution in $\mu$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()

    png = outdir / "overall_mu_corrected_vs_energy_integrated_models.png"
    csv = outdir / "overall_mu_corrected_vs_energy_integrated_models.csv"
    fig.savefig(png)
    plt.close(fig)
    mu_df.to_csv(csv, index=False)
    return png, csv


def plot_raw_vs_corrected(hist_df, outdir):
    df = normalized_hist_columns(hist_df)
    raw = df["count"].to_numpy(dtype=float)
    raw_norm = normalize_to_max(raw)
    edges = np.r_[df["theta_low_deg"].to_numpy(), df["theta_high_deg"].iloc[-1]]

    fig, ax = plt.subplots()
    plot_step_with_markers(
        ax,
        edges=edges,
        centers=df["theta_center_deg"].to_numpy(),
        values=raw_norm,
        yerr=None,
        label=r"Raw $\theta$ histogram",
    )
    plot_step_with_markers(
        ax,
        edges=edges,
        centers=df["theta_center_deg"].to_numpy(),
        values=df["dN_dOmega_norm"].to_numpy(),
        yerr=None,
        label=r"Corrected by $\Delta\Omega$",
    )
    ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
    ax.set_ylabel("Normalized distribution")
    ax.set_title("Effect of the solid-angle correction")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()

    png = outdir / "overall_theta_raw_vs_solid_angle_corrected.png"
    fig.savefig(png)
    plt.close(fig)
    return png


def plot_arti_energy_slices(df, energy_bins, theta_edges, outdir):
    """One figure comparing only ARTI corrected angular distributions by energy band."""
    fig, ax = plt.subplots()
    rows = []

    for lo, hi in energy_bins:
        sub = df[(df["E_total_GeV"] >= lo) & (df["E_total_GeV"] < hi)]
        if len(sub) < 20:
            continue

        hdf = corrected_theta_histogram(sub["theta_rad"].to_numpy(), theta_edges)
        hdf = normalized_hist_columns(hdf)

        label = f"{lo:g}–{hi:g} GeV, N={len(sub)}"
        ax.step(
            np.r_[hdf["theta_low_deg"].to_numpy(), hdf["theta_high_deg"].iloc[-1]],
            np.r_[hdf["dN_dOmega_norm"].to_numpy(), hdf["dN_dOmega_norm"].iloc[-1]],
            where="post",
            label=label,
        )
        # Step only: do not add a second marker layer for ARTI.

        tmp = hdf.copy()
        tmp["E_low_GeV"] = lo
        tmp["E_high_GeV"] = hi
        tmp["N_band"] = len(sub)
        tmp["E_median_GeV"] = float(np.median(sub["E_total_GeV"]))
        rows.append(tmp)

    ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
    ax.set_ylabel(r"Normalized $dN/d\Omega$")
    ax.set_title("ARTI SHW angular distribution by energy band")
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()

    png = outdir / "arti_energy_slices_corrected.png"
    csv = outdir / "arti_energy_slices_corrected.csv"
    fig.savefig(png)
    plt.close(fig)

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(csv, index=False)
    else:
        pd.DataFrame().to_csv(csv, index=False)

    return png, csv


def plot_energy_band_vs_models(df, energy_bins, theta_edges, outdir):
    """
    For each energy band:
    1. Corrected comparison: ARTI dN/dOmega vs Phi(E_med, theta).
    2. Raw comparison: ARTI raw theta histogram vs Phi(E_med, theta)*DeltaOmega.
    """
    outputs = []
    summary_rows = []

    centers = 0.5*(theta_edges[:-1] + theta_edges[1:])
    centers_deg = np.rad2deg(centers)
    edges_deg = np.rad2deg(theta_edges)

    for lo, hi in energy_bins:
        sub = df[(df["E_total_GeV"] >= lo) & (df["E_total_GeV"] < hi)]
        n_band = len(sub)

        if n_band < 20:
            summary_rows.append({
                "E_low_GeV": lo,
                "E_high_GeV": hi,
                "N_band": n_band,
                "E_median_GeV": np.nan,
                "status": "skipped: too few events",
            })
            continue

        E_med = float(np.median(sub["E_total_GeV"]))
        hdf = corrected_theta_histogram(sub["theta_rad"].to_numpy(), theta_edges)
        hdf = normalized_hist_columns(hdf)

        # ---------------- corrected comparison ----------------
        model_shapes = model_shapes_at_fixed_energy(centers, E_med, h=0.0)

        fig, ax = plt.subplots()

        plot_step_with_markers(
            ax,
            edges=edges_deg,
            centers=hdf["theta_center_deg"].to_numpy(),
            values=hdf["dN_dOmega_norm"].to_numpy(),
            yerr=hdf["dN_dOmega_err_norm"].to_numpy(),
            label=f"ARTI SHW, {lo:g}–{hi:g} GeV, N={n_band}",
        )

        for name, vals in model_shapes.items():
            ax.plot(centers_deg, vals, label=f"{name} at E_med={E_med:.2g} GeV")

        ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
        ax.set_ylabel(r"Normalized $dN/d\Omega$")
        ax.set_title(f"Corrected angular shape, E = {lo:g}–{hi:g} GeV")
        ax.set_xlim(0.0, 90.0)
        ax.set_ylim(bottom=0.0)
        ax.legend()
        fig.tight_layout()

        tag = f"E_{str(lo).replace('.', 'p')}_{str(hi).replace('.', 'p')}_GeV"
        png_corr = outdir / f"energy_band_{tag}_corrected_vs_models.png"
        fig.savefig(png_corr)
        plt.close(fig)

        table_corr = hdf.copy()
        table_corr["E_low_GeV"] = lo
        table_corr["E_high_GeV"] = hi
        table_corr["E_median_GeV"] = E_med
        table_corr["N_band"] = n_band
        for name, vals in model_shapes.items():
            table_corr[f"model_{name}_fixedE_norm"] = vals

        csv_corr = outdir / f"energy_band_{tag}_corrected_vs_models.csv"
        table_corr.to_csv(csv_corr, index=False)

        outputs.append(png_corr)
        outputs.append(csv_corr)

        # ---------------- raw comparison with Jacobian ----------------
        raw_counts = hdf["count"].to_numpy(dtype=float)
        raw_norm = normalize_to_max(raw_counts)
        raw_model_shapes = model_raw_theta_shapes_at_fixed_energy(theta_edges, E_med, h=0.0)

        fig, ax = plt.subplots()
        plot_step_with_markers(
            ax,
            edges=edges_deg,
            centers=centers_deg,
            values=raw_norm,
            yerr=None,
            label=f"ARTI raw θ, {lo:g}–{hi:g} GeV, N={n_band}",
        )
        for name, vals in raw_model_shapes.items():
            ax.plot(centers_deg, vals, label=fr"{name} $\times \Delta\Omega$")

        ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
        ax.set_ylabel("Normalized raw-bin shape")
        ax.set_title(f"Raw θ-bin shape including the Jacobian, E = {lo:g}–{hi:g} GeV")
        ax.set_xlim(0.0, 90.0)
        ax.set_ylim(bottom=0.0)
        ax.legend()
        fig.tight_layout()

        png_raw = outdir / f"energy_band_{tag}_raw_theta_with_jacobian_vs_models.png"
        fig.savefig(png_raw)
        plt.close(fig)

        table_raw = pd.DataFrame({
            "theta_low_deg": np.rad2deg(theta_edges[:-1]),
            "theta_high_deg": np.rad2deg(theta_edges[1:]),
            "theta_center_deg": centers_deg,
            "count": raw_counts,
            "count_norm": raw_norm,
            "delta_omega_sr": delta_omega_from_theta_edges(theta_edges),
            "E_low_GeV": lo,
            "E_high_GeV": hi,
            "E_median_GeV": E_med,
            "N_band": n_band,
        })
        for name, vals in raw_model_shapes.items():
            table_raw[f"model_{name}_times_deltaOmega_norm"] = vals

        csv_raw = outdir / f"energy_band_{tag}_raw_theta_with_jacobian_vs_models.csv"
        table_raw.to_csv(csv_raw, index=False)

        outputs.append(png_raw)
        outputs.append(csv_raw)

        summary_rows.append({
            "E_low_GeV": lo,
            "E_high_GeV": hi,
            "N_band": n_band,
            "E_median_GeV": E_med,
            "theta_bin_width_deg": float(np.rad2deg(theta_edges[1] - theta_edges[0])),
            "status": "ok",
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = outdir / "energy_band_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)

    return outputs


# ----------------------------------------------------------------------
# New absolute-flux plots
# ----------------------------------------------------------------------
def plot_overall_theta_absolute_flux(df_abs: pd.DataFrame, emin: float, emax: float, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    edges_deg = np.r_[df_abs["theta_low_deg"].to_numpy(), df_abs["theta_high_deg"].iloc[-1]]
    centers_deg = df_abs["theta_center_deg"].to_numpy()

    plot_step_with_markers(
        ax,
        edges=edges_deg,
        centers=centers_deg,
        values=df_abs["sim_flux_allE_m2_s_sr"].to_numpy(),
        yerr=df_abs["sim_flux_allE_err_m2_s_sr"].to_numpy(),
        label="ARTI SHW, all energies",
    )
    plot_step_with_markers(
        ax,
        edges=edges_deg,
        centers=centers_deg,
        values=df_abs["sim_flux_match_model_range_m2_s_sr"].to_numpy(),
        yerr=df_abs["sim_flux_match_model_range_err_m2_s_sr"].to_numpy(),
        label=f"ARTI SHW, matched to {emin:g}–{emax:g} GeV",
    )

    for col in [c for c in df_abs.columns if c.startswith("model_") and c.endswith("_m2_s_sr")]:
        label = col.replace("model_", "").replace("_m2_s_sr", "").replace("_", "/")
        ax.plot(centers_deg, df_abs[col], label=label)

    ax.set_yscale("log")
    ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
    ax.set_ylabel(r"Integrated angular flux (m$^{-2}$ s$^{-1}$ sr$^{-1}$)")
    ax.set_title("Absolute angular flux: SHW vs integrated models")
    ax.set_xlim(0.0, 90.0)
    ax.legend(ncol=2)
    fig.tight_layout()

    png = outdir / "overall_theta_absolute_flux_vs_models_log.png"
    fig.savefig(png)
    plt.close(fig)
    return png


def plot_absolute_energy_bands_vs_models(df_abs_bands: pd.DataFrame, energy_bins: List[Tuple[float, float]], outdir: Path):
    n_panels = len(energy_bins)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 4.0*nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (lo, hi) in zip(axes, energy_bins):
        sub = df_abs_bands[(df_abs_bands["E_low_GeV"] == lo) & (df_abs_bands["E_high_GeV"] == hi)]
        if sub.empty:
            ax.set_visible(False)
            continue

        edges_deg = np.r_[sub["theta_low_deg"].to_numpy(), sub["theta_high_deg"].iloc[-1]]
        centers_deg = sub["theta_center_deg"].to_numpy()

        plot_step_with_markers(
            ax,
            edges=edges_deg,
            centers=centers_deg,
            values=sub["sim_flux_m2_s_sr"].to_numpy(),
            yerr=sub["sim_flux_err_m2_s_sr"].to_numpy(),
            label=f"ARTI SHW, N={int(sub['N_band'].iloc[0])}",
            )

        for col in [c for c in sub.columns if c.startswith("model_") and c.endswith("_m2_s_sr")]:
            label = col.replace("model_", "").replace("_m2_s_sr", "").replace("_", "/")
            ax.plot(centers_deg, sub[col], label=label)

        ax.set_yscale("log")
        ax.set_title(f"{lo:g}–{hi:g} GeV")
        ax.set_ylabel(r"m$^{-2}$ s$^{-1}$ sr$^{-1}$")
        ax.set_xlim(0.0, 90.0)
        ax.legend(fontsize=7)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    for ax in axes[max(0, n_panels - ncols):n_panels]:
        ax.set_xlabel(r"Zenith angle $\theta$ (deg)")

    fig.suptitle("Absolute angular flux by energy band", y=0.995)
    fig.tight_layout()
    png = outdir / "absolute_energy_bands_flux_vs_models_log.png"
    fig.savefig(png)
    plt.close(fig)
    return png


def plot_summary_reyna_bugaev_vs_arti(df_abs_bands: pd.DataFrame, energy_bins: List[Tuple[float, float]], outdir: Path):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    for lo, hi in energy_bins:
        sub = df_abs_bands[(df_abs_bands["E_low_GeV"] == lo) & (df_abs_bands["E_high_GeV"] == hi)]
        if sub.empty:
            continue

        label = f"{lo:g}–{hi:g} GeV"
        centers_deg = sub["theta_center_deg"].to_numpy()
        edges_deg = np.r_[sub["theta_low_deg"].to_numpy(), sub["theta_high_deg"].iloc[-1]]

        # Left: Reyna/Bugaev model only.
        model_col = "model_Reyna_Bugaev_m2_s_sr"
        line = ax_l.plot(centers_deg, sub[model_col].to_numpy(), label=label)
        color = line[0].get_color()

        # Right: ARTI using same color, step + separate markers.
        plot_step_with_markers(
            ax_r,
            edges=edges_deg,
            centers=centers_deg,
            values=sub["sim_flux_m2_s_sr"].to_numpy(),
            yerr=sub["sim_flux_err_m2_s_sr"].to_numpy(),
            label=label,
            color=color,
            )

    ax_l.set_yscale("log")
    ax_r.set_yscale("log")

    ax_l.set_title("Reyna/Bugaev by energy band")
    ax_r.set_title("ARTI SHW by energy band")

    for ax in (ax_l, ax_r):
        ax.set_xlim(0.0, 90.0)
        ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
        ax.legend(fontsize=8)

    ax_l.set_ylabel(r"Integrated angular flux (m$^{-2}$ s$^{-1}$ sr$^{-1}$)")
    fig.tight_layout()

    png = outdir / "summary_energy_bands_reyna_bugaev_vs_arti.png"
    fig.savefig(png)
    plt.close(fig)
    return png


# ----------------------------------------------------------------------
# Summary / args helpers
# ----------------------------------------------------------------------
def write_summary(df, outdir, altitude_m, emin, emax):
    summary = pd.DataFrame({
        "quantity": [
            "N_muons",
            "N_mu_minus_0005",
            "N_mu_plus_0006",
            "pz_positive",
            "pz_negative",
            "theta_min_deg",
            "theta_median_deg",
            "theta_max_deg",
            "p_min_GeV",
            "p_median_GeV",
            "p_max_GeV",
            "Etotal_min_GeV",
            "Etotal_median_GeV",
            "Etotal_max_GeV",
            "altitude_used_m",
            "model_energy_integral_min_GeV",
            "model_energy_integral_max_GeV",
        ],
        "value": [
            len(df),
            int((df["pid"] == "0005").sum()),
            int((df["pid"] == "0006").sum()),
            int((df["pz_GeV"] > 0).sum()),
            int((df["pz_GeV"] < 0).sum()),
            float(df["theta_deg"].min()),
            float(df["theta_deg"].median()),
            float(df["theta_deg"].max()),
            float(df["p_GeV"].min()),
            float(df["p_GeV"].median()),
            float(df["p_GeV"].max()),
            float(df["E_total_GeV"].min()),
            float(df["E_total_GeV"].median()),
            float(df["E_total_GeV"].max()),
            altitude_m,
            emin,
            emax,
        ],
    })
    path = outdir / "analysis_summary.csv"
    summary.to_csv(path, index=False)
    return path


def parse_energy_bins(s: str) -> List[Tuple[float, float]]:
    bins = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lohi = chunk.split(":")
        if len(lohi) != 2:
            raise ValueError(f"Bad energy bin '{chunk}'. Use 'low:high,low:high'.")
        lo = float(lohi[0])
        hi = float(lohi[1])
        if hi <= lo:
            raise ValueError(f"Bad energy bin '{chunk}': high must be larger than low.")
        bins.append((lo, hi))
    if not bins:
        raise ValueError("No valid energy bins were provided.")
    return bins


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Solid-angle-corrected angular analysis of ARTI .shw muons."
    )
    parser.add_argument("--shw", required=True, type=Path, help="Input .shw file.")
    parser.add_argument("--outdir", default=Path("angular_analysis_out"), type=Path, help="Output directory.")
    parser.add_argument("--altitude-m", default=893.0, type=float, help="Observation altitude in m a.s.l.")
    parser.add_argument("--theta-bin-deg", default=2.0, type=float, help="Theta bin width for overall plots.")
    parser.add_argument("--slice-theta-bin-deg", default=5.0, type=float, help="Theta bin width for energy-band plots.")
    parser.add_argument("--model-emin-GeV", default=1.0, type=float, help="Minimum total energy for model integration.")
    parser.add_argument("--model-emax-GeV", default=1.0e5, type=float, help="Maximum total energy for model integration.")
    parser.add_argument("--model-grid", default=2500, type=int, help="Log-grid points for energy integration.")
    parser.add_argument(
        "--energy-bins",
        default="0.5:1.5,1.5:4.5,4.5:15,15:50,50:200",
        help="Energy bands in total GeV, format 'low:high,low:high'.",
    )
    parser.add_argument(
        "--shw-time-s",
        default=None,
        type=float,
        help="If provided, generate extra absolute-flux outputs assuming an effective area of 1 m^2.",
    )
    args = parser.parse_args(argv)

    set_article_style()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.shw.exists():
        raise FileNotFoundError(f"Input .shw not found: {args.shw}")

    print(f"[1] Reading {args.shw}")
    df = read_shw_muons(args.shw)
    df.to_csv(args.outdir / "muons_angles_energy_table.csv", index=False)

    summary_path = write_summary(
        df,
        args.outdir,
        args.altitude_m,
        args.model_emin_GeV,
        args.model_emax_GeV,
    )

    print(f"[2] Muons read: {len(df)}")
    print(f"    Summary: {summary_path}")

    print("[3] Overall angular histograms")
    theta_edges = theta_edges_from_bin_width(args.theta_bin_deg)
    overall_hist = corrected_theta_histogram(df["theta_rad"].to_numpy(), theta_edges)
    overall_mu = corrected_mu_histogram(df["theta_rad"].to_numpy(), n_bins=40)

    outputs = []
    outputs.extend(plot_overall_theta_corrected(
        overall_hist,
        h=args.altitude_m,
        emin=args.model_emin_GeV,
        emax=args.model_emax_GeV,
        n_grid=args.model_grid,
        outdir=args.outdir,
    ))
    outputs.extend(plot_overall_mu_corrected(
        overall_mu,
        h=args.altitude_m,
        emin=args.model_emin_GeV,
        emax=args.model_emax_GeV,
        n_grid=args.model_grid,
        outdir=args.outdir,
    ))
    outputs.append(plot_raw_vs_corrected(overall_hist, args.outdir))

    print("[4] Energy-band angular comparisons")
    energy_bins = parse_energy_bins(args.energy_bins)
    slice_edges = theta_edges_from_bin_width(args.slice_theta_bin_deg)
    outputs.extend(plot_arti_energy_slices(df, energy_bins, slice_edges, args.outdir))
    outputs.extend(plot_energy_band_vs_models(df, energy_bins, slice_edges, args.outdir))

    # Optional absolute-flux outputs.
    if args.shw_time_s is not None:
        if args.shw_time_s <= 0:
            raise ValueError("--shw-time-s must be positive.")

        print("[5] Absolute-flux outputs")
        abs_overall = build_absolute_overall_table(
            df=df,
            theta_edges=slice_edges,
            time_s=args.shw_time_s,
            area_m2=AREA_M2_DEFAULT,
            altitude_m=args.altitude_m,
            emin=args.model_emin_GeV,
            emax=args.model_emax_GeV,
            n_grid=args.model_grid,
        )
        abs_overall_csv = args.outdir / "absolute_overall_theta_flux.csv"
        abs_overall.to_csv(abs_overall_csv, index=False)
        outputs.append(abs_overall_csv)

        abs_bands = build_absolute_energy_band_table(
            df=df,
            energy_bins=energy_bins,
            theta_edges=slice_edges,
            time_s=args.shw_time_s,
            area_m2=AREA_M2_DEFAULT,
            altitude_m=args.altitude_m,
            n_grid=args.model_grid,
        )
        abs_bands_csv = args.outdir / "absolute_energy_band_theta_flux.csv"
        abs_bands.to_csv(abs_bands_csv, index=False)
        outputs.append(abs_bands_csv)

        abs_run_summary = pd.DataFrame({
            "quantity": ["shw_time_s", "assumed_area_m2", "absolute_theta_bin_deg"],
            "value": [args.shw_time_s, AREA_M2_DEFAULT, args.slice_theta_bin_deg],
        })
        abs_run_summary_csv = args.outdir / "absolute_run_summary.csv"
        abs_run_summary.to_csv(abs_run_summary_csv, index=False)
        outputs.append(abs_run_summary_csv)

        outputs.append(plot_overall_theta_absolute_flux(
            abs_overall,
            emin=args.model_emin_GeV,
            emax=args.model_emax_GeV,
            outdir=args.outdir,
        ))
        outputs.append(plot_absolute_energy_bands_vs_models(
            abs_bands,
            energy_bins=energy_bins,
            outdir=args.outdir,
        ))
        outputs.append(plot_summary_reyna_bugaev_vs_arti(
            abs_bands,
            energy_bins=energy_bins,
            outdir=args.outdir,
        ))

    manifest = pd.DataFrame({"output": [str(p) for p in outputs]})
    manifest_path = args.outdir / "outputs_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("[OK] Analysis finished")
    print(f"Output directory: {args.outdir}")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
