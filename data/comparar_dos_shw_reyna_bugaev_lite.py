#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-light comparison of two ARTI/CORSIKA .shw files against Reyna/Bugaev.

This version does NOT load all muons into memory. It streams each .shw file
line by line and only accumulates angular/energy histograms.

Expected .shw 12-column format:
    CorsikaId px py pz x y z shower_id prm_id prm_energy prm_theta prm_phi

Muon IDs accepted:
    0005 / 5 : mu+
    0006 / 6 : mu-

Main normalization:
    flux(theta) = counts / (area_m2 * time_s * DeltaOmega)

Units:
    flux in m^-2 s^-1 sr^-1
    model curves integrated in energy and converted from cm^-2 to m^-2
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


E_MU_GEV = 0.1056583755
MUON_IDS = {"0005", "0006"}
CM2_TO_M2 = 1.0e4


# ----------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------
def set_article_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (7.6, 5.0),
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


def step_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return np.r_[values, values[-1]]


def positive_or_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y) & (y > 0.0), y, np.nan)


def normalize_to_max(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    good = np.isfinite(y) & (y > 0.0)
    if not np.any(good):
        return np.full_like(y, np.nan, dtype=float)
    return y / np.nanmax(y[good])


# ----------------------------------------------------------------------
# Binning / geometry
# ----------------------------------------------------------------------
def theta_edges_from_bin_width(theta_bin_deg: float, theta_max_deg: float) -> np.ndarray:
    if theta_bin_deg <= 0:
        raise ValueError("theta bin width must be positive")
    if theta_max_deg <= 0 or theta_max_deg > 180:
        raise ValueError("theta max must be in (0, 180]")
    n = int(round(theta_max_deg / theta_bin_deg))
    if not np.isclose(n * theta_bin_deg, theta_max_deg):
        raise ValueError("theta_max_deg must be an integer multiple of theta_bin_deg")
    return np.deg2rad(np.linspace(0.0, theta_max_deg, n + 1))


def delta_omega_from_theta_edges(theta_edges: np.ndarray, phi_span_rad: float = 2.0*np.pi) -> np.ndarray:
    return phi_span_rad * (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))


def parse_energy_bins(s: str) -> List[Tuple[float, float]]:
    bins: List[Tuple[float, float]] = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        pieces = chunk.split(":")
        if len(pieces) != 2:
            raise ValueError(f"Bad energy bin '{chunk}'. Use low:high,low:high")
        lo = float(pieces[0])
        hi = float(pieces[1])
        if hi <= lo:
            raise ValueError(f"Bad energy bin '{chunk}': high must be larger than low")
        bins.append((lo, hi))
    if not bins:
        raise ValueError("No energy bins were provided")
    return bins


# ----------------------------------------------------------------------
# Reyna/Bugaev model
# ----------------------------------------------------------------------
def safe_cos(theta: np.ndarray) -> np.ndarray:
    return np.clip(np.cos(theta), 1.0e-8, 1.0)


def reyna_bugaev(E: np.ndarray, theta: np.ndarray, h_m: float) -> np.ndarray:
    """
    Differential muon flux following the Reyna/Bugaev parameterization.

    The returned units follow the usual parameterization scale, cm^-2 s^-1 sr^-1 GeV^-1.
    The altitude factor is the same empirical correction used in the user's previous script.
    """
    E = np.asarray(E, dtype=float)
    theta = np.asarray(theta, dtype=float)

    p = np.sqrt(np.maximum(E*E - E_MU_GEV*E_MU_GEV, 0.0))

    Ab = 0.00253
    a0 = 0.2455
    a1 = 1.288
    a2 = -0.2555
    a3 = 0.0209

    c = safe_cos(theta)
    pcos = np.maximum(p*c, 1.0e-12)
    y = np.log10(pcos)

    phi_b = Ab * (pcos ** (-(a3*y**3 + a2*y**2 + a1*y + a0)))
    phi_rb = (c**3) * phi_b

    h0 = 4900.0 + 750.0 * p
    h_corr = np.exp(-h_m / h0)

    return phi_rb * h_corr


def integrate_reyna_bugaev_energy(theta_array: np.ndarray, h_m: float, emin: float, emax: float, n_grid: int) -> np.ndarray:
    """Energy-integrated Reyna/Bugaev flux in m^-2 s^-1 sr^-1."""
    emin_eff = max(float(emin), E_MU_GEV * 1.000001)
    emax_eff = float(emax)
    if emax_eff <= emin_eff:
        return np.full_like(theta_array, np.nan, dtype=float)

    egrid = np.logspace(np.log10(emin_eff), np.log10(emax_eff), int(n_grid))
    out = []
    for th in theta_array:
        yy = reyna_bugaev(egrid, np.full_like(egrid, th), h_m)
        yy = np.where(np.isfinite(yy), yy, 0.0)
        out.append(np.trapz(yy, egrid) * CM2_TO_M2)
    return np.asarray(out)


# ----------------------------------------------------------------------
# Streaming SHW reader
# ----------------------------------------------------------------------
@dataclass
class StreamResult:
    path: Path
    label: str
    time_s: float
    total_lines: int
    muons_found: int
    muons_used_overall: int
    non_muon_lines: int
    bad_lines: int
    theta_outside: int
    pz_positive: int
    pz_negative: int
    pid_0005: int
    pid_0006: int
    e_min: float
    e_max: float
    theta_min_deg: float
    theta_max_deg: float
    counts_overall: np.ndarray
    counts_bands: np.ndarray
    counts_bands_total: np.ndarray


def read_shw_streaming(
    path: Path,
    label: str,
    time_s: float,
    theta_edges: np.ndarray,
    energy_bins: List[Tuple[float, float]],
    model_emin: float,
    model_emax: float,
    overall_all_energies: bool,
    max_muons: int | None,
    progress_lines: int,
) -> StreamResult:
    ntheta = len(theta_edges) - 1
    nbands = len(energy_bins)

    counts_overall = np.zeros(ntheta, dtype=np.int64)
    counts_bands = np.zeros((nbands, ntheta), dtype=np.int64)
    counts_bands_total = np.zeros(nbands, dtype=np.int64)

    total_lines = 0
    muons_found = 0
    muons_used_overall = 0
    non_muon_lines = 0
    bad_lines = 0
    theta_outside = 0
    pz_positive = 0
    pz_negative = 0
    pid_0005 = 0
    pid_0006 = 0

    e_min = math.inf
    e_max = -math.inf
    theta_min_deg = math.inf
    theta_max_deg = -math.inf

    theta_max_rad = float(theta_edges[-1])

    with path.open("r", errors="ignore") as f:
        for raw in f:
            total_lines += 1

            if progress_lines > 0 and total_lines % progress_lines == 0:
                print(f"    {label}: {total_lines:,} lines read, {muons_found:,} muons found", flush=True)

            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 12:
                bad_lines += 1
                continue

            pid = parts[0].strip().zfill(4)
            if pid not in MUON_IDS:
                non_muon_lines += 1
                continue

            try:
                px = float(parts[1])
                py = float(parts[2])
                pz = float(parts[3])
            except ValueError:
                bad_lines += 1
                continue

            p = math.sqrt(px*px + py*py + pz*pz)
            if not math.isfinite(p) or p <= 0.0:
                bad_lines += 1
                continue

            muons_found += 1
            if pid == "0005":
                pid_0005 += 1
            elif pid == "0006":
                pid_0006 += 1

            if pz > 0:
                pz_positive += 1
            elif pz < 0:
                pz_negative += 1

            cos_th = max(-1.0, min(1.0, pz / p))
            theta = math.acos(cos_th)
            theta_deg = math.degrees(theta)
            e_total = math.sqrt(p*p + E_MU_GEV*E_MU_GEV)

            e_min = min(e_min, e_total)
            e_max = max(e_max, e_total)
            theta_min_deg = min(theta_min_deg, theta_deg)
            theta_max_deg = max(theta_max_deg, theta_deg)

            # Histogram only inside the requested theta range.
            if theta < 0.0 or theta > theta_max_rad:
                theta_outside += 1
                if max_muons is not None and muons_found >= max_muons:
                    break
                continue

            ith = np.searchsorted(theta_edges, theta, side="right") - 1
            if ith < 0 or ith >= ntheta:
                # Includes theta exactly equal to the upper edge.
                if np.isclose(theta, theta_edges[-1]):
                    ith = ntheta - 1
                else:
                    theta_outside += 1
                    if max_muons is not None and muons_found >= max_muons:
                        break
                    continue

            use_overall = overall_all_energies or (model_emin <= e_total < model_emax)
            if use_overall:
                counts_overall[ith] += 1
                muons_used_overall += 1

            for ib, (lo, hi) in enumerate(energy_bins):
                if lo <= e_total < hi:
                    counts_bands_total[ib] += 1
                    counts_bands[ib, ith] += 1
                    break

            if max_muons is not None and muons_found >= max_muons:
                break

    if muons_found == 0:
        raise RuntimeError(f"No muons with IDs {sorted(MUON_IDS)} found in {path}")

    if not np.isfinite(e_min):
        e_min = np.nan
    if not np.isfinite(e_max):
        e_max = np.nan
    if not np.isfinite(theta_min_deg):
        theta_min_deg = np.nan
    if not np.isfinite(theta_max_deg):
        theta_max_deg = np.nan

    return StreamResult(
        path=path,
        label=label,
        time_s=time_s,
        total_lines=total_lines,
        muons_found=muons_found,
        muons_used_overall=muons_used_overall,
        non_muon_lines=non_muon_lines,
        bad_lines=bad_lines,
        theta_outside=theta_outside,
        pz_positive=pz_positive,
        pz_negative=pz_negative,
        pid_0005=pid_0005,
        pid_0006=pid_0006,
        e_min=e_min,
        e_max=e_max,
        theta_min_deg=theta_min_deg,
        theta_max_deg=theta_max_deg,
        counts_overall=counts_overall,
        counts_bands=counts_bands,
        counts_bands_total=counts_bands_total,
    )


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def flux_from_counts(counts: np.ndarray, domega: np.ndarray, time_s: float, area_m2: float) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    denom = area_m2 * float(time_s) * np.asarray(domega, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = counts / denom
        err = np.sqrt(counts) / denom
    return flux, err


def make_overall_table(
    res_a: StreamResult,
    res_b: StreamResult,
    theta_edges: np.ndarray,
    area_m2: float,
    altitude_m: float,
    model_emin: float,
    model_emax: float,
    model_grid: int,
) -> pd.DataFrame:
    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    centers_deg = np.rad2deg(centers)
    domega = delta_omega_from_theta_edges(theta_edges)

    flux_a, err_a = flux_from_counts(res_a.counts_overall, domega, res_a.time_s, area_m2)
    flux_b, err_b = flux_from_counts(res_b.counts_overall, domega, res_b.time_s, area_m2)
    rb = integrate_reyna_bugaev_energy(centers, altitude_m, model_emin, model_emax, model_grid)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_a = flux_a / rb
        ratio_b = flux_b / rb
        ratio_b_to_a = flux_b / flux_a

    return pd.DataFrame({
        "theta_low_deg": np.rad2deg(theta_edges[:-1]),
        "theta_high_deg": np.rad2deg(theta_edges[1:]),
        "theta_center_deg": centers_deg,
        "theta_center_rad": centers,
        "delta_omega_sr": domega,
        f"count_{res_a.label}": res_a.counts_overall,
        f"flux_{res_a.label}_m2_s_sr": flux_a,
        f"flux_err_{res_a.label}_m2_s_sr": err_a,
        f"count_{res_b.label}": res_b.counts_overall,
        f"flux_{res_b.label}_m2_s_sr": flux_b,
        f"flux_err_{res_b.label}_m2_s_sr": err_b,
        "model_Reyna_Bugaev_m2_s_sr": rb,
        f"ratio_{res_a.label}_to_RB": ratio_a,
        f"ratio_{res_b.label}_to_RB": ratio_b,
        f"ratio_{res_b.label}_to_{res_a.label}": ratio_b_to_a,
    })


def make_energy_band_table(
    res_a: StreamResult,
    res_b: StreamResult,
    theta_edges: np.ndarray,
    energy_bins: List[Tuple[float, float]],
    area_m2: float,
    altitude_m: float,
    model_grid: int,
) -> pd.DataFrame:
    rows = []
    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    centers_deg = np.rad2deg(centers)
    domega = delta_omega_from_theta_edges(theta_edges)

    for ib, (lo, hi) in enumerate(energy_bins):
        flux_a, err_a = flux_from_counts(res_a.counts_bands[ib], domega, res_a.time_s, area_m2)
        flux_b, err_b = flux_from_counts(res_b.counts_bands[ib], domega, res_b.time_s, area_m2)
        rb = integrate_reyna_bugaev_energy(centers, altitude_m, lo, hi, model_grid)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_a = flux_a / rb
            ratio_b = flux_b / rb
            ratio_b_to_a = flux_b / flux_a

        rows.append(pd.DataFrame({
            "E_low_GeV": lo,
            "E_high_GeV": hi,
            "N_band_" + res_a.label: res_a.counts_bands_total[ib],
            "N_band_" + res_b.label: res_b.counts_bands_total[ib],
            "theta_low_deg": np.rad2deg(theta_edges[:-1]),
            "theta_high_deg": np.rad2deg(theta_edges[1:]),
            "theta_center_deg": centers_deg,
            "theta_center_rad": centers,
            "delta_omega_sr": domega,
            f"count_{res_a.label}": res_a.counts_bands[ib],
            f"flux_{res_a.label}_m2_s_sr": flux_a,
            f"flux_err_{res_a.label}_m2_s_sr": err_a,
            f"count_{res_b.label}": res_b.counts_bands[ib],
            f"flux_{res_b.label}_m2_s_sr": flux_b,
            f"flux_err_{res_b.label}_m2_s_sr": err_b,
            "model_Reyna_Bugaev_m2_s_sr": rb,
            f"ratio_{res_a.label}_to_RB": ratio_a,
            f"ratio_{res_b.label}_to_RB": ratio_b,
            f"ratio_{res_b.label}_to_{res_a.label}": ratio_b_to_a,
        }))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def make_integrated_band_rates(
    df_bands: pd.DataFrame,
    energy_bins: List[Tuple[float, float]],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    rows = []
    for lo, hi in energy_bins:
        sub = df_bands[(df_bands["E_low_GeV"] == lo) & (df_bands["E_high_GeV"] == hi)]
        if sub.empty:
            continue
        domega = sub["delta_omega_sr"].to_numpy(dtype=float)
        rb = sub["model_Reyna_Bugaev_m2_s_sr"].to_numpy(dtype=float)
        fa = sub[f"flux_{label_a}_m2_s_sr"].to_numpy(dtype=float)
        fb = sub[f"flux_{label_b}_m2_s_sr"].to_numpy(dtype=float)

        rate_rb = np.nansum(rb * domega)
        rate_a = np.nansum(fa * domega)
        rate_b = np.nansum(fb * domega)

        rows.append({
            "E_low_GeV": lo,
            "E_high_GeV": hi,
            f"integrated_flux_{label_a}_m2_s": rate_a,
            f"integrated_flux_{label_b}_m2_s": rate_b,
            "integrated_flux_Reyna_Bugaev_m2_s": rate_rb,
            f"ratio_{label_a}_to_RB": rate_a / rate_rb if rate_rb > 0 else np.nan,
            f"ratio_{label_b}_to_RB": rate_b / rate_rb if rate_rb > 0 else np.nan,
            f"ratio_{label_b}_to_{label_a}": rate_b / rate_a if rate_a > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def write_run_summary(res_a: StreamResult, res_b: StreamResult, outdir: Path, args) -> Path:
    rows = []
    for res in (res_a, res_b):
        rows.extend([
            {"sample": res.label, "quantity": "path", "value": str(res.path)},
            {"sample": res.label, "quantity": "time_s", "value": res.time_s},
            {"sample": res.label, "quantity": "total_lines_read", "value": res.total_lines},
            {"sample": res.label, "quantity": "muons_found", "value": res.muons_found},
            {"sample": res.label, "quantity": "muons_used_overall", "value": res.muons_used_overall},
            {"sample": res.label, "quantity": "pid_0005_mu_plus", "value": res.pid_0005},
            {"sample": res.label, "quantity": "pid_0006_mu_minus", "value": res.pid_0006},
            {"sample": res.label, "quantity": "pz_positive", "value": res.pz_positive},
            {"sample": res.label, "quantity": "pz_negative", "value": res.pz_negative},
            {"sample": res.label, "quantity": "theta_outside_requested_range", "value": res.theta_outside},
            {"sample": res.label, "quantity": "bad_lines", "value": res.bad_lines},
            {"sample": res.label, "quantity": "non_muon_lines", "value": res.non_muon_lines},
            {"sample": res.label, "quantity": "E_total_min_GeV", "value": res.e_min},
            {"sample": res.label, "quantity": "E_total_max_GeV", "value": res.e_max},
            {"sample": res.label, "quantity": "theta_min_deg", "value": res.theta_min_deg},
            {"sample": res.label, "quantity": "theta_max_deg", "value": res.theta_max_deg},
        ])

    rows.extend([
        {"sample": "config", "quantity": "area_m2", "value": args.area_m2},
        {"sample": "config", "quantity": "altitude_m", "value": args.altitude_m},
        {"sample": "config", "quantity": "theta_bin_deg", "value": args.theta_bin_deg},
        {"sample": "config", "quantity": "theta_max_deg", "value": args.theta_max_deg},
        {"sample": "config", "quantity": "model_emin_GeV", "value": args.model_emin_GeV},
        {"sample": "config", "quantity": "model_emax_GeV", "value": args.model_emax_GeV},
        {"sample": "config", "quantity": "overall_all_energies", "value": args.overall_all_energies},
    ])

    path = outdir / "run_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_overall_flux(df: pd.DataFrame, label_a: str, label_b: str, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    edges = np.r_[df["theta_low_deg"].to_numpy(), df["theta_high_deg"].iloc[-1]]
    centers = df["theta_center_deg"].to_numpy()

    ya = positive_or_nan(df[f"flux_{label_a}_m2_s_sr"].to_numpy())
    yb = positive_or_nan(df[f"flux_{label_b}_m2_s_sr"].to_numpy())
    ym = positive_or_nan(df["model_Reyna_Bugaev_m2_s_sr"].to_numpy())

    ax.step(edges, step_values(ya), where="post", label=label_a)
    ax.step(edges, step_values(yb), where="post", label=label_b)
    ax.plot(centers, ym, label="Reyna/Bugaev", color="black")

    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
    ax.set_ylabel(r"Integrated angular flux (m$^{-2}$ s$^{-1}$ sr$^{-1}$)")
    ax.set_title("Two SHW samples vs Reyna/Bugaev")
    ax.legend()
    fig.tight_layout()

    path = outdir / "overall_theta_flux_two_shw_vs_reyna_bugaev.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_overall_shape(df: pd.DataFrame, label_a: str, label_b: str, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    edges = np.r_[df["theta_low_deg"].to_numpy(), df["theta_high_deg"].iloc[-1]]
    centers = df["theta_center_deg"].to_numpy()

    ya = normalize_to_max(df[f"flux_{label_a}_m2_s_sr"].to_numpy())
    yb = normalize_to_max(df[f"flux_{label_b}_m2_s_sr"].to_numpy())
    ym = normalize_to_max(df["model_Reyna_Bugaev_m2_s_sr"].to_numpy())

    ax.step(edges, step_values(ya), where="post", label=label_a)
    ax.step(edges, step_values(yb), where="post", label=label_b)
    ax.plot(centers, ym, label="Reyna/Bugaev", color="black")

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
    ax.set_ylabel("Normalized shape")
    ax.set_title("Angular-shape comparison")
    ax.legend()
    fig.tight_layout()

    path = outdir / "overall_theta_shape_norm_two_shw_vs_reyna_bugaev.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_overall_ratio(df: pd.DataFrame, label_a: str, label_b: str, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    edges = np.r_[df["theta_low_deg"].to_numpy(), df["theta_high_deg"].iloc[-1]]

    ya = positive_or_nan(df[f"ratio_{label_a}_to_RB"].to_numpy())
    yb = positive_or_nan(df[f"ratio_{label_b}_to_RB"].to_numpy())

    ax.step(edges, step_values(ya), where="post", label=f"{label_a}/RB")
    ax.step(edges, step_values(yb), where="post", label=f"{label_b}/RB")
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="unity")

    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
    ax.set_ylabel("Flux ratio to Reyna/Bugaev")
    ax.set_title("Ratio to model")
    ax.legend()
    fig.tight_layout()

    path = outdir / "overall_theta_flux_ratio_to_reyna_bugaev.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_energy_band_flux(df: pd.DataFrame, energy_bins: List[Tuple[float, float]], label_a: str, label_b: str, outdir: Path) -> Path:
    n = len(energy_bins)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 4.1*nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (lo, hi) in zip(axes, energy_bins):
        sub = df[(df["E_low_GeV"] == lo) & (df["E_high_GeV"] == hi)]
        if sub.empty:
            ax.set_visible(False)
            continue
        edges = np.r_[sub["theta_low_deg"].to_numpy(), sub["theta_high_deg"].iloc[-1]]
        centers = sub["theta_center_deg"].to_numpy()
        ya = positive_or_nan(sub[f"flux_{label_a}_m2_s_sr"].to_numpy())
        yb = positive_or_nan(sub[f"flux_{label_b}_m2_s_sr"].to_numpy())
        ym = positive_or_nan(sub["model_Reyna_Bugaev_m2_s_sr"].to_numpy())

        ax.step(edges, step_values(ya), where="post", label=label_a)
        ax.step(edges, step_values(yb), where="post", label=label_b)
        ax.plot(centers, ym, color="black", label="Reyna/Bugaev")
        ax.set_yscale("log")
        ax.set_title(f"{lo:g}–{hi:g} GeV")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylabel(r"m$^{-2}$ s$^{-1}$ sr$^{-1}$")
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n-ncols):n]:
        ax.set_xlabel(r"Zenith angle $\theta$ (deg)")

    fig.suptitle("Energy-band angular flux", y=0.995)
    fig.tight_layout()
    path = outdir / "energy_bands_flux_two_shw_vs_reyna_bugaev.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_energy_band_ratio(df: pd.DataFrame, energy_bins: List[Tuple[float, float]], label_a: str, label_b: str, outdir: Path) -> Path:
    n = len(energy_bins)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 4.1*nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (lo, hi) in zip(axes, energy_bins):
        sub = df[(df["E_low_GeV"] == lo) & (df["E_high_GeV"] == hi)]
        if sub.empty:
            ax.set_visible(False)
            continue
        edges = np.r_[sub["theta_low_deg"].to_numpy(), sub["theta_high_deg"].iloc[-1]]
        ya = positive_or_nan(sub[f"ratio_{label_a}_to_RB"].to_numpy())
        yb = positive_or_nan(sub[f"ratio_{label_b}_to_RB"].to_numpy())

        ax.step(edges, step_values(ya), where="post", label=f"{label_a}/RB")
        ax.step(edges, step_values(yb), where="post", label=f"{label_b}/RB")
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"{lo:g}–{hi:g} GeV")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylabel("ratio")
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n-ncols):n]:
        ax.set_xlabel(r"Zenith angle $\theta$ (deg)")

    fig.suptitle("Energy-band ratio to Reyna/Bugaev", y=0.995)
    fig.tight_layout()
    path = outdir / "energy_bands_ratio_to_reyna_bugaev.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_summary_reyna_bugaev_vs_two_shw(df: pd.DataFrame, energy_bins: List[Tuple[float, float]], label_a: str, label_b: str, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=True)
    ax_l, ax_m, ax_r = axes

    for lo, hi in energy_bins:
        sub = df[(df["E_low_GeV"] == lo) & (df["E_high_GeV"] == hi)]
        if sub.empty:
            continue
        label = f"{lo:g}–{hi:g} GeV"
        centers = sub["theta_center_deg"].to_numpy()
        edges = np.r_[sub["theta_low_deg"].to_numpy(), sub["theta_high_deg"].iloc[-1]]

        line = ax_l.plot(centers, positive_or_nan(sub["model_Reyna_Bugaev_m2_s_sr"].to_numpy()), label=label)
        color = line[0].get_color()
        ax_m.step(edges, step_values(positive_or_nan(sub[f"flux_{label_a}_m2_s_sr"].to_numpy())), where="post", label=label, color=color)
        ax_r.step(edges, step_values(positive_or_nan(sub[f"flux_{label_b}_m2_s_sr"].to_numpy())), where="post", label=label, color=color)

    ax_l.set_title("Reyna/Bugaev")
    ax_m.set_title(label_a)
    ax_r.set_title(label_b)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlim(df["theta_low_deg"].min(), df["theta_high_deg"].max())
        ax.set_xlabel(r"Zenith angle $\theta$ (deg)")
        ax.legend(fontsize=8)

    ax_l.set_ylabel(r"Integrated angular flux (m$^{-2}$ s$^{-1}$ sr$^{-1}$)")
    fig.tight_layout()
    path = outdir / "summary_energy_bands_reyna_bugaev_vs_two_shw.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_integrated_rates(df_rates: pd.DataFrame, label_a: str, label_b: str, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    xlabels = [f"{lo:g}–{hi:g}" for lo, hi in zip(df_rates["E_low_GeV"], df_rates["E_high_GeV"])]
    x = np.arange(len(xlabels))
    width = 0.25

    ax.bar(x - width, df_rates[f"integrated_flux_{label_a}_m2_s"], width, label=label_a)
    ax.bar(x, df_rates[f"integrated_flux_{label_b}_m2_s"], width, label=label_b)
    ax.bar(x + width, df_rates["integrated_flux_Reyna_Bugaev_m2_s"], width, label="Reyna/Bugaev")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_xlabel("Energy band (GeV)")
    ax.set_ylabel(r"Integrated flux over angular range (m$^{-2}$ s$^{-1}$)")
    ax.set_title("Integrated flux by energy band")
    ax.legend()
    fig.tight_layout()

    path = outdir / "energy_band_integrated_rates_two_shw_vs_reyna_bugaev.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Memory-light comparison of two .shw files against Reyna/Bugaev."
    )
    parser.add_argument("--shw-a", required=True, type=Path, help="First input .shw file")
    parser.add_argument("--time-a-s", required=True, type=float, help="Normalization time for SHW A, in seconds")
    parser.add_argument("--label-a", default="SHW_A", help="Label for first sample")
    parser.add_argument("--shw-b", required=True, type=Path, help="Second input .shw file")
    parser.add_argument("--time-b-s", required=True, type=float, help="Normalization time for SHW B, in seconds")
    parser.add_argument("--label-b", default="SHW_B", help="Label for second sample")
    parser.add_argument("--outdir", default=Path("comparacion_shw_RB"), type=Path, help="Output directory")
    parser.add_argument("--area-m2", default=1.0, type=float, help="Effective area in m^2. Default: 1")
    parser.add_argument("--altitude-m", default=893.0, type=float, help="Observation altitude in m a.s.l.")
    parser.add_argument("--theta-bin-deg", default=5.0, type=float, help="Theta bin width in degrees")
    parser.add_argument("--theta-max-deg", default=90.0, type=float, help="Maximum theta included in histograms")
    parser.add_argument("--model-emin-GeV", default=1.0, type=float, help="Minimum total energy for overall Reyna/Bugaev integration")
    parser.add_argument("--model-emax-GeV", default=1.0e5, type=float, help="Maximum total energy for overall Reyna/Bugaev integration")
    parser.add_argument("--model-grid", default=1200, type=int, help="Energy grid points for model integration")
    parser.add_argument(
        "--energy-bins",
        default="0.5:1.5,1.5:4.5,4.5:15,15:50,50:200",
        help="Energy bands in total GeV, format 'low:high,low:high'",
    )
    parser.add_argument(
        "--overall-all-energies",
        action="store_true",
        help="Use all SHW muons for the overall histogram. Default: only model energy range.",
    )
    parser.add_argument(
        "--max-muons",
        default=None,
        type=int,
        help="Optional limit on number of muons read per file. Useful for fast tests.",
    )
    parser.add_argument(
        "--progress-lines",
        default=2_000_000,
        type=int,
        help="Print progress every N input lines. Use 0 to disable.",
    )

    args = parser.parse_args(argv)

    if args.time_a_s <= 0 or args.time_b_s <= 0:
        raise ValueError("Both normalization times must be positive")
    if args.area_m2 <= 0:
        raise ValueError("--area-m2 must be positive")
    if not args.shw_a.exists():
        raise FileNotFoundError(f"Input SHW A not found: {args.shw_a}")
    if not args.shw_b.exists():
        raise FileNotFoundError(f"Input SHW B not found: {args.shw_b}")

    set_article_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    energy_bins = parse_energy_bins(args.energy_bins)
    theta_edges = theta_edges_from_bin_width(args.theta_bin_deg, args.theta_max_deg)

    print(f"[1] Streaming {args.shw_a}", flush=True)
    res_a = read_shw_streaming(
        path=args.shw_a,
        label=args.label_a,
        time_s=args.time_a_s,
        theta_edges=theta_edges,
        energy_bins=energy_bins,
        model_emin=args.model_emin_GeV,
        model_emax=args.model_emax_GeV,
        overall_all_energies=args.overall_all_energies,
        max_muons=args.max_muons,
        progress_lines=args.progress_lines,
    )
    print(f"    {args.label_a}: {res_a.muons_found:,} muons found; {res_a.muons_used_overall:,} used in overall histogram", flush=True)

    print(f"[2] Streaming {args.shw_b}", flush=True)
    res_b = read_shw_streaming(
        path=args.shw_b,
        label=args.label_b,
        time_s=args.time_b_s,
        theta_edges=theta_edges,
        energy_bins=energy_bins,
        model_emin=args.model_emin_GeV,
        model_emax=args.model_emax_GeV,
        overall_all_energies=args.overall_all_energies,
        max_muons=args.max_muons,
        progress_lines=args.progress_lines,
    )
    print(f"    {args.label_b}: {res_b.muons_found:,} muons found; {res_b.muons_used_overall:,} used in overall histogram", flush=True)

    print("[3] Building tables", flush=True)
    summary_path = write_run_summary(res_a, res_b, args.outdir, args)

    df_overall = make_overall_table(
        res_a=res_a,
        res_b=res_b,
        theta_edges=theta_edges,
        area_m2=args.area_m2,
        altitude_m=args.altitude_m,
        model_emin=args.model_emin_GeV,
        model_emax=args.model_emax_GeV,
        model_grid=args.model_grid,
    )
    overall_csv = args.outdir / "overall_theta_flux_comparison.csv"
    df_overall.to_csv(overall_csv, index=False)

    df_bands = make_energy_band_table(
        res_a=res_a,
        res_b=res_b,
        theta_edges=theta_edges,
        energy_bins=energy_bins,
        area_m2=args.area_m2,
        altitude_m=args.altitude_m,
        model_grid=args.model_grid,
    )
    bands_csv = args.outdir / "energy_band_theta_flux_comparison.csv"
    df_bands.to_csv(bands_csv, index=False)

    df_rates = make_integrated_band_rates(df_bands, energy_bins, args.label_a, args.label_b)
    rates_csv = args.outdir / "energy_band_integrated_rates.csv"
    df_rates.to_csv(rates_csv, index=False)

    print("[4] Plotting", flush=True)
    outputs = [summary_path, overall_csv, bands_csv, rates_csv]
    outputs.append(plot_overall_flux(df_overall, args.label_a, args.label_b, args.outdir))
    outputs.append(plot_overall_shape(df_overall, args.label_a, args.label_b, args.outdir))
    outputs.append(plot_overall_ratio(df_overall, args.label_a, args.label_b, args.outdir))
    outputs.append(plot_energy_band_flux(df_bands, energy_bins, args.label_a, args.label_b, args.outdir))
    outputs.append(plot_energy_band_ratio(df_bands, energy_bins, args.label_a, args.label_b, args.outdir))
    outputs.append(plot_summary_reyna_bugaev_vs_two_shw(df_bands, energy_bins, args.label_a, args.label_b, args.outdir))
    outputs.append(plot_integrated_rates(df_rates, args.label_a, args.label_b, args.outdir))

    manifest = args.outdir / "outputs_manifest.csv"
    pd.DataFrame({"output": [str(p) for p in outputs]}).to_csv(manifest, index=False)

    print("[OK] Finished", flush=True)
    print(f"Output directory: {args.outdir}", flush=True)
    print(f"Manifest: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
