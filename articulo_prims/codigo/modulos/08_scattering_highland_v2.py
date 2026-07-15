#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_scattering_highland_v2.py
----------------------------
Analytic Highland scattering diagnostic for absorption muography.

Main differences vs the previous version:
- Keeps the CSV output per energy factor.
- Produces COMBINED 1x3 figures (one panel per energy factor) for each point.
- Applies a default theta cut at 90 deg to exclude below-horizon / upgoing angular cells.
- Uses square-display regridding only for visualization, so the bins look square and well proportioned.

Typical usage
-------------
python 08_scattering_highland_v2.py \
    --indir ./run_machin/03_ecrit \
    --outdir ./run_machin/07_scattering \
    --points P1 P2 P4 P5 \
    --energy-factors 1.0 1.5 2.0 \
    --theta-max 90 \
    --square-display \
    --display-step 0.5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MUON_MASS_GEV = 0.10565837
DEFAULT_POINTS = ("P1", "P2", "P4", "P5")
DEFAULT_X0_ROCK_G_CM2 = 26.54


def set_article_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.grid": False,
    })


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col in df.columns:
        low = col.lower()
        for cand in candidates:
            if cand.lower() in low:
                return col
    if required:
        raise KeyError(f"Could not find columns {list(candidates)} in {list(df.columns)}")
    return None


def infer_bin_width(values: np.ndarray, fallback: float | None = None) -> float:
    vals = np.asarray(values, dtype=float)
    vals = np.sort(np.unique(vals[np.isfinite(vals)]))
    if vals.size >= 2:
        diffs = np.diff(vals)
        diffs = diffs[diffs > 0]
        if diffs.size:
            return float(np.median(diffs))
    if fallback is not None:
        return float(fallback)
    raise ValueError("Could not infer bin width.")


def centers_to_edges(values: np.ndarray, fallback_width: float | None = None) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = np.sort(np.unique(vals[np.isfinite(vals)]))
    if vals.size == 0:
        raise ValueError("Empty coordinate array.")
    if vals.size == 1:
        if fallback_width is None:
            fallback_width = 1.0
        half = 0.5 * float(fallback_width)
        return np.array([vals[0] - half, vals[0] + half], dtype=float)
    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - (mids[0] - vals[0])
    last = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]]).astype(float)


def build_grid(df: pd.DataFrame, theta_col: str, phi_col: str, value_col: str):
    theta_centers = np.array(sorted(df[theta_col].dropna().unique()), dtype=float)
    phi_centers = np.array(sorted(df[phi_col].dropna().unique()), dtype=float)
    Z = np.full((len(theta_centers), len(phi_centers)), np.nan, dtype=float)
    t2i = {round(v, 10): i for i, v in enumerate(theta_centers)}
    p2j = {round(v, 10): j for j, v in enumerate(phi_centers)}
    for _, row in df.iterrows():
        t = row[theta_col]
        p = row[phi_col]
        if not np.isfinite(t) or not np.isfinite(p):
            continue
        i = t2i.get(round(float(t), 10))
        j = p2j.get(round(float(p), 10))
        if i is None or j is None:
            continue
        Z[i, j] = row[value_col]
    return phi_centers, theta_centers, Z


def momentum_beta_from_total_energy(E_total_GeV: np.ndarray, mass_GeV: float = MUON_MASS_GEV):
    E = np.asarray(E_total_GeV, dtype=float)
    p2 = E*E - mass_GeV*mass_GeV
    p = np.full_like(E, np.nan, dtype=float)
    beta = np.full_like(E, np.nan, dtype=float)
    valid = np.isfinite(E) & (p2 > 0.0)
    p[valid] = np.sqrt(p2[valid])
    beta[valid] = p[valid] / E[valid]
    return p, beta


def highland_theta0_rad(X_g_cm2, p_GeV, beta, X0_g_cm2, charge_number=1.0):
    X = np.asarray(X_g_cm2, dtype=float)
    p = np.asarray(p_GeV, dtype=float)
    b = np.asarray(beta, dtype=float)
    theta0 = np.full_like(X, np.nan, dtype=float)
    x_over_x0 = X / X0_g_cm2
    valid = (
        np.isfinite(x_over_x0) & (x_over_x0 > 0.0) &
        np.isfinite(p) & (p > 0.0) &
        np.isfinite(b) & (b > 0.0)
    )
    if not np.any(valid):
        return theta0
    correction = 1.0 + 0.038 * np.log(x_over_x0[valid])
    correction = np.where(correction > 0.0, correction, np.nan)
    theta0[valid] = (
        0.0136 / (b[valid] * p[valid]) *
        abs(charge_number) *
        np.sqrt(x_over_x0[valid]) *
        correction
    )
    return theta0


def load_ecrit_table(path: Path, rho_g_cm3: float):
    df = pd.read_csv(path)
    theta_col = find_column(df, ["theta_deg", "theta", "zenith_deg"])
    phi_col = find_column(df, ["phi_deg", "phi_rel_deg", "phi", "azimuth_deg"])
    ecrit_col = find_column(df, ["Ecrit_total_GeV", "E_total_crit_GeV", "Ecrit_GeV", "E_total_GeV"])
    x_col = find_column(df, ["X_g_cm2", "opacity_g_cm2", "opacity"], required=False)
    length_col = find_column(df, ["length_inside_m", "rock_length_m", "L_m", "length_m", "longitud_m", "length"], required=False)

    df = df.copy()
    df[theta_col] = pd.to_numeric(df[theta_col], errors="coerce")
    df[phi_col] = pd.to_numeric(df[phi_col], errors="coerce")
    df[ecrit_col] = pd.to_numeric(df[ecrit_col], errors="coerce")

    if x_col is not None:
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        X = df[x_col].to_numpy(dtype=float)
    elif length_col is not None:
        df[length_col] = pd.to_numeric(df[length_col], errors="coerce")
        X = rho_g_cm3 * df[length_col].to_numpy(dtype=float) * 100.0
        x_col = "X_g_cm2"
        df[x_col] = X
    else:
        raise KeyError(f"{path.name} has no X_g_cm2 and no rock length column.")

    df["X_g_cm2"] = pd.to_numeric(df[x_col], errors="coerce")
    df["Ecrit_total_GeV"] = pd.to_numeric(df[ecrit_col], errors="coerce")
    return df, theta_col, phi_col


def compute_scattering_for_factor(df, theta_col, phi_col, energy_factor, X0_g_cm2,
                                  charge_number, theta_bin_deg, phi_bin_deg):
    out = df.copy()
    Ecrit = out["Ecrit_total_GeV"].to_numpy(dtype=float)
    X = out["X_g_cm2"].to_numpy(dtype=float)

    E_ref = energy_factor * Ecrit
    p_ref, beta = momentum_beta_from_total_energy(E_ref)
    x_over_x0 = X / X0_g_cm2
    theta0_rad = highland_theta0_rad(X, p_ref, beta, X0_g_cm2, charge_number=charge_number)

    theta0_deg = np.rad2deg(theta0_rad)
    theta0_mrad = 1.0e3 * theta0_rad
    theta_rms_2d_deg = math.sqrt(2.0) * theta0_deg

    theta_vals = out[theta_col].to_numpy(dtype=float)
    phi_effective_deg = np.abs(np.sin(np.deg2rad(theta_vals))) * phi_bin_deg
    pixel_min_deg = np.minimum(theta_bin_deg, phi_effective_deg)

    with np.errstate(divide="ignore", invalid="ignore"):
        theta0_over_theta_bin = theta0_deg / theta_bin_deg
        theta0_over_phi_eff_bin = theta0_deg / phi_effective_deg
        theta0_over_pixel_min = theta0_deg / pixel_min_deg
        theta2d_over_pixel_min = theta_rms_2d_deg / pixel_min_deg

    length_col = find_column(out, ["length_inside_m", "rock_length_m", "L_m", "length_m", "longitud_m", "length"], required=False)
    if length_col is not None:
        L_m = pd.to_numeric(out[length_col], errors="coerce").to_numpy(dtype=float)
    else:
        L_m = np.full_like(theta0_rad, np.nan, dtype=float)

    out["energy_factor"] = energy_factor
    out["X0_g_cm2"] = X0_g_cm2
    out["x_over_X0"] = x_over_x0
    out["Eref_total_GeV"] = E_ref
    out["p_ref_GeV_c"] = p_ref
    out["beta_ref"] = beta
    out["theta_bin_deg"] = theta_bin_deg
    out["phi_bin_deg"] = phi_bin_deg
    out["phi_effective_bin_deg"] = phi_effective_deg
    out["pixel_min_bin_deg"] = pixel_min_deg
    out["theta0_proj_rad"] = theta0_rad
    out["theta0_proj_deg"] = theta0_deg
    out["theta0_proj_mrad"] = theta0_mrad
    out["theta_rms_2d_deg"] = theta_rms_2d_deg
    out["theta0_over_theta_bin"] = theta0_over_theta_bin
    out["theta0_over_phi_effective_bin"] = theta0_over_phi_eff_bin
    out["theta0_over_pixel_min_bin"] = theta0_over_pixel_min
    out["theta2d_over_pixel_min_bin"] = theta2d_over_pixel_min
    out["lateral_rms_proj_m"] = L_m * theta0_rad / math.sqrt(3.0)
    out["lateral_simple_Ltheta_m"] = L_m * theta0_rad
    return out


def robust_vmax(values: np.ndarray, percentile: float = 98.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vmax = float(np.nanpercentile(vals, percentile))
    if not np.isfinite(vmax):
        return None
    return vmax


def factor_tag(factor: float) -> str:
    return f"f{factor:.2f}".replace(".", "p").replace("-", "m")


def summarize(df: pd.DataFrame, point: str, factor: float):
    def q(col: str, p: float):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.nanpercentile(vals, p)) if vals.size else float("nan")

    vals_ratio = pd.to_numeric(df["theta0_over_pixel_min_bin"], errors="coerce").to_numpy(dtype=float)
    vals_ratio = vals_ratio[np.isfinite(vals_ratio)]
    if vals_ratio.size:
        frac_gt_01 = float(np.mean(vals_ratio > 0.1))
        frac_gt_03 = float(np.mean(vals_ratio > 0.3))
        frac_gt_10 = float(np.mean(vals_ratio > 1.0))
    else:
        frac_gt_01 = frac_gt_03 = frac_gt_10 = float("nan")

    return {
        "point": point,
        "energy_factor": factor,
        "n_cells": int(df.shape[0]),
        "theta0_mrad_median": q("theta0_proj_mrad", 50),
        "theta0_mrad_p90": q("theta0_proj_mrad", 90),
        "theta0_deg_median": q("theta0_proj_deg", 50),
        "theta0_deg_p90": q("theta0_proj_deg", 90),
        "ratio_pixel_min_median": q("theta0_over_pixel_min_bin", 50),
        "ratio_pixel_min_p90": q("theta0_over_pixel_min_bin", 90),
        "ratio_pixel_min_max": q("theta0_over_pixel_min_bin", 100),
        "frac_cells_ratio_gt_0p1": frac_gt_01,
        "frac_cells_ratio_gt_0p3": frac_gt_03,
        "frac_cells_ratio_gt_1p0": frac_gt_10,
        "lateral_rms_proj_m_median": q("lateral_rms_proj_m", 50),
        "lateral_rms_proj_m_p90": q("lateral_rms_proj_m", 90),
        "x_over_X0_median": q("x_over_X0", 50),
        "Eref_total_GeV_median": q("Eref_total_GeV", 50),
    }


def square_display_from_df(df, theta_col, phi_col, value_col,
                           theta_min, theta_max, phi_min, phi_max,
                           display_step, theta_bin_deg, phi_bin_deg):
    phis, thetas, Z_src = build_grid(df, theta_col, phi_col, value_col)
    ph_edges_src = centers_to_edges(phis, fallback_width=phi_bin_deg)
    th_edges_src = centers_to_edges(thetas, fallback_width=theta_bin_deg)

    theta_edges_disp = np.arange(theta_min, theta_max + display_step, display_step)
    phi_edges_disp = np.arange(phi_min, phi_max + display_step, display_step)
    theta_centers_disp = 0.5 * (theta_edges_disp[:-1] + theta_edges_disp[1:])
    phi_centers_disp = 0.5 * (phi_edges_disp[:-1] + phi_edges_disp[1:])

    Z_disp = np.full((len(theta_centers_disp), len(phi_centers_disp)), np.nan, dtype=float)
    for i, th in enumerate(theta_centers_disp):
        i_src = np.searchsorted(th_edges_src, th, side="right") - 1
        if i_src < 0 or i_src >= Z_src.shape[0]:
            continue
        for j, ph in enumerate(phi_centers_disp):
            j_src = np.searchsorted(ph_edges_src, ph, side="right") - 1
            if j_src < 0 or j_src >= Z_src.shape[1]:
                continue
            Z_disp[i, j] = Z_src[i_src, j_src]

    return phi_edges_disp, theta_edges_disp, Z_disp


def native_display_from_df(df, theta_col, phi_col, value_col, theta_bin_deg, phi_bin_deg):
    phis, thetas, Z = build_grid(df, theta_col, phi_col, value_col)
    phi_edges = centers_to_edges(phis, fallback_width=phi_bin_deg)
    theta_edges = centers_to_edges(thetas, fallback_width=theta_bin_deg)
    return phi_edges, theta_edges, Z


def make_triptych(scattering_by_factor: list[pd.DataFrame], factors: list[float],
                  theta_col: str, phi_col: str, value_col: str,
                  out_png: Path, cbar_label: str, title_prefix: str,
                  theta_bin_deg: float, phi_bin_deg: float,
                  theta_min: float, theta_max: float,
                  phi_min: float, phi_max: float,
                  square_display: bool, display_step: float,
                  fixed_vmax: float | None = None):

    canvases = []
    all_vals = []
    for df in scattering_by_factor:
        if square_display:
            phi_edges, theta_edges, Z = square_display_from_df(
                df, theta_col, phi_col, value_col,
                theta_min, theta_max, phi_min, phi_max,
                display_step, theta_bin_deg, phi_bin_deg
            )
        else:
            phi_edges, theta_edges, Z = native_display_from_df(df, theta_col, phi_col, value_col,
                                                               theta_bin_deg, phi_bin_deg)
        mask_theta = (0.5 * (theta_edges[:-1] + theta_edges[1:]) >= theta_min) & \
                     (0.5 * (theta_edges[:-1] + theta_edges[1:]) <= theta_max)
        if not np.all(mask_theta):
            Z = Z[mask_theta, :]
            theta_edges = np.concatenate([theta_edges[:-1][mask_theta], [theta_edges[1:][mask_theta][-1]]])
        canvases.append((phi_edges, theta_edges, Z))
        vals = Z[np.isfinite(Z)]
        if vals.size:
            all_vals.append(vals)

    vmax = fixed_vmax
    if vmax is None and all_vals:
        vmax = robust_vmax(np.concatenate(all_vals), percentile=98.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)
    mappable = None
    for ax, (factor, (phi_edges, theta_edges, Z)) in zip(axes, zip(factors, canvases)):
        kwargs = {"shading": "flat", "cmap": "viridis"}
        if vmax is not None:
            kwargs["vmax"] = vmax
        mappable = ax.pcolormesh(phi_edges, theta_edges, Z, **kwargs)
        ax.set_xlim(phi_min, phi_max)
        ax.set_ylim(theta_max, theta_min)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(rf"$E_{{ref}} = {factor:g}\,E_{{crit}}$")
        ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
        ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
        ax.set_xticks(np.arange(np.ceil(phi_min/20)*20, phi_max + 1e-6, 20))
        ax.set_yticks(np.arange(np.ceil(theta_min/10)*10, theta_max + 1e-6, 10))

    cbar = fig.colorbar(mappable, ax=axes, shrink=0.96, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(title_prefix, fontsize=12)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def process_point(point, indir, outdir, energy_factors, rho_g_cm3, X0_g_cm2,
                  charge_number, theta_bin_arg, phi_bin_arg,
                  theta_min_arg, theta_max_arg, phi_min_arg, phi_max_arg,
                  square_display, display_step):
    in_csv = indir / f"ecrit_table_{point}.csv"
    if not in_csv.exists():
        print(f"[WARN] Missing {in_csv}. Skipping {point}.")
        return []

    df, theta_col, phi_col = load_ecrit_table(in_csv, rho_g_cm3=rho_g_cm3)
    df = df.dropna(subset=[theta_col, phi_col, "X_g_cm2", "Ecrit_total_GeV"]).copy()
    if df.empty:
        print(f"[WARN] No valid rows in {in_csv}. Skipping {point}.")
        return []

    # Hard cut to avoid theta > 90 deg by default.
    if theta_min_arg is None:
        theta_min = float(np.nanmin(df[theta_col].to_numpy(dtype=float)))
    else:
        theta_min = theta_min_arg
    theta_max = min(theta_max_arg, 90.0)
    df = df[(df[theta_col] >= theta_min) & (df[theta_col] <= theta_max)].copy()

    if df.empty:
        print(f"[WARN] No cells remain after theta cut for {point}.")
        return []

    theta_bin = infer_bin_width(df[theta_col].to_numpy(dtype=float), fallback=theta_bin_arg)
    phi_bin = infer_bin_width(df[phi_col].to_numpy(dtype=float), fallback=phi_bin_arg)

    if phi_min_arg is None:
        phi_min = float(np.nanmin(df[phi_col].to_numpy(dtype=float)))
    else:
        phi_min = phi_min_arg
    if phi_max_arg is None:
        phi_max = float(np.nanmax(df[phi_col].to_numpy(dtype=float)))
    else:
        phi_max = phi_max_arg

    if display_step is None:
        display_step = min(theta_bin, phi_bin)

    point_dir = outdir / point
    point_dir.mkdir(parents=True, exist_ok=True)

    scattering_by_factor = []
    summaries = []
    for factor in energy_factors:
        tag = factor_tag(factor)
        scat = compute_scattering_for_factor(
            df=df,
            theta_col=theta_col,
            phi_col=phi_col,
            energy_factor=factor,
            X0_g_cm2=X0_g_cm2,
            charge_number=charge_number,
            theta_bin_deg=theta_bin,
            phi_bin_deg=phi_bin,
        )
        out_csv = point_dir / f"scattering_table_{point}_{tag}.csv"
        scat.to_csv(out_csv, index=False)
        scattering_by_factor.append(scat)

        summary = summarize(scat, point=point, factor=factor)
        summary.update({
            "input_csv": str(in_csv),
            "output_csv": str(out_csv),
            "theta_bin_deg_inferred": theta_bin,
            "phi_bin_deg_inferred": phi_bin,
            "theta_min_used": theta_min,
            "theta_max_used": theta_max,
            "phi_min_used": phi_min,
            "phi_max_used": phi_max,
            "display_step_deg": display_step,
        })
        summaries.append(summary)

    make_triptych(
        scattering_by_factor, energy_factors, theta_col, phi_col,
        value_col="theta0_proj_mrad",
        out_png=point_dir / f"theta0_mrad_triptych_{point}.png",
        cbar_label=r"$\theta_0$ (mrad)",
        title_prefix=f"Projected RMS multiple scattering — {point}",
        theta_bin_deg=theta_bin, phi_bin_deg=phi_bin,
        theta_min=theta_min, theta_max=theta_max,
        phi_min=phi_min, phi_max=phi_max,
        square_display=square_display, display_step=display_step,
    )

    make_triptych(
        scattering_by_factor, energy_factors, theta_col, phi_col,
        value_col="theta0_proj_deg",
        out_png=point_dir / f"theta0_deg_triptych_{point}.png",
        cbar_label=r"$\theta_0$ (deg)",
        title_prefix=f"Projected RMS multiple scattering — {point}",
        theta_bin_deg=theta_bin, phi_bin_deg=phi_bin,
        theta_min=theta_min, theta_max=theta_max,
        phi_min=phi_min, phi_max=phi_max,
        square_display=square_display, display_step=display_step,
    )

    make_triptych(
        scattering_by_factor, energy_factors, theta_col, phi_col,
        value_col="theta0_over_pixel_min_bin",
        out_png=point_dir / f"theta0_over_pixel_min_triptych_{point}.png",
        cbar_label=r"$\theta_0 / \Delta\alpha_{\min}$",
        title_prefix=f"Scattering relative to angular pixel — {point}",
        theta_bin_deg=theta_bin, phi_bin_deg=phi_bin,
        theta_min=theta_min, theta_max=theta_max,
        phi_min=phi_min, phi_max=phi_max,
        square_display=square_display, display_step=display_step,
        fixed_vmax=1.0,
    )

    make_triptych(
        scattering_by_factor, energy_factors, theta_col, phi_col,
        value_col="lateral_rms_proj_m",
        out_png=point_dir / f"lateral_rms_proj_m_triptych_{point}.png",
        cbar_label=r"$y_{\mathrm{rms}}$ (m)",
        title_prefix=f"Approx. projected lateral RMS — {point}",
        theta_bin_deg=theta_bin, phi_bin_deg=phi_bin,
        theta_min=theta_min, theta_max=theta_max,
        phi_min=phi_min, phi_max=phi_max,
        square_display=square_display, display_step=display_step,
    )

    print(f"[OK] {point}: combined 1x3 figures created with theta <= {theta_max} deg")
    return summaries


def build_parser():
    ap = argparse.ArgumentParser(description="Analytic Highland scattering diagnostic with 1x3 comparison figures.")
    ap.add_argument("--indir", default="outputs", type=Path, help="Directory with ecrit_table_{POINT}.csv files.")
    ap.add_argument("--outdir", default="outputs_scattering", type=Path, help="Output directory.")
    ap.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS), help="Points to process.")
    ap.add_argument("--energy-factors", nargs="+", type=float, default=[1.0, 1.5, 2.0],
                    help="Reference total energy factors: Eref_total = factor * Ecrit_total.")
    ap.add_argument("--rho", type=float, default=2.65,
                    help="Rock density in g/cm^3, used only if X_g_cm2 is absent.")
    ap.add_argument("--X0", type=float, default=DEFAULT_X0_ROCK_G_CM2,
                    help="Radiation length in g/cm^2. Default: 26.54.")
    ap.add_argument("--charge", type=float, default=1.0, help="Particle charge number |z|. Default: 1.")
    ap.add_argument("--theta-bin-deg", type=float, default=None, help="Fallback theta bin width.")
    ap.add_argument("--phi-bin-deg", type=float, default=None, help="Fallback phi bin width.")
    ap.add_argument("--theta-min", type=float, default=None, help="Minimum theta to show/use. Default: inferred from data.")
    ap.add_argument("--theta-max", type=float, default=90.0, help="Maximum theta to show/use. Default: 90 deg.")
    ap.add_argument("--phi-min", type=float, default=None, help="Minimum phi to show. Default: inferred from data.")
    ap.add_argument("--phi-max", type=float, default=None, help="Maximum phi to show. Default: inferred from data.")
    ap.add_argument("--square-display", action="store_true", help="Regrid only for display to square angular bins.")
    ap.add_argument("--display-step", type=float, default=None,
                    help="Square display step in deg. Default: min(theta_bin, phi_bin).")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_article_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for point in args.points:
        summaries = process_point(
            point=point,
            indir=args.indir,
            outdir=args.outdir,
            energy_factors=args.energy_factors,
            rho_g_cm3=args.rho,
            X0_g_cm2=args.X0,
            charge_number=args.charge,
            theta_bin_arg=args.theta_bin_deg,
            phi_bin_arg=args.phi_bin_deg,
            theta_min_arg=args.theta_min,
            theta_max_arg=args.theta_max,
            phi_min_arg=args.phi_min,
            phi_max_arg=args.phi_max,
            square_display=args.square_display,
            display_step=args.display_step,
        )
        all_summaries.extend(summaries)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_csv = args.outdir / "scattering_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[DONE] Summary: {summary_csv}")
    else:
        print("[WARN] No outputs created. Check --indir and point names.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
