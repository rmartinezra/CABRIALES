#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_inside_volcano_maps_merged.py

Une dos tareas que antes estaban separadas:

1) Cuenta muones dentro de la geometría angular del volcán.
2) Genera figuras finales con el estilo de plot_4panel_muon_maps.py:
   - canvas angular fijo,
   - bins visualmente cuadrados,
   - ceros en blanco,
   - escala lineal con percentil superior,
   - escala logarítmica,
   - figuras individuales,
   - figura 2x2 para P1/P2/P4/P5.

Por defecto procesa RAW y FILTERED.

RAW:
  usa un solo .shw para todos los puntos y lo lee una sola vez.

FILTERED:
  usa un .shw filtrado distinto por punto:
      {stem}_filtered_P1.shw
      {stem}_filtered_P2.shw
      ...
  Esto evita el error físico de usar el filtrado de P1 para todos los puntos.

Ejemplo típico:

python3 07_inside_volcano_maps_merged.py \
  --raw-shw data/bga_CNF_604800s.shw \
  --filtered-dir run_machin_7dias/04_filtered \
  --geom-dir run_machin_7dias/02_lengths \
  --outdir run_machin_7dias/06_inside_volcano \
  --theta-min 0 \
  --theta-max 90 \
  --display-theta-min 0 \
  --display-theta-max 90 \
  --display-phi-min -60 \
  --display-phi-max 60 \
  --display-step 0.5

Salidas:

run_machin_7dias/06_inside_volcano/
├── raw/P1/counts_inside_volcano_P1.csv
├── raw/P1/counts_inside_volcano_P1.png
├── ...
├── filtered/P1/counts_inside_volcano_P1.csv
├── ...
└── figures/
    ├── raw/inside_volcano_raw_P1_linear.png
    ├── raw/inside_volcano_raw_P1_log.png
    ├── filtered/inside_volcano_filtered_P1_linear.png
    ├── filtered/inside_volcano_filtered_P1_log.png
    ├── inside_volcano_raw_4panel_linear.png
    ├── inside_volcano_raw_4panel_log.png
    ├── inside_volcano_filtered_4panel_linear.png
    └── inside_volcano_filtered_4panel_log.png
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from shw_io import open_shw_bytes, parse_muon_parts, shw_stem, stream_size_hint, theta_phi_from_momentum
from plot_style import apply_scientific_style


# ---------------------------------------------------------------------
# Site constants
# ---------------------------------------------------------------------
SUMMIT = (4.486552, -75.388975)
POINTS = {
    "P1": (4.492298, -75.381092),
    "P2": (4.494946, -75.388110),
    "P4": (4.476500, -75.386500),
    "P5": (4.488500, -75.379500),
}
ORDER = ["P1", "P2", "P4", "P5"]
MUON_IDS_B = {b"0005", b"0006", b"5", b"6"}


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------
def setup_style() -> None:
    apply_scientific_style()


# ---------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------
def azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def solid_angle_per_bin(theta_edges_deg: np.ndarray, phi_edges_deg: np.ndarray) -> np.ndarray:
    th = np.deg2rad(theta_edges_deg)
    ph = np.deg2rad(phi_edges_deg)
    return (np.cos(th[:-1]) - np.cos(th[1:]))[:, None] * np.diff(ph)[None, :]


def infer_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(sorted(np.unique(np.round(np.asarray(centers, dtype=float), 10))), dtype=float)
    if centers.size < 2:
        raise ValueError("Need at least two centers to infer bin edges.")
    mids = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([
        [centers[0] - (mids[0] - centers[0])],
        mids,
        [centers[-1] + (centers[-1] - mids[-1])],
    ])


def nearest_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True):
    lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    for col in df.columns:
        low = col.lower()
        if any(cand.lower() in low for cand in candidates):
            return col

    if required:
        raise KeyError(f"No column found. Tried {list(candidates)}. Available: {list(df.columns)}")
    return None


def guess_mask_column(df: pd.DataFrame, theta_col: str, phi_col: str, explicit: str | None):
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"Mask column {explicit!r} not found. Available: {list(df.columns)}")
        return explicit

    groups = [
        ["rock_length_m", "L_rock_m", "length_rock_m", "rock_m", "L_m", "length_m", "rock_length", "length"],
        ["Ecrit_total_GeV", "Ecrit_GeV", "Tcrit_GeV", "E_min_GeV", "Emin_GeV"],
        ["blocked", "inside", "mask"],
    ]

    excluded = {theta_col, phi_col}
    for group in groups:
        for col in df.columns:
            if col in excluded:
                continue
            low = col.lower()
            if any(k.lower() in low for k in group):
                return col

    for col in df.columns:
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col]):
            return col

    raise KeyError(f"Could not infer mask column. Available: {list(df.columns)}")


def load_geometry(
    geom_csv: Path,
    mask_col: str | None,
    mask_min: float,
    theta_min: float,
    theta_max: float,
):
    df = pd.read_csv(geom_csv)

    theta_col = nearest_column(df, ["theta_deg", "theta_center_deg", "theta", "Theta", "zenith_deg", "theta_z_deg"])
    phi_col = nearest_column(df, ["phi_rel_deg", "phi_deg", "phi", "azimuth_deg", "az_deg"])
    mcol = guess_mask_column(df, theta_col, phi_col, mask_col)

    df = df.copy()
    df[theta_col] = pd.to_numeric(df[theta_col], errors="coerce")
    df[phi_col] = pd.to_numeric(df[phi_col], errors="coerce")
    df[mcol] = pd.to_numeric(df[mcol], errors="coerce")

    df = df.dropna(subset=[theta_col, phi_col])
    df = df[(df[theta_col] >= theta_min) & (df[theta_col] <= theta_max)]

    if df.empty:
        raise RuntimeError(f"No valid angular rows after theta cut in {geom_csv}")

    theta_centers = np.array(sorted(df[theta_col].dropna().unique()), dtype=float)
    phi_centers = np.array(sorted(df[phi_col].dropna().unique()), dtype=float)
    theta_edges = infer_edges_from_centers(theta_centers)
    phi_edges = infer_edges_from_centers(phi_centers)

    mask = np.zeros((len(theta_centers), len(phi_centers)), dtype=bool)
    ti = {round(v, 10): i for i, v in enumerate(theta_centers)}
    pj = {round(v, 10): j for j, v in enumerate(phi_centers)}

    for row in df.itertuples(index=False):
        th = float(getattr(row, theta_col))
        ph = float(getattr(row, phi_col))
        val = getattr(row, mcol)
        val = float(val) if pd.notna(val) else np.nan

        i = ti.get(round(th, 10))
        j = pj.get(round(ph, 10))
        if i is not None and j is not None and np.isfinite(val) and val > mask_min:
            mask[i, j] = True

    return {
        "theta_edges": theta_edges,
        "phi_edges": phi_edges,
        "mask": mask,
        "mask_col": mcol,
        "geom_csv": Path(geom_csv),
        "H_inside": np.zeros_like(mask, dtype=np.int64),
        "H_all": np.zeros_like(mask, dtype=np.int64),
        "n_in_grid": 0,
        "n_inside": 0,
        "n_outside": 0,
    }


def load_geometries(
    points: list[str],
    geom_dir: Path | None,
    geom_template: str | None,
    mask_col: str | None,
    mask_min: float,
    theta_min: float,
    theta_max: float,
):
    geoms = {}
    for point in points:
        geom_csv = Path(geom_template.format(point=point)) if geom_template else Path(geom_dir) / f"rock_length_{point}.csv"
        if not geom_csv.exists():
            raise FileNotFoundError(f"No encontré geometría para {point}: {geom_csv}")

        g = load_geometry(
            geom_csv=geom_csv,
            mask_col=mask_col,
            mask_min=mask_min,
            theta_min=theta_min,
            theta_max=theta_max,
        )

        plat, plon = POINTS[point]
        az_geo = azimuth_deg(plat, plon, SUMMIT[0], SUMMIT[1])
        g["phi0"] = (90.0 - az_geo) % 360.0
        geoms[point] = g

        print(
            f"  {point}: {geom_csv} | bins={g['mask'].shape} | "
            f"inside cells={int(g['mask'].sum())}"
        )

    return geoms


# ---------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------
def update_event(geoms: dict, theta_deg: float, phi_abs: float):
    for _, g in geoms.items():
        phi_rel = (phi_abs - g["phi0"]) % 360.0
        if phi_rel > 180.0:
            phi_rel -= 360.0

        i = np.searchsorted(g["theta_edges"], theta_deg, side="right") - 1
        if i < 0 or i >= g["mask"].shape[0]:
            continue

        j = np.searchsorted(g["phi_edges"], phi_rel, side="right") - 1
        if j < 0 or j >= g["mask"].shape[1]:
            continue

        g["H_all"][i, j] += 1
        g["n_in_grid"] += 1

        if g["mask"][i, j]:
            g["H_inside"][i, j] += 1
            g["n_inside"] += 1
        else:
            g["n_outside"] += 1


def count_file(
    shw: Path,
    geoms: dict,
    only_muons: bool = True,
    shw_format: str = "auto",
    shw_member: str | None = None,
    progress_update_mb: float = 32.0,
    head: int = 0,
):
    total = stream_size_hint(shw)
    update_every = max(1, int(progress_update_mb * 1024 * 1024))
    pending = 0

    n_lines = 0
    n_particles = 0
    n_muons = 0
    next_report = update_every

    print(f"  reading: {shw}")

    with open_shw_bytes(shw, member_name=shw_member) as f:
        for raw in f:
            n_lines += 1
            pending += len(raw)

            if pending >= next_report:
                frac = 100.0 * pending / total if total else 100.0
                print(f"    progress: {frac:5.1f}%")
                next_report += update_every

            s = raw.strip()
            if not s or s.startswith(b"#"):
                continue

            parts = s.split()
            rec = parse_muon_parts(parts, shw_format=shw_format, only_muons=only_muons)
            if rec is None:
                continue

            n_particles += 1
            if rec.pid in MUON_IDS_B:
                n_muons += 1
            angles = theta_phi_from_momentum(rec.px, rec.py, rec.pz)
            if angles is None:
                continue
            theta, phi = angles

            update_event(geoms, theta, phi)

            if head > 0 and all(g["n_inside"] >= head for g in geoms.values()):
                break

    return {
        "n_lines_read": n_lines,
        "n_particles_read": n_particles,
        "n_muons_read": n_muons,
    }


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------
def point_grid_for_plot(g: dict):
    H_plot = g["H_inside"].astype(float)
    H_plot[~g["mask"]] = np.nan

    return {
        "theta_edges": g["theta_edges"],
        "phi_edges": g["phi_edges"],
        "H_plot": H_plot,
    }


def save_point_tables(outdir: Path, point: str, g: dict, global_summary: dict, mask_min: float):
    outdir = Path(outdir) / point
    outdir.mkdir(parents=True, exist_ok=True)

    th_edges = g["theta_edges"]
    ph_edges = g["phi_edges"]
    mask = g["mask"]
    H_inside = g["H_inside"]
    H_all = g["H_all"]

    th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])
    ph_centers = 0.5 * (ph_edges[:-1] + ph_edges[1:])
    TH, PH = np.meshgrid(th_centers, ph_centers, indexing="ij")
    domega = solid_angle_per_bin(th_edges, ph_edges)

    with np.errstate(divide="ignore", invalid="ignore"):
        I = H_inside.astype(float) / domega
        I[~mask] = np.nan
        I[~np.isfinite(I)] = np.nan

    counts_csv = outdir / f"counts_inside_volcano_{point}.csv"
    dno_csv = outdir / f"dNdOmega_inside_volcano_{point}.csv"
    summary_csv = outdir / f"inside_volcano_summary_{point}.csv"

    pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "inside_volcano_geometry": mask.ravel().astype(int),
        "count_inside_geometry": H_inside.ravel(),
        "count_all_in_grid": H_all.ravel(),
    }).to_csv(counts_csv, index=False)

    pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "inside_volcano_geometry": mask.ravel().astype(int),
        "count_inside_geometry": H_inside.ravel(),
        "delta_omega_sr": domega.ravel(),
        "dN_dOmega_inside_count_per_sr": I.ravel(),
    }).to_csv(dno_csv, index=False)

    summary = {
        "point": point,
        **global_summary,
        "n_events_in_angular_grid": g["n_in_grid"],
        "n_events_inside_volcano_geometry": g["n_inside"],
        "n_events_in_grid_but_outside_geometry": g["n_outside"],
        "fraction_inside_given_in_grid": g["n_inside"] / g["n_in_grid"] if g["n_in_grid"] else np.nan,
        "mask_column": g["mask_col"],
        "mask_min": mask_min,
        "geometry_csv": str(g["geom_csv"]),
    }
    pd.DataFrame({"quantity": list(summary.keys()), "value": list(summary.values())}).to_csv(summary_csv, index=False)

    return [counts_csv, dno_csv, summary_csv]


# ---------------------------------------------------------------------
# Plotting with square-display configuration
# ---------------------------------------------------------------------
def build_square_display_grid(
    plot_data: dict,
    display_step: float,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
):
    theta_edges_display = np.arange(theta_min, theta_max + display_step, display_step)
    phi_edges_display = np.arange(phi_min, phi_max + display_step, display_step)
    theta_centers_display = 0.5 * (theta_edges_display[:-1] + theta_edges_display[1:])
    phi_centers_display = 0.5 * (phi_edges_display[:-1] + phi_edges_display[1:])

    out = {}

    for point, d in plot_data.items():
        H_src = d["H_plot"]
        th_edges_src = d["theta_edges"]
        ph_edges_src = d["phi_edges"]

        H_disp = np.full((len(theta_centers_display), len(phi_centers_display)), np.nan, dtype=float)

        for i, th in enumerate(theta_centers_display):
            i_src = np.searchsorted(th_edges_src, th, side="right") - 1
            if i_src < 0 or i_src >= H_src.shape[0]:
                continue

            for j, ph in enumerate(phi_centers_display):
                j_src = np.searchsorted(ph_edges_src, ph, side="right") - 1
                if j_src < 0 or j_src >= H_src.shape[1]:
                    continue
                H_disp[i, j] = H_src[i_src, j_src]

        out[point] = {
            "theta_edges": theta_edges_display,
            "phi_edges": phi_edges_display,
            "H_plot": H_disp,
        }

    return out


def clean_for_plot(H: np.ndarray, blank_zeros: bool = True):
    Z = np.asarray(H, dtype=float).copy()
    Z[~np.isfinite(Z)] = np.nan
    if blank_zeros:
        Z[Z <= 0] = np.nan
    return Z


def positive_values(plot_data: dict):
    blocks = []
    for d in plot_data.values():
        vals = d["H_plot"][np.isfinite(d["H_plot"]) & (d["H_plot"] > 0)]
        if vals.size:
            blocks.append(vals)
    if not blocks:
        return np.array([], dtype=float)
    return np.concatenate(blocks)


def format_axes(ax, theta_min, theta_max, phi_min, phi_max):
    ax.set_xlim(phi_min, phi_max)
    ax.set_ylim(theta_max, theta_min)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
    ax.set_xticks(np.arange(np.ceil(phi_min / 20) * 20, phi_max + 1, 20))
    ax.set_yticks(np.arange(np.ceil(theta_min / 10) * 10, theta_max + 1, 10))


def make_individual_figures(
    plot_data: dict,
    outdir: Path,
    prefix: str,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    vmax_percentile: float,
    blank_zeros: bool,
):
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for point in ORDER:
        if point not in plot_data:
            continue

        d = plot_data[point]
        H = clean_for_plot(d["H_plot"], blank_zeros=blank_zeros)
        pos = H[np.isfinite(H) & (H > 0)]

        if pos.size:
            vmax_linear = np.nanpercentile(pos, vmax_percentile)
            vmax_log = np.nanmax(pos)
            vmin_log = max(1.0, np.nanmin(pos))
        else:
            vmax_linear = 1.0
            vmax_log = 1.0
            vmin_log = 1.0

        # Linear
        fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
        im = ax.pcolormesh(
            d["phi_edges"],
            d["theta_edges"],
            H,
            shading="flat",
            vmin=0,
            vmax=vmax_linear,
        )
        format_axes(ax, theta_min, theta_max, phi_min, phi_max)
        ax.set_title(f"{point}")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Counts per angular bin")
        out_png = outdir / f"{prefix}_{point}_linear.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_png)

        # Log
        fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
        if pos.size:
            im = ax.pcolormesh(
                d["phi_edges"],
                d["theta_edges"],
                H,
                shading="flat",
                norm=LogNorm(vmin=vmin_log, vmax=vmax_log),
            )
        else:
            im = ax.pcolormesh(d["phi_edges"], d["theta_edges"], H, shading="flat")
        format_axes(ax, theta_min, theta_max, phi_min, phi_max)
        ax.set_title(f"{point} (log scale)")
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("Counts per angular bin")
        out_png = outdir / f"{prefix}_{point}_log.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_png)

    return outputs


def make_4panel_figures(
    plot_data: dict,
    outdir: Path,
    prefix: str,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    vmax_percentile: float,
    blank_zeros: bool,
):
    outdir.mkdir(parents=True, exist_ok=True)
    if any(point not in plot_data for point in ORDER):
        return []

    labels = ["(a) P1", "(b) P2", "(c) P4", "(d) P5"]
    pos = positive_values(plot_data)

    if pos.size:
        vmax_linear = np.nanpercentile(pos, vmax_percentile)
        vmax_log = np.nanmax(pos)
        vmin_log = max(1.0, np.nanmin(pos))
    else:
        vmax_linear = 1.0
        vmax_log = 1.0
        vmin_log = 1.0

    outputs = []

    # Linear 2x2
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), constrained_layout=True)
    axes = axes.ravel()
    mappable = None

    for ax, point, label in zip(axes, ORDER, labels):
        d = plot_data[point]
        H = clean_for_plot(d["H_plot"], blank_zeros=blank_zeros)
        mappable = ax.pcolormesh(
            d["phi_edges"],
            d["theta_edges"],
            H,
            shading="flat",
            vmin=0,
            vmax=vmax_linear,
        )
        format_axes(ax, theta_min, theta_max, phi_min, phi_max)
        ax.set_title(label)

    cbar = fig.colorbar(mappable, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label("Counts per angular bin")
    out_png = outdir / f"{prefix}_4panel_linear.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out_png)

    # Log 2x2
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.6), constrained_layout=True)
    axes = axes.ravel()
    mappable = None

    for ax, point, label in zip(axes, ORDER, labels):
        d = plot_data[point]
        H = clean_for_plot(d["H_plot"], blank_zeros=blank_zeros)
        if pos.size:
            mappable = ax.pcolormesh(
                d["phi_edges"],
                d["theta_edges"],
                H,
                shading="flat",
                norm=LogNorm(vmin=vmin_log, vmax=vmax_log),
            )
        else:
            mappable = ax.pcolormesh(d["phi_edges"], d["theta_edges"], H, shading="flat")
        format_axes(ax, theta_min, theta_max, phi_min, phi_max)
        ax.set_title(label)

    cbar = fig.colorbar(mappable, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label("Counts per angular bin")
    out_png = outdir / f"{prefix}_4panel_log.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    outputs.append(out_png)

    # Summary
    lines = ["point,n_positive,min_positive,median_positive,p99_positive,max_positive"]
    for point in ORDER:
        d = plot_data[point]
        H = clean_for_plot(d["H_plot"], blank_zeros=blank_zeros)
        p = H[np.isfinite(H) & (H > 0)]
        if p.size:
            lines.append(
                f"{point},{p.size},{np.nanmin(p):.0f},{np.nanmedian(p):.6g},"
                f"{np.nanpercentile(p,99):.6g},{np.nanmax(p):.0f}"
            )
        else:
            lines.append(f"{point},0,nan,nan,nan,nan")

    summary_csv = outdir / f"{prefix}_4panel_summary.csv"
    summary_csv.write_text("\n".join(lines), encoding="utf-8")
    outputs.append(summary_csv)

    return outputs


def save_plots_for_source(
    source_name: str,
    plot_data: dict,
    figures_dir: Path,
    display_step: float,
    display_theta_min: float,
    display_theta_max: float,
    display_phi_min: float,
    display_phi_max: float,
    vmax_percentile: float,
    blank_zeros: bool,
):
    prefix = f"inside_volcano_{source_name}"

    display_data = build_square_display_grid(
        plot_data,
        display_step=display_step,
        theta_min=display_theta_min,
        theta_max=display_theta_max,
        phi_min=display_phi_min,
        phi_max=display_phi_max,
    )

    outputs = []

    outputs.extend(make_individual_figures(
        display_data,
        outdir=figures_dir / source_name,
        prefix=prefix,
        theta_min=display_theta_min,
        theta_max=display_theta_max,
        phi_min=display_phi_min,
        phi_max=display_phi_max,
        vmax_percentile=vmax_percentile,
        blank_zeros=blank_zeros,
    ))

    outputs.extend(make_4panel_figures(
        display_data,
        outdir=figures_dir,
        prefix=prefix,
        theta_min=display_theta_min,
        theta_max=display_theta_max,
        phi_min=display_phi_min,
        phi_max=display_phi_max,
        vmax_percentile=vmax_percentile,
        blank_zeros=blank_zeros,
    ))

    return outputs


# ---------------------------------------------------------------------
# Source orchestration
# ---------------------------------------------------------------------
def resolve_filtered_path(
    point: str,
    raw_stem: str,
    filtered_dir: Path | None,
    filtered_template: str | None,
    filtered_stem: str | None,
):
    stem = filtered_stem or raw_stem

    if filtered_template:
        return Path(filtered_template.format(point=point, stem=stem, raw_stem=raw_stem))

    if filtered_dir:
        base = Path(filtered_dir) / f"{stem}_filtered_{point}.shw"
        for candidate in (base, Path(str(base) + ".gz"), Path(str(base) + ".xz"), Path(str(base) + ".bz2")):
            if candidate.exists():
                return candidate
        return base

    return Path("04_filtered") / f"{stem}_filtered_{point}.shw"


def process_raw(args):
    print("\n[RAW] Loading geometries")
    geoms = load_geometries(
        points=args.points,
        geom_dir=args.geom_dir,
        geom_template=args.geom_template,
        mask_col=args.mask_col,
        mask_min=args.mask_min,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
    )

    print("[RAW] Counting all points in one SHW read")
    global_summary = count_file(
        args.raw_shw,
        geoms,
        only_muons=args.only_muons,
        shw_format=args.shw_format,
        shw_member=args.shw_member,
        progress_update_mb=args.progress_update_mb,
        head=args.head,
    )

    source_outdir = args.outdir / "raw"
    outputs = []
    plot_data = {}

    print("[RAW] Saving tables")
    for point in args.points:
        outputs.extend(save_point_tables(source_outdir, point, geoms[point], global_summary, args.mask_min))
        plot_data[point] = point_grid_for_plot(geoms[point])
        print(f"  {point}: inside={geoms[point]['n_inside']} | in_grid={geoms[point]['n_in_grid']}")

    outputs.extend(save_plots_for_source(
        "raw",
        plot_data,
        figures_dir=args.outdir / "figures",
        display_step=args.display_step,
        display_theta_min=args.display_theta_min,
        display_theta_max=args.display_theta_max,
        display_phi_min=args.display_phi_min,
        display_phi_max=args.display_phi_max,
        vmax_percentile=args.vmax_percentile,
        blank_zeros=args.blank_zeros,
    ))

    return outputs



def process_filtered_point_worker(payload: dict):
    """
    Worker independiente para un punto filtrado.

    Cada punto filtrado tiene su propio .shw. Por eso esta etapa sí se puede
    paralelizar de forma segura por punto.
    """
    point = payload["point"]

    shw = resolve_filtered_path(
        point=point,
        raw_stem=payload["raw_stem"],
        filtered_dir=payload["filtered_dir"],
        filtered_template=payload["filtered_template"],
        filtered_stem=payload["filtered_stem"],
    )

    if not shw.exists():
        raise FileNotFoundError(
            f"No encontré filtered SHW para {point}: {shw}\n"
            f"Usa --filtered-dir, --filtered-template, o corre con --source raw."
        )

    print(f"[FILTERED] {point}: {shw}")

    geoms = load_geometries(
        points=[point],
        geom_dir=payload["geom_dir"],
        geom_template=payload["geom_template"],
        mask_col=payload["mask_col"],
        mask_min=payload["mask_min"],
        theta_min=payload["theta_min"],
        theta_max=payload["theta_max"],
    )

    global_summary = count_file(
        shw,
        geoms,
        only_muons=payload["only_muons"],
        shw_format=payload["shw_format"],
        progress_update_mb=payload["progress_update_mb"],
        head=payload["head"],
    )

    outputs = save_point_tables(
        payload["source_outdir"],
        point,
        geoms[point],
        global_summary,
        payload["mask_min"],
    )

    return {
        "point": point,
        "outputs": outputs,
        "plot_data": point_grid_for_plot(geoms[point]),
        "n_inside": geoms[point]["n_inside"],
        "n_in_grid": geoms[point]["n_in_grid"],
    }


def process_filtered(args):
    print("\n[FILTERED] Counting each point with its own filtered SHW")
    source_outdir = args.outdir / "filtered"

    outputs = []
    plot_data = {}
    raw_stem = shw_stem(args.raw_shw)

    payloads = []
    for point in args.points:
        payloads.append({
            "point": point,
            "raw_stem": raw_stem,
            "filtered_dir": args.filtered_dir,
            "filtered_template": args.filtered_template,
            "filtered_stem": args.filtered_stem,
            "geom_dir": args.geom_dir,
            "geom_template": args.geom_template,
            "mask_col": args.mask_col,
            "mask_min": args.mask_min,
            "theta_min": args.theta_min,
            "theta_max": args.theta_max,
            "source_outdir": source_outdir,
            "only_muons": args.only_muons,
            "shw_format": args.shw_format,
            "progress_update_mb": args.progress_update_mb,
            "head": args.head,
        })

    workers = max(1, int(args.filtered_workers))

    if workers > 1 and len(payloads) > 1:
        print(f"[FILTERED] parallel workers: {workers}")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_point = {
                ex.submit(process_filtered_point_worker, payload): payload["point"]
                for payload in payloads
            }
            results = []
            for fut in as_completed(future_to_point):
                results.append(fut.result())

        results.sort(key=lambda item: ORDER.index(item["point"]))
    else:
        results = [process_filtered_point_worker(payload) for payload in payloads]

    for item in results:
        point = item["point"]
        outputs.extend(item["outputs"])
        plot_data[point] = item["plot_data"]
        print(f"  {point}: inside={item['n_inside']} | in_grid={item['n_in_grid']}")

    outputs.extend(save_plots_for_source(
        "filtered",
        plot_data,
        figures_dir=args.outdir / "figures",
        display_step=args.display_step,
        display_theta_min=args.display_theta_min,
        display_theta_max=args.display_theta_max,
        display_phi_min=args.display_phi_min,
        display_phi_max=args.display_phi_max,
        vmax_percentile=args.vmax_percentile,
        blank_zeros=args.blank_zeros,
    ))

    return outputs


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="Cuenta muones dentro del volcán y genera figuras individuales + 2x2 para raw y filtered."
    )

    ap.add_argument("--raw-shw", "--shw", dest="raw_shw", required=True, type=Path,
                    help="Archivo .shw raw. También se usa su stem para inferir nombres filtrados.")
    ap.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="auto",
                    help="Formato de entrada. auto detecta ARTI 12 columnas o CNFId energy theta px py pz h bx bz.")
    ap.add_argument("--shw-member", default=None,
                    help="Nombre del miembro dentro de un .tar/.tar.gz. Si se omite, toma el primer .shw.")
    ap.add_argument("--filtered-dir", default=None, type=Path,
                    help="Carpeta con {stem}_filtered_P*.shw")
    ap.add_argument("--filtered-template", default=None,
                    help="Plantilla para filtrados. Variables: {point}, {stem}, {raw_stem}")
    ap.add_argument("--filtered-stem", default=None,
                    help="Stem alternativo para filtrados si difiere del raw_shw.stem")

    ap.add_argument("--source", choices=["raw", "filtered", "both"], default="both",
                    help="Qué fuentes procesar. Default: both.")
    ap.add_argument("--points", nargs="+", default=ORDER, choices=ORDER)

    ap.add_argument("--geom-dir", default=None, type=Path, help="Carpeta con rock_length_P*.csv")
    ap.add_argument("--geom-template", default=None,
                    help="Plantilla para geometría. Ejemplo: '/path/rock_length_{point}.csv'")

    ap.add_argument("--outdir", required=True, type=Path)

    ap.add_argument("--mask-col", default=None)
    ap.add_argument("--mask-min", default=0.0, type=float)
    ap.add_argument("--theta-min", default=0.0, type=float,
                    help="Theta mínimo para contar dentro de la geometría")
    ap.add_argument("--theta-max", default=90.0, type=float,
                    help="Theta máximo para contar dentro de la geometría")

    ap.add_argument("--display-theta-min", default=0.0, type=float,
                    help="Theta mínimo del canvas de las figuras finales")
    ap.add_argument("--display-theta-max", default=90.0, type=float,
                    help="Theta máximo del canvas de las figuras finales")
    ap.add_argument("--display-phi-min", default=-60.0, type=float,
                    help="Phi mínimo del canvas de las figuras finales")
    ap.add_argument("--display-phi-max", default=60.0, type=float,
                    help="Phi máximo del canvas de las figuras finales")
    ap.add_argument("--display-step", default=0.5, type=float,
                    help="Bineado visual cuadrado de las figuras finales, en grados")
    ap.add_argument("--vmax-percentile", default=99.0, type=float,
                    help="Percentil superior para escala lineal")
    ap.add_argument("--show-zeros", dest="blank_zeros", action="store_false",
                    help="Muestra ceros en vez de dejarlos en blanco")
    ap.set_defaults(blank_zeros=True)

    ap.add_argument("--include-all", dest="only_muons", action="store_false",
                    help="Incluye todas las partículas. Default: sólo muones 0005/0006.")
    ap.set_defaults(only_muons=True)

    ap.add_argument("--progress-update-mb", default=32.0, type=float)
    ap.add_argument("--filtered-workers", default=1, type=int,
                    help="Número de procesos para la etapa filtered. Seguro porque cada punto usa su propio .shw.")
    ap.add_argument("--head", default=0, type=int,
                    help="Debug: detiene el conteo tras N eventos dentro por punto.")

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not args.raw_shw.exists():
        raise FileNotFoundError(args.raw_shw)
    if args.geom_template is None and args.geom_dir is None:
        raise ValueError("Usa --geom-dir o --geom-template")

    setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    if args.source in ("raw", "both"):
        all_outputs.extend(process_raw(args))

    if args.source in ("filtered", "both"):
        all_outputs.extend(process_filtered(args))

    manifest = args.outdir / "inside_volcano_merged_manifest.csv"
    pd.DataFrame({"path": [str(p) for p in all_outputs]}).to_csv(manifest, index=False)

    print("\n[OK] Finalizado.")
    print(f"[OK] Manifest: {manifest}")
    print(f"[OK] Figures:  {args.outdir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
