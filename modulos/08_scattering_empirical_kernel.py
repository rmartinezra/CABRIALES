#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_scattering_empirical_kernel.py
---------------------------------
Empirical Geant4 scattering diagnostics for Machín muography.

This script is intentionally parallel to 08_scattering_highland_v2.py. It reads
Ecrit tables and an empirical kernel library extracted from ROOT, then writes
per-cell diagnostics and triptych figures by energy factor.

The recommended downstream model is the full empirical kernel. The columns
`theta0_proj_mrad` and `theta0_proj_deg` are exported only as compatibility
aliases for scripts that expect a Highland-like sigma column.
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

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
    from scipy.spatial import cKDTree, Delaunay
except Exception as exc:  # pragma: no cover
    raise SystemExit("scipy is required for empirical-kernel interpolation") from exc

MUON_MASS_GEV = 0.10565837
DEFAULT_POINTS = ("P1", "P2", "P4", "P5")


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
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        low = col.lower()
        if any(cand.lower() in low for cand in candidates):
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
        half = 0.5 * float(fallback_width or 1.0)
        return np.array([vals[0] - half, vals[0] + half], dtype=float)
    mids = 0.5 * (vals[:-1] + vals[1:])
    return np.concatenate([[vals[0] - (mids[0] - vals[0])], mids, [vals[-1] + (vals[-1] - mids[-1])]]).astype(float)


def factor_tag(factor: float) -> str:
    return f"f{factor:.2f}".replace(".", "p").replace("-", "m")


def normalize_probability(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=float).copy()
    p[~np.isfinite(p)] = 0.0
    p[p < 0.0] = 0.0
    s = float(np.sum(p))
    if s <= 0.0 or not np.isfinite(s):
        return np.zeros_like(p)
    return p / s


class EmpiricalKernelModel:
    """Interpolate K_G4(delta theta | L, Ekin) in u=log(L), v=log(E/L)."""

    def __init__(self, npz_path: Path, interp_method: str = "rbf_linear"):
        self.path = Path(npz_path)
        self.interp_method = interp_method
        data = np.load(self.path, allow_pickle=False)

        required = ["centers_mrad", "edges_mrad", "probabilities", "L_m", "E_in_GeV", "clean_for_kernel"]
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"Kernel library missing keys: {missing}. Available: {data.files}")

        self.centers_mrad = np.asarray(data["centers_mrad"], dtype=float)
        self.edges_mrad = np.asarray(data["edges_mrad"], dtype=float)
        self.probabilities = np.asarray(data["probabilities"], dtype=float)
        self.L_m = np.asarray(data["L_m"], dtype=float)
        self.E_in_GeV = np.asarray(data["E_in_GeV"], dtype=float)
        self.clean_for_kernel = np.asarray(data["clean_for_kernel"], dtype=bool)

        if self.probabilities.shape != (self.L_m.size, self.centers_mrad.size):
            raise ValueError(
                "Expected probabilities with shape (n_kernels, n_centers). "
                f"Got {self.probabilities.shape}, L={self.L_m.size}, centers={self.centers_mrad.size}."
            )

        valid = (
            self.clean_for_kernel &
            np.isfinite(self.L_m) & (self.L_m > 0.0) &
            np.isfinite(self.E_in_GeV) & (self.E_in_GeV > 0.0)
        )
        valid &= np.isfinite(self.probabilities).all(axis=1)
        if not np.any(valid):
            raise ValueError("No valid empirical kernels found after clean_for_kernel filtering.")

        self.valid_mask = valid
        self.points_uv = np.column_stack([
            np.log(self.L_m[valid]),
            np.log(self.E_in_GeV[valid] / self.L_m[valid]),
        ])
        self.values = np.vstack([normalize_probability(row) for row in self.probabilities[valid]])

        self.u_min, self.v_min = np.nanmin(self.points_uv, axis=0)
        self.u_max, self.v_max = np.nanmax(self.points_uv, axis=0)
        self.tree = cKDTree(self.points_uv)
        self.nearest_interp = NearestNDInterpolator(self.points_uv, self.values)

        self.delaunay = None
        if self.points_uv.shape[0] >= 3:
            try:
                self.delaunay = Delaunay(self.points_uv)
            except Exception:
                self.delaunay = None

        self.linear_interp = None
        if self.points_uv.shape[0] >= 3:
            try:
                self.linear_interp = LinearNDInterpolator(self.points_uv, self.values, fill_value=np.nan)
            except Exception:
                self.linear_interp = None

        self.rbf_interp = None
        if self.points_uv.shape[0] >= 2:
            try:
                neighbors = min(40, self.points_uv.shape[0])
                self.rbf_interp = RBFInterpolator(
                    self.points_uv,
                    self.values,
                    kernel="linear",
                    neighbors=neighbors,
                )
            except Exception:
                self.rbf_interp = None

    def _outside_domain(self, uv: np.ndarray) -> bool:
        u, v = float(uv[0]), float(uv[1])
        outside_box = (u < self.u_min) or (u > self.u_max) or (v < self.v_min) or (v > self.v_max)
        if outside_box:
            return True
        if self.delaunay is not None:
            try:
                return bool(self.delaunay.find_simplex(uv.reshape(1, -1))[0] < 0)
            except Exception:
                return outside_box
        return outside_box

    def _nearest(self, uv: np.ndarray) -> np.ndarray:
        _, idx = self.tree.query(uv.reshape(1, -1), k=1)
        return self.values[int(np.ravel(idx)[0])].copy()

    def predict_kernel(self, L_m: float, E_GeV: float):
        meta = {"used_nearest_fallback": False, "outside_domain": False, "valid": False}
        if not (np.isfinite(L_m) and np.isfinite(E_GeV) and L_m > 0.0 and E_GeV > 0.0):
            return self.centers_mrad, np.zeros_like(self.centers_mrad), meta

        uv = np.array([math.log(float(L_m)), math.log(float(E_GeV) / float(L_m))], dtype=float)
        meta["outside_domain"] = self._outside_domain(uv)

        prob = None
        if self.interp_method == "nearest" or meta["outside_domain"]:
            prob = self._nearest(uv)
            meta["used_nearest_fallback"] = True
        elif self.interp_method == "linear":
            if self.linear_interp is not None:
                try:
                    prob = np.asarray(self.linear_interp(uv.reshape(1, -1))[0], dtype=float)
                except Exception:
                    prob = None
            if prob is None or (not np.isfinite(prob).all()) or np.sum(np.clip(prob, 0, None)) <= 0.0:
                prob = self._nearest(uv)
                meta["used_nearest_fallback"] = True
        elif self.interp_method == "rbf_linear":
            if self.rbf_interp is not None:
                try:
                    prob = np.asarray(self.rbf_interp(uv.reshape(1, -1))[0], dtype=float)
                except Exception:
                    prob = None
            if prob is None or (not np.isfinite(prob).all()) or np.sum(np.clip(prob, 0, None)) <= 0.0:
                prob = self._nearest(uv)
                meta["used_nearest_fallback"] = True
        else:
            raise ValueError(f"Unknown interpolation method: {self.interp_method}")

        prob = normalize_probability(prob)
        meta["valid"] = bool(np.sum(prob) > 0.0)
        return self.centers_mrad, prob, meta


def weighted_abs_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.abs(np.asarray(x, dtype=float))
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return 0.0
    xs = x[m]
    ws = w[m]
    order = np.argsort(xs)
    xs = xs[order]
    ws = ws[order]
    cdf = np.cumsum(ws)
    cdf /= cdf[-1]
    return float(np.interp(q, cdf, xs))


def kernel_metrics(centers_mrad: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    p = normalize_probability(prob)
    if np.sum(p) <= 0.0:
        return {
            "RMS_empirical_mrad": 0.0,
            "RMS_empirical_deg": 0.0,
            "abs68_mrad": 0.0,
            "abs90_mrad": 0.0,
            "abs95_mrad": 0.0,
            "Tail5_empirical": 0.0,
            "Tail10_empirical": 0.0,
            "Tail20_empirical": 0.0,
        }
    c = np.asarray(centers_mrad, dtype=float)
    rms = math.sqrt(float(np.sum(p * c * c)))
    return {
        "RMS_empirical_mrad": rms,
        "RMS_empirical_deg": math.degrees(rms / 1000.0),
        "abs68_mrad": weighted_abs_quantile(c, p, 0.68),
        "abs90_mrad": weighted_abs_quantile(c, p, 0.90),
        "abs95_mrad": weighted_abs_quantile(c, p, 0.95),
        "Tail5_empirical": float(np.sum(p[np.abs(c) > 5.0])),
        "Tail10_empirical": float(np.sum(p[np.abs(c) > 10.0])),
        "Tail20_empirical": float(np.sum(p[np.abs(c) > 20.0])),
    }


def load_ecrit_table(path: Path):
    df = pd.read_csv(path)
    theta_col = find_column(df, ["theta_deg", "theta", "zenith_deg"])
    phi_col = find_column(df, ["phi_rel_deg", "phi_deg", "phi", "azimuth_deg"])
    length_col = find_column(df, ["length_inside_m", "L_m", "rock_length_m", "length_m", "longitud_m", "length"])
    tcrit_col = find_column(df, ["Tcrit_GeV", "T_crit_GeV", "kinetic_crit_GeV"], required=False)
    etot_col = find_column(df, ["Ecrit_total_GeV", "E_total_crit_GeV", "Ecrit_GeV", "E_total_GeV"], required=False)
    x_col = find_column(df, ["X_g_cm2", "opacity_g_cm2", "opacity"], required=False)

    df = df.copy()
    for c in [theta_col, phi_col, length_col, tcrit_col, etot_col, x_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if x_col is None:
        df["X_g_cm2"] = np.nan
    elif x_col != "X_g_cm2":
        df["X_g_cm2"] = df[x_col]

    if tcrit_col is None and etot_col is None:
        raise KeyError(f"{path.name} needs Tcrit_GeV or Ecrit_total_GeV.")

    return df, theta_col, phi_col, length_col, tcrit_col, etot_col


def compute_for_factor(df: pd.DataFrame, theta_col: str, phi_col: str, length_col: str,
                       tcrit_col: str | None, etot_col: str | None, factor: float,
                       theta_bin_deg: float, phi_bin_deg: float, model: EmpiricalKernelModel) -> pd.DataFrame:
    out = df.copy()
    L = pd.to_numeric(out[length_col], errors="coerce").to_numpy(dtype=float)
    theta_vals = pd.to_numeric(out[theta_col], errors="coerce").to_numpy(dtype=float)

    if tcrit_col is not None:
        Tcrit = pd.to_numeric(out[tcrit_col], errors="coerce").to_numpy(dtype=float)
        Eref_kin = factor * Tcrit
    else:
        Etotal = pd.to_numeric(out[etot_col], errors="coerce").to_numpy(dtype=float)
        Eref_kin = factor * np.maximum(Etotal - MUON_MASS_GEV, 0.0)

    phi_eff_deg = np.abs(np.sin(np.deg2rad(theta_vals))) * phi_bin_deg
    pixel_min_deg = np.minimum(theta_bin_deg, phi_eff_deg)

    rows = []
    for Li, Ei in zip(L, Eref_kin):
        if not (np.isfinite(Li) and np.isfinite(Ei) and Li > 0.0 and Ei > 0.0):
            metrics = kernel_metrics(model.centers_mrad, np.zeros_like(model.centers_mrad))
            metrics.update({
                "kernel_valid": 0,
                "kernel_nearest_fallback": 0,
                "kernel_outside_domain": 0,
            })
        else:
            centers, prob, meta = model.predict_kernel(float(Li), float(Ei))
            metrics = kernel_metrics(centers, prob)
            metrics.update({
                "kernel_valid": int(meta["valid"]),
                "kernel_nearest_fallback": int(meta["used_nearest_fallback"]),
                "kernel_outside_domain": int(meta["outside_domain"]),
            })
        rows.append(metrics)

    met = pd.DataFrame(rows)
    for col in met.columns:
        out[col] = met[col].to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = out["RMS_empirical_deg"].to_numpy(dtype=float) / pixel_min_deg
    ratio[~np.isfinite(ratio)] = np.nan

    out["energy_factor"] = factor
    out["Eref_kinetic_GeV"] = Eref_kin
    out["Eref_total_GeV_compat"] = Eref_kin + MUON_MASS_GEV
    out["theta_bin_deg"] = theta_bin_deg
    out["phi_bin_deg"] = phi_bin_deg
    out["phi_effective_bin_deg"] = phi_eff_deg
    out["pixel_min_bin_deg"] = pixel_min_deg
    out["RMS_empirical_over_pixel_min_bin"] = ratio

    # Compatibility only: these are not a Gaussian sigma model.
    out["theta0_proj_mrad"] = out["RMS_empirical_mrad"]
    out["theta0_proj_deg"] = out["RMS_empirical_deg"]
    out["theta0_over_pixel_min_bin"] = out["RMS_empirical_over_pixel_min_bin"]
    return out


def summarize(df: pd.DataFrame, point: str, factor: float, in_csv: Path, out_csv: Path,
              theta_bin: float, phi_bin: float, theta_min: float, theta_max: float,
              phi_min: float, phi_max: float, display_step: float) -> dict[str, float | str | int]:
    def pct(col: str, p: float) -> float:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.nanpercentile(vals, p)) if vals.size else float("nan")

    valid = pd.to_numeric(df["kernel_valid"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    fallback = pd.to_numeric(df["kernel_nearest_fallback"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    outside = pd.to_numeric(df["kernel_outside_domain"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0

    return {
        "point": point,
        "energy_factor": factor,
        "n_cells": int(len(df)),
        "n_cells_valid": int(np.sum(valid)),
        "RMS_empirical_mrad_median": pct("RMS_empirical_mrad", 50),
        "RMS_empirical_mrad_p90": pct("RMS_empirical_mrad", 90),
        "Tail10_empirical_median": pct("Tail10_empirical", 50),
        "Tail10_empirical_p90": pct("Tail10_empirical", 90),
        "RMS_empirical_over_pixel_min_bin_median": pct("RMS_empirical_over_pixel_min_bin", 50),
        "RMS_empirical_over_pixel_min_bin_p90": pct("RMS_empirical_over_pixel_min_bin", 90),
        "fraction_nearest_fallback": float(np.mean(fallback[valid])) if np.any(valid) else float("nan"),
        "fraction_outside_domain": float(np.mean(outside[valid])) if np.any(valid) else float("nan"),
        "input_csv": str(in_csv),
        "output_csv": str(out_csv),
        "theta_bin_deg_inferred": theta_bin,
        "phi_bin_deg_inferred": phi_bin,
        "theta_min_used": theta_min,
        "theta_max_used": theta_max,
        "phi_min_used": phi_min,
        "phi_max_used": phi_max,
        "display_step_deg": display_step,
    }


def build_grid(df: pd.DataFrame, theta_col: str, phi_col: str, value_col: str):
    theta = np.array(sorted(df[theta_col].dropna().unique()), dtype=float)
    phi = np.array(sorted(df[phi_col].dropna().unique()), dtype=float)
    Z = np.full((len(theta), len(phi)), np.nan, dtype=float)
    ti = {round(v, 10): i for i, v in enumerate(theta)}
    pj = {round(v, 10): j for j, v in enumerate(phi)}
    for _, row in df.iterrows():
        i = ti.get(round(float(row[theta_col]), 10))
        j = pj.get(round(float(row[phi_col]), 10))
        if i is not None and j is not None:
            Z[i, j] = row[value_col]
    return phi, theta, Z


def square_display_from_df(df, theta_col, phi_col, value_col,
                           theta_min, theta_max, phi_min, phi_max,
                           display_step, theta_bin_deg, phi_bin_deg):
    phis, thetas, Z_src = build_grid(df, theta_col, phi_col, value_col)
    ph_edges_src = centers_to_edges(phis, fallback_width=phi_bin_deg)
    th_edges_src = centers_to_edges(thetas, fallback_width=theta_bin_deg)

    theta_edges = np.arange(theta_min, theta_max + display_step, display_step)
    phi_edges = np.arange(phi_min, phi_max + display_step, display_step)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    Z = np.full((len(theta_centers), len(phi_centers)), np.nan, dtype=float)

    for i, th in enumerate(theta_centers):
        ii = np.searchsorted(th_edges_src, th, side="right") - 1
        if ii < 0 or ii >= Z_src.shape[0]:
            continue
        for j, ph in enumerate(phi_centers):
            jj = np.searchsorted(ph_edges_src, ph, side="right") - 1
            if 0 <= jj < Z_src.shape[1]:
                Z[i, j] = Z_src[ii, jj]
    return phi_edges, theta_edges, Z


def native_display_from_df(df, theta_col, phi_col, value_col, theta_bin_deg, phi_bin_deg):
    phis, thetas, Z = build_grid(df, theta_col, phi_col, value_col)
    return centers_to_edges(phis, phi_bin_deg), centers_to_edges(thetas, theta_bin_deg), Z


def robust_vmax(values: np.ndarray, percentile: float = 98.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vmax = float(np.nanpercentile(vals, percentile))
    return vmax if np.isfinite(vmax) and vmax > 0 else None


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
                display_step, theta_bin_deg, phi_bin_deg,
            )
        else:
            phi_edges, theta_edges, Z = native_display_from_df(df, theta_col, phi_col, value_col,
                                                               theta_bin_deg, phi_bin_deg)
        canvases.append((phi_edges, theta_edges, Z))
        vals = Z[np.isfinite(Z)]
        if vals.size:
            all_vals.append(vals)

    vmax = fixed_vmax if fixed_vmax is not None else (robust_vmax(np.concatenate(all_vals)) if all_vals else None)
    fig, axes = plt.subplots(1, len(factors), figsize=(4.55 * len(factors), 4.6), constrained_layout=True)
    if len(factors) == 1:
        axes = [axes]
    mappable = None
    for ax, factor, canvas in zip(axes, factors, canvases):
        phi_edges, theta_edges, Z = canvas
        kwargs = {"shading": "flat", "cmap": "viridis"}
        if vmax is not None:
            kwargs["vmax"] = vmax
        mappable = ax.pcolormesh(phi_edges, theta_edges, Z, **kwargs)
        ax.set_xlim(phi_min, phi_max)
        ax.set_ylim(theta_max, theta_min)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(rf"$T_{{ref}} = {factor:g}\,T_{{crit}}$")
        ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
        ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
        ax.set_xticks(np.arange(np.ceil(phi_min/20)*20, phi_max + 1e-6, 20))
        ax.set_yticks(np.arange(np.ceil(theta_min/10)*10, theta_max + 1e-6, 10))
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.96, pad=0.02)
        cbar.set_label(cbar_label)
    fig.suptitle(title_prefix, fontsize=12)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def process_point(point: str, indir: Path, outdir: Path, energy_factors: list[float],
                  model: EmpiricalKernelModel, theta_bin_arg: float | None,
                  phi_bin_arg: float | None, theta_min_arg: float | None,
                  theta_max_arg: float, phi_min_arg: float | None,
                  phi_max_arg: float | None, square_display: bool,
                  display_step: float | None) -> list[dict]:
    in_csv = indir / f"ecrit_table_{point}.csv"
    if not in_csv.exists():
        print(f"[WARN] Missing {in_csv}. Skipping {point}.")
        return []

    df, theta_col, phi_col, length_col, tcrit_col, etot_col = load_ecrit_table(in_csv)
    df = df.dropna(subset=[theta_col, phi_col, length_col]).copy()
    if df.empty:
        print(f"[WARN] No valid rows in {in_csv}. Skipping {point}.")
        return []

    theta_min = float(np.nanmin(df[theta_col])) if theta_min_arg is None else theta_min_arg
    theta_max = min(float(theta_max_arg), 90.0)
    df = df[(df[theta_col] >= theta_min) & (df[theta_col] <= theta_max)].copy()
    if df.empty:
        print(f"[WARN] No cells remain after theta cut for {point}.")
        return []

    if phi_min_arg is not None:
        df = df[df[phi_col] >= phi_min_arg].copy()
    if phi_max_arg is not None:
        df = df[df[phi_col] <= phi_max_arg].copy()
    if df.empty:
        print(f"[WARN] No cells remain after phi cut for {point}.")
        return []

    theta_bin = infer_bin_width(df[theta_col].to_numpy(dtype=float), theta_bin_arg)
    phi_bin = infer_bin_width(df[phi_col].to_numpy(dtype=float), phi_bin_arg)
    phi_min = float(np.nanmin(df[phi_col])) if phi_min_arg is None else phi_min_arg
    phi_max = float(np.nanmax(df[phi_col])) if phi_max_arg is None else phi_max_arg
    if display_step is None:
        display_step = min(theta_bin, phi_bin)

    point_dir = outdir / point
    point_dir.mkdir(parents=True, exist_ok=True)

    all_scat = []
    summaries = []
    for factor in energy_factors:
        tag = factor_tag(factor)
        scat = compute_for_factor(
            df=df,
            theta_col=theta_col,
            phi_col=phi_col,
            length_col=length_col,
            tcrit_col=tcrit_col,
            etot_col=etot_col,
            factor=factor,
            theta_bin_deg=theta_bin,
            phi_bin_deg=phi_bin,
            model=model,
        )
        out_csv = point_dir / f"scattering_empirical_table_{point}_{tag}.csv"
        scat.to_csv(out_csv, index=False)
        all_scat.append(scat)
        summaries.append(summarize(scat, point, factor, in_csv, out_csv,
                                   theta_bin, phi_bin, theta_min, theta_max,
                                   phi_min, phi_max, display_step))

    make_triptych(
        all_scat, energy_factors, theta_col, phi_col, "RMS_empirical_mrad",
        point_dir / f"RMS_empirical_mrad_triptych_{point}.png",
        r"RMS empirical (mrad)", f"Empirical Geant4 RMS — {point}",
        theta_bin, phi_bin, theta_min, theta_max, phi_min, phi_max,
        square_display, display_step,
    )
    make_triptych(
        all_scat, energy_factors, theta_col, phi_col, "Tail10_empirical",
        point_dir / f"Tail10_empirical_triptych_{point}.png",
        r"$P(|\Delta\alpha|>10\,\mathrm{mrad})$", f"Empirical Geant4 tails — {point}",
        theta_bin, phi_bin, theta_min, theta_max, phi_min, phi_max,
        square_display, display_step, fixed_vmax=1.0,
    )
    make_triptych(
        all_scat, energy_factors, theta_col, phi_col, "RMS_empirical_over_pixel_min_bin",
        point_dir / f"RMS_empirical_over_pixel_min_triptych_{point}.png",
        r"RMS / $\Delta\alpha_{\min}$", f"Empirical scattering relative to angular pixel — {point}",
        theta_bin, phi_bin, theta_min, theta_max, phi_min, phi_max,
        square_display, display_step, fixed_vmax=1.0,
    )

    print(f"[OK] {point}: empirical scattering tables and triptychs created")
    return summaries


def build_parser():
    ap = argparse.ArgumentParser(description="Empirical Geant4 scattering diagnostics with 1xN comparison figures.")
    ap.add_argument("--indir", type=Path, default=Path("outputs"), help="Directory with ecrit_table_{POINT}.csv files.")
    ap.add_argument("--outdir", type=Path, default=Path("outputs_scattering_empirical"), help="Output directory.")
    ap.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS), help="Points to process.")
    ap.add_argument("--kernel-library", type=Path, required=True, help="Path to empirical_kernel_library.npz")
    ap.add_argument("--energy-factors", nargs="+", type=float, default=[1.0, 1.5, 2.0], help="Reference kinetic energy factors: Tref = factor*Tcrit.")
    ap.add_argument("--interp-method", choices=["rbf_linear", "linear", "nearest"], default="rbf_linear")
    ap.add_argument("--theta-bin-deg", type=float, default=None, help="Fallback theta bin width.")
    ap.add_argument("--phi-bin-deg", type=float, default=None, help="Fallback phi bin width.")
    ap.add_argument("--theta-min", type=float, default=None, help="Minimum theta to show/use. Default: inferred from data.")
    ap.add_argument("--theta-max", type=float, default=90.0, help="Maximum theta to show/use. Default: 90 deg.")
    ap.add_argument("--phi-min", type=float, default=None, help="Minimum phi to show. Default: inferred from data.")
    ap.add_argument("--phi-max", type=float, default=None, help="Maximum phi to show. Default: inferred from data.")
    ap.add_argument("--square-display", action="store_true", help="Regrid only for display to square angular bins.")
    ap.add_argument("--display-step", type=float, default=None, help="Square display step in deg. Default: min(theta_bin, phi_bin).")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_article_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    model = EmpiricalKernelModel(args.kernel_library, interp_method=args.interp_method)
    summaries = []
    for point in args.points:
        summaries.extend(process_point(
            point=point,
            indir=args.indir,
            outdir=args.outdir,
            energy_factors=args.energy_factors,
            model=model,
            theta_bin_arg=args.theta_bin_deg,
            phi_bin_arg=args.phi_bin_deg,
            theta_min_arg=args.theta_min,
            theta_max_arg=args.theta_max,
            phi_min_arg=args.phi_min,
            phi_max_arg=args.phi_max,
            square_display=args.square_display,
            display_step=args.display_step,
        ))

    if summaries:
        summary_csv = args.outdir / "scattering_empirical_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_csv, index=False)
        print(f"[DONE] Summary: {summary_csv}")
    else:
        print("[WARN] No empirical scattering outputs were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
