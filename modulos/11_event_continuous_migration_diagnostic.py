#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuous event-level angular migration diagnostic.

This script complements the binned event-MC pipeline.  It keeps one row per muon
with its continuous input direction and a sampled continuous output direction,
then makes both point-cloud and binned views of the same migration.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script needs scipy. Install it with: pip install scipy") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modulos.empirical_kernel_io import TailAwareEmpiricalKernel, load_empirical_kernel_library
from modulos.plot_style import apply_scientific_style
from modulos.shw_io import MUON_MASS_GEV, open_shw_bytes, parse_muon_parts, theta_phi_from_momentum


SUMMIT = (4.486552, -75.388975)
POINTS = {
    "P1": (4.492298, -75.381092),
    "P2": (4.494946, -75.388110),
    "P4": (4.476500, -75.386500),
    "P5": (4.488500, -75.379500),
}


@dataclass
class Grid:
    theta: np.ndarray
    phi: np.ndarray
    theta_edges: np.ndarray
    phi_edges: np.ndarray
    length_m: np.ndarray
    inside_mask: np.ndarray
    filled: np.ndarray
    phi0: float


@dataclass
class KernelPrediction:
    centers_mrad: np.ndarray
    probability_per_bin: np.ndarray
    used_nearest_fallback: bool
    outside_domain: bool
    valid: bool


def azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def phi0_for_point(point: str) -> float:
    plat, plon = POINTS[point]
    az_geo = azimuth_deg(plat, plon, SUMMIT[0], SUMMIT[1])
    return (90.0 - az_geo) % 360.0


def find_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        low = col.lower()
        if any(cand.lower() in low for cand in candidates):
            return col
    if required:
        raise KeyError(f"No encontre columnas {list(candidates)}. Disponibles: {list(df.columns)}")
    return None


def centers_to_edges(centers: np.ndarray, fallback_step: float = 1.0) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    c = np.array(sorted(np.unique(c[np.isfinite(c)])), dtype=float)
    if c.size == 0:
        raise ValueError("Empty coordinate array")
    if c.size == 1:
        half = 0.5 * fallback_step
        return np.array([c[0] - half, c[0] + half], dtype=float)
    mids = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mids[0] - c[0])], mids, [c[-1] + (c[-1] - mids[-1])]])


def bin_width(centers: np.ndarray) -> float:
    c = np.asarray(centers, dtype=float)
    c = np.array(sorted(np.unique(c[np.isfinite(c)])), dtype=float)
    if c.size < 2:
        return 1.0
    d = np.diff(c)
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def wrap180(phi: float) -> float:
    out = ((float(phi) + 180.0) % 360.0) - 180.0
    return 180.0 if out == -180.0 else out


def load_grid(ecrit_csv: Path, point: str, theta_min: float | None, theta_max: float | None,
              phi_min: float | None, phi_max: float | None) -> Grid:
    df = pd.read_csv(ecrit_csv)
    th_col = find_col(df, ["theta_deg", "theta", "zenith_deg"])
    ph_col = find_col(df, ["phi_rel_deg", "phi_deg", "phi", "azimuth_deg"])
    length_col = find_col(df, ["length_inside_m", "L_m", "rock_length_m", "length_m", "longitud_m", "length"])
    inside_col = find_col(df, ["inside_volcano_geometry", "inside_geometry", "inside"], required=False)

    df = df.copy()
    for col in (th_col, ph_col, length_col, inside_col):
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[th_col, ph_col])
    if theta_min is not None:
        df = df[df[th_col] >= theta_min]
    if theta_max is not None:
        df = df[df[th_col] <= theta_max]
    if phi_min is not None:
        df = df[df[ph_col] >= phi_min]
    if phi_max is not None:
        df = df[df[ph_col] <= phi_max]
    if df.empty:
        raise RuntimeError(f"No angular cells remain after cuts in {ecrit_csv}")

    theta = np.array(sorted(df[th_col].unique()), dtype=float)
    phi = np.array(sorted(df[ph_col].unique()), dtype=float)
    theta_edges = centers_to_edges(theta, bin_width(theta))
    phi_edges = centers_to_edges(phi, bin_width(phi))
    length = np.zeros((len(theta), len(phi)), dtype=float)
    inside = np.zeros_like(length, dtype=bool)
    filled = np.zeros_like(length, dtype=bool)
    ti = {round(v, 10): i for i, v in enumerate(theta)}
    pj = {round(v, 10): j for j, v in enumerate(phi)}

    for row in df.itertuples(index=False):
        row_map = dict(zip(df.columns, row))
        i = ti.get(round(float(row_map[th_col]), 10))
        j = pj.get(round(float(row_map[ph_col]), 10))
        if i is None or j is None:
            continue
        L = row_map[length_col]
        length[i, j] = float(L) if pd.notna(L) and np.isfinite(L) else 0.0
        if inside_col is not None:
            inside[i, j] = bool(row_map[inside_col])
        else:
            inside[i, j] = length[i, j] > 0.0
        filled[i, j] = True

    return Grid(
        theta=theta,
        phi=phi,
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        length_m=length,
        inside_mask=inside,
        filled=filled,
        phi0=phi0_for_point(point),
    )


class EmpiricalKernelModel:
    def __init__(self, npz_path: Path, interp_method: str = "tail-aware", rbf_smoothing: float = 0.0) -> None:
        self.tail_aware = None
        if interp_method == "tail-aware":
            self.tail_aware = TailAwareEmpiricalKernel(npz_path, energy_cache_dlog=0.02)
            self.kernel_family = self.tail_aware.kernel_family
            self.centers_mrad = self.tail_aware.centers_mrad
            self.edges_mrad = self.tail_aware.edges_mrad
            self.widths_mrad = self.tail_aware.widths_mrad
            self.interp_method = interp_method
            self.rbf_smoothing = rbf_smoothing
            return
        lib = load_empirical_kernel_library(npz_path)
        self.kernel_family = lib.family
        self.centers_mrad = lib.centers_mrad
        self.edges_mrad = lib.edges_mrad
        self.widths_mrad = np.diff(self.edges_mrad)
        self.probabilities = lib.probabilities
        self.L_m = lib.L_m
        self.E_in_GeV = lib.E_in_GeV
        self.clean = lib.clean_for_kernel
        self.interp_method = interp_method

        valid = (
            self.clean
            & np.isfinite(self.L_m) & (self.L_m > 0)
            & np.isfinite(self.E_in_GeV) & (self.E_in_GeV > 0)
            & np.isfinite(self.probabilities).all(axis=1)
            & (self.probabilities.sum(axis=1) > 0)
        )
        if np.sum(valid) < 4:
            raise RuntimeError("Too few clean kernels in empirical library.")

        self.train_probs = self.probabilities[valid]
        self.X = np.column_stack([
            np.log(self.L_m[valid]),
            np.log(self.E_in_GeV[valid] / self.L_m[valid]),
        ])
        self.X_min = self.X.min(axis=0)
        self.X_max = self.X.max(axis=0)
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0)
        self.X_std[self.X_std == 0] = 1.0
        self.X_scaled = (self.X - self.X_mean) / self.X_std
        self.nearest = NearestNDInterpolator(self.X, self.train_probs)
        self.linear = None
        self.rbf = None
        if interp_method == "linear":
            self.linear = LinearNDInterpolator(self.X, self.train_probs, fill_value=np.nan)
        elif interp_method == "rbf_linear":
            self.rbf = RBFInterpolator(self.X_scaled, self.train_probs, kernel="linear", smoothing=rbf_smoothing)
        elif interp_method != "nearest":
            raise ValueError("interp_method must be: linear, rbf_linear, nearest")

    def _normalize(self, probability: np.ndarray) -> tuple[np.ndarray, bool]:
        p = np.asarray(probability, dtype=float).copy()
        p[~np.isfinite(p)] = 0.0
        p[p < 0.0] = 0.0
        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            return np.zeros_like(self.centers_mrad), False
        p /= s
        p = 0.5 * (p + p[::-1])
        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            return np.zeros_like(self.centers_mrad), False
        return p / s, True

    def predict_kernel(self, L_m: float, E_GeV: float) -> KernelPrediction:
        if self.tail_aware is not None:
            pred = self.tail_aware.predict_kernel(L_m, E_GeV)
            return KernelPrediction(
                pred.centers_mrad,
                pred.probability_per_bin,
                pred.used_nearest_fallback,
                pred.outside_domain,
                pred.valid,
            )
        if (not np.isfinite(L_m)) or (not np.isfinite(E_GeV)) or L_m <= 0.0 or E_GeV <= 0.0:
            return KernelPrediction(self.centers_mrad.copy(), np.zeros_like(self.centers_mrad), False, False, False)
        q = np.array([math.log(float(L_m)), math.log(float(E_GeV) / float(L_m))], dtype=float)
        outside = bool(np.any(q < self.X_min) or np.any(q > self.X_max))
        used_nearest = False
        if self.interp_method == "nearest" or outside:
            pred = self.nearest(q)
            used_nearest = True
        elif self.interp_method == "linear":
            pred = self.linear(q)
            if np.ndim(pred) > 1:
                pred = pred[0]
            if np.any(~np.isfinite(pred)) or np.all(np.asarray(pred) <= 0.0):
                pred = self.nearest(q)
                used_nearest = True
        else:
            qs = (q - self.X_mean) / self.X_std
            pred = self.rbf(qs[None, :])[0]
            if np.any(~np.isfinite(pred)) or np.all(np.asarray(pred) <= 0.0):
                pred = self.nearest(q)
                used_nearest = True
        if np.ndim(pred) > 1:
            pred = pred[0]
        p, ok = self._normalize(pred)
        if not ok:
            pred = self.nearest(q)
            if np.ndim(pred) > 1:
                pred = pred[0]
            used_nearest = True
            p, ok = self._normalize(pred)
        return KernelPrediction(self.centers_mrad.copy(), p, used_nearest, outside, ok)


def sample_delta_mrad(rng: np.random.Generator, centers: np.ndarray, probability: np.ndarray,
                      widths_mrad: np.ndarray, threshold: float) -> float:
    p = np.asarray(probability, dtype=float).copy()
    density = p / widths_mrad
    if threshold > 0.0:
        p[density < threshold] = 0.0
    p[~np.isfinite(p)] = 0.0
    p[p < 0.0] = 0.0
    s = float(p.sum())
    if s <= 0.0 or not np.isfinite(s):
        return 0.0
    cdf = np.cumsum(p / s)
    cdf[-1] = 1.0
    idx = int(np.searchsorted(cdf, rng.random(), side="right"))
    idx = min(idx, len(centers) - 1)
    return float(centers[idx])


def make_events(args, grid: Grid, model: EmpiricalKernelModel) -> pd.DataFrame:
    rng = np.random.default_rng(args.random_seed)
    rows = []
    event_id = 0
    rad_to_mrad = 1000.0 * math.pi / 180.0

    with open_shw_bytes(args.shw, member_name=args.shw_member) as handle:
        for raw in handle:
            s = raw.strip()
            if not s or s.startswith(b"#"):
                continue
            rec = parse_muon_parts(s.split(), shw_format=args.shw_format, only_muons=True)
            if rec is None:
                continue
            angles = theta_phi_from_momentum(rec.px, rec.py, rec.pz)
            if angles is None:
                continue
            theta_in, phi_abs = angles
            phi_in = wrap180((phi_abs - grid.phi0) % 360.0)
            i = int(np.searchsorted(grid.theta_edges, theta_in, side="right") - 1)
            j = int(np.searchsorted(grid.phi_edges, phi_in, side="right") - 1)
            if i < 0 or i >= len(grid.theta) or j < 0 or j >= len(grid.phi) or not grid.filled[i, j]:
                continue

            event_id += 1
            L = float(grid.length_m[i, j])
            T = float(rec.e_total_GeV - MUON_MASS_GEV)
            inside_source = bool(grid.inside_mask[i, j])
            theta_out = float(theta_in)
            phi_out = float(phi_in)
            dtheta_mrad = 0.0
            dphi_eff_mrad = 0.0
            used_nearest = False
            outside_domain = False
            valid_kernel = False
            smeared = False

            if inside_source and L > 0.0 and T > 0.0 and np.isfinite(T):
                pred = model.predict_kernel(L, T)
                used_nearest = pred.used_nearest_fallback
                outside_domain = pred.outside_domain
                valid_kernel = pred.valid
                if pred.valid:
                    dtheta_mrad = sample_delta_mrad(
                        rng, pred.centers_mrad, pred.probability_per_bin, model.widths_mrad, args.kernel_threshold
                    )
                    dphi_eff_mrad = sample_delta_mrad(
                        rng, pred.centers_mrad, pred.probability_per_bin, model.widths_mrad, args.kernel_threshold
                    )
                    theta_out = theta_in + dtheta_mrad / rad_to_mrad
                    sin_th = max(abs(math.sin(math.radians(theta_in))), 1e-3)
                    phi_out = wrap180(phi_in + dphi_eff_mrad / (rad_to_mrad * sin_th))
                    smeared = True

            rows.append({
                "event_id": event_id,
                "theta_in_deg": theta_in,
                "phi_in_deg": phi_in,
                "theta_out_deg": theta_out,
                "phi_out_deg": phi_out,
                "delta_theta_mrad": dtheta_mrad,
                "delta_phi_eff_mrad": dphi_eff_mrad,
                "kinetic_GeV": T,
                "length_inside_m": L,
                "inside_source": int(inside_source),
                "smeared": int(smeared),
                "used_nearest_fallback": int(used_nearest),
                "outside_domain": int(outside_domain),
                "valid_kernel": int(valid_kernel),
            })
            if args.head and event_id >= args.head:
                break
    return pd.DataFrame(rows)


def bin_events(theta: np.ndarray, phi: np.ndarray, grid: Grid) -> np.ndarray:
    hist, _, _ = np.histogram2d(theta, phi, bins=[grid.theta_edges, grid.phi_edges])
    return hist


def apply_axes(ax, grid: Grid, args) -> None:
    ax.set_xlim(args.phi_min, args.phi_max)
    ax.set_ylim(args.theta_max, args.theta_min)
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
    ax.set_aspect("equal", adjustable="box")
    try:
        ax.contour(grid.phi, grid.theta, grid.inside_mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    except Exception:
        pass


def plot_point_cloud(df: pd.DataFrame, grid: Grid, args, subset_name: str, out_png: Path) -> None:
    if subset_name == "inside":
        data = df[df["inside_source"] == 1].copy()
        title = "P1 inside-volcano muons"
    else:
        data = df.copy()
        title = "P1 all measured muons: inside + sky"
    inside = data["inside_source"].to_numpy(bool)
    inside_size = 56 if subset_name == "inside" else 22

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].scatter(data.loc[~inside, "phi_in_deg"], data.loc[~inside, "theta_in_deg"], s=5, c="0.55", alpha=0.55, label="sky")
    axes[0].scatter(data.loc[inside, "phi_in_deg"], data.loc[inside, "theta_in_deg"], s=inside_size, c="#ffb000", edgecolors="k", linewidths=0.25, label="inside")
    axes[0].set_title("Input continuous positions")

    axes[1].scatter(data.loc[~inside, "phi_out_deg"], data.loc[~inside, "theta_out_deg"], s=5, c="0.55", alpha=0.55, label="sky")
    axes[1].scatter(data.loc[inside, "phi_out_deg"], data.loc[inside, "theta_out_deg"], s=inside_size, c="#d62728", edgecolors="k", linewidths=0.25, label="inside after kernel")
    axes[1].set_title("Sampled final positions")

    moved = data[data["smeared"] == 1]
    axes[2].scatter(data["phi_in_deg"], data["theta_in_deg"], s=4, c="0.80", alpha=0.40)
    if not moved.empty:
        axes[2].quiver(
            moved["phi_in_deg"], moved["theta_in_deg"],
            moved["phi_out_deg"] - moved["phi_in_deg"],
            moved["theta_out_deg"] - moved["theta_in_deg"],
            angles="xy", scale_units="xy", scale=1.0, width=0.003,
            color="#d62728", alpha=0.85,
        )
    axes[2].set_title("Input -> final displacement")

    for ax in axes:
        apply_axes(ax, grid, args)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(f"{title} | continuous event diagnostic", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_inside_zoom(df: pd.DataFrame, grid: Grid, args, out_png: Path) -> None:
    data = df[df["inside_source"] == 1].copy()
    if data.empty:
        return
    phi_vals = np.concatenate([data["phi_in_deg"].to_numpy(float), data["phi_out_deg"].to_numpy(float)])
    theta_vals = np.concatenate([data["theta_in_deg"].to_numpy(float), data["theta_out_deg"].to_numpy(float)])
    phi_pad = max(3.0, 0.25 * (float(phi_vals.max() - phi_vals.min()) + 1e-9))
    theta_pad = max(3.0, 0.25 * (float(theta_vals.max() - theta_vals.min()) + 1e-9))

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    ax.scatter(data["phi_in_deg"], data["theta_in_deg"], s=90, c="#ffb000", edgecolors="k", label="input")
    ax.scatter(data["phi_out_deg"], data["theta_out_deg"], s=90, c="#d62728", edgecolors="k", label="final")
    ax.quiver(
        data["phi_in_deg"], data["theta_in_deg"],
        data["phi_out_deg"] - data["phi_in_deg"],
        data["theta_out_deg"] - data["theta_in_deg"],
        angles="xy", scale_units="xy", scale=1.0, width=0.0045,
        color="#d62728", alpha=0.85,
    )
    for _, row in data.iterrows():
        ax.text(row["phi_out_deg"] + 0.25, row["theta_out_deg"] + 0.25, str(int(row["event_id"])), fontsize=8)
    try:
        ax.contour(grid.phi, grid.theta, grid.inside_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)
    except Exception:
        pass
    ax.set_xlim(float(phi_vals.min() - phi_pad), float(phi_vals.max() + phi_pad))
    ax.set_ylim(float(theta_vals.max() + theta_pad), float(theta_vals.min() - theta_pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
    ax.set_title("P1 inside-volcano continuous migration zoom")
    ax.legend(loc="best", frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=240)
    plt.close(fig)


def plot_binned(df: pd.DataFrame, grid: Grid, args, subset_name: str, out_png: Path) -> None:
    if subset_name == "inside":
        data = df[df["inside_source"] == 1].copy()
        title = "P1 inside-volcano muons"
    else:
        data = df.copy()
        title = "P1 all measured muons: inside + sky"
    H_in = bin_events(data["theta_in_deg"].to_numpy(float), data["phi_in_deg"].to_numpy(float), grid)
    H_out = bin_events(data["theta_out_deg"].to_numpy(float), data["phi_out_deg"].to_numpy(float), grid)
    delta = H_out - H_in
    vmax = max(float(H_in.max()), float(H_out.max()), 1.0)
    finite_delta = delta[np.isfinite(delta)]
    dv = float(np.nanpercentile(np.abs(finite_delta), 99)) if finite_delta.size else 1.0
    dv = max(dv, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, H, name in ((axes[0], H_in, "Input binned"), (axes[1], H_out, "Final binned")):
        im = ax.pcolormesh(grid.phi_edges, grid.theta_edges, H, shading="flat", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{name} | total={H.sum():.0f}")
        fig.colorbar(im, ax=ax, shrink=0.9, label="counts")
        apply_axes(ax, grid, args)

    im = axes[2].pcolormesh(
        grid.phi_edges, grid.theta_edges, delta, shading="flat", cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-dv, vcenter=0.0, vmax=dv),
    )
    axes[2].set_title("Final - input")
    fig.colorbar(im, ax=axes[2], shrink=0.9, label="counts")
    apply_axes(axes[2], grid, args)

    fig.suptitle(f"{title} | finite angular bins", fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Continuous per-muon angular migration diagnostic.")
    ap.add_argument("--point", default="P1", choices=sorted(POINTS))
    ap.add_argument("--shw", type=Path, required=True)
    ap.add_argument("--shw-format", default="auto", choices=["auto", "arti12", "cnf9"])
    ap.add_argument("--shw-member", default=None)
    ap.add_argument("--ecrit-csv", type=Path, required=True)
    ap.add_argument(
        "--kernel-library",
        type=Path,
        default=REPO_ROOT / "modulos/hybrid_empirical_kernel_library.npz",
        help="Empirical kernel library (default: bundled hybrid full-tail model).",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--interp-method", choices=["tail-aware", "linear", "rbf_linear", "nearest"], default="tail-aware")
    ap.add_argument("--rbf-smoothing", type=float, default=0.0)
    ap.add_argument("--kernel-threshold", type=float, default=0.0)
    ap.add_argument("--theta-min", type=float, default=60.0)
    ap.add_argument("--theta-max", type=float, default=90.0)
    ap.add_argument("--phi-min", type=float, default=-60.0)
    ap.add_argument("--phi-max", type=float, default=60.0)
    ap.add_argument("--random-seed", type=int, default=12345)
    ap.add_argument("--head", type=int, default=0)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    apply_scientific_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    grid = load_grid(args.ecrit_csv, args.point, args.theta_min, args.theta_max, args.phi_min, args.phi_max)
    model = EmpiricalKernelModel(args.kernel_library, args.interp_method, args.rbf_smoothing)
    events = make_events(args, grid, model)
    events_csv = args.outdir / f"continuous_migration_events_{args.point}.csv"
    events.to_csv(events_csv, index=False)

    summary = {
        "point": args.point,
        "kernel_library": str(args.kernel_library),
        "kernel_family": model.kernel_family,
        "interp_method": args.interp_method,
        "kernel_threshold": float(args.kernel_threshold),
        "events_all": int(len(events)),
        "events_inside_source": int(events["inside_source"].sum()) if not events.empty else 0,
        "events_smeared": int(events["smeared"].sum()) if not events.empty else 0,
        "nearest_fallback": int(events["used_nearest_fallback"].sum()) if not events.empty else 0,
        "outside_domain": int(events["outside_domain"].sum()) if not events.empty else 0,
        "events_csv": str(events_csv),
    }
    pd.DataFrame([summary]).to_csv(args.outdir / f"continuous_migration_summary_{args.point}.csv", index=False)

    plot_point_cloud(events, grid, args, "all", args.outdir / f"continuous_points_all_{args.point}.png")
    plot_point_cloud(events, grid, args, "inside", args.outdir / f"continuous_points_inside_{args.point}.png")
    plot_inside_zoom(events, grid, args, args.outdir / f"continuous_points_inside_zoom_{args.point}.png")
    plot_binned(events, grid, args, "all", args.outdir / f"binned_from_continuous_all_{args.point}.png")
    plot_binned(events, grid, args, "inside", args.outdir / f"binned_from_continuous_inside_{args.point}.png")

    print("[OK] Continuous migration diagnostic")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
