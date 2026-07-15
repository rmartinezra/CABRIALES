#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_apply_event_by_event_empirical_mc.py
--------------------------------------
Event-by-event Monte Carlo angular smearing using an empirical Geant4 kernel.

This script avoids the E = factor*Tcrit approximation. For each muon in a .shw
file, it uses the muon's kinetic energy and the rock length of its angular cell:

    K_G4(Delta theta | L(theta,phi), T_muon)

Typical filtered run:

python3 10_apply_event_by_event_empirical_mc.py \
  --points P1 P2 P4 P5 \
  --shw-template 'run_machin/04_filtered/bga_CNF_604800s_filtered_{point}.shw' \
  --ecrit-template 'run_machin/03_ecrit/ecrit_table_{point}.csv' \
  --kernel-library 'analysis_mcs_kernel_fixed/model/empirical_kernel_library.npz' \
  --outdir 'run_machin/09_event_mc_empirical' \
  --workers 4 \
  --random-seed 12345

Typical raw run, one SHW reused for all points:

python3 10_apply_event_by_event_empirical_mc.py \
  --points P1 P2 P4 P5 \
  --shw data/bga_CNF_604800s.shw \
  --ecrit-template 'run_machin/03_ecrit/ecrit_table_{point}.csv' \
  --kernel-library 'analysis_mcs_kernel_fixed/model/empirical_kernel_library.npz' \
  --outdir 'run_machin/09_event_mc_empirical_raw' \
  --workers 4

Notes
-----
- px, py, pz are assumed to be in GeV/c, as in the existing pipeline.
- kinetic energy is computed as sqrt(p^2 + m_mu^2) - m_mu.
- theta and phi conventions match the existing 05/06 scripts:
      theta = acos(pz/|p|)
      phi_abs = atan2(py, px) mapped to [0,360)
      phi_rel = phi_abs - phi0(point), optionally wrapped to [-180,180)
- The output map is produced by sampling one destination angular bin per input muon.
- A cache is used for (source cell, quantized kinetic energy) to avoid rebuilding
  the same 2D kernel many times.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    from empirical_kernel_io import TailAwareEmpiricalKernel, load_empirical_kernel_library
    from plot_style import apply_scientific_style
    from shw_io import open_shw_bytes, parse_muon_parts, stream_size_hint, theta_phi_from_momentum
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from empirical_kernel_io import TailAwareEmpiricalKernel, load_empirical_kernel_library
    from plot_style import apply_scientific_style
    from shw_io import open_shw_bytes, parse_muon_parts, stream_size_hint, theta_phi_from_momentum

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script needs scipy. Install it with: pip install scipy") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    class tqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def update(self, n=1):
            pass
        def close(self):
            pass


MUON_MASS_GEV = 0.10565837
MUON_IDS_B = {b"0005", b"0006", b"5", b"6"}  # mu-/mu+
DEFAULT_POINTS = ("P1", "P2", "P4", "P5")
SUMMIT = (4.486552, -75.388975)
POINTS = {
    "P1": (4.492298, -75.381092),
    "P2": (4.494946, -75.388110),
    "P4": (4.476500, -75.386500),
    "P5": (4.488500, -75.379500),
}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def setup_style() -> None:
    apply_scientific_style()


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
        for cand in candidates:
            if cand.lower() in low:
                return col
    if required:
        raise KeyError(f"No encontré columnas {list(candidates)}. Disponibles: {list(df.columns)}")
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


def safe_rel_delta(out: np.ndarray, inp: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (out - inp) / inp
        rel[~np.isfinite(rel)] = np.nan
    return rel


def output_prefix_for_source_mode(source_mode: str) -> str:
    return "event_mc_inside_source" if source_mode == "inside" else "event_mc"


# -----------------------------------------------------------------------------
# Empirical kernel model
# -----------------------------------------------------------------------------
@dataclass
class KernelPrediction:
    centers_mrad: np.ndarray
    probability_per_bin: np.ndarray
    used_nearest_fallback: bool
    outside_domain: bool
    valid: bool


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
        self.rbf_smoothing = rbf_smoothing

        valid = (
            self.clean
            & np.isfinite(self.L_m) & (self.L_m > 0)
            & np.isfinite(self.E_in_GeV) & (self.E_in_GeV > 0)
            & np.isfinite(self.probabilities).all(axis=1)
            & (self.probabilities.sum(axis=1) > 0)
        )
        if np.sum(valid) < 4:
            raise RuntimeError("Too few clean kernels in empirical library.")

        self.train_L = self.L_m[valid]
        self.train_E = self.E_in_GeV[valid]
        self.train_probs = self.probabilities[valid]
        self.X = np.column_stack([np.log(self.train_L), np.log(self.train_E / self.train_L)])
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
        elif interp_method == "nearest":
            pass
        else:
            raise ValueError("interp_method must be: linear, rbf_linear, nearest")

    def _query(self, L_m: float, E_GeV: float) -> np.ndarray:
        return np.array([math.log(float(L_m)), math.log(float(E_GeV) / float(L_m))], dtype=float)

    def _normalize(self, p: np.ndarray) -> tuple[np.ndarray, bool]:
        p = np.asarray(p, dtype=float).copy()
        p[~np.isfinite(p)] = 0.0
        p[p < 0.0] = 0.0
        s = float(np.sum(p))
        if s <= 0.0 or not np.isfinite(s):
            return np.zeros_like(self.centers_mrad, dtype=float), False
        p /= s
        p = 0.5 * (p + p[::-1])
        s = float(np.sum(p))
        if s <= 0.0 or not np.isfinite(s):
            return np.zeros_like(self.centers_mrad, dtype=float), False
        p /= s
        return p, True

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

        q = self._query(L_m, E_GeV)
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
        elif self.interp_method == "rbf_linear":
            qs = (q - self.X_mean) / self.X_std
            pred = self.rbf(qs[None, :])[0]
            if np.any(~np.isfinite(pred)) or np.all(np.asarray(pred) <= 0.0):
                pred = self.nearest(q)
                used_nearest = True
        else:  # pragma: no cover
            raise ValueError(self.interp_method)

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


# -----------------------------------------------------------------------------
# Geometry / angular grid
# -----------------------------------------------------------------------------
@dataclass
class GridInfo:
    point: str
    theta: np.ndarray
    phi: np.ndarray
    theta_edges: np.ndarray
    phi_edges: np.ndarray
    L_grid: np.ndarray
    inside_mask: np.ndarray
    filled: np.ndarray
    phi0: float
    theta_col: str
    phi_col: str
    L_col: str


def load_ecrit_grid(ecrit_csv: Path, point: str, theta_min: float | None, theta_max: float | None,
                    phi_min: float | None, phi_max: float | None) -> GridInfo:
    df = pd.read_csv(ecrit_csv)
    th_col = find_col(df, ["theta_deg", "theta", "zenith_deg"])
    ph_col = find_col(df, ["phi_rel_deg", "phi_deg", "phi", "azimuth_deg"])
    L_col = find_col(df, ["length_inside_m", "L_m", "rock_length_m", "length_m", "longitud_m", "length"])

    df = df.copy()
    for c in (th_col, ph_col, L_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[th_col, ph_col]).copy()

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
    th_edges = centers_to_edges(theta, bin_width(theta))
    ph_edges = centers_to_edges(phi, bin_width(phi))

    L_grid = np.zeros((len(theta), len(phi)), dtype=float)
    filled = np.zeros_like(L_grid, dtype=bool)
    ti = {round(v, 10): i for i, v in enumerate(theta)}
    pj = {round(v, 10): j for j, v in enumerate(phi)}

    for row in df[[th_col, ph_col, L_col]].itertuples(index=False, name=None):
        th, ph, L = row
        i = ti.get(round(float(th), 10))
        j = pj.get(round(float(ph), 10))
        if i is None or j is None:
            continue
        L_grid[i, j] = float(L) if pd.notna(L) and np.isfinite(L) else 0.0
        filled[i, j] = True

    inside = filled & np.isfinite(L_grid) & (L_grid > 0.0)
    return GridInfo(
        point=point,
        theta=theta,
        phi=phi,
        theta_edges=th_edges,
        phi_edges=ph_edges,
        L_grid=L_grid,
        inside_mask=inside,
        filled=filled,
        phi0=phi0_for_point(point),
        theta_col=th_col,
        phi_col=ph_col,
        L_col=L_col,
    )


# -----------------------------------------------------------------------------
# 2D angular kernel cache and sampling
# -----------------------------------------------------------------------------
@dataclass
class Cached2DKernel:
    flat_indices: np.ndarray
    cdf: np.ndarray
    used_nearest_fallback: bool
    outside_domain: bool
    valid: bool


class Kernel2DCache:
    def __init__(
        self,
        model: EmpiricalKernelModel,
        grid: GridInfo,
        kernel_threshold: float,
        max_kernel_radius_mrad: float | None,
        energy_cache_dlog: float,
        max_cache_items: int,
    ) -> None:
        self.model = model
        self.grid = grid
        self.kernel_threshold = float(kernel_threshold)
        self.max_kernel_radius_mrad = max_kernel_radius_mrad
        self.energy_cache_dlog = float(energy_cache_dlog)
        self.max_cache_items = int(max_cache_items)
        self.cache: dict[tuple[int, int, int | float], Cached2DKernel] = {}
        self.TH, self.PH = np.meshgrid(grid.theta, grid.phi, indexing="ij")

    def energy_key_and_value(self, E: float) -> tuple[int | float, float]:
        if self.energy_cache_dlog > 0:
            key = int(round(math.log(max(E, 1e-12)) / self.energy_cache_dlog))
            E_use = math.exp(key * self.energy_cache_dlog)
            return key, E_use
        # Exact energy: useful for debugging, bad for speed.
        return round(float(E), 6), float(E)

    def get(self, i_src: int, j_src: int, E_GeV: float) -> Cached2DKernel:
        key_E, E_use = self.energy_key_and_value(E_GeV)
        key = (int(i_src), int(j_src), key_E)
        item = self.cache.get(key)
        if item is not None:
            return item

        if len(self.cache) >= self.max_cache_items:
            self.cache.clear()

        item = self._build(i_src, j_src, E_use)
        self.cache[key] = item
        return item

    def _build(self, i_src: int, j_src: int, E_GeV: float) -> Cached2DKernel:
        L = float(self.grid.L_grid[i_src, j_src])
        if (not np.isfinite(L)) or L <= 0.0 or (not np.isfinite(E_GeV)) or E_GeV <= 0.0:
            flat = np.array([i_src * len(self.grid.phi) + j_src], dtype=np.int64)
            return Cached2DKernel(flat, np.array([1.0]), False, False, False)

        pred = self.model.predict_kernel(L, E_GeV)
        if not pred.valid:
            flat = np.array([i_src * len(self.grid.phi) + j_src], dtype=np.int64)
            return Cached2DKernel(flat, np.array([1.0]), pred.used_nearest_fallback, pred.outside_domain, False)

        density = pred.probability_per_bin / self.model.widths_mrad
        density = np.asarray(density, dtype=float)
        density[~np.isfinite(density)] = 0.0
        density[density < 0.0] = 0.0

        positive = density > self.kernel_threshold
        if np.any(positive):
            radius = float(np.nanmax(np.abs(self.model.centers_mrad[positive])))
        else:
            radius = float(np.nanmax(np.abs(self.model.centers_mrad)))
        if self.max_kernel_radius_mrad is not None and self.max_kernel_radius_mrad > 0:
            radius = min(radius, float(self.max_kernel_radius_mrad))
        radius = max(radius, 1.0)

        th0 = float(self.grid.theta[i_src])
        ph0 = float(self.grid.phi[j_src])
        sin_th = abs(math.sin(math.radians(th0)))
        deg_radius_theta = math.degrees(radius / 1000.0)
        deg_radius_phi = math.degrees(radius / 1000.0) / max(sin_th, 1e-3)

        tmask = (self.grid.theta >= th0 - deg_radius_theta) & (self.grid.theta <= th0 + deg_radius_theta)
        pmask = (self.grid.phi >= ph0 - deg_radius_phi) & (self.grid.phi <= ph0 + deg_radius_phi)
        if not np.any(tmask) or not np.any(pmask):
            flat = np.array([i_src * len(self.grid.phi) + j_src], dtype=np.int64)
            return Cached2DKernel(flat, np.array([1.0]), pred.used_nearest_fallback, pred.outside_domain, False)

        ii = np.where(tmask)[0]
        jj = np.where(pmask)[0]
        TH = self.TH[np.ix_(ii, jj)]
        PH = self.PH[np.ix_(ii, jj)]
        valid = self.grid.filled[np.ix_(ii, jj)]

        dth_mrad = (TH - th0) * math.pi / 180.0 * 1000.0
        dph_mrad = (PH - ph0) * math.pi / 180.0 * 1000.0
        dph_eff_mrad = sin_th * dph_mrad

        Kth = np.interp(dth_mrad.ravel(), self.model.centers_mrad, density, left=0.0, right=0.0).reshape(dth_mrad.shape)
        Kph = np.interp(dph_eff_mrad.ravel(), self.model.centers_mrad, density, left=0.0, right=0.0).reshape(dph_eff_mrad.shape)
        W = Kth * Kph
        W[~valid] = 0.0
        W[W < 0.0] = 0.0
        s = float(np.sum(W))
        if s <= 0.0 or not np.isfinite(s):
            flat = np.array([i_src * len(self.grid.phi) + j_src], dtype=np.int64)
            return Cached2DKernel(flat, np.array([1.0]), pred.used_nearest_fallback, pred.outside_domain, False)
        W /= s

        nonzero = W.ravel() > 0.0
        # Map local flattened indices to global flattened indices.
        II, JJ = np.meshgrid(ii, jj, indexing="ij")
        global_flat = (II.ravel()[nonzero] * len(self.grid.phi) + JJ.ravel()[nonzero]).astype(np.int64)
        probs = W.ravel()[nonzero].astype(float)
        probs /= probs.sum()
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        return Cached2DKernel(global_flat, cdf, pred.used_nearest_fallback, pred.outside_domain, True)


# -----------------------------------------------------------------------------
# SHW processing
# -----------------------------------------------------------------------------
@dataclass
class ProcessResult:
    point: str
    summary: dict


def process_point(payload: dict) -> ProcessResult:
    point = payload["point"]
    shw_path = Path(payload["shw_path"]) if payload.get("shw_path") else None
    event_cache_path = Path(payload["event_cache_path"]) if payload.get("event_cache_path") else None
    ecrit_csv = Path(payload["ecrit_csv"])
    outdir = Path(payload["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(payload["random_seed"]))
    model = EmpiricalKernelModel(Path(payload["kernel_library"]), payload["interp_method"], float(payload["rbf_smoothing"]))
    grid = load_ecrit_grid(
        ecrit_csv,
        point,
        payload["theta_min"], payload["theta_max"], payload["phi_min"], payload["phi_max"],
    )
    cache = Kernel2DCache(
        model=model,
        grid=grid,
        kernel_threshold=float(payload["kernel_threshold"]),
        max_kernel_radius_mrad=payload["max_kernel_radius_mrad"],
        energy_cache_dlog=float(payload["energy_cache_dlog"]),
        max_cache_items=int(payload["max_cache_items"]),
    )

    H_in = np.zeros_like(grid.L_grid, dtype=np.int64)
    H_out = np.zeros_like(grid.L_grid, dtype=np.int64)

    n_lines = 0
    n_particles = 0
    n_muons = 0
    n_in_grid_total = 0
    n_in_grid = 0
    n_skipped_source_outside_geometry = 0
    n_identity = 0
    n_smeared = 0
    n_nearest = 0
    n_outside = 0
    n_no_support = 0
    n_energy_nonpositive = 0

    def handle_event(theta: float, phi_rel: float, T: float) -> bool:
        nonlocal n_in_grid_total, n_in_grid, n_skipped_source_outside_geometry
        nonlocal n_identity, n_smeared, n_nearest, n_outside, n_no_support, n_energy_nonpositive

        i = int(np.searchsorted(grid.theta_edges, theta, side="right") - 1)
        if i < 0 or i >= len(grid.theta):
            return False
        j = int(np.searchsorted(grid.phi_edges, phi_rel, side="right") - 1)
        if j < 0 or j >= len(grid.phi):
            return False
        if not grid.filled[i, j]:
            return False

        n_in_grid_total += 1
        if payload["source_mode"] == "inside" and not grid.inside_mask[i, j]:
            n_skipped_source_outside_geometry += 1
            return False

        n_in_grid += 1
        H_in[i, j] += 1

        if T <= 0.0 or not math.isfinite(T):
            H_out[i, j] += 1
            n_identity += 1
            n_energy_nonpositive += 1
            return True

        if grid.L_grid[i, j] <= 0.0:
            H_out[i, j] += 1
            n_identity += 1
            return True

        k2d = cache.get(i, j, T)
        if k2d.used_nearest_fallback:
            n_nearest += 1
        if k2d.outside_domain:
            n_outside += 1
        if not k2d.valid:
            n_no_support += 1

        if k2d.flat_indices.size == 1:
            H_out.ravel()[k2d.flat_indices[0]] += 1
            n_identity += 1
            return True

        u = rng.random()
        k = int(np.searchsorted(k2d.cdf, u, side="right"))
        if k >= len(k2d.flat_indices):
            k = len(k2d.flat_indices) - 1
        H_out.ravel()[k2d.flat_indices[k]] += 1
        n_smeared += 1
        return True

    if event_cache_path is not None:
        if not event_cache_path.exists():
            raise FileNotFoundError(event_cache_path)
        with np.load(event_cache_path) as data:
            theta_arr = np.asarray(data["theta_deg"], dtype=float)
            phi_arr = np.asarray(data["phi_rel_deg"], dtype=float)
            if "kinetic_GeV" in data:
                kinetic_arr = np.asarray(data["kinetic_GeV"], dtype=float)
            else:
                kinetic_arr = np.asarray(data["total_E_GeV"], dtype=float) - MUON_MASS_GEV
        total = int(theta_arr.size)
        pbar = tqdm(total=total, unit="event", desc=f"event-MC cache {point}", disable=bool(payload["no_progress"]))
        for theta, phi_rel, T in zip(theta_arr, phi_arr, kinetic_arr):
            n_lines += 1
            n_particles += 1
            n_muons += 1
            handle_event(float(theta), float(phi_rel), float(T))
            pbar.update(1)
            if payload["head"] and n_in_grid >= int(payload["head"]):
                break
        pbar.close()
    else:
        if shw_path is None or not shw_path.exists():
            raise FileNotFoundError(shw_path)
        total_bytes = stream_size_hint(shw_path)
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"event-MC {point}", disable=bool(payload["no_progress"]))

        with open_shw_bytes(shw_path, member_name=payload.get("shw_member")) as f:
            for raw in f:
                n_lines += 1
                pbar.update(len(raw))
                s = raw.strip()
                if not s or s.startswith(b"#"):
                    continue
                parts = s.split()
                rec = parse_muon_parts(parts, shw_format=payload["shw_format"], only_muons=payload["only_muons"])
                if rec is None:
                    continue
                n_particles += 1
                if rec.pid in MUON_IDS_B:
                    n_muons += 1
                if payload["discard_upgoing"] and rec.pz > 0.0:
                    continue
                angles = theta_phi_from_momentum(rec.px, rec.py, rec.pz)
                if angles is None:
                    continue
                theta, phi_abs = angles
                phi_rel = (phi_abs - grid.phi0) % 360.0
                if payload["wrap180"] and phi_rel > 180.0:
                    phi_rel -= 360.0
                T = rec.e_total_GeV - MUON_MASS_GEV
                handle_event(theta, phi_rel, T)

                if payload["head"] and n_in_grid >= int(payload["head"]):
                    break

        pbar.close()

    point_dir = outdir / point
    point_dir.mkdir(parents=True, exist_ok=True)

    output_paths = save_tables_and_plots(point, grid, H_in, H_out, point_dir, payload)

    total_in = int(H_in.sum())
    total_out = int(H_out.sum())
    rel_change = (total_out - total_in) / total_in if total_in else np.nan
    rel = safe_rel_delta(H_out.astype(float), H_in.astype(float))
    finite_rel = rel[np.isfinite(rel)]
    inside = grid.inside_mask
    input_inside = int(H_in[inside].sum())
    input_outside = int(H_in[~inside].sum())
    smeared_inside = int(H_out[inside].sum())
    smeared_outside = int(H_out[~inside].sum())
    prefix = output_prefix_for_source_mode(payload["source_mode"])

    summary = {
        "point": point,
        "shw_path": str(shw_path) if shw_path is not None else "",
        "event_cache_path": str(event_cache_path) if event_cache_path is not None else "",
        "ecrit_csv": str(ecrit_csv),
        "kernel_library": str(payload["kernel_library"]),
        "interp_method": payload["interp_method"],
        "kernel_family": model.kernel_family,
        "kernel_tail_policy": model.tail_aware.policy_description if model.tail_aware is not None else "legacy",
        "kernel_support_mrad": f"{model.edges_mrad[0]:g},{model.edges_mrad[-1]:g}",
        "source_mode": payload["source_mode"],
        "random_seed": int(payload["random_seed"]),
        "energy_cache_dlog": float(payload["energy_cache_dlog"]),
        "n_lines_read": int(n_lines),
        "n_particles_read": int(n_particles),
        "n_muons_read": int(n_muons),
        "n_events_in_grid_before_source_cut": int(n_in_grid_total),
        "n_skipped_source_outside_geometry": int(n_skipped_source_outside_geometry),
        "n_events_in_grid": int(n_in_grid),
        "input_total": total_in,
        "smeared_total": total_out,
        "input_inside_geometry": input_inside,
        "input_outside_geometry": input_outside,
        "smeared_inside_geometry": smeared_inside,
        "smeared_outside_geometry": smeared_outside,
        "inside_geometry_net_change": smeared_inside - input_inside,
        "outside_geometry_net_change": smeared_outside - input_outside,
        "relative_total_change": rel_change,
        "n_sources_identity": int(n_identity),
        "n_sources_smeared": int(n_smeared),
        "n_sources_nearest_fallback": int(n_nearest),
        "n_sources_outside_domain": int(n_outside),
        "n_sources_no_support": int(n_no_support),
        "n_energy_nonpositive": int(n_energy_nonpositive),
        "kernel_cache_items_final": int(len(cache.cache)),
        "p90_abs_relative_delta": float(np.nanpercentile(np.abs(finite_rel), 90)) if finite_rel.size else np.nan,
        "p99_abs_relative_delta": float(np.nanpercentile(np.abs(finite_rel), 99)) if finite_rel.size else np.nan,
        "theta_min": float(grid.theta.min()),
        "theta_max": float(grid.theta.max()),
        "phi_min": float(grid.phi.min()),
        "phi_max": float(grid.phi.max()),
        "output_table": str(point_dir / f"{prefix}_smearing_table_{point}.csv"),
        "inside_output_table": str(point_dir / f"{prefix}_retained_inside_table_{point}.csv"),
        "comparison_png": str(point_dir / f"{prefix}_smearing_comparison_{point}.png"),
        "inside_comparison_png": str(point_dir / f"{prefix}_retained_inside_comparison_{point}.png"),
    }
    summary.update(output_paths)
    pd.DataFrame([summary]).to_csv(point_dir / f"event_mc_summary_{point}.csv", index=False)
    return ProcessResult(point=point, summary=summary)


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def output_table(grid: GridInfo, H_in: np.ndarray, H_out: np.ndarray, inside_only: bool = False) -> pd.DataFrame:
    TH, PH = np.meshgrid(grid.theta, grid.phi, indexing="ij")
    inp = H_in.astype(float)
    out = H_out.astype(float)
    if inside_only:
        mask = grid.inside_mask
        inp = np.where(mask, inp, 0.0)
        out = np.where(mask, out, 0.0)
    delta = out - inp
    rel = safe_rel_delta(out, inp)
    return pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "inside_volcano_geometry": grid.inside_mask.ravel().astype(int),
        "length_inside_m": grid.L_grid.ravel(),
        "input_count": inp.ravel(),
        "smeared_count": out.ravel(),
        "delta_smeared_minus_input": delta.ravel(),
        "relative_delta": rel.ravel(),
    })


def step_tag(step: float) -> str:
    return f"bin{step:.2f}deg".replace(".", "p").replace("-", "m")


def edges_for_step(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("--display-step must be positive")
    n = max(1, int(math.ceil((stop - start) / step)))
    edges = start + np.arange(n + 1, dtype=float) * step
    edges[-1] = stop
    return edges


def rebin_to_step(grid: GridInfo, H_in: np.ndarray, H_out: np.ndarray, step: float) -> tuple[GridInfo, np.ndarray, np.ndarray]:
    """Aggregate native MC cells to a coarser angular bin."""
    theta_edges = edges_for_step(float(grid.theta_edges[0]), float(grid.theta_edges[-1]), step)
    phi_edges = edges_for_step(float(grid.phi_edges[0]), float(grid.phi_edges[-1]), step)
    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    ti = np.searchsorted(theta_edges, grid.theta, side="right") - 1
    pj = np.searchsorted(phi_edges, grid.phi, side="right") - 1
    valid_t = (ti >= 0) & (ti < len(theta))
    valid_p = (pj >= 0) & (pj < len(phi))

    out_shape = (len(theta), len(phi))
    rebinned_in = np.zeros(out_shape, dtype=float)
    rebinned_out = np.zeros(out_shape, dtype=float)
    L_sum = np.zeros(out_shape, dtype=float)
    L_count = np.zeros(out_shape, dtype=float)
    inside_count = np.zeros(out_shape, dtype=np.int64)
    filled_count = np.zeros(out_shape, dtype=np.int64)

    for i_old, i_new in enumerate(ti):
        if not valid_t[i_old]:
            continue
        for j_old, j_new in enumerate(pj):
            if not valid_p[j_old]:
                continue
            rebinned_in[i_new, j_new] += float(H_in[i_old, j_old])
            rebinned_out[i_new, j_new] += float(H_out[i_old, j_old])
            if grid.filled[i_old, j_old]:
                filled_count[i_new, j_new] += 1
                L = float(grid.L_grid[i_old, j_old])
                if np.isfinite(L):
                    L_sum[i_new, j_new] += L
                    L_count[i_new, j_new] += 1.0
            if grid.inside_mask[i_old, j_old]:
                inside_count[i_new, j_new] += 1

    L_grid = np.divide(L_sum, L_count, out=np.zeros_like(L_sum), where=L_count > 0)
    coarse_grid = GridInfo(
        point=grid.point,
        theta=theta,
        phi=phi,
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        L_grid=L_grid,
        inside_mask=inside_count > 0,
        filled=filled_count > 0,
        phi0=grid.phi0,
        theta_col=grid.theta_col,
        phi_col=grid.phi_col,
        L_col=grid.L_col,
    )
    return coarse_grid, rebinned_in, rebinned_out


def prepare_plot(Z: np.ndarray, blank_zeros: bool = True) -> np.ndarray:
    Zp = np.asarray(Z, dtype=float).copy()
    Zp[~np.isfinite(Zp)] = np.nan
    if blank_zeros:
        Zp[Zp <= 0] = np.nan
    return Zp


def format_axes(ax, grid: GridInfo) -> None:
    ax.set_xlim(float(grid.phi_edges[0]), float(grid.phi_edges[-1]))
    ax.set_ylim(float(grid.theta_edges[-1]), float(grid.theta_edges[0]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")


def plot_comparison(grid: GridInfo, H_in: np.ndarray, H_out: np.ndarray, out_png: Path, title: str,
                    inside_only: bool, blank_zeros: bool, vmax_percentile: float, rel_vmax_percentile: float) -> None:
    inp = H_in.astype(float)
    out = H_out.astype(float)
    if inside_only:
        inp = np.where(grid.inside_mask, inp, 0.0)
        out = np.where(grid.inside_mask, out, 0.0)
    rel = safe_rel_delta(out, inp)

    common = np.concatenate([inp.ravel(), out.ravel()])
    common = common[np.isfinite(common) & (common > 0)]
    vmax = float(np.nanpercentile(common, vmax_percentile)) if common.size else None

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
    panels = [
        (inp, "Input map", "Counts", False),
        (out, "After event-by-event MC", "Counts", False),
        (rel, "Relative change", r"$(N_{MC}-N_{in})/N_{in}$", True),
    ]
    for ax, (Z, ttl, label, diverging) in zip(axes, panels):
        kwargs = {"shading": "flat"}
        Zp = Z
        if diverging:
            vals = Zp[np.isfinite(Zp)]
            if vals.size:
                rv = float(np.nanpercentile(np.abs(vals), rel_vmax_percentile))
                if np.isfinite(rv) and rv > 0:
                    kwargs["norm"] = TwoSlopeNorm(vmin=-rv, vcenter=0.0, vmax=rv)
            kwargs["cmap"] = "coolwarm"
        else:
            Zp = prepare_plot(Zp, blank_zeros=blank_zeros)
            kwargs["cmap"] = "viridis"
            if vmax is not None and np.isfinite(vmax) and vmax > 0:
                kwargs["vmax"] = vmax
        im = ax.pcolormesh(grid.phi_edges, grid.theta_edges, Zp, **kwargs)
        format_axes(ax, grid)
        ax.set_title(ttl)
        cb = fig.colorbar(im, ax=ax, shrink=0.92)
        cb.set_label(label)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_tables_and_plots(point: str, grid: GridInfo, H_in: np.ndarray, H_out: np.ndarray,
                          point_dir: Path, payload: dict) -> dict[str, str]:
    prefix = output_prefix_for_source_mode(payload["source_mode"])
    if payload["source_mode"] == "inside":
        title_full = f"Event-by-event empirical MC from inside-volcano sources — {point}"
        title_retained = f"Retained inside-volcano counts after event-by-event empirical MC — {point}"
    else:
        title_full = f"Event-by-event empirical MC smearing — {point}"
        title_retained = f"Inside-volcano counts after event-by-event empirical MC — {point}"

    outputs: dict[str, str] = {}
    table_path = point_dir / f"{prefix}_smearing_table_{point}.csv"
    retained_table_path = point_dir / f"{prefix}_retained_inside_table_{point}.csv"
    comparison_path = point_dir / f"{prefix}_smearing_comparison_{point}.png"
    retained_comparison_path = point_dir / f"{prefix}_retained_inside_comparison_{point}.png"

    output_table(grid, H_in, H_out, inside_only=False).to_csv(table_path, index=False)
    output_table(grid, H_in, H_out, inside_only=True).to_csv(retained_table_path, index=False)
    plot_comparison(
        grid, H_in, H_out,
        comparison_path,
        title=title_full,
        inside_only=False,
        blank_zeros=bool(payload["blank_zeros"]),
        vmax_percentile=float(payload["vmax_percentile"]),
        rel_vmax_percentile=float(payload["relative_vmax_percentile"]),
    )
    plot_comparison(
        grid, H_in, H_out,
        retained_comparison_path,
        title=title_retained,
        inside_only=True,
        blank_zeros=bool(payload["blank_zeros"]),
        vmax_percentile=float(payload["vmax_percentile"]),
        rel_vmax_percentile=float(payload["relative_vmax_percentile"]),
    )
    outputs.update({
        "output_table": str(table_path),
        "inside_output_table": str(retained_table_path),
        "comparison_png": str(comparison_path),
        "inside_comparison_png": str(retained_comparison_path),
    })

    display_step = payload.get("display_step")
    if display_step is not None:
        display_step = float(display_step)
        native_step = max(bin_width(grid.theta), bin_width(grid.phi))
        if display_step > native_step + 1e-9:
            tag = step_tag(display_step)
            bgrid, bH_in, bH_out = rebin_to_step(grid, H_in, H_out, display_step)
            inside_in = np.where(grid.inside_mask, H_in, 0.0)
            inside_out = np.where(grid.inside_mask, H_out, 0.0)
            inside_grid, bH_in_inside, bH_out_inside = rebin_to_step(grid, inside_in, inside_out, display_step)

            binned_table = point_dir / f"{prefix}_smearing_binned_{tag}_table_{point}.csv"
            binned_png = point_dir / f"{prefix}_smearing_binned_{tag}_comparison_{point}.png"
            binned_inside_table = point_dir / f"{prefix}_retained_inside_binned_{tag}_table_{point}.csv"
            binned_inside_png = point_dir / f"{prefix}_retained_inside_binned_{tag}_comparison_{point}.png"

            output_table(bgrid, bH_in, bH_out, inside_only=False).to_csv(binned_table, index=False)
            output_table(inside_grid, bH_in_inside, bH_out_inside, inside_only=False).to_csv(
                binned_inside_table, index=False
            )
            plot_comparison(
                bgrid, bH_in, bH_out,
                binned_png,
                title=f"{title_full} ({display_step:g} deg bins)",
                inside_only=False,
                blank_zeros=bool(payload["blank_zeros"]),
                vmax_percentile=float(payload["vmax_percentile"]),
                rel_vmax_percentile=float(payload["relative_vmax_percentile"]),
            )
            plot_comparison(
                inside_grid, bH_in_inside, bH_out_inside,
                binned_inside_png,
                title=f"{title_retained} ({display_step:g} deg bins)",
                inside_only=False,
                blank_zeros=bool(payload["blank_zeros"]),
                vmax_percentile=float(payload["vmax_percentile"]),
                rel_vmax_percentile=float(payload["relative_vmax_percentile"]),
            )
            outputs.update({
                "binned_display_step_deg": f"{display_step:g}",
                "binned_output_table": str(binned_table),
                "binned_inside_output_table": str(binned_inside_table),
                "binned_comparison_png": str(binned_png),
                "binned_inside_comparison_png": str(binned_inside_png),
            })
    return outputs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Event-by-event empirical Geant4-kernel MC smearing for SHW muons.")
    ap.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS), choices=list(DEFAULT_POINTS))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shw", type=Path, default=None, help="One SHW file reused for all points, usually raw.")
    src.add_argument("--shw-template", default=None, help="Template per point, e.g. run/04_filtered/stem_filtered_{point}.shw")
    src.add_argument("--event-cache", type=Path, default=None, help="One cached events_*.npz file for a single-point run.")
    src.add_argument("--event-cache-template", default=None, help="Template per point, e.g. run/04_event_cache/events_{point}.npz")
    ap.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="auto",
                    help="Input layout. auto detects ARTI-style 12-col or CNF 9-col SHW lines.")
    ap.add_argument("--shw-member", default=None,
                    help="Optional member path inside a .tar/.tar.gz input. Default: first .shw-like file.")
    ap.add_argument("--ecrit-template", required=True, help="Template: run/03_ecrit/ecrit_table_{point}.csv")
    ap.add_argument(
        "--kernel-library",
        type=Path,
        default=Path(__file__).resolve().parent / "hybrid_empirical_kernel_library.npz",
        help="Empirical kernel library (default: bundled hybrid full-tail model).",
    )
    ap.add_argument("--outdir", type=Path, default=Path("09_event_mc_empirical"))

    ap.add_argument("--interp-method", default="tail-aware", choices=["tail-aware", "linear", "rbf_linear", "nearest"])
    ap.add_argument("--rbf-smoothing", type=float, default=0.0)
    ap.add_argument("--energy-cache-dlog", type=float, default=0.05,
                    help="Log-energy bin size for cache. 0.05 ≈ 5%%. Use 0 for exact energy, slower.")
    ap.add_argument("--max-cache-items", type=int, default=20000)
    ap.add_argument("--kernel-threshold", type=float, default=0.0,
                    help="Density threshold used only to estimate local support radius.")
    ap.add_argument("--max-kernel-radius-mrad", type=float, default=None,
                    help="Optional hard radius in mrad to reduce cost. Default: use library support.")

    ap.add_argument("--theta-min", type=float, default=None)
    ap.add_argument("--theta-max", type=float, default=90.0)
    ap.add_argument("--phi-min", type=float, default=-60.0)
    ap.add_argument("--phi-max", type=float, default=60.0)
    ap.add_argument("--display-step", type=float, default=None,
                    help="Optional coarser angular bin in degrees for detector-resolution muograms.")
    ap.add_argument("--wrap180", dest="wrap180", action="store_true", default=True)
    ap.add_argument("--no-wrap180", dest="wrap180", action="store_false")
    ap.add_argument("--source-mode", choices=["all", "inside"], default="all",
                    help="all: smear all events in the angular grid. inside: use only events whose source cell is inside the volcano geometry, then allow them to migrate to the full angular grid.")
    ap.add_argument("--source-inside-only", dest="source_mode", action="store_const", const="inside",
                    help="Alias for --source-mode inside.")
    ap.add_argument("--discard-upgoing", action="store_true")
    ap.add_argument("--include-all", dest="only_muons", action="store_false", help="Include all particles; default only muons.")
    ap.set_defaults(only_muons=True)
    ap.add_argument("--head", type=int, default=0, help="Debug: stop after N events in grid per point.")

    ap.add_argument("--workers", type=int, default=1, help="Parallel workers over points.")
    ap.add_argument("--random-seed", type=int, default=12345)
    ap.add_argument("--no-progress", action="store_true")

    ap.add_argument("--blank-zeros", dest="blank_zeros", action="store_true", default=True)
    ap.add_argument("--show-zeros", dest="blank_zeros", action="store_false")
    ap.add_argument("--vmax-percentile", type=float, default=99.0)
    ap.add_argument("--relative-vmax-percentile", type=float, default=98.0)
    return ap


def main(argv=None) -> int:
    setup_style()
    args = parser().parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    payloads = []
    for k, point in enumerate(args.points):
        if args.event_cache_template:
            shw_path = None
            event_cache_path = Path(args.event_cache_template.format(point=point))
        elif args.event_cache:
            shw_path = None
            event_cache_path = args.event_cache
        elif args.shw_template:
            shw_path = Path(args.shw_template.format(point=point))
            event_cache_path = None
        else:
            shw_path = args.shw
            event_cache_path = None
        ecrit_csv = Path(args.ecrit_template.format(point=point))
        payloads.append({
            "point": point,
            "shw_path": str(shw_path) if shw_path is not None else "",
            "event_cache_path": str(event_cache_path) if event_cache_path is not None else "",
            "shw_format": args.shw_format,
            "shw_member": args.shw_member,
            "ecrit_csv": str(ecrit_csv),
            "kernel_library": str(args.kernel_library),
            "outdir": str(args.outdir),
            "interp_method": args.interp_method,
            "rbf_smoothing": args.rbf_smoothing,
            "energy_cache_dlog": args.energy_cache_dlog,
            "max_cache_items": args.max_cache_items,
            "kernel_threshold": args.kernel_threshold,
            "max_kernel_radius_mrad": args.max_kernel_radius_mrad,
            "theta_min": args.theta_min,
            "theta_max": args.theta_max,
            "phi_min": args.phi_min,
            "phi_max": args.phi_max,
            "display_step": args.display_step,
            "wrap180": args.wrap180,
            "source_mode": args.source_mode,
            "discard_upgoing": args.discard_upgoing,
            "only_muons": args.only_muons,
            "head": args.head,
            "random_seed": args.random_seed + 1009 * k,
            "no_progress": args.no_progress,
            "blank_zeros": args.blank_zeros,
            "vmax_percentile": args.vmax_percentile,
            "relative_vmax_percentile": args.relative_vmax_percentile,
        })

    results: list[ProcessResult] = []
    workers = max(1, int(args.workers))
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_to_point = {ex.submit(process_point, p): p["point"] for p in payloads}
            for fut in as_completed(fut_to_point):
                results.append(fut.result())
    else:
        for p in payloads:
            results.append(process_point(p))

    order = {p: i for i, p in enumerate(args.points)}
    results.sort(key=lambda r: order[r.point])
    summary = pd.DataFrame([r.summary for r in results])
    summary.to_csv(args.outdir / "event_mc_smearing_summary.csv", index=False)

    print("\n[OK] Event-by-event empirical MC finished")
    print(f"Summary: {args.outdir / 'event_mc_smearing_summary.csv'}")
    for r in results:
        s = r.summary
        print(
            f"  {r.point}: mode={s['source_mode']} in={s['input_total']} out={s['smeared_total']} "
            f"inside={s['input_inside_geometry']}->{s['smeared_inside_geometry']} "
            f"outside={s['input_outside_geometry']}->{s['smeared_outside_geometry']} "
            f"rel={s['relative_total_change']:.3e} smeared={s['n_sources_smeared']} "
            f"nearest={s['n_sources_nearest_fallback']} outside_domain={s['n_sources_outside_domain']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
