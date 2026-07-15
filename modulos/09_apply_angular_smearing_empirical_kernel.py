#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_apply_angular_smearing_empirical_kernel.py
--------------------------------------------
Apply the full empirical Geant4 angular migration kernel to theta-phi maps.

This is not a Gaussian sigma smearing. For each source angular bin, the script
queries K_G4(delta alpha | L, T_ref) and builds a separable 2D kernel in
(theta, phi_eff):

    W(target|source) ∝ K(delta theta) K(sin(theta_source) delta phi)

The deterministic mode conserves total counts by construction after per-source
renormalization over valid target bins.
"""
from __future__ import annotations

import argparse
import math
import re
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
    from scipy.spatial import cKDTree, Delaunay
except Exception as exc:  # pragma: no cover
    raise SystemExit("scipy is required for empirical-kernel interpolation") from exc

try:
    from empirical_kernel_io import TailAwareEmpiricalKernel, load_empirical_kernel_library
    from plot_style import apply_scientific_style
except ModuleNotFoundError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from empirical_kernel_io import TailAwareEmpiricalKernel, load_empirical_kernel_library
    from plot_style import apply_scientific_style

MUON_MASS_GEV = 0.10565837
DEFAULT_POINTS = ("P1", "P2", "P4", "P5")
VALUE_CANDIDATES = (
    "count", "counts", "count_inside_geometry", "count_all_in_grid",
    "dN_dOmega_count_per_sr", "dN_dOmega_inside_count_per_sr",
    "flux", "intensity", "N_abs", "N",
)


def setup_style() -> None:
    apply_scientific_style()


def factor_tag(factor: float) -> str:
    return f"f{factor:.2f}".replace(".", "p").replace("-", "m")


def infer_tag_from_path(path: Path) -> str:
    m = re.search(r"_(f\d+p\d+|fm\d+p\d+)\.csv$", path.name)
    return m.group(1) if m else path.stem


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
        raise KeyError(f"No encontré columnas {list(candidates)}. Disponibles: {list(df.columns)}")
    return None


def centers_to_edges(centers: np.ndarray, fallback_step: float = 1.0) -> np.ndarray:
    c = np.asarray(centers, dtype=float)
    c = np.array(sorted(np.unique(c[np.isfinite(c)])), dtype=float)
    if c.size == 0:
        raise ValueError("No hay centros válidos.")
    if c.size == 1:
        return np.array([c[0] - 0.5 * fallback_step, c[0] + 0.5 * fallback_step])
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


def normalize_probability(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=float).copy()
    p[~np.isfinite(p)] = 0.0
    p[p < 0.0] = 0.0
    s = float(np.sum(p))
    if s <= 0.0 or not np.isfinite(s):
        return np.zeros_like(p)
    return p / s


class EmpiricalKernelModel:
    def __init__(self, npz_path: Path, interp_method: str = "tail-aware"):
        self.path = Path(npz_path)
        self.interp_method = interp_method
        self.tail_aware = None
        if interp_method == "tail-aware":
            self.tail_aware = TailAwareEmpiricalKernel(self.path)
            self.kernel_family = self.tail_aware.kernel_family
            self.centers_mrad = self.tail_aware.centers_mrad
            self.edges_mrad = self.tail_aware.edges_mrad
            return
        lib = load_empirical_kernel_library(self.path)
        self.kernel_family = lib.family
        self.centers_mrad = lib.centers_mrad
        self.edges_mrad = lib.edges_mrad
        self.probabilities = lib.probabilities
        self.L_m = lib.L_m
        self.E_in_GeV = lib.E_in_GeV
        self.clean_for_kernel = lib.clean_for_kernel

        if self.probabilities.shape != (self.L_m.size, self.centers_mrad.size):
            raise ValueError(
                "Expected probabilities with shape (n_kernels, n_centers). "
                f"Got {self.probabilities.shape}, L={self.L_m.size}, centers={self.centers_mrad.size}."
            )

        valid = (
            self.clean_for_kernel &
            np.isfinite(self.L_m) & (self.L_m > 0.0) &
            np.isfinite(self.E_in_GeV) & (self.E_in_GeV > 0.0) &
            np.isfinite(self.probabilities).all(axis=1)
        )
        if not np.any(valid):
            raise ValueError("No valid empirical kernels found after clean_for_kernel filtering.")

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
                self.rbf_interp = RBFInterpolator(
                    self.points_uv,
                    self.values,
                    kernel="linear",
                    neighbors=min(40, self.points_uv.shape[0]),
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
        if self.tail_aware is not None:
            pred = self.tail_aware.predict_kernel(L_m, E_GeV)
            meta = {
                "used_nearest_fallback": pred.used_nearest_fallback,
                "outside_domain": pred.outside_domain,
                "valid": pred.valid,
                "interpolation_mode": pred.interpolation_mode,
                "tail_policy": pred.tail_policy,
            }
            return pred.centers_mrad, pred.probability_per_bin, meta
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


class DummyKernelModel:
    """Small self-contained kernel for --run-toy-test."""
    centers_mrad = np.array([-15.0, -7.5, 0.0, 7.5, 15.0])

    def predict_kernel(self, L_m: float, E_GeV: float):
        p = normalize_probability(np.array([0.08, 0.22, 0.40, 0.22, 0.08]))
        return self.centers_mrad, p, {"used_nearest_fallback": False, "outside_domain": False, "valid": True}


def read_map(csv_path: Path, value_col: str | None):
    df = pd.read_csv(csv_path)
    th_col = find_col(df, ["theta_deg", "theta", "zenith_deg"])
    ph_col = find_col(df, ["phi_rel_deg", "phi_deg", "phi", "azimuth_deg"])
    val_col = value_col if value_col else find_col(df, VALUE_CANDIDATES)
    if val_col not in df.columns:
        raise KeyError(f"La columna {val_col} no existe en {csv_path}")

    df = df.copy()
    for c in (th_col, ph_col, val_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[th_col, ph_col, val_col])

    theta = np.array(sorted(df[th_col].unique()), dtype=float)
    phi = np.array(sorted(df[ph_col].unique()), dtype=float)
    H = np.zeros((len(theta), len(phi)), dtype=float)
    filled = np.zeros_like(H, dtype=bool)
    ti = {round(v, 10): i for i, v in enumerate(theta)}
    pj = {round(v, 10): j for j, v in enumerate(phi)}

    for _, row in df.iterrows():
        i = ti.get(round(float(row[th_col]), 10))
        j = pj.get(round(float(row[ph_col]), 10))
        if i is not None and j is not None:
            H[i, j] = float(row[val_col])
            filled[i, j] = True

    return {
        "theta": theta,
        "phi": phi,
        "theta_edges": centers_to_edges(theta, bin_width(theta)),
        "phi_edges": centers_to_edges(phi, bin_width(phi)),
        "H": H,
        "filled": filled,
        "value_col": val_col,
    }


def cut_window(info: dict, theta_min, theta_max, phi_min, phi_max):
    th, ph = info["theta"], info["phi"]
    mt = np.ones(th.shape, dtype=bool)
    mp = np.ones(ph.shape, dtype=bool)
    if theta_min is not None:
        mt &= th >= theta_min
    if theta_max is not None:
        mt &= th <= theta_max
    if phi_min is not None:
        mp &= ph >= phi_min
    if phi_max is not None:
        mp &= ph <= phi_max
    if not np.any(mt) or not np.any(mp):
        raise RuntimeError("La ventana angular solicitada no contiene datos.")
    out = info.copy()
    out["theta"] = th[mt]
    out["phi"] = ph[mp]
    out["theta_edges"] = centers_to_edges(out["theta"], bin_width(out["theta"]))
    out["phi_edges"] = centers_to_edges(out["phi"], bin_width(out["phi"]))
    out["H"] = info["H"][np.ix_(mt, mp)]
    out["filled"] = info["filled"][np.ix_(mt, mp)]
    return out


def read_ecrit_arrays(ecrit_csv: Path, theta: np.ndarray, phi: np.ndarray, factor: float):
    df = pd.read_csv(ecrit_csv)
    th_col = find_col(df, ["theta_deg", "theta", "zenith_deg"])
    ph_col = find_col(df, ["phi_rel_deg", "phi_deg", "phi", "azimuth_deg"])
    L_col = find_col(df, ["length_inside_m", "L_m", "rock_length_m", "length_m", "longitud_m", "length"])
    T_col = find_col(df, ["Tcrit_GeV", "T_crit_GeV", "kinetic_crit_GeV"], required=False)
    Etot_col = find_col(df, ["Ecrit_total_GeV", "E_total_crit_GeV", "Ecrit_GeV", "E_total_GeV"], required=False)
    if T_col is None and Etot_col is None:
        raise KeyError(f"{ecrit_csv} needs Tcrit_GeV or Ecrit_total_GeV")

    df = df.copy()
    for c in [th_col, ph_col, L_col, T_col, Etot_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[th_col, ph_col, L_col])

    L = np.zeros((len(theta), len(phi)), dtype=float)
    E = np.zeros_like(L)
    has = np.zeros_like(L, dtype=bool)
    th_edges = centers_to_edges(theta, bin_width(theta))
    ph_edges = centers_to_edges(phi, bin_width(phi))

    for _, row in df.iterrows():
        i = np.searchsorted(th_edges, float(row[th_col]), side="right") - 1
        j = np.searchsorted(ph_edges, float(row[ph_col]), side="right") - 1
        if i < 0 or i >= L.shape[0] or j < 0 or j >= L.shape[1]:
            continue
        Li = float(row[L_col]) if pd.notna(row[L_col]) else 0.0
        if T_col is not None:
            Ei = factor * float(row[T_col]) if pd.notna(row[T_col]) else 0.0
        else:
            Etot = float(row[Etot_col]) if pd.notna(row[Etot_col]) else 0.0
            Ei = factor * max(Etot - MUON_MASS_GEV, 0.0)
        L[i, j] = Li
        E[i, j] = Ei
        has[i, j] = True
    return L, E, has


def kernel_values_on_grid(centers_mrad: np.ndarray, probability: np.ndarray, x_mrad: np.ndarray,
                          threshold: float | None):
    p = normalize_probability(probability)
    y = np.interp(x_mrad, centers_mrad, p, left=0.0, right=0.0)
    y[~np.isfinite(y)] = 0.0
    if threshold is not None and threshold > 0:
        y[y < threshold] = 0.0
    return y


def maybe_round_nonnegative(value: float) -> int:
    return int(max(0, round(float(value))))


def smear_empirical(H: np.ndarray, theta: np.ndarray, phi: np.ndarray, filled: np.ndarray,
                    L_grid: np.ndarray, E_grid: np.ndarray, model,
                    stochastic: bool = False, rng=None,
                    kernel_threshold: float | None = None,
                    max_kernel_radius_mrad: float | None = None):
    H = np.asarray(H, dtype=float)
    out = np.zeros_like(H, dtype=float)

    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    n_sources = n_identity = n_smeared = 0
    n_nearest = n_outside = 0

    default_radius = float(np.nanmax(np.abs(getattr(model, "centers_mrad", np.array([60.0])))))
    if max_kernel_radius_mrad is not None and max_kernel_radius_mrad > 0:
        default_radius = min(default_radius, float(max_kernel_radius_mrad))

    rad_to_mrad = 1000.0 * math.pi / 180.0

    for i in range(len(theta)):
        for j in range(len(phi)):
            val = H[i, j]
            if not filled[i, j] or not np.isfinite(val) or val == 0.0:
                continue
            n_sources += 1

            L = float(L_grid[i, j]) if np.isfinite(L_grid[i, j]) else 0.0
            E = float(E_grid[i, j]) if np.isfinite(E_grid[i, j]) else 0.0
            if L <= 0.0 or E <= 0.0:
                out[i, j] += val
                n_identity += 1
                continue

            centers, prob, meta = model.predict_kernel(L, E)
            if not meta.get("valid", False):
                out[i, j] += val
                n_identity += 1
                continue

            if meta.get("used_nearest_fallback", False):
                n_nearest += 1
            if meta.get("outside_domain", False):
                n_outside += 1

            theta0 = float(theta[i])
            phi0 = float(phi[j])
            sin_th = abs(math.sin(math.radians(theta0)))

            theta_radius_deg = default_radius / rad_to_mrad
            if sin_th > 1e-4:
                phi_radius_deg = default_radius / (rad_to_mrad * sin_th)
            else:
                phi_radius_deg = np.inf

            mt = np.abs(theta - theta0) <= theta_radius_deg + 1e-12
            if np.isfinite(phi_radius_deg):
                mp = np.abs(phi - phi0) <= phi_radius_deg + 1e-12
            else:
                mp = np.ones_like(phi, dtype=bool)

            if not np.any(mt) or not np.any(mp):
                out[i, j] += val
                n_identity += 1
                continue

            theta_sub = theta[mt]
            phi_sub = phi[mp]
            filled_sub = filled[np.ix_(mt, mp)]
            dtheta_mrad = (theta_sub[:, None] - theta0) * rad_to_mrad
            dphi_mrad = (phi_sub[None, :] - phi0) * rad_to_mrad
            dphi_eff_mrad = sin_th * dphi_mrad

            ktheta = kernel_values_on_grid(centers, prob, dtheta_mrad, kernel_threshold)
            kphi = kernel_values_on_grid(centers, prob, dphi_eff_mrad, kernel_threshold)
            W = ktheta * kphi
            W[~filled_sub] = 0.0

            s = float(np.nansum(W))
            if s <= 0.0 or not np.isfinite(s):
                out[i, j] += val
                n_identity += 1
                continue
            W /= s

            target = out[np.ix_(mt, mp)]
            if stochastic:
                n = maybe_round_nonnegative(val)
                if n > 0:
                    sampled = rng.multinomial(n=n, pvals=W.ravel()).reshape(W.shape)
                    target += sampled
            else:
                target += val * W
            out[np.ix_(mt, mp)] = target
            n_smeared += 1

    return out, {
        "n_sources_nonzero": n_sources,
        "n_sources_identity": n_identity,
        "n_sources_smeared": n_smeared,
        "n_sources_nearest_fallback": n_nearest,
        "n_sources_outside_domain": n_outside,
    }


def output_table(theta, phi, H_in, H_out, L_grid, E_grid):
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    delta = H_out - H_in
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = delta / H_in
        rel[~np.isfinite(rel)] = np.nan
    return pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "input_value": H_in.ravel(),
        "smeared_value": H_out.ravel(),
        "delta_smeared_minus_input": delta.ravel(),
        "relative_delta": rel.ravel(),
        "length_inside_m": L_grid.ravel(),
        "Eref_kinetic_GeV": E_grid.ravel(),
    })


def display_canvas(theta, phi, Z, theta_min, theta_max, phi_min, phi_max, square, step):
    if not square:
        return centers_to_edges(theta, bin_width(theta)), centers_to_edges(phi, bin_width(phi)), Z

    if step is None:
        step = min(bin_width(theta), bin_width(phi))

    th_edges_src = centers_to_edges(theta, bin_width(theta))
    ph_edges_src = centers_to_edges(phi, bin_width(phi))
    th_edges = np.arange(theta_min, theta_max + step, step)
    ph_edges = np.arange(phi_min, phi_max + step, step)
    th_c = 0.5 * (th_edges[:-1] + th_edges[1:])
    ph_c = 0.5 * (ph_edges[:-1] + ph_edges[1:])
    Z2 = np.full((len(th_c), len(ph_c)), np.nan, dtype=float)

    for i, th in enumerate(th_c):
        ii = np.searchsorted(th_edges_src, th, side="right") - 1
        if ii < 0 or ii >= Z.shape[0]:
            continue
        for j, ph in enumerate(ph_c):
            jj = np.searchsorted(ph_edges_src, ph, side="right") - 1
            if 0 <= jj < Z.shape[1]:
                Z2[i, j] = Z[ii, jj]
    return th_edges, ph_edges, Z2


def prepare_plot_array(Z, blank_zeros=True):
    Zp = np.asarray(Z, dtype=float).copy()
    Zp[~np.isfinite(Zp)] = np.nan
    if blank_zeros:
        Zp[Zp <= 0] = np.nan
    return Zp


def apply_axes_format(ax, theta_min, theta_max, phi_min, phi_max):
    ax.set_xlim(phi_min, phi_max)
    ax.set_ylim(theta_max, theta_min)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
    ax.set_xticks(np.arange(np.ceil(phi_min / 20) * 20, phi_max + 1, 20))
    ax.set_yticks(np.arange(np.ceil(theta_min / 10) * 10, theta_max + 1, 10))


def plot_comparison(theta, phi, H_in, H_out, out_png: Path, point: str, tag: str,
                    value_label: str, theta_min, theta_max, phi_min, phi_max,
                    square, step, blank_zeros=True, vmax_percentile=99.0,
                    rel_vmax_percentile=98.0, title_prefix="Empirical angular smearing"):
    delta = H_out - H_in
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = delta / H_in
        rel[~np.isfinite(rel)] = np.nan

    panels = [
        (H_in, "Input map", value_label, False),
        (H_out, "After empirical kernel smearing", value_label, False),
        (rel, "Relative change", r"$(N_{smear}-N_{in})/N_{in}$", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), constrained_layout=True)
    common = np.concatenate([H_in.ravel(), H_out.ravel()])
    common = common[np.isfinite(common) & (common > 0)]
    common_vmax = np.nanpercentile(common, vmax_percentile) if common.size else None

    for ax, (Z, title, label, div) in zip(axes, panels):
        th_edges, ph_edges, Zp = display_canvas(theta, phi, Z, theta_min, theta_max, phi_min, phi_max, square, step)
        kwargs = {"shading": "flat"}
        if div:
            vals = Zp[np.isfinite(Zp)]
            if vals.size:
                vmax = np.nanpercentile(np.abs(vals), rel_vmax_percentile)
                if np.isfinite(vmax) and vmax > 0:
                    kwargs["norm"] = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            kwargs["cmap"] = "coolwarm"
        else:
            Zp = prepare_plot_array(Zp, blank_zeros=blank_zeros)
            kwargs["cmap"] = "viridis"
            if common_vmax is not None and np.isfinite(common_vmax) and common_vmax > 0:
                kwargs["vmax"] = common_vmax

        im = ax.pcolormesh(ph_edges, th_edges, Zp, **kwargs)
        apply_axes_format(ax, theta_min, theta_max, phi_min, phi_max)
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, shrink=0.92)
        cb.set_label(label)

    fig.suptitle(f"{title_prefix} — {point} — {tag}", fontsize=12)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def process_dataset(map_csv: Path, ecrit_csv: Path, point: str, outdir: Path, tag: str,
                    energy_factor: float, args, dataset_name: str, title_prefix: str,
                    prefix_stem: str, model, rng):
    raw = read_map(map_csv, args.value_col)
    tmin = np.nanmin(raw["theta"]) if args.theta_min is None else args.theta_min
    tmax = np.nanmax(raw["theta"]) if args.theta_max is None else args.theta_max
    pmin = np.nanmin(raw["phi"]) if args.phi_min is None else args.phi_min
    pmax = np.nanmax(raw["phi"]) if args.phi_max is None else args.phi_max
    info = cut_window(raw, tmin, tmax, pmin, pmax)

    L_grid, E_grid, has_ecrit = read_ecrit_arrays(ecrit_csv, info["theta"], info["phi"], energy_factor)
    # Missing Ecrit rows are treated as clear/identity by zero L and zero E.
    L_grid[~has_ecrit] = 0.0
    E_grid[~has_ecrit] = 0.0

    H_in = info["H"]
    H_out, stats = smear_empirical(
        H=H_in,
        theta=info["theta"],
        phi=info["phi"],
        filled=info["filled"],
        L_grid=L_grid,
        E_grid=E_grid,
        model=model,
        stochastic=args.stochastic,
        rng=rng,
        kernel_threshold=args.kernel_threshold,
        max_kernel_radius_mrad=args.max_kernel_radius_mrad,
    )

    point_dir = outdir / point
    point_dir.mkdir(parents=True, exist_ok=True)
    out_csv = point_dir / f"{prefix_stem}_table_{point}_{tag}.csv"
    output_table(info["theta"], info["phi"], H_in, H_out, L_grid, E_grid).to_csv(out_csv, index=False)

    comparison_png = point_dir / f"{prefix_stem}_comparison_{point}_{tag}.png"
    plot_comparison(
        info["theta"], info["phi"], H_in, H_out, comparison_png,
        point, tag, info["value_col"], tmin, tmax, pmin, pmax,
        args.square_display, args.display_step, args.blank_zeros,
        args.vmax_percentile, args.relative_vmax_percentile,
        title_prefix=title_prefix,
    )

    total_in = float(np.nansum(H_in))
    total_out = float(np.nansum(H_out))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (H_out - H_in) / H_in
        rel[~np.isfinite(rel)] = np.nan
    finite_rel = rel[np.isfinite(rel)]
    relative_total_change = (total_out - total_in) / total_in if total_in else np.nan

    if (not args.stochastic) and total_in > 0 and abs(relative_total_change) > 1e-6:
        print(
            f"[WARN] conservation drift {dataset_name} {point} {tag}: "
            f"relative_total_change={relative_total_change:.3e}"
        )

    summary = {
        "dataset": dataset_name,
        "point": point,
        "tag": tag,
        "energy_factor": energy_factor,
        "map_csv": str(map_csv),
        "ecrit_csv": str(ecrit_csv),
        "value_col": info["value_col"],
        "theta_min": tmin,
        "theta_max": tmax,
        "phi_min": pmin,
        "phi_max": pmax,
        "stochastic": int(args.stochastic),
        "n_theta_bins": len(info["theta"]),
        "n_phi_bins": len(info["phi"]),
        "input_total": total_in,
        "smeared_total": total_out,
        "relative_total_change": relative_total_change,
        "p90_abs_relative_delta": float(np.nanpercentile(np.abs(finite_rel), 90)) if finite_rel.size else np.nan,
        "p99_abs_relative_delta": float(np.nanpercentile(np.abs(finite_rel), 99)) if finite_rel.size else np.nan,
        "output_csv": str(out_csv),
        "comparison_png": str(comparison_png),
        **stats,
    }
    print(f"[OK] {dataset_name} {point} {tag}: total_in={total_in:.6g}, total_out={total_out:.6g}, rel={relative_total_change:.3e}")
    return summary


def run_toy_test() -> int:
    args = argparse.Namespace(
        value_col=None,
        theta_min=None,
        theta_max=None,
        phi_min=None,
        phi_max=None,
        stochastic=False,
        kernel_threshold=None,
        max_kernel_radius_mrad=None,
        square_display=True,
        display_step=1.0,
        blank_zeros=True,
        vmax_percentile=99.0,
        relative_vmax_percentile=98.0,
    )
    theta = np.arange(0.5, 5.5, 1.0)
    phi = np.arange(-2.0, 3.0, 1.0)
    H = np.zeros((len(theta), len(phi)), dtype=float)
    H[2, 2] = 100.0
    filled = np.ones_like(H, dtype=bool)
    L = np.ones_like(H) * 500.0
    E = np.ones_like(H) * 500.0
    out, stats = smear_empirical(H, theta, phi, filled, L, E, DummyKernelModel(), stochastic=False, rng=None)
    rel = abs(float(np.sum(out) - np.sum(H))) / float(np.sum(H))
    ok = rel < 1e-12 and stats["n_sources_smeared"] == 1
    print(f"TOY TEST: {'PASS' if ok else 'FAIL'} | relative_total_change={rel:.3e} | stats={stats}")
    return 0 if ok else 2


def parser():
    ap = argparse.ArgumentParser(description="Aplica smearing angular con kernel empírico Geant4 completo.")
    ap.add_argument("--map-csv", type=Path, default=None, help="Mapa angular principal para una corrida.")
    ap.add_argument("--inside-map-csv", type=Path, default=None, help="Mapa angular inside-volcano opcional.")
    ap.add_argument("--point", default=None)
    ap.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS))
    ap.add_argument("--energy-factors", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    ap.add_argument("--map-template", default=None)
    ap.add_argument("--inside-map-template", default=None)
    ap.add_argument("--ecrit-template", default=None)
    ap.add_argument("--ecrit-dir", type=Path, default=None)
    ap.add_argument(
        "--kernel-library",
        type=Path,
        default=Path(__file__).resolve().parent / "hybrid_empirical_kernel_library.npz",
        help="Empirical kernel library (default: bundled hybrid full-tail model).",
    )
    ap.add_argument("--outdir", type=Path, default=Path("outputs_smearing_empirical"))
    ap.add_argument("--interp-method", choices=["tail-aware", "rbf_linear", "linear", "nearest"], default="tail-aware")
    ap.add_argument("--value-col", default=None)

    ap.add_argument("--theta-min", type=float, default=None, help="Mínimo theta a usar. Default: inferido desde CSV")
    ap.add_argument("--theta-max", type=float, default=90.0)
    ap.add_argument("--phi-min", type=float, default=-60.0)
    ap.add_argument("--phi-max", type=float, default=60.0)
    ap.add_argument("--display-step", type=float, default=0.5)
    ap.add_argument("--square-display", dest="square_display", action="store_true", default=True)
    ap.add_argument("--native-display", dest="square_display", action="store_false")
    ap.add_argument("--blank-zeros", dest="blank_zeros", action="store_true", default=True)
    ap.add_argument("--show-zeros", dest="blank_zeros", action="store_false")
    ap.add_argument("--vmax-percentile", type=float, default=99.0)
    ap.add_argument("--relative-vmax-percentile", type=float, default=98.0)

    ap.add_argument("--stochastic", action="store_true", help="Genera una realización Monte Carlo ruidosa del smearing.")
    ap.add_argument("--random-seed", type=int, default=12345)
    ap.add_argument("--kernel-threshold", type=float, default=None, help="Descarta pesos K menores que este valor antes de renormalizar.")
    ap.add_argument("--max-kernel-radius-mrad", type=float, default=None, help="Limita el radio angular evaluado del kernel.")
    ap.add_argument("--run-toy-test", action="store_true")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    setup_style()

    if args.run_toy_test:
        return run_toy_test()

    if args.kernel_library is None:
        raise SystemExit("Se requiere --kernel-library, salvo cuando usas --run-toy-test.")
    model = EmpiricalKernelModel(args.kernel_library, interp_method=args.interp_method)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    jobs = []
    if args.map_csv or args.inside_map_csv:
        if args.map_csv is None:
            raise SystemExit("Modo simple requiere --map-csv.")
        if args.ecrit_template is None and args.ecrit_dir is None:
            raise SystemExit("Modo simple requiere --ecrit-template o --ecrit-dir.")
        point = args.point or "POINT"
        for factor in args.energy_factors:
            tag = factor_tag(factor)
            if args.ecrit_template is not None:
                ecrit_csv = Path(args.ecrit_template.format(point=point, factor=factor, tag=tag))
            else:
                ecrit_csv = args.ecrit_dir / f"ecrit_table_{point}.csv"
            jobs.append({
                "point": point,
                "tag": tag,
                "factor": factor,
                "map_csv": args.map_csv,
                "inside_map_csv": args.inside_map_csv,
                "ecrit_csv": ecrit_csv,
            })
    else:
        if args.map_template is None:
            raise SystemExit("Modo lote requiere --map-template, o usa --map-csv.")
        if args.ecrit_template is None and args.ecrit_dir is None:
            raise SystemExit("Modo lote requiere --ecrit-template o --ecrit-dir.")
        for point in args.points:
            for factor in args.energy_factors:
                tag = factor_tag(factor)
                inside_map_csv = None
                if args.inside_map_template is not None:
                    inside_map_csv = Path(args.inside_map_template.format(point=point, factor=factor, tag=tag))
                if args.ecrit_template is not None:
                    ecrit_csv = Path(args.ecrit_template.format(point=point, factor=factor, tag=tag))
                else:
                    ecrit_csv = args.ecrit_dir / f"ecrit_table_{point}.csv"
                jobs.append({
                    "point": point,
                    "tag": tag,
                    "factor": factor,
                    "map_csv": Path(args.map_template.format(point=point, factor=factor, tag=tag)),
                    "inside_map_csv": inside_map_csv,
                    "ecrit_csv": ecrit_csv,
                })

    summaries = []
    for job in jobs:
        point = job["point"]
        tag = job["tag"]
        factor = job["factor"]
        map_csv = job["map_csv"]
        inside_map_csv = job["inside_map_csv"]
        ecrit_csv = job["ecrit_csv"]

        if not map_csv.exists():
            print(f"[WARN] No existe map CSV: {map_csv}. Salto.")
            continue
        if not ecrit_csv.exists():
            print(f"[WARN] No existe Ecrit CSV: {ecrit_csv}. Salto.")
            continue

        summaries.append(process_dataset(
            map_csv=map_csv,
            ecrit_csv=ecrit_csv,
            point=point,
            outdir=args.outdir,
            tag=tag,
            energy_factor=factor,
            args=args,
            dataset_name="full_map",
            title_prefix="Empirical angular smearing",
            prefix_stem="smearing_empirical",
            model=model,
            rng=rng,
        ))

        if inside_map_csv is not None:
            if inside_map_csv.exists():
                summaries.append(process_dataset(
                    map_csv=inside_map_csv,
                    ecrit_csv=ecrit_csv,
                    point=point,
                    outdir=args.outdir,
                    tag=tag,
                    energy_factor=factor,
                    args=args,
                    dataset_name="inside_volcano",
                    title_prefix="Inside-volcano counts after empirical smearing",
                    prefix_stem="inside_volcano_smearing_empirical",
                    model=model,
                    rng=rng,
                ))
            else:
                print(f"[WARN] No existe inside-map CSV: {inside_map_csv}. Se omite inside-volcano.")

    if summaries:
        summary_csv = args.outdir / "smearing_empirical_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_csv, index=False)
        print(f"[DONE] Summary: {summary_csv}")
    else:
        print("[WARN] No se generaron salidas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
