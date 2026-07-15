#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build compact per-point event caches and filtered map products in one SHW pass.

This is the fast path for large runs:

  SHW -> Ecrit filter -> per-point event cache
      -> 05_plots/filtered maps
      -> 06_inside_volcano/filtered tables and figures

It intentionally does not write filtered SHW files. Downstream event-by-event MC can
read the cache directly, avoiding another full pass through the input.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from matplotlib.colors import LogNorm

try:
    from plot_style import COUNTS_CMAP, apply_scientific_style, finite_percentile, format_angular_axes, style_colorbar
    from shw_io import MUON_MASS_GEV, open_shw_bytes, parse_muon_parts, stream_size_hint, theta_phi_from_momentum
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_style import COUNTS_CMAP, apply_scientific_style, finite_percentile, format_angular_axes, style_colorbar
    from shw_io import MUON_MASS_GEV, open_shw_bytes, parse_muon_parts, stream_size_hint, theta_phi_from_momentum

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


SUMMIT = (4.486552, -75.388975)
POINTS = {
    "P1": (4.492298, -75.381092),
    "P2": (4.494946, -75.388110),
    "P4": (4.476500, -75.386500),
    "P5": (4.488500, -75.379500),
}
ORDER = ["P1", "P2", "P4", "P5"]
MUON_IDS_B = {b"0005", b"0006", b"5", b"6"}


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


def pick_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
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
    c = np.asarray(sorted(np.unique(np.asarray(centers, dtype=float))), dtype=float)
    if c.size == 0:
        raise ValueError("Empty coordinate array.")
    if c.size == 1:
        half = 0.5 * fallback_step
        return np.array([c[0] - half, c[0] + half], dtype=float)
    mids = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mids[0] - c[0])], mids, [c[-1] + (c[-1] - mids[-1])]])


def nearest_index(arr: np.ndarray, x: float, tol: float) -> int | None:
    n = len(arr)
    if n == 0:
        return None
    k = int(np.searchsorted(arr, x))
    best = None
    best_d = float("inf")
    for cand in (k, k - 1):
        if 0 <= cand < n:
            d = abs(float(arr[cand]) - x)
            if d < best_d:
                best = cand
                best_d = d
    if best is None or best_d > tol:
        return None
    return best


def solid_angle_per_bin(theta_edges_deg: np.ndarray, phi_edges_deg: np.ndarray) -> np.ndarray:
    th = np.deg2rad(theta_edges_deg)
    ph = np.deg2rad(phi_edges_deg)
    return (np.cos(th[:-1]) - np.cos(th[1:]))[:, None] * np.diff(ph)[None, :]


@dataclass
class PointCache:
    point: str
    theta: np.ndarray
    phi: np.ndarray
    theta_edges: np.ndarray
    phi_edges: np.ndarray
    ecrit: np.ndarray
    valid: np.ndarray
    inside_mask: np.ndarray
    length_inside_m: np.ndarray
    clear_sky: np.ndarray
    phi0: float
    plot_theta_edges: np.ndarray
    plot_phi_edges: np.ndarray
    plot_counts: np.ndarray
    grid_counts_all: np.ndarray
    grid_counts_inside: np.ndarray
    theta_events: list[float]
    phi_rel_events: list[float]
    kinetic_events: list[float]
    total_energy_events: list[float]
    length_events: list[float]
    inside_events: list[int]
    theta_index_events: list[int]
    phi_index_events: list[int]
    n_in_angular_grid_before_filter: int = 0
    n_out_of_ecrit_grid: int = 0
    n_failed_ecrit: int = 0
    n_kept: int = 0
    n_kept_without_ecrit_cell: int = 0
    n_kept_in_ecrit_grid: int = 0


def load_point_cache(ecrit_dir: Path, point: str, plot_theta_edges: np.ndarray, plot_phi_edges: np.ndarray) -> PointCache:
    path = ecrit_dir / f"ecrit_table_{point}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    theta_col = pick_column(df, ["theta_deg", "theta", "zenith_deg"])
    phi_col = pick_column(df, ["phi_deg", "phi_rel_deg", "phi", "azimuth_deg"])
    ecrit_col = pick_column(df, ["Ecrit_total_GeV", "Ecrit_total", "Ecrit"])
    length_col = pick_column(df, ["length_inside_m", "L_m", "rock_length_m", "length_m", "longitud_m", "length"], required=False)
    inside_col = pick_column(df, ["inside_volcano_geometry", "inside", "mask", "blocked"], required=False)
    clear_col = pick_column(df, ["clear_sky_geometry", "clear_sky"], required=False)

    keep_cols = [theta_col, phi_col, ecrit_col]
    for col in (length_col, inside_col, clear_col):
        if col and col not in keep_cols:
            keep_cols.append(col)
    df = df[keep_cols].copy()
    for col in keep_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[theta_col, phi_col]).copy()

    theta = np.array(sorted(df[theta_col].unique()), dtype=float)
    phi = np.array(sorted(df[phi_col].unique()), dtype=float)
    theta_edges = centers_to_edges(theta, fallback_step=0.5)
    phi_edges = centers_to_edges(phi, fallback_step=0.5)

    shape = (len(theta), len(phi))
    ecrit = np.full(shape, np.nan, dtype=float)
    length = np.zeros(shape, dtype=float)
    inside = np.zeros(shape, dtype=bool)
    clear = np.zeros(shape, dtype=bool)
    valid = np.zeros(shape, dtype=bool)
    ti = {round(v, 10): i for i, v in enumerate(theta)}
    pj = {round(v, 10): j for j, v in enumerate(phi)}

    for row in df.itertuples(index=False):
        th = float(getattr(row, theta_col))
        ph = float(getattr(row, phi_col))
        i = ti.get(round(th, 10))
        j = pj.get(round(ph, 10))
        if i is None or j is None:
            continue
        valid[i, j] = True
        ecrit_val = getattr(row, ecrit_col)
        ecrit[i, j] = float(ecrit_val) if pd.notna(ecrit_val) else np.nan
        if length_col:
            val = getattr(row, length_col)
            length[i, j] = float(val) if pd.notna(val) and np.isfinite(val) else 0.0
        if inside_col:
            val = getattr(row, inside_col)
            inside[i, j] = bool(pd.notna(val) and float(val) > 0.0)
        else:
            inside[i, j] = length[i, j] > 0.0
        if clear_col:
            val = getattr(row, clear_col)
            clear[i, j] = bool(pd.notna(val) and float(val) > 0.0)
        else:
            clear[i, j] = valid[i, j] & ~inside[i, j]

    return PointCache(
        point=point,
        theta=theta,
        phi=phi,
        theta_edges=theta_edges,
        phi_edges=phi_edges,
        ecrit=ecrit,
        valid=valid,
        inside_mask=inside,
        length_inside_m=length,
        clear_sky=clear,
        phi0=phi0_for_point(point),
        plot_theta_edges=plot_theta_edges,
        plot_phi_edges=plot_phi_edges,
        plot_counts=np.zeros((len(plot_theta_edges) - 1, len(plot_phi_edges) - 1), dtype=np.int64),
        grid_counts_all=np.zeros(shape, dtype=np.int64),
        grid_counts_inside=np.zeros(shape, dtype=np.int64),
        theta_events=[],
        phi_rel_events=[],
        kinetic_events=[],
        total_energy_events=[],
        length_events=[],
        inside_events=[],
        theta_index_events=[],
        phi_index_events=[],
    )


def match_ecrit_cell(point: PointCache, theta_deg: float, phi_rel: float, tol_theta: float, tol_phi: float) -> tuple[int, int] | None:
    phi_candidates = (phi_rel, phi_rel if phi_rel <= 180.0 else phi_rel - 360.0)
    i = nearest_index(point.theta, theta_deg, tol_theta)
    if i is None:
        return None
    for phi_c in phi_candidates:
        j = nearest_index(point.phi, phi_c, tol_phi)
        if j is not None:
            return i, j
    return None


def plot_bin(point: PointCache, theta_deg: float, phi_rel: float) -> tuple[int, int] | None:
    i = int(np.searchsorted(point.plot_theta_edges, theta_deg, side="right") - 1)
    j = int(np.searchsorted(point.plot_phi_edges, phi_rel, side="right") - 1)
    if 0 <= i < point.plot_counts.shape[0] and 0 <= j < point.plot_counts.shape[1]:
        return i, j
    return None


def add_event_to_cache(
    point: PointCache,
    theta_deg: float,
    phi_rel: float,
    total_energy: float,
    i: int,
    j: int,
    cache_inside_only: bool = False,
) -> None:
    length = float(point.length_inside_m[i, j])
    inside = bool(point.inside_mask[i, j])
    point.n_kept_in_ecrit_grid += 1
    point.grid_counts_all[i, j] += 1
    if inside:
        point.grid_counts_inside[i, j] += 1
    if cache_inside_only and not inside:
        return
    point.theta_events.append(float(theta_deg))
    point.phi_rel_events.append(float(phi_rel))
    point.total_energy_events.append(float(total_energy))
    point.kinetic_events.append(float(total_energy - MUON_MASS_GEV))
    point.length_events.append(length)
    point.inside_events.append(1 if inside else 0)
    point.theta_index_events.append(int(i))
    point.phi_index_events.append(int(j))


def nearest_indices_vectorized(arr: np.ndarray, values: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(arr)
    idx = np.zeros(values.shape, dtype=np.int64)
    if n == 0 or values.size == 0:
        return idx, np.zeros(values.shape, dtype=bool)

    k = np.searchsorted(arr, values)
    k0 = np.clip(k, 0, n - 1)
    k1 = np.clip(k - 1, 0, n - 1)
    d0 = np.abs(arr[k0] - values)
    d1 = np.abs(arr[k1] - values)
    use_prev = d1 < d0
    idx = np.where(use_prev, k1, k0).astype(np.int64)
    dist = np.where(use_prev, d1, d0)
    valid = np.isfinite(values) & (dist <= tol)
    return idx, valid


def append_selected_events(
    point: PointCache,
    theta_deg: np.ndarray,
    phi_rel: np.ndarray,
    total_energy: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    cache_inside_only: bool = False,
) -> None:
    if theta_deg.size == 0:
        return
    inside = point.inside_mask[i, j]
    point.n_kept_in_ecrit_grid += int(theta_deg.size)
    flat = i.astype(np.int64) * len(point.phi) + j.astype(np.int64)
    counts = np.bincount(flat, minlength=point.grid_counts_all.size).reshape(point.grid_counts_all.shape)
    point.grid_counts_all += counts
    point.grid_counts_inside += counts * point.inside_mask.astype(np.int64)

    if cache_inside_only:
        theta_deg = theta_deg[inside]
        phi_rel = phi_rel[inside]
        total_energy = total_energy[inside]
        i = i[inside]
        j = j[inside]
        inside = inside[inside]
        if theta_deg.size == 0:
            return

    length = point.length_inside_m[i, j].astype(np.float32, copy=False)
    point.theta_events.append(theta_deg.astype(np.float32, copy=True))
    point.phi_rel_events.append(phi_rel.astype(np.float32, copy=True))
    point.total_energy_events.append(total_energy.astype(np.float32, copy=True))
    point.kinetic_events.append((total_energy - MUON_MASS_GEV).astype(np.float32, copy=True))
    point.length_events.append(length)
    point.inside_events.append(inside.astype(np.uint8, copy=True))
    point.theta_index_events.append(i.astype(np.int32, copy=True))
    point.phi_index_events.append(j.astype(np.int32, copy=True))


def add_plot_counts(point: PointCache, theta_deg: np.ndarray, phi_rel: np.ndarray) -> None:
    if theta_deg.size == 0:
        return
    H, _, _ = np.histogram2d(theta_deg, phi_rel, bins=[point.plot_theta_edges, point.plot_phi_edges])
    point.plot_counts += H.astype(np.int64)


def process_kinematic_arrays(args, points: dict[str, PointCache], theta: np.ndarray, phi_abs: np.ndarray,
                             total_energy: np.ndarray, pz_positive: np.ndarray | None) -> int:
    theta = np.asarray(theta, dtype=float)
    phi_abs = np.asarray(phi_abs, dtype=float)
    total_energy = np.asarray(total_energy, dtype=float)
    if pz_positive is None:
        pz_positive = np.zeros(theta.shape, dtype=np.uint8)
    else:
        pz_positive = np.asarray(pz_positive, dtype=np.uint8)

    base = np.isfinite(theta) & np.isfinite(phi_abs) & np.isfinite(total_energy)
    if args.discard_upgoing:
        base &= pz_positive == 0
    n_discarded_upgoing = int(np.count_nonzero(np.isfinite(theta) & (pz_positive != 0))) if args.discard_upgoing else 0
    if not np.any(base):
        return n_discarded_upgoing

    theta_b = theta[base]
    phi_abs_b = phi_abs[base]
    total_b = total_energy[base]

    for point in points.values():
        phi_rel = (phi_abs_b - point.phi0) % 360.0
        if args.wrap180:
            phi_rel = np.where(phi_rel > 180.0, phi_rel - 360.0, phi_rel)

        i, ok_i = nearest_indices_vectorized(point.theta, theta_b, args.tol_theta)
        j, ok_j = nearest_indices_vectorized(point.phi, phi_rel, args.tol_phi)
        matched = ok_i & ok_j
        point.n_out_of_ecrit_grid += int(np.count_nonzero(~matched))

        keep_outside = (~matched) & bool(args.treat_out_of_grid_as_clear)
        if np.any(keep_outside):
            point.n_kept += int(np.count_nonzero(keep_outside))
            point.n_kept_without_ecrit_cell += int(np.count_nonzero(keep_outside))
            add_plot_counts(point, theta_b[keep_outside], phi_rel[keep_outside])

        if not np.any(matched):
            continue

        im = i[matched]
        jm = j[matched]
        em = total_b[matched]
        thm = theta_b[matched]
        phm = phi_rel[matched]
        point.n_in_angular_grid_before_filter += int(im.size)

        valid_cell = point.valid[im, jm] & np.isfinite(point.ecrit[im, jm])
        keep_m = np.zeros(im.shape, dtype=bool)
        if np.any(valid_cell):
            keep_m[valid_cell] = em[valid_cell] >= point.ecrit[im[valid_cell], jm[valid_cell]]
        if args.treat_out_of_grid_as_clear:
            keep_m[~valid_cell] = True

        point.n_failed_ecrit += int(np.count_nonzero(~keep_m))
        if not np.any(keep_m):
            continue

        point.n_kept += int(np.count_nonzero(keep_m))
        add_plot_counts(point, thm[keep_m], phm[keep_m])
        append_selected_events(
            point,
            thm[keep_m],
            phm[keep_m],
            em[keep_m],
            im[keep_m],
            jm[keep_m],
            cache_inside_only=(args.event_cache_source_mode == "inside"),
        )

    return n_discarded_upgoing


def process_shw(args, points: dict[str, PointCache]) -> dict[str, int]:
    total_bytes = stream_size_hint(args.shw)
    update_bytes = max(1, int(args.progress_update_mb * 1024 * 1024))
    pending_update = 0
    stats = {
        "n_lines_read": 0,
        "n_particles_read": 0,
        "n_muons_read": 0,
        "n_discarded_upgoing": 0,
        "n_bad_momentum": 0,
    }

    with open_shw_bytes(args.shw, member_name=args.shw_member) as fin:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="event-cache")
        for raw in fin:
            stats["n_lines_read"] += 1
            pending_update += len(raw)
            if pending_update >= update_bytes:
                pbar.update(pending_update)
                pending_update = 0

            s = raw.strip()
            if not s or s.startswith(b"#"):
                continue

            rec = parse_muon_parts(s.split(), shw_format=args.shw_format, only_muons=True)
            if rec is None:
                continue
            stats["n_particles_read"] += 1
            if rec.pid in MUON_IDS_B:
                stats["n_muons_read"] += 1
            if args.discard_upgoing and rec.pz > 0.0:
                stats["n_discarded_upgoing"] += 1
                continue
            angles = theta_phi_from_momentum(rec.px, rec.py, rec.pz)
            if angles is None:
                stats["n_bad_momentum"] += 1
                continue
            theta_deg, phi_abs = angles

            for point in points.values():
                phi_rel = (phi_abs - point.phi0) % 360.0
                if args.wrap180 and phi_rel > 180.0:
                    phi_rel -= 360.0

                matched = match_ecrit_cell(point, theta_deg, phi_rel, args.tol_theta, args.tol_phi)
                if matched is None:
                    point.n_out_of_ecrit_grid += 1
                    keep = bool(args.treat_out_of_grid_as_clear)
                    if keep:
                        point.n_kept += 1
                        pbin = plot_bin(point, theta_deg, phi_rel)
                        if pbin is not None:
                            point.plot_counts[pbin] += 1
                        point.n_kept_without_ecrit_cell += 1
                    continue

                i, j = matched
                point.n_in_angular_grid_before_filter += 1
                valid = bool(point.valid[i, j])
                ecrit = float(point.ecrit[i, j])
                if (not valid) or (not np.isfinite(ecrit)):
                    keep = bool(args.treat_out_of_grid_as_clear)
                else:
                    keep = float(rec.e_total_GeV) >= ecrit
                if not keep:
                    point.n_failed_ecrit += 1
                    continue

                point.n_kept += 1
                pbin = plot_bin(point, theta_deg, phi_rel)
                if pbin is not None:
                    point.plot_counts[pbin] += 1
                add_event_to_cache(
                    point,
                    theta_deg,
                    phi_rel,
                    rec.e_total_GeV,
                    i,
                    j,
                    cache_inside_only=(args.event_cache_source_mode == "inside"),
                )

            if args.head and min(p.n_kept for p in points.values()) >= args.head:
                break

        if pending_update:
            pbar.update(pending_update)
        pbar.close()
    return stats


def iter_kinematic_cache(cache_path: Path):
    cache_path = Path(cache_path)
    if cache_path.is_dir():
        manifest_path = cache_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No encontre manifest.json en {cache_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks_dir = cache_path / "chunks"
        for chunk in manifest.get("chunks", []):
            path = chunks_dir / chunk["file"]
            with np.load(path) as data:
                yield manifest, {
                    "theta_deg": np.asarray(data["theta_deg"]),
                    "phi_abs_deg": np.asarray(data["phi_abs_deg"]),
                    "total_E_GeV": np.asarray(data["total_E_GeV"]),
                    "pz_positive": np.asarray(data["pz_positive"]) if "pz_positive" in data else None,
                }
        return

    with np.load(cache_path) as data:
        manifest = {"n_lines_read": 0, "n_events": int(data["theta_deg"].size), "n_muons_read": int(data["theta_deg"].size)}
        yield manifest, {
            "theta_deg": np.asarray(data["theta_deg"]),
            "phi_abs_deg": np.asarray(data["phi_abs_deg"]),
            "total_E_GeV": np.asarray(data["total_E_GeV"]),
            "pz_positive": np.asarray(data["pz_positive"]) if "pz_positive" in data else None,
        }


def process_kinematic_cache(args, points: dict[str, PointCache]) -> dict[str, int]:
    cache_path = Path(args.kinematic_cache)
    manifest_path = cache_path / "manifest.json" if cache_path.is_dir() else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.exists() else {}
    total_events = int(manifest.get("n_events", 0))
    stats = {
        "n_lines_read": int(manifest.get("n_lines_read", 0)),
        "n_particles_read": 0,
        "n_muons_read": 0,
        "n_discarded_upgoing": 0,
        "n_bad_momentum": 0,
    }
    pbar = tqdm(total=total_events or None, unit="event", unit_scale=True, desc="event-cache from kinematics")
    for _, chunk in iter_kinematic_cache(cache_path):
        theta = chunk["theta_deg"]
        n = int(theta.size)
        stats["n_particles_read"] += n
        stats["n_muons_read"] += n
        stats["n_discarded_upgoing"] += process_kinematic_arrays(
            args,
            points,
            theta=theta,
            phi_abs=chunk["phi_abs_deg"],
            total_energy=chunk["total_E_GeV"],
            pz_positive=chunk.get("pz_positive"),
        )
        pbar.update(n)
        if args.head and min(p.n_kept for p in points.values()) >= args.head:
            break
    pbar.close()
    return stats


def concat_or_array(blocks, dtype) -> np.ndarray:
    if not blocks:
        return np.asarray([], dtype=dtype)
    first = blocks[0]
    if isinstance(first, np.ndarray):
        return np.concatenate([np.asarray(block, dtype=dtype) for block in blocks]).astype(dtype, copy=False)
    return np.asarray(blocks, dtype=dtype)


def save_npz(point: PointCache, outdir: Path, compressed: bool) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"events_{point.point}.npz"
    arrays = {
        "point": np.array(point.point),
        "theta_deg": concat_or_array(point.theta_events, np.float32),
        "phi_rel_deg": concat_or_array(point.phi_rel_events, np.float32),
        "kinetic_GeV": concat_or_array(point.kinetic_events, np.float32),
        "total_E_GeV": concat_or_array(point.total_energy_events, np.float32),
        "length_inside_m": concat_or_array(point.length_events, np.float32),
        "inside_source": concat_or_array(point.inside_events, np.uint8),
        "theta_index": concat_or_array(point.theta_index_events, np.int32),
        "phi_index": concat_or_array(point.phi_index_events, np.int32),
        "theta_centers": point.theta.astype(np.float32),
        "phi_centers": point.phi.astype(np.float32),
        "theta_edges": point.theta_edges.astype(np.float32),
        "phi_edges": point.phi_edges.astype(np.float32),
        "ecrit_total_GeV": point.ecrit.astype(np.float32),
        "length_grid_m": point.length_inside_m.astype(np.float32),
        "inside_mask": point.inside_mask.astype(np.uint8),
        "filled": point.valid.astype(np.uint8),
    }
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return path


def cached_event_count(point: PointCache) -> int:
    if not point.theta_events:
        return 0
    return int(sum(np.asarray(block).size for block in point.theta_events))


def save_theta_phi_maps(point: PointCache, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    th_edges = point.plot_theta_edges
    ph_edges = point.plot_phi_edges
    H = point.plot_counts
    th_cent = 0.5 * (th_edges[:-1] + th_edges[1:])
    ph_cent = 0.5 * (ph_edges[:-1] + ph_edges[1:])
    TH, PH = np.meshgrid(th_cent, ph_cent, indexing="ij")
    domega = solid_angle_per_bin(th_edges, ph_edges)

    counts_csv = outdir / f"theta_phi_counts_{point.point}.csv"
    dno_csv = outdir / f"theta_phi_dNdOmega_{point.point}.csv"
    counts_png = outdir / f"theta_phi_counts_{point.point}.png"
    dno_png = outdir / f"theta_phi_dNdOmega_{point.point}.png"

    pd.DataFrame({"theta_deg": TH.ravel(), "phi_rel_deg": PH.ravel(), "count": H.ravel()}).to_csv(counts_csv, index=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        H_domega = H.astype(float) / domega
        H_domega[~np.isfinite(H_domega)] = np.nan
    pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "count": H.ravel(),
        "delta_omega_sr": domega.ravel(),
        "dN_dOmega_count_per_sr": H_domega.ravel(),
    }).to_csv(dno_csv, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    vmax = finite_percentile(H, 99.5, positive_only=True, fallback=max(float(H.max()), 1.0))
    im = ax.pcolormesh(ph_edges, th_edges, H, shading="flat", cmap=COUNTS_CMAP, vmin=0.0, vmax=vmax, rasterized=True)
    format_angular_axes(ax, th_edges[0], th_edges[-1], ph_edges[0], ph_edges[-1])
    ax.set_title(f"Muon counts | {point.point} | N={int(H.sum())}")
    style_colorbar(fig.colorbar(im, ax=ax, shrink=0.92), "Counts")
    fig.savefig(counts_png)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    vmax = finite_percentile(H_domega, 99.5, positive_only=True, fallback=None)
    im = ax.pcolormesh(ph_edges, th_edges, H_domega, shading="flat", cmap=COUNTS_CMAP, vmin=0.0, vmax=vmax, rasterized=True)
    format_angular_axes(ax, th_edges[0], th_edges[-1], ph_edges[0], ph_edges[-1])
    ax.set_title(f"Muon intensity proxy | {point.point}")
    style_colorbar(fig.colorbar(im, ax=ax, shrink=0.92), r"Counts sr$^{-1}$")
    fig.savefig(dno_png)
    plt.close(fig)
    return [counts_csv, dno_csv, counts_png, dno_png]


def save_inside_tables(point: PointCache, outdir: Path, global_stats: dict[str, int]) -> list[Path]:
    point_dir = outdir / "filtered" / point.point
    point_dir.mkdir(parents=True, exist_ok=True)
    th_cent, ph_cent = point.theta, point.phi
    TH, PH = np.meshgrid(th_cent, ph_cent, indexing="ij")
    domega = solid_angle_per_bin(point.theta_edges, point.phi_edges)
    H_inside = point.grid_counts_inside
    H_all = point.grid_counts_all
    with np.errstate(divide="ignore", invalid="ignore"):
        I = H_inside.astype(float) / domega
        I[~point.inside_mask] = np.nan
        I[~np.isfinite(I)] = np.nan

    counts_csv = point_dir / f"counts_inside_volcano_{point.point}.csv"
    dno_csv = point_dir / f"dNdOmega_inside_volcano_{point.point}.csv"
    summary_csv = point_dir / f"inside_volcano_summary_{point.point}.csv"
    pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "inside_volcano_geometry": point.inside_mask.ravel().astype(int),
        "count_inside_geometry": H_inside.ravel(),
        "count_all_in_grid": H_all.ravel(),
    }).to_csv(counts_csv, index=False)
    pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "inside_volcano_geometry": point.inside_mask.ravel().astype(int),
        "count_inside_geometry": H_inside.ravel(),
        "delta_omega_sr": domega.ravel(),
        "dN_dOmega_inside_count_per_sr": I.ravel(),
    }).to_csv(dno_csv, index=False)

    summary = {
        **global_stats,
        "point": point.point,
        "n_events_in_angular_grid": int(point.grid_counts_all.sum()),
        "n_events_inside_volcano_geometry": int(point.grid_counts_inside.sum()),
        "n_events_in_grid_but_outside_geometry": int(point.grid_counts_all.sum() - point.grid_counts_inside.sum()),
        "fraction_inside_given_in_grid": float(point.grid_counts_inside.sum() / point.grid_counts_all.sum()) if point.grid_counts_all.sum() else np.nan,
        "mask_column": "inside_volcano_geometry",
        "mask_min": 0.0,
        "geometry_csv": "from_ecrit_table",
    }
    pd.DataFrame({"quantity": list(summary.keys()), "value": list(summary.values())}).to_csv(summary_csv, index=False)
    return [counts_csv, dno_csv, summary_csv]


def square_display(point: PointCache, display_step: float, theta_min: float, theta_max: float, phi_min: float, phi_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    th_edges = np.arange(theta_min, theta_max + display_step, display_step)
    ph_edges = np.arange(phi_min, phi_max + display_step, display_step)
    th_cent = 0.5 * (th_edges[:-1] + th_edges[1:])
    ph_cent = 0.5 * (ph_edges[:-1] + ph_edges[1:])
    out = np.full((len(th_cent), len(ph_cent)), np.nan, dtype=float)
    H = point.grid_counts_inside.astype(float)
    H[~point.inside_mask] = np.nan
    for i, th in enumerate(th_cent):
        src_i = int(np.searchsorted(point.theta_edges, th, side="right") - 1)
        if src_i < 0 or src_i >= H.shape[0]:
            continue
        for j, ph in enumerate(ph_cent):
            src_j = int(np.searchsorted(point.phi_edges, ph, side="right") - 1)
            if src_j < 0 or src_j >= H.shape[1]:
                continue
            out[i, j] = H[src_i, src_j]
    return th_edges, ph_edges, out


def save_inside_figures(point: PointCache, figures_dir: Path, args) -> list[Path]:
    outdir = figures_dir / "filtered"
    outdir.mkdir(parents=True, exist_ok=True)
    th_edges, ph_edges, H = square_display(
        point,
        args.inside_display_step,
        args.inside_display_theta_min,
        args.inside_display_theta_max,
        args.inside_display_phi_min,
        args.inside_display_phi_max,
    )
    if not args.inside_show_zeros:
        H = H.copy()
        H[H <= 0] = np.nan
    pos = H[np.isfinite(H) & (H > 0)]
    vmax_linear = float(np.nanpercentile(pos, args.inside_vmax_percentile)) if pos.size else 1.0
    vmax_log = float(np.nanmax(pos)) if pos.size else 1.0
    vmin_log = max(1.0, float(np.nanmin(pos))) if pos.size else 1.0

    linear_png = outdir / f"inside_volcano_filtered_{point.point}_linear.png"
    log_png = outdir / f"inside_volcano_filtered_{point.point}_log.png"

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.pcolormesh(ph_edges, th_edges, H, shading="flat", cmap=COUNTS_CMAP, vmin=0.0, vmax=vmax_linear)
    format_angular_axes(ax, args.inside_display_theta_min, args.inside_display_theta_max, args.inside_display_phi_min, args.inside_display_phi_max)
    ax.set_title(f"Inside-volcano counts | filtered | {point.point}")
    style_colorbar(fig.colorbar(im, ax=ax, pad=0.02), "Counts per angular bin")
    fig.savefig(linear_png)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    if pos.size:
        im = ax.pcolormesh(ph_edges, th_edges, H, shading="flat", cmap=COUNTS_CMAP, norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
    else:
        im = ax.pcolormesh(ph_edges, th_edges, H, shading="flat", cmap=COUNTS_CMAP)
    format_angular_axes(ax, args.inside_display_theta_min, args.inside_display_theta_max, args.inside_display_phi_min, args.inside_display_phi_max)
    ax.set_title(f"Inside-volcano counts | filtered | {point.point} | log")
    style_colorbar(fig.colorbar(im, ax=ax, pad=0.02), "Counts per angular bin")
    fig.savefig(log_png)
    plt.close(fig)
    return [linear_png, log_png]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "point", "kind", "path"])
        writer.writeheader()
        writer.writerows(rows)


def save_point_products(
    pc: PointCache,
    args,
    stats: dict[str, int],
    plot_dir: Path,
    figures_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    rows: list[dict[str, str]] = []
    cache_path = save_npz(pc, args.cache_outdir, compressed=not args.uncompressed_cache)
    rows.append({"stage": "04_event_cache", "point": pc.point, "kind": "event_cache_npz", "path": str(cache_path)})
    for path in save_theta_phi_maps(pc, plot_dir):
        rows.append({"stage": "05_plots_filtered", "point": pc.point, "kind": path.stem, "path": str(path)})
    for path in save_inside_tables(pc, args.inside_outdir, stats):
        rows.append({"stage": "06_inside_volcano_filtered", "point": pc.point, "kind": path.stem, "path": str(path)})
    for path in save_inside_figures(pc, figures_dir, args):
        rows.append({"stage": "06_inside_volcano_filtered", "point": pc.point, "kind": path.stem, "path": str(path)})

    summary_row = {
        "point": pc.point,
        **stats,
        "n_in_angular_grid_before_filter": pc.n_in_angular_grid_before_filter,
        "n_out_of_ecrit_grid": pc.n_out_of_ecrit_grid,
        "n_failed_ecrit": pc.n_failed_ecrit,
        "n_kept": pc.n_kept,
        "n_kept_without_ecrit_cell": pc.n_kept_without_ecrit_cell,
        "n_kept_in_ecrit_grid": pc.n_kept_in_ecrit_grid,
        "n_kept_inside_geometry": int(pc.grid_counts_inside.sum()),
        "n_event_cache_rows": cached_event_count(pc),
        "n_plot_counts": int(pc.plot_counts.sum()),
        "cache_path": str(cache_path),
    }
    return rows, summary_row


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build CABRIALES event cache and filtered maps in one SHW pass.")
    ap.add_argument("--points", nargs="+", default=ORDER, choices=ORDER)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shw", type=Path)
    src.add_argument("--kinematic-cache", type=Path,
                     help="Chunked cache directory produced by 04_build_kinematic_cache.py.")
    ap.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="auto")
    ap.add_argument("--shw-member", default=None)
    ap.add_argument("--ecrit-dir", required=True, type=Path)
    ap.add_argument("--cache-outdir", required=True, type=Path)
    ap.add_argument("--plot-outdir", required=True, type=Path)
    ap.add_argument("--inside-outdir", required=True, type=Path)
    ap.add_argument("--bins-theta", type=int, default=60)
    ap.add_argument("--bins-phi", type=int, default=40)
    ap.add_argument("--plot-theta-min", type=float, default=60.0)
    ap.add_argument("--plot-theta-max", type=float, default=90.0)
    ap.add_argument("--plot-phi-min", type=float, default=-50.0)
    ap.add_argument("--plot-phi-max", type=float, default=50.0)
    ap.add_argument("--tol-phi", type=float, default=0.51)
    ap.add_argument("--tol-theta", type=float, default=0.51)
    ap.add_argument("--treat-out-of-grid-as-clear", type=int, choices=[0, 1], default=1)
    ap.add_argument("--discard-upgoing", action="store_true")
    ap.add_argument("--no-wrap180", dest="wrap180", action="store_false")
    ap.set_defaults(wrap180=True)
    ap.add_argument("--progress-update-mb", type=float, default=32.0)
    ap.add_argument("--uncompressed-cache", action="store_true")
    ap.add_argument("--event-cache-source-mode", choices=["all", "inside"], default="all",
                    help="all caches every filtered event; inside caches only inside-volcano source events for event-MC inside mode.")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--inside-display-theta-min", type=float, default=0.0)
    ap.add_argument("--inside-display-theta-max", type=float, default=90.0)
    ap.add_argument("--inside-display-phi-min", type=float, default=-60.0)
    ap.add_argument("--inside-display-phi-max", type=float, default=60.0)
    ap.add_argument("--inside-display-step", type=float, default=0.5)
    ap.add_argument("--inside-vmax-percentile", type=float, default=99.0)
    ap.add_argument("--inside-show-zeros", action="store_true")
    return ap


def main(argv=None) -> int:
    apply_scientific_style()
    args = parser().parse_args(argv)
    for directory in (args.cache_outdir, args.plot_outdir, args.inside_outdir):
        directory.mkdir(parents=True, exist_ok=True)
    plot_dir = args.plot_outdir / "filtered"
    figures_dir = args.inside_outdir / "figures"

    th_edges = np.linspace(min(args.plot_theta_min, args.plot_theta_max), max(args.plot_theta_min, args.plot_theta_max), args.bins_theta + 1)
    ph_edges = np.linspace(min(args.plot_phi_min, args.plot_phi_max), max(args.plot_phi_min, args.plot_phi_max), args.bins_phi + 1)
    rows: list[dict[str, str]] = []
    summary_rows = []

    if args.kinematic_cache is not None and len(args.points) > 1:
        print("[INFO] Kinematic cache multi-point: processing one point at a time to keep RAM bounded.")
        for point in args.points:
            pc = load_point_cache(args.ecrit_dir, point, th_edges, ph_edges)
            print(f"  {point}: ecrit grid={pc.ecrit.shape}, inside cells={int(pc.inside_mask.sum())}")
            stats = process_kinematic_cache(args, {point: pc})
            point_rows, summary_row = save_point_products(pc, args, stats, plot_dir, figures_dir)
            rows.extend(point_rows)
            summary_rows.append(summary_row)
            print(
                f"  {point}: kept={summary_row['n_kept']} cache={summary_row['n_kept_in_ecrit_grid']} "
                f"inside={summary_row['n_kept_inside_geometry']} plot={summary_row['n_plot_counts']}"
            )
    else:
        points = {
            point: load_point_cache(args.ecrit_dir, point, th_edges, ph_edges)
            for point in args.points
        }
        for point, pc in points.items():
            print(f"  {point}: ecrit grid={pc.ecrit.shape}, inside cells={int(pc.inside_mask.sum())}")

        if args.kinematic_cache is not None:
            stats = process_kinematic_cache(args, points)
        else:
            stats = process_shw(args, points)

        for point in args.points:
            pc = points[point]
            point_rows, summary_row = save_point_products(pc, args, stats, plot_dir, figures_dir)
            rows.extend(point_rows)
            summary_rows.append(summary_row)

    summary_path = args.cache_outdir / "event_cache_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    rows.append({"stage": "04_event_cache", "point": "ALL", "kind": "summary", "path": str(summary_path)})

    manifest_path = args.inside_outdir / "inside_volcano_merged_manifest.csv"
    write_manifest(manifest_path, rows)

    print("\n[OK] Fast event-cache path finished")
    print(f"Summary: {summary_path}")
    for row in summary_rows:
        print(
            f"  {row['point']}: kept={row['n_kept']} cache={row['n_kept_in_ecrit_grid']} "
            f"inside={row['n_kept_inside_geometry']} plot={row['n_plot_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
