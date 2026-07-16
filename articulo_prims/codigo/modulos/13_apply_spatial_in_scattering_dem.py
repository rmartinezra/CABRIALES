#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial DEM in-scattering diagnostic for CABRIALES.

This module complements ``12_apply_in_scattering_background.py``.  Module 12 is
angular-only: it assigns a rock column to an angular ray from P1.  This module
adds the missing spatial degree of freedom by sampling a physical start
position on a configurable source surface around the DEM and transporting the
muon through the DEM volume.

The calculation is still a diagnostic, not a detector-rate prediction, unless a
real detector aperture is supplied and the source-surface normalization is
calibrated.  By default it answers:

    can a muon that crosses DEM rock outside the angular mask leave the rock
    with a direction that falls inside the volcano angular mask seen from P1?
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
for _path in (MODULE_DIR, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def load_sibling_module(alias: str, filename: str):
    path = MODULE_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude cargar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


INSCAT = load_sibling_module("cabriales_in_scattering_angular", "12_apply_in_scattering_background.py")
EVENT_CACHE = INSCAT.EVENT_CACHE
EVENT_MC = INSCAT.EVENT_MC
LENGTHS = INSCAT.LENGTHS
MUON_MASS_GEV = INSCAT.MUON_MASS_GEV

try:
    from plot_style import apply_scientific_style
    from empirical_kernel_io import mcs_momentum_scale
except ModuleNotFoundError:  # pragma: no cover
    def apply_scientific_style() -> None:
        plt.rcParams.update({"savefig.dpi": 260, "savefig.bbox": "tight"})
    from modulos.empirical_kernel_io import mcs_momentum_scale

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


@dataclass
class DemContext:
    point: str
    plat: float
    plon: float
    az_center_deg: float
    interp: object
    bbox_xy: tuple[float, float, float, float]
    z_plane_m: float
    z_min_m: float
    source_area_m2: float


@dataclass
class VolcanoSurfaceSampler:
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    theta_los_deg: np.ndarray
    phi_los_deg: np.ndarray
    edge_distance_m: np.ndarray
    height_fraction: np.ndarray
    grid_step_m: float
    horizontal_area_m2: float
    n_candidate_points: int
    n_inside_acceptance: int
    n_after_edge_guard: int
    n_after_height_guard: int


@dataclass
class TrackResult:
    touched_rock: bool
    survived: bool
    accepted: bool
    theta_final_deg: float
    phi_final_deg: float
    final_i: int | None
    final_j: int | None
    rock_length_m: float
    closest_approach_m: float
    used_nearest: int
    outside_domain: int
    no_support: int
    n_steps_rock: int
    final_in_domain: bool = True
    final_position_m: tuple[float, float, float] | None = None
    first_rock_position_m: tuple[float, float, float] | None = None
    first_rock_surface_height_m: float | None = None
    n_kernel_full_tail_steps: int = 0
    n_kernel_core_steps: int = 0
    n_step_deflections_gt_300_mrad: int = 0
    n_step_deflections_gt_500_mrad: int = 0
    n_step_deflections_gt_1000_mrad: int = 0
    n_kernel_extrapolated_low_energy_steps: int = 0
    n_kernel_extrapolated_high_energy_steps: int = 0
    n_kernel_extrapolated_hull_steps: int = 0
    n_kernel_momentum_scaled_steps: int = 0


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def meters_from_latlon(lat: float, lon: float, plat: float, plon: float) -> tuple[float, float]:
    dy = (float(lat) - float(plat)) * math.pi / 180.0 * LENGTHS.R_EARTH
    dx = (float(lon) - float(plon)) * math.pi / 180.0 * LENGTHS.R_EARTH * math.cos(math.radians(plat))
    return dx, dy


def latlon_from_meters(x_east_m: float, y_north_m: float, plat: float, plon: float) -> tuple[float, float]:
    dlat, dlon = LENGTHS.meters_to_deg(float(x_east_m), float(y_north_m), float(plat))
    return float(plat + dlat), float(plon + dlon)


def build_dem_context(point: str, hgt_dir: Path, margin_m: float, plane_margin_m: float) -> DemContext:
    hgt_paths = [hgt_dir / name for name in LENGTHS.HGT_ORDER]
    missing = [str(path) for path in hgt_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan HGT:\n  - " + "\n  - ".join(missing))

    mosaic, lats, lons = LENGTHS.mosaic_two_hgt([str(path) for path in hgt_paths])
    crop, crop_lats, crop_lons = LENGTHS.crop_dem(mosaic, lats, lons, LENGTHS.BBOX)
    interp = LENGTHS.make_interp(crop, crop_lats, crop_lons)

    plat, plon = LENGTHS.POINTS[point]
    az_center = LENGTHS.azimuth_deg(plat, plon, LENGTHS.SUMMIT[0], LENGTHS.SUMMIT[1])

    lat_min, lat_max, lon_min, lon_max = LENGTHS.BBOX
    corners = [
        meters_from_latlon(lat_min, lon_min, plat, plon),
        meters_from_latlon(lat_min, lon_max, plat, plon),
        meters_from_latlon(lat_max, lon_min, plat, plon),
        meters_from_latlon(lat_max, lon_max, plat, plon),
    ]
    xs = np.array([c[0] for c in corners], dtype=float)
    ys = np.array([c[1] for c in corners], dtype=float)
    x_min = float(xs.min() - margin_m)
    x_max = float(xs.max() + margin_m)
    y_min = float(ys.min() - margin_m)
    y_max = float(ys.max() + margin_m)

    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        raise RuntimeError("DEM crop sin alturas finitas.")
    z_plane = float(np.nanmax(finite) + plane_margin_m)
    z_min = float(np.nanmin(finite) - plane_margin_m)
    area = float((x_max - x_min) * (y_max - y_min))
    return DemContext(point, plat, plon, az_center, interp, (x_min, x_max, y_min, y_max), z_plane, z_min, area)


def unit_from_theta_phi(theta_deg: float, phi_rel_deg: float, az_center_deg: float) -> np.ndarray:
    theta = math.radians(float(theta_deg))
    az = math.radians(float(az_center_deg) + float(phi_rel_deg))
    return np.array([
        math.sin(theta) * math.sin(az),
        math.sin(theta) * math.cos(az),
        math.cos(theta),
    ], dtype=float)


def theta_phi_from_unit(u: np.ndarray, az_center_deg: float) -> tuple[float, float]:
    u = np.asarray(u, dtype=float)
    n = float(np.linalg.norm(u))
    if n <= 0.0 or not np.isfinite(n):
        return float("nan"), float("nan")
    u = u / n
    theta = math.degrees(math.acos(max(-1.0, min(1.0, float(u[2])))))
    az = math.degrees(math.atan2(float(u[0]), float(u[1])))
    phi_rel = INSCAT.wrap180(az - float(az_center_deg))
    return theta, phi_rel


def perpendicular_basis(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(v, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    e1 = np.cross(v, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, v)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def scatter_direction(v: np.ndarray, dtheta_rad: float, dphi_rad: float) -> np.ndarray:
    e1, e2 = perpendicular_basis(v)
    out = np.asarray(v, dtype=float) + float(dtheta_rad) * e1 + float(dphi_rad) * e2
    n = float(np.linalg.norm(out))
    if n <= 0.0 or not np.isfinite(n):
        return np.asarray(v, dtype=float)
    return out / n


def sample_kernel_component_mrad(rng, prediction, widths_mrad: np.ndarray, threshold: float) -> float:
    """Sample directly from a cached CDF when the full PDF is retained."""
    if threshold <= 0.0 and prediction.sampling_cdf.size:
        idx = int(np.searchsorted(prediction.sampling_cdf, rng.random(), side="right"))
        idx = min(idx, prediction.centers_mrad.size - 1)
        return float(prediction.centers_mrad[idx])
    return float(
        INSCAT.sample_delta_mrad(
            rng,
            prediction.centers_mrad,
            prediction.probability_per_bin,
            widths_mrad,
            threshold,
        )
    )


def inside_bbox(x: float, y: float, bbox_xy: tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = bbox_xy
    return x_min <= x <= x_max and y_min <= y <= y_max


def dem_height(ctx: DemContext, x: float, y: float) -> float:
    lat, lon = latlon_from_meters(x, y, ctx.plat, ctx.plon)
    return float(ctx.interp(np.array([lat]), np.array([lon]))[0])


def instrument_height_m(ctx: DemContext) -> float:
    z0 = float(ctx.interp(np.array([ctx.plat]), np.array([ctx.plon]))[0])
    return z0 + float(getattr(LENGTHS, "HEIGHT_OFFSET_M", 2.0))


def dem_crop_bbox_xy(ctx: DemContext) -> tuple[float, float, float, float]:
    lat_min, lat_max, lon_min, lon_max = LENGTHS.BBOX
    corners = [
        meters_from_latlon(lat_min, lon_min, ctx.plat, ctx.plon),
        meters_from_latlon(lat_min, lon_max, ctx.plat, ctx.plon),
        meters_from_latlon(lat_max, lon_min, ctx.plat, ctx.plon),
        meters_from_latlon(lat_max, lon_max, ctx.plat, ctx.plon),
    ]
    xs = np.array([c[0] for c in corners], dtype=float)
    ys = np.array([c[1] for c in corners], dtype=float)
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def edge_distance_xy(x: float, y: float, bbox_xy: tuple[float, float, float, float]) -> float:
    x_min, x_max, y_min, y_max = bbox_xy
    return float(min(float(x) - x_min, x_max - float(x), float(y) - y_min, y_max - float(y)))


def los_theta_phi_from_position(ctx: DemContext, x: float, y: float, z: float) -> tuple[float, float]:
    u = np.array([float(x), float(y), float(z) - instrument_height_m(ctx)], dtype=float)
    n = float(np.linalg.norm(u))
    if n <= 0.0 or not np.isfinite(n):
        return float("nan"), float("nan")
    return theta_phi_from_unit(u / n, ctx.az_center_deg)


def build_volcano_surface_sampler(ctx: DemContext, grid, args) -> VolcanoSurfaceSampler:
    step = float(args.volcano_surface_grid_step_m)
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("--volcano-surface-grid-step-m must be positive")
    edge_guard_m = max(0.0, float(args.volcano_surface_edge_guard_m))
    min_height_frac = float(args.volcano_surface_min_height_frac)
    if not (0.0 <= min_height_frac <= 1.0):
        raise ValueError("--volcano-surface-min-height-frac must be in [0, 1]")

    x_min, x_max, y_min, y_max = dem_crop_bbox_xy(ctx)
    xs = np.arange(x_min + 0.5 * step, x_max, step, dtype=float)
    ys = np.arange(y_min + 0.5 * step, y_max, step, dtype=float)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError("DEM crop demasiado pequeno para --volcano-surface-grid-step-m")

    XX, YY = np.meshgrid(xs, ys)
    dlat, dlon = LENGTHS.meters_to_deg(XX.ravel(), YY.ravel(), ctx.plat)
    lat = ctx.plat + dlat
    lon = ctx.plon + dlon
    z = np.asarray(ctx.interp(lat, lon), dtype=float)
    finite = np.isfinite(z)
    n_candidates = int(np.count_nonzero(finite))
    if n_candidates == 0:
        raise RuntimeError("No hay puntos finitos de DEM para construir la superficie volcanica.")

    x = XX.ravel().astype(float)
    y = YY.ravel().astype(float)
    z_inst = instrument_height_m(ctx)
    u = np.column_stack([x, y, z - z_inst])
    n = np.linalg.norm(u, axis=1)
    valid_los = finite & np.isfinite(n) & (n > 0.0)
    u[valid_los] /= n[valid_los, None]
    theta = np.full(x.shape, np.nan, dtype=float)
    phi = np.full(x.shape, np.nan, dtype=float)
    theta[valid_los] = np.degrees(np.arccos(np.clip(u[valid_los, 2], -1.0, 1.0)))
    az = np.degrees(np.arctan2(u[valid_los, 0], u[valid_los, 1]))
    phi[valid_los] = np.array([INSCAT.wrap180(v - ctx.az_center_deg) for v in az], dtype=float)

    inside_acceptance = np.zeros(x.shape, dtype=bool)
    inside_acceptance[valid_los] = INSCAT.vectorized_inside_acceptance(grid, theta[valid_los], phi[valid_los])
    n_inside_acceptance = int(np.count_nonzero(inside_acceptance))

    edge_dist = np.minimum.reduce([x - x_min, x_max - x, y - y_min, y_max - y]).astype(float)
    edge_ok = edge_dist >= edge_guard_m
    n_after_edge = int(np.count_nonzero(inside_acceptance & edge_ok))

    z_finite = z[finite]
    z_min = float(np.nanmin(z_finite))
    z_max = float(np.nanmax(z_finite))
    if z_max > z_min:
        height_frac = (z - z_min) / (z_max - z_min)
    else:
        height_frac = np.zeros_like(z, dtype=float)
    height_ok = height_frac >= min_height_frac
    n_after_height = int(np.count_nonzero(inside_acceptance & edge_ok & height_ok))

    target = inside_acceptance & edge_ok & height_ok & finite
    if not np.any(target):
        raise RuntimeError(
            "La superficie volcanica objetivo quedo vacia. Reduce --volcano-surface-edge-guard-m "
            "o --volcano-surface-min-height-frac."
        )

    area = float(np.count_nonzero(target) * step * step)
    return VolcanoSurfaceSampler(
        x_m=x[target].astype(float),
        y_m=y[target].astype(float),
        z_m=z[target].astype(float),
        theta_los_deg=theta[target].astype(float),
        phi_los_deg=phi[target].astype(float),
        edge_distance_m=edge_dist[target].astype(float),
        height_fraction=height_frac[target].astype(float),
        grid_step_m=step,
        horizontal_area_m2=area,
        n_candidate_points=n_candidates,
        n_inside_acceptance=n_inside_acceptance,
        n_after_edge_guard=n_after_edge,
        n_after_height_guard=n_after_height,
    )


def closest_approach_to_origin(pos: np.ndarray, direction: np.ndarray) -> float:
    v = np.asarray(direction, dtype=float)
    v = v / np.linalg.norm(v)
    p = np.asarray(pos, dtype=float)
    t = -float(np.dot(p, v))
    q = p + max(0.0, t) * v
    return float(np.linalg.norm(q))


ENTRY_FACES = ("top", "west", "east", "south", "north")


def parse_entry_face_importance(spec: str | None) -> dict[str, float]:
    factors = {face: 1.0 for face in ENTRY_FACES}
    if spec is None or not str(spec).strip():
        return factors
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Formato invalido en --entry-face-importance: {item!r}. Usa face:factor.")
        face, value = item.split(":", 1)
        face = face.strip().lower()
        if face not in factors:
            opts = ", ".join(ENTRY_FACES)
            raise ValueError(f"Cara invalida en --entry-face-importance: {face!r}. Opciones: {opts}")
        factor = float(value)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("--entry-face-importance requiere factores positivos para mantener estimador no sesgado.")
        factors[face] = factor
    return factors


def sample_entry_position(
    ctx: DemContext,
    direction: np.ndarray,
    rng: np.random.Generator,
    args,
    volcano_sampler: VolcanoSurfaceSampler | None = None,
) -> tuple[np.ndarray | None, str | None, float]:
    x_min, x_max, y_min, y_max = ctx.bbox_xy
    z_min = float(ctx.z_min_m)
    z_max = float(ctx.z_plane_m)
    if args.source_surface == "top-plane":
        return (
            np.array([float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)), z_max], dtype=float),
            "top",
            1.0,
        )
    if args.source_surface == "volcano-surface":
        if volcano_sampler is None or volcano_sampler.x_m.size == 0:
            return None, None, 1.0
        idx = int(rng.integers(0, volcano_sampler.x_m.size))
        offset = float(args.volcano_surface_start_offset_m)
        pos = np.array([
            float(volcano_sampler.x_m[idx]),
            float(volcano_sampler.y_m[idx]),
            float(volcano_sampler.z_m[idx]) + max(0.0, offset),
        ], dtype=float)
        return pos, "volcano_surface", 1.0

    dx = max(0.0, float(x_max - x_min))
    dy = max(0.0, float(y_max - y_min))
    dz = max(0.0, float(z_max - z_min))
    v = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0.0 or not np.isfinite(n):
        return None, None, 1.0
    v = v / n

    faces: list[tuple[str, float]] = []
    if v[2] < 0.0:
        faces.append(("top", -float(v[2]) * dx * dy))
    if v[0] > 0.0:
        faces.append(("west", float(v[0]) * dy * dz))
    elif v[0] < 0.0:
        faces.append(("east", -float(v[0]) * dy * dz))
    if v[1] > 0.0:
        faces.append(("south", float(v[1]) * dx * dz))
    elif v[1] < 0.0:
        faces.append(("north", -float(v[1]) * dx * dz))

    weights = np.array([w for _, w in faces], dtype=float)
    weights[~np.isfinite(weights)] = 0.0
    weights[weights < 0.0] = 0.0
    total = float(weights.sum())
    if total <= 0.0:
        return None, None, 1.0

    factors = np.array([args.entry_face_importance_weights.get(face, 1.0) for face, _ in faces], dtype=float)
    biased_weights = weights * factors
    biased_total = float(biased_weights.sum())
    if biased_total <= 0.0 or not np.isfinite(biased_total):
        return None, None, 1.0
    physical_prob = weights / total
    biased_prob = biased_weights / biased_total
    idx = int(np.searchsorted(np.cumsum(biased_prob), rng.random(), side="right"))
    idx = min(idx, len(faces) - 1)
    face = faces[idx][0]
    importance_weight = float(physical_prob[idx] / biased_prob[idx])

    if face == "top":
        pos = np.array([float(rng.uniform(x_min, x_max)), float(rng.uniform(y_min, y_max)), z_max], dtype=float)
    elif face == "west":
        pos = np.array([x_min, float(rng.uniform(y_min, y_max)), float(rng.uniform(z_min, z_max))], dtype=float)
    elif face == "east":
        pos = np.array([x_max, float(rng.uniform(y_min, y_max)), float(rng.uniform(z_min, z_max))], dtype=float)
    elif face == "south":
        pos = np.array([float(rng.uniform(x_min, x_max)), y_min, float(rng.uniform(z_min, z_max))], dtype=float)
    else:
        pos = np.array([float(rng.uniform(x_min, x_max)), y_max, float(rng.uniform(z_min, z_max))], dtype=float)
    return pos, face, importance_weight


def propagate_spatial_track(
    theta_initial_deg: float,
    phi_initial_deg: float,
    kinetic0_GeV: float,
    start_pos: np.ndarray,
    ctx: DemContext,
    grid,
    model,
    energy_loss,
    rng: np.random.Generator,
    args,
) -> TrackResult:
    # Direction stored in CABRIALES is the apparent line-of-sight direction.
    # The physical muon travels in the opposite direction.
    apparent_initial = unit_from_theta_phi(theta_initial_deg, phi_initial_deg, ctx.az_center_deg)
    v = -apparent_initial
    n = float(np.linalg.norm(v))
    if n <= 0.0 or not np.isfinite(n):
        return TrackResult(False, False, False, np.nan, np.nan, None, None, 0.0, np.inf, 0, 0, 0, 0)
    v /= n
    if v[2] >= -1e-6:
        return TrackResult(False, False, False, np.nan, np.nan, None, None, 0.0, np.inf, 0, 0, 0, 0)

    pos = np.asarray(start_pos, dtype=float).copy()
    initial_prev_pos: np.ndarray | None = None
    initial_prev_gap: float | None = None
    if args.source_surface == "volcano-surface":
        start_topo = dem_height(ctx, float(pos[0]), float(pos[1]))
        start_gap = float(pos[2]) - float(start_topo) if np.isfinite(start_topo) else float("nan")
        probe = pos + float(args.volcano_surface_entry_check_m) * v
        probe_topo = dem_height(ctx, float(probe[0]), float(probe[1]))
        probe_gap = float(probe[2]) - float(probe_topo) if np.isfinite(probe_topo) else float("nan")
        if (not np.isfinite(probe_gap)) or probe_gap >= 0.0:
            return TrackResult(False, False, False, np.nan, np.nan, None, None, 0.0, np.inf, 0, 0, 0, 0)
        # Keep the entry probe only as a direction check. Start immediately
        # inside the interpolated surface so a 10 m transport does not omit its
        # first measured kernel slab.
        if np.isfinite(start_gap) and start_gap >= 0.0:
            denom = float(start_gap - probe_gap)
            t = float(start_gap / denom) if denom > 0.0 else 0.0
            t = min(1.0, max(0.0, t))
            crossing = pos + t * (probe - pos)
        else:
            crossing = pos.copy()
        crossing_topo = dem_height(ctx, float(crossing[0]), float(crossing[1]))
        if np.isfinite(crossing_topo):
            crossing[2] = float(crossing_topo)
        initial_prev_pos = crossing.copy()
        initial_prev_gap = 0.0
        pos = crossing + 1e-3 * v
    kinetic = float(kinetic0_GeV)
    touched = False
    first_rock_pos: np.ndarray | None = None
    first_rock_topo: float | None = None
    prev_pos: np.ndarray | None = initial_prev_pos
    prev_gap: float | None = initial_prev_gap
    rock_length = 0.0
    used_nearest = 0
    outside_domain = 0
    no_support = 0
    n_steps_rock = 0
    n_kernel_full_tail_steps = 0
    n_kernel_core_steps = 0
    n_step_deflections_gt_300_mrad = 0
    n_step_deflections_gt_500_mrad = 0
    n_step_deflections_gt_1000_mrad = 0
    n_kernel_extrapolated_low_energy_steps = 0
    n_kernel_extrapolated_high_energy_steps = 0
    n_kernel_extrapolated_hull_steps = 0
    n_kernel_momentum_scaled_steps = 0
    rad_to_mrad = 1000.0

    core_energy_bounds = None
    if getattr(model, "tail_aware", None) is not None:
        core_energy_bounds = model.tail_aware.core_energy_bounds(float(args.ray_step_m))

    max_steps = int(max(1, math.ceil(float(args.max_track_m) / float(args.ray_step_m))))
    for _ in range(max_steps):
        if not inside_bbox(float(pos[0]), float(pos[1]), ctx.bbox_xy) or pos[2] < ctx.z_min_m:
            break
        topo = dem_height(ctx, float(pos[0]), float(pos[1]))
        gap = float(pos[2]) - float(topo) if np.isfinite(topo) else float("nan")
        inside_rock = np.isfinite(topo) and gap < 0.0
        step = float(args.ray_step_m)
        transport_direction = v.copy()
        if inside_rock:
            if not touched:
                first_rock_pos = pos.copy()
                first_rock_topo = float(topo)
                if prev_pos is not None and prev_gap is not None and np.isfinite(prev_gap) and prev_gap >= 0.0:
                    denom = float(prev_gap - gap)
                    t = float(prev_gap / denom) if denom != 0.0 else 1.0
                    t = min(1.0, max(0.0, t))
                    first_rock_pos = prev_pos + t * (pos - prev_pos)
                    topo_c = dem_height(ctx, float(first_rock_pos[0]), float(first_rock_pos[1]))
                    if np.isfinite(topo_c):
                        first_rock_topo = float(topo_c)
                        first_rock_pos[2] = float(topo_c)
            touched = True
            n_steps_rock += 1
            rock_length += step
            if kinetic <= 0.0 or not np.isfinite(kinetic):
                return TrackResult(
                    touched, False, False, np.nan, np.nan, None, None, rock_length, np.inf,
                    used_nearest, outside_domain, no_support, n_steps_rock,
                    n_kernel_full_tail_steps=n_kernel_full_tail_steps,
                    n_kernel_core_steps=n_kernel_core_steps,
                    n_step_deflections_gt_300_mrad=n_step_deflections_gt_300_mrad,
                    n_step_deflections_gt_500_mrad=n_step_deflections_gt_500_mrad,
                    n_step_deflections_gt_1000_mrad=n_step_deflections_gt_1000_mrad,
                    n_kernel_extrapolated_low_energy_steps=n_kernel_extrapolated_low_energy_steps,
                    n_kernel_extrapolated_high_energy_steps=n_kernel_extrapolated_high_energy_steps,
                    n_kernel_extrapolated_hull_steps=n_kernel_extrapolated_hull_steps,
                    n_kernel_momentum_scaled_steps=n_kernel_momentum_scaled_steps,
                )
            kinetic_next = energy_loss.advance(kinetic, step)
            if kinetic_next is None or kinetic_next <= 0.0 or not np.isfinite(kinetic_next):
                return TrackResult(
                    touched, False, False, np.nan, np.nan, None, None, rock_length, np.inf,
                    used_nearest, outside_domain, no_support, n_steps_rock,
                    n_kernel_full_tail_steps=n_kernel_full_tail_steps,
                    n_kernel_core_steps=n_kernel_core_steps,
                    n_step_deflections_gt_300_mrad=n_step_deflections_gt_300_mrad,
                    n_step_deflections_gt_500_mrad=n_step_deflections_gt_500_mrad,
                    n_step_deflections_gt_1000_mrad=n_step_deflections_gt_1000_mrad,
                    n_kernel_extrapolated_low_energy_steps=n_kernel_extrapolated_low_energy_steps,
                    n_kernel_extrapolated_high_energy_steps=n_kernel_extrapolated_high_energy_steps,
                    n_kernel_extrapolated_hull_steps=n_kernel_extrapolated_hull_steps,
                    n_kernel_momentum_scaled_steps=n_kernel_momentum_scaled_steps,
                )
            if not args.disable_scattering:
                pred = model.predict_kernel(step, kinetic)
                if pred.used_nearest_fallback:
                    used_nearest += 1
                if pred.outside_domain:
                    outside_domain += 1
                    if core_energy_bounds is None:
                        n_kernel_extrapolated_hull_steps += 1
                    elif kinetic < core_energy_bounds[0]:
                        n_kernel_extrapolated_low_energy_steps += 1
                    elif kinetic > core_energy_bounds[1]:
                        n_kernel_extrapolated_high_energy_steps += 1
                    else:
                        n_kernel_extrapolated_hull_steps += 1
                if not pred.valid:
                    no_support += 1
                else:
                    if pred.tail_policy == "body_quantile_tail_histogram_linear":
                        n_kernel_full_tail_steps += 1
                    else:
                        n_kernel_core_steps += 1
                    a_mrad = sample_kernel_component_mrad(rng, pred, model.widths_mrad, args.kernel_threshold)
                    b_mrad = sample_kernel_component_mrad(rng, pred, model.widths_mrad, args.kernel_threshold)
                    extrapolation_scale = 1.0
                    if (
                        pred.outside_domain
                        and core_energy_bounds is not None
                        and args.kernel_energy_extrapolation == "momentum-scale"
                    ):
                        if kinetic < core_energy_bounds[0]:
                            reference_energy = core_energy_bounds[0]
                        elif kinetic > core_energy_bounds[1]:
                            reference_energy = core_energy_bounds[1]
                        else:
                            reference_energy = None
                        if reference_energy is not None:
                            extrapolation_scale = mcs_momentum_scale(kinetic, reference_energy)
                            n_kernel_momentum_scaled_steps += 1
                    a_mrad *= extrapolation_scale
                    b_mrad *= extrapolation_scale
                    step_deflection_mrad = math.hypot(a_mrad, b_mrad) * float(args.kernel_scale)
                    n_step_deflections_gt_300_mrad += int(step_deflection_mrad > 300.0)
                    n_step_deflections_gt_500_mrad += int(step_deflection_mrad > 500.0)
                    n_step_deflections_gt_1000_mrad += int(step_deflection_mrad > 1000.0)
                    a = a_mrad * float(args.kernel_scale) / rad_to_mrad
                    b = b_mrad * float(args.kernel_scale) / rad_to_mrad
                    v = scatter_direction(v, a, b)
            kinetic = kinetic_next
        prev_pos = pos.copy()
        prev_gap = gap
        # Condensed-history step: move through the slab along the incoming
        # direction and apply its angular kick at the segment endpoint.
        pos = pos + step * transport_direction

    if not touched:
        return TrackResult(False, False, False, np.nan, np.nan, None, None, 0.0, np.inf, used_nearest, outside_domain, no_support, n_steps_rock)

    apparent_final = -v
    theta_final, phi_final = theta_phi_from_unit(apparent_final, ctx.az_center_deg)
    final_in_domain = bool(INSCAT.source_in_domain(theta_final, phi_final, args))
    dest = INSCAT.find_cell(grid, theta_final, phi_final)
    accepted = False
    final_i = final_j = None
    if dest is not None:
        final_i, final_j = dest
        accepted = final_in_domain and bool(grid.inside_mask[final_i, final_j])

    closest = closest_approach_to_origin(pos, v)
    if args.observer_radius_m is not None and args.observer_radius_m > 0.0:
        accepted = accepted and (closest <= float(args.observer_radius_m))
    return TrackResult(
        touched,
        True,
        accepted,
        theta_final,
        phi_final,
        final_i,
        final_j,
        rock_length,
        closest,
        used_nearest,
        outside_domain,
        no_support,
        n_steps_rock,
        final_in_domain,
        (float(pos[0]), float(pos[1]), float(pos[2])),
        None if first_rock_pos is None else (float(first_rock_pos[0]), float(first_rock_pos[1]), float(first_rock_pos[2])),
        None if first_rock_topo is None else float(first_rock_topo),
        n_kernel_full_tail_steps,
        n_kernel_core_steps,
        n_step_deflections_gt_300_mrad,
        n_step_deflections_gt_500_mrad,
        n_step_deflections_gt_1000_mrad,
        n_kernel_extrapolated_low_energy_steps,
        n_kernel_extrapolated_high_energy_steps,
        n_kernel_extrapolated_hull_steps,
        n_kernel_momentum_scaled_steps,
    )


def iter_kinematic_events(args, grid, stats=None):
    cache = Path(args.kinematic_cache)
    total = None
    manifest = cache / "manifest.json"
    if manifest.exists():
        try:
            total = int(json.loads(manifest.read_text(encoding="utf-8")).get("n_events", 0))
        except Exception:
            total = None
    if args.head and total is not None:
        total = min(total, int(args.head))

    sample_probability = float(args.sample_probability)
    event_weight = 1.0 / sample_probability
    sample_rng = np.random.default_rng(int(args.seed) + 1000003 + 7919 * int(args.chunk_index))
    progress_total = None if int(args.chunk_count) > 1 else total
    pbar = tqdm(total=progress_total, unit="event", unit_scale=True, desc="spatial in-scattering", disable=args.no_progress)
    n_seen_global = 0
    for chunk_ordinal, (_, chunk) in enumerate(EVENT_CACHE.iter_kinematic_cache(cache)):
        theta = np.asarray(chunk["theta_deg"], dtype=float)
        chunk_size = int(theta.size)
        n_take = chunk_size
        if args.head:
            remaining = max(0, int(args.head) - n_seen_global)
            n_take = min(n_take, remaining)
        if n_take <= 0:
            pbar.close()
            return

        chunk_start_index = n_seen_global
        process_chunk = (chunk_ordinal % int(args.chunk_count)) == int(args.chunk_index)
        n_seen_global += n_take
        if stats is not None:
            stats["n_chunks_seen"] += 1
        if not process_chunk:
            if args.head and n_seen_global >= int(args.head):
                pbar.close()
                return
            continue
        if stats is not None:
            stats["n_chunks_processed"] += 1

        phi_abs = np.asarray(chunk["phi_abs_deg"], dtype=float)[:n_take]
        total_e = np.asarray(chunk["total_E_GeV"], dtype=float)[:n_take]
        theta = theta[:n_take]
        pz_positive = chunk.get("pz_positive")
        if pz_positive is None:
            pz_positive = np.zeros(theta.shape, dtype=np.uint8)
        else:
            pz_positive = np.asarray(pz_positive, dtype=np.uint8)[:n_take]
        phi_rel = (phi_abs - grid.phi0) % 360.0
        phi_rel = np.where(phi_rel > 180.0, phi_rel - 360.0, phi_rel)

        if sample_probability < 1.0:
            keep = sample_rng.random(n_take) < sample_probability
        else:
            keep = np.ones(n_take, dtype=bool)

        pbar.update(n_take)
        if stats is not None:
            stats["n_flux_events_read"] += n_take
            stats["n_flux_events_sampled"] += int(np.count_nonzero(keep))

        for idx in np.flatnonzero(keep):
            yield (
                float(theta[idx]),
                float(phi_rel[idx]),
                float(total_e[idx]),
                bool(pz_positive[idx]),
                event_weight,
                int(chunk_start_index + idx),
            )
        if args.head and n_seen_global >= int(args.head):
            pbar.close()
            return
    pbar.close()


def grid_distance_to_acceptance_deg(grid, distance_deg: np.ndarray, theta_deg: float, phi_deg: float) -> float:
    cell = INSCAT.find_cell(grid, theta_deg, phi_deg)
    if cell is None:
        return float("inf")
    i, j = cell
    value = float(distance_deg[i, j])
    return value if np.isfinite(value) else float("inf")


def plot_map(path: Path, theta_edges, phi_edges, values, title: str, cbar_label: str, log: bool = True) -> None:
    vals = values[np.isfinite(values) & (values > 0)]
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    kwargs = {"shading": "flat", "cmap": "magma"}
    if log:
        kwargs["norm"] = LogNorm(vmin=1.0, vmax=max(1.0, float(vals.max())))
    im = ax.pcolormesh(phi_edges, theta_edges, np.where(values > 0, values, np.nan), **kwargs)
    ax.set_xlim(float(phi_edges[0]), float(phi_edges[-1]))
    ax.set_ylim(float(theta_edges[-1]), float(theta_edges[0]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label(cbar_label)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Spatial DEM diagnostic for external -> volcano-mask in-scattering.")
    ap.add_argument("--kinematic-cache", required=True, type=Path)
    ap.add_argument(
        "--kernel-npz",
        type=Path,
        default=MODULE_DIR / "hybrid_empirical_kernel_library.npz",
        help="Empirical MCS kernel (default: bundled hybrid full-tail model).",
    )
    ap.add_argument("--acceptance-map", required=True, type=Path)
    ap.add_argument("--length-map", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--point", choices=["P1", "P2", "P4", "P5"], default=None)
    ap.add_argument("--acceptance-mask-col", default=None)
    ap.add_argument("--acceptance-mask-min", type=float, default=0.0)
    ap.add_argument("--hgt-dir", default="auto")
    ap.add_argument("--range-file", type=Path, default=None)
    ap.add_argument("--head", type=int, default=10000)
    ap.add_argument("--chunk-index", type=int, default=0, help="Worker index for chunk-parallel cache processing.")
    ap.add_argument("--chunk-count", type=int, default=1, help="Number of chunk-parallel workers.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ray-step-m", type=float, default=10.0,
                    help="Paso geometrico y espesor incremental del kernel dentro de roca.")
    ap.add_argument("--max-track-m", type=float, default=9000.0)
    ap.add_argument("--source-plane-margin-m", type=float, default=250.0)
    ap.add_argument("--source-plane-height-margin-m", type=float, default=500.0)
    ap.add_argument("--source-surface", choices=["entry-box", "top-plane", "volcano-surface"], default="entry-box",
                    help="Superficie de muestreo espacial: caja de entrada DEM, plano superior legacy o superficie DEM dentro de la mascara del volcan.")
    ap.add_argument("--volcano-surface-grid-step-m", type=float, default=50.0,
                    help="Paso de grilla horizontal para construir la superficie volcanica objetivo.")
    ap.add_argument("--volcano-surface-edge-guard-m", type=float, default=500.0,
                    help="Margen minimo contra el borde del recorte DEM para puntos de superficie volcanica.")
    ap.add_argument("--volcano-surface-min-height-frac", type=float, default=0.0,
                    help="Altura relativa minima [0,1] dentro del recorte DEM para puntos de superficie volcanica.")
    ap.add_argument("--volcano-surface-start-offset-m", type=float, default=1.0,
                    help="Offset vertical sobre la superficie DEM antes de iniciar el transporte.")
    ap.add_argument("--volcano-surface-entry-check-m", type=float, default=10.0,
                    help="Distancia de prueba para exigir que el muon entre en roca desde el punto de superficie volcanica.")
    ap.add_argument("--entry-face-importance", default="",
                    help="Importance sampling por cara, e.g. south:4,west:4,top:1,east:0.5,north:0.5. Pesos positivos conservan estimador no sesgado.")
    ap.add_argument("--position-samples-per-muon", type=int, default=1)
    ap.add_argument("--sample-probability", type=float, default=1.0,
                    help="Probabilidad de muestrear cada evento del cache; los eventos conservados pesan 1/p.")
    ap.add_argument("--min-survival-rock-m", type=float, default=10.0,
                    help="Prefiltro CSDA de rango inicial; no impone una longitud minima a las trayectorias aceptadas.")
    ap.add_argument("--observer-radius-m", type=float, default=0.0,
                    help="Si >0 exige que la trayectoria final pase a esta distancia de P1. 0 desactiva chequeo.")
    ap.add_argument("--theta-min-deg", type=float, default=0.0)
    ap.add_argument("--theta-max-deg", type=float, default=180.0)
    ap.add_argument("--max-angular-margin-deg", type=float, default=None,
                    help="Si se define, solo propaga direcciones externas a esta distancia angular de la mascara aceptada.")
    ap.add_argument("--phi-min-deg", type=float, default=-180.0)
    ap.add_argument("--phi-max-deg", type=float, default=180.0)
    ap.add_argument("--discard-upgoing", action="store_true")
    ap.add_argument("--rho", type=float, default=2.65)
    ap.add_argument("--interp-method", choices=["tail-aware", "linear", "rbf_linear", "nearest"], default="tail-aware")
    ap.add_argument("--rbf-smoothing", type=float, default=0.0)
    ap.add_argument("--kernel-threshold", type=float, default=0.0)
    ap.add_argument("--kernel-scale", type=float, default=1.0)
    ap.add_argument(
        "--kernel-energy-extrapolation",
        choices=["momentum-scale", "nearest"],
        default="momentum-scale",
        help="Fuera del rango energetico reescala el angulo empirico con 1/(beta*p), o usa el vecino sin correccion.",
    )
    ap.add_argument("--disable-scattering", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    return ap


def main(argv=None) -> int:
    apply_scientific_style()
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.ray_step_m < 1.0 or args.max_track_m <= 0.0:
        raise ValueError("--ray-step-m must be >= 1 m and --max-track-m must be positive")
    if int(args.chunk_count) <= 0:
        raise ValueError("--chunk-count must be positive")
    if int(args.chunk_index) < 0 or int(args.chunk_index) >= int(args.chunk_count):
        raise ValueError("--chunk-index must satisfy 0 <= index < chunk-count")
    if args.position_samples_per_muon <= 0:
        raise ValueError("--position-samples-per-muon must be positive")
    if not (0.0 < float(args.sample_probability) <= 1.0):
        raise ValueError("--sample-probability must be in (0, 1]")
    if float(args.min_survival_rock_m) < 0.0:
        raise ValueError("--min-survival-rock-m must be non-negative")
    if float(args.volcano_surface_grid_step_m) <= 0.0:
        raise ValueError("--volcano-surface-grid-step-m must be positive")
    if float(args.volcano_surface_edge_guard_m) < 0.0:
        raise ValueError("--volcano-surface-edge-guard-m must be non-negative")
    if float(args.volcano_surface_start_offset_m) < 0.0:
        raise ValueError("--volcano-surface-start-offset-m must be non-negative")
    if float(args.volcano_surface_entry_check_m) <= 0.0:
        raise ValueError("--volcano-surface-entry-check-m must be positive")
    if not (0.0 <= float(args.volcano_surface_min_height_frac) <= 1.0):
        raise ValueError("--volcano-surface-min-height-frac must be in [0, 1]")
    if args.max_angular_margin_deg is not None and float(args.max_angular_margin_deg) < 0.0:
        raise ValueError("--max-angular-margin-deg must be non-negative")
    args.entry_face_importance_weights = parse_entry_face_importance(args.entry_face_importance)

    point, grid, _ = INSCAT.load_grid_and_acceptance(args)
    hgt_dir = INSCAT.find_hgt_dir(args.hgt_dir, args.length_map)
    if hgt_dir is None:
        raise FileNotFoundError("No encontré HGT. Usa --hgt-dir.")
    ctx = build_dem_context(point, hgt_dir, args.source_plane_margin_m, args.source_plane_height_margin_m)
    dem_bbox = dem_crop_bbox_xy(ctx)
    volcano_sampler: VolcanoSurfaceSampler | None = None
    volcano_surface_target_path: Path | None = None
    if args.source_surface == "volcano-surface":
        volcano_sampler = build_volcano_surface_sampler(ctx, grid, args)
        volcano_surface_target_path = args.output_dir / "volcano_surface_target_points.csv"
        pd.DataFrame({
            "x_m": volcano_sampler.x_m,
            "y_m": volcano_sampler.y_m,
            "z_m": volcano_sampler.z_m,
            "theta_los_deg": volcano_sampler.theta_los_deg,
            "phi_los_deg": volcano_sampler.phi_los_deg,
            "edge_distance_m": volcano_sampler.edge_distance_m,
            "height_fraction": volcano_sampler.height_fraction,
        }).to_csv(volcano_surface_target_path, index=False)
    range_file = INSCAT.find_range_file(args.range_file, args.length_map)
    if range_file is None:
        raise FileNotFoundError("No encontré data_rock.dat/muon_range_table.csv")
    energy_loss, range_table_path = INSCAT.load_energy_loss(range_file, args.output_dir, args.rho)
    model = EVENT_MC.EmpiricalKernelModel(args.kernel_npz, args.interp_method, args.rbf_smoothing)
    kernel_step_is_native = bool(
        model.tail_aware is not None
        and np.any(np.isclose(model.tail_aware.transport_L_nodes_m, float(args.ray_step_m)))
    )
    print(
        f"[KERNEL] family={model.kernel_family} method={args.interp_method} "
        f"bins={len(model.centers_mrad)} support_mrad="
        f"[{model.edges_mrad[0]:g}, {model.edges_mrad[-1]:g}] "
        f"threshold={args.kernel_threshold:g}"
    )
    distance_deg_grid = INSCAT.angular_distance_to_acceptance(grid, grid.inside_mask)

    source_grid = INSCAT.build_source_grid(args, grid)
    source_counts = np.zeros((len(source_grid["theta"]), len(source_grid["phi"])), dtype=float)
    final_counts = np.zeros_like(grid.L_grid, dtype=float)
    rock_lengths: list[float] = []
    closest_distances: list[float] = []
    accepted_rows: list[dict[str, float | int | str]] = []

    stats = {
        "n_flux_events_read": 0,
        "n_flux_events_sampled": 0,
        "n_chunks_seen": 0,
        "n_chunks_processed": 0,
        "n_position_samples": 0,
        "n_prefilter_insufficient_range": 0,
        "n_invalid_initial_energy": 0,
        "n_entry_face_top": 0,
        "n_entry_face_west": 0,
        "n_entry_face_east": 0,
        "n_entry_face_south": 0,
        "n_entry_face_north": 0,
        "n_entry_face_volcano_surface": 0,
        "n_entry_face_none": 0,
        "n_discarded_upgoing": 0,
        "n_outside_source_domain": 0,
        "n_initial_inside_acceptance_skipped": 0,
        "n_outside_angular_margin": 0,
        "n_non_downward_for_source_plane": 0,
        "n_tracks_without_dem_rock": 0,
        "n_tracks_touched_rock": 0,
        "n_tracks_not_survived": 0,
        "n_tracks_survived": 0,
        "n_tracks_final_outside_domain": 0,
        "n_tracks_final_inside_acceptance": 0,
        "n_accepted_source_map_misses": 0,
        "n_kernel_nearest_fallback_steps": 0,
        "n_kernel_outside_domain_steps": 0,
        "n_kernel_no_support_steps": 0,
        "n_kernel_full_tail_steps": 0,
        "n_kernel_core_steps": 0,
        "n_kernel_extrapolated_low_energy_steps": 0,
        "n_kernel_extrapolated_high_energy_steps": 0,
        "n_kernel_extrapolated_hull_steps": 0,
        "n_kernel_momentum_scaled_steps": 0,
        "n_step_deflections_gt_300_mrad": 0,
        "n_step_deflections_gt_500_mrad": 0,
        "n_step_deflections_gt_1000_mrad": 0,
    }

    x_min, x_max, y_min, y_max = ctx.bbox_xy
    print("[INFO] Spatial DEM in-scattering diagnostic")
    print("[INFO] spatial_positions_sampled=true detector_intersection_checked=" + str(args.observer_radius_m > 0.0).lower())
    print(f"[INFO] point={point} source_plane_area_m2={ctx.source_area_m2:.6g} z_plane_m={ctx.z_plane_m:.3f}")
    print(f"[INFO] sample_probability={args.sample_probability:g} event_weight={1.0 / float(args.sample_probability):.6g}")
    print(f"[INFO] chunk_index={args.chunk_index} chunk_count={args.chunk_count}")
    print(f"[INFO] min_survival_rock_m={args.min_survival_rock_m:g}")
    print(f"[INFO] ray_step_m={args.ray_step_m:g} kernel_step_is_native={str(kernel_step_is_native).lower()}")
    print(f"[INFO] max_angular_margin_deg={args.max_angular_margin_deg}")
    print(f"[INFO] source_surface={args.source_surface}")
    if volcano_sampler is not None:
        print(
            "[INFO] volcano_surface_points="
            f"{volcano_sampler.x_m.size} area_horizontal_m2={volcano_sampler.horizontal_area_m2:.6g} "
            f"grid_step_m={volcano_sampler.grid_step_m:g} edge_guard_m={args.volcano_surface_edge_guard_m:g} "
            f"min_height_frac={args.volcano_surface_min_height_frac:g}"
        )
    print(f"[INFO] entry_face_importance={args.entry_face_importance_weights}")

    min_survival_X_gcm2 = float(energy_loss.rho_g_cm3) * float(args.min_survival_rock_m) * 100.0
    for theta_i, phi_i, total_e, pz_positive, event_weight, event_index in iter_kinematic_events(args, grid, stats):
        if args.discard_upgoing and pz_positive:
            stats["n_discarded_upgoing"] += 1
            continue
        if not INSCAT.source_in_domain(theta_i, phi_i, args):
            stats["n_outside_source_domain"] += 1
            continue
        src_cell = INSCAT.find_cell(grid, theta_i, phi_i)
        if src_cell is not None and bool(grid.inside_mask[src_cell[0], src_cell[1]]):
            stats["n_initial_inside_acceptance_skipped"] += 1
            continue
        initial_distance_to_acceptance = grid_distance_to_acceptance_deg(grid, distance_deg_grid, theta_i, phi_i)
        if args.max_angular_margin_deg is not None and initial_distance_to_acceptance > float(args.max_angular_margin_deg):
            stats["n_outside_angular_margin"] += 1
            continue
        kinetic0 = float(total_e - MUON_MASS_GEV)
        if kinetic0 <= 0.0 or not np.isfinite(kinetic0):
            continue
        if min_survival_X_gcm2 > 0.0 and energy_loss.range_for_kinetic(kinetic0) <= min_survival_X_gcm2:
            stats["n_prefilter_insufficient_range"] += 1
            continue
        sample_weight = float(event_weight) / float(args.position_samples_per_muon)
        for position_sample_index in range(args.position_samples_per_muon):
            stats["n_position_samples"] += 1
            # A per-event stream keeps the same physical muon reproducible even
            # when an earlier event exits sooner or a transport check changes.
            rng = np.random.default_rng(np.random.SeedSequence([
                int(args.seed),
                int(event_index & 0xFFFFFFFF),
                int(event_index >> 32),
                int(position_sample_index),
            ]))
            entry_direction = -unit_from_theta_phi(theta_i, phi_i, ctx.az_center_deg)
            start_pos, entry_face, face_importance_weight = sample_entry_position(ctx, entry_direction, rng, args, volcano_sampler)
            if start_pos is None or entry_face is None:
                stats["n_entry_face_none"] += 1
                continue
            stats[f"n_entry_face_{entry_face}"] += 1
            track_weight = sample_weight * float(face_importance_weight)
            result = propagate_spatial_track(theta_i, phi_i, kinetic0, start_pos, ctx, grid, model, energy_loss, rng, args)
            stats["n_kernel_nearest_fallback_steps"] += result.used_nearest
            stats["n_kernel_outside_domain_steps"] += result.outside_domain
            stats["n_kernel_no_support_steps"] += result.no_support
            stats["n_kernel_full_tail_steps"] += result.n_kernel_full_tail_steps
            stats["n_kernel_core_steps"] += result.n_kernel_core_steps
            stats["n_kernel_extrapolated_low_energy_steps"] += result.n_kernel_extrapolated_low_energy_steps
            stats["n_kernel_extrapolated_high_energy_steps"] += result.n_kernel_extrapolated_high_energy_steps
            stats["n_kernel_extrapolated_hull_steps"] += result.n_kernel_extrapolated_hull_steps
            stats["n_kernel_momentum_scaled_steps"] += result.n_kernel_momentum_scaled_steps
            stats["n_step_deflections_gt_300_mrad"] += result.n_step_deflections_gt_300_mrad
            stats["n_step_deflections_gt_500_mrad"] += result.n_step_deflections_gt_500_mrad
            stats["n_step_deflections_gt_1000_mrad"] += result.n_step_deflections_gt_1000_mrad
            if not result.touched_rock:
                # Includes directions that do not enter from the source plane.
                line = unit_from_theta_phi(theta_i, phi_i, ctx.az_center_deg)
                if (-line)[2] >= -1e-6:
                    stats["n_non_downward_for_source_plane"] += 1
                else:
                    stats["n_tracks_without_dem_rock"] += 1
                continue
            stats["n_tracks_touched_rock"] += 1
            if not result.survived:
                stats["n_tracks_not_survived"] += 1
                continue
            stats["n_tracks_survived"] += 1
            if not result.final_in_domain:
                stats["n_tracks_final_outside_domain"] += 1
            if not result.accepted or result.final_i is None or result.final_j is None:
                continue
            stats["n_tracks_final_inside_acceptance"] += 1
            final_counts[result.final_i, result.final_j] += track_weight
            source_cell = INSCAT.cell_from_edges(source_grid["theta_edges"], source_grid["phi_edges"], theta_i, phi_i)
            if source_cell is None:
                stats["n_accepted_source_map_misses"] += 1
            else:
                source_counts[source_cell] += track_weight
            final_pos = result.final_position_m or (np.nan, np.nan, np.nan)
            first_rock_pos = result.first_rock_position_m or (np.nan, np.nan, np.nan)
            if np.all(np.isfinite(np.asarray(first_rock_pos, dtype=float))):
                theta_contact, phi_contact = los_theta_phi_from_position(ctx, first_rock_pos[0], first_rock_pos[1], first_rock_pos[2])
                contact_inside = INSCAT.is_inside_acceptance(grid, theta_contact, phi_contact)
                contact_edge = edge_distance_xy(first_rock_pos[0], first_rock_pos[1], dem_bbox)
            else:
                theta_contact = phi_contact = contact_edge = float("nan")
                contact_inside = False
            accepted_rows.append({
                "accepted_id": int(len(accepted_rows) + 1),
                "source_event_index": int(event_index),
                "position_sample_index": int(position_sample_index),
                "start_x_m": float(start_pos[0]),
                "start_y_m": float(start_pos[1]),
                "start_z_m": float(start_pos[2]),
                "first_rock_x_m": float(first_rock_pos[0]),
                "first_rock_y_m": float(first_rock_pos[1]),
                "first_rock_z_m": float(first_rock_pos[2]),
                "first_rock_topo_m": float(result.first_rock_surface_height_m) if result.first_rock_surface_height_m is not None else float("nan"),
                "first_rock_edge_distance_m": float(contact_edge),
                "theta_first_rock_los_deg": float(theta_contact),
                "phi_first_rock_los_deg": float(phi_contact),
                "first_rock_los_inside_acceptance": int(bool(contact_inside)),
                "initial_distance_to_acceptance_deg": float(initial_distance_to_acceptance),
                "final_x_m": float(final_pos[0]),
                "final_y_m": float(final_pos[1]),
                "final_z_m": float(final_pos[2]),
                "theta_initial_deg": float(theta_i),
                "phi_initial_deg": float(phi_i),
                "theta_final_deg": float(result.theta_final_deg),
                "phi_final_deg": float(result.phi_final_deg),
                "kinetic_initial_GeV": float(kinetic0),
                "rock_length_m": float(result.rock_length_m),
                "n_steps_rock": int(result.n_steps_rock),
                "n_kernel_full_tail_steps": int(result.n_kernel_full_tail_steps),
                "n_kernel_core_steps": int(result.n_kernel_core_steps),
                "n_kernel_momentum_scaled_steps": int(result.n_kernel_momentum_scaled_steps),
                "n_step_deflections_gt_300_mrad": int(result.n_step_deflections_gt_300_mrad),
                "n_step_deflections_gt_500_mrad": int(result.n_step_deflections_gt_500_mrad),
                "n_step_deflections_gt_1000_mrad": int(result.n_step_deflections_gt_1000_mrad),
                "closest_approach_m": float(result.closest_approach_m),
                "entry_face": str(entry_face),
                "base_sample_weight": float(sample_weight),
                "face_importance_weight": float(face_importance_weight),
                "sample_weight": float(track_weight),
                "final_i": int(result.final_i),
                "final_j": int(result.final_j),
            })
            rock_lengths.append(float(result.rock_length_m))
            closest_distances.append(float(result.closest_approach_m))

    masked_final = np.where(grid.inside_mask, final_counts, np.nan)
    np.save(args.output_dir / "spatial_final_counts_theta_phi.npy", masked_final)
    np.save(args.output_dir / "spatial_source_counts_theta_phi.npy", source_counts)
    INSCAT.save_map_csv(args.output_dir / "spatial_final_counts_theta_phi.csv", grid, masked_final, grid.filled & (~grid.inside_mask), np.zeros_like(grid.L_grid))
    INSCAT.save_source_map_csv(args.output_dir / "spatial_source_counts_theta_phi.csv", source_grid, source_counts)
    accepted_tracks_path = args.output_dir / "spatial_accepted_tracks.csv"
    accepted_columns = [
        "accepted_id",
        "source_event_index", "position_sample_index",
        "start_x_m", "start_y_m", "start_z_m",
        "first_rock_x_m", "first_rock_y_m", "first_rock_z_m", "first_rock_topo_m",
        "first_rock_edge_distance_m", "theta_first_rock_los_deg", "phi_first_rock_los_deg", "first_rock_los_inside_acceptance",
        "initial_distance_to_acceptance_deg",
        "final_x_m", "final_y_m", "final_z_m",
        "theta_initial_deg", "phi_initial_deg", "theta_final_deg", "phi_final_deg",
        "kinetic_initial_GeV", "rock_length_m", "n_steps_rock",
        "n_kernel_full_tail_steps", "n_kernel_core_steps",
        "n_kernel_momentum_scaled_steps",
        "n_step_deflections_gt_300_mrad", "n_step_deflections_gt_500_mrad", "n_step_deflections_gt_1000_mrad",
        "closest_approach_m",
        "entry_face", "base_sample_weight", "face_importance_weight", "sample_weight", "final_i", "final_j",
    ]
    accepted_df = pd.DataFrame(accepted_rows, columns=accepted_columns)
    accepted_df.to_csv(accepted_tracks_path, index=False)
    accepted_weights = accepted_df["sample_weight"].to_numpy(dtype=float) if not accepted_df.empty else np.array([], dtype=float)
    accepted_weight_sum = float(np.sum(accepted_weights)) if accepted_weights.size else 0.0
    accepted_weight_sum_sq = float(np.sum(accepted_weights * accepted_weights)) if accepted_weights.size else 0.0
    accepted_effective_sample_size = (
        float((accepted_weight_sum * accepted_weight_sum) / accepted_weight_sum_sq)
        if accepted_weight_sum_sq > 0.0
        else 0.0
    )
    accepted_relative_mc_se = (
        float(math.sqrt(accepted_weight_sum_sq) / accepted_weight_sum)
        if accepted_weight_sum > 0.0
        else None
    )

    figure_paths: dict[str, str] = {}
    if not args.no_figures:
        if volcano_sampler is not None:
            target_png = args.output_dir / "volcano_surface_target_points_xy.png"
            fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
            sc = ax.scatter(
                volcano_sampler.x_m / 1000.0,
                volcano_sampler.y_m / 1000.0,
                c=volcano_sampler.height_fraction,
                s=12,
                cmap="viridis",
                alpha=0.82,
                linewidths=0.0,
            )
            ax.scatter([0.0], [0.0], marker="*", s=130, color="black", label=point)
            ax.set_xlabel(f"East from {point} (km)")
            ax.set_ylabel(f"North from {point} (km)")
            ax.set_title("Volcano-surface target points selected from DEM and angular mask")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            cb = fig.colorbar(sc, ax=ax, shrink=0.92)
            cb.set_label("DEM height fraction")
            fig.savefig(target_png, bbox_inches="tight")
            plt.close(fig)
            figure_paths["volcano_surface_target_points_xy_png"] = str(target_png)

        final_png = args.output_dir / "spatial_final_accepted_map.png"
        source_png = args.output_dir / "spatial_source_external_map.png"
        plot_map(final_png, grid.theta_edges, grid.phi_edges, masked_final, "Spatial DEM: final directions inside volcano mask", "Counts")
        plot_map(source_png, source_grid["theta_edges"], source_grid["phi_edges"], source_counts, "Spatial DEM: initial external directions that scatter inside", "Counts")
        figure_paths["spatial_final_accepted_map_png"] = str(final_png)
        figure_paths["spatial_source_external_map_png"] = str(source_png)
        if not accepted_df.empty:
            arrow_png = args.output_dir / "spatial_accepted_muon_arrows_theta_phi.png"
            fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
            sc0 = ax.scatter(
                accepted_df["phi_initial_deg"],
                accepted_df["theta_initial_deg"],
                c=accepted_df["kinetic_initial_GeV"],
                marker="x",
                s=42,
                cmap="viridis",
                label="initial external",
            )
            ax.scatter(
                accepted_df["phi_final_deg"],
                accepted_df["theta_final_deg"],
                c=accepted_df["kinetic_initial_GeV"],
                marker="o",
                s=46,
                cmap="viridis",
                edgecolor="white",
                linewidth=0.6,
                label="final accepted",
            )
            for row in accepted_df.itertuples(index=False):
                ax.annotate(
                    "",
                    xy=(row.phi_final_deg, row.theta_final_deg),
                    xytext=(row.phi_initial_deg, row.theta_initial_deg),
                    arrowprops={"arrowstyle": "->", "lw": 0.9, "alpha": 0.75, "color": "tab:red"},
                )
            ax.set_xlim(float(source_grid["phi_edges"][0]), float(source_grid["phi_edges"][-1]))
            ax.set_ylim(float(source_grid["theta_edges"][-1]), float(source_grid["theta_edges"][0]))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
            ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
            ax.set_title("Accepted in-scattering tracks: initial external to final accepted")
            if len(accepted_df) <= 80:
                for row in accepted_df.itertuples(index=False):
                    ax.text(row.phi_final_deg, row.theta_final_deg, str(int(row.accepted_id)), fontsize=7, color="white", ha="center", va="center")
            ax.legend(loc="best", fontsize=8)
            cb = fig.colorbar(sc0, ax=ax, shrink=0.92)
            cb.set_label("Initial kinetic energy (GeV)")
            fig.savefig(arrow_png, bbox_inches="tight")
            plt.close(fig)
            figure_paths["spatial_accepted_muon_arrows_theta_phi_png"] = str(arrow_png)

            plan_png = args.output_dir / "spatial_accepted_muon_tracks_xy.png"
            fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
            face_colors = {"top": "tab:blue", "west": "tab:orange", "east": "tab:green", "south": "tab:red", "north": "tab:purple", "volcano_surface": "tab:brown"}
            for face, group in accepted_df.groupby("entry_face"):
                color = face_colors.get(str(face), "black")
                ax.scatter(group["start_x_m"] / 1000.0, group["start_y_m"] / 1000.0, s=42, marker="x", color=color, label=f"start {face}")
                ax.scatter(group["final_x_m"] / 1000.0, group["final_y_m"] / 1000.0, s=42, marker="o", facecolor="none", edgecolor=color)
            for row in accepted_df.itertuples(index=False):
                color = face_colors.get(str(row.entry_face), "black")
                ax.plot([row.start_x_m / 1000.0, row.final_x_m / 1000.0], [row.start_y_m / 1000.0, row.final_y_m / 1000.0], color=color, lw=0.9, alpha=0.7)
            if len(accepted_df) <= 80:
                for row in accepted_df.itertuples(index=False):
                    ax.text(row.final_x_m / 1000.0, row.final_y_m / 1000.0, str(int(row.accepted_id)), fontsize=7, ha="center", va="center")
            ax.set_xlabel(f"East from {point} (km)")
            ax.set_ylabel(f"North from {point} (km)")
            ax.set_title("Accepted in-scattering tracks in DEM box, plan view")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.savefig(plan_png, bbox_inches="tight")
            plt.close(fig)
            figure_paths["spatial_accepted_muon_tracks_xy_png"] = str(plan_png)

            contact_png = args.output_dir / "spatial_first_rock_contact_xy.png"
            fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
            sc = ax.scatter(
                accepted_df["first_rock_x_m"] / 1000.0,
                accepted_df["first_rock_y_m"] / 1000.0,
                c=accepted_df["rock_length_m"],
                s=34,
                cmap="magma",
                alpha=0.86,
                edgecolor="white",
                linewidth=0.35,
            )
            ax.scatter([0.0], [0.0], marker="*", s=130, color="black", label=point)
            ax.set_xlabel(f"East from {point} (km)")
            ax.set_ylabel(f"North from {point} (km)")
            ax.set_title("Accepted in-scattering first DEM-rock contact points")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            cb = fig.colorbar(sc, ax=ax, shrink=0.92)
            cb.set_label("Rock length (m)")
            fig.savefig(contact_png, bbox_inches="tight")
            plt.close(fig)
            figure_paths["spatial_first_rock_contact_xy_png"] = str(contact_png)
        if rock_lengths:
            fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
            ax.hist(rock_lengths, bins=50, histtype="stepfilled", alpha=0.85)
            ax.set_xlabel("DEM rock path length (m)")
            ax.set_ylabel("Accepted tracks")
            ax.set_title("Spatial DEM accepted track lengths")
            path = args.output_dir / "spatial_accepted_rock_length_hist.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            figure_paths["spatial_accepted_rock_length_hist_png"] = str(path)

    summary = {
        "created_at": now_stamp(),
        "module": Path(__file__).name,
        "point": point,
        "spatial_positions_sampled": True,
        "detector_intersection_checked": bool(args.observer_radius_m > 0.0),
        "physical_scope_note": (
            "Samples physical positions on the configured source surface around the DEM and transports tracks through DEM rock. "
            "Unless observer_radius_m > 0, this is not a detector-intersection/rate calculation."
        ),
        "inputs": {
            "kinematic_cache": str(args.kinematic_cache),
            "kernel_npz": str(args.kernel_npz),
            "acceptance_map": str(args.acceptance_map),
            "length_map": str(args.length_map),
            "hgt_dir": str(hgt_dir),
            "range_file": str(range_file),
            "range_table_used": str(range_table_path),
        },
        "parameters": {
            "head": int(args.head),
            "chunk_index": int(args.chunk_index),
            "chunk_count": int(args.chunk_count),
            "seed": int(args.seed),
            "ray_step_m": float(args.ray_step_m),
            "kernel_step_is_native": kernel_step_is_native,
            "transport_scheme": "fixed_rock_step_condensed_history_endpoint_kick",
            "max_track_m": float(args.max_track_m),
            "source_plane_margin_m": float(args.source_plane_margin_m),
            "source_plane_height_margin_m": float(args.source_plane_height_margin_m),
            "source_plane_area_m2": float(ctx.source_area_m2),
            "source_surface": str(args.source_surface),
            "max_angular_margin_deg": float(args.max_angular_margin_deg) if args.max_angular_margin_deg is not None else None,
            "volcano_surface_grid_step_m": float(args.volcano_surface_grid_step_m),
            "volcano_surface_edge_guard_m": float(args.volcano_surface_edge_guard_m),
            "volcano_surface_min_height_frac": float(args.volcano_surface_min_height_frac),
            "volcano_surface_start_offset_m": float(args.volcano_surface_start_offset_m),
            "volcano_surface_entry_check_m": float(args.volcano_surface_entry_check_m),
            "volcano_surface_horizontal_area_m2": None if volcano_sampler is None else float(volcano_sampler.horizontal_area_m2),
            "volcano_surface_n_points": 0 if volcano_sampler is None else int(volcano_sampler.x_m.size),
            "volcano_surface_n_candidate_points": 0 if volcano_sampler is None else int(volcano_sampler.n_candidate_points),
            "volcano_surface_n_inside_acceptance": 0 if volcano_sampler is None else int(volcano_sampler.n_inside_acceptance),
            "volcano_surface_n_after_edge_guard": 0 if volcano_sampler is None else int(volcano_sampler.n_after_edge_guard),
            "volcano_surface_n_after_height_guard": 0 if volcano_sampler is None else int(volcano_sampler.n_after_height_guard),
            "entry_face_importance": dict(args.entry_face_importance_weights),
            "observer_radius_m": float(args.observer_radius_m),
            "position_samples_per_muon": int(args.position_samples_per_muon),
            "sample_probability": float(args.sample_probability),
            "sample_event_weight": float(1.0 / float(args.sample_probability)),
            "min_survival_rock_m": float(args.min_survival_rock_m),
            "min_survival_rock_m_role": "initial_CSDA_range_prefilter_not_minimum_accepted_path",
            "random_stream_policy": "deterministic_per_source_event_and_position_sample",
            "rho_g_cm3": float(args.rho),
            "interp_method": args.interp_method,
            "kernel_family": model.kernel_family,
            "kernel_tail_policy": model.tail_aware.policy_description if model.tail_aware is not None else "legacy",
            "kernel_support_mrad": [float(model.edges_mrad[0]), float(model.edges_mrad[-1])],
            "kernel_energy_cache_dlog": float(model.tail_aware.energy_cache_dlog) if model.tail_aware is not None else 0.0,
            "kernel_threshold": float(args.kernel_threshold),
            "kernel_scale": float(args.kernel_scale),
            "kernel_energy_extrapolation": str(args.kernel_energy_extrapolation),
            "kernel_energy_extrapolation_note": (
                "The complete empirical PDF is sampled first; outside measured energy bounds, "
                "the sampled angle is mapped with the standard 1/(beta*p) MCS scaling."
            ),
            "disable_scattering": bool(args.disable_scattering),
        },
        "stats": stats,
        "sampling_note": (
            "Counts are weighted by 1/sample_probability. With sample_probability < 1, outputs estimate the scanned cache, "
            "not an exact transported count."
        ),
        "weighted_accepted_count": float(np.nansum(masked_final)),
        "weighted_accepted_source_count": float(np.nansum(source_counts)),
        "accepted_weight_sum_sq": accepted_weight_sum_sq,
        "accepted_effective_sample_size": accepted_effective_sample_size,
        "accepted_relative_mc_se": accepted_relative_mc_se,
        "accepted_rock_length_m": {
            "count": int(len(rock_lengths)),
            "median": float(np.median(rock_lengths)) if rock_lengths else None,
            "p16": float(np.percentile(rock_lengths, 16)) if rock_lengths else None,
            "p84": float(np.percentile(rock_lengths, 84)) if rock_lengths else None,
        },
        "outputs": {
            "spatial_final_counts_theta_phi_npy": str(args.output_dir / "spatial_final_counts_theta_phi.npy"),
            "spatial_final_counts_theta_phi_csv": str(args.output_dir / "spatial_final_counts_theta_phi.csv"),
            "spatial_source_counts_theta_phi_npy": str(args.output_dir / "spatial_source_counts_theta_phi.npy"),
            "spatial_source_counts_theta_phi_csv": str(args.output_dir / "spatial_source_counts_theta_phi.csv"),
            "spatial_accepted_tracks_csv": str(accepted_tracks_path),
            **({"volcano_surface_target_points_csv": str(volcano_surface_target_path)} if volcano_surface_target_path is not None else {}),
            **figure_paths,
        },
    }
    with (args.output_dir / "spatial_in_scattering_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Spatial DEM in-scattering finished")
    print(f"  accepted count: {summary['weighted_accepted_count']:.6g}")
    print(f"  summary: {args.output_dir / 'spatial_in_scattering_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
