#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate angular-only in-scattering background from external muons.

This module estimates muons whose initial angular cell is outside an accepted
geometric mask but whose final, scattered direction lands inside that mask:

    external initial direction -> accepted final pixel

It is intentionally different from the event-by-event internal migration module,
which studies accepted/source pixels migrating into final accepted pixels.

Important physical scope
------------------------
This first implementation is angular-only. It uses the existing CABRIALES
theta-phi grids, rock-length map, CSDA energy loss and empirical MCS kernel, but
it does not verify a full 3D intersection with the detector's physical area.
That limitation is written to stdout and to in_scattering_summary.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
for _path in (MODULE_DIR, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from shw_io import (
        MUON_MASS_GEV,
        open_shw_bytes,
        parse_muon_parts,
        stream_size_hint,
        theta_phi_from_momentum,
    )
except ModuleNotFoundError:  # pragma: no cover
    from modulos.shw_io import (
        MUON_MASS_GEV,
        open_shw_bytes,
        parse_muon_parts,
        stream_size_hint,
        theta_phi_from_momentum,
    )

try:
    from plot_style import apply_scientific_style
except ModuleNotFoundError:  # pragma: no cover
    def apply_scientific_style() -> None:
        plt.rcParams.update({"savefig.dpi": 260, "savefig.bbox": "tight"})

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


def load_sibling_module(alias: str, filename: str):
    path = MODULE_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude cargar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


EVENT_MC = load_sibling_module("cabriales_event_mc_v2", "10_apply_event_by_event_empirical_mc_v2.py")
EVENT_CACHE = load_sibling_module("cabriales_event_cache", "04_build_event_cache.py")
ECRIT = load_sibling_module("cabriales_ecrit", "03_ecrit_heatmaps.py")
INSIDE = load_sibling_module("cabriales_inside_volcano", "07_inside_volcano_maps_merged.py")
LENGTHS = load_sibling_module("cabriales_lengths", "02_longitud.py")

try:
    DIAG = load_sibling_module("cabriales_continuous_migration_diagnostic", "11_event_continuous_migration_diagnostic.py")
    sample_delta_mrad = DIAG.sample_delta_mrad
except Exception:  # pragma: no cover
    def sample_delta_mrad(rng, centers, probability, widths_mrad, threshold):
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


@dataclass
class RangeEnergyLoss:
    range_gcm2: np.ndarray
    kinetic_GeV: np.ndarray
    rho_g_cm3: float

    def range_for_kinetic(self, kinetic_GeV: float) -> float:
        if not np.isfinite(kinetic_GeV) or kinetic_GeV <= 0.0:
            return 0.0
        return float(np.interp(kinetic_GeV, self.kinetic_GeV, self.range_gcm2))

    def kinetic_for_range(self, range_gcm2: float) -> float:
        if not np.isfinite(range_gcm2) or range_gcm2 <= 0.0:
            return 0.0
        return float(np.interp(range_gcm2, self.range_gcm2, self.kinetic_GeV))

    def advance(self, kinetic_GeV: float, step_m: float) -> float | None:
        current_range = self.range_for_kinetic(kinetic_GeV)
        dX_gcm2 = float(self.rho_g_cm3) * float(step_m) * 100.0
        remaining = current_range - dX_gcm2
        if remaining <= 0.0 or not np.isfinite(remaining):
            return None
        return self.kinetic_for_range(remaining)


@dataclass
class FluxEvent:
    theta_deg: float
    phi_rel_deg: float
    total_E_GeV: float
    pz_positive: bool


@dataclass
class PropagationResult:
    accepted: bool
    survived: bool
    theta_final_deg: float
    phi_final_deg: float
    kinetic_final_GeV: float
    final_i: int | None
    final_j: int | None
    deflection_final_deg: float
    used_nearest: int
    outside_domain: int
    no_support: int
    n_steps: int


@dataclass
class ExternalLengthModel:
    mode: str
    point: str
    grid: object
    hgt_dir: Path | None
    s_max_m: float
    ray_step_m: float
    cache_step_deg: float
    interp: object | None = None
    plat: float | None = None
    plon: float | None = None
    az_center: float | None = None
    z0: float | None = None
    dem_ready: bool = False
    length_cache: dict[tuple[int, int], tuple[float, str]] | None = None

    def ensure_dem(self) -> None:
        if self.dem_ready or self.mode == "length-map":
            return
        if self.hgt_dir is None:
            raise FileNotFoundError("--hgt-dir es requerido para calcular longitudes externas con DEM.")
        hgt_paths = [self.hgt_dir / name for name in LENGTHS.HGT_ORDER]
        missing = [str(path) for path in hgt_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Faltan HGT para longitud externa:\n  - " + "\n  - ".join(missing))

        mosaic, lats, lons = LENGTHS.mosaic_two_hgt([str(path) for path in hgt_paths])
        crop, crop_lats, crop_lons = LENGTHS.crop_dem(mosaic, lats, lons, LENGTHS.BBOX)
        self.interp = LENGTHS.make_interp(crop, crop_lats, crop_lons)
        self.plat, self.plon = LENGTHS.POINTS[self.point]
        self.az_center = LENGTHS.azimuth_deg(self.plat, self.plon, LENGTHS.SUMMIT[0], LENGTHS.SUMMIT[1])
        self.z0 = float(self.interp(np.array([self.plat]), np.array([self.plon]))[0])
        self.dem_ready = True

    def from_length_map(self, theta_deg: float, phi_rel_deg: float) -> float | None:
        cell = find_cell(self.grid, theta_deg, phi_rel_deg)
        if cell is None:
            return None
        i, j = cell
        length = float(self.grid.L_grid[i, j])
        if np.isfinite(length) and length > 0.0:
            return length
        return None

    def from_dem(self, theta_deg: float, phi_rel_deg: float) -> float:
        self.ensure_dem()
        if self.interp is None or self.plat is None or self.plon is None or self.az_center is None or self.z0 is None:
            return 0.0
        return float(
            LENGTHS.inside_length_one(
                self.plat,
                self.plon,
                self.az_center,
                self.z0,
                float(phi_rel_deg),
                float(theta_deg),
                self.interp,
                s_max=float(self.s_max_m),
                s_step=float(self.ray_step_m),
            )
        )

    def length_for(self, theta_deg: float, phi_rel_deg: float) -> tuple[float, str]:
        if self.length_cache is None:
            self.length_cache = {}
        if self.cache_step_deg > 0.0 and np.isfinite(self.cache_step_deg):
            key = (
                int(round(float(theta_deg) / self.cache_step_deg)),
                int(round(float(phi_rel_deg) / self.cache_step_deg)),
            )
            cached = self.length_cache.get(key)
            if cached is not None:
                return cached
        else:
            key = None

        if self.mode in ("length-map", "hybrid"):
            length = self.from_length_map(theta_deg, phi_rel_deg)
            if length is not None:
                out = (length, "length-map")
                if key is not None:
                    self.length_cache[key] = out
                return out
            if self.mode == "length-map":
                out = (0.0, "length-map-missing")
                if key is not None:
                    self.length_cache[key] = out
                return out
        out = (self.from_dem(theta_deg, phi_rel_deg), "dem")
        if key is not None:
            self.length_cache[key] = out
        return out


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def wrap180(phi_deg: float) -> float:
    out = float(phi_deg) % 360.0
    if out > 180.0:
        out -= 360.0
    return out


def infer_point_from_paths(paths: Iterable[Path]) -> str | None:
    for path in paths:
        if path is None:
            continue
        m = re.search(r"(?:^|[_/-])(P[1245])(?:[_./-]|$)", str(path))
        if m:
            return m.group(1)
    return None


def find_range_file(explicit: Path | None, length_map: Path) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = [
        length_map.parent / "muon_range_table.csv",
        length_map.parent / "data_rock.dat",
        length_map.parent.parent / "00_inputs" / "muon_range_table.csv",
        length_map.parent.parent / "00_inputs" / "data_rock.dat",
        PROJECT_ROOT / "data" / "muon_range_table.csv",
        PROJECT_ROOT / "data" / "data_rock.dat",
        Path.cwd() / "data" / "muon_range_table.csv",
        Path.cwd() / "data" / "data_rock.dat",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_hgt_dir(explicit: str | Path | None, length_map: Path) -> Path | None:
    if explicit is not None and str(explicit).strip().lower() != "auto":
        return Path(explicit)
    candidates = [
        length_map.parent,
        length_map.parent.parent / "00_inputs",
        PROJECT_ROOT / "data",
        Path.cwd() / "data",
    ]
    for directory in candidates:
        if all((directory / name).exists() for name in LENGTHS.HGT_ORDER):
            return directory
    return None


def load_energy_loss(range_file: Path, output_dir: Path, rho_g_cm3: float) -> tuple[RangeEnergyLoss, Path]:
    range_file = Path(range_file)
    if range_file.suffix.lower() == ".csv":
        table_path = range_file
    else:
        cache_dir = output_dir / "_range_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        table_path = cache_dir / "muon_range_table_from_data_rock.csv"
        ECRIT.parse_data_rock_to_csv(range_file, table_path)

    df = pd.read_csv(table_path)
    t_col = EVENT_MC.find_col(df, ["T_MeV", "kinetic_MeV", "T"], required=True)
    r_col = EVENT_MC.find_col(df, ["CSDA_gcm2", "range_gcm2", "CSDA Range"], required=True)
    kinetic = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float) * 1e-3
    range_gcm2 = pd.to_numeric(df[r_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(kinetic) & np.isfinite(range_gcm2) & (kinetic > 0.0) & (range_gcm2 > 0.0)
    if np.count_nonzero(ok) < 2:
        raise RuntimeError(f"Tabla CSDA inválida: {table_path}")
    kinetic = kinetic[ok]
    range_gcm2 = range_gcm2[ok]
    order = np.argsort(kinetic)
    return RangeEnergyLoss(range_gcm2=range_gcm2[order], kinetic_GeV=kinetic[order], rho_g_cm3=rho_g_cm3), table_path


def source_in_domain(theta_deg: float, phi_rel_deg: float, args) -> bool:
    if args.theta_min_deg is not None and theta_deg < args.theta_min_deg:
        return False
    if args.theta_max_deg is not None and theta_deg > args.theta_max_deg:
        return False
    if args.phi_min_deg is not None and phi_rel_deg < args.phi_min_deg:
        return False
    if args.phi_max_deg is not None and phi_rel_deg > args.phi_max_deg:
        return False
    return True


def align_mask_to_grid(acc: dict, grid) -> np.ndarray:
    acc_theta = 0.5 * (np.asarray(acc["theta_edges"][:-1], dtype=float) + np.asarray(acc["theta_edges"][1:], dtype=float))
    acc_phi = 0.5 * (np.asarray(acc["phi_edges"][:-1], dtype=float) + np.asarray(acc["phi_edges"][1:], dtype=float))
    acc_mask = np.asarray(acc["mask"], dtype=bool)

    if (
        acc_mask.shape == grid.L_grid.shape
        and np.allclose(acc_theta, grid.theta, rtol=0.0, atol=1e-6)
        and np.allclose(acc_phi, grid.phi, rtol=0.0, atol=1e-6)
    ):
        return acc_mask & grid.filled

    mask = np.zeros_like(grid.L_grid, dtype=bool)
    ti = {round(float(v), 10): i for i, v in enumerate(grid.theta)}
    pj = {round(float(v), 10): j for j, v in enumerate(grid.phi)}
    for ia, th in enumerate(acc_theta):
        i = ti.get(round(float(th), 10))
        if i is None:
            continue
        for ja, ph in enumerate(acc_phi):
            if not acc_mask[ia, ja]:
                continue
            j = pj.get(round(float(ph), 10))
            if j is not None:
                mask[i, j] = True
    return mask & grid.filled


def load_acceptance_map(path: Path, explicit_mask_col: str | None, mask_min: float, theta_max: float | None) -> dict:
    df = pd.read_csv(path)
    theta_col = EVENT_MC.find_col(df, ["theta_deg", "theta_center_deg", "theta", "Theta", "zenith_deg", "theta_z_deg"])
    phi_col = EVENT_MC.find_col(df, ["phi_rel_deg", "phi_deg", "phi", "azimuth_deg", "az_deg"])
    mask_col = explicit_mask_col
    if mask_col is None:
        for candidate in ("inside_volcano_geometry", "blocked_geometry", "blocked", "inside", "mask"):
            if candidate in df.columns:
                mask_col = candidate
                break
    if mask_col is None:
        mask_col = EVENT_MC.find_col(df, ["inside_volcano_geometry", "blocked", "inside", "mask", "length_inside_m"])
    if mask_col not in df.columns:
        raise KeyError(f"Mask column {mask_col!r} not found in {path}")

    df = df.copy()
    for col in (theta_col, phi_col, mask_col):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[theta_col, phi_col, mask_col])
    if theta_max is not None:
        df = df[df[theta_col] <= theta_max]
    if df.empty:
        raise RuntimeError(f"No acceptance rows remain after cuts in {path}")

    theta = np.array(sorted(df[theta_col].unique()), dtype=float)
    phi = np.array(sorted(df[phi_col].unique()), dtype=float)
    theta_edges = EVENT_MC.centers_to_edges(theta, EVENT_MC.bin_width(theta))
    phi_edges = EVENT_MC.centers_to_edges(phi, EVENT_MC.bin_width(phi))
    mask = np.zeros((len(theta), len(phi)), dtype=bool)
    ti = {round(float(v), 10): i for i, v in enumerate(theta)}
    pj = {round(float(v), 10): j for j, v in enumerate(phi)}
    for row in df[[theta_col, phi_col, mask_col]].itertuples(index=False, name=None):
        th, ph, val = row
        i = ti.get(round(float(th), 10))
        j = pj.get(round(float(ph), 10))
        if i is not None and j is not None and np.isfinite(val) and float(val) > mask_min:
            mask[i, j] = True
    return {
        "theta_edges": theta_edges,
        "phi_edges": phi_edges,
        "mask": mask,
        "mask_col": mask_col,
        "geom_csv": Path(path),
    }


def load_grid_and_acceptance(args):
    point = args.point or infer_point_from_paths([args.acceptance_map, args.length_map])
    if point is None:
        raise ValueError("No pude inferir --point desde los nombres de archivo. Usa --point P1/P2/P4/P5.")

    grid = EVENT_MC.load_ecrit_grid(
        Path(args.length_map),
        point,
        theta_min=None,
        theta_max=args.theta_max_deg,
        phi_min=None,
        phi_max=None,
    )
    acc = load_acceptance_map(
        Path(args.acceptance_map),
        explicit_mask_col=args.acceptance_mask_col,
        mask_min=args.acceptance_mask_min,
        theta_max=args.theta_max_deg,
    )
    inside_mask = align_mask_to_grid(acc, grid)
    grid.inside_mask[:, :] = inside_mask
    return point, grid, acc


def angular_distance_to_acceptance(grid, inside_mask: np.ndarray) -> np.ndarray:
    if not np.any(inside_mask):
        return np.full_like(grid.L_grid, np.inf, dtype=float)
    theta_step = EVENT_MC.bin_width(grid.theta)
    phi_step = EVENT_MC.bin_width(grid.phi)
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(~inside_mask, sampling=(theta_step, phi_step))
    except Exception:
        accepted = np.column_stack(np.where(inside_mask))
        out = np.full_like(grid.L_grid, np.inf, dtype=float)
        for i in range(out.shape[0]):
            dtheta = (grid.theta[accepted[:, 0]] - grid.theta[i]) ** 2
            for j in range(out.shape[1]):
                dphi = (grid.phi[accepted[:, 1]] - grid.phi[j]) ** 2
                out[i, j] = math.sqrt(float(np.min(dtheta + dphi)))
        return out


def accepted_centers(grid) -> np.ndarray:
    ii, jj = np.where(grid.inside_mask)
    if ii.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack([grid.theta[ii], grid.phi[jj]]).astype(float)


def distance_to_acceptance_deg(theta_deg: float, phi_rel_deg: float, centers: np.ndarray) -> float:
    if centers.size == 0:
        return float("inf")
    dtheta2 = (centers[:, 0] - float(theta_deg)) ** 2
    dphi2 = (centers[:, 1] - float(phi_rel_deg)) ** 2
    return math.sqrt(float(np.min(dtheta2 + dphi2)))


def is_inside_acceptance(grid, theta_deg: float, phi_rel_deg: float) -> bool:
    cell = find_cell(grid, theta_deg, phi_rel_deg)
    if cell is None:
        return False
    i, j = cell
    return bool(grid.inside_mask[i, j])


def find_cell(grid, theta_deg: float, phi_rel_deg: float) -> tuple[int, int] | None:
    i = int(np.searchsorted(grid.theta_edges, theta_deg, side="right") - 1)
    if i < 0 or i >= len(grid.theta):
        return None
    j = int(np.searchsorted(grid.phi_edges, phi_rel_deg, side="right") - 1)
    if j < 0 or j >= len(grid.phi):
        return None
    if not grid.filled[i, j]:
        return None
    return i, j


def iter_shw_events(args, grid) -> Iterator[FluxEvent]:
    total_bytes = stream_size_hint(args.input_shw)
    with open_shw_bytes(args.input_shw, member_name=args.shw_member) as handle:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="in-scattering SHW", disable=args.no_progress)
        for raw in handle:
            pbar.update(len(raw))
            s = raw.strip()
            if not s or s.startswith(b"#"):
                continue
            rec = parse_muon_parts(s.split(), shw_format=args.shw_format, only_muons=True)
            if rec is None:
                continue
            angles = theta_phi_from_momentum(rec.px, rec.py, rec.pz)
            if angles is None:
                continue
            theta, phi_abs = angles
            phi_rel = wrap180((phi_abs - grid.phi0) % 360.0)
            yield FluxEvent(theta, phi_rel, float(rec.e_total_GeV), bool(rec.pz > 0.0))
        pbar.close()


def iter_kinematic_events(args, grid) -> Iterator[FluxEvent]:
    total = None
    cache = Path(args.kinematic_cache)
    manifest = cache / "manifest.json"
    if manifest.exists():
        try:
            total = int(json.loads(manifest.read_text(encoding="utf-8")).get("n_events", 0))
        except Exception:
            total = None

    pbar = tqdm(total=total, unit="event", unit_scale=True, desc="in-scattering kinematic-cache", disable=args.no_progress)
    for _, chunk in EVENT_CACHE.iter_kinematic_cache(cache):
        theta = np.asarray(chunk["theta_deg"], dtype=float)
        phi_abs = np.asarray(chunk["phi_abs_deg"], dtype=float)
        total_e = np.asarray(chunk["total_E_GeV"], dtype=float)
        pz_positive = chunk.get("pz_positive")
        if pz_positive is None:
            pz_positive = np.zeros(theta.shape, dtype=np.uint8)
        else:
            pz_positive = np.asarray(pz_positive, dtype=np.uint8)
        phi_rel = (phi_abs - grid.phi0) % 360.0
        phi_rel = np.where(phi_rel > 180.0, phi_rel - 360.0, phi_rel)
        for th, ph, energy, up in zip(theta, phi_rel, total_e, pz_positive):
            yield FluxEvent(float(th), float(ph), float(energy), bool(up))
        pbar.update(int(theta.size))
    pbar.close()


def vectorized_inside_acceptance(grid, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    i = np.searchsorted(grid.theta_edges, theta, side="right") - 1
    j = np.searchsorted(grid.phi_edges, phi, side="right") - 1
    ok = (i >= 0) & (i < len(grid.theta)) & (j >= 0) & (j < len(grid.phi))
    inside = np.zeros(theta.shape, dtype=bool)
    if np.any(ok):
        inside[ok] = grid.inside_mask[i[ok], j[ok]]
    return inside


def length_for_events(
    theta: np.ndarray,
    phi: np.ndarray,
    length_model: ExternalLengthModel,
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.zeros(theta.shape, dtype=float)
    source_codes = np.zeros(theta.shape, dtype=np.uint8)
    if theta.size == 0:
        return lengths, source_codes

    step = float(length_model.cache_step_deg)
    if step > 0.0 and np.isfinite(step):
        kt = np.rint(theta / step).astype(np.int64)
        kp = np.rint(phi / step).astype(np.int64)
        packed = kt * 1_000_000 + (kp + 500_000)
        unique, inverse = np.unique(packed, return_inverse=True)
        unique_lengths = np.zeros(unique.shape, dtype=float)
        unique_sources = np.zeros(unique.shape, dtype=np.uint8)
        for u_idx, key in enumerate(unique):
            kt_u = int(key // 1_000_000)
            kp_u = int(key - kt_u * 1_000_000 - 500_000)
            length, source = length_model.length_for(kt_u * step, kp_u * step)
            unique_lengths[u_idx] = length
            unique_sources[u_idx] = 1 if source == "length-map" else (2 if source == "dem" else 0)
        lengths = unique_lengths[inverse]
        source_codes = unique_sources[inverse]
        return lengths, source_codes

    for idx, (th, ph) in enumerate(zip(theta, phi)):
        length, source = length_model.length_for(float(th), float(ph))
        lengths[idx] = length
        source_codes[idx] = 1 if source == "length-map" else (2 if source == "dem" else 0)
    return lengths, source_codes


def build_source_grid(args, grid) -> dict[str, np.ndarray]:
    theta_step = EVENT_MC.bin_width(grid.theta)
    phi_step = EVENT_MC.bin_width(grid.phi)
    theta_min = 0.0 if args.theta_min_deg is None else float(args.theta_min_deg)
    theta_max = 180.0 if args.theta_max_deg is None else float(args.theta_max_deg)
    phi_min = -180.0 if args.phi_min_deg is None else float(args.phi_min_deg)
    phi_max = 180.0 if args.phi_max_deg is None else float(args.phi_max_deg)

    theta_edges = np.arange(theta_min, theta_max + 0.5 * theta_step, theta_step, dtype=float)
    phi_edges = np.arange(phi_min, phi_max + 0.5 * phi_step, phi_step, dtype=float)
    if theta_edges[-1] < theta_max:
        theta_edges = np.append(theta_edges, theta_max)
    if phi_edges[-1] < phi_max:
        phi_edges = np.append(phi_edges, phi_max)
    theta_edges[-1] = theta_max
    phi_edges[-1] = phi_max

    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    inside = vectorized_inside_acceptance(grid, TH.ravel(), PH.ravel()).reshape(TH.shape)
    return {
        "theta": theta,
        "phi": phi,
        "theta_edges": theta_edges,
        "phi_edges": phi_edges,
        "inside_acceptance": inside,
    }


def cell_from_edges(theta_edges: np.ndarray, phi_edges: np.ndarray, theta_deg: float, phi_deg: float) -> tuple[int, int] | None:
    i = int(np.searchsorted(theta_edges, theta_deg, side="right") - 1)
    j = int(np.searchsorted(phi_edges, phi_deg, side="right") - 1)
    if i == len(theta_edges) - 1 and np.isclose(theta_deg, theta_edges[-1]):
        i -= 1
    if j == len(phi_edges) - 1 and np.isclose(phi_deg, phi_edges[-1]):
        j -= 1
    if 0 <= i < len(theta_edges) - 1 and 0 <= j < len(phi_edges) - 1:
        return i, j
    return None


def process_kinematic_cache_fast(
    args,
    grid,
    source_grid: dict[str, np.ndarray],
    length_model: ExternalLengthModel,
    energy_loss: RangeEnergyLoss,
    model,
    rng: np.random.Generator,
    counts: np.ndarray,
    source_counts: np.ndarray,
    accepted_deflection_deg: list[float],
    accepted_energy_initial: list[float],
    accepted_length_m: list[float],
    stats: dict,
) -> None:
    """Fast chunked path for the huge 90-day kinematic caches.

    Screening is vectorized per chunk. We only enter the Python/kernel loop for
    external muons with positive material length and enough CSDA range to
    survive the material column.
    """
    total = None
    cache = Path(args.kinematic_cache)
    manifest = cache / "manifest.json"
    if manifest.exists():
        try:
            total = int(json.loads(manifest.read_text(encoding="utf-8")).get("n_events", 0))
        except Exception:
            total = None

    sample_weight = 1.0 / float(args.n_samples_per_muon)
    selected_so_far = 0
    pbar = tqdm(total=total, unit="event", unit_scale=True, desc="in-scattering fast-cache", disable=args.no_progress)

    for _, chunk in EVENT_CACHE.iter_kinematic_cache(cache):
        theta_all = np.asarray(chunk["theta_deg"], dtype=float)
        phi_abs_all = np.asarray(chunk["phi_abs_deg"], dtype=float)
        total_e_all = np.asarray(chunk["total_E_GeV"], dtype=float)
        pz_positive = chunk.get("pz_positive")
        if pz_positive is None:
            pz_positive = np.zeros(theta_all.shape, dtype=np.uint8)
        else:
            pz_positive = np.asarray(pz_positive, dtype=np.uint8)

        n_chunk = int(theta_all.size)
        phi_all = (phi_abs_all - grid.phi0) % 360.0
        phi_all = np.where(phi_all > 180.0, phi_all - 360.0, phi_all)

        domain = np.isfinite(theta_all) & np.isfinite(phi_all) & np.isfinite(total_e_all)
        if args.theta_min_deg is not None:
            domain &= theta_all >= args.theta_min_deg
        if args.theta_max_deg is not None:
            domain &= theta_all <= args.theta_max_deg
        if args.phi_min_deg is not None:
            domain &= phi_all >= args.phi_min_deg
        if args.phi_max_deg is not None:
            domain &= phi_all <= args.phi_max_deg
        if args.discard_upgoing:
            up = pz_positive != 0
            stats["n_discarded_upgoing"] += int(np.count_nonzero(domain & up))
            domain &= ~up

        inside = vectorized_inside_acceptance(grid, theta_all, phi_all)
        selected = domain & (~inside)

        if args.max_angular_margin_deg is not None:
            # Margin mode is rare for production. Use the exact scalar distance
            # only for already-external candidates.
            idx = np.where(selected)[0]
            keep = np.zeros(idx.shape, dtype=bool)
            centers = accepted_centers(grid)
            for n, original_idx in enumerate(idx):
                keep[n] = distance_to_acceptance_deg(
                    float(theta_all[original_idx]), float(phi_all[original_idx]), centers
                ) <= float(args.max_angular_margin_deg)
            margin_reject_idx = idx[~keep]
            selected_idx = idx[keep]
        else:
            margin_reject_idx = np.empty(0, dtype=int)
            selected_idx = np.where(selected)[0]

        if args.head:
            remaining = int(args.head) - selected_so_far
            if remaining <= 0:
                break
            if selected_idx.size > remaining:
                selected_idx = selected_idx[:remaining]
                read_limit = int(selected_idx[-1]) + 1 if selected_idx.size else n_chunk
                stop_after_chunk = True
            else:
                read_limit = n_chunk
                stop_after_chunk = False
        else:
            read_limit = n_chunk
            stop_after_chunk = False

        stats["n_flux_events_read"] += read_limit
        stats["n_in_source_domain"] += int(np.count_nonzero(domain[:read_limit]))
        stats["n_initial_inside_acceptance_skipped"] += int(np.count_nonzero((domain & inside)[:read_limit]))
        if margin_reject_idx.size:
            stats["n_outside_margin_skipped"] += int(np.count_nonzero(margin_reject_idx < read_limit))
        pbar.update(read_limit)

        if selected_idx.size == 0:
            if stop_after_chunk:
                break
            continue

        selected_so_far += int(selected_idx.size)
        stats["n_selected_external_muons"] += int(selected_idx.size)

        theta = theta_all[selected_idx]
        phi = phi_all[selected_idx]
        total_e = total_e_all[selected_idx]
        kinetic = total_e - MUON_MASS_GEV
        lengths, sources = length_for_events(theta, phi, length_model)
        stats["n_length_from_map"] += int(np.count_nonzero(sources == 1))
        stats["n_length_from_dem"] += int(np.count_nonzero(sources == 2))

        positive_length = np.isfinite(lengths) & (lengths > 0.0) & np.isfinite(kinetic) & (kinetic > 0.0)
        stats["n_selected_without_positive_length"] += int(selected_idx.size - np.count_nonzero(positive_length))
        if not np.any(positive_length):
            if stop_after_chunk:
                break
            continue
        stats["n_positive_length_external_muons"] += int(np.count_nonzero(positive_length))

        theta_p = theta[positive_length]
        phi_p = phi[positive_length]
        kinetic_p = kinetic[positive_length]
        length_p = lengths[positive_length]

        ranges = np.interp(kinetic_p, energy_loss.kinetic_GeV, energy_loss.range_gcm2)
        required = float(args.rho) * length_p * 100.0
        can_survive = ranges > required
        n_fail = int(np.count_nonzero(~can_survive))
        if n_fail:
            stats["n_survival_prefilter_rejected"] += n_fail
            stats["n_samples_total"] += n_fail * int(args.n_samples_per_muon)
            stats["n_samples_not_survived"] += n_fail * int(args.n_samples_per_muon)
        if not np.any(can_survive):
            if stop_after_chunk:
                break
            continue

        for theta0, phi0, kinetic0, length_m in zip(
            theta_p[can_survive],
            phi_p[can_survive],
            kinetic_p[can_survive],
            length_p[can_survive],
        ):
            for _ in range(args.n_samples_per_muon):
                stats["n_samples_total"] += 1
                result = propagate_external_muon(
                    float(theta0),
                    float(phi0),
                    float(kinetic0),
                    float(length_m),
                    model,
                    energy_loss,
                    rng,
                    args,
                    grid,
                )
                stats["n_kernel_nearest_fallback_steps"] += result.used_nearest
                stats["n_kernel_outside_domain_steps"] += result.outside_domain
                stats["n_kernel_no_support_steps"] += result.no_support
                if not result.survived:
                    stats["n_samples_not_survived"] += 1
                    continue
                stats["n_samples_survived"] += 1
                if not result.accepted or result.final_i is None or result.final_j is None:
                    continue
                stats["n_samples_accepted"] += 1
                counts[result.final_i, result.final_j] += sample_weight
                source_cell = cell_from_edges(source_grid["theta_edges"], source_grid["phi_edges"], float(theta0), float(phi0))
                if source_cell is None:
                    stats["n_accepted_source_map_misses"] += 1
                else:
                    source_counts[source_cell] += sample_weight
                accepted_deflection_deg.append(float(result.deflection_final_deg))
                accepted_energy_initial.append(float(kinetic0))
                accepted_length_m.append(float(length_m))

        if stop_after_chunk:
            break

    pbar.close()


def propagate_external_muon(
    theta0: float,
    phi0: float,
    kinetic0: float,
    length_m: float,
    model,
    energy_loss: RangeEnergyLoss,
    rng: np.random.Generator,
    args,
    grid,
) -> PropagationResult:
    theta = float(theta0)
    phi = float(phi0)
    kinetic = float(kinetic0)
    used_nearest = 0
    outside_domain = 0
    no_support = 0
    n_steps = 0
    rad_to_mrad = 1000.0 * math.pi / 180.0

    if length_m <= 0.0 or not np.isfinite(length_m):
        return PropagationResult(False, False, theta, phi, kinetic, None, None, 0.0, 0, 0, 0, 0)

    remaining = float(length_m)
    while remaining > 1e-9:
        step = min(float(args.step_m), remaining)
        n_steps += 1

        if not args.disable_scattering:
            pred = model.predict_kernel(step, kinetic)
            if pred.used_nearest_fallback:
                used_nearest += 1
            if pred.outside_domain:
                outside_domain += 1
            if not pred.valid:
                no_support += 1
            else:
                dtheta_mrad = sample_delta_mrad(
                    rng,
                    pred.centers_mrad,
                    pred.probability_per_bin,
                    model.widths_mrad,
                    args.kernel_threshold,
                ) * float(args.kernel_scale)
                dphi_eff_mrad = sample_delta_mrad(
                    rng,
                    pred.centers_mrad,
                    pred.probability_per_bin,
                    model.widths_mrad,
                    args.kernel_threshold,
                ) * float(args.kernel_scale)
                theta += dtheta_mrad / rad_to_mrad
                sin_th = max(abs(math.sin(math.radians(theta))), 1e-3)
                phi = wrap180(phi + dphi_eff_mrad / (rad_to_mrad * sin_th))

        kinetic_next = energy_loss.advance(kinetic, step)
        if kinetic_next is None or kinetic_next <= 0.0 or not np.isfinite(kinetic_next):
            return PropagationResult(
                False,
                False,
                theta,
                phi,
                0.0,
                None,
                None,
                math.hypot(theta - theta0, phi - phi0),
                used_nearest,
                outside_domain,
                no_support,
                n_steps,
            )
        kinetic = kinetic_next
        remaining -= step

        if theta < 0.0 or theta > 180.0 or not np.isfinite(theta) or not np.isfinite(phi):
            return PropagationResult(False, False, theta, phi, kinetic, None, None, np.nan, used_nearest, outside_domain, no_support, n_steps)

    dest = find_cell(grid, theta, phi)
    accepted = False
    final_i = final_j = None
    if dest is not None:
        final_i, final_j = dest
        accepted = bool(grid.inside_mask[final_i, final_j])
    return PropagationResult(
        accepted,
        True,
        theta,
        phi,
        kinetic,
        final_i,
        final_j,
        math.hypot(theta - theta0, phi - phi0),
        used_nearest,
        outside_domain,
        no_support,
        n_steps,
    )


def save_map_csv(path: Path, grid, counts: np.ndarray, extended_external: np.ndarray, distance_deg: np.ndarray) -> None:
    TH, PH = np.meshgrid(grid.theta, grid.phi, indexing="ij")
    df = pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "inside_acceptance": grid.inside_mask.ravel().astype(int),
        "extended_external_region": extended_external.ravel().astype(int),
        "distance_to_acceptance_deg": distance_deg.ravel(),
        "in_scattering_count": counts.ravel(),
    })
    df.to_csv(path, index=False)


def save_source_map_csv(path: Path, source_grid: dict[str, np.ndarray], source_counts: np.ndarray) -> None:
    theta = np.asarray(source_grid["theta"], dtype=float)
    phi = np.asarray(source_grid["phi"], dtype=float)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    inside = np.asarray(source_grid["inside_acceptance"], dtype=bool)
    counts = np.asarray(source_counts, dtype=float)
    df = pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "inside_acceptance": inside.ravel().astype(int),
        "external_source_region": (~inside).ravel().astype(int),
        "in_scattering_source_count": counts.ravel(),
    })
    df.to_csv(path, index=False)


def plot_extended_region(path: Path, grid, extended_external: np.ndarray) -> None:
    Z = np.zeros_like(grid.L_grid, dtype=float)
    Z[extended_external] = 1.0
    Z[grid.inside_mask] = 2.0
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    im = ax.pcolormesh(grid.phi_edges, grid.theta_edges, Z, shading="flat", cmap="viridis", vmin=0.0, vmax=2.0)
    EVENT_MC.format_axes(ax, grid)
    ax.set_title("Accepted mask and external angular margin")
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_ticks([0.0, 1.0, 2.0])
    cb.set_ticklabels(["outside", "external margin", "accepted"])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def format_source_axes(ax, source_grid: dict[str, np.ndarray]) -> None:
    theta_edges = np.asarray(source_grid["theta_edges"], dtype=float)
    phi_edges = np.asarray(source_grid["phi_edges"], dtype=float)
    ax.set_xlim(float(phi_edges[0]), float(phi_edges[-1]))
    ax.set_ylim(float(theta_edges[-1]), float(theta_edges[0]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")


def plot_source_counts(path: Path, source_grid: dict[str, np.ndarray], source_counts: np.ndarray, grid) -> None:
    vals = source_counts[np.isfinite(source_counts) & (source_counts > 0)]
    if vals.size == 0:
        return
    from matplotlib.colors import LogNorm

    Z = np.where(source_counts > 0, source_counts, np.nan)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    im = ax.pcolormesh(
        source_grid["phi_edges"],
        source_grid["theta_edges"],
        Z,
        shading="flat",
        cmap="magma",
        norm=LogNorm(vmin=1.0, vmax=max(1.0, float(np.nanmax(vals)))),
    )
    try:
        PH, TH = np.meshgrid(grid.phi, grid.theta)
        ax.contour(PH, TH, grid.inside_mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.75, alpha=0.85)
    except Exception:
        pass
    format_source_axes(ax, source_grid)
    ax.set_title("Initial external directions accepted by in-scattering")
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label("Accepted source counts")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_counts(path: Path, grid, counts: np.ndarray) -> None:
    Z = np.where(grid.inside_mask, counts, np.nan)
    vals = Z[np.isfinite(Z) & (Z > 0)]
    vmax = float(np.nanpercentile(vals, 99.0)) if vals.size else 1.0
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    im = ax.pcolormesh(grid.phi_edges, grid.theta_edges, Z, shading="flat", cmap="magma", vmin=0.0, vmax=vmax)
    EVENT_MC.format_axes(ax, grid)
    ax.set_title("Accepted in-scattering contamination")
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label("Weighted counts")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hist(path: Path, values: list[float], xlabel: str, title: str) -> bool:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.hist(arr, bins=50, histtype="stepfilled", alpha=0.82)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    ax.set_title(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Angular-only external -> accepted in-scattering background estimator."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-shw", type=Path, default=None, help="Open-flux SHW/tar input.")
    src.add_argument("--kinematic-cache", type=Path, default=None, help="Chunked kinematic cache from 04_build_kinematic_cache.py.")
    ap.add_argument(
        "--kernel-npz",
        type=Path,
        default=MODULE_DIR / "hybrid_empirical_kernel_library.npz",
        help="Empirical MCS kernel (default: bundled hybrid full-tail model).",
    )
    ap.add_argument("--acceptance-map", required=True, type=Path, help="CSV with accepted angular mask.")
    ap.add_argument("--length-map", required=True, type=Path, help="rock_length/ecrit CSV with length_inside_m.")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--point", choices=["P1", "P2", "P4", "P5"], default=None)
    ap.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="auto")
    ap.add_argument("--shw-member", default=None)

    ap.add_argument("--step-m", type=float, default=100.0)
    ap.add_argument("--n-samples-per-muon", type=int, default=1)
    ap.add_argument("--max-angular-margin-deg", type=float, default=None,
                    help="Restricción opcional de borde. Si se omite, usa todo F \\ A.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=1,
                    help="Reservado para paralelismo futuro. El kinematic-cache usa cribado vectorizado monoproceso.")
    ap.add_argument("--discard-upgoing", action="store_true")
    ap.add_argument("--theta-min-deg", type=float, default=0.0,
                    help="Theta mínimo del dominio físico de fuentes F.")
    ap.add_argument("--theta-max-deg", type=float, default=180.0,
                    help="Theta máximo del dominio físico de fuentes F.")
    ap.add_argument("--phi-min-deg", type=float, default=-180.0,
                    help="Phi relativo mínimo del dominio físico de fuentes F.")
    ap.add_argument("--phi-max-deg", type=float, default=180.0,
                    help="Phi relativo máximo del dominio físico de fuentes F.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug-trajectories", action="store_true")

    ap.add_argument("--acceptance-mask-col", default=None)
    ap.add_argument("--acceptance-mask-min", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=2.65, help="Effective rock density for CSDA energy loss.")
    ap.add_argument("--range-file", type=Path, default=None, help="Optional muon_range_table.csv or data_rock.dat.")
    ap.add_argument("--external-length-mode", choices=["hybrid", "dem", "length-map"], default="hybrid",
                    help="hybrid usa length-map si tiene L>0 y DEM como fallback para el complemento.")
    ap.add_argument("--hgt-dir", default="auto", help="Carpeta con HGT para longitudes externas por DEM.")
    ap.add_argument("--external-s-max-m", type=float, default=float(LENGTHS.R_MAX_M),
                    help="Longitud máxima del rayo DEM para fuentes externas.")
    ap.add_argument("--external-ray-step-m", type=float, default=float(LENGTHS.S_STEP_M),
                    help="Paso del trazador DEM para longitud externa.")
    ap.add_argument("--length-cache-step-deg", type=float, default=0.5,
                    help="Cuantización angular para cachear longitudes externas por DEM. 0 desactiva.")
    ap.add_argument("--interp-method", choices=["tail-aware", "linear", "rbf_linear", "nearest"], default="tail-aware")
    ap.add_argument("--rbf-smoothing", type=float, default=0.0)
    ap.add_argument("--kernel-threshold", type=float, default=0.0)
    ap.add_argument("--kernel-scale", type=float, default=1.0, help="Scale sampled angular deflections for validation studies.")
    ap.add_argument("--disable-scattering", action="store_true")
    ap.add_argument("--head", type=int, default=0, help="Debug: stop after N selected external muons.")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    return ap


def main(argv=None) -> int:
    apply_scientific_style()
    args = parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.step_m <= 0.0:
        raise ValueError("--step-m must be positive")
    if args.n_samples_per_muon <= 0:
        raise ValueError("--n-samples-per-muon must be positive")
    if args.max_angular_margin_deg is not None and args.max_angular_margin_deg < 0.0:
        raise ValueError("--max-angular-margin-deg must be non-negative")
    if args.theta_min_deg is not None and args.theta_max_deg is not None and args.theta_min_deg >= args.theta_max_deg:
        raise ValueError("--theta-min-deg must be smaller than --theta-max-deg")
    if args.phi_min_deg is not None and args.phi_max_deg is not None and args.phi_min_deg >= args.phi_max_deg:
        raise ValueError("--phi-min-deg must be smaller than --phi-max-deg")
    if args.external_s_max_m <= 0.0 or args.external_ray_step_m <= 0.0:
        raise ValueError("--external-s-max-m and --external-ray-step-m must be positive")
    if args.workers != 1:
        print("[WARN] --workers se acepta por compatibilidad, pero este módulo angular-only corre secuencialmente para reproducibilidad.")

    for label, path in [
        ("kernel", args.kernel_npz),
        ("acceptance-map", args.acceptance_map),
        ("length-map", args.length_map),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"No encontré {label}: {path}")
    if args.input_shw is not None and not args.input_shw.exists():
        raise FileNotFoundError(args.input_shw)
    if args.kinematic_cache is not None and not args.kinematic_cache.exists():
        raise FileNotFoundError(args.kinematic_cache)

    point, grid, acc = load_grid_and_acceptance(args)
    distance_deg = angular_distance_to_acceptance(grid, grid.inside_mask)
    if args.max_angular_margin_deg is None:
        extended_external = grid.filled & (~grid.inside_mask)
    else:
        extended_external = (
            grid.filled
            & (~grid.inside_mask)
            & np.isfinite(distance_deg)
            & (distance_deg <= float(args.max_angular_margin_deg))
        )
    acc_centers = accepted_centers(grid)

    range_file = find_range_file(args.range_file, args.length_map)
    if range_file is None:
        raise FileNotFoundError("No encontré muon_range_table.csv/data_rock.dat para actualizar energía por CSDA.")

    hgt_dir = find_hgt_dir(args.hgt_dir, args.length_map)
    if args.external_length_mode in ("dem", "hybrid") and hgt_dir is None:
        raise FileNotFoundError("No encontré HGT para --external-length-mode dem/hybrid. Usa --hgt-dir.")

    if args.dry_run:
        print("[DRY-RUN] In-scattering angular-only")
        print(f"  point: {point}")
        print(f"  accepted cells: {int(grid.inside_mask.sum())}")
        print(f"  source domain F: theta=[{args.theta_min_deg}, {args.theta_max_deg}], phi=[{args.phi_min_deg}, {args.phi_max_deg}]")
        print(f"  margin restriction: {args.max_angular_margin_deg}")
        print(f"  external cells visible on length-map grid: {int(extended_external.sum())}")
        print(f"  range file: {range_file}")
        print(f"  hgt dir: {hgt_dir}")
        print("  spatial_detector_intersection_checked: false")
        return 0

    energy_loss, range_table_path = load_energy_loss(range_file, args.output_dir, args.rho)
    model = EVENT_MC.EmpiricalKernelModel(args.kernel_npz, args.interp_method, args.rbf_smoothing)
    print(
        f"[KERNEL] family={model.kernel_family} method={args.interp_method} "
        f"bins={len(model.centers_mrad)} support_mrad="
        f"[{model.edges_mrad[0]:g}, {model.edges_mrad[-1]:g}] "
        f"threshold={args.kernel_threshold:g}"
    )
    length_model = ExternalLengthModel(
        mode=args.external_length_mode,
        point=point,
        grid=grid,
        hgt_dir=hgt_dir,
        s_max_m=args.external_s_max_m,
        ray_step_m=args.external_ray_step_m,
        cache_step_deg=args.length_cache_step_deg,
    )
    rng = np.random.default_rng(args.seed)

    print("[INFO] In-scattering angular-only estimator")
    print("[INFO] spatial_detector_intersection_checked=false")
    print("[INFO] Se usa la máscara angular aceptada; no se verifica el área física del detector.")
    print(f"[INFO] point={point} accepted_cells={int(grid.inside_mask.sum())} external_cells_on_length_grid={int(extended_external.sum())}")
    print(f"[INFO] source domain F: theta=[{args.theta_min_deg}, {args.theta_max_deg}], phi=[{args.phi_min_deg}, {args.phi_max_deg}], margin={args.max_angular_margin_deg}")
    print(f"[INFO] external length mode={args.external_length_mode}, hgt_dir={hgt_dir}")

    counts = np.zeros_like(grid.L_grid, dtype=float)
    source_grid = build_source_grid(args, grid)
    source_counts = np.zeros((len(source_grid["theta"]), len(source_grid["phi"])), dtype=float)
    debug_rows: list[dict[str, object]] = []
    accepted_deflection_deg: list[float] = []
    accepted_energy_initial: list[float] = []
    accepted_length_m: list[float] = []

    stats = {
        "n_flux_events_read": 0,
        "n_discarded_upgoing": 0,
        "n_in_source_domain": 0,
        "n_initial_inside_acceptance_skipped": 0,
        "n_outside_margin_skipped": 0,
        "n_selected_external_muons": 0,
        "n_selected_without_positive_length": 0,
        "n_positive_length_external_muons": 0,
        "n_survival_prefilter_rejected": 0,
        "n_length_from_map": 0,
        "n_length_from_dem": 0,
        "n_samples_total": 0,
        "n_samples_not_survived": 0,
        "n_samples_survived": 0,
        "n_samples_accepted": 0,
        "n_accepted_source_map_misses": 0,
        "n_kernel_nearest_fallback_steps": 0,
        "n_kernel_outside_domain_steps": 0,
        "n_kernel_no_support_steps": 0,
    }

    processing_mode = "event-iterator"
    if args.kinematic_cache is not None and not args.debug_trajectories:
        processing_mode = "kinematic-cache-vectorized"
        print("[INFO] processing_mode=kinematic-cache-vectorized")
        process_kinematic_cache_fast(
            args,
            grid,
            source_grid,
            length_model,
            energy_loss,
            model,
            rng,
            counts,
            source_counts,
            accepted_deflection_deg,
            accepted_energy_initial,
            accepted_length_m,
            stats,
        )
    else:
        if args.kinematic_cache is not None and args.debug_trajectories:
            print("[INFO] processing_mode=event-iterator porque --debug-trajectories necesita guardar eventos aceptados.")
        event_iter = iter_shw_events(args, grid) if args.input_shw is not None else iter_kinematic_events(args, grid)
        sample_weight = 1.0 / float(args.n_samples_per_muon)
        for event in event_iter:
            stats["n_flux_events_read"] += 1
            if args.discard_upgoing and event.pz_positive:
                stats["n_discarded_upgoing"] += 1
                continue
            if not source_in_domain(event.theta_deg, event.phi_rel_deg, args):
                continue
            stats["n_in_source_domain"] += 1
            if is_inside_acceptance(grid, event.theta_deg, event.phi_rel_deg):
                stats["n_initial_inside_acceptance_skipped"] += 1
                continue

            if args.max_angular_margin_deg is not None:
                src_distance = distance_to_acceptance_deg(event.theta_deg, event.phi_rel_deg, acc_centers)
                if src_distance > float(args.max_angular_margin_deg):
                    stats["n_outside_margin_skipped"] += 1
                    continue
            else:
                src_distance = np.nan

            length_m, length_source = length_model.length_for(event.theta_deg, event.phi_rel_deg)
            if length_source == "length-map":
                stats["n_length_from_map"] += 1
            elif length_source == "dem":
                stats["n_length_from_dem"] += 1

            stats["n_selected_external_muons"] += 1
            if length_m <= 0.0 or not np.isfinite(length_m):
                stats["n_selected_without_positive_length"] += 1
                if args.head and stats["n_selected_external_muons"] >= args.head:
                    break
                continue
            stats["n_positive_length_external_muons"] += 1

            kinetic0 = float(event.total_E_GeV - MUON_MASS_GEV)
            for sample_idx in range(args.n_samples_per_muon):
                stats["n_samples_total"] += 1
                result = propagate_external_muon(
                    event.theta_deg,
                    event.phi_rel_deg,
                    kinetic0,
                    length_m,
                    model,
                    energy_loss,
                    rng,
                    args,
                    grid,
                )
                stats["n_kernel_nearest_fallback_steps"] += result.used_nearest
                stats["n_kernel_outside_domain_steps"] += result.outside_domain
                stats["n_kernel_no_support_steps"] += result.no_support
                if not result.survived:
                    stats["n_samples_not_survived"] += 1
                    continue
                stats["n_samples_survived"] += 1
                if not result.accepted or result.final_i is None or result.final_j is None:
                    continue

                stats["n_samples_accepted"] += 1
                counts[result.final_i, result.final_j] += sample_weight
                source_cell = cell_from_edges(
                    source_grid["theta_edges"],
                    source_grid["phi_edges"],
                    event.theta_deg,
                    event.phi_rel_deg,
                )
                if source_cell is None:
                    stats["n_accepted_source_map_misses"] += 1
                else:
                    source_counts[source_cell] += sample_weight
                accepted_deflection_deg.append(float(result.deflection_final_deg))
                accepted_energy_initial.append(float(kinetic0))
                accepted_length_m.append(float(length_m))
                if args.debug_trajectories:
                    debug_rows.append({
                        "theta_initial_deg": event.theta_deg,
                        "phi_initial_deg": event.phi_rel_deg,
                        "theta_final_deg": result.theta_final_deg,
                        "phi_final_deg": result.phi_final_deg,
                        "source_theta_index": "",
                        "source_phi_index": "",
                        "final_theta_index": result.final_i,
                        "final_phi_index": result.final_j,
                        "kinetic_initial_GeV": kinetic0,
                        "kinetic_final_GeV": result.kinetic_final_GeV,
                        "length_m": length_m,
                        "length_source": length_source,
                        "distance_source_to_acceptance_deg": float(src_distance),
                        "deflection_final_deg": result.deflection_final_deg,
                        "sample_index": sample_idx,
                    })

            if args.head and stats["n_selected_external_muons"] >= args.head:
                break

    masked_counts = np.where(grid.inside_mask, counts, np.nan)
    np.save(args.output_dir / "masked_counts_theta_phi.npy", masked_counts)
    np.save(args.output_dir / "source_counts_theta_phi.npy", source_counts)
    save_map_csv(args.output_dir / "masked_counts_theta_phi.csv", grid, masked_counts, extended_external, distance_deg)
    save_source_map_csv(args.output_dir / "source_counts_theta_phi.csv", source_grid, source_counts)
    if args.debug_trajectories:
        pd.DataFrame(debug_rows).to_csv(args.output_dir / "debug_accepted_trajectories.csv", index=False)

    figure_paths: dict[str, str] = {}
    if not args.no_figures:
        extended_png = args.output_dir / "extended_angular_region.png"
        accepted_png = args.output_dir / "in_scattering_accepted_map.png"
        source_png = args.output_dir / "in_scattering_source_map.png"
        plot_extended_region(extended_png, grid, extended_external)
        plot_counts(accepted_png, grid, counts)
        plot_source_counts(source_png, source_grid, source_counts, grid)
        figure_paths["extended_angular_region_png"] = str(extended_png)
        figure_paths["accepted_map_png"] = str(accepted_png)
        figure_paths["source_map_png"] = str(source_png)
        hist_specs = [
            ("final_deflection_hist_png", args.output_dir / "final_deflection_hist.png", accepted_deflection_deg, "Final angular displacement (deg)", "Accepted external muons"),
            ("initial_energy_hist_png", args.output_dir / "initial_energy_accepted_hist.png", accepted_energy_initial, "Initial kinetic energy (GeV)", "Accepted external muons"),
            ("length_hist_png", args.output_dir / "rock_length_accepted_hist.png", accepted_length_m, "Source-cell rock length (m)", "Accepted external muons"),
        ]
        for key, path, values, xlabel, title in hist_specs:
            if plot_hist(path, values, xlabel, title):
                figure_paths[key] = str(path)

    summary = {
        "created_at": now_stamp(),
        "module": Path(__file__).name,
        "point": point,
        "angular_only": True,
        "spatial_detector_intersection_checked": False,
        "physical_scope_note": (
            "Angular-only estimate: uses theta-phi mask and rock length map; "
            "does not verify 3D intersection with the detector physical area."
        ),
        "inputs": {
            "input_shw": str(args.input_shw) if args.input_shw is not None else None,
            "kinematic_cache": str(args.kinematic_cache) if args.kinematic_cache is not None else None,
            "kernel_npz": str(args.kernel_npz),
            "acceptance_map": str(args.acceptance_map),
            "length_map": str(args.length_map),
            "range_file": str(range_file),
            "range_table_used": str(range_table_path),
            "hgt_dir": str(hgt_dir) if hgt_dir is not None else None,
        },
        "parameters": {
            "processing_mode": processing_mode,
            "range_survival_prefilter": bool(processing_mode == "kinematic-cache-vectorized"),
            "step_m": float(args.step_m),
            "n_samples_per_muon": int(args.n_samples_per_muon),
            "max_angular_margin_deg": float(args.max_angular_margin_deg) if args.max_angular_margin_deg is not None else None,
            "seed": int(args.seed),
            "discard_upgoing": bool(args.discard_upgoing),
            "theta_min_deg": float(args.theta_min_deg) if args.theta_min_deg is not None else None,
            "theta_max_deg": float(args.theta_max_deg) if args.theta_max_deg is not None else None,
            "phi_min_deg": float(args.phi_min_deg) if args.phi_min_deg is not None else None,
            "phi_max_deg": float(args.phi_max_deg) if args.phi_max_deg is not None else None,
            "rho_g_cm3": float(args.rho),
            "external_length_mode": args.external_length_mode,
            "external_s_max_m": float(args.external_s_max_m),
            "external_ray_step_m": float(args.external_ray_step_m),
            "length_cache_step_deg": float(args.length_cache_step_deg),
            "interp_method": args.interp_method,
            "kernel_family": model.kernel_family,
            "kernel_tail_policy": "body_quantile_tail_histogram_linear" if args.interp_method == "tail-aware" else "legacy",
            "kernel_support_mrad": [float(model.edges_mrad[0]), float(model.edges_mrad[-1])],
            "kernel_energy_cache_dlog": float(model.tail_aware.energy_cache_dlog) if model.tail_aware is not None else 0.0,
            "kernel_threshold": float(args.kernel_threshold),
            "kernel_scale": float(args.kernel_scale),
            "disable_scattering": bool(args.disable_scattering),
        },
        "source_domain_definition": {
            "F": "theta/phi domain given by theta_min/max and phi_min/max",
            "A": "accepted angular mask from acceptance_map",
            "E": "F minus A, optionally restricted by max_angular_margin_deg",
            "full_complement_used": args.max_angular_margin_deg is None,
            "masked_counts_theta_phi": "final reconstructed accepted pixel after scattering",
            "source_counts_theta_phi": "initial external direction of muons that ended accepted",
        },
        "grid": {
            "theta_min_deg": float(grid.theta_edges[0]),
            "theta_max_deg": float(grid.theta_edges[-1]),
            "phi_min_deg": float(grid.phi_edges[0]),
            "phi_max_deg": float(grid.phi_edges[-1]),
            "n_theta": int(len(grid.theta)),
            "n_phi": int(len(grid.phi)),
            "accepted_cells": int(grid.inside_mask.sum()),
            "extended_external_cells": int(extended_external.sum()),
            "acceptance_mask_column": acc.get("mask_col", None),
        },
        "source_grid": {
            "theta_min_deg": float(source_grid["theta_edges"][0]),
            "theta_max_deg": float(source_grid["theta_edges"][-1]),
            "phi_min_deg": float(source_grid["phi_edges"][0]),
            "phi_max_deg": float(source_grid["phi_edges"][-1]),
            "n_theta": int(len(source_grid["theta"])),
            "n_phi": int(len(source_grid["phi"])),
            "theta_step_deg": float(EVENT_MC.bin_width(source_grid["theta"])),
            "phi_step_deg": float(EVENT_MC.bin_width(source_grid["phi"])),
        },
        "stats": stats,
        "weighted_accepted_count": float(np.nansum(masked_counts)),
        "weighted_accepted_source_count": float(np.nansum(source_counts)),
        "outputs": {
            "masked_counts_theta_phi_npy": str(args.output_dir / "masked_counts_theta_phi.npy"),
            "masked_counts_theta_phi_csv": str(args.output_dir / "masked_counts_theta_phi.csv"),
            "source_counts_theta_phi_npy": str(args.output_dir / "source_counts_theta_phi.npy"),
            "source_counts_theta_phi_csv": str(args.output_dir / "source_counts_theta_phi.csv"),
            **figure_paths,
        },
        "validation_expectations": {
            "max_angular_margin_deg_zero_should_select_no_external_margin_cells": bool(args.max_angular_margin_deg == 0.0 and int(extended_external.sum()) == 0),
            "disable_scattering_should_give_zero_or_near_zero_external_acceptance": bool(args.disable_scattering),
            "accepted_sources_should_be_near_acceptance_edge_checked_by_distance_to_acceptance_deg": True,
            "energy_survival_checked_with_csda": True,
            "external_complement_can_include_vertical_floor_directions_if_theta_range_includes_them": True,
        },
        "performance_notes": {
            "length_cache_entries": len(length_model.length_cache or {}),
            "vectorized_cache_path_used": bool(processing_mode == "kinematic-cache-vectorized"),
            "workers_used": 1,
        },
    }
    summary_path = args.output_dir / "in_scattering_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] In-scattering angular-only finished")
    print(f"  weighted accepted count: {summary['weighted_accepted_count']:.6g}")
    print(f"  map: {args.output_dir / 'masked_counts_theta_phi.csv'}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
