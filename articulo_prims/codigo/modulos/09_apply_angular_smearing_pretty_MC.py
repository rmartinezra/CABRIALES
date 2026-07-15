#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_apply_angular_smearing_pretty_v2.py
-------------------------------------
Apply Gaussian angular smearing to theta-phi muography maps.

New features compared with earlier versions:
- clearer plotting defaults (same angular window / square display style as article-like maps)
- optional stochastic smearing realization (multinomial per source bin) if you want a noisier map
- optional second figure for a pre-filtered 'inside volcano' map, so you can compare
  the counts inside the volcano before and after angular smearing.

Conceptually:
- deterministic mode computes the expected smeared map (smooth by construction)
- stochastic mode draws one Monte Carlo realization of that redistribution (noisier)

Example, single point with an extra inside-volcano map:
python3 09_apply_angular_smearing_pretty_v2.py \
  --map-csv ./run_machin/05_plots/filtered/theta_phi_counts_P4.csv \
  --inside-map-csv ./run_machin/06_inside_volcano/counts_inside_volcano_P4.csv \
  --scat-csv ./run_machin/07_scattering/P4/scattering_table_P4_f1p00.csv \
  --point P4 \
  --outdir ./run_machin/08_smearing \
  --stochastic --random-seed 12345

Example, batch mode with inside-volcano maps:
python3 09_apply_angular_smearing_pretty_v2.py \
  --points P1 P2 P4 P5 \
  --energy-factors 1.0 1.5 2.0 \
  --map-template './run_machin/05_plots/filtered/theta_phi_counts_{point}.csv' \
  --inside-map-template './run_machin/06_inside_volcano/counts_inside_volcano_{point}.csv' \
  --scat-template './run_machin/07_scattering/{point}/scattering_table_{point}_{tag}.csv' \
  --outdir ./run_machin/08_smearing
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

DEFAULT_POINTS = ("P1", "P2", "P4", "P5")
VALUE_CANDIDATES = (
    "count", "counts", "count_inside_geometry", "count_all_in_grid",
    "dN_dOmega_count_per_sr", "dN_dOmega_inside_count_per_sr",
    "flux", "intensity", "N_abs", "N",
)


def setup_style() -> None:
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

    th = np.array(sorted(df[th_col].unique()), dtype=float)
    ph = np.array(sorted(df[ph_col].unique()), dtype=float)
    H = np.zeros((len(th), len(ph)), dtype=float)
    filled = np.zeros_like(H, dtype=bool)
    ti = {round(v, 10): i for i, v in enumerate(th)}
    pj = {round(v, 10): j for j, v in enumerate(ph)}

    for _, r in df.iterrows():
        i = ti.get(round(float(r[th_col]), 10))
        j = pj.get(round(float(r[ph_col]), 10))
        if i is not None and j is not None:
            H[i, j] = float(r[val_col])
            filled[i, j] = True

    return {
        "theta": th,
        "phi": ph,
        "theta_edges": centers_to_edges(th, bin_width(th)),
        "phi_edges": centers_to_edges(ph, bin_width(ph)),
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
    return out, mt, mp


def read_sigma(scat_csv: Path, full_info: dict, sigma_col: str, theta_min, theta_max, phi_min, phi_max):
    df = pd.read_csv(scat_csv)
    th_col = find_col(df, ["theta_deg", "theta", "zenith_deg"])
    ph_col = find_col(df, ["phi_rel_deg", "phi_deg", "phi", "azimuth_deg"])
    sig_col = sigma_col if sigma_col in df.columns else find_col(df, [sigma_col])

    df = df.copy()
    for c in (th_col, ph_col, sig_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[th_col, ph_col, sig_col])

    if theta_min is not None:
        df = df[df[th_col] >= theta_min]
    if theta_max is not None:
        df = df[df[th_col] <= theta_max]
    if phi_min is not None:
        df = df[df[ph_col] >= phi_min]
    if phi_max is not None:
        df = df[df[ph_col] <= phi_max]

    sigma = np.zeros_like(full_info["H"], dtype=float)
    has = np.zeros_like(full_info["H"], dtype=bool)

    for _, r in df.iterrows():
        i = np.searchsorted(full_info["theta_edges"], float(r[th_col]), side="right") - 1
        j = np.searchsorted(full_info["phi_edges"], float(r[ph_col]), side="right") - 1
        sig = float(r[sig_col])
        if 0 <= i < sigma.shape[0] and 0 <= j < sigma.shape[1] and np.isfinite(sig) and sig > 0:
            sigma[i, j] = max(sigma[i, j], sig)
            has[i, j] = True
    return sigma, has


def dphi_array(phi_grid, phi0, wrap_phi: bool):
    d = phi_grid - phi0
    if wrap_phi:
        d = (d + 180.0) % 360.0 - 180.0
    return d


def compute_kernel(theta, phi, filled, i_src, j_src, sig, kernel_radius_sigma, wrap_phi=False):
    theta0 = float(theta[i_src])
    phi0 = float(phi[j_src])
    sin_th = abs(math.sin(math.radians(theta0)))
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    radius = kernel_radius_sigma * sig
    dth = TH - theta0
    dph = dphi_array(PH, phi0, wrap_phi)
    da2 = dth * dth + (sin_th * dph) * (sin_th * dph)
    mask = (da2 <= radius * radius) & filled
    if not np.any(mask):
        return None
    W = np.zeros((len(theta), len(phi)), dtype=float)
    W[mask] = np.exp(-0.5 * da2[mask] / (sig * sig))
    s = float(np.nansum(W))
    if s <= 0 or not np.isfinite(s):
        return None
    W /= s
    return W


def maybe_round_nonnegative(value: float) -> int:
    return int(max(0, round(float(value))))


def smear(H, theta, phi, sigma_deg, has_sigma, filled,
          kernel_radius_sigma=4.0, detector_sigma_deg=0.0,
          sigma_scale=1.0, wrap_phi=False, renormalize=True,
          min_sigma_deg=1e-6, stochastic=False, rng=None):
    H = np.asarray(H, dtype=float)
    out = np.zeros_like(H, dtype=float)
    n_smeared = n_identity = n_sources = 0

    for i in range(len(theta)):
        for j in range(len(phi)):
            val = H[i, j]
            if not filled[i, j] or not np.isfinite(val) or val == 0:
                continue
            n_sources += 1
            sig_sc = sigma_deg[i, j] if has_sigma[i, j] else 0.0
            sig = math.sqrt((sigma_scale * sig_sc) ** 2 + detector_sigma_deg ** 2)
            if (not np.isfinite(sig)) or sig <= min_sigma_deg:
                out[i, j] += val
                n_identity += 1
                continue

            W = compute_kernel(theta, phi, filled, i, j, sig, kernel_radius_sigma, wrap_phi=wrap_phi)
            if W is None:
                out[i, j] += val
                n_identity += 1
                continue

            if stochastic:
                n = maybe_round_nonnegative(val)
                if n == 0:
                    continue
                probs = W.ravel()
                sampled = rng.multinomial(n=n, pvals=probs)
                out += sampled.reshape(W.shape)
            else:
                if renormalize:
                    out += val * W
                else:
                    out += val * (W * np.nansum(W))
            n_smeared += 1

    return out, {
        "n_sources_nonzero": n_sources,
        "n_sources_identity": n_identity,
        "n_sources_smeared": n_smeared,
    }


def output_table(theta, phi, H_in, H_out, sigma, has):
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
        "sigma_smearing_deg": sigma.ravel(),
        "has_scattering_sigma": has.ravel().astype(int),
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
                    rel_vmax_percentile=98.0, title_prefix="Angular smearing diagnostic"):
    delta = H_out - H_in
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = delta / H_in
        rel[~np.isfinite(rel)] = np.nan

    panels = [
        (H_in, "Input map", value_label, False),
        (H_out, "After angular smearing", value_label, False),
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


def process_dataset(map_csv, scat_csv, point, outdir, tag, args, dataset_name,
                    title_prefix, prefix_stem, rng):
    raw = read_map(map_csv, args.value_col)
    tmin = np.nanmin(raw["theta"]) if args.theta_min is None else args.theta_min
    tmax = np.nanmax(raw["theta"]) if args.theta_max is None else args.theta_max
    pmin = np.nanmin(raw["phi"]) if args.phi_min is None else args.phi_min
    pmax = np.nanmax(raw["phi"]) if args.phi_max is None else args.phi_max

    full_sigma, full_has = read_sigma(scat_csv, raw, args.sigma_col, tmin, tmax, pmin, pmax)
    info, mt, mp = cut_window(raw, tmin, tmax, pmin, pmax)
    sigma = full_sigma[np.ix_(mt, mp)]
    has = full_has[np.ix_(mt, mp)]

    H_in = info["H"]
    H_out, stats = smear(
        H_in, info["theta"], info["phi"], sigma, has, info["filled"],
        args.kernel_radius_sigma, args.detector_sigma_deg, args.sigma_scale,
        args.wrap_phi, args.renormalize, stochastic=args.stochastic, rng=rng,
    )

    point_dir = outdir / point
    point_dir.mkdir(parents=True, exist_ok=True)

    out_csv = point_dir / f"{prefix_stem}_table_{point}_{tag}.csv"
    output_table(info["theta"], info["phi"], H_in, H_out, sigma, has).to_csv(out_csv, index=False)

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

    summary = {
        "dataset": dataset_name,
        "point": point,
        "tag": tag,
        "map_csv": str(map_csv),
        "scat_csv": str(scat_csv),
        "value_col": info["value_col"],
        "sigma_col": args.sigma_col,
        "theta_min": tmin,
        "theta_max": tmax,
        "phi_min": pmin,
        "phi_max": pmax,
        "stochastic": int(args.stochastic),
        "n_theta_bins": len(info["theta"]),
        "n_phi_bins": len(info["phi"]),
        "n_cells_with_scattering_sigma": int(np.sum(has)),
        "input_total": total_in,
        "smeared_total": total_out,
        "relative_total_change": (total_out - total_in) / total_in if total_in else np.nan,
        "p90_abs_relative_delta": float(np.nanpercentile(np.abs(finite_rel), 90)) if finite_rel.size else np.nan,
        "p99_abs_relative_delta": float(np.nanpercentile(np.abs(finite_rel), 99)) if finite_rel.size else np.nan,
        "output_csv": str(out_csv),
        "comparison_png": str(comparison_png),
        **stats,
    }
    print(f"[OK] {dataset_name} {point} {tag}: total_in={total_in:.6g}, total_out={total_out:.6g}, rel={summary['relative_total_change']:.3e}")
    return summary


def parser():
    ap = argparse.ArgumentParser(description="Aplica smearing angular gaussiano a un mapa theta-phi.")
    ap.add_argument("--map-csv", type=Path, default=None, help="Mapa angular principal para una corrida.")
    ap.add_argument("--inside-map-csv", type=Path, default=None, help="Mapa angular ya filtrado dentro del volcán (opcional).")
    ap.add_argument("--scat-csv", type=Path, default=None, help="Tabla de scattering para una corrida.")
    ap.add_argument("--point", default=None)
    ap.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS))
    ap.add_argument("--energy-factors", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    ap.add_argument("--map-template", default=None)
    ap.add_argument("--inside-map-template", default=None)
    ap.add_argument("--scat-template", default=None)
    ap.add_argument("--outdir", type=Path, default=Path("outputs_smearing"))
    ap.add_argument("--value-col", default=None)
    ap.add_argument("--sigma-col", default="theta0_proj_deg")

    ap.add_argument("--theta-min", type=float, default=None, help="Mínimo theta a graficar/usar. Default: inferido desde el CSV")
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

    ap.add_argument("--kernel-radius-sigma", type=float, default=4.0)
    ap.add_argument("--detector-sigma-deg", type=float, default=0.0)
    ap.add_argument("--sigma-scale", type=float, default=1.0)
    ap.add_argument("--no-renormalize", dest="renormalize", action="store_false")
    ap.set_defaults(renormalize=True)
    ap.add_argument("--wrap-phi", action="store_true")

    ap.add_argument("--stochastic", action="store_true", help="Genera una realización Monte Carlo ruidosa del smearing.")
    ap.add_argument("--random-seed", type=int, default=12345, help="Semilla del RNG para el modo --stochastic.")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    jobs = []
    if args.map_csv or args.scat_csv or args.inside_map_csv:
        if not (args.map_csv and args.scat_csv):
            raise SystemExit("Usa --map-csv y --scat-csv juntos en modo simple.")
        jobs.append({
            "point": args.point or "POINT",
            "tag": infer_tag_from_path(args.scat_csv),
            "map_csv": args.map_csv,
            "inside_map_csv": args.inside_map_csv,
            "scat_csv": args.scat_csv,
        })
    else:
        if args.map_template is None:
            raise SystemExit("Modo lote requiere --map-template, o usa --map-csv/--scat-csv.")
        scat_template = args.scat_template or "outputs_scattering/{point}/scattering_table_{point}_{tag}.csv"
        for point in args.points:
            for factor in args.energy_factors:
                tag = factor_tag(factor)
                inside_map_csv = None
                if args.inside_map_template is not None:
                    inside_map_csv = Path(args.inside_map_template.format(point=point, factor=factor, tag=tag))
                jobs.append({
                    "point": point,
                    "tag": tag,
                    "map_csv": Path(args.map_template.format(point=point, factor=factor, tag=tag)),
                    "inside_map_csv": inside_map_csv,
                    "scat_csv": Path(scat_template.format(point=point, factor=factor, tag=tag)),
                })

    summaries = []
    for job in jobs:
        point = job["point"]
        tag = job["tag"]
        map_csv = job["map_csv"]
        scat_csv = job["scat_csv"]
        inside_map_csv = job["inside_map_csv"]

        if not map_csv.exists():
            print(f"[WARN] No existe map CSV: {map_csv}. Salto.")
            continue
        if not scat_csv.exists():
            print(f"[WARN] No existe scattering CSV: {scat_csv}. Salto.")
            continue

        summaries.append(process_dataset(
            map_csv=map_csv, scat_csv=scat_csv, point=point, outdir=args.outdir, tag=tag,
            args=args, dataset_name="full_map", title_prefix="Angular smearing diagnostic",
            prefix_stem="smearing", rng=rng,
        ))

        if inside_map_csv is not None:
            if inside_map_csv.exists():
                summaries.append(process_dataset(
                    map_csv=inside_map_csv, scat_csv=scat_csv, point=point, outdir=args.outdir, tag=tag,
                    args=args, dataset_name="inside_volcano", title_prefix="Inside-volcano counts after smearing",
                    prefix_stem="inside_volcano_smearing", rng=rng,
                ))
            else:
                print(f"[WARN] No existe inside-map CSV: {inside_map_csv}. Se omite la figura inside-volcano.")

    if summaries:
        summary_csv = args.outdir / "smearing_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_csv, index=False)
        print(f"[DONE] Summary: {summary_csv}")
    else:
        print("[WARN] No se generaron salidas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
