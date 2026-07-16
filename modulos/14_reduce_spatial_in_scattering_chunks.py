#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reduce chunk-parallel outputs from 13_apply_spatial_in_scattering_dem.py."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def centers_to_edges(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vals = np.unique(vals[np.isfinite(vals)])
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if vals.size == 1:
        step = 0.5
        return np.array([vals[0] - step / 2.0, vals[0] + step / 2.0], dtype=float)
    mids = 0.5 * (vals[:-1] + vals[1:])
    first = vals[0] - (mids[0] - vals[0])
    last = vals[-1] + (vals[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def plot_grid_csv(path: Path, csv_path: Path, count_col: str, title: str, cbar: str) -> None:
    df = pd.read_csv(csv_path)
    vals = df[count_col].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals) & (vals > 0.0)]
    if finite.size == 0:
        return
    theta_vals = np.sort(df["theta_deg"].unique())
    phi_vals = np.sort(df["phi_rel_deg"].unique())
    pivot = df.pivot(index="theta_deg", columns="phi_rel_deg", values=count_col).reindex(index=theta_vals, columns=phi_vals)
    z = pivot.to_numpy(dtype=float)
    theta_edges = centers_to_edges(theta_vals)
    phi_edges = centers_to_edges(phi_vals)
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    im = ax.pcolormesh(
        phi_edges,
        theta_edges,
        np.where(z > 0.0, z, np.nan),
        shading="flat",
        cmap="magma",
        norm=LogNorm(vmin=1.0, vmax=max(1.0, float(finite.max()))),
    )
    ax.set_xlim(float(phi_edges[0]), float(phi_edges[-1]))
    ax.set_ylim(float(theta_edges[-1]), float(theta_edges[0]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Relative azimuth $\phi$ (deg)")
    ax.set_ylabel(r"Zenith angle $\theta$ (deg)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.92)
    cb.set_label(cbar)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sum_numeric_stats(summaries: list[dict]) -> dict:
    keys = sorted(set().union(*(s.get("stats", {}).keys() for s in summaries)))
    out = {}
    for key in keys:
        vals = [s.get("stats", {}).get(key) for s in summaries]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals if v is not None):
            out[key] = int(sum(v for v in vals if v is not None))
    return out


def reduce_csv_counts(input_dirs: list[Path], filename: str, count_col: str, output_path: Path, final_masked: bool) -> None:
    frames = []
    for d in input_dirs:
        p = d / filename
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return
    base = frames[0].drop(columns=[count_col]).copy()
    counts = np.zeros(len(base), dtype=float)
    for df in frames:
        vals = df[count_col].fillna(0.0).to_numpy(dtype=float)
        counts += vals
    if final_masked and "inside_acceptance" in base.columns:
        inside = base["inside_acceptance"].to_numpy(dtype=int) == 1
        base[count_col] = np.where(inside, counts, np.nan)
    else:
        base[count_col] = counts
    base.to_csv(output_path, index=False)


def reduce_npy(input_dirs: list[Path], filename: str, output_path: Path, masked_nan: bool) -> None:
    arrays = []
    for d in input_dirs:
        p = d / filename
        if p.exists():
            arrays.append(np.load(p))
    if not arrays:
        return
    total = np.zeros_like(np.nan_to_num(arrays[0], nan=0.0), dtype=float)
    for arr in arrays:
        total += np.nan_to_num(arr, nan=0.0)
    if masked_nan:
        total = np.where(np.isnan(arrays[0]), np.nan, total)
    np.save(output_path, total)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Reduce spatial in-scattering chunk outputs.")
    ap.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--label", default="reduced")
    ap.add_argument("--no-figures", action="store_true")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    input_dirs = [p for p in args.input_dirs if (p / "spatial_in_scattering_summary.json").exists()]
    if not input_dirs:
        raise FileNotFoundError("No input summaries found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [json.load(open(d / "spatial_in_scattering_summary.json", encoding="utf-8")) for d in input_dirs]
    first = summaries[0]

    reduce_npy(input_dirs, "spatial_final_counts_theta_phi.npy", args.output_dir / "spatial_final_counts_theta_phi.npy", masked_nan=True)
    reduce_npy(input_dirs, "spatial_source_counts_theta_phi.npy", args.output_dir / "spatial_source_counts_theta_phi.npy", masked_nan=False)
    reduce_csv_counts(input_dirs, "spatial_final_counts_theta_phi.csv", "in_scattering_count", args.output_dir / "spatial_final_counts_theta_phi.csv", final_masked=True)
    reduce_csv_counts(input_dirs, "spatial_source_counts_theta_phi.csv", "in_scattering_source_count", args.output_dir / "spatial_source_counts_theta_phi.csv", final_masked=False)

    track_frames = []
    for d in input_dirs:
        p = d / "spatial_accepted_tracks.csv"
        if p.exists():
            df = pd.read_csv(p)
            if not df.empty:
                df["chunk_output_dir"] = str(d)
                track_frames.append(df)
    if track_frames:
        tracks = pd.concat(track_frames, ignore_index=True)
        tracks["accepted_id"] = np.arange(1, len(tracks) + 1, dtype=int)
    else:
        tracks = pd.DataFrame()
    tracks_path = args.output_dir / "spatial_accepted_tracks.csv"
    tracks.to_csv(tracks_path, index=False)

    target_src = input_dirs[0] / "volcano_surface_target_points.csv"
    target_dst = args.output_dir / "volcano_surface_target_points.csv"
    if target_src.exists():
        shutil.copy2(target_src, target_dst)

    accepted_weights = tracks["sample_weight"].to_numpy(dtype=float) if (not tracks.empty and "sample_weight" in tracks) else np.array([], dtype=float)
    accepted_sum = float(np.sum(accepted_weights)) if accepted_weights.size else 0.0
    accepted_sum_sq = float(np.sum(accepted_weights * accepted_weights)) if accepted_weights.size else 0.0
    eff_n = float((accepted_sum * accepted_sum) / accepted_sum_sq) if accepted_sum_sq > 0.0 else 0.0
    rel_se = float(math.sqrt(accepted_sum_sq) / accepted_sum) if accepted_sum > 0.0 else None

    stats = sum_numeric_stats(summaries)
    params = dict(first.get("parameters", {}))
    params["reduced_chunk_count"] = len(input_dirs)
    params.pop("chunk_index", None)

    area = params.get("volcano_surface_horizontal_area_m2")
    n_flux = stats.get("n_flux_events_read", 0)
    area_est = None
    if area is not None:
        area = float(area)
        lo = max(0.0, accepted_sum - 1.96 * math.sqrt(accepted_sum)) if accepted_sum > 0 else 0.0
        hi = accepted_sum + 1.96 * math.sqrt(accepted_sum) if accepted_sum > 0 else 0.0
        area_est = {
            "effective_area_m2": area,
            "accepted_count_90d": accepted_sum,
            "probability_per_flux_muon": float(accepted_sum / n_flux) if n_flux else None,
            "diagnostic_per_day_per_m2": float(accepted_sum / 90.0),
            "ideal_surface_rate_muons_per_m2_day": float(accepted_sum / 90.0),
            "ideal_surface_rate_definition": (
                "weighted_accepted_count / 90 days using the original 1 m2 CNF flux normalization; "
                "this is an ideal injection-surface rate, not a detector rate"
            ),
            "equivalent_area_scaled_exposure_days_for_mc_count": float(90.0 / area) if area > 0.0 else None,
            "equivalent_area_scaled_exposure_seconds_for_mc_count": (
                float(90.0 * 86400.0 / area) if area > 0.0 else None
            ),
            "equivalent_exposure_definition": (
                "time needed on the full effective injection area to accumulate the same weighted accepted "
                "count as the 90-day, 1 m2 Monte Carlo normalization"
            ),
            "area_scaled_count_90d": float(accepted_sum * area),
            "area_scaled_count_per_day": float(accepted_sum * area / 90.0),
            "poisson95_area_scaled_count_90d_low": float(lo * area),
            "poisson95_area_scaled_count_90d_high": float(hi * area),
            "poisson95_area_scaled_count_per_day_low": float(lo * area / 90.0),
            "poisson95_area_scaled_count_per_day_high": float(hi * area / 90.0),
        }

    rock_lengths = tracks["rock_length_m"].to_numpy(dtype=float) if (not tracks.empty and "rock_length_m" in tracks) else np.array([])
    summary = {
        "created_at": now_stamp(),
        "module": Path(__file__).name,
        "label": args.label,
        "reduced_from": [str(d) for d in input_dirs],
        "point": first.get("point"),
        "spatial_positions_sampled": first.get("spatial_positions_sampled", True),
        "detector_intersection_checked": first.get("detector_intersection_checked", False),
        "physical_scope_note": first.get("physical_scope_note"),
        "inputs": first.get("inputs", {}),
        "parameters": params,
        "stats": stats,
        "weighted_accepted_count": accepted_sum,
        "weighted_accepted_source_count": accepted_sum,
        "accepted_weight_sum_sq": accepted_sum_sq,
        "accepted_effective_sample_size": eff_n,
        "accepted_relative_mc_se": rel_se,
        "accepted_rock_length_m": {
            "count": int(rock_lengths.size),
            "median": float(np.median(rock_lengths)) if rock_lengths.size else None,
            "p16": float(np.percentile(rock_lengths, 16)) if rock_lengths.size else None,
            "p84": float(np.percentile(rock_lengths, 84)) if rock_lengths.size else None,
        },
        "area_effective_estimate": area_est,
        "outputs": {
            "spatial_final_counts_theta_phi_npy": str(args.output_dir / "spatial_final_counts_theta_phi.npy"),
            "spatial_final_counts_theta_phi_csv": str(args.output_dir / "spatial_final_counts_theta_phi.csv"),
            "spatial_source_counts_theta_phi_npy": str(args.output_dir / "spatial_source_counts_theta_phi.npy"),
            "spatial_source_counts_theta_phi_csv": str(args.output_dir / "spatial_source_counts_theta_phi.csv"),
            "spatial_accepted_tracks_csv": str(tracks_path),
            "volcano_surface_target_points_csv": str(target_dst) if target_dst.exists() else None,
        },
    }

    if not args.no_figures:
        point_label = str(first.get("point", "P"))
        final_png = args.output_dir / "spatial_final_accepted_map.png"
        source_png = args.output_dir / "spatial_source_external_map.png"
        plot_grid_csv(final_png, args.output_dir / "spatial_final_counts_theta_phi.csv", "in_scattering_count", "Spatial DEM: final directions inside volcano mask", "Counts")
        plot_grid_csv(source_png, args.output_dir / "spatial_source_counts_theta_phi.csv", "in_scattering_source_count", "Spatial DEM: initial external directions that scatter inside", "Counts")
        summary["outputs"]["spatial_final_accepted_map_png"] = str(final_png)
        summary["outputs"]["spatial_source_external_map_png"] = str(source_png)
        if not tracks.empty:
            contact_png = args.output_dir / "spatial_first_rock_contact_xy.png"
            fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
            sc = ax.scatter(tracks["first_rock_x_m"] / 1000.0, tracks["first_rock_y_m"] / 1000.0, c=tracks["rock_length_m"], s=34, cmap="magma", alpha=0.86, edgecolor="white", linewidth=0.35)
            ax.scatter([0.0], [0.0], marker="*", s=130, color="black", label=point_label)
            ax.set_xlabel(f"East from {point_label} (km)")
            ax.set_ylabel(f"North from {point_label} (km)")
            ax.set_title("Accepted in-scattering first DEM-rock contact points")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.colorbar(sc, ax=ax, shrink=0.92).set_label("Rock length (m)")
            fig.savefig(contact_png, bbox_inches="tight")
            plt.close(fig)
            summary["outputs"]["spatial_first_rock_contact_xy_png"] = str(contact_png)
            hist_png = args.output_dir / "spatial_accepted_rock_length_hist.png"
            fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
            ax.hist(rock_lengths, bins=50, histtype="stepfilled", alpha=0.85)
            ax.set_xlabel("DEM rock path length (m)")
            ax.set_ylabel("Accepted tracks")
            ax.set_title("Spatial DEM accepted track lengths")
            fig.savefig(hist_png, bbox_inches="tight")
            plt.close(fig)
            summary["outputs"]["spatial_accepted_rock_length_hist_png"] = str(hist_png)
        if target_dst.exists():
            target = pd.read_csv(target_dst)
            target_png = args.output_dir / "volcano_surface_target_points_xy.png"
            fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
            sc = ax.scatter(target["x_m"] / 1000.0, target["y_m"] / 1000.0, c=target["height_fraction"], s=12, cmap="viridis", alpha=0.82, linewidths=0.0)
            ax.scatter([0.0], [0.0], marker="*", s=130, color="black", label=point_label)
            ax.set_xlabel(f"East from {point_label} (km)")
            ax.set_ylabel(f"North from {point_label} (km)")
            ax.set_title("Volcano-surface target points selected from DEM and angular mask")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.colorbar(sc, ax=ax, shrink=0.92).set_label("DEM height fraction")
            fig.savefig(target_png, bbox_inches="tight")
            plt.close(fig)
            summary["outputs"]["volcano_surface_target_points_xy_png"] = str(target_png)

    with (args.output_dir / "spatial_in_scattering_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] reduced {len(input_dirs)} chunks -> {args.output_dir}")
    print(f"  accepted count: {accepted_sum:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
