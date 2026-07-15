#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the full 90-day spatial in-scattering DEM calculation for P1/P2/P4/P5.

This is a reproducibility wrapper around:

* modulos/13_apply_spatial_in_scattering_dem.py
* modulos/14_reduce_spatial_in_scattering_chunks.py

It keeps each point independent, runs chunk workers in parallel per point, and
writes one reduced directory per point plus a combined campaign summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

from progress import format_duration


POINTS = ("P1", "P2", "P4", "P5")


@dataclass(frozen=True)
class TransportConfig:
    sample_probability: float = 1.0
    head: int = 0
    base_seed: int = 12345
    ray_step_m: float = 100.0
    max_track_m: float = 9000.0
    grid_step_m: float = 50.0
    edge_guard_m: float = 500.0
    min_height_frac: float = 0.15
    start_offset_m: float = 1.0
    entry_check_m: float = 10.0
    theta_max_deg: float = 90.0
    max_angular_margin_deg: float = 5.0
    min_survival_rock_m: float = 100.0
    kernel_scale: float = 1.0
    disable_scattering: bool = False


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run full spatial in-scattering campaign for Machin points.")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("run_machin90dia_allpoints_full/10_in_scattering_background/machin90d_4points_volcano_surface_workers8"),
    )
    ap.add_argument("--points", nargs="+", choices=POINTS, default=list(POINTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="Remove existing per-point output before running it.")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--continue-on-existing", action="store_true", help="Reuse already reduced points when present.")
    ap.add_argument(
        "--kinematic-cache",
        type=Path,
        default=Path(
            os.environ.get(
                "CABRIALES_90D_CACHE",
                "/home/rafael/proyectos/CNF/muon-cnf-toolkit/machin90dia_kinematic_cache",
            )
        ),
    )
    ap.add_argument("--kernel-npz", type=Path, default=Path("modulos/hybrid_empirical_kernel_library.npz"))
    ap.add_argument("--interp-method", choices=["tail-aware", "linear", "rbf_linear", "nearest"], default="tail-aware")
    ap.add_argument("--kernel-threshold", type=float, default=0.0)
    ap.add_argument("--ecrit-root", type=Path, default=Path("run_machin90dia_allpoints_full/03_ecrit"))
    ap.add_argument("--hgt-dir", default="data")
    ap.add_argument("--range-file", type=Path, default=Path("data/data_rock.dat"))
    ap.add_argument("--sample-probability", type=float, default=1.0)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ray-step-m", type=float, default=100.0)
    ap.add_argument("--max-track-m", type=float, default=9000.0)
    ap.add_argument("--volcano-surface-grid-step-m", type=float, default=50.0)
    ap.add_argument("--volcano-surface-edge-guard-m", type=float, default=500.0)
    ap.add_argument("--volcano-surface-min-height-frac", type=float, default=0.15)
    ap.add_argument("--volcano-surface-start-offset-m", type=float, default=1.0)
    ap.add_argument("--volcano-surface-entry-check-m", type=float, default=10.0)
    ap.add_argument("--theta-max-deg", type=float, default=90.0)
    ap.add_argument("--max-angular-margin-deg", type=float, default=5.0)
    ap.add_argument("--min-survival-rock-m", type=float, default=100.0)
    ap.add_argument("--kernel-scale", type=float, default=1.0)
    ap.add_argument("--disable-scattering", action="store_true")
    ap.add_argument(
        "--status-interval-s",
        type=float,
        default=30.0,
        help="Seconds between progress messages while chunk workers are active; 0 disables heartbeats.",
    )
    return ap


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_root(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def check_inputs(root: Path, args: argparse.Namespace) -> None:
    required = [
        args.kinematic_cache,
        args.kernel_npz,
        args.ecrit_root,
        Path(args.hgt_dir),
        args.range_file,
    ]
    for p in required:
        q = resolve_from_root(root, Path(p))
        if not q.exists():
            raise FileNotFoundError(f"Missing required input: {q}")
    for point in args.points:
        p = args.ecrit_root / f"ecrit_table_{point}.csv"
        q = resolve_from_root(root, p)
        if not q.exists():
            raise FileNotFoundError(f"Missing ecrit table for {point}: {q}")


def point_seed(base_seed: int, point: str) -> int:
    return base_seed + 100000 * POINTS.index(point)


def run_command(
    cmd: list[str],
    log_path: Path,
    cwd: Path,
    *,
    label: str | None = None,
    status_interval_s: float = 0.0,
) -> int:
    """Run a command with full output in a log and optional terminal heartbeat."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if label:
        print(f"[START] {label} | log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
        while True:
            if status_interval_s <= 0:
                returncode = proc.wait()
                break
            try:
                returncode = proc.wait(timeout=status_interval_s)
                break
            except subprocess.TimeoutExpired:
                print(
                    f"[RUNNING] {label or Path(cmd[0]).name} "
                    f"| elapsed={format_duration(time.monotonic() - started)} "
                    f"| log={log_path}",
                    flush=True,
                )
    if label:
        state = "OK" if returncode == 0 else f"ERROR rc={returncode}"
        print(
            f"[{state}] {label} | elapsed={format_duration(time.monotonic() - started)}",
            flush=True,
        )
    return int(returncode)


def path_arg(root: Path, p: Path) -> str:
    return str(resolve_from_root(root, p))


def build_worker_cmd(
    root: Path,
    args: argparse.Namespace,
    cfg: TransportConfig,
    point: str,
    worker_index: int,
    chunk_dir: Path,
) -> list[str]:
    ecrit = args.ecrit_root / f"ecrit_table_{point}.csv"
    cmd = [
        sys.executable,
        str(root / "modulos/13_apply_spatial_in_scattering_dem.py"),
        "--kinematic-cache", path_arg(root, args.kinematic_cache),
        "--kernel-npz", path_arg(root, args.kernel_npz),
        "--acceptance-map", path_arg(root, ecrit),
        "--length-map", path_arg(root, ecrit),
        "--output-dir", str(resolve_from_root(root, chunk_dir)),
        "--point", point,
        "--hgt-dir", path_arg(root, Path(args.hgt_dir)),
        "--range-file", path_arg(root, args.range_file),
        "--head", str(cfg.head),
        "--chunk-count", str(args.workers),
        "--chunk-index", str(worker_index),
        "--seed", str(point_seed(cfg.base_seed, point)),
        "--ray-step-m", str(cfg.ray_step_m),
        "--max-track-m", str(cfg.max_track_m),
        "--source-surface", "volcano-surface",
        "--volcano-surface-grid-step-m", str(cfg.grid_step_m),
        "--volcano-surface-edge-guard-m", str(cfg.edge_guard_m),
        "--volcano-surface-min-height-frac", str(cfg.min_height_frac),
        "--volcano-surface-start-offset-m", str(cfg.start_offset_m),
        "--volcano-surface-entry-check-m", str(cfg.entry_check_m),
        "--theta-max-deg", str(cfg.theta_max_deg),
        "--max-angular-margin-deg", str(cfg.max_angular_margin_deg),
        "--min-survival-rock-m", str(cfg.min_survival_rock_m),
        "--sample-probability", str(cfg.sample_probability),
        "--kernel-scale", str(cfg.kernel_scale),
        "--interp-method", str(args.interp_method),
        "--kernel-threshold", str(args.kernel_threshold),
        "--no-progress",
        "--no-figures",
    ]
    if cfg.disable_scattering:
        cmd.append("--disable-scattering")
    return cmd


def summarize_point(point_dir: Path, point: str) -> dict[str, object]:
    summary_path = point_dir / "reduced" / "spatial_in_scattering_summary.json"
    if not summary_path.exists():
        return {"point": point, "status": "missing_summary", "summary_json": str(summary_path)}
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    stats = summary.get("stats", {})
    area = summary.get("area_effective_estimate") or {}
    return {
        "point": point,
        "status": "ok",
        "weighted_accepted_count": summary.get("weighted_accepted_count"),
        "accepted_effective_sample_size": summary.get("accepted_effective_sample_size"),
        "accepted_relative_mc_se": summary.get("accepted_relative_mc_se"),
        "n_flux_events_read": stats.get("n_flux_events_read"),
        "n_initial_inside_acceptance_skipped": stats.get("n_initial_inside_acceptance_skipped"),
        "n_outside_angular_margin": stats.get("n_outside_angular_margin"),
        "n_tracks_touched_rock": stats.get("n_tracks_touched_rock"),
        "n_tracks_not_survived": stats.get("n_tracks_not_survived"),
        "n_tracks_survived": stats.get("n_tracks_survived"),
        "n_tracks_final_inside_acceptance": stats.get("n_tracks_final_inside_acceptance"),
        "effective_area_m2": area.get("effective_area_m2"),
        "area_scaled_count_90d": area.get("area_scaled_count_90d"),
        "area_scaled_count_per_day": area.get("area_scaled_count_per_day"),
        "poisson95_area_scaled_count_per_day_low": area.get("poisson95_area_scaled_count_per_day_low"),
        "poisson95_area_scaled_count_per_day_high": area.get("poisson95_area_scaled_count_per_day_high"),
        "summary_json": str(summary_path),
    }


def run_point(root: Path, args: argparse.Namespace, cfg: TransportConfig, point: str) -> dict[str, object]:
    point_dir = resolve_from_root(root, args.out_root / point)
    reduced_summary = point_dir / "reduced" / "spatial_in_scattering_summary.json"
    if reduced_summary.exists() and args.continue_on_existing and not args.force:
        print(f"[SKIP] {point}: reduced output already exists")
        return summarize_point(point_dir, point)
    if point_dir.exists() and args.force:
        shutil.rmtree(point_dir)
    point_dir.mkdir(parents=True, exist_ok=True)
    chunks_root = point_dir / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "point": point,
        "workers": args.workers,
        "point_seed": point_seed(cfg.base_seed, point),
        "transport_config": asdict(cfg),
        "kernel": {
            "path": str(resolve_from_root(root, args.kernel_npz)),
            "interp_method": args.interp_method,
            "kernel_threshold": float(args.kernel_threshold),
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (point_dir / "campaign_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"[RUN] {point}: workers={args.workers} "
        f"seed={point_seed(cfg.base_seed, point)} p={cfg.sample_probability}",
        flush=True,
    )
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_worker = {}
        for worker in range(args.workers):
            chunk_dir = chunks_root / f"w{worker:02d}"
            cmd = build_worker_cmd(root, args, cfg, point, worker, chunk_dir)
            chunk_dir.mkdir(parents=True, exist_ok=True)
            (chunk_dir / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
            future = executor.submit(run_command, cmd, chunk_dir / "run.log", root)
            future_to_worker[future] = worker

        failed = []
        completed = 0
        pending = set(future_to_worker)
        while pending:
            timeout = args.status_interval_s if args.status_interval_s > 0 else None
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                print(
                    f"[RUNNING] {point}: chunks={completed}/{args.workers} "
                    f"elapsed={format_duration(time.monotonic() - t0)}",
                    flush=True,
                )
                continue
            for future in done:
                worker = future_to_worker[future]
                try:
                    rc = future.result()
                except Exception as exc:  # Preserve the worker identity in terminal output.
                    print(f"[ERROR] {point}/w{worker:02d}: {exc}", flush=True)
                    rc = 1
                if rc != 0:
                    failed.append((worker, rc))
                completed += 1
            print(
                f"[PROGRESS] {point}: chunks={completed}/{args.workers} "
                f"elapsed={format_duration(time.monotonic() - t0)}",
                flush=True,
            )
    if failed:
        print(f"[FAIL] {point}: chunk failures {failed}")
        return {
            "point": point,
            "status": "failed_chunks",
            "failures": failed,
            "elapsed_s": time.monotonic() - t0,
        }

    input_dirs = [chunks_root / f"w{i:02d}" for i in range(args.workers)]
    reduce_cmd = [
        sys.executable,
        str(root / "modulos/14_reduce_spatial_in_scattering_chunks.py"),
        "--output-dir", str(point_dir / "reduced"),
        "--label", f"{point}_machin90d_spatial_in_scattering",
        "--input-dirs",
        *map(str, input_dirs),
    ]
    if args.no_figures:
        reduce_cmd.append("--no-figures")
    rc = run_command(
        reduce_cmd,
        point_dir / "reduce.log",
        root,
        label=f"{point}/reduce",
        status_interval_s=args.status_interval_s,
    )
    if rc != 0:
        print(f"[FAIL] {point}: reducer rc={rc}")
        return {
            "point": point,
            "status": "failed_reduce",
            "elapsed_s": time.monotonic() - t0,
        }
    row = summarize_point(point_dir, point)
    row["elapsed_s"] = time.monotonic() - t0
    print(
        f"[OK] {point}: accepted={row.get('weighted_accepted_count')} "
        f"area/day={row.get('area_scaled_count_per_day')} elapsed={row['elapsed_s']/60.0:.1f} min"
    )
    return row


def write_campaign_summary(out_root: Path, rows: list[dict[str, object]], cfg: TransportConfig, args: argparse.Namespace, root: Path) -> None:
    out_root = resolve_from_root(root, out_root)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    totals = {
        "weighted_accepted_count": sum(float(r.get("weighted_accepted_count") or 0.0) for r in ok_rows),
        "area_scaled_count_90d": sum(float(r.get("area_scaled_count_90d") or 0.0) for r in ok_rows),
        "area_scaled_count_per_day": sum(float(r.get("area_scaled_count_per_day") or 0.0) for r in ok_rows),
        "effective_area_m2": sum(float(r.get("effective_area_m2") or 0.0) for r in ok_rows),
    }
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "module": Path(__file__).name,
        "points": list(args.points),
        "workers_per_point": args.workers,
        "transport_config": asdict(cfg),
        "kernel": {
            "path": str(resolve_from_root(root, args.kernel_npz)),
            "interp_method": args.interp_method,
            "kernel_threshold": float(args.kernel_threshold),
            "tail_policy": "tail-aware_full-tail-domain__core-rbf_broad-domain" if args.interp_method == "tail-aware" else "legacy",
        },
        "status": "ok" if len(ok_rows) == len(rows) else "partial_or_failed",
        "totals": totals,
        "physical_scope_note": (
            "Ideal volcano-surface angular in-scattering diagnostic. Counts are scaled by the sampled "
            "horizontal DEM target area and are not detector rates unless an additional detector area/geometry "
            "normalization is applied."
        ),
        "rows": rows,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "four_point_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else ["point", "status"]
    with (out_root / "four_point_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "README.txt").write_text(
        "\n".join(
            [
                "CABRIALES spatial in-scattering 90-day four-point campaign",
                "",
                "Each point directory contains chunk worker outputs and a reduced result.",
                "Main files:",
                "- four_point_summary.json",
                "- four_point_summary.csv",
                "- P*/reduced/spatial_in_scattering_summary.json",
                "- P*/reduced/spatial_final_accepted_map.png",
                "- P*/reduced/spatial_source_external_map.png",
                "- P*/reduced/spatial_first_rock_contact_xy.png",
                "",
                summary["physical_scope_note"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.status_interval_s < 0:
        raise ValueError("--status-interval-s must be non-negative")
    if args.force and args.continue_on_existing:
        raise ValueError("--force and --continue-on-existing cannot be used together")
    root = repo_root()
    check_inputs(root, args)
    resolve_from_root(root, args.out_root).mkdir(parents=True, exist_ok=True)
    cfg = TransportConfig(
        sample_probability=args.sample_probability,
        head=args.head,
        base_seed=args.seed,
        ray_step_m=args.ray_step_m,
        max_track_m=args.max_track_m,
        grid_step_m=args.volcano_surface_grid_step_m,
        edge_guard_m=args.volcano_surface_edge_guard_m,
        min_height_frac=args.volcano_surface_min_height_frac,
        start_offset_m=args.volcano_surface_start_offset_m,
        entry_check_m=args.volcano_surface_entry_check_m,
        theta_max_deg=args.theta_max_deg,
        max_angular_margin_deg=args.max_angular_margin_deg,
        min_survival_rock_m=args.min_survival_rock_m,
        kernel_scale=args.kernel_scale,
        disable_scattering=args.disable_scattering,
    )
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    for index, point in enumerate(args.points, start=1):
        print(f"[CAMPAIGN {index}/{len(args.points)}] point={point}", flush=True)
        row = run_point(root, args, cfg, point)
        rows.append(row)
        write_campaign_summary(args.out_root, rows, cfg, args, root)
        if row.get("status") != "ok":
            print(f"[STOP] {point} did not finish cleanly")
            return 2
    write_campaign_summary(args.out_root, rows, cfg, args, root)
    print(f"[DONE] four-point campaign elapsed={format_duration(time.monotonic() - started)}")
    print(f"summary_json={resolve_from_root(root, args.out_root) / 'four_point_summary.json'}")
    print(f"summary_csv={resolve_from_root(root, args.out_root) / 'four_point_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
