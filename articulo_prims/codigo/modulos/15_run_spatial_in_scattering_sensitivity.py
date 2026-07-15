#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an 8-worker spatial in-scattering sensitivity campaign for CABRIALES."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CampaignConfig:
    name: str
    seed: int = 12345
    max_angular_margin_deg: float = 5.0
    edge_guard_m: float = 500.0
    grid_step_m: float = 50.0
    min_height_frac: float = 0.15
    ray_step_m: float = 100.0
    min_survival_rock_m: float = 100.0
    kernel_scale: float = 1.0
    disable_scattering: bool = False
    extra: dict[str, object] = field(default_factory=dict)


DEFAULT_CONFIGS = [
    CampaignConfig("baseline_seed12345", seed=12345),
    CampaignConfig("baseline_seed24680", seed=24680),
    CampaignConfig("baseline_seed97531", seed=97531),
    CampaignConfig("margin3_seed12345", seed=12345, max_angular_margin_deg=3.0),
    CampaignConfig("margin8_seed12345", seed=12345, max_angular_margin_deg=8.0),
    CampaignConfig("edge1000_seed12345", seed=12345, edge_guard_m=1000.0),
    CampaignConfig("ray50_seed12345", seed=12345, ray_step_m=50.0, min_survival_rock_m=100.0),
    CampaignConfig("noscatter_seed12345", seed=12345, disable_scattering=True),
]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run spatial in-scattering sensitivity campaign with chunk workers.")
    ap.add_argument("--out-root", type=Path, default=Path("run_machin90dia_allpoints_full/10_in_scattering_background/sensitivity_p50_workers8"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-probability", type=float, default=0.5)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="Optional config names to run.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
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
    ap.add_argument("--kernel-npz", type=Path, default=Path("modulos/empirical_kernel_library.npz"))
    ap.add_argument("--acceptance-map", type=Path, default=Path("run_machin90dia_p1_fastcache/03_ecrit/ecrit_table_P1.csv"))
    ap.add_argument("--length-map", type=Path, default=Path("run_machin90dia_p1_fastcache/03_ecrit/ecrit_table_P1.csv"))
    ap.add_argument("--hgt-dir", default="data")
    ap.add_argument("--range-file", type=Path, default=Path("data/data_rock.dat"))
    ap.add_argument("--point", default="P1")
    return ap


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def build_worker_cmd(args, cfg: CampaignConfig, worker_index: int, chunk_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "modulos/13_apply_spatial_in_scattering_dem.py",
        "--kinematic-cache", str(args.kinematic_cache),
        "--kernel-npz", str(args.kernel_npz),
        "--acceptance-map", str(args.acceptance_map),
        "--length-map", str(args.length_map),
        "--output-dir", str(chunk_dir),
        "--point", str(args.point),
        "--hgt-dir", str(args.hgt_dir),
        "--range-file", str(args.range_file),
        "--head", str(args.head),
        "--chunk-count", str(args.workers),
        "--chunk-index", str(worker_index),
        "--seed", str(cfg.seed),
        "--ray-step-m", str(cfg.ray_step_m),
        "--max-track-m", "9000",
        "--source-surface", "volcano-surface",
        "--volcano-surface-grid-step-m", str(cfg.grid_step_m),
        "--volcano-surface-edge-guard-m", str(cfg.edge_guard_m),
        "--volcano-surface-min-height-frac", str(cfg.min_height_frac),
        "--volcano-surface-entry-check-m", "10",
        "--theta-max-deg", "90",
        "--max-angular-margin-deg", str(cfg.max_angular_margin_deg),
        "--min-survival-rock-m", str(cfg.min_survival_rock_m),
        "--sample-probability", str(args.sample_probability),
        "--kernel-scale", str(cfg.kernel_scale),
        "--no-progress",
        "--no-figures",
    ]
    if cfg.disable_scattering:
        cmd.append("--disable-scattering")
    return cmd


def summarize_config(config_dir: Path, cfg: CampaignConfig) -> dict[str, object]:
    summary_path = config_dir / "reduced" / "spatial_in_scattering_summary.json"
    if not summary_path.exists():
        return {"config": cfg.name, "status": "missing_summary"}
    s = json.load(open(summary_path, encoding="utf-8"))
    area = s.get("area_effective_estimate") or {}
    stats = s.get("stats", {})
    return {
        "config": cfg.name,
        "status": "ok",
        "seed": cfg.seed,
        "max_angular_margin_deg": cfg.max_angular_margin_deg,
        "edge_guard_m": cfg.edge_guard_m,
        "grid_step_m": cfg.grid_step_m,
        "ray_step_m": cfg.ray_step_m,
        "disable_scattering": cfg.disable_scattering,
        "weighted_accepted_count": s.get("weighted_accepted_count"),
        "accepted_effective_sample_size": s.get("accepted_effective_sample_size"),
        "accepted_relative_mc_se": s.get("accepted_relative_mc_se"),
        "n_flux_events_read": stats.get("n_flux_events_read"),
        "n_tracks_touched_rock": stats.get("n_tracks_touched_rock"),
        "n_tracks_survived": stats.get("n_tracks_survived"),
        "area_m2": area.get("effective_area_m2"),
        "area_scaled_count_90d": area.get("area_scaled_count_90d"),
        "area_scaled_count_per_day": area.get("area_scaled_count_per_day"),
        "summary_json": str(summary_path),
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    args.out_root.mkdir(parents=True, exist_ok=True)
    configs = DEFAULT_CONFIGS
    if args.only:
        wanted = set(args.only)
        configs = [cfg for cfg in configs if cfg.name in wanted]
        missing = sorted(wanted - {cfg.name for cfg in configs})
        if missing:
            raise ValueError(f"Unknown configs: {missing}")

    campaign_rows: list[dict[str, object]] = []
    t_campaign = time.time()
    for cfg in configs:
        config_dir = args.out_root / cfg.name
        reduced_summary = config_dir / "reduced" / "spatial_in_scattering_summary.json"
        if reduced_summary.exists() and not args.force:
            print(f"[SKIP] {cfg.name} already reduced")
            campaign_rows.append(summarize_config(config_dir, cfg))
            continue
        config_dir.mkdir(parents=True, exist_ok=True)
        chunks_root = config_dir / "chunks"
        chunks_root.mkdir(parents=True, exist_ok=True)
        meta = {
            "config": cfg.__dict__,
            "workers": args.workers,
            "sample_probability": args.sample_probability,
            "head": args.head,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (config_dir / "campaign_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[RUN] {cfg.name} workers={args.workers} p={args.sample_probability}")
        t0 = time.time()
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for worker in range(args.workers):
                chunk_dir = chunks_root / f"w{worker:02d}"
                cmd = build_worker_cmd(args, cfg, worker, chunk_dir)
                (chunk_dir / "command.txt").parent.mkdir(parents=True, exist_ok=True)
                (chunk_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
                futures.append((worker, ex.submit(run_command, cmd, chunk_dir / "run.log")))
            failed = []
            for worker, fut in futures:
                rc = fut.result()
                if rc != 0:
                    failed.append((worker, rc))
        if failed:
            print(f"[FAIL] {cfg.name}: {failed}")
            return 2
        input_dirs = [chunks_root / f"w{i:02d}" for i in range(args.workers)]
        reduce_cmd = [
            sys.executable,
            "modulos/14_reduce_spatial_in_scattering_chunks.py",
            "--output-dir", str(config_dir / "reduced"),
            "--label", cfg.name,
            "--input-dirs", *map(str, input_dirs),
        ]
        if args.no_figures:
            reduce_cmd.append("--no-figures")
        rc = run_command(reduce_cmd, config_dir / "reduce.log")
        if rc != 0:
            print(f"[FAIL] reducer {cfg.name}: rc={rc}")
            return 3
        elapsed = time.time() - t0
        row = summarize_config(config_dir, cfg)
        row["elapsed_s"] = elapsed
        campaign_rows.append(row)
        print(f"[OK] {cfg.name}: accepted={row.get('weighted_accepted_count')} elapsed={elapsed/60:.1f} min")

    summary_csv = args.out_root / "sensitivity_summary.csv"
    keys = sorted(set().union(*(row.keys() for row in campaign_rows)))
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(campaign_rows)
    (args.out_root / "sensitivity_summary.json").write_text(json.dumps(campaign_rows, indent=2), encoding="utf-8")
    print(f"[DONE] campaign elapsed={(time.time()-t_campaign)/60:.1f} min")
    print(f"summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
