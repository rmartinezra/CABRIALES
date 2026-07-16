#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience entry point for common CABRIALES workflows.

This wrapper does not reimplement physics. It builds the validated commands for
orquestador_machin.py and the spatial in-scattering campaign runner, prints them,
and then executes them from the repository root.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from modulos.progress import format_duration


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POINTS = ("P1", "P2", "P4", "P5")
DEFAULT_PIPELINE_90D_OUT = Path("run_machin90dia_allpoints_full")
DEFAULT_BACKGROUND_RUN_STEM = "machin90d_4points_volcano_surface"
DEFAULT_KERNEL = Path("modulos/hybrid_empirical_kernel_library.npz")
DEFAULT_ROCK_STEP_M = 10.0
DEFAULT_MIN_SURVIVAL_ROCK_M = DEFAULT_ROCK_STEP_M


def default_background_90d_out(
    pipeline_outdir: Path | str,
    workers: int,
    ray_step_m: float = DEFAULT_ROCK_STEP_M,
) -> Path:
    """Name the background campaign after its transport configuration."""
    if workers < 1:
        raise ValueError("El background espacial requiere al menos un worker.")
    if ray_step_m < 1.0:
        raise ValueError("El paso de transporte debe ser >= 1 m.")
    step_label = f"{float(ray_step_m):g}".replace(".", "p")
    return (
        Path(pipeline_outdir)
        / "10_in_scattering_background"
        / f"{DEFAULT_BACKGROUND_RUN_STEM}_step{step_label}m_workers{workers}"
    )


def detect_default_90d_cache() -> Path:
    """Prefer an explicit env var, then local and legacy cache locations."""
    configured = os.environ.get("CABRIALES_90D_CACHE")
    if configured:
        return Path(configured).expanduser()
    candidates = (
        PROJECT_ROOT / "data" / "cache" / "machin90dia_kinematic_cache",
        PROJECT_ROOT.parent / "CNF" / "muon-cnf-toolkit" / "machin90dia_kinematic_cache",
    )
    return next((path for path in candidates if (path / "manifest.json").is_file()), candidates[0])


DEFAULT_90D_CACHE = detect_default_90d_cache()


def q(cmd: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run(cmd: Sequence[object], *, label: str, dry_run: bool = False) -> int:
    """Run one workflow step and keep terminal state concise and explicit."""
    print(f"\n[START] {label}", flush=True)
    print("$ " + q(cmd), flush=True)
    if dry_run:
        print(f"[DRY-RUN] {label}", flush=True)
        return 0
    started = time.monotonic()
    proc = subprocess.run([str(part) for part in cmd], cwd=PROJECT_ROOT)
    state = "OK" if proc.returncode == 0 else f"ERROR rc={proc.returncode}"
    print(f"[{state}] {label} | elapsed={format_duration(time.monotonic() - started)}", flush=True)
    return int(proc.returncode)


def point_args(points: Iterable[str]) -> list[str]:
    pts = list(points)
    if not pts:
        raise ValueError("Debes indicar al menos un punto.")
    return pts


def append_force(cmd: list[object], enabled: bool) -> None:
    if enabled:
        cmd.append("--force")


def build_smoke_cmd(args: argparse.Namespace, extra: list[str]) -> list[object]:
    cmd: list[object] = [
        sys.executable,
        "orquestador_machin.py",
        "--profile", "bariloche-smoke",
        "--outdir", args.outdir,
        "--status-interval-s", str(args.status_interval_s),
    ]
    append_force(cmd, args.force)
    cmd.extend(extra)
    return cmd


def build_machin90d_cmd(args: argparse.Namespace, extra: list[str]) -> list[object]:
    cache = Path(args.kinematic_cache).expanduser()
    shw = Path(args.shw).expanduser() if args.shw else None
    if not args.dry_run and not (cache / "manifest.json").exists() and shw is None:
        raise FileNotFoundError(
            f"No existe el cache cinematico {cache}/manifest.json. "
            "Usa --shw para construirlo o pasa --kinematic-cache a un cache existente."
        )

    cmd: list[object] = [
        sys.executable,
        "orquestador_machin.py",
        "--scripts-dir", "modulos",
        "--hgt-dir", "data",
        "--range-file", "data/data_rock.dat",
        "--kinematic-cache", cache,
        "--outdir", args.outdir,
        "--points", *point_args(args.points),
        "--storage-profile", "compact",
        "--fast-cache",
        "--plot-source", "filtered",
        "--inside-volcano-source", "filtered",
        "--scattering-model", "empirical",
        "--empirical-kernel-library", args.kernel_npz,
        "--smearing-source", "filtered",
        "--run-event-mc",
        "--event-mc-source", "filtered",
        "--event-mc-source-mode", "inside",
        "--empirical-interp-method", "tail-aware",
        "--empirical-kernel-threshold", str(args.empirical_kernel_threshold),
        "--parallel-jobs", str(args.workers),
        "--inside-filtered-workers", str(args.workers),
        "--event-mc-workers", str(args.workers),
        "--status-interval-s", str(args.status_interval_s),
    ]
    if shw is not None:
        cmd.extend(["--shw", shw, "--shw-format", args.shw_format])
        if args.shw_member:
            cmd.extend(["--shw-member", args.shw_member])
    if args.discard_upgoing:
        cmd.append("--discard-upgoing")
    if args.skip_event_mc:
        cmd.append("--skip-event-mc")
    if args.head > 0:
        cmd.extend(["--event-mc-head", str(args.head)])
    append_force(cmd, args.force)
    cmd.extend(extra)
    return cmd


def build_background90d_cmd(
    args: argparse.Namespace,
    extra: list[str],
    *,
    out_root: Path | str | None = None,
    ecrit_root: Path | str | None = None,
) -> list[object]:
    cmd: list[object] = [
        sys.executable,
        "modulos/16_run_spatial_in_scattering_4points.py",
        "--out-root", out_root if out_root is not None else args.out_root,
        "--workers", str(args.workers),
        "--sample-probability", str(args.sample_probability),
        "--seed", str(args.seed),
        "--kinematic-cache", args.kinematic_cache,
        "--ecrit-root", ecrit_root if ecrit_root is not None else args.ecrit_root,
        "--kernel-npz", args.kernel_npz,
        "--interp-method", "tail-aware",
        "--kernel-threshold", str(args.empirical_kernel_threshold),
        "--ray-step-m", str(args.ray_step_m),
        "--min-survival-rock-m", str(args.min_survival_rock_m),
        "--kernel-energy-extrapolation", str(args.kernel_energy_extrapolation),
        "--points", *point_args(args.points),
        "--status-interval-s", str(args.status_interval_s),
    ]
    if args.head > 0:
        cmd.extend(["--head", str(args.head)])
    if args.no_figures:
        cmd.append("--no-figures")
    if args.continue_on_existing:
        cmd.append("--continue-on-existing")
    append_force(cmd, args.force)
    cmd.extend(extra)
    return cmd


def cmd_smoke(args: argparse.Namespace, extra: list[str]) -> int:
    rc = run(build_smoke_cmd(args, extra), label="Smoke test", dry_run=args.dry_run)
    if rc != 0 or args.no_validate:
        return rc
    return run(
        [sys.executable, "validar_corrida.py", args.outdir],
        label="Validacion del smoke test",
        dry_run=args.dry_run,
    )


def cmd_validate(args: argparse.Namespace, extra: list[str]) -> int:
    cmd: list[object] = [sys.executable, "validar_corrida.py", args.outdir]
    cmd.extend(extra)
    return run(cmd, label="Validacion de corrida", dry_run=args.dry_run)


def cmd_kernel_smoke(args: argparse.Namespace, extra: list[str]) -> int:
    """Exercise the bundled full-tail predictor at one physical point."""
    cmd: list[object] = [
        sys.executable,
        "modulos/tail_aware_transport.py",
        "--model", args.kernel_npz,
        "--L", str(args.length_m),
        "--E", str(args.energy_gev),
        "--method", "tail-aware",
        "--out", args.out,
    ]
    cmd.extend(extra)
    return run(cmd, label="Kernel full-tail tail-aware", dry_run=args.dry_run)


def cmd_machin90d(args: argparse.Namespace, extra: list[str]) -> int:
    rc = run(build_machin90d_cmd(args, extra), label="Pipeline Machin 90 dias", dry_run=args.dry_run)
    if rc != 0 or args.no_validate:
        return rc
    return run(
        [sys.executable, "validar_corrida.py", args.outdir],
        label="Validacion del pipeline",
        dry_run=args.dry_run,
    )


def validate_background(out_root: Path | str, points: Iterable[str], *, dry_run: bool) -> int:
    """Check that the reduced four-point campaign finished and is readable."""
    summary_path = PROJECT_ROOT / Path(out_root) / "four_point_summary.json"
    print(f"\n[START] Validacion del background | summary={summary_path}", flush=True)
    if dry_run:
        print("[DRY-RUN] Validacion del background", flush=True)
        return 0
    if not summary_path.is_file():
        print(f"[ERROR] Falta el resumen del background: {summary_path}", flush=True)
        return 2
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    expected = set(points)
    completed = {str(row.get("point")) for row in summary.get("rows", []) if row.get("status") == "ok"}
    missing = sorted(expected - completed)
    if summary.get("status") != "ok" or missing:
        print(
            f"[ERROR] Background incompleto | status={summary.get('status')} "
            f"| puntos_faltantes={missing}",
            flush=True,
        )
        return 2
    accepted = (summary.get("totals") or {}).get("weighted_accepted_count")
    print(f"[OK] Background completo | puntos={len(completed)} | accepted={accepted}", flush=True)
    return 0


def cmd_background90d(args: argparse.Namespace, extra: list[str]) -> int:
    out_root = args.out_root or default_background_90d_out(
        DEFAULT_PIPELINE_90D_OUT,
        args.workers,
        args.ray_step_m,
    )
    rc = run(
        build_background90d_cmd(args, extra, out_root=out_root),
        label="Background espacial 90 dias",
        dry_run=args.dry_run,
    )
    if rc != 0 or args.no_validate:
        return rc
    return validate_background(out_root, args.points, dry_run=args.dry_run)


def cmd_full(args: argparse.Namespace, extra: list[str]) -> int:
    print("CABRIALES FULL: pipeline -> background espacial -> validacion", flush=True)
    background_out_root = args.background_out_root or default_background_90d_out(
        args.pipeline_outdir,
        args.workers,
        args.ray_step_m,
    )
    # Keep pass-through arguments on the pipeline step, where most framework flags live.
    pipeline_args = argparse.Namespace(**vars(args))
    pipeline_args.outdir = args.pipeline_outdir
    pipeline_args.no_validate = True
    rc = run(
        build_machin90d_cmd(pipeline_args, extra),
        label="FULL 1/3 - Pipeline Machin 90 dias",
        dry_run=args.dry_run,
    )
    if rc != 0:
        return rc

    background_args = argparse.Namespace(**vars(args))
    background_args.out_root = background_out_root
    background_args.ecrit_root = str(Path(args.pipeline_outdir) / "03_ecrit")
    rc = run(
        build_background90d_cmd(background_args, []),
        label="FULL 2/3 - Background espacial",
        dry_run=args.dry_run,
    )
    if rc != 0 or args.no_validate:
        return rc
    rc = run(
        [sys.executable, "validar_corrida.py", args.pipeline_outdir],
        label="FULL 3/3 - Validacion del pipeline",
        dry_run=args.dry_run,
    )
    if rc != 0:
        return rc
    return validate_background(background_out_root, args.points, dry_run=args.dry_run)


def add_progress_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--status-interval-s",
        type=float,
        default=30.0,
        help="Segundos entre actualizaciones de progreso durante etapas largas; 0 las desactiva.",
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Entrada simple para CABRIALES: smoke, pipeline Machin 90d y background espacial.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Verifica el pipeline con data/bariloche_5min.shw.")
    smoke.add_argument("--outdir", default="run_bariloche_smoke")
    smoke.add_argument("--force", action="store_true")
    smoke.add_argument("--no-validate", action="store_true")
    smoke.add_argument("--dry-run", action="store_true")
    add_progress_argument(smoke)
    smoke.set_defaults(func=cmd_smoke)

    validate = sub.add_parser("validate", help="Valida una corrida existente con validar_corrida.py.")
    validate.add_argument("outdir")
    validate.add_argument("--dry-run", action="store_true")
    validate.set_defaults(func=cmd_validate)

    kernel_smoke = sub.add_parser(
        "kernel-smoke",
        help="Verifica normalizacion, interpolacion y colas del kernel hibrido.",
    )
    kernel_smoke.add_argument("--kernel-npz", default=str(DEFAULT_KERNEL))
    kernel_smoke.add_argument("--length-m", type=float, default=80.0)
    kernel_smoke.add_argument("--energy-gev", type=float, default=39.67)
    kernel_smoke.add_argument("--out", default="outputs/kernel_smoke/kernel_L80_E39p67.csv")
    kernel_smoke.add_argument("--dry-run", action="store_true")
    kernel_smoke.set_defaults(func=cmd_kernel_smoke)

    mach = sub.add_parser("machin90d", help="Corre el pipeline 90 dias Machin con fast-cache y event-MC.")
    mach.add_argument("--outdir", default=str(DEFAULT_PIPELINE_90D_OUT))
    mach.add_argument("--kinematic-cache", default=str(DEFAULT_90D_CACHE))
    mach.add_argument("--shw", default=None, help="Opcional: SHW/tar para construir el cache si no existe.")
    mach.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="cnf9")
    mach.add_argument("--shw-member", default=None)
    mach.add_argument("--points", nargs="+", choices=DEFAULT_POINTS, default=list(DEFAULT_POINTS))
    mach.add_argument("--workers", type=int, default=0, help="0 autodetecta en el orquestador.")
    mach.add_argument("--kernel-npz", default=str(DEFAULT_KERNEL))
    mach.add_argument("--empirical-kernel-threshold", type=float, default=0.0)
    mach.add_argument("--head", type=int, default=0, help="Limita el event-MC por punto; 0 usa todos los eventos.")
    mach.add_argument("--discard-upgoing", action="store_true")
    mach.add_argument("--skip-event-mc", action="store_true")
    mach.add_argument("--force", action="store_true")
    mach.add_argument("--no-validate", action="store_true")
    mach.add_argument("--dry-run", action="store_true")
    add_progress_argument(mach)
    mach.set_defaults(func=cmd_machin90d)

    bg = sub.add_parser("background90d", help="Corre in-scattering espacial 90 dias para P1/P2/P4/P5.")
    bg.add_argument(
        "--out-root",
        default=None,
        help="Salida opcional; por defecto termina en volcano_surface_workersN.",
    )
    bg.add_argument("--kinematic-cache", default=str(DEFAULT_90D_CACHE))
    bg.add_argument("--ecrit-root", default=str(DEFAULT_PIPELINE_90D_OUT / "03_ecrit"))
    bg.add_argument("--points", nargs="+", choices=DEFAULT_POINTS, default=list(DEFAULT_POINTS))
    bg.add_argument("--workers", type=int, default=10)
    bg.add_argument("--sample-probability", type=float, default=1.0)
    bg.add_argument("--seed", type=int, default=12345)
    bg.add_argument("--kernel-npz", default=str(DEFAULT_KERNEL))
    bg.add_argument("--empirical-kernel-threshold", type=float, default=0.0)
    bg.add_argument("--ray-step-m", type=float, default=DEFAULT_ROCK_STEP_M)
    bg.add_argument("--min-survival-rock-m", type=float, default=DEFAULT_MIN_SURVIVAL_ROCK_M)
    bg.add_argument("--kernel-energy-extrapolation", choices=["momentum-scale", "nearest"], default="momentum-scale")
    bg.add_argument("--head", type=int, default=0, help="Limita los eventos de flujo por punto; 0 usa todo el cache.")
    bg.add_argument("--force", action="store_true")
    bg.add_argument("--continue-on-existing", action="store_true")
    bg.add_argument("--no-figures", action="store_true")
    bg.add_argument("--no-validate", action="store_true")
    bg.add_argument("--dry-run", action="store_true")
    add_progress_argument(bg)
    bg.set_defaults(func=cmd_background90d)

    all90d = sub.add_parser(
        "full",
        aliases=["all90d"],
        help="Corre y valida todo: pipeline 90 dias y background espacial.",
    )
    all90d.add_argument("--pipeline-outdir", default=str(DEFAULT_PIPELINE_90D_OUT))
    all90d.add_argument(
        "--background-out-root",
        default=None,
        help="Salida opcional; por defecto usa pipeline-outdir y termina en workersN.",
    )
    all90d.add_argument("--kinematic-cache", default=str(DEFAULT_90D_CACHE))
    all90d.add_argument("--shw", default=None)
    all90d.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="cnf9")
    all90d.add_argument("--shw-member", default=None)
    all90d.add_argument("--points", nargs="+", choices=DEFAULT_POINTS, default=list(DEFAULT_POINTS))
    all90d.add_argument("--workers", type=int, default=10)
    all90d.add_argument("--sample-probability", type=float, default=1.0)
    all90d.add_argument("--seed", type=int, default=12345)
    all90d.add_argument("--kernel-npz", default=str(DEFAULT_KERNEL))
    all90d.add_argument("--empirical-kernel-threshold", type=float, default=0.0)
    all90d.add_argument("--ray-step-m", type=float, default=DEFAULT_ROCK_STEP_M)
    all90d.add_argument("--min-survival-rock-m", type=float, default=DEFAULT_MIN_SURVIVAL_ROCK_M)
    all90d.add_argument("--kernel-energy-extrapolation", choices=["momentum-scale", "nearest"], default="momentum-scale")
    all90d.add_argument("--head", type=int, default=0, help="Limita event-MC y background por punto; 0 usa todo el cache.")
    all90d.add_argument("--discard-upgoing", action="store_true")
    all90d.add_argument("--skip-event-mc", action="store_true")
    all90d.add_argument("--force", action="store_true")
    all90d.add_argument("--continue-on-existing", action="store_true")
    all90d.add_argument("--no-figures", action="store_true")
    all90d.add_argument("--no-validate", action="store_true")
    all90d.add_argument("--dry-run", action="store_true")
    add_progress_argument(all90d)
    all90d.set_defaults(func=cmd_full)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if getattr(args, "head", 0) < 0:
        parser.error("--head must be non-negative")
    if extra and extra[0] == "--":
        extra = extra[1:]
    return int(args.func(args, extra))


if __name__ == "__main__":
    raise SystemExit(main())
