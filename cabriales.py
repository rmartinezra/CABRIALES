#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience entry point for common CABRIALES workflows.

This wrapper does not reimplement physics. It builds the validated commands for
orquestador_machin.py and the spatial in-scattering campaign runner, prints them,
and then executes them from the repository root.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POINTS = ("P1", "P2", "P4", "P5")
DEFAULT_90D_CACHE = Path("/home/rafael/proyectos/CNF/muon-cnf-toolkit/machin90dia_kinematic_cache")
DEFAULT_PIPELINE_90D_OUT = Path("run_machin90dia_allpoints_full")
DEFAULT_BACKGROUND_90D_OUT = Path(
    "run_machin90dia_p1_fastcache/10_in_scattering_background/"
    "machin90d_4points_volcano_surface_workers8"
)


def q(cmd: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run(cmd: Sequence[object], *, dry_run: bool = False) -> int:
    print("\n$ " + q(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run([str(part) for part in cmd], cwd=PROJECT_ROOT)
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
        "--smearing-source", "filtered",
        "--run-event-mc",
        "--event-mc-source", "filtered",
        "--event-mc-source-mode", "inside",
        "--empirical-interp-method", "linear",
        "--empirical-kernel-threshold", str(args.empirical_kernel_threshold),
        "--parallel-jobs", str(args.workers),
        "--inside-filtered-workers", str(args.workers),
        "--event-mc-workers", str(args.workers),
    ]
    if shw is not None:
        cmd.extend(["--shw", shw, "--shw-format", args.shw_format])
        if args.shw_member:
            cmd.extend(["--shw-member", args.shw_member])
    if args.discard_upgoing:
        cmd.append("--discard-upgoing")
    if args.skip_event_mc:
        cmd.append("--skip-event-mc")
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
        "--points", *point_args(args.points),
    ]
    if args.no_figures:
        cmd.append("--no-figures")
    if args.continue_on_existing:
        cmd.append("--continue-on-existing")
    append_force(cmd, args.force)
    cmd.extend(extra)
    return cmd


def cmd_smoke(args: argparse.Namespace, extra: list[str]) -> int:
    rc = run(build_smoke_cmd(args, extra), dry_run=args.dry_run)
    if rc != 0 or args.no_validate:
        return rc
    return run([sys.executable, "validar_corrida.py", args.outdir], dry_run=args.dry_run)


def cmd_validate(args: argparse.Namespace, extra: list[str]) -> int:
    cmd: list[object] = [sys.executable, "validar_corrida.py", args.outdir]
    cmd.extend(extra)
    return run(cmd, dry_run=args.dry_run)


def cmd_machin90d(args: argparse.Namespace, extra: list[str]) -> int:
    rc = run(build_machin90d_cmd(args, extra), dry_run=args.dry_run)
    if rc != 0 or args.no_validate:
        return rc
    return run([sys.executable, "validar_corrida.py", args.outdir], dry_run=args.dry_run)


def cmd_background90d(args: argparse.Namespace, extra: list[str]) -> int:
    return run(build_background90d_cmd(args, extra), dry_run=args.dry_run)


def cmd_all90d(args: argparse.Namespace, extra: list[str]) -> int:
    # Keep pass-through arguments on the pipeline step, where most framework flags live.
    pipeline_args = argparse.Namespace(**vars(args))
    pipeline_args.outdir = args.pipeline_outdir
    pipeline_args.no_validate = True
    rc = run(build_machin90d_cmd(pipeline_args, extra), dry_run=args.dry_run)
    if rc != 0:
        return rc

    background_args = argparse.Namespace(**vars(args))
    background_args.out_root = args.background_out_root
    background_args.ecrit_root = str(Path(args.pipeline_outdir) / "03_ecrit")
    rc = run(build_background90d_cmd(background_args, []), dry_run=args.dry_run)
    if rc != 0 or args.no_validate:
        return rc
    return run([sys.executable, "validar_corrida.py", args.pipeline_outdir], dry_run=args.dry_run)


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
    smoke.set_defaults(func=cmd_smoke)

    validate = sub.add_parser("validate", help="Valida una corrida existente con validar_corrida.py.")
    validate.add_argument("outdir")
    validate.add_argument("--dry-run", action="store_true")
    validate.set_defaults(func=cmd_validate)

    mach = sub.add_parser("machin90d", help="Corre el pipeline 90 dias Machin con fast-cache y event-MC.")
    mach.add_argument("--outdir", default=str(DEFAULT_PIPELINE_90D_OUT))
    mach.add_argument("--kinematic-cache", default=str(DEFAULT_90D_CACHE))
    mach.add_argument("--shw", default=None, help="Opcional: SHW/tar para construir el cache si no existe.")
    mach.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="cnf9")
    mach.add_argument("--shw-member", default=None)
    mach.add_argument("--points", nargs="+", choices=DEFAULT_POINTS, default=list(DEFAULT_POINTS))
    mach.add_argument("--workers", type=int, default=0, help="0 autodetecta en el orquestador.")
    mach.add_argument("--empirical-kernel-threshold", type=float, default=0.001)
    mach.add_argument("--discard-upgoing", action="store_true")
    mach.add_argument("--skip-event-mc", action="store_true")
    mach.add_argument("--force", action="store_true")
    mach.add_argument("--no-validate", action="store_true")
    mach.add_argument("--dry-run", action="store_true")
    mach.set_defaults(func=cmd_machin90d)

    bg = sub.add_parser("background90d", help="Corre in-scattering espacial 90 dias para P1/P2/P4/P5.")
    bg.add_argument("--out-root", default=str(DEFAULT_BACKGROUND_90D_OUT))
    bg.add_argument("--kinematic-cache", default=str(DEFAULT_90D_CACHE))
    bg.add_argument("--ecrit-root", default=str(DEFAULT_PIPELINE_90D_OUT / "03_ecrit"))
    bg.add_argument("--points", nargs="+", choices=DEFAULT_POINTS, default=list(DEFAULT_POINTS))
    bg.add_argument("--workers", type=int, default=8)
    bg.add_argument("--sample-probability", type=float, default=1.0)
    bg.add_argument("--seed", type=int, default=12345)
    bg.add_argument("--force", action="store_true")
    bg.add_argument("--continue-on-existing", action="store_true")
    bg.add_argument("--no-figures", action="store_true")
    bg.add_argument("--dry-run", action="store_true")
    bg.set_defaults(func=cmd_background90d)

    all90d = sub.add_parser("all90d", help="Corre pipeline 90 dias y luego background espacial.")
    all90d.add_argument("--pipeline-outdir", default=str(DEFAULT_PIPELINE_90D_OUT))
    all90d.add_argument("--background-out-root", default=str(DEFAULT_PIPELINE_90D_OUT / "10_in_scattering_background" / "machin90d_4points_volcano_surface_workers8"))
    all90d.add_argument("--kinematic-cache", default=str(DEFAULT_90D_CACHE))
    all90d.add_argument("--shw", default=None)
    all90d.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="cnf9")
    all90d.add_argument("--shw-member", default=None)
    all90d.add_argument("--points", nargs="+", choices=DEFAULT_POINTS, default=list(DEFAULT_POINTS))
    all90d.add_argument("--workers", type=int, default=8)
    all90d.add_argument("--sample-probability", type=float, default=1.0)
    all90d.add_argument("--seed", type=int, default=12345)
    all90d.add_argument("--empirical-kernel-threshold", type=float, default=0.001)
    all90d.add_argument("--discard-upgoing", action="store_true")
    all90d.add_argument("--skip-event-mc", action="store_true")
    all90d.add_argument("--force", action="store_true")
    all90d.add_argument("--continue-on-existing", action="store_true")
    all90d.add_argument("--no-figures", action="store_true")
    all90d.add_argument("--no-validate", action="store_true")
    all90d.add_argument("--dry-run", action="store_true")
    all90d.set_defaults(func=cmd_all90d)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return int(args.func(args, extra))


if __name__ == "__main__":
    raise SystemExit(main())
