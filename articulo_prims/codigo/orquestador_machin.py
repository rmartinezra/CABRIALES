#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orquestador_machin.py
---------------------
Ejecuta la cadena completa para Machín usando los scripts existentes:

  01_puntos.py                 -> FOV, DEM + abanicos, blocked_angles_P*.csv
  02_longitud.py               -> longitudes dentro de roca, rock_length_P*.csv
  03_ecrit_heatmaps.py         -> tablas/mapas de energía crítica, ecrit_table_P*.csv
  06_filter_muons_by_ecrit.py  -> filtrado de muones ARTI por punto
  05_plot_theta_phi.py         -> mapas θ–φ de conteo de muones
  07_inside_volcano_maps_merged.py -> cuentas dentro del volcán + figuras individuales + 2x2
  08_scattering_highland_v2.py -> diagnóstico Highland de dispersión angular
  09_apply_angular_smearing_pretty_MC.py -> smearing angular sobre mapas θ–φ

Diseño:
- No modifica los scripts originales.
- Aísla el problema de 01_puntos.py, que escribe siempre en ./outputs.
- Ordena salidas por etapa.
- Guarda logs por etapa.
- Verifica entradas y salidas críticas.

Ejemplo mínimo:

  python orquestador_machin.py \
    --scripts-dir . \
    --hgt-dir ./data \
    --range-file ./data/data_rock.dat \
    --shw ./arti/input.shw \
    --outdir ./run_machin \
    --points P1 P2 P4 P5 \
    --rho 2.65 \
    --discard-upgoing

Si no tienes .shw todavía, puedes correr sólo geometría + longitudes + Ecrit:

  python orquestador_machin.py \
    --scripts-dir . \
    --hgt-dir ./data \
    --range-file ./data/data_rock.dat \
    --outdir ./run_machin
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from modulos.progress import format_duration
from modulos.shw_io import output_template_for_compression, shw_stem as detect_shw_stem


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_HGT = ("N04W076.hgt", "N04W075.hgt")
DEFAULT_POINTS = ("P1", "P2", "P4", "P5")
DEFAULT_STATUS_INTERVAL_S = 30.0
STATUS_INTERVAL_S = DEFAULT_STATUS_INTERVAL_S

SCRIPT_01 = "01_puntos.py"
SCRIPT_02 = "02_longitud.py"
SCRIPT_03 = "03_ecrit_heatmaps.py"
SCRIPT_04_KINEMATIC_CACHE = "04_build_kinematic_cache.py"
SCRIPT_04_EVENT_CACHE = "04_build_event_cache.py"
SCRIPT_05 = "05_plot_theta_phi.py"
SCRIPT_06 = "06_filter_muons_by_ecrit.py"  # versión rápida multi-punto en esta rama
SCRIPT_07_MERGED = "07_inside_volcano_maps_merged.py"
SCRIPT_07_FAST = "07_inside_volcano_allpoints_fast.py"
SCRIPT_07_LEGACY = "07_plot_counts_inside_volcano_geometry.py"
SCRIPT_08 = "08_scattering_highland_v2.py"
SCRIPT_09 = "09_apply_angular_smearing_pretty_MC.py"
SCRIPT_08_EMPIRICAL = "08_scattering_empirical_kernel.py"
SCRIPT_09_EMPIRICAL = "09_apply_angular_smearing_empirical_kernel.py"
SCRIPT_10_EVENT_MC = "10_apply_event_by_event_empirical_mc_v2.py"
SCRIPT_12_IN_SCATTERING = "12_apply_in_scattering_background.py"
SCRIPT_13_SPATIAL_IN_SCATTERING = "13_apply_spatial_in_scattering_dem.py"
SCRIPT_4PANEL = "plot_4panel_muon_maps.py"


@dataclass
class StageResult:
    name: str
    command: list[str]
    cwd: str
    log: str
    elapsed_s: float
    returncode: int
    status: str


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    if str(path).strip().lower() == "auto":
        return None
    return Path(path).expanduser().resolve()


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path.resolve()
    return None


def resolve_scripts_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        Path.cwd() / "modulos",
        PROJECT_ROOT / "modulos",
        Path.cwd(),
        PROJECT_ROOT,
    ]
    found = first_existing(p for p in candidates if (p / SCRIPT_01).exists())
    if found is None:
        raise FileNotFoundError(
            "No pude autodetectar la carpeta de scripts. Usa --scripts-dir modulos."
        )
    return found


def resolve_hgt_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        Path.cwd() / "data",
        PROJECT_ROOT / "data",
        Path.cwd(),
        PROJECT_ROOT,
    ]
    found = first_existing(
        p for p in candidates
        if all((p / fname).exists() for fname in REQUIRED_HGT)
    )
    if found is None:
        raise FileNotFoundError(
            "No pude autodetectar N04W076.hgt y N04W075.hgt. Usa --hgt-dir data."
        )
    return found


def find_script(
    filename: str,
    scripts_dir: Path,
    label: str,
    alternate_names: Sequence[str] = (),
    required: bool = True,
) -> Path | None:
    dirs = [
        scripts_dir,
        PROJECT_ROOT,
        PROJECT_ROOT / "modulos",
        Path.cwd(),
        Path.cwd() / "modulos",
    ]
    candidates = [directory / filename for directory in dirs]
    for directory in dirs:
        for name in alternate_names:
            candidates.append(directory / name)
    found = first_existing(candidates)
    if found is None and required:
        tried = "\n  - ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"No encontré script {label}. Busqué:\n  - {tried}")
    return found


def find_empirical_kernel(explicit: Path | None, scripts_dir: Path, hgt_dir: Path) -> Path | None:
    if explicit is not None:
        require_file(explicit, "empirical_kernel_library.npz")
        return explicit
    env_kernel = os.environ.get("CABRIALES_EMPIRICAL_KERNEL")
    return first_existing([
        Path(env_kernel).expanduser() if env_kernel else None,
        scripts_dir / "hybrid_empirical_kernel_library.npz",
        PROJECT_ROOT / "modulos" / "hybrid_empirical_kernel_library.npz",
        scripts_dir / "empirical_kernel_library.npz",
        PROJECT_ROOT / "modulos" / "empirical_kernel_library.npz",
        hgt_dir / "empirical_kernel_library.npz",
        PROJECT_ROOT / "data" / "empirical_kernel_library.npz",
        Path.cwd() / "empirical_kernel_library.npz",
    ])


def normalize_workers(value: int | None, max_reasonable: int) -> int:
    if value is not None and value > 0:
        return int(value)
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, max_reasonable))


def detect_compute_backend(requested: str) -> dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi") or (
        "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None
    )
    gpu_visible = bool(nvidia_smi)
    cupy_available = importlib.util.find_spec("cupy") is not None
    gpu_ready = gpu_visible and cupy_available
    active = "gpu" if requested in ("auto", "gpu") and gpu_ready else "cpu"
    reason = None
    if requested == "gpu" and not gpu_ready:
        reason = "GPU visible, pero falta CuPy; continúo en CPU." if gpu_visible else "No hay GPU visible; continúo en CPU."
    elif requested == "auto" and gpu_visible and not cupy_available:
        reason = "GPU visible, pero no hay backend CuPy instalado; uso CPU."
    elif requested == "cpu":
        reason = "CPU forzado por --compute-device cpu."
    return {
        "requested": requested,
        "active": active,
        "gpu_visible": gpu_visible,
        "cupy_available": cupy_available,
        "nvidia_smi": nvidia_smi,
        "note": reason,
    }


def apply_profile(args: argparse.Namespace) -> None:
    if args.profile == "standard":
        return
    if args.profile == "bariloche-smoke":
        if args.shw is None:
            args.shw = "data/bariloche_5min.shw"
        if args.outdir == "run_machin":
            args.outdir = "run_bariloche_smoke"
        args.plot_source = "both"
        args.inside_volcano_source = "both"
        args.scattering_model = "both"
        args.smearing_source = "both"
        args.run_event_mc = True
        args.event_mc_source = "both"
        args.event_mc_source_mode = "all"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_and_create(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, overwrite: bool = False) -> None:
    """Crea symlink si puede; si no, copia. Útil para HGT grandes."""
    src = src.resolve()
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return
    ensure_dir(dst.parent)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree_contents(src_dir: Path, dst_dir: Path, overwrite: bool = True) -> None:
    ensure_dir(dst_dir)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            if overwrite and target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            if overwrite and (target.exists() or target.is_symlink()):
                target.unlink()
            shutil.copy2(item, target)


def require_file(path: Path, label: str = "archivo") -> None:
    if not path.exists():
        raise FileNotFoundError(f"No encontré {label}: {path}")


def require_files(paths: Iterable[Path], stage: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        msg = "\n  - ".join(missing)
        raise FileNotFoundError(f"Faltan salidas requeridas después de {stage}:\n  - {msg}")


def filtered_shw_path(filtered_dir: Path, input_stem: str, point: str, compression: str) -> Path:
    template = str(filtered_dir / f"{input_stem}_filtered_{point}.shw")
    return Path(output_template_for_compression(template, compression))


def check_scripts(scripts_dir: Path) -> dict[str, Path]:
    scripts = {
        "geometry": scripts_dir / SCRIPT_01,
        "lengths": scripts_dir / SCRIPT_02,
        "ecrit": scripts_dir / SCRIPT_03,
        "plot": scripts_dir / SCRIPT_05,
        "filter": scripts_dir / SCRIPT_06,
        "scattering": scripts_dir / SCRIPT_08,
        "smearing": scripts_dir / SCRIPT_09,
    }
    for name, path in scripts.items():
        require_file(path, f"script {name}")
    return scripts


def find_range_file(explicit: Path | None, search_dirs: Sequence[Path]) -> Path | None:
    if explicit is not None:
        require_file(explicit, "tabla CSDA / data_rock")
        return explicit
    candidates = ("muon_range_table.csv", "data_rock.dat")
    for directory in search_dirs:
        if directory is None:
            continue
        for fname in candidates:
            p = directory / fname
            if p.exists():
                return p.resolve()
    return None


def run_command(
    name: str,
    cmd: Sequence[str],
    cwd: Path,
    log_dir: Path,
    dry_run: bool = False,
) -> StageResult:
    """Run one stage, stream output to its log, and report live status."""
    ensure_dir(log_dir)
    log_path = log_dir / f"{name}.log"
    start = time.monotonic()
    printable = " ".join(str(c) for c in cmd)
    print(f"[START] {name} | log={log_path}", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}\n")
        log.write(f"# time: {now_stamp()}\n")
        log.write(f"# cwd: {cwd}\n")
        log.write(f"# cmd: {printable}\n\n")

        if dry_run:
            elapsed = time.monotonic() - start
            print(f"[DRY-RUN] {name} | cmd={printable}", flush=True)
            return StageResult(
                name,
                list(map(str, cmd)),
                str(cwd),
                str(log_path),
                elapsed,
                0,
                "DRY-RUN",
            )

        proc = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while True:
                if STATUS_INTERVAL_S <= 0:
                    returncode = proc.wait()
                    break
                try:
                    returncode = proc.wait(timeout=STATUS_INTERVAL_S)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - start
                    print(
                        f"[RUNNING] {name} | elapsed={format_duration(elapsed)} "
                        f"| log={log_path}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise

    elapsed = time.monotonic() - start
    status = "OK" if returncode == 0 else "ERROR"
    result = StageResult(name, list(map(str, cmd)), str(cwd), str(log_path), elapsed, returncode, status)
    if returncode != 0:
        print(f"[ERROR] {name} | rc={returncode} | log={log_path}", flush=True)
        raise RuntimeError(
            f"Falló la etapa {name} con código {returncode}. Revisa el log: {log_path}"
        )
    print(f"[OK] {name} | elapsed={format_duration(elapsed)}", flush=True)
    return result



def run_command_batch(
    jobs: Sequence[tuple[str, Sequence[str]]],
    cwd: Path,
    log_dir: Path,
    dry_run: bool = False,
    parallel_jobs: int = 1,
) -> list[StageResult]:
    """
    Ejecuta una lista de comandos.

    Si parallel_jobs <= 1, corre secuencialmente.
    Si parallel_jobs > 1, corre comandos independientes en paralelo.
    Cada comando escribe su propio log, por lo que es seguro para etapas por punto.
    """
    if not jobs:
        return []

    if parallel_jobs <= 1 or len(jobs) == 1:
        results = []
        for name, cmd in jobs:
            results.append(run_command(name, cmd, cwd=cwd, log_dir=log_dir, dry_run=dry_run))
        return results

    results: list[StageResult] = []
    with ThreadPoolExecutor(max_workers=parallel_jobs) as ex:
        future_to_name = {
            ex.submit(run_command, name, cmd, cwd, log_dir, dry_run): name
            for name, cmd in jobs
        }
        for fut in as_completed(future_to_name):
            results.append(fut.result())

    # Orden estable por nombre de etapa para que el manifest sea reproducible.
    results.sort(key=lambda r: r.name)
    return results


def write_outputs_index(outdir: Path, rows: list[dict[str, str]]) -> None:
    csv_path = outdir / "pipeline_outputs.csv"
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "point", "kind", "path"])
        writer.writeheader()
        writer.writerows(rows)


def cleanup_empty_run_dirs(outdir: Path) -> list[Path]:
    """Remove empty run directories while preserving Highland placeholders."""
    keep_names = {"07_scattering", "08_smearing"}
    removed: list[Path] = []
    if not outdir.exists():
        return removed

    directories = [p for p in outdir.rglob("*") if p.is_dir()]
    directories.sort(key=lambda p: len(p.parts), reverse=True)
    for directory in directories:
        if directory.name in keep_names:
            continue
        try:
            directory.rmdir()
        except OSError:
            continue
        removed.append(directory)
    return removed


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Orquestador completo: FOV -> longitud -> Ecrit -> filtro -> mapas -> inside -> scattering -> smearing."
    )
    ap.add_argument("--profile", choices=["standard", "bariloche-smoke"], default="standard",
                    help="Preset cómodo. bariloche-smoke usa data/bariloche_5min.shw, módulos, kernel empírico y event-MC.")
    ap.add_argument("--scripts-dir", default="auto", help="Carpeta de scripts. Default: autodetecta modulos/.")
    ap.add_argument("--hgt-dir", default="auto", help="Carpeta con N04W076.hgt y N04W075.hgt. Default: autodetecta data/.")
    ap.add_argument("--range-file", default=None, help="data_rock.dat o muon_range_table.csv. Si no se da, se busca automáticamente.")
    ap.add_argument("--shw", default=None, help="Archivo .shw, .shw.gz/.xz/.bz2 o .tar/.tar.gz con un .shw dentro. Si se omite, no se filtran muones ni se grafican conteos.")
    ap.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="auto",
                    help="Formato de líneas SHW: auto, arti12 o cnf9 (CNFId energy theta px py pz h bx bz).")
    ap.add_argument("--shw-member", default=None,
                    help="Ruta interna del .shw dentro del .tar si hay más de un archivo. Default: primer .shw encontrado.")
    ap.add_argument("--storage-profile", choices=["normal", "compact"], default="normal",
                    help="normal conserva salidas históricas; compact comprime los .shw filtrados si no se indicó otra compresión.")
    ap.add_argument("--filtered-compression", choices=["none", "gz", "xz", "bz2"], default="none",
                    help="Compresión de 04_filtered/*.shw. gz suele ser el mejor equilibrio entre espacio y velocidad.")
    ap.add_argument("--fast-cache", action="store_true",
                    help="Ruta rápida filtered-only: lee el SHW una vez, evita escribir .shw filtrados y genera cache/mapas/inside.")
    ap.add_argument("--kinematic-cache", default=None,
                    help="Directorio de cache cinemático global. Si no existe y hay --shw, se construye antes de --fast-cache.")
    ap.add_argument("--rebuild-kinematic-cache", action="store_true",
                    help="Regenera --kinematic-cache aunque ya exista.")
    ap.add_argument("--kinematic-cache-chunk-events", type=int, default=1_000_000,
                    help="Eventos por chunk al construir el cache cinemático.")
    ap.add_argument("--outdir", default="run_machin", help="Carpeta raíz de salida")
    ap.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS), choices=list(DEFAULT_POINTS), help="Puntos a procesar")
    ap.add_argument("--rho", type=float, default=2.65, help="Densidad efectiva de roca en g/cm^3")
    ap.add_argument("--tol-phi", type=float, default=0.51, help="Tolerancia angular en φ para filtrar .shw")
    ap.add_argument("--tol-theta", type=float, default=0.51, help="Tolerancia angular en θ para filtrar .shw")
    ap.add_argument("--treat-out-of-grid-as-clear", type=int, choices=[0, 1], default=1, help="1 conserva muones fuera de la grilla Ecrit")
    ap.add_argument("--discard-upgoing", action="store_true", help="Descarta muones con pz > 0 en el filtrado")
    ap.add_argument("--plot-source", choices=["none", "raw", "filtered", "both"], default="both", help="Qué .shw graficar con 05_plot_theta_phi.py")
    ap.add_argument("--inside-volcano-source", choices=["none", "raw", "filtered", "both"], default="both", help="Grafica cuentas dentro del volcán. Default: raw y filtered.")
    ap.add_argument("--inside-mask-min", type=float, default=0.0, help="Umbral para máscara de volcán: celda dentro si mask_col > inside_mask_min")
    ap.add_argument("--inside-mask-col", default=None, help="Columna para definir máscara; por defecto se autodetecta en rock_length_P*.csv")
    ap.add_argument("--inside-theta-min", type=float, default=0.0, help="Theta mínimo para contar dentro de la geometría")
    ap.add_argument("--inside-theta-max", type=float, default=90.0, help="Theta máximo para contar dentro de la geometría")
    ap.add_argument("--inside-display-theta-min", type=float, default=0.0, help="Theta mínimo del canvas final inside-volcano")
    ap.add_argument("--inside-display-theta-max", type=float, default=90.0, help="Theta máximo del canvas final inside-volcano")
    ap.add_argument("--inside-display-phi-min", type=float, default=-60.0, help="Phi mínimo del canvas final inside-volcano")
    ap.add_argument("--inside-display-phi-max", type=float, default=60.0, help="Phi máximo del canvas final inside-volcano")
    ap.add_argument("--inside-display-step", type=float, default=0.5, help="Bineado visual cuadrado del plot final inside-volcano")
    ap.add_argument("--inside-vmax-percentile", type=float, default=99.0, help="Percentil superior de color para escala lineal inside-volcano")
    ap.add_argument("--inside-filtered-workers", type=int, default=0, help="Procesos para inside-volcano filtered; 0 autodetecta.")
    ap.add_argument("--inside-show-zeros", action="store_true", help="Muestra ceros en los mapas inside-volcano; por defecto quedan en blanco")
    ap.add_argument("--filter-mode", choices=["fast", "legacy"], default="fast", help="fast usa 06 multi-punto; legacy usa la interfaz vieja por punto")
    ap.add_argument("--parallel-jobs", type=int, default=0, help="Número de procesos paralelos para etapas por punto; 0 autodetecta.")
    ap.add_argument("--make-4panel", action="store_true", help="Genera figuras 2x2 con plot_4panel_muon_maps.py al final")
    ap.add_argument("--fourpanel-display-step", type=float, default=0.5, help="Paso angular visual para figuras 2x2 con square-display")

    # 08_scattering_highland_v2.py
    ap.add_argument("--skip-scattering", action="store_true", help="No ejecuta 08_scattering_highland_v2.py")
    ap.add_argument("--scattering-energy-factors", nargs="+", type=float, default=[1.0, 1.5, 2.0], help="Factores Eref_total = factor * Ecrit_total para el 08")
    ap.add_argument("--scattering-X0", type=float, default=26.54, help="Longitud de radiación de roca en g/cm^2 para Highland")
    ap.add_argument("--scattering-charge", type=float, default=1.0, help="Número de carga |z| para Highland")
    ap.add_argument("--scattering-theta-bin-deg", type=float, default=None, help="Fallback de bin θ para el 08")
    ap.add_argument("--scattering-phi-bin-deg", type=float, default=None, help="Fallback de bin φ para el 08")
    ap.add_argument("--scattering-theta-min", type=float, default=None, help="Theta mínimo para el 08; default inferido")
    ap.add_argument("--scattering-theta-max", type=float, default=90.0, help="Theta máximo para el 08")
    ap.add_argument("--scattering-phi-min", type=float, default=None, help="Phi mínimo para el 08; default inferido")
    ap.add_argument("--scattering-phi-max", type=float, default=None, help="Phi máximo para el 08; default inferido")
    ap.add_argument("--scattering-display-step", type=float, default=0.5, help="Paso visual cuadrado para las figuras del 08")
    ap.add_argument("--scattering-square-display", dest="scattering_square_display", action="store_true", default=True, help="Usa canvas cuadrado en el 08")
    ap.add_argument("--scattering-native-display", dest="scattering_square_display", action="store_false", help="Usa display nativo en el 08")

    # 09_apply_angular_smearing_pretty_MC.py
    ap.add_argument("--skip-smearing", action="store_true", help="No ejecuta 09_apply_angular_smearing_pretty_MC.py")
    ap.add_argument("--smearing-source", choices=["raw", "filtered", "both"], default="filtered", help="Fuente de mapas para aplicar smearing. Usa both para correr raw y filtered.")
    ap.add_argument("--smearing-energy-factors", nargs="+", type=float, default=None, help="Factores para el 09; default: usa --scattering-energy-factors")
    ap.add_argument("--smearing-sigma-col", default="theta0_proj_deg", help="Columna de sigma angular en scattering_table")
    ap.add_argument("--smearing-theta-min", type=float, default=None, help="Theta mínimo para el 09; default inferido desde CSV")
    ap.add_argument("--smearing-theta-max", type=float, default=90.0, help="Theta máximo para el 09")
    ap.add_argument("--smearing-phi-min", type=float, default=-60.0, help="Phi mínimo para el 09")
    ap.add_argument("--smearing-phi-max", type=float, default=60.0, help="Phi máximo para el 09")
    ap.add_argument("--smearing-display-step", type=float, default=0.5, help="Paso visual cuadrado para el 09")
    ap.add_argument("--smearing-vmax-percentile", type=float, default=99.0, help="Percentil superior de color para conteos en el 09")
    ap.add_argument("--smearing-relative-vmax-percentile", type=float, default=98.0, help="Percentil superior para cambio relativo en el 09")
    ap.add_argument("--smearing-kernel-radius-sigma", type=float, default=4.0, help="Radio del kernel en sigmas para el 09")
    ap.add_argument("--smearing-detector-sigma-deg", type=float, default=0.0, help="Sigma angular instrumental extra para el 09")
    ap.add_argument("--smearing-sigma-scale", type=float, default=1.0, help="Factor multiplicativo de sigma para el 09")
    ap.add_argument("--smearing-stochastic", action="store_true", help="Usa realización Monte Carlo en el 09")
    ap.add_argument("--smearing-random-seed", type=int, default=12345, help="Semilla RNG para --smearing-stochastic")

    # Rama empírica Geant4, paralela a Highland
    ap.add_argument("--scattering-model", choices=["highland", "empirical", "both"], default="highland",
                    help="Modelo de dispersión angular a ejecutar. Default: highland para conservar compatibilidad.")
    ap.add_argument("--empirical-kernel-library", default=None,
                    help="Ruta a empirical_kernel_library.npz. Default: autodetecta en modulos/ o data/.")
    ap.add_argument("--empirical-interp-method", choices=["tail-aware", "rbf_linear", "linear", "nearest"], default="tail-aware",
                    help="Interpolacion del kernel; tail-aware preserva las colas medidas y el hard scattering.")
    ap.add_argument("--empirical-energy-factors", nargs="+", type=float, default=None,
                    help="Factores Tref = factor*Tcrit para la rama empírica. Default: usa --scattering-energy-factors.")
    ap.add_argument("--empirical-stochastic", action="store_true",
                    help="Usa realización Monte Carlo para el smearing empírico.")
    ap.add_argument("--empirical-random-seed", type=int, default=12345,
                    help="Semilla RNG para --empirical-stochastic.")
    ap.add_argument("--empirical-display-step", type=float, default=None,
                    help="Paso visual del smearing empírico. Default: usa --smearing-display-step.")
    ap.add_argument("--empirical-theta-min", type=float, default=None,
                    help="Theta mínimo para la rama empírica. Default: usa --smearing-theta-min.")
    ap.add_argument("--empirical-theta-max", type=float, default=None,
                    help="Theta máximo para la rama empírica. Default: usa --smearing-theta-max.")
    ap.add_argument("--empirical-phi-min", type=float, default=None,
                    help="Phi mínimo para la rama empírica. Default: usa --smearing-phi-min.")
    ap.add_argument("--empirical-phi-max", type=float, default=None,
                    help="Phi máximo para la rama empírica. Default: usa --smearing-phi-max.")
    ap.add_argument("--empirical-kernel-threshold", type=float, default=0.0,
                    help="Corte de densidad del kernel. Default 0 conserva las colas de hard scattering.")
    ap.add_argument("--empirical-max-kernel-radius-mrad", type=float, default=None,
                    help="Opcional: limita el radio angular evaluado del kernel empírico.")

    # modulos/10_apply_event_by_event_empirical_mc_v2.py
    ap.add_argument("--run-event-mc", action="store_true",
                    help="Ejecuta el MC empírico evento-por-evento usando energía real del .shw y L(theta,phi).")
    ap.add_argument("--skip-event-mc", action="store_true",
                    help="No ejecuta el MC evento-por-evento, aunque --run-event-mc esté activo.")
    ap.add_argument("--event-mc-source", choices=["raw", "filtered", "both"], default="filtered",
                    help="Fuente .shw para el MC evento-por-evento. Usa both para correr raw y filtered.")
    ap.add_argument("--event-mc-source-mode", choices=["all", "inside"], default="inside",
                    help="all: propaga todos los muones. inside: sólo usa como fuente los que originalmente estaban dentro del borde.")
    ap.add_argument("--event-mc-workers", type=int, default=None,
                    help="Workers para el MC evento-por-evento. Default: usa --parallel-jobs.")
    ap.add_argument("--event-mc-energy-cache-dlog", type=float, default=0.05,
                    help="Cuantización logarítmica de energía para cachear kernels. 0 desactiva cache aproximada.")
    ap.add_argument("--event-mc-kernel-threshold", type=float, default=None,
                    help="Umbral de densidad para soporte local del MC evento-por-evento. Default: usa --empirical-kernel-threshold.")
    ap.add_argument("--event-mc-max-kernel-radius-mrad", type=float, default=None,
                    help="Radio máximo opcional del kernel en el MC evento-por-evento.")
    ap.add_argument("--event-mc-random-seed", type=int, default=12345,
                    help="Semilla RNG para el MC evento-por-evento.")
    ap.add_argument("--event-mc-head", type=int, default=0,
                    help="Limita los eventos procesados por punto en el event-MC; 0 usa todos.")
    ap.add_argument("--event-mc-theta-min", type=float, default=None,
                    help="Theta mínimo para el canvas del MC evento-por-evento. Default: usa --plot-theta-min.")
    ap.add_argument("--event-mc-theta-max", type=float, default=None,
                    help="Theta máximo para el canvas del MC evento-por-evento. Default: usa --plot-theta-max.")
    ap.add_argument("--event-mc-phi-min", type=float, default=None,
                    help="Phi mínimo para el canvas del MC evento-por-evento. Default: usa --plot-phi-min.")
    ap.add_argument("--event-mc-phi-max", type=float, default=None,
                    help="Phi máximo para el canvas del MC evento-por-evento. Default: usa --plot-phi-max.")
    ap.add_argument("--event-mc-display-step", type=float, default=2.5,
                    help="Paso angular extra para muogramas rebineados del MC evento-por-evento. Default: 2.5 grados.")

    # modulos/12_apply_in_scattering_background.py
    ap.add_argument("--run-in-scattering", action="store_true",
                    help="Ejecuta estimación angular-only de contaminación externa -> acceptance por MCS.")
    ap.add_argument("--skip-in-scattering", action="store_true",
                    help="No ejecuta in-scattering aunque --run-in-scattering esté activo.")
    ap.add_argument("--in-scattering-step-m", type=float, default=100.0,
                    help="Paso de propagación en roca para in-scattering.")
    ap.add_argument("--in-scattering-samples-per-muon", type=int, default=1,
                    help="Muestras MC por muón externo seleccionado.")
    ap.add_argument("--in-scattering-max-angular-margin-deg", type=float, default=None,
                    help="Margen angular externo opcional. Si se omite, usa todo el complemento F\\A.")
    ap.add_argument("--in-scattering-seed", type=int, default=12345,
                    help="Semilla RNG para in-scattering.")
    ap.add_argument("--in-scattering-workers", type=int, default=1,
                    help="Reservado para el módulo 12. Usa --parallel-jobs para correr varios puntos en paralelo; kinematic-cache usa cribado vectorizado por punto.")
    ap.add_argument("--in-scattering-theta-min-deg", type=float, default=None,
                    help="Theta mínimo del dominio físico F. Default: módulo 12 usa 0.")
    ap.add_argument("--in-scattering-theta-max-deg", type=float, default=None,
                    help="Theta máximo del dominio físico F. Default: módulo 12 usa 180.")
    ap.add_argument("--in-scattering-phi-min-deg", type=float, default=None,
                    help="Phi relativo mínimo del dominio físico F. Default: módulo 12 usa -180.")
    ap.add_argument("--in-scattering-phi-max-deg", type=float, default=None,
                    help="Phi relativo máximo del dominio físico F. Default: módulo 12 usa 180.")
    ap.add_argument("--in-scattering-external-length-mode", choices=["hybrid", "dem", "length-map"], default="hybrid",
                    help="Fuente de longitud externa: length-map, DEM o híbrida.")
    ap.add_argument("--in-scattering-external-s-max-m", type=float, default=5000.0,
                    help="Longitud máxima del rayo DEM externo.")
    ap.add_argument("--in-scattering-external-ray-step-m", type=float, default=5.0,
                    help="Paso del trazador DEM externo.")
    ap.add_argument("--in-scattering-length-cache-step-deg", type=float, default=0.5,
                    help="Cuantización angular para cachear longitudes externas por DEM.")
    ap.add_argument("--in-scattering-kernel-threshold", type=float, default=None,
                    help="Umbral de densidad del kernel para muestrear deflexiones. Default: usa --empirical-kernel-threshold.")
    ap.add_argument("--in-scattering-kernel-scale", type=float, default=1.0,
                    help="Escala artificial de deflexión para validaciones.")
    ap.add_argument("--in-scattering-disable-scattering", action="store_true",
                    help="Propaga energía sin deflexión angular; útil para validar aceptación externa ~0.")
    ap.add_argument("--in-scattering-debug-trajectories", action="store_true",
                    help="Guarda CSV de trayectorias aceptadas.")
    ap.add_argument("--in-scattering-no-figures", action="store_true",
                    help="No genera figuras opcionales de in-scattering.")
    ap.add_argument("--in-scattering-head", type=int, default=0,
                    help="Debug: detiene cada punto tras N muones externos seleccionados.")

    # modulos/13_apply_spatial_in_scattering_dem.py
    ap.add_argument("--run-spatial-in-scattering", action="store_true",
                    help="Ejecuta diagnóstico espacial DEM de in-scattering externo -> máscara angular.")
    ap.add_argument("--skip-spatial-in-scattering", action="store_true",
                    help="No ejecuta in-scattering espacial aunque --run-spatial-in-scattering esté activo.")
    ap.add_argument("--spatial-in-scattering-ray-step-m", type=float, default=100.0,
                    help="Paso de transporte DEM para in-scattering espacial.")
    ap.add_argument("--spatial-in-scattering-sample-probability", type=float, default=0.01,
                    help="Probabilidad de muestrear cada evento del cache; eventos retenidos pesan 1/p.")
    ap.add_argument("--spatial-in-scattering-samples-per-muon", type=int, default=1,
                    help="Muestras de posición por muón muestreado.")
    ap.add_argument("--spatial-in-scattering-source-surface", choices=["entry-box", "top-plane", "volcano-surface"], default="entry-box",
                    help="Superficie espacial de entrada: caja DEM, plano superior legacy o superficie DEM del volcan.")
    ap.add_argument("--spatial-in-scattering-volcano-surface-grid-step-m", type=float, default=50.0,
                    help="Paso de grilla para construir la superficie volcanica del modulo 13.")
    ap.add_argument("--spatial-in-scattering-volcano-surface-edge-guard-m", type=float, default=500.0,
                    help="Margen contra el borde DEM para superficie volcanica del modulo 13.")
    ap.add_argument("--spatial-in-scattering-volcano-surface-min-height-frac", type=float, default=0.0,
                    help="Altura relativa minima [0,1] de la superficie volcanica objetivo.")
    ap.add_argument("--spatial-in-scattering-volcano-surface-entry-check-m", type=float, default=10.0,
                    help="Distancia de prueba para exigir entrada inmediata en roca desde superficie volcanica.")
    ap.add_argument("--spatial-in-scattering-max-angular-margin-deg", type=float, default=None,
                    help="Solo propaga direcciones externas a esta distancia angular de la mascara aceptada.")
    ap.add_argument("--spatial-in-scattering-entry-face-importance", default="",
                    help="Importance sampling por cara para módulo 13, e.g. south:4,west:4,top:1,east:0.5,north:0.5.")
    ap.add_argument("--spatial-in-scattering-min-survival-rock-m", type=float, default=None,
                    help="Corte temprano por rango CSDA; default del módulo: ray-step-m.")
    ap.add_argument("--spatial-in-scattering-seed", type=int, default=12345,
                    help="Semilla RNG para in-scattering espacial.")
    ap.add_argument("--spatial-in-scattering-head", type=int, default=0,
                    help="Debug: detiene cada punto tras N eventos de flujo leídos; 0 usa todo el cache.")
    ap.add_argument("--spatial-in-scattering-observer-radius-m", type=float, default=0.0,
                    help="Si >0 exige paso cerca de P1; 0 mantiene diagnóstico angular DEM sin detector físico.")
    ap.add_argument("--spatial-in-scattering-kernel-scale", type=float, default=1.0,
                    help="Escala artificial de deflexión para validaciones.")
    ap.add_argument("--spatial-in-scattering-disable-scattering", action="store_true",
                    help="Control: transporte sin deflexión angular.")
    ap.add_argument("--spatial-in-scattering-no-figures", action="store_true",
                    help="No genera figuras opcionales del diagnóstico espacial.")
    ap.add_argument("--plot-theta-min", type=float, default=60.0)
    ap.add_argument("--plot-theta-max", type=float, default=90.0)
    ap.add_argument("--plot-phi-min", type=float, default=-50.0)
    ap.add_argument("--plot-phi-max", type=float, default=50.0)
    ap.add_argument("--bins-theta", type=int, default=60)
    ap.add_argument("--bins-phi", type=int, default=40)
    ap.add_argument("--compute-device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="Backend deseado. Hoy GPU requiere CuPy; si no está disponible se informa y se usa CPU.")
    ap.add_argument("--status-interval-s", type=float, default=DEFAULT_STATUS_INTERVAL_S,
                    help="Segundos entre mensajes de estado durante etapas largas; 0 desactiva el latido.")
    ap.add_argument("--force", action="store_true", help="Borra la carpeta de salida antes de correr")
    ap.add_argument("--dry-run", action="store_true", help="Sólo imprime/guarda comandos; no ejecuta scripts")
    ap.add_argument("--skip-geometry", action="store_true")
    ap.add_argument("--skip-lengths", action="store_true")
    ap.add_argument("--skip-ecrit", action="store_true")
    ap.add_argument("--skip-filter", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    return ap


def main() -> int:
    global STATUS_INTERVAL_S
    args = build_parser().parse_args()
    if args.status_interval_s < 0:
        raise ValueError("--status-interval-s no puede ser negativo.")
    STATUS_INTERVAL_S = float(args.status_interval_s)
    apply_profile(args)
    if args.storage_profile == "compact" and args.filtered_compression == "none":
        args.filtered_compression = "gz"
    if args.fast_cache:
        if args.plot_source == "both":
            args.plot_source = "filtered"
        if args.inside_volcano_source == "both":
            args.inside_volcano_source = "filtered"
        if args.smearing_source == "both":
            args.smearing_source = "filtered"
        if args.event_mc_source == "both":
            args.event_mc_source = "filtered"
        if args.plot_source == "raw":
            raise ValueError("--fast-cache es filtered-only; usa --plot-source filtered/none.")
        if args.inside_volcano_source == "raw":
            raise ValueError("--fast-cache es filtered-only; usa --inside-volcano-source filtered/none.")
        if args.smearing_source == "raw":
            raise ValueError("--fast-cache es filtered-only; usa --smearing-source filtered.")
        if args.event_mc_source == "raw":
            raise ValueError("--fast-cache es filtered-only; usa --event-mc-source filtered.")

    scripts_dir = resolve_scripts_dir(resolve_path(args.scripts_dir))
    hgt_dir = resolve_hgt_dir(resolve_path(args.hgt_dir))
    outdir = resolve_path(args.outdir)
    range_file_arg = resolve_path(args.range_file)
    shw = resolve_path(args.shw)
    kinematic_cache = resolve_path(args.kinematic_cache)
    empirical_kernel_library_arg = resolve_path(args.empirical_kernel_library)

    assert outdir is not None
    args.parallel_jobs = normalize_workers(args.parallel_jobs, len(args.points))
    args.inside_filtered_workers = normalize_workers(args.inside_filtered_workers, len(args.points))
    compute_backend = detect_compute_backend(args.compute_device)
    if compute_backend["note"]:
        print(f"[COMPUTE] {compute_backend['note']}")

    scripts = check_scripts(scripts_dir)
    empirical_kernel_library = find_empirical_kernel(empirical_kernel_library_arg, scripts_dir, hgt_dir)
    if args.fast_cache:
        scripts["event_cache"] = find_script(SCRIPT_04_EVENT_CACHE, scripts_dir, "event_cache")
        if kinematic_cache is not None:
            scripts["kinematic_cache"] = find_script(SCRIPT_04_KINEMATIC_CACHE, scripts_dir, "kinematic_cache")

    if args.scattering_model in ("empirical", "both"):
        scripts["scattering_empirical"] = find_script(SCRIPT_08_EMPIRICAL, scripts_dir, "scattering_empirical")
        scripts["smearing_empirical"] = find_script(SCRIPT_09_EMPIRICAL, scripts_dir, "smearing_empirical")
        if empirical_kernel_library is None:
            raise FileNotFoundError(
                "No encontré empirical_kernel_library.npz. Usa --empirical-kernel-library ruta/al/kernel.npz"
            )
        require_file(empirical_kernel_library, "empirical_kernel_library.npz")

    if args.run_event_mc and (not args.skip_event_mc):
        scripts["event_mc_empirical"] = find_script(
            SCRIPT_10_EVENT_MC,
            scripts_dir,
            "event_mc_empirical",
            alternate_names=("10_apply_event_by_event_empirical_mc.py",),
        )
        if empirical_kernel_library is None:
            raise FileNotFoundError(
                "No encontré empirical_kernel_library.npz. Usa --empirical-kernel-library ruta/al/kernel.npz"
            )
        require_file(empirical_kernel_library, "empirical_kernel_library.npz")

    if args.run_in_scattering and (not args.skip_in_scattering):
        scripts["in_scattering"] = find_script(
            SCRIPT_12_IN_SCATTERING,
            scripts_dir,
            "in_scattering",
        )
        if empirical_kernel_library is None:
            raise FileNotFoundError(
                "No encontré empirical_kernel_library.npz. Usa --empirical-kernel-library ruta/al/kernel.npz"
            )
        require_file(empirical_kernel_library, "empirical_kernel_library.npz")
        if shw is None and kinematic_cache is None:
            raise FileNotFoundError(
                "--run-in-scattering requiere --shw o --kinematic-cache para leer flujo abierto."
            )

    if args.run_spatial_in_scattering and (not args.skip_spatial_in_scattering):
        scripts["spatial_in_scattering"] = find_script(
            SCRIPT_13_SPATIAL_IN_SCATTERING,
            scripts_dir,
            "spatial_in_scattering",
        )
        if empirical_kernel_library is None:
            raise FileNotFoundError(
                "No encontré empirical_kernel_library.npz. Usa --empirical-kernel-library ruta/al/kernel.npz"
            )
        require_file(empirical_kernel_library, "empirical_kernel_library.npz")
        if kinematic_cache is None:
            raise FileNotFoundError(
                "--run-spatial-in-scattering requiere --kinematic-cache o una corrida que construya cache cinemático."
            )

    if args.inside_volcano_source != "none":
        merged_07 = scripts_dir / SCRIPT_07_MERGED
        if merged_07.exists():
            scripts["inside_volcano_merged"] = merged_07
        else:
            raise FileNotFoundError(
                f"No encontré {merged_07}. Este orquestador usa el 07 unido "
                f"para generar tablas, figuras individuales y figuras 2x2 con bineado cuadrado."
            )

    if args.make_4panel:
        scripts["plot_4panel"] = find_script(SCRIPT_4PANEL, scripts_dir, "plot_4panel", required=False)
        if scripts["plot_4panel"] is None:
            print("[WARN] --make-4panel omitido: no encontré plot_4panel_muon_maps.py.")
            args.make_4panel = False

    require_file(hgt_dir / REQUIRED_HGT[0], REQUIRED_HGT[0])
    require_file(hgt_dir / REQUIRED_HGT[1], REQUIRED_HGT[1])
    if shw is not None:
        require_file(shw, "archivo .shw")
    input_shw_stem = detect_shw_stem(shw) if shw is not None else "input"

    if args.force and outdir.exists():
        shutil.rmtree(outdir)
    ensure_dir(outdir)

    dirs = {
        "inputs": outdir / "00_inputs",
        "kinematic_cache_stage": outdir / "00_kinematic_cache",
        "geometry": outdir / "01_geometry",
        "lengths": outdir / "02_lengths",
        "ecrit": outdir / "03_ecrit",
        "event_cache": outdir / "04_event_cache",
        "filtered": outdir / "04_filtered",
        "plots": outdir / "05_plots",
        "inside_volcano": outdir / "06_inside_volcano",
        "scattering": outdir / "07_scattering",
        "smearing": outdir / "08_smearing",
        "scattering_empirical": outdir / "07_scattering_empirical",
        "smearing_empirical": outdir / "08_smearing_empirical",
        "event_mc_empirical": outdir / "09_event_mc_empirical",
        "in_scattering": outdir / "10_in_scattering_background",
        "logs": outdir / "logs",
        "work_geometry": outdir / "_work_geometry",
    }
    for key, directory in dirs.items():
        if key != "work_geometry":
            ensure_dir(directory)

    # Enlace/copia de entradas básicas para trazabilidad.
    for fname in REQUIRED_HGT:
        link_or_copy(hgt_dir / fname, dirs["inputs"] / fname, overwrite=args.force)

    if shw is not None:
        link_or_copy(shw, dirs["inputs"] / shw.name, overwrite=args.force)

    range_file = find_range_file(
        range_file_arg,
        search_dirs=[dirs["inputs"], hgt_dir, scripts_dir, Path.cwd()],
    )
    if (not args.skip_ecrit) and range_file is None:
        raise FileNotFoundError(
            "No encontré data_rock.dat ni muon_range_table.csv. Usa --range-file /ruta/data_rock.dat"
        )
    if range_file is not None:
        link_or_copy(range_file, dirs["inputs"] / range_file.name, overwrite=args.force)
    if empirical_kernel_library is not None:
        link_or_copy(empirical_kernel_library, dirs["inputs"] / empirical_kernel_library.name, overwrite=args.force)

    stage_results: list[StageResult] = []
    output_rows: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # 00. Cache cinemático global opcional
    # ------------------------------------------------------------------
    if args.fast_cache and kinematic_cache is not None:
        build_kinematic = args.rebuild_kinematic_cache or not (kinematic_cache / "manifest.json").exists()
        if build_kinematic:
            if shw is None:
                raise FileNotFoundError("--kinematic-cache no existe y no se puede construir sin --shw")
            print(f"[SETUP] Cache cinemático compacto -> {kinematic_cache}")
            cmd = [
                sys.executable,
                scripts["kinematic_cache"],
                "--shw", shw,
                "--shw-format", args.shw_format,
                "--out", kinematic_cache,
                "--chunk-events", str(args.kinematic_cache_chunk_events),
                "--force",
            ]
            if args.shw_member:
                cmd.extend(["--shw-member", args.shw_member])
            result = run_command("00_kinematic_cache", cmd, cwd=outdir, log_dir=dirs["logs"], dry_run=args.dry_run)
            stage_results.append(result)
        else:
            print(f"[SETUP] Reuso cache cinemático -> {kinematic_cache}")

        output_rows.append({
            "stage": "00_kinematic_cache",
            "point": "ALL",
            "kind": "manifest",
            "path": str(kinematic_cache / "manifest.json"),
        })
        if not args.dry_run:
            require_file(kinematic_cache / "manifest.json", "manifest cache cinemático")

    # ------------------------------------------------------------------
    # 01. Geometría / FOV
    # ------------------------------------------------------------------
    if not args.skip_geometry:
        remove_and_create(dirs["work_geometry"])
        for fname in REQUIRED_HGT:
            link_or_copy(hgt_dir / fname, dirs["work_geometry"] / fname, overwrite=True)

        cmd = [sys.executable, scripts["geometry"]]
        print(f"[PHASE 1/9] Geometría y ángulos bloqueados -> {dirs['geometry']}")
        result = run_command("01_geometry", cmd, cwd=dirs["work_geometry"], log_dir=dirs["logs"], dry_run=args.dry_run)
        stage_results.append(result)

        if not args.dry_run:
            copy_tree_contents(dirs["work_geometry"] / "outputs", dirs["geometry"], overwrite=True)
            for fname in REQUIRED_HGT:
                link_or_copy(hgt_dir / fname, dirs["geometry"] / fname, overwrite=True)

            required = [dirs["geometry"] / f"blocked_angles_{p}.csv" for p in args.points]
            required.append(dirs["geometry"] / "dem_fans.png")
            require_files(required, "01_geometry")
            shutil.rmtree(dirs["work_geometry"], ignore_errors=True)

        output_rows.append({"stage": "01_geometry", "point": "ALL", "kind": "dem", "path": str(dirs["geometry"] / "dem_fans.png")})
        for p in args.points:
            output_rows.extend([
                {"stage": "01_geometry", "point": p, "kind": "blocked_angles", "path": str(dirs["geometry"] / f"blocked_angles_{p}.csv")},
                {"stage": "01_geometry", "point": p, "kind": "fov_png", "path": str(dirs["geometry"] / f"fov_{p}.png")},
            ])
    else:
        for fname in REQUIRED_HGT:
            link_or_copy(hgt_dir / fname, dirs["geometry"] / fname, overwrite=args.force)

    # ------------------------------------------------------------------
    # 02. Longitudes dentro de roca
    # ------------------------------------------------------------------
    if not args.skip_lengths:
        cmd = [
            sys.executable,
            scripts["lengths"],
            "--data_dir", dirs["geometry"],
            "--outdir", dirs["lengths"],
        ]
        print(f"[PHASE 2/9] Longitudes dentro de roca -> {dirs['lengths']}")
        result = run_command("02_lengths", cmd, cwd=outdir, log_dir=dirs["logs"], dry_run=args.dry_run)
        stage_results.append(result)

        if not args.dry_run:
            required = [dirs["lengths"] / f"rock_length_{p}.csv" for p in args.points]
            required.append(dirs["lengths"] / "summary.csv")
            require_files(required, "02_lengths")

        for p in args.points:
            output_rows.extend([
                {"stage": "02_lengths", "point": p, "kind": "rock_length", "path": str(dirs["lengths"] / f"rock_length_{p}.csv")},
                {"stage": "02_lengths", "point": p, "kind": "rock_heatmap", "path": str(dirs["lengths"] / f"heatmap_{p}.png")},
            ])
        output_rows.append({"stage": "02_lengths", "point": "ALL", "kind": "summary", "path": str(dirs["lengths"] / "summary.csv")})

    # ------------------------------------------------------------------
    # 03. Energía crítica
    # ------------------------------------------------------------------
    if not args.skip_ecrit:
        if range_file is not None:
            link_or_copy(range_file, dirs["lengths"] / range_file.name, overwrite=args.force)

        cmd = [
            sys.executable,
            scripts["ecrit"],
            "--indir", dirs["lengths"],
            "--outdir", dirs["ecrit"],
            "--rho", str(args.rho),
            "--points", *args.points,
        ]
        print(f"[PHASE 3/9] Energía crítica -> {dirs['ecrit']}")
        result = run_command("03_ecrit", cmd, cwd=outdir, log_dir=dirs["logs"], dry_run=args.dry_run)
        stage_results.append(result)

        if not args.dry_run:
            required = [dirs["ecrit"] / f"ecrit_table_{p}.csv" for p in args.points]
            require_files(required, "03_ecrit")

        for p in args.points:
            output_rows.extend([
                {"stage": "03_ecrit", "point": p, "kind": "ecrit_table", "path": str(dirs["ecrit"] / f"ecrit_table_{p}.csv")},
                {"stage": "03_ecrit", "point": p, "kind": "Tcrit_heatmap", "path": str(dirs["ecrit"] / f"Tcrit_heatmap_{p}.png")},
                {"stage": "03_ecrit", "point": p, "kind": "Etotal_heatmap", "path": str(dirs["ecrit"] / f"Etotal_heatmap_{p}.png")},
            ])

    # ------------------------------------------------------------------
    # 04-fast. Cache compacto + mapas filtered en una sola lectura del SHW
    # ------------------------------------------------------------------
    do_fast_cache = args.fast_cache and ((shw is not None) or (kinematic_cache is not None)) and (not args.skip_filter)
    reuse_fast_cache = args.fast_cache and args.skip_filter
    fast_cache_products_available = do_fast_cache or reuse_fast_cache
    event_cache_by_point: dict[str, Path] = {}
    if do_fast_cache:
        print(f"[PHASE 4/9] Cache rápido de eventos + mapas filtered -> {dirs['event_cache']}")
        cmd = [
            sys.executable,
            scripts["event_cache"],
            "--points", *args.points,
            "--ecrit-dir", dirs["ecrit"],
            "--cache-outdir", dirs["event_cache"],
            "--plot-outdir", dirs["plots"],
            "--inside-outdir", dirs["inside_volcano"],
            "--bins-theta", str(args.bins_theta),
            "--bins-phi", str(args.bins_phi),
            "--plot-theta-min", str(args.plot_theta_min),
            "--plot-theta-max", str(args.plot_theta_max),
            "--plot-phi-min", str(args.plot_phi_min),
            "--plot-phi-max", str(args.plot_phi_max),
            "--tol-phi", str(args.tol_phi),
            "--tol-theta", str(args.tol_theta),
            "--treat-out-of-grid-as-clear", str(args.treat_out_of_grid_as_clear),
            "--inside-display-theta-min", str(args.inside_display_theta_min),
            "--inside-display-theta-max", str(args.inside_display_theta_max),
            "--inside-display-phi-min", str(args.inside_display_phi_min),
            "--inside-display-phi-max", str(args.inside_display_phi_max),
            "--inside-display-step", str(args.inside_display_step),
            "--inside-vmax-percentile", str(args.inside_vmax_percentile),
        ]
        if kinematic_cache is not None:
            cmd.extend(["--kinematic-cache", kinematic_cache])
        else:
            cmd.extend(["--shw", shw, "--shw-format", args.shw_format])
        if args.shw_member and kinematic_cache is None:
            cmd.extend(["--shw-member", args.shw_member])
        if args.discard_upgoing:
            cmd.append("--discard-upgoing")
        if args.inside_show_zeros:
            cmd.append("--inside-show-zeros")
        if args.run_event_mc and (not args.skip_event_mc) and args.event_mc_source_mode == "inside":
            cmd.extend(["--event-cache-source-mode", "inside"])

        result = run_command("04_event_cache_fast", cmd, cwd=outdir, log_dir=dirs["logs"], dry_run=args.dry_run)
        stage_results.append(result)

        for p in args.points:
            cache_path = dirs["event_cache"] / f"events_{p}.npz"
            event_cache_by_point[p] = cache_path
            output_rows.extend([
                {"stage": "04_event_cache", "point": p, "kind": "event_cache_npz", "path": str(cache_path)},
                {"stage": "05_plots_filtered", "point": p, "kind": "theta_phi_png", "path": str(dirs["plots"] / "filtered" / f"theta_phi_counts_{p}.png")},
                {"stage": "05_plots_filtered", "point": p, "kind": "theta_phi_csv", "path": str(dirs["plots"] / "filtered" / f"theta_phi_counts_{p}.csv")},
                {"stage": "05_plots_filtered", "point": p, "kind": "theta_phi_dNdOmega_png", "path": str(dirs["plots"] / "filtered" / f"theta_phi_dNdOmega_{p}.png")},
                {"stage": "05_plots_filtered", "point": p, "kind": "theta_phi_dNdOmega_csv", "path": str(dirs["plots"] / "filtered" / f"theta_phi_dNdOmega_{p}.csv")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "counts_inside_csv", "path": str(dirs["inside_volcano"] / "filtered" / p / f"counts_inside_volcano_{p}.csv")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "dNdOmega_inside_csv", "path": str(dirs["inside_volcano"] / "filtered" / p / f"dNdOmega_inside_volcano_{p}.csv")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "summary", "path": str(dirs["inside_volcano"] / "filtered" / p / f"inside_volcano_summary_{p}.csv")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "individual_linear_png", "path": str(dirs["inside_volcano"] / "figures" / "filtered" / f"inside_volcano_filtered_{p}_linear.png")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "individual_log_png", "path": str(dirs["inside_volcano"] / "figures" / "filtered" / f"inside_volcano_filtered_{p}_log.png")},
            ])
        output_rows.append({"stage": "04_event_cache", "point": "ALL", "kind": "summary", "path": str(dirs["event_cache"] / "event_cache_summary.csv")})
        output_rows.append({"stage": "06_inside_volcano", "point": "ALL", "kind": "merged_manifest", "path": str(dirs["inside_volcano"] / "inside_volcano_merged_manifest.csv")})

        if not args.dry_run:
            required = []
            for p in args.points:
                required.extend([
                    dirs["event_cache"] / f"events_{p}.npz",
                    dirs["plots"] / "filtered" / f"theta_phi_counts_{p}.csv",
                    dirs["inside_volcano"] / "filtered" / p / f"counts_inside_volcano_{p}.csv",
                ])
            require_files(required, "04_event_cache_fast")

    if reuse_fast_cache:
        print("[PHASE 4/9] Reuso productos fast-cache existentes (--skip-filter).")
        for p in args.points:
            cache_path = dirs["event_cache"] / f"events_{p}.npz"
            event_cache_by_point[p] = cache_path
            output_rows.extend([
                {"stage": "04_event_cache", "point": p, "kind": "event_cache_npz", "path": str(cache_path)},
                {"stage": "05_plots_filtered", "point": p, "kind": "theta_phi_png", "path": str(dirs["plots"] / "filtered" / f"theta_phi_counts_{p}.png")},
                {"stage": "05_plots_filtered", "point": p, "kind": "theta_phi_csv", "path": str(dirs["plots"] / "filtered" / f"theta_phi_counts_{p}.csv")},
                {"stage": "05_plots_filtered", "point": p, "kind": "theta_phi_dNdOmega_png", "path": str(dirs["plots"] / "filtered" / f"theta_phi_dNdOmega_{p}.png")},
                {"stage": "05_plots_filtered", "point": p, "kind": "theta_phi_dNdOmega_csv", "path": str(dirs["plots"] / "filtered" / f"theta_phi_dNdOmega_{p}.csv")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "counts_inside_csv", "path": str(dirs["inside_volcano"] / "filtered" / p / f"counts_inside_volcano_{p}.csv")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "dNdOmega_inside_csv", "path": str(dirs["inside_volcano"] / "filtered" / p / f"dNdOmega_inside_volcano_{p}.csv")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "summary", "path": str(dirs["inside_volcano"] / "filtered" / p / f"inside_volcano_summary_{p}.csv")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "individual_linear_png", "path": str(dirs["inside_volcano"] / "figures" / "filtered" / f"inside_volcano_filtered_{p}_linear.png")},
                {"stage": "06_inside_volcano_filtered", "point": p, "kind": "individual_log_png", "path": str(dirs["inside_volcano"] / "figures" / "filtered" / f"inside_volcano_filtered_{p}_log.png")},
            ])
        output_rows.append({"stage": "04_event_cache", "point": "ALL", "kind": "summary", "path": str(dirs["event_cache"] / "event_cache_summary.csv")})
        output_rows.append({"stage": "06_inside_volcano", "point": "ALL", "kind": "merged_manifest", "path": str(dirs["inside_volcano"] / "inside_volcano_merged_manifest.csv")})
        if not args.dry_run:
            required = []
            for p in args.points:
                required.extend([
                    dirs["event_cache"] / f"events_{p}.npz",
                    dirs["plots"] / "filtered" / f"theta_phi_counts_{p}.csv",
                    dirs["inside_volcano"] / "filtered" / p / f"counts_inside_volcano_{p}.csv",
                ])
            required.append(dirs["event_cache"] / "event_cache_summary.csv")
            require_files(required, "04_event_cache_fast reuse")

    # ------------------------------------------------------------------
    # 04. Filtro de muones por Ecrit
    # ------------------------------------------------------------------
    filtered_by_point: dict[str, Path] = {}
    do_filter = (shw is not None) and (not args.skip_filter) and (not args.fast_cache)
    if do_filter:
        print(f"[PHASE 4/9] Filtrado de muones .shw -> {dirs['filtered']}")

        for p in args.points:
            filtered_by_point[p] = filtered_shw_path(dirs["filtered"], input_shw_stem, p, args.filtered_compression)

        if args.filter_mode == "fast":
            # Nueva interfaz: un solo proceso lee el .shw una vez y escribe todos los puntos.
            cmd = [
                sys.executable,
                scripts["filter"],
                "--points", *args.points,
                "--shw", shw,
                "--shw-format", args.shw_format,
                "--indir", dirs["ecrit"],
                "--outdir", dirs["filtered"],
                "--output-compression", args.filtered_compression,
                "--tol-phi", str(args.tol_phi),
                "--tol-theta", str(args.tol_theta),
                "--treat-out-of-grid-as-clear", str(args.treat_out_of_grid_as_clear),
            ]
            if args.shw_member:
                cmd.extend(["--shw-member", args.shw_member])
            if args.discard_upgoing:
                cmd.append("--discard-upgoing")

            result = run_command("04_filter_allpoints_fast", cmd, cwd=outdir, log_dir=dirs["logs"], dry_run=args.dry_run)
            stage_results.append(result)

        else:
            # Interfaz vieja: un proceso por punto.
            jobs = []
            for p in args.points:
                out_shw = filtered_by_point[p]
                cmd = [
                    sys.executable,
                    scripts["filter"],
                    "--point", p,
                    "--shw", shw,
                    "--shw-format", args.shw_format,
                    "--indir", dirs["ecrit"],
                    "--out", out_shw,
                    "--tol_phi", str(args.tol_phi),
                    "--tol_theta", str(args.tol_theta),
                    "--treat_out_of_grid_as_clear", str(args.treat_out_of_grid_as_clear),
                ]
                if args.shw_member:
                    cmd.extend(["--shw-member", args.shw_member])
                if args.discard_upgoing:
                    cmd.append("--discard_upgoing")
                jobs.append((f"04_filter_{p}", cmd))

            stage_results.extend(
                run_command_batch(
                    jobs,
                    cwd=outdir,
                    log_dir=dirs["logs"],
                    dry_run=args.dry_run,
                    parallel_jobs=args.parallel_jobs,
                )
            )

        for p, out_shw in filtered_by_point.items():
            output_rows.append({"stage": "04_filtered", "point": p, "kind": "filtered_shw", "path": str(out_shw)})

        if not args.dry_run:
            require_files(filtered_by_point.values(), "04_filtered")

    elif args.fast_cache and fast_cache_products_available:
        print("[PHASE 4/9] Filtrado .shw omitido: --fast-cache ya generó cache y productos filtered.")
    elif shw is None:
        print("[PHASE 4/9] Sin --shw/cache cinemático: salto filtrado de muones.")
    else:
        print("[PHASE 4/9] Filtrado omitido por bandera.")

    # ------------------------------------------------------------------
    # 05. Mapas θ–φ de conteo
    # ------------------------------------------------------------------
    make_plots = (shw is not None) and (not args.skip_plots) and (args.plot_source != "none") and (not args.fast_cache)
    if make_plots:
        print(f"[PHASE 5/9] Mapas θ–φ -> {dirs['plots']}")
        sources: list[tuple[str, Path | None]] = []
        if args.plot_source in ("raw", "both"):
            sources.append(("raw", shw))
        if args.plot_source in ("filtered", "both"):
            sources.append(("filtered", None))

        plot_jobs: list[tuple[str, Sequence[str]]] = []

        for source_name, source_path in sources:
            plot_dir = dirs["plots"] / source_name
            ensure_dir(plot_dir)
            for p in args.points:
                if source_name == "filtered":
                    source_for_point = filtered_by_point.get(
                        p,
                        filtered_shw_path(dirs["filtered"], input_shw_stem, p, args.filtered_compression),
                    )
                else:
                    source_for_point = source_path

                if source_for_point is None:
                    continue

                cmd = [
                    sys.executable,
                    scripts["plot"],
                    "--point", p,
                    "--shw", source_for_point,
                    "--shw-format", args.shw_format,
                    "--outdir", plot_dir,
                    "--bins-theta", str(args.bins_theta),
                    "--bins-phi", str(args.bins_phi),
                    "--theta-min", str(args.plot_theta_min),
                    "--theta-max", str(args.plot_theta_max),
                    "--phi-min", str(args.plot_phi_min),
                    "--phi-max", str(args.plot_phi_max),
                ]
                if args.shw_member and source_name == "raw":
                    cmd.extend(["--shw-member", args.shw_member])
                plot_jobs.append((f"05_plot_{source_name}_{p}", cmd))

                output_rows.extend([
                    {"stage": f"05_plots_{source_name}", "point": p, "kind": "theta_phi_png", "path": str(plot_dir / f"theta_phi_counts_{p}.png")},
                    {"stage": f"05_plots_{source_name}", "point": p, "kind": "theta_phi_csv", "path": str(plot_dir / f"theta_phi_counts_{p}.csv")},
                    {"stage": f"05_plots_{source_name}", "point": p, "kind": "theta_phi_dNdOmega_png", "path": str(plot_dir / f"theta_phi_dNdOmega_{p}.png")},
                    {"stage": f"05_plots_{source_name}", "point": p, "kind": "theta_phi_dNdOmega_csv", "path": str(plot_dir / f"theta_phi_dNdOmega_{p}.csv")},
                ])

        stage_results.extend(
            run_command_batch(
                plot_jobs,
                cwd=outdir,
                log_dir=dirs["logs"],
                dry_run=args.dry_run,
                parallel_jobs=args.parallel_jobs,
            )
        )
    elif args.fast_cache and fast_cache_products_available:
        print("[PHASE 5/9] Mapas θ–φ filtered ya generados por --fast-cache.")
    elif shw is None:
        print("[PHASE 5/9] Sin --shw/cache cinemático: salto mapas θ–φ.")
    else:
        print("[PHASE 5/9] Mapas θ–φ omitidos por bandera.")


    # ------------------------------------------------------------------
    # 06. Mapas de cuentas sólo dentro de la geometría del volcán
    # ------------------------------------------------------------------
    make_inside = (shw is not None) and (args.inside_volcano_source != "none") and (not args.fast_cache)
    if make_inside:
        print(f"[PHASE 6/9] Cuentas dentro del volcán + figuras finales -> {dirs['inside_volcano']}")

        # 07_inside_volcano_maps_merged.py une:
        #   - conteo raw all-points en una sola lectura del .shw,
        #   - conteo filtered usando un .shw filtrado por punto,
        #   - tablas por punto,
        #   - figuras individuales,
        #   - figuras 2x2,
        #   - bineado visual cuadrado.
        cmd = [
            sys.executable,
            scripts["inside_volcano_merged"],
            "--source", args.inside_volcano_source,
            "--points", *args.points,
            "--raw-shw", shw,
            "--shw-format", args.shw_format,
            "--filtered-dir", dirs["filtered"],
            "--geom-dir", dirs["lengths"],
            "--outdir", dirs["inside_volcano"],
            "--mask-min", str(args.inside_mask_min),
            "--theta-min", str(args.inside_theta_min),
            "--theta-max", str(args.inside_theta_max),
            "--display-theta-min", str(args.inside_display_theta_min),
            "--display-theta-max", str(args.inside_display_theta_max),
            "--display-phi-min", str(args.inside_display_phi_min),
            "--display-phi-max", str(args.inside_display_phi_max),
            "--display-step", str(args.inside_display_step),
            "--vmax-percentile", str(args.inside_vmax_percentile),
            "--filtered-workers", str(args.inside_filtered_workers),
        ]
        if args.shw_member:
            cmd.extend(["--shw-member", args.shw_member])
        if args.inside_mask_col:
            cmd.extend(["--mask-col", args.inside_mask_col])
        if args.inside_show_zeros:
            cmd.append("--show-zeros")

        result = run_command(
            "06_inside_volcano_merged",
            cmd,
            cwd=outdir,
            log_dir=dirs["logs"],
            dry_run=args.dry_run,
        )
        stage_results.append(result)

        # Registrar salidas esperadas.
        for source_name in ("raw", "filtered"):
            if args.inside_volcano_source not in (source_name, "both"):
                continue

            for p in args.points:
                out_inside = dirs["inside_volcano"] / source_name / p
                output_rows.extend([
                    {"stage": f"06_inside_volcano_{source_name}", "point": p, "kind": "counts_inside_csv", "path": str(out_inside / f"counts_inside_volcano_{p}.csv")},
                    {"stage": f"06_inside_volcano_{source_name}", "point": p, "kind": "dNdOmega_inside_csv", "path": str(out_inside / f"dNdOmega_inside_volcano_{p}.csv")},
                    {"stage": f"06_inside_volcano_{source_name}", "point": p, "kind": "summary", "path": str(out_inside / f"inside_volcano_summary_{p}.csv")},
                    {"stage": f"06_inside_volcano_{source_name}", "point": p, "kind": "individual_linear_png", "path": str(dirs["inside_volcano"] / "figures" / source_name / f"inside_volcano_{source_name}_{p}_linear.png")},
                    {"stage": f"06_inside_volcano_{source_name}", "point": p, "kind": "individual_log_png", "path": str(dirs["inside_volcano"] / "figures" / source_name / f"inside_volcano_{source_name}_{p}_log.png")},
                ])

            if set(args.points) == set(DEFAULT_POINTS):
                output_rows.extend([
                    {"stage": f"06_inside_volcano_{source_name}", "point": "ALL", "kind": "fourpanel_linear_png", "path": str(dirs["inside_volcano"] / "figures" / f"inside_volcano_{source_name}_4panel_linear.png")},
                    {"stage": f"06_inside_volcano_{source_name}", "point": "ALL", "kind": "fourpanel_log_png", "path": str(dirs["inside_volcano"] / "figures" / f"inside_volcano_{source_name}_4panel_log.png")},
                    {"stage": f"06_inside_volcano_{source_name}", "point": "ALL", "kind": "fourpanel_summary_csv", "path": str(dirs["inside_volcano"] / "figures" / f"inside_volcano_{source_name}_4panel_summary.csv")},
                ])

        output_rows.append({
            "stage": "06_inside_volcano",
            "point": "ALL",
            "kind": "merged_manifest",
            "path": str(dirs["inside_volcano"] / "inside_volcano_merged_manifest.csv"),
        })

    elif args.fast_cache and fast_cache_products_available:
        print("[PHASE 6/9] Cuentas inside-volcano filtered ya generadas por --fast-cache.")
    elif shw is None:
        print("[PHASE 6/9] Sin --shw/cache cinemático: salto cuentas dentro de la geometría del volcán.")
    else:
        print("[PHASE 6/9] Cuentas dentro de geometría omitidas (--inside-volcano-source none).")


    # ------------------------------------------------------------------
    # 07. Diagnóstico Highland de dispersión angular
    # ------------------------------------------------------------------
    make_scattering = (not args.skip_scattering) and (args.scattering_model in ("highland", "both"))
    if make_scattering:
        print(f"[PHASE 7/9] Scattering Highland -> {dirs['scattering']}")
        cmd = [
            sys.executable,
            scripts["scattering"],
            "--indir", dirs["ecrit"],
            "--outdir", dirs["scattering"],
            "--points", *args.points,
            "--energy-factors", *[str(x) for x in args.scattering_energy_factors],
            "--rho", str(args.rho),
            "--X0", str(args.scattering_X0),
            "--charge", str(args.scattering_charge),
            "--theta-max", str(args.scattering_theta_max),
            "--display-step", str(args.scattering_display_step),
        ]
        if args.scattering_theta_min is not None:
            cmd.extend(["--theta-min", str(args.scattering_theta_min)])
        if args.scattering_phi_min is not None:
            cmd.extend(["--phi-min", str(args.scattering_phi_min)])
        if args.scattering_phi_max is not None:
            cmd.extend(["--phi-max", str(args.scattering_phi_max)])
        if args.scattering_theta_bin_deg is not None:
            cmd.extend(["--theta-bin-deg", str(args.scattering_theta_bin_deg)])
        if args.scattering_phi_bin_deg is not None:
            cmd.extend(["--phi-bin-deg", str(args.scattering_phi_bin_deg)])
        if args.scattering_square_display:
            cmd.append("--square-display")

        result = run_command(
            "07_scattering_highland",
            cmd,
            cwd=outdir,
            log_dir=dirs["logs"],
            dry_run=args.dry_run,
        )
        stage_results.append(result)

        if not args.dry_run:
            required = [dirs["scattering"] / f"scattering_summary.csv"]
            for p in args.points:
                for factor in args.scattering_energy_factors:
                    tag = f"f{factor:.2f}".replace(".", "p").replace("-", "m")
                    required.append(dirs["scattering"] / p / f"scattering_table_{p}_{tag}.csv")
            require_files(required, "07_scattering_highland")

        output_rows.append({"stage": "07_scattering", "point": "ALL", "kind": "summary", "path": str(dirs["scattering"] / "scattering_summary.csv")})
        for p in args.points:
            for factor in args.scattering_energy_factors:
                tag = f"f{factor:.2f}".replace(".", "p").replace("-", "m")
                output_rows.append({"stage": "07_scattering", "point": p, "kind": f"scattering_table_{tag}", "path": str(dirs["scattering"] / p / f"scattering_table_{p}_{tag}.csv")})
            output_rows.extend([
                {"stage": "07_scattering", "point": p, "kind": "theta0_mrad_triptych", "path": str(dirs["scattering"] / p / f"theta0_mrad_triptych_{p}.png")},
                {"stage": "07_scattering", "point": p, "kind": "theta0_deg_triptych", "path": str(dirs["scattering"] / p / f"theta0_deg_triptych_{p}.png")},
                {"stage": "07_scattering", "point": p, "kind": "theta0_over_pixel_min_triptych", "path": str(dirs["scattering"] / p / f"theta0_over_pixel_min_triptych_{p}.png")},
                {"stage": "07_scattering", "point": p, "kind": "lateral_rms_proj_m_triptych", "path": str(dirs["scattering"] / p / f"lateral_rms_proj_m_triptych_{p}.png")},
            ])
    else:
        print("[PHASE 7/9] Scattering omitido por bandera.")

    # ------------------------------------------------------------------
    # 07b. Diagnóstico empírico Geant4 de dispersión angular
    # ------------------------------------------------------------------
    make_scattering_empirical = (not args.skip_scattering) and (args.scattering_model in ("empirical", "both"))
    empirical_factors = args.empirical_energy_factors or args.scattering_energy_factors
    empirical_theta_min = args.empirical_theta_min if args.empirical_theta_min is not None else args.smearing_theta_min
    empirical_theta_max = args.empirical_theta_max if args.empirical_theta_max is not None else args.smearing_theta_max
    empirical_phi_min = args.empirical_phi_min if args.empirical_phi_min is not None else args.smearing_phi_min
    empirical_phi_max = args.empirical_phi_max if args.empirical_phi_max is not None else args.smearing_phi_max
    empirical_display_step = args.empirical_display_step if args.empirical_display_step is not None else args.smearing_display_step

    if make_scattering_empirical:
        print(f"[PHASE 7/9 EMPIRICAL] Scattering empírico Geant4 -> {dirs['scattering_empirical']}")
        cmd = [
            sys.executable,
            scripts["scattering_empirical"],
            "--indir", dirs["ecrit"],
            "--outdir", dirs["scattering_empirical"],
            "--points", *args.points,
            "--kernel-library", empirical_kernel_library,
            "--energy-factors", *[str(x) for x in empirical_factors],
            "--interp-method", args.empirical_interp_method,
            "--theta-max", str(empirical_theta_max),
            "--display-step", str(empirical_display_step),
            "--square-display",
        ]
        if empirical_theta_min is not None:
            cmd.extend(["--theta-min", str(empirical_theta_min)])
        if empirical_phi_min is not None:
            cmd.extend(["--phi-min", str(empirical_phi_min)])
        if empirical_phi_max is not None:
            cmd.extend(["--phi-max", str(empirical_phi_max)])
        if args.scattering_theta_bin_deg is not None:
            cmd.extend(["--theta-bin-deg", str(args.scattering_theta_bin_deg)])
        if args.scattering_phi_bin_deg is not None:
            cmd.extend(["--phi-bin-deg", str(args.scattering_phi_bin_deg)])

        result = run_command(
            "07_scattering_empirical",
            cmd,
            cwd=outdir,
            log_dir=dirs["logs"],
            dry_run=args.dry_run,
        )
        stage_results.append(result)

        if not args.dry_run:
            required = [dirs["scattering_empirical"] / "scattering_empirical_summary.csv"]
            for p in args.points:
                for factor in empirical_factors:
                    tag = f"f{factor:.2f}".replace(".", "p").replace("-", "m")
                    required.append(dirs["scattering_empirical"] / p / f"scattering_empirical_table_{p}_{tag}.csv")
            require_files(required, "07_scattering_empirical")

        output_rows.append({"stage": "07_scattering_empirical", "point": "ALL", "kind": "summary", "path": str(dirs["scattering_empirical"] / "scattering_empirical_summary.csv")})
        for p in args.points:
            for factor in empirical_factors:
                tag = f"f{factor:.2f}".replace(".", "p").replace("-", "m")
                output_rows.append({"stage": "07_scattering_empirical", "point": p, "kind": f"scattering_empirical_table_{tag}", "path": str(dirs["scattering_empirical"] / p / f"scattering_empirical_table_{p}_{tag}.csv")})
            output_rows.extend([
                {"stage": "07_scattering_empirical", "point": p, "kind": "RMS_empirical_mrad_triptych", "path": str(dirs["scattering_empirical"] / p / f"RMS_empirical_mrad_triptych_{p}.png")},
                {"stage": "07_scattering_empirical", "point": p, "kind": "Tail10_empirical_triptych", "path": str(dirs["scattering_empirical"] / p / f"Tail10_empirical_triptych_{p}.png")},
                {"stage": "07_scattering_empirical", "point": p, "kind": "RMS_empirical_over_pixel_min_triptych", "path": str(dirs["scattering_empirical"] / p / f"RMS_empirical_over_pixel_min_triptych_{p}.png")},
            ])
    elif args.scattering_model in ("empirical", "both"):
        print("[PHASE 7/9 EMPIRICAL] Scattering empírico omitido por bandera.")

    # ------------------------------------------------------------------
    # 08. Smearing angular sobre mapas θ–φ
    # ------------------------------------------------------------------
    make_smearing = (
        (not args.skip_smearing)
        and ((shw is not None) or fast_cache_products_available)
        and (args.plot_source != "none")
        and (args.inside_volcano_source != "none")
        and (args.scattering_model in ("highland", "both"))
    )
    if make_smearing:
        if args.smearing_source == "both":
            smearing_sources = ["raw", "filtered"]
        else:
            smearing_sources = [args.smearing_source]

        # Validación: sólo se puede hacer smearing sobre fuentes que fueron generadas antes.
        missing_sources = []
        for src in smearing_sources:
            if args.plot_source not in (src, "both"):
                missing_sources.append(f"05_plots/{src}")
            if args.inside_volcano_source not in (src, "both"):
                missing_sources.append(f"06_inside_volcano/{src}")
        if missing_sources:
            raise RuntimeError(
                "Pediste --smearing-source pero no se generaron todos sus insumos: "
                + ", ".join(missing_sources)
                + ". Usa --plot-source both y --inside-volcano-source both, o cambia --smearing-source."
            )

        smear_factors = args.smearing_energy_factors or args.scattering_energy_factors
        scat_template = str(dirs["scattering"] / "{point}" / "scattering_table_{point}_{tag}.csv")

        for smearing_source in smearing_sources:
            # Para una sola fuente mantenemos la salida histórica en 08_smearing/.
            # Para both se separa en 08_smearing/raw/ y 08_smearing/filtered/ para no sobrescribir.
            smearing_outdir = dirs["smearing"] if len(smearing_sources) == 1 else dirs["smearing"] / smearing_source
            ensure_dir(smearing_outdir)

            print(f"[PHASE 8/9 HIGHLAND] Smearing angular ({smearing_source}) -> {smearing_outdir}")

            map_template = str(dirs["plots"] / smearing_source / "theta_phi_counts_{point}.csv")
            inside_template = str(dirs["inside_volcano"] / smearing_source / "{point}" / "counts_inside_volcano_{point}.csv")

            if not args.dry_run:
                required = []
                for p in args.points:
                    required.append(Path(map_template.format(point=p)))
                    required.append(Path(inside_template.format(point=p)))
                    for factor in smear_factors:
                        tag = f"f{factor:.2f}".replace(".", "p").replace("-", "m")
                        required.append(Path(scat_template.format(point=p, factor=factor, tag=tag)))
                require_files(required, f"08_smearing inputs ({smearing_source})")

            cmd = [
                sys.executable,
                scripts["smearing"],
                "--points", *args.points,
                "--energy-factors", *[str(x) for x in smear_factors],
                "--map-template", map_template,
                "--inside-map-template", inside_template,
                "--scat-template", scat_template,
                "--outdir", smearing_outdir,
                "--sigma-col", args.smearing_sigma_col,
                "--theta-max", str(args.smearing_theta_max),
                "--phi-min", str(args.smearing_phi_min),
                "--phi-max", str(args.smearing_phi_max),
                "--display-step", str(args.smearing_display_step),
                "--vmax-percentile", str(args.smearing_vmax_percentile),
                "--relative-vmax-percentile", str(args.smearing_relative_vmax_percentile),
                "--kernel-radius-sigma", str(args.smearing_kernel_radius_sigma),
                "--detector-sigma-deg", str(args.smearing_detector_sigma_deg),
                "--sigma-scale", str(args.smearing_sigma_scale),
                "--random-seed", str(args.smearing_random_seed),
            ]
            if args.smearing_theta_min is not None:
                cmd.extend(["--theta-min", str(args.smearing_theta_min)])
            if args.smearing_stochastic:
                cmd.append("--stochastic")

            result = run_command(
                f"08_angular_smearing_{smearing_source}",
                cmd,
                cwd=outdir,
                log_dir=dirs["logs"],
                dry_run=args.dry_run,
            )
            stage_results.append(result)

            if not args.dry_run:
                require_file(smearing_outdir / "smearing_summary.csv", f"summary de smearing ({smearing_source})")

            output_rows.append({"stage": f"08_smearing_{smearing_source}", "point": "ALL", "kind": "summary", "path": str(smearing_outdir / "smearing_summary.csv")})
            for p in args.points:
                for factor in smear_factors:
                    tag = f"f{factor:.2f}".replace(".", "p").replace("-", "m")
                    output_rows.extend([
                        {"stage": f"08_smearing_{smearing_source}", "point": p, "kind": f"smearing_table_{tag}", "path": str(smearing_outdir / p / f"smearing_table_{p}_{tag}.csv")},
                        {"stage": f"08_smearing_{smearing_source}", "point": p, "kind": f"smearing_comparison_{tag}", "path": str(smearing_outdir / p / f"smearing_comparison_{p}_{tag}.png")},
                        {"stage": f"08_smearing_{smearing_source}", "point": p, "kind": f"inside_volcano_smearing_table_{tag}", "path": str(smearing_outdir / p / f"inside_volcano_smearing_table_{p}_{tag}.csv")},
                        {"stage": f"08_smearing_{smearing_source}", "point": p, "kind": f"inside_volcano_smearing_comparison_{tag}", "path": str(smearing_outdir / p / f"inside_volcano_smearing_comparison_{p}_{tag}.png")},
                    ])
    elif shw is None and not fast_cache_products_available:
        print("[PHASE 8/9 HIGHLAND] Sin --shw/cache cinemático: salto smearing angular Highland.")
    elif args.scattering_model in ("highland", "both"):
        print("[PHASE 8/9 HIGHLAND] Smearing Highland omitido por bandera o faltan mapas fuente.")

    # ------------------------------------------------------------------
    # 08b. Smearing angular empírico sobre mapas θ–φ
    # ------------------------------------------------------------------
    make_smearing_empirical = (
        (not args.skip_smearing)
        and ((shw is not None) or fast_cache_products_available)
        and (args.plot_source != "none")
        and (args.inside_volcano_source != "none")
        and (args.scattering_model in ("empirical", "both"))
    )
    if make_smearing_empirical:
        if args.smearing_source == "both":
            empirical_smearing_sources = ["raw", "filtered"]
        else:
            empirical_smearing_sources = [args.smearing_source]

        missing_sources = []
        for src in empirical_smearing_sources:
            if args.plot_source not in (src, "both"):
                missing_sources.append(f"05_plots/{src}")
            if args.inside_volcano_source not in (src, "both"):
                missing_sources.append(f"06_inside_volcano/{src}")
        if missing_sources:
            raise RuntimeError(
                "Pediste smearing empírico pero no se generaron todos sus insumos: "
                + ", ".join(missing_sources)
                + ". Usa --plot-source both y --inside-volcano-source both, o cambia --smearing-source."
            )

        for smearing_source in empirical_smearing_sources:
            smearing_outdir = dirs["smearing_empirical"] if len(empirical_smearing_sources) == 1 else dirs["smearing_empirical"] / smearing_source
            ensure_dir(smearing_outdir)

            print(f"[PHASE 8/9 EMPIRICAL] Smearing angular empírico ({smearing_source}) -> {smearing_outdir}")

            map_template = str(dirs["plots"] / smearing_source / "theta_phi_counts_{point}.csv")
            inside_template = str(dirs["inside_volcano"] / smearing_source / "{point}" / "counts_inside_volcano_{point}.csv")
            ecrit_template = str(dirs["ecrit"] / "ecrit_table_{point}.csv")

            if not args.dry_run:
                required = []
                for p in args.points:
                    required.append(Path(map_template.format(point=p)))
                    required.append(Path(inside_template.format(point=p)))
                    required.append(Path(ecrit_template.format(point=p)))
                require_files(required, f"08_smearing_empirical inputs ({smearing_source})")

            cmd = [
                sys.executable,
                scripts["smearing_empirical"],
                "--points", *args.points,
                "--energy-factors", *[str(x) for x in empirical_factors],
                "--map-template", map_template,
                "--inside-map-template", inside_template,
                "--ecrit-template", ecrit_template,
                "--kernel-library", empirical_kernel_library,
                "--outdir", smearing_outdir,
                "--interp-method", args.empirical_interp_method,
                "--theta-max", str(empirical_theta_max),
                "--phi-min", str(empirical_phi_min),
                "--phi-max", str(empirical_phi_max),
                "--display-step", str(empirical_display_step),
                "--vmax-percentile", str(args.smearing_vmax_percentile),
                "--relative-vmax-percentile", str(args.smearing_relative_vmax_percentile),
                "--random-seed", str(args.empirical_random_seed),
            ]
            if empirical_theta_min is not None:
                cmd.extend(["--theta-min", str(empirical_theta_min)])
            if args.empirical_stochastic:
                cmd.append("--stochastic")
            if args.empirical_kernel_threshold is not None:
                cmd.extend(["--kernel-threshold", str(args.empirical_kernel_threshold)])
            if args.empirical_max_kernel_radius_mrad is not None:
                cmd.extend(["--max-kernel-radius-mrad", str(args.empirical_max_kernel_radius_mrad)])

            result = run_command(
                f"08_angular_smearing_empirical_{smearing_source}",
                cmd,
                cwd=outdir,
                log_dir=dirs["logs"],
                dry_run=args.dry_run,
            )
            stage_results.append(result)

            if not args.dry_run:
                require_file(smearing_outdir / "smearing_empirical_summary.csv", f"summary de smearing empírico ({smearing_source})")

            output_rows.append({"stage": f"08_smearing_empirical_{smearing_source}", "point": "ALL", "kind": "summary", "path": str(smearing_outdir / "smearing_empirical_summary.csv")})
            for p in args.points:
                for factor in empirical_factors:
                    tag = f"f{factor:.2f}".replace(".", "p").replace("-", "m")
                    output_rows.extend([
                        {"stage": f"08_smearing_empirical_{smearing_source}", "point": p, "kind": f"smearing_empirical_table_{tag}", "path": str(smearing_outdir / p / f"smearing_empirical_table_{p}_{tag}.csv")},
                        {"stage": f"08_smearing_empirical_{smearing_source}", "point": p, "kind": f"smearing_empirical_comparison_{tag}", "path": str(smearing_outdir / p / f"smearing_empirical_comparison_{p}_{tag}.png")},
                        {"stage": f"08_smearing_empirical_{smearing_source}", "point": p, "kind": f"inside_volcano_smearing_empirical_table_{tag}", "path": str(smearing_outdir / p / f"inside_volcano_smearing_empirical_table_{p}_{tag}.csv")},
                        {"stage": f"08_smearing_empirical_{smearing_source}", "point": p, "kind": f"inside_volcano_smearing_empirical_comparison_{tag}", "path": str(smearing_outdir / p / f"inside_volcano_smearing_empirical_comparison_{p}_{tag}.png")},
                    ])
    elif shw is None and not fast_cache_products_available:
        print("[PHASE 8/9 EMPIRICAL] Sin --shw/cache cinemático: salto smearing angular empírico.")
    elif args.scattering_model in ("empirical", "both"):
        print("[PHASE 8/9 EMPIRICAL] Smearing empírico omitido por bandera o faltan mapas fuente.")

    # ------------------------------------------------------------------
    # 08c. MC empírico evento-por-evento
    # ------------------------------------------------------------------
    make_event_mc = (
        args.run_event_mc
        and (not args.skip_event_mc)
        and ((shw is not None) or fast_cache_products_available)
    )
    if make_event_mc:
        if empirical_kernel_library is None:
            raise FileNotFoundError("--empirical-kernel-library es requerido con --run-event-mc")

        ecrit_template = str(dirs["ecrit"] / "ecrit_table_{point}.csv")
        event_workers = args.event_mc_workers if args.event_mc_workers is not None else args.parallel_jobs
        event_workers = max(1, int(event_workers))
        event_theta_min = args.event_mc_theta_min if args.event_mc_theta_min is not None else args.plot_theta_min
        event_theta_max = args.event_mc_theta_max if args.event_mc_theta_max is not None else args.plot_theta_max
        event_phi_min = args.event_mc_phi_min if args.event_mc_phi_min is not None else args.plot_phi_min
        event_phi_max = args.event_mc_phi_max if args.event_mc_phi_max is not None else args.plot_phi_max
        event_display_step = args.event_mc_display_step
        event_kernel_threshold = (
            args.event_mc_kernel_threshold
            if args.event_mc_kernel_threshold is not None
            else args.empirical_kernel_threshold
        )
        event_sources = ["raw", "filtered"] if args.event_mc_source == "both" else [args.event_mc_source]

        for event_source in event_sources:
            if event_source == "filtered":
                if args.fast_cache:
                    shw_template = None
                    event_cache_template = str(dirs["event_cache"] / "events_{point}.npz")
                else:
                    shw_template = str(
                        filtered_shw_path(dirs["filtered"], input_shw_stem, "{point}", args.filtered_compression)
                    )
                    event_cache_template = None
                event_outdir = dirs["event_mc_empirical"] / "filtered"
            else:
                if shw is None:
                    raise RuntimeError(
                        "Event-MC raw requiere --shw. Con --kinematic-cache/--fast-cache usa "
                        "--event-mc-source filtered."
                    )
                shw_template = None
                event_cache_template = None
                event_outdir = dirs["event_mc_empirical"] / "raw"

            ensure_dir(event_outdir)

            if not args.dry_run:
                required = [Path(ecrit_template.format(point=p)) for p in args.points]
                if event_source == "filtered" and args.fast_cache:
                    required.extend(Path(event_cache_template.format(point=p)) for p in args.points)
                elif event_source == "filtered":
                    required.extend(Path(shw_template.format(point=p)) for p in args.points)
                require_files(required, f"08c_event_mc_empirical inputs ({event_source})")

            print(f"[PHASE 8/9 EVENT-MC] Event-by-event empirical MC ({event_source}, source-mode={args.event_mc_source_mode}) -> {event_outdir}")
            cmd = [
                sys.executable,
                scripts["event_mc_empirical"],
                "--points", *args.points,
                "--ecrit-template", ecrit_template,
                "--kernel-library", empirical_kernel_library,
                "--outdir", event_outdir,
                "--shw-format", args.shw_format,
                "--workers", str(event_workers),
                "--interp-method", args.empirical_interp_method,
                "--energy-cache-dlog", str(args.event_mc_energy_cache_dlog),
                "--source-mode", args.event_mc_source_mode,
                "--theta-min", str(event_theta_min),
                "--theta-max", str(event_theta_max),
                "--phi-min", str(event_phi_min),
                "--phi-max", str(event_phi_max),
                "--random-seed", str(args.event_mc_random_seed),
            ]
            if event_kernel_threshold is not None:
                cmd.extend(["--kernel-threshold", str(event_kernel_threshold)])
            if args.event_mc_max_kernel_radius_mrad is not None:
                cmd.extend(["--max-kernel-radius-mrad", str(args.event_mc_max_kernel_radius_mrad)])
            if event_display_step is not None:
                cmd.extend(["--display-step", str(event_display_step)])
            if args.event_mc_head:
                cmd.extend(["--head", str(args.event_mc_head)])
            if event_source == "filtered" and args.fast_cache:
                cmd.extend(["--event-cache-template", event_cache_template])
            elif event_source == "filtered":
                cmd.extend(["--shw-template", shw_template])
            else:
                cmd.extend(["--shw", shw])
            if args.shw_member:
                cmd.extend(["--shw-member", args.shw_member])

            result = run_command(
                f"08c_event_mc_empirical_{event_source}_{args.event_mc_source_mode}",
                cmd,
                cwd=outdir,
                log_dir=dirs["logs"],
                dry_run=args.dry_run,
            )
            stage_results.append(result)

            if not args.dry_run:
                require_file(event_outdir / "event_mc_smearing_summary.csv", f"summary event-MC empírico ({event_source})")

            stage = f"08c_event_mc_empirical_{event_source}_{args.event_mc_source_mode}"
            output_rows.append({"stage": stage, "point": "ALL", "kind": "summary", "path": str(event_outdir / "event_mc_smearing_summary.csv")})
            prefix = "event_mc_inside_source" if args.event_mc_source_mode == "inside" else "event_mc"
            for p in args.points:
                point_dir = event_outdir / p
                output_rows.extend([
                    {"stage": stage, "point": p, "kind": f"{prefix}_smearing_table", "path": str(point_dir / f"{prefix}_smearing_table_{p}.csv")},
                    {"stage": stage, "point": p, "kind": f"{prefix}_smearing_comparison", "path": str(point_dir / f"{prefix}_smearing_comparison_{p}.png")},
                    {"stage": stage, "point": p, "kind": f"{prefix}_retained_inside_table", "path": str(point_dir / f"{prefix}_retained_inside_table_{p}.csv")},
                    {"stage": stage, "point": p, "kind": f"{prefix}_retained_inside_comparison", "path": str(point_dir / f"{prefix}_retained_inside_comparison_{p}.png")},
                    {"stage": stage, "point": p, "kind": "event_mc_summary", "path": str(point_dir / f"event_mc_summary_{p}.csv")},
                ])
                if event_display_step is not None and event_display_step > 0.5:
                    tag = f"bin{event_display_step:.2f}deg".replace(".", "p").replace("-", "m")
                    output_rows.extend([
                        {"stage": stage, "point": p, "kind": f"{prefix}_smearing_binned_{tag}_table", "path": str(point_dir / f"{prefix}_smearing_binned_{tag}_table_{p}.csv")},
                        {"stage": stage, "point": p, "kind": f"{prefix}_smearing_binned_{tag}_comparison", "path": str(point_dir / f"{prefix}_smearing_binned_{tag}_comparison_{p}.png")},
                        {"stage": stage, "point": p, "kind": f"{prefix}_retained_inside_binned_{tag}_table", "path": str(point_dir / f"{prefix}_retained_inside_binned_{tag}_table_{p}.csv")},
                        {"stage": stage, "point": p, "kind": f"{prefix}_retained_inside_binned_{tag}_comparison", "path": str(point_dir / f"{prefix}_retained_inside_binned_{tag}_comparison_{p}.png")},
                    ])
    elif args.run_event_mc and args.skip_event_mc:
        print("[PHASE 8/9 EVENT-MC] Event-by-event empirical MC omitido por --skip-event-mc.")

    # ------------------------------------------------------------------
    # 08d. Contaminación angular por in-scattering externo -> acceptance
    # ------------------------------------------------------------------
    make_in_scattering = (
        args.run_in_scattering
        and (not args.skip_in_scattering)
        and ((shw is not None) or (kinematic_cache is not None))
    )
    if make_in_scattering:
        if empirical_kernel_library is None:
            raise FileNotFoundError("--empirical-kernel-library es requerido con --run-in-scattering")

        print(f"[PHASE 8/9 IN-SCATTERING] In-scattering angular-only externo -> acceptance -> {dirs['in_scattering']}")
        in_scattering_kernel_threshold = (
            args.in_scattering_kernel_threshold
            if args.in_scattering_kernel_threshold is not None
            else args.empirical_kernel_threshold
        )

        jobs = []
        for k, p in enumerate(args.points):
            point_outdir = dirs["in_scattering"] / p
            ensure_dir(point_outdir)
            ecrit_csv = dirs["ecrit"] / f"ecrit_table_{p}.csv"
            if not args.dry_run:
                require_file(ecrit_csv, f"ecrit/length map para in-scattering ({p})")

            cmd = [
                sys.executable,
                scripts["in_scattering"],
                "--kernel-npz", empirical_kernel_library,
                "--acceptance-map", ecrit_csv,
                "--length-map", ecrit_csv,
                "--output-dir", point_outdir,
                "--point", p,
                "--step-m", str(args.in_scattering_step_m),
                "--n-samples-per-muon", str(args.in_scattering_samples_per_muon),
                "--seed", str(args.in_scattering_seed + 1009 * k),
                "--workers", str(args.in_scattering_workers),
                "--rho", str(args.rho),
                "--external-length-mode", args.in_scattering_external_length_mode,
                "--hgt-dir", hgt_dir,
                "--external-s-max-m", str(args.in_scattering_external_s_max_m),
                "--external-ray-step-m", str(args.in_scattering_external_ray_step_m),
                "--length-cache-step-deg", str(args.in_scattering_length_cache_step_deg),
                "--interp-method", args.empirical_interp_method,
                "--kernel-threshold", str(in_scattering_kernel_threshold),
                "--kernel-scale", str(args.in_scattering_kernel_scale),
            ]
            if args.in_scattering_max_angular_margin_deg is not None:
                cmd.extend(["--max-angular-margin-deg", str(args.in_scattering_max_angular_margin_deg)])
            if args.in_scattering_theta_min_deg is not None:
                cmd.extend(["--theta-min-deg", str(args.in_scattering_theta_min_deg)])
            if args.in_scattering_theta_max_deg is not None:
                cmd.extend(["--theta-max-deg", str(args.in_scattering_theta_max_deg)])
            if args.in_scattering_phi_min_deg is not None:
                cmd.extend(["--phi-min-deg", str(args.in_scattering_phi_min_deg)])
            if args.in_scattering_phi_max_deg is not None:
                cmd.extend(["--phi-max-deg", str(args.in_scattering_phi_max_deg)])
            if shw is not None:
                cmd.extend(["--input-shw", shw, "--shw-format", args.shw_format])
                if args.shw_member:
                    cmd.extend(["--shw-member", args.shw_member])
            else:
                cmd.extend(["--kinematic-cache", kinematic_cache])
            if range_file is not None:
                cmd.extend(["--range-file", range_file])
            if args.discard_upgoing:
                cmd.append("--discard-upgoing")
            if args.in_scattering_disable_scattering:
                cmd.append("--disable-scattering")
            if args.in_scattering_debug_trajectories:
                cmd.append("--debug-trajectories")
            if args.in_scattering_no_figures:
                cmd.append("--no-figures")
            if args.in_scattering_head:
                cmd.extend(["--head", str(args.in_scattering_head)])

            jobs.append((f"08d_in_scattering_{p}", cmd))

            output_rows.extend([
                {"stage": "08d_in_scattering", "point": p, "kind": "masked_counts_npy", "path": str(point_outdir / "masked_counts_theta_phi.npy")},
                {"stage": "08d_in_scattering", "point": p, "kind": "masked_counts_csv", "path": str(point_outdir / "masked_counts_theta_phi.csv")},
                {"stage": "08d_in_scattering", "point": p, "kind": "source_counts_npy", "path": str(point_outdir / "source_counts_theta_phi.npy")},
                {"stage": "08d_in_scattering", "point": p, "kind": "source_counts_csv", "path": str(point_outdir / "source_counts_theta_phi.csv")},
                {"stage": "08d_in_scattering", "point": p, "kind": "summary_json", "path": str(point_outdir / "in_scattering_summary.json")},
            ])
            if not args.in_scattering_no_figures:
                output_rows.extend([
                    {"stage": "08d_in_scattering", "point": p, "kind": "extended_region_png", "path": str(point_outdir / "extended_angular_region.png")},
                    {"stage": "08d_in_scattering", "point": p, "kind": "accepted_map_png", "path": str(point_outdir / "in_scattering_accepted_map.png")},
                    {"stage": "08d_in_scattering", "point": p, "kind": "source_map_png", "path": str(point_outdir / "in_scattering_source_map.png")},
                    {"stage": "08d_in_scattering", "point": p, "kind": "final_deflection_hist_png", "path": str(point_outdir / "final_deflection_hist.png")},
                    {"stage": "08d_in_scattering", "point": p, "kind": "initial_energy_hist_png", "path": str(point_outdir / "initial_energy_accepted_hist.png")},
                    {"stage": "08d_in_scattering", "point": p, "kind": "length_hist_png", "path": str(point_outdir / "rock_length_accepted_hist.png")},
                ])
            if args.in_scattering_debug_trajectories:
                output_rows.append(
                    {"stage": "08d_in_scattering", "point": p, "kind": "debug_trajectories_csv", "path": str(point_outdir / "debug_accepted_trajectories.csv")}
                )

        stage_results.extend(
            run_command_batch(
                jobs,
                cwd=outdir,
                log_dir=dirs["logs"],
                dry_run=args.dry_run,
                parallel_jobs=args.parallel_jobs,
            )
        )
    elif args.run_in_scattering and args.skip_in_scattering:
        print("[PHASE 8/9 IN-SCATTERING] In-scattering omitido por --skip-in-scattering.")
    elif args.run_in_scattering:
        print("[PHASE 8/9 IN-SCATTERING] In-scattering omitido: falta --shw o --kinematic-cache.")

    # ------------------------------------------------------------------
    # 08e. In-scattering espacial DEM externo -> máscara angular
    # ------------------------------------------------------------------
    make_spatial_in_scattering = (
        args.run_spatial_in_scattering
        and (not args.skip_spatial_in_scattering)
        and (kinematic_cache is not None)
    )
    if make_spatial_in_scattering:
        if empirical_kernel_library is None:
            raise FileNotFoundError("--empirical-kernel-library es requerido con --run-spatial-in-scattering")

        print(f"[PHASE 8/9 SPATIAL] In-scattering espacial DEM externo -> máscara angular -> {dirs['in_scattering']}")
        spatial_kernel_threshold = (
            args.in_scattering_kernel_threshold
            if args.in_scattering_kernel_threshold is not None
            else args.empirical_kernel_threshold
        )

        jobs = []
        for k, p in enumerate(args.points):
            point_outdir = dirs["in_scattering"] / f"{p}_spatial_dem"
            ensure_dir(point_outdir)
            ecrit_csv = dirs["ecrit"] / f"ecrit_table_{p}.csv"
            if not args.dry_run:
                require_file(ecrit_csv, f"ecrit/length map para in-scattering espacial ({p})")

            cmd = [
                sys.executable,
                scripts["spatial_in_scattering"],
                "--kinematic-cache", kinematic_cache,
                "--kernel-npz", empirical_kernel_library,
                "--acceptance-map", ecrit_csv,
                "--length-map", ecrit_csv,
                "--output-dir", point_outdir,
                "--point", p,
                "--hgt-dir", hgt_dir,
                "--ray-step-m", str(args.spatial_in_scattering_ray_step_m),
                "--position-samples-per-muon", str(args.spatial_in_scattering_samples_per_muon),
                "--sample-probability", str(args.spatial_in_scattering_sample_probability),
                "--source-surface", args.spatial_in_scattering_source_surface,
                "--volcano-surface-grid-step-m", str(args.spatial_in_scattering_volcano_surface_grid_step_m),
                "--volcano-surface-edge-guard-m", str(args.spatial_in_scattering_volcano_surface_edge_guard_m),
                "--volcano-surface-min-height-frac", str(args.spatial_in_scattering_volcano_surface_min_height_frac),
                "--volcano-surface-entry-check-m", str(args.spatial_in_scattering_volcano_surface_entry_check_m),
                "--observer-radius-m", str(args.spatial_in_scattering_observer_radius_m),
                "--seed", str(args.spatial_in_scattering_seed + 1009 * k),
                "--rho", str(args.rho),
                "--interp-method", args.empirical_interp_method,
                "--kernel-threshold", str(spatial_kernel_threshold),
                "--kernel-scale", str(args.spatial_in_scattering_kernel_scale),
            ]
            if args.spatial_in_scattering_max_angular_margin_deg is not None:
                cmd.extend(["--max-angular-margin-deg", str(args.spatial_in_scattering_max_angular_margin_deg)])
            if args.spatial_in_scattering_entry_face_importance:
                cmd.extend(["--entry-face-importance", args.spatial_in_scattering_entry_face_importance])
            if range_file is not None:
                cmd.extend(["--range-file", range_file])
            if args.spatial_in_scattering_min_survival_rock_m is not None:
                cmd.extend(["--min-survival-rock-m", str(args.spatial_in_scattering_min_survival_rock_m)])
            if args.discard_upgoing:
                cmd.append("--discard-upgoing")
            if args.spatial_in_scattering_disable_scattering:
                cmd.append("--disable-scattering")
            if args.spatial_in_scattering_no_figures:
                cmd.append("--no-figures")
            if args.spatial_in_scattering_head:
                cmd.extend(["--head", str(args.spatial_in_scattering_head)])

            jobs.append((f"08e_spatial_in_scattering_{p}", cmd))
            output_rows.extend([
                {"stage": "08e_spatial_in_scattering", "point": p, "kind": "final_counts_npy", "path": str(point_outdir / "spatial_final_counts_theta_phi.npy")},
                {"stage": "08e_spatial_in_scattering", "point": p, "kind": "final_counts_csv", "path": str(point_outdir / "spatial_final_counts_theta_phi.csv")},
                {"stage": "08e_spatial_in_scattering", "point": p, "kind": "source_counts_npy", "path": str(point_outdir / "spatial_source_counts_theta_phi.npy")},
                {"stage": "08e_spatial_in_scattering", "point": p, "kind": "source_counts_csv", "path": str(point_outdir / "spatial_source_counts_theta_phi.csv")},
                {"stage": "08e_spatial_in_scattering", "point": p, "kind": "accepted_tracks_csv", "path": str(point_outdir / "spatial_accepted_tracks.csv")},
                {"stage": "08e_spatial_in_scattering", "point": p, "kind": "volcano_surface_target_points_csv", "path": str(point_outdir / "volcano_surface_target_points.csv")},
                {"stage": "08e_spatial_in_scattering", "point": p, "kind": "summary_json", "path": str(point_outdir / "spatial_in_scattering_summary.json")},
            ])
            if not args.spatial_in_scattering_no_figures:
                output_rows.extend([
                    {"stage": "08e_spatial_in_scattering", "point": p, "kind": "final_map_png", "path": str(point_outdir / "spatial_final_accepted_map.png")},
                    {"stage": "08e_spatial_in_scattering", "point": p, "kind": "source_map_png", "path": str(point_outdir / "spatial_source_external_map.png")},
                    {"stage": "08e_spatial_in_scattering", "point": p, "kind": "accepted_arrows_theta_phi_png", "path": str(point_outdir / "spatial_accepted_muon_arrows_theta_phi.png")},
                    {"stage": "08e_spatial_in_scattering", "point": p, "kind": "accepted_tracks_xy_png", "path": str(point_outdir / "spatial_accepted_muon_tracks_xy.png")},
                    {"stage": "08e_spatial_in_scattering", "point": p, "kind": "first_rock_contact_xy_png", "path": str(point_outdir / "spatial_first_rock_contact_xy.png")},
                    {"stage": "08e_spatial_in_scattering", "point": p, "kind": "volcano_surface_target_points_xy_png", "path": str(point_outdir / "volcano_surface_target_points_xy.png")},
                    {"stage": "08e_spatial_in_scattering", "point": p, "kind": "rock_length_hist_png", "path": str(point_outdir / "spatial_accepted_rock_length_hist.png")},
                ])

        stage_results.extend(
            run_command_batch(
                jobs,
                cwd=outdir,
                log_dir=dirs["logs"],
                dry_run=args.dry_run,
                parallel_jobs=args.parallel_jobs,
            )
        )
    elif args.run_spatial_in_scattering and args.skip_spatial_in_scattering:
        print("[PHASE 8/9 SPATIAL] In-scattering espacial omitido por --skip-spatial-in-scattering.")
    elif args.run_spatial_in_scattering:
        print("[PHASE 8/9 SPATIAL] In-scattering espacial omitido: falta --kinematic-cache.")

    # ------------------------------------------------------------------
    # 09. Figuras 2x2 tipo artículo, opcional
    # ------------------------------------------------------------------
    if args.make_4panel and shw is not None:
        fig_dir = outdir / "09_figures_4panel"
        ensure_dir(fig_dir)

        fourpanel_jobs: list[tuple[str, Sequence[str]]] = []

        def add_4panel_job(stage_name: str, prefix: str, paths: dict[str, Path]) -> None:
            missing = [p for p in paths.values() if not p.exists()]
            if missing and not args.dry_run:
                print(f"[WARN] No genero {prefix}: faltan {len(missing)} CSV.")
                return
            cmd = [
                sys.executable,
                scripts["plot_4panel"],
                "--p1", paths["P1"],
                "--p2", paths["P2"],
                "--p4", paths["P4"],
                "--p5", paths["P5"],
                "--outdir", fig_dir,
                "--prefix", prefix,
                "--theta-min", str(args.inside_theta_min),
                "--theta-max", str(args.inside_theta_max),
                "--phi-min", "-60",
                "--phi-max", "60",
                "--blank-zeros",
                "--square-display",
                "--display-step", str(args.fourpanel_display_step),
                "--title", "",
                "--title-log", "",
            ]
            fourpanel_jobs.append((stage_name, cmd))
            output_rows.extend([
                {"stage": "09_figures_4panel", "point": "ALL", "kind": f"{prefix}_linear", "path": str(fig_dir / f"{prefix}_linear.png")},
                {"stage": "09_figures_4panel", "point": "ALL", "kind": f"{prefix}_log", "path": str(fig_dir / f"{prefix}_log.png")},
                {"stage": "09_figures_4panel", "point": "ALL", "kind": f"{prefix}_summary", "path": str(fig_dir / f"{prefix}_summary.csv")},
            ])

        if args.plot_source in ("raw", "both"):
            add_4panel_job(
                "09_4panel_theta_phi_raw",
                "theta_phi_raw",
                {p: dirs["plots"] / "raw" / f"theta_phi_counts_{p}.csv" for p in DEFAULT_POINTS},
            )
        if args.plot_source in ("filtered", "both"):
            add_4panel_job(
                "09_4panel_theta_phi_filtered",
                "theta_phi_filtered",
                {p: dirs["plots"] / "filtered" / f"theta_phi_counts_{p}.csv" for p in DEFAULT_POINTS},
            )
        # Las figuras inside-volcano ya las genera 07_inside_volcano_maps_merged.py.
        # Este bloque opcional queda sólo para theta_phi_counts de 05_plots.

        if fourpanel_jobs:
            print(f"[PHASE 9/9] Figuras 2x2 -> {fig_dir}")
            stage_results.extend(
                run_command_batch(
                    fourpanel_jobs,
                    cwd=outdir,
                    log_dir=dirs["logs"],
                    dry_run=args.dry_run,
                    parallel_jobs=args.parallel_jobs,
                )
            )

    # Índices finales
    write_outputs_index(outdir, output_rows)
    removed_empty_dirs = [] if args.dry_run else cleanup_empty_run_dirs(outdir)
    manifest = {
        "created_at": now_stamp(),
        "parameters": vars(args),
        "resolved_paths": {
            "scripts_dir": str(scripts_dir),
            "hgt_dir": str(hgt_dir),
            "range_file": str(range_file) if range_file else None,
            "empirical_kernel_library": str(empirical_kernel_library) if empirical_kernel_library else None,
            "shw": str(shw) if shw else None,
            "kinematic_cache": str(kinematic_cache) if kinematic_cache else None,
            "outdir": str(outdir),
        },
        "compute_backend": compute_backend,
        "directories": {k: str(v) for k, v in dirs.items()},
        "cleanup": {
            "removed_empty_dirs": [str(p) for p in removed_empty_dirs],
            "preserved_empty_dir_names": ["07_scattering", "08_smearing"],
        },
        "stages": [asdict(r) for r in stage_results],
    }
    with (outdir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n[DONE] Pipeline terminada.")
    print(f"[DONE] Salidas: {outdir}")
    print(f"[DONE] Índice: {outdir / 'pipeline_outputs.csv'}")
    print(f"[DONE] Manifiesto: {outdir / 'run_manifest.json'}")
    print(f"[DONE] Logs: {dirs['logs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
