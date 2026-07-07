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


REQUIRED_HGT = ("N04W076.hgt", "N04W075.hgt")
DEFAULT_POINTS = ("P1", "P2", "P4", "P5")

SCRIPT_01 = "01_puntos.py"
SCRIPT_02 = "02_longitud.py"
SCRIPT_03 = "03_ecrit_heatmaps.py"
SCRIPT_05 = "05_plot_theta_phi.py"
SCRIPT_06 = "06_filter_muons_by_ecrit.py"  # versión rápida multi-punto en esta rama
SCRIPT_07_MERGED = "07_inside_volcano_maps_merged.py"
SCRIPT_07_FAST = "07_inside_volcano_allpoints_fast.py"
SCRIPT_07_LEGACY = "07_plot_counts_inside_volcano_geometry.py"
SCRIPT_08 = "08_scattering_highland_v2.py"
SCRIPT_09 = "09_apply_angular_smearing_pretty_MC.py"
SCRIPT_08_EMPIRICAL = "08_scattering_empirical_kernel.py"
SCRIPT_09_EMPIRICAL = "09_apply_angular_smearing_empirical_kernel.py"
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
    return Path(path).expanduser().resolve()


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
    ensure_dir(log_dir)
    log_path = log_dir / f"{name}.log"
    start = time.time()
    printable = " ".join(str(c) for c in cmd)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}\n")
        log.write(f"# time: {now_stamp()}\n")
        log.write(f"# cwd: {cwd}\n")
        log.write(f"# cmd: {printable}\n\n")

        if dry_run:
            elapsed = time.time() - start
            return StageResult(name, list(map(str, cmd)), str(cwd), str(log_path), elapsed, 0, "DRY-RUN")

        proc = subprocess.run(
            list(map(str, cmd)),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        if proc.stdout:
            log.write(proc.stdout)

    elapsed = time.time() - start
    status = "OK" if proc.returncode == 0 else "ERROR"
    result = StageResult(name, list(map(str, cmd)), str(cwd), str(log_path), elapsed, proc.returncode, status)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Falló la etapa {name} con código {proc.returncode}. Revisa el log: {log_path}"
        )
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Orquestador completo: FOV -> longitud -> Ecrit -> filtro -> mapas -> inside -> scattering -> smearing."
    )
    ap.add_argument("--scripts-dir", default=".", help="Carpeta donde están 01_puntos.py ... 06_filter_muons_by_ecrit.py")
    ap.add_argument("--hgt-dir", required=True, help="Carpeta con N04W076.hgt y N04W075.hgt")
    ap.add_argument("--range-file", default=None, help="data_rock.dat o muon_range_table.csv. Si no se da, se busca automáticamente.")
    ap.add_argument("--shw", default=None, help="Archivo .shw de ARTI. Si se omite, no se filtran muones ni se grafican conteos.")
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
    ap.add_argument("--inside-filtered-workers", type=int, default=1, help="Procesos para inside-volcano filtered; cada punto usa su propio .shw")
    ap.add_argument("--inside-show-zeros", action="store_true", help="Muestra ceros en los mapas inside-volcano; por defecto quedan en blanco")
    ap.add_argument("--filter-mode", choices=["fast", "legacy"], default="fast", help="fast usa 06 multi-punto; legacy usa la interfaz vieja por punto")
    ap.add_argument("--parallel-jobs", type=int, default=1, help="Número de procesos paralelos para etapas independientes por punto")
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
                    help="Ruta a empirical_kernel_library.npz. Requerido si --scattering-model empirical/both.")
    ap.add_argument("--empirical-interp-method", choices=["rbf_linear", "linear", "nearest"], default="rbf_linear",
                    help="Interpolación interna del kernel empírico en log(L), log(E/L).")
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
    ap.add_argument("--empirical-kernel-threshold", type=float, default=None,
                    help="Opcional: descarta pesos K menores que este valor en el smearing empírico.")
    ap.add_argument("--empirical-max-kernel-radius-mrad", type=float, default=None,
                    help="Opcional: limita el radio angular evaluado del kernel empírico.")
    ap.add_argument("--plot-theta-min", type=float, default=60.0)
    ap.add_argument("--plot-theta-max", type=float, default=90.0)
    ap.add_argument("--plot-phi-min", type=float, default=-50.0)
    ap.add_argument("--plot-phi-max", type=float, default=50.0)
    ap.add_argument("--bins-theta", type=int, default=60)
    ap.add_argument("--bins-phi", type=int, default=40)
    ap.add_argument("--force", action="store_true", help="Borra la carpeta de salida antes de correr")
    ap.add_argument("--dry-run", action="store_true", help="Sólo imprime/guarda comandos; no ejecuta scripts")
    ap.add_argument("--skip-geometry", action="store_true")
    ap.add_argument("--skip-lengths", action="store_true")
    ap.add_argument("--skip-ecrit", action="store_true")
    ap.add_argument("--skip-filter", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    scripts_dir = resolve_path(args.scripts_dir)
    hgt_dir = resolve_path(args.hgt_dir)
    outdir = resolve_path(args.outdir)
    range_file_arg = resolve_path(args.range_file)
    shw = resolve_path(args.shw)
    empirical_kernel_library = resolve_path(args.empirical_kernel_library)

    assert scripts_dir is not None
    assert hgt_dir is not None
    assert outdir is not None

    scripts = check_scripts(scripts_dir)

    if args.scattering_model in ("empirical", "both"):
        scripts["scattering_empirical"] = scripts_dir / SCRIPT_08_EMPIRICAL
        scripts["smearing_empirical"] = scripts_dir / SCRIPT_09_EMPIRICAL
        require_file(scripts["scattering_empirical"], "script scattering_empirical")
        require_file(scripts["smearing_empirical"], "script smearing_empirical")
        if empirical_kernel_library is None:
            raise FileNotFoundError("--empirical-kernel-library es requerido con --scattering-model empirical/both")
        require_file(empirical_kernel_library, "empirical_kernel_library.npz")

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
        scripts["plot_4panel"] = scripts_dir / SCRIPT_4PANEL
        require_file(scripts["plot_4panel"], "script plot_4panel")

    require_file(hgt_dir / REQUIRED_HGT[0], REQUIRED_HGT[0])
    require_file(hgt_dir / REQUIRED_HGT[1], REQUIRED_HGT[1])
    if shw is not None:
        require_file(shw, "archivo .shw")

    if args.force and outdir.exists():
        shutil.rmtree(outdir)
    ensure_dir(outdir)

    dirs = {
        "inputs": outdir / "00_inputs",
        "geometry": outdir / "01_geometry",
        "lengths": outdir / "02_lengths",
        "ecrit": outdir / "03_ecrit",
        "filtered": outdir / "04_filtered",
        "plots": outdir / "05_plots",
        "inside_volcano": outdir / "06_inside_volcano",
        "scattering": outdir / "07_scattering",
        "smearing": outdir / "08_smearing",
        "scattering_empirical": outdir / "07_scattering_empirical",
        "smearing_empirical": outdir / "08_smearing_empirical",
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
    # 01. Geometría / FOV
    # ------------------------------------------------------------------
    if not args.skip_geometry:
        remove_and_create(dirs["work_geometry"])
        for fname in REQUIRED_HGT:
            link_or_copy(hgt_dir / fname, dirs["work_geometry"] / fname, overwrite=True)

        cmd = [sys.executable, scripts["geometry"]]
        print(f"[1/5] Geometría y ángulos bloqueados -> {dirs['geometry']}")
        result = run_command("01_geometry", cmd, cwd=dirs["work_geometry"], log_dir=dirs["logs"], dry_run=args.dry_run)
        stage_results.append(result)

        if not args.dry_run:
            copy_tree_contents(dirs["work_geometry"] / "outputs", dirs["geometry"], overwrite=True)
            for fname in REQUIRED_HGT:
                link_or_copy(hgt_dir / fname, dirs["geometry"] / fname, overwrite=True)

            required = [dirs["geometry"] / f"blocked_angles_{p}.csv" for p in args.points]
            required.append(dirs["geometry"] / "dem_fans.png")
            require_files(required, "01_geometry")

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
        print(f"[2/5] Longitudes dentro de roca -> {dirs['lengths']}")
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
        print(f"[3/5] Energía crítica -> {dirs['ecrit']}")
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
    # 04. Filtro de muones por Ecrit
    # ------------------------------------------------------------------
    filtered_by_point: dict[str, Path] = {}
    do_filter = (shw is not None) and (not args.skip_filter)
    if do_filter:
        print(f"[4/5] Filtrado de muones .shw -> {dirs['filtered']}")
        shw_stem = shw.stem if shw is not None else "input"

        for p in args.points:
            filtered_by_point[p] = dirs["filtered"] / f"{shw_stem}_filtered_{p}.shw"

        if args.filter_mode == "fast":
            # Nueva interfaz: un solo proceso lee el .shw una vez y escribe todos los puntos.
            cmd = [
                sys.executable,
                scripts["filter"],
                "--points", *args.points,
                "--shw", shw,
                "--indir", dirs["ecrit"],
                "--outdir", dirs["filtered"],
                "--tol-phi", str(args.tol_phi),
                "--tol-theta", str(args.tol_theta),
                "--treat-out-of-grid-as-clear", str(args.treat_out_of_grid_as_clear),
            ]
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
                    "--indir", dirs["ecrit"],
                    "--out", out_shw,
                    "--tol_phi", str(args.tol_phi),
                    "--tol_theta", str(args.tol_theta),
                    "--treat_out_of_grid_as_clear", str(args.treat_out_of_grid_as_clear),
                ]
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

    elif shw is None:
        print("[4/5] Sin --shw: salto filtrado de muones.")
    else:
        print("[4/5] Filtrado omitido por bandera.")

    # ------------------------------------------------------------------
    # 05. Mapas θ–φ de conteo
    # ------------------------------------------------------------------
    make_plots = (shw is not None) and (not args.skip_plots) and (args.plot_source != "none")
    if make_plots:
        print(f"[5/5] Mapas θ–φ -> {dirs['plots']}")
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
                    source_for_point = filtered_by_point.get(p, dirs["filtered"] / f"{shw.stem}_filtered_{p}.shw")
                else:
                    source_for_point = source_path

                if source_for_point is None:
                    continue

                cmd = [
                    sys.executable,
                    scripts["plot"],
                    "--point", p,
                    "--shw", source_for_point,
                    "--outdir", plot_dir,
                    "--bins-theta", str(args.bins_theta),
                    "--bins-phi", str(args.bins_phi),
                    "--theta-min", str(args.plot_theta_min),
                    "--theta-max", str(args.plot_theta_max),
                    "--phi-min", str(args.plot_phi_min),
                    "--phi-max", str(args.plot_phi_max),
                ]
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
    elif shw is None:
        print("[5/5] Sin --shw: salto mapas θ–φ.")
    else:
        print("[5/5] Mapas θ–φ omitidos por bandera.")


    # ------------------------------------------------------------------
    # 06. Mapas de cuentas sólo dentro de la geometría del volcán
    # ------------------------------------------------------------------
    make_inside = (shw is not None) and (args.inside_volcano_source != "none")
    if make_inside:
        print(f"[6/6] Cuentas dentro del volcán + figuras finales -> {dirs['inside_volcano']}")

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
            "--raw-shw", shw,
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

    elif shw is None:
        print("[6/6] Sin --shw: salto cuentas dentro de la geometría del volcán.")
    else:
        print("[6/6] Cuentas dentro de geometría omitidas (--inside-volcano-source none).")


    # ------------------------------------------------------------------
    # 07. Diagnóstico Highland de dispersión angular
    # ------------------------------------------------------------------
    make_scattering = (not args.skip_scattering) and (args.scattering_model in ("highland", "both"))
    if make_scattering:
        print(f"[7/9] Scattering Highland -> {dirs['scattering']}")
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
        print("[7/9] Scattering omitido por bandera.")

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
        print(f"[7b/9] Scattering empírico Geant4 -> {dirs['scattering_empirical']}")
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
        print("[7b/9] Scattering empírico omitido por bandera.")

    # ------------------------------------------------------------------
    # 08. Smearing angular sobre mapas θ–φ
    # ------------------------------------------------------------------
    make_smearing = (
        (not args.skip_smearing)
        and (shw is not None)
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

            print(f"[8/9] Smearing angular ({smearing_source}) -> {smearing_outdir}")

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
    elif shw is None:
        print("[8/9] Sin --shw: salto smearing angular Highland.")
    elif args.scattering_model in ("highland", "both"):
        print("[8/9] Smearing Highland omitido por bandera o faltan mapas fuente.")

    # ------------------------------------------------------------------
    # 08b. Smearing angular empírico sobre mapas θ–φ
    # ------------------------------------------------------------------
    make_smearing_empirical = (
        (not args.skip_smearing)
        and (shw is not None)
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

            print(f"[8b/9] Smearing angular empírico ({smearing_source}) -> {smearing_outdir}")

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
    elif shw is None:
        print("[8b/9] Sin --shw: salto smearing angular empírico.")
    elif args.scattering_model in ("empirical", "both"):
        print("[8b/9] Smearing empírico omitido por bandera o faltan mapas fuente.")

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
            print(f"[9/9] Figuras 2x2 -> {fig_dir}")
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
    manifest = {
        "created_at": now_stamp(),
        "parameters": vars(args),
        "resolved_paths": {
            "scripts_dir": str(scripts_dir),
            "hgt_dir": str(hgt_dir),
            "range_file": str(range_file) if range_file else None,
            "empirical_kernel_library": str(empirical_kernel_library) if empirical_kernel_library else None,
            "shw": str(shw) if shw else None,
            "outdir": str(outdir),
        },
        "directories": {k: str(v) for k, v in dirs.items()},
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
    from orquestador_machin_with_eventmc import main as unified_main

    raise SystemExit(unified_main())
