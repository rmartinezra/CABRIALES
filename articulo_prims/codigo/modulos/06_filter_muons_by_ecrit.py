#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_filter_muons_by_ecrit_allpoints_fast.py

Filtro rápido de muones por Ecrit para varios puntos en UNA sola lectura del .shw.

Ventajas:
- Lee el .shw una sola vez para P1/P2/P4/P5.
- Escribe un .shw filtrado por punto.
- Evita pandas MultiIndex dentro del loop.
- Evita np.argmin por evento usando searchsorted.
- Actualiza tqdm por bloques de MB, no por cada línea.

Mantiene la convención angular:
    theta = acos(pz/|p|)
    phi   = atan2(py, px) en [0, 360)
    phi_rel = (phi - phi0_point) mod 360, con opción equivalente [-180,180)

Uso:
python3 06_filter_muons_by_ecrit_allpoints_fast.py \
  --points P1 P2 P4 P5 \
  --shw ./data/bga-2212-01_043200.shw \
  --indir ./run_machin/03_ecrit \
  --outdir ./run_machin/04_filtered \
  --tol-phi 0.51 \
  --tol-theta 0.51 \
  --treat-out-of-grid-as-clear 1
"""

from __future__ import annotations

import argparse
import math
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import pandas as pd
from shw_io import (
    open_output_bytes,
    open_shw_bytes,
    output_template_for_compression,
    parse_muon_parts,
    shw_stem,
    stream_size_hint,
    theta_phi_from_momentum,
)

try:
    from tqdm import tqdm
except Exception:
    class tqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())
        def update(self, n=1):
            pass
        def close(self):
            pass

SUMMIT = (4.486552, -75.388975)
POINTS = {
    "P1": (4.492298, -75.381092),
    "P2": (4.494946, -75.388110),
    "P4": (4.476500, -75.386500),
    "P5": (4.488500, -75.379500),
}


def azimuth_deg(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def pick_column(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    for c in df.columns:
        low = c.lower()
        if any(k.lower() in low for k in keys):
            return c
    return None


def load_ecrit_grid(indir: Path, point: str):
    path = indir / f"ecrit_table_{point}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encontré {path}")

    df = pd.read_csv(path)

    phi_col = pick_column(df, ["phi_deg", "phi", "phi_rel_deg"])
    theta_col = pick_column(df, ["theta_deg", "theta"])
    ecrit_col = "Ecrit_total_GeV" if "Ecrit_total_GeV" in df.columns else pick_column(df, ["Ecrit_total", "Ecrit"])

    if phi_col is None or theta_col is None or ecrit_col is None:
        raise ValueError(
            f"No pude identificar columnas en {path}. "
            f"phi_col={phi_col}, theta_col={theta_col}, ecrit_col={ecrit_col}. "
            f"Columnas: {list(df.columns)}"
        )

    df = df[[phi_col, theta_col, ecrit_col]].copy()
    df[phi_col] = pd.to_numeric(df[phi_col], errors="coerce").round(3)
    df[theta_col] = pd.to_numeric(df[theta_col], errors="coerce").round(3)
    df[ecrit_col] = pd.to_numeric(df[ecrit_col], errors="coerce")
    df = df.dropna(subset=[phi_col, theta_col])

    phi_centers = np.sort(df[phi_col].unique().astype(float))
    theta_centers = np.sort(df[theta_col].unique().astype(float))

    phi_index = {round(float(v), 3): j for j, v in enumerate(phi_centers)}
    theta_index = {round(float(v), 3): i for i, v in enumerate(theta_centers)}

    ecrit = np.full((len(theta_centers), len(phi_centers)), np.nan, dtype=float)
    valid = np.zeros_like(ecrit, dtype=bool)

    # itertuples con name=None evita problemas si los nombres tienen caracteres raros.
    for ph, th, ec in df[[phi_col, theta_col, ecrit_col]].itertuples(index=False, name=None):
        ph = round(float(ph), 3)
        th = round(float(th), 3)
        i = theta_index[th]
        j = phi_index[ph]
        ecrit[i, j] = float(ec) if pd.notna(ec) else np.nan
        valid[i, j] = True

    plat, plon = POINTS[point]
    az_geo = azimuth_deg(plat, plon, SUMMIT[0], SUMMIT[1])
    phi0 = (90.0 - az_geo) % 360.0

    return {
        "point": point,
        "phi": phi_centers,
        "theta": theta_centers,
        "ecrit": ecrit,
        "valid": valid,
        "phi0": phi0,
    }


def nearest_index(arr: np.ndarray, x: float, tol: float):
    n = len(arr)
    if n == 0:
        return None

    k = int(np.searchsorted(arr, x))
    best = None
    best_d = float("inf")

    if 0 <= k < n:
        d = abs(float(arr[k]) - x)
        if d < best_d:
            best = k
            best_d = d
    if 0 <= k - 1 < n:
        d = abs(float(arr[k - 1]) - x)
        if d < best_d:
            best = k - 1
            best_d = d

    if best is None or best_d > tol:
        return None
    return best


def should_keep_for_grid(E_tot, theta_deg, phi_abs, grid, tol_phi, tol_theta, treat_clear):
    phi_rel = (phi_abs - grid["phi0"]) % 360.0
    phi_candidates = (phi_rel, phi_rel if phi_rel <= 180.0 else phi_rel - 360.0)

    for phi_c in phi_candidates:
        j = nearest_index(grid["phi"], phi_c, tol_phi)
        if j is None:
            continue

        i = nearest_index(grid["theta"], theta_deg, tol_theta)
        if i is None:
            continue

        if not grid["valid"][i, j]:
            return bool(treat_clear)

        Ecrit = grid["ecrit"][i, j]
        if not math.isfinite(Ecrit):
            return bool(treat_clear)

        return E_tot >= Ecrit

    # Fuera de la grilla angular o sin matching dentro de tolerancia.
    return bool(treat_clear)


def main():
    ap = argparse.ArgumentParser(description="Filtro rápido multi-punto de muones por Ecrit.")
    ap.add_argument("--points", nargs="+", default=["P1", "P2", "P4", "P5"], choices=list(POINTS.keys()))
    ap.add_argument("--shw", required=True, type=Path)
    ap.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="auto",
                    help="Formato de entrada. auto detecta ARTI 12 columnas o CNFId energy theta px py pz h bx bz.")
    ap.add_argument("--shw-member", default=None,
                    help="Nombre del miembro dentro de un .tar/.tar.gz. Si se omite, toma el primer .shw.")
    ap.add_argument("--indir", required=True, type=Path, help="Directorio con ecrit_table_P*.csv")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--out-template", default="{stem}_filtered_{point}.shw", help="Variables: {stem}, {point}")
    ap.add_argument("--output-compression", choices=["none", "gz", "xz", "bz2"], default="none",
                    help="Comprime los .shw filtrados. Recomendado para corridas grandes: gz.")
    ap.add_argument("--tol-phi", "--tol_phi", dest="tol_phi", type=float, default=0.51)
    ap.add_argument("--tol-theta", "--tol_theta", dest="tol_theta", type=float, default=0.51)
    ap.add_argument("--discard-upgoing", "--discard_upgoing", dest="discard_upgoing", action="store_true")
    ap.add_argument("--treat-out-of-grid-as-clear", "--treat_out_of_grid_as_clear", dest="treat_clear", type=int, choices=[0, 1], default=1)
    ap.add_argument("--progress-update-mb", type=float, default=16.0, help="Actualizar tqdm cada N MB para reducir overhead")
    args = ap.parse_args()

    if not args.shw.exists():
        raise FileNotFoundError(args.shw)

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[1] Cargando tablas Ecrit")
    grids = {}
    for p in args.points:
        grids[p] = load_ecrit_grid(args.indir, p)
        print(f"  {p}: theta={len(grids[p]['theta'])}, phi={len(grids[p]['phi'])}")

    stem = shw_stem(args.shw)
    out_template = output_template_for_compression(args.out_template, args.output_compression)
    out_paths = {p: args.outdir / out_template.format(stem=stem, point=p) for p in args.points}

    total_bytes = stream_size_hint(args.shw)
    update_bytes = max(1, int(args.progress_update_mb * 1024 * 1024))
    pending_update = 0

    cand_mu = 0
    kept = {p: 0 for p in args.points}
    read_lines = 0

    print("[2] Filtrando en una sola lectura del .shw")
    with ExitStack() as stack:
        fout = {p: stack.enter_context(open_output_bytes(out_paths[p])) for p in args.points}
        with open_shw_bytes(args.shw, member_name=args.shw_member) as fin:
            pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Filtrando multi-punto")

            for raw in fin:
                read_lines += 1
                pending_update += len(raw)

                if pending_update >= update_bytes:
                    pbar.update(pending_update)
                    pending_update = 0

                s = raw.strip()
                if not s:
                    continue

                if s.startswith(b"#"):
                    for fp in fout.values():
                        fp.write(raw)
                    continue

                parts = s.split()
                rec = parse_muon_parts(parts, shw_format=args.shw_format, only_muons=True)
                if rec is None:
                    continue

                if args.discard_upgoing and rec.pz > 0.0:
                    continue

                angles = theta_phi_from_momentum(rec.px, rec.py, rec.pz)
                if angles is None:
                    continue

                cand_mu += 1

                theta_deg, phi_abs = angles
                E_tot = rec.e_total_GeV

                for point, grid in grids.items():
                    if should_keep_for_grid(E_tot, theta_deg, phi_abs, grid, args.tol_phi, args.tol_theta, args.treat_clear):
                        fout[point].write(raw)
                        kept[point] += 1

            if pending_update:
                pbar.update(pending_update)
            pbar.close()

    print("[OK] Finalizado")
    print(f"  líneas leídas: {read_lines}")
    print(f"  muones candidatos: {cand_mu}")
    for p in args.points:
        print(f"  {p}: {kept[p]} muones escritos -> {out_paths[p]}")


if __name__ == "__main__":
    main()
