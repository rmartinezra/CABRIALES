#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import statistics
from pathlib import Path

TEMPLATE = r'''# TestEm5 near-threshold muon transport
/control/verbose 1
/run/verbose 1
/process/em/verbose 0

/testem/det/setWorldMat G4_Galactic
/testem/det/setAbsMat __MATERIAL__
/testem/det/setAbsThick __L_M__ m
/testem/det/setAbsYZ __ABS_YZ_M__ m
/testem/det/setWorldX __WORLD_X_M__ m
/testem/det/setWorldYZ __WORLD_YZ_M__ m

/testem/phys/addPhysics emstandard_opt4
/run/setCut __CUT_CM__ cm
/run/initialize

/testem/gun/setDefault
/gun/particle __PARTICLE__
/gun/energy __E_GEV__ GeV

/analysis/setFileName __OUTBASE__
/analysis/h1/set 10 400 0.0 __E_GEV__ GeV
/analysis/h1/set 12 400 0.0 __ANGLE_MAX_MRAD__ mrad
/analysis/h1/set 13 800 -__ANGLE_MAX_MRAD__ __ANGLE_MAX_MRAD__ mrad
/analysis/h1/set 14 800 -__POS_MAX_M__ __POS_MAX_M__ m
/analysis/h1/set 15 400 0.0 __RADIUS_MAX_M__ m

/testem/stack/killSecondaries
/random/setSeeds __SEED1__ __SEED2__
/run/printProgress __PRINT_PROGRESS__
/run/beamOn __N_EVENTS__
'''


def finite_float(x):
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def find_col(columns, candidates):
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for col in cols:
        low = col.lower()
        if any(c.lower() in low for c in candidates):
            return col
    raise KeyError(f"No encontré {candidates}. Columnas: {cols}")


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path.resolve()}")
    with path.open(newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError('El CSV no tiene encabezado')
        lc = find_col(r.fieldnames, ['length_inside_m','L_m','rock_length_m','length_m','length'])
        ec = find_col(r.fieldnames, ['Tcrit_GeV','T_crit_GeV','kinetic_crit_GeV'])
        rows = []
        for row in r:
            L = finite_float(row.get(lc)); E = finite_float(row.get(ec))
            if L is not None and E is not None and L > 0 and E > 0:
                rows.append((L,E))
    if not rows:
        raise RuntimeError('No encontré pares positivos (L,Tcrit)')
    return rows, lc, ec


def load_pairs(items):
    rows = []
    for item in items or []:
        try:
            a,b = item.split(':',1)
            L,E = float(a), float(b)
        except Exception as exc:
            raise ValueError(f"Par inválido {item!r}; use L:Tcrit, por ejemplo 300:120") from exc
        if L <= 0 or E <= 0:
            raise ValueError(f"Par no positivo: {item}")
        rows.append((L,E))
    return rows


def representative(rows, requested):
    tol = max(5.0, 0.01*requested)
    near = [(L,E) for L,E in rows if abs(L-requested) <= tol]
    if not near:
        near = [min(rows, key=lambda p: abs(p[0]-requested))]
    return statistics.median(x[0] for x in near), statistics.median(x[1] for x in near), len(near)


def tag(x, n=2):
    return f"{x:.{n}f}".replace('.','p').replace('-','m')


def fill(template, values):
    out = template
    for k,v in values.items():
        out = out.replace(f"__{k}__", str(v))
    leftovers = [s for s in out.split() if s.startswith('__') and s.endswith('__')]
    if leftovers:
        raise RuntimeError(f"Tokens sin reemplazar: {leftovers}")
    return out


def main():
    ap = argparse.ArgumentParser(description='Genera macros TestEm5 cerca de Tcrit sin pandas')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--ecrit-csv', type=Path)
    src.add_argument('--pairs', nargs='+', help='Pares L:Tcrit, ejemplo 100:45.2 300:123.8')
    ap.add_argument('--lengths', nargs='+', type=float, required=True)
    ap.add_argument('--factors', nargs='+', type=float, default=[0.90,0.95,1.00,1.05,1.10,1.20,1.50])
    ap.add_argument('--events', type=int, default=50000)
    ap.add_argument('--particle', choices=['mu-','mu+'], default='mu-')
    ap.add_argument('--material', default='G4_SILICON_DIOXIDE')
    ap.add_argument('--cut-cm', type=float, default=1.0)
    ap.add_argument('--angle-max-mrad', type=float, default=200.0)
    ap.add_argument('--executable', default='./TestEm5')
    ap.add_argument('--outdir', type=Path, default=Path('macros_near_threshold'))
    args = ap.parse_args()

    if args.events <= 0:
        ap.error('--events debe ser positivo')
    if any(x <= 0 for x in args.lengths):
        ap.error('--lengths debe contener valores positivos')
    if any(x <= 0 for x in args.factors):
        ap.error('--factors debe contener valores positivos')

    if args.ecrit_csv:
        rows, lc, ec = load_csv(args.ecrit_csv)
        source_desc = f"{args.ecrit_csv.resolve()} [{lc}, {ec}]"
    else:
        rows = load_pairs(args.pairs)
        source_desc = 'manual --pairs'

    exe = Path(args.executable).expanduser().resolve()
    if not exe.exists():
        raise FileNotFoundError(f"No existe TestEm5: {exe}")

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    commands = ['#!/usr/bin/env bash','set -euo pipefail','HERE="$(cd "$(dirname "$0")" && pwd)"','cd "$HERE"','']
    run_index = 0

    for requested in args.lengths:
        L,Tcrit,nrows = representative(rows, requested)
        # Broad enough slab/world to avoid lateral leakage in this pilot.
        half_spread = max(20.0, L*math.tan(args.angle_max_mrad/1000.0))
        abs_yz = 2.4*half_spread
        world_yz = 1.2*abs_yz
        world_x = L + 20.0
        pos_max = max(20.0, half_spread)
        radius_max = math.sqrt(2.0)*pos_max

        for factor in args.factors:
            run_index += 1
            E = factor*Tcrit
            base = f"near_threshold_L{tag(L,1)}m_f{tag(factor,2)}_{args.particle.replace('+','plus').replace('-','minus')}"
            macro = outdir / f"{base}.mac"
            values = {
                'MATERIAL':args.material, 'L_M':f'{L:.6f}', 'ABS_YZ_M':f'{abs_yz:.6f}',
                'WORLD_X_M':f'{world_x:.6f}', 'WORLD_YZ_M':f'{world_yz:.6f}',
                'CUT_CM':f'{args.cut_cm:.6f}', 'PARTICLE':args.particle,
                'E_GEV':f'{E:.10g}', 'OUTBASE':base,
                'ANGLE_MAX_MRAD':f'{args.angle_max_mrad:.6f}',
                'POS_MAX_M':f'{pos_max:.6f}', 'RADIUS_MAX_M':f'{radius_max:.6f}',
                'SEED1':100003+2*run_index, 'SEED2':200003+2*run_index,
                'PRINT_PROGRESS':max(1,args.events//20), 'N_EVENTS':args.events,
            }
            macro.write_text(fill(TEMPLATE, values), encoding='utf-8')
            commands += [f'echo "Running {base}"', f'{shlex.quote(str(exe))} {shlex.quote(macro.name)} > {shlex.quote(base+".log")} 2>&1', '']
            manifest.append({
                'requested_length_m':requested,'length_used_m':L,'rows_used':nrows,
                'Tcrit_kinetic_GeV':Tcrit,'energy_factor':factor,'Ekin_GeV':E,
                'particle':args.particle,'events':args.events,'material':args.material,
                'macro':macro.name,'output_base':base,
            })

    with (outdir/'near_threshold_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
    runner = outdir/'run_all.sh'
    runner.write_text('\n'.join(commands)+'\n',encoding='utf-8'); os.chmod(runner,0o755)

    print(f'Fuente: {source_desc}')
    print(f'Pares válidos: {len(rows)}')
    print(f'Macros generados: {len(manifest)}')
    print(f'Salida: {outdir}')
    print(f'Ejecutable: {exe}')
    print(f'Ejecute: bash {runner}')

if __name__ == '__main__':
    main()
