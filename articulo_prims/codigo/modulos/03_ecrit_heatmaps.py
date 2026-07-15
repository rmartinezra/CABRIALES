#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_ecrit_heatmaps.py
--------------------
Definitivo: calcula y exporta mapas 2D de energía crítica Ecrit(θ,φ) para P1,P2,P4,P5.
- Entrada geométrica: rock_length_{P}.csv (salida de 02_longitud.py), con columnas (phi, theta, length_inside_m).
- Física de frenado: tabla CSDA de muones en roca estándar tomada de data_rock.dat (o muon_range_table.csv).
- Define Ecrit como energía CINÉTICA mínima (GeV) tal que Range(T) ≥ ρ * L (g/cm²).
- También exporta E_total = T + m_mu.

Uso típico:
  python 05_ecrit_heatmaps.py --indir ./outputs --outdir ./outputs_ecrit --rho 2.65 --points P1 P2 P4 P5

Notas:
- Si existe muon_range_table.csv en --indir se usa; si no, se parsea data_rock.dat para generarla.
- Los mapas sólo tienen valores donde hay bloqueo (celdas presentes en rock_length_{P}.csv); el resto se deja NaN.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

MUON_MASS_GEV = 0.10565837  # GeV

def parse_data_rock_to_csv(data_rock_path: Path, out_csv: Path) -> None:
    """Parsea data_rock.dat y guarda CSV con columnas T_MeV, CSDA_gcm2."""
    txt = data_rock_path.read_text(errors="ignore").splitlines()
    start_idx = None
    for i, ln in enumerate(txt):
        if "CSDA Range" in ln and i+1 < len(txt) and "[g/cm^2]" in txt[i+1]:
            start_idx = i + 2
            break
    if start_idx is None:
        start_idx = 10
    rows = []
    for ln in txt[start_idx:]:
        if not ln.strip():
            continue
        toks = re.findall(r"[-+]?\d+\.\d+E[+-]\d+|[-+]?\d+\.\d+|\d+", ln)
        if len(toks) < 9:
            continue
        vals = list(map(float, toks[:11]))
        rows.append(vals)
    if not rows:
        raise RuntimeError(f"No pude extraer tabla CSDA desde {data_rock_path.name}")
    arr = np.asarray(rows, dtype=float)
    T_MeV = arr[:, 0]
    CSDA = arr[:, 8]
    df = pd.DataFrame({"T_MeV": T_MeV, "CSDA_gcm2": CSDA}).dropna()
    df = df[df["T_MeV"] > 0].sort_values("T_MeV").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def load_csda_table(indir: Path):
    """Devuelve (R_sorted[g/cm^2], T_sorted[MeV])."""
    csv_path = indir / "muon_range_table.csv"
    if not csv_path.exists():
        dat = indir / "data_rock.dat"
        if not dat.exists():
            raise FileNotFoundError(f"No encontré {csv_path} ni {dat}. Provee uno de los dos en {indir}.")
        parse_data_rock_to_csv(dat, csv_path)
    df = pd.read_csv(csv_path)
    T = df["T_MeV"].to_numpy(dtype=float)
    R = df["CSDA_gcm2"].to_numpy(dtype=float)
    m = np.isfinite(T) & np.isfinite(R)
    T, R = T[m], R[m]
    order = np.argsort(R)
    return R[order], T[order]

def invert_range_to_Tgev(R_sorted, T_sorted_MeV, X_gcm2):
    """
    Invertir rango CSDA -> energía cinética crítica.

    Importante para la nueva geometría completa:
    las celdas fuera del volcán tienen L=0 y por tanto X=0. En esas celdas
    la energía cinética crítica debe ser 0, no el primer punto de la tabla CSDA.
    """
    X = np.asarray(X_gcm2, dtype=float)
    T_mev = np.zeros_like(X, dtype=float)
    m = np.isfinite(X) & (X > 0.0)
    if np.any(m):
        X_clip = np.clip(X[m], R_sorted.min(), R_sorted.max())
        T_mev[m] = np.interp(X_clip, R_sorted, T_sorted_MeV)
    T_mev[~np.isfinite(T_mev)] = np.nan
    return T_mev * 1e-3

def centers_to_edges(vals):
    vals = np.asarray(sorted(np.unique(vals)))
    if len(vals) == 1:
        dv = 0.5
        return np.array([vals[0]-dv, vals[0]+dv], dtype=float)
    mids = (vals[1:] + vals[:-1]) / 2.0
    left_gap = vals[1] - vals[0]
    right_gap = vals[-1] - vals[-2]
    edges = np.concatenate([[vals[0] - left_gap/2], mids, [vals[-1] + right_gap/2]])
    return edges

def build_grid(df, phi_col, theta_col, val_col):
    phis = np.sort(df[phi_col].unique())
    thetas = np.sort(df[theta_col].unique())
    Z = np.full((len(thetas), len(phis)), np.nan, dtype=float)
    p2j = {p: j for j, p in enumerate(phis)}
    t2i = {t: i for i, t in enumerate(thetas)}
    for _, row in df.iterrows():
        j = p2j[row[phi_col]]
        i = t2i[row[theta_col]]
        Z[i, j] = row[val_col]
    return phis, thetas, Z

def process_point(rock_csv, R_sorted, T_sorted_MeV, outdir, rho_g_cm3):
    df = pd.read_csv(rock_csv)
    def pick(keys):
        for k in keys:
            if k in df.columns: return k
        for c in df.columns:
            if any(k.lower() in c.lower() for k in keys):
                return c
        return None
    phi_col = pick(["phi_deg","phi"])
    theta_col = pick(["theta_deg","theta"])
    L_col = pick(["length_inside_m","L_m","length_m","longitud_m","longitud","length"])
    if not all([phi_col, theta_col, L_col]):
        raise ValueError(f"Columnas faltantes en {rock_csv.name}. Tengo: {df.columns.tolist()}" )

    L_cm = df[L_col].to_numpy(dtype=float) * 100.0
    X = rho_g_cm3 * L_cm  # g/cm^2
    Tcrit_GeV = invert_range_to_Tgev(R_sorted, T_sorted_MeV, X)
    Etot_GeV = Tcrit_GeV + MUON_MASS_GEV

    dfo = df.copy()
    dfo["X_g_cm2"] = X
    dfo["Tcrit_GeV"] = Tcrit_GeV
    dfo["Ecrit_total_GeV"] = Etot_GeV

    phi_cent, theta_cent, Z_T = build_grid(dfo, phi_col, theta_col, "Tcrit_GeV")
    _, _, Z_E = build_grid(dfo, phi_col, theta_col, "Ecrit_total_GeV")

    # Sólo para visualización: deja en blanco las celdas sin roca.
    # La tabla CSV conserva esas celdas con Tcrit=0 y Ecrit_total=m_mu.
    if L_col is not None:
        _, _, Z_L = build_grid(dfo, phi_col, theta_col, L_col)
        clear = ~(np.isfinite(Z_L) & (Z_L > 0.0))
        Z_T = Z_T.copy(); Z_E = Z_E.copy()
        Z_T[clear] = np.nan
        Z_E[clear] = np.nan

    phi_edges = centers_to_edges(phi_cent)
    theta_edges = centers_to_edges(theta_cent)

    point = rock_csv.stem.replace("rock_length_","" )
    outdir.mkdir(parents=True, exist_ok=True)
    csv_out = outdir / f"ecrit_table_{point}.csv"
    dfo.to_csv(csv_out, index=False)

    # Heatmap Tcrit
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8.4, 6.0))
    PH, TH = np.meshgrid(phi_edges,theta_edges)
    plt.pcolormesh(PH, TH, Z_T, shading="flat",vmax=2000)
    plt.gca().invert_yaxis()
    plt.xlabel("Relative azimuth φ (deg) [0 points to summit]")
    plt.ylabel("Zenith θ (deg)")
    plt.title(f"T_crit(θ,φ) — {point}")
    plt.ylim(90,50)
    cbar = plt.colorbar()
    cbar.set_label("T_crit (GeV)")
    plt.tight_layout()
    png_T = outdir / f"Tcrit_heatmap_{point}.png"
    plt.savefig(png_T, dpi=180, bbox_inches="tight")
    plt.close()

    # Heatmap E_total
    plt.figure(figsize=(8.4, 6.0))
    plt.pcolormesh(PH, TH, Z_E,vmax=4000 ,shading="flat")
    plt.gca().invert_yaxis()
    plt.xlabel("Relative azimuth φ (deg) [0 points to summit]")
    plt.ylabel("Zenith θ (deg)")
    plt.title(f"E_total,crit(θ,φ) — {point}")
    cbar = plt.colorbar()
    plt.ylim(90,50)
    cbar.set_label("E_total,crit (GeV)")
    plt.tight_layout()
    png_E = outdir / f"Etotal_heatmap_{point}.png"
    plt.savefig(png_E, dpi=180, bbox_inches="tight")
    plt.close()

    return csv_out, png_T, png_E

def main():
    ap = argparse.ArgumentParser(description="Ecrit heatmaps for P1,P2,P4,P5 from rock_length CSVs + CSDA table.")
    ap.add_argument("--indir", default="outputs", help="Directorio con rock_length_*.csv y data_rock.dat/muon_range_table.csv")
    ap.add_argument("--outdir", default="outputs", help="Directorio de salida")
    ap.add_argument("--rho", type=float, default=2.65, help="Densidad efectiva (g/cm^3)")
    ap.add_argument("--points", nargs="+", default=["P1","P2","P4","P5"], help="Lista de puntos a procesar")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    R_sorted, T_sorted_MeV = load_csda_table(indir)

    for P in args.points:
        rock_csv = indir / f"rock_length_{P}.csv"
        if not rock_csv.exists():
            print(f"[WARN] No encuentro {rock_csv}. Salto {P}.")
            continue
        csv_out, png_T, png_E = process_point(rock_csv, R_sorted, T_sorted_MeV, outdir, args.rho)
        print(f"[OK] {P}: {csv_out.name}, {png_T.name}, {png_E.name}")

    print("[DONE] Revisa los PNG/CSV en", outdir)

if __name__ == "__main__":
    main()
