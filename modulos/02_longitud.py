#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fov_rock_lengths_simple.py
--------------------------
Uso mínimo: solo pásame una carpeta con los archivos y lo hago para P1, P2, P4, P5.
Requisitos dentro de --data_dir:
  - N04W076.hgt, N04W075.hgt
  - blocked_angles_P1.csv, blocked_angles_P2.csv, blocked_angles_P4.csv, blocked_angles_P5.csv

Ejemplo:
  python fov_rock_lengths_simple.py --data_dir ./data --outdir ./outputs_rock

Parámetros físicos fijos (puedes cambiarlos abajo):
  - R_MAX = 5000 m, paso s = 5 m, altura instrumento = 2 m
  - φ=0 hacia la cima; dx_east = r*sin(AZ), dy_north = r*cos(AZ)

Autor: GPT-5 Thinking
"""
import os
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config fijos del sitio (Cerro Machín)
# -----------------------------
BBOX = (4.466944, 4.500833, -75.404720, -75.372694)
SUMMIT = (4.486552, -75.388975)
POINTS = {
    "P1": (4.492298, -75.381092),
    "P2": (4.494946, -75.388110),
    "P4": (4.4765,   -75.3865),
    "P5": (4.4885,   -75.3795),
}
HGT_ORDER = ["N04W076.hgt", "N04W075.hgt"]  # oeste -> este

# Física / muestreo
R_MAX_M = 5000.0
S_STEP_M = 5.0
HEIGHT_OFFSET_M = 2.0

# -----------------------------
# Utilidades
# -----------------------------
R_EARTH = 6371000.0

def meters_to_deg(dx_east_m, dy_north_m, at_lat_deg):
    dlat_deg = (dy_north_m / R_EARTH) * (180.0 / np.pi)
    dlon_deg = (dx_east_m / (R_EARTH * np.cos(np.radians(at_lat_deg)))) * (180.0 / np.pi)
    return dlat_deg, dlon_deg

def azimuth_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    th = np.degrees(np.arctan2(x, y))
    return (th + 360.0) % 360.0

def load_hgt(path):
    with open(path, "rb") as f:
        data = f.read()
    n = int(np.sqrt(len(data) // 2))
    arr = np.frombuffer(data, dtype=">i2").reshape((n, n)).astype(np.float32)
    arr[arr == -32768] = np.nan
    return arr

def tile_latlon_vectors(tile_name):
    name = os.path.splitext(os.path.basename(tile_name))[0]
    hemi_lat = 1 if name[0] == "N" else -1
    hemi_lon = -1 if name[3] == "W" else 1
    base_lat = int(name[1:3]) * hemi_lat
    base_lon = int(name[4:7]) * hemi_lon
    n = 3601
    lats = np.linspace(base_lat + 1, base_lat, n, endpoint=True)  # N->S
    lons = np.linspace(base_lon, base_lon + 1, n, endpoint=True)  # W->E
    return lats, lons

def mosaic_two_hgt(files):
    A = load_hgt(files[0])
    B = load_hgt(files[1])
    latsA, lonsA = tile_latlon_vectors(files[0])
    latsB, lonsB = tile_latlon_vectors(files[1])
    if not np.allclose(latsA, latsB, atol=1e-9):
        raise ValueError("Latitude vectors of tiles do not align.")
    lats = latsA
    lons = np.concatenate([lonsA, lonsB[1:]])
    mosaic = np.concatenate([A, B[:, 1:]], axis=1)
    return mosaic, lats, lons

def crop_dem(mosaic, lats, lons, bbox):
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    crop = mosaic[np.ix_(lat_mask, lon_mask)]
    crop_lats = lats[lat_mask]
    crop_lons = lons[lon_mask]
    return crop, crop_lats, crop_lons

def make_interp(crop, crop_lats, crop_lons):
    dlat = float(abs(crop_lats[1] - crop_lats[0]))
    dlon = float(abs(crop_lons[1] - crop_lons[0]))
    nlat, nlon = crop.shape

    def interp_elev_vec_nd(lat, lon):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        out = np.full(lat.shape, np.nan, dtype=np.float32)
        m = (lat >= crop_lats.min()) & (lat <= crop_lats.max()) & (lon >= crop_lons.min()) & (lon <= crop_lons.max())
        if not np.any(m):
            return out
        latm = lat[m]; lonm = lon[m]
        i = ((crop_lats.max() - latm) / dlat)
        j = ((lonm - crop_lons.min()) / dlon)
        i0 = np.floor(i).astype(int); j0 = np.floor(j).astype(int)
        di = i - i0; dj = j - j0
        i1 = np.clip(i0 + 1, 0, nlat - 1); j1 = np.clip(j0 + 1, 0, nlon - 1)
        Q11 = crop[i0, j0]; Q21 = crop[i1, j0]
        Q12 = crop[i0, j1]; Q22 = crop[i1, j1]
        z = (Q11*(1-di)*(1-dj) + Q21*(di)*(1-dj) + Q12*(1-di)*dj + Q22*di*dj).astype(np.float32)
        out[m] = z
        return out

    return interp_elev_vec_nd

def inside_length_one(plat, plon, az_center, z0, phi_deg, theta_deg, interp,
                      s_max=R_MAX_M, s_step=S_STEP_M, h_offset=HEIGHT_OFFSET_M):
    th = math.radians(theta_deg)
    AZ = math.radians(az_center + phi_deg)
    sin_th = math.sin(th); cos_th = math.cos(th)
    sin_AZ = math.sin(AZ); cos_AZ = math.cos(AZ)

    s_vals = np.arange(0.0, s_max + 1e-6, s_step, dtype=np.float32)
    r_vals = s_vals * sin_th
    z_ray  = z0 + h_offset + s_vals * cos_th

    dxe = r_vals * sin_AZ
    dyn = r_vals * cos_AZ
    dlat_deg, dlon_deg = meters_to_deg(dxe, dyn, plat)
    lat_path = plat + dlat_deg
    lon_path = plon + dlon_deg

    topo = interp(lat_path, lon_path)
    valid = ~np.isnan(topo)
    if not np.any(valid):
        return 0.0
    first_invalid = np.argmax(~valid) if np.any(~valid) else -1
    if first_invalid > 0:
        s_vals = s_vals[:first_invalid]
        z_ray = z_ray[:first_invalid]
        topo = topo[:first_invalid]
    elif first_invalid == 0:
        return 0.0

    inside = z_ray < topo
    if inside.size < 2:
        return 0.0

    total = 0.0
    for i in range(inside.size - 1):
        a, b = inside[i], inside[i+1]
        if a and b:
            total += s_step
        elif a != b:
            zr0, zr1 = z_ray[i], z_ray[i+1]
            tp0, tp1 = topo[i], topo[i+1]
            denom = (zr1 - zr0) - (tp1 - tp0)
            t_cross = (tp0 - zr0) / denom if denom != 0 else 0.5
            t_cross = 0.0 if t_cross < 0 else (1.0 if t_cross > 1 else t_cross)
            if a and not b:
                total += t_cross * s_step
            else:
                total += (1.0 - t_cross) * s_step
    return float(total)

def centers_to_edges(vals):
    vals = np.asarray(sorted(np.unique(vals)))
    if len(vals) == 1:
        dv = 0.5
        return np.array([vals[0]-dv, vals[0]+dv])
    mids = (vals[1:] + vals[:-1]) / 2.0
    left_gap = vals[1] - vals[0]
    right_gap = vals[-1] - vals[-2]
    edges = np.concatenate([[vals[0] - left_gap/2], mids, [vals[-1] + right_gap/2]])
    return edges

def pick_column(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        low = col.lower()
        if any(cand.lower() in low for cand in candidates):
            return col
    if required:
        raise KeyError(f"No encontré columnas {list(candidates)}. Disponibles: {list(df.columns)}")
    return None

def infer_geometry_mask(df):
    """
    Devuelve una máscara booleana de celdas que realmente interceptan topografía.

    Compatible con:
      - nuevo blocked_angles_P*.csv: trae inside_volcano_geometry
      - formato viejo: sólo phi/theta, se asume que todas las filas son bloqueadas
    """
    col = pick_column(
        df,
        ["inside_volcano_geometry", "blocked_geometry", "blocked", "inside", "mask"],
        required=False,
    )
    if col is None:
        return np.ones(len(df), dtype=bool), None
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return vals > 0.0, col

def save_heatmap(df, out_png, title):
    #df = df[df["theta_deg"] <= 85.0].copy()
    thetas = np.sort(df["theta_deg"].unique())
    phis = np.sort(df["phi_deg"].unique())
    grid = np.full((len(thetas), len(phis)), np.nan, dtype=np.float32)
    t2i = {t: i for i, t in enumerate(thetas)}
    p2j = {p: j for j, p in enumerate(phis)}
    for _, row in df.iterrows():
        i = t2i[row["theta_deg"]]; j = p2j[row["phi_deg"]]
        val = float(row["length_inside_m"])
        grid[i, j] = val if val > 0 else np.nan

    theta_edges = centers_to_edges(thetas)
    phi_edges = centers_to_edges(phis)
    PH_e, TH_e = np.meshgrid(phi_edges, theta_edges)

    plt.figure(figsize=(8, 6))
    plt.grid(False)
    plt.pcolormesh(PH_e, TH_e, grid,vmax=2000 , shading="auto")
    plt.gca().invert_yaxis()
    plt.xlabel("Relative azimuth φ (deg) [0 = to summit]")
    plt.ylabel("Zenith θ (deg)")
    plt.ylim(90,50)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Length inside rock (m)")
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.axhline(90.0, linestyle=":", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Inside-rock lengths for blocked rays (simple, 4 puntos)")
    ap.add_argument("--data_dir",default="outputs",help="Carpeta con HGT y blocked_angles_P*.csv")
    ap.add_argument("--outdir", default="outputs",help="Carpeta de salida")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Cargar DEM
    hgt_paths = [str(data_dir / HGT_ORDER[0]), str(data_dir / HGT_ORDER[1])]
    mosaic, lats, lons = mosaic_two_hgt(hgt_paths)
    crop, crop_lats, crop_lons = crop_dem(mosaic, lats, lons, BBOX)
    interp = make_interp(crop, crop_lats, crop_lons)

    # Azimut al volcán y z0
    az_center = {name: azimuth_deg(lat, lon, SUMMIT[0], SUMMIT[1]) for name, (lat, lon) in POINTS.items()}
    z0_map = {name: float(interp(np.array([lat]), np.array([lon]))[0]) for name, (lat, lon) in POINTS.items()}

    summary = []
    for name in ["P1", "P2", "P4", "P5"]:
        csv_in = data_dir / f"blocked_angles_{name}.csv"
        if not csv_in.exists():
            print(f"[!] No encontrado: {csv_in}. Me lo salto.")
            continue
        df_in = pd.read_csv(csv_in)
        phi_col = pick_column(df_in, ["phi_deg", "phi_rel_deg", "phi"])
        theta_col = pick_column(df_in, ["theta_deg", "theta", "zenith_deg"])

        df_in = df_in.copy()
        df_in[phi_col] = pd.to_numeric(df_in[phi_col], errors="coerce")
        df_in[theta_col] = pd.to_numeric(df_in[theta_col], errors="coerce")
        df_in = df_in.dropna(subset=[phi_col, theta_col]).reset_index(drop=True)

        # En el formato nuevo, blocked_angles_P*.csv contiene TODA la grilla
        # angular y una columna inside_volcano_geometry. Para ahorrar tiempo y
        # conservar el canvas completo, sólo calculamos longitud en las celdas
        # que sí interceptan topografía; las demás quedan con L=0.
        inside_mask, geom_col = infer_geometry_mask(df_in)
        if "inside_volcano_geometry" not in df_in.columns:
            df_in["inside_volcano_geometry"] = inside_mask.astype(np.uint8)

        phis = df_in[phi_col].to_numpy(np.float32)
        thetas = df_in[theta_col].to_numpy(np.float32)

        lengths = np.zeros(len(df_in), dtype=np.float32)
        plat, plon = POINTS[name]
        for k, (phi, theta, do_calc) in enumerate(zip(phis, thetas, inside_mask)):
            if not do_calc:
                continue
            lengths[k] = inside_length_one(
                plat, plon, az_center[name], z0_map[name],
                float(phi), float(theta), interp
            )

        df_out = df_in.copy()
        if phi_col != "phi_deg":
            df_out["phi_deg"] = df_out[phi_col]
        if theta_col != "theta_deg":
            df_out["theta_deg"] = df_out[theta_col]
        df_out["length_inside_m"] = lengths

        # Orden estable de columnas principales, conservando columnas extra.
        first_cols = ["phi_deg", "theta_deg", "inside_volcano_geometry", "length_inside_m"]
        extra_cols = [c for c in df_out.columns if c not in first_cols]
        df_out = df_out[first_cols + extra_cols]

        out_csv = outdir / f"rock_length_{name}.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"[+] {name}: guardado {out_csv}")

        out_png = outdir / f"heatmap_{name}.png"
        save_heatmap(df_out, out_png, f"{name} — Inside-rock path length (m)")
        print(f"[+] {name}: guardado {out_png}")

        summary.append({
            "point": name,
            "n_cells": int(df_out.shape[0]),
            "n_inside_geometry": int(df_out["inside_volcano_geometry"].sum()) if "inside_volcano_geometry" in df_out.columns else int(df_out.shape[0]),
            "n_positive_length": int((df_out["length_inside_m"] > 0).sum()),
            "mean_length_m": float(df_out.loc[df_out["length_inside_m"] > 0, "length_inside_m"].mean() or 0.0),
            "median_length_m": float(df_out.loc[df_out["length_inside_m"] > 0, "length_inside_m"].median() or 0.0),
            "max_length_m": float(df_out["length_inside_m"].max() or 0.0),
            "min_positive_length_m": float(df_out.loc[df_out["length_inside_m"] > 0, "length_inside_m"].min() or 0.0),
            "csv": str(out_csv),
            "heatmap": str(out_png),
        })

    if summary:
        df_sum = pd.DataFrame(summary)
        sum_csv = outdir / "summary.csv"
        df_sum.to_csv(sum_csv, index=False)
        print(f"[+] Resumen: {sum_csv}")
        print(df_sum.to_string(index=False))
    else:
        print("[!] No se generaron salidas. Revisa la carpeta y nombres de archivos.")

if __name__ == "__main__":
    main()
