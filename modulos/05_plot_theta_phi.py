#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Heatmap θ–φ (color = conteo de muones) rápido y robusto.
# - Lee .shw (ARTI 12 cols) en streaming (O(1) memoria)
# - Progreso con tqdm (porcentaje por bytes leídos)
# - Límites por defecto: φ ∈ [-50, 50], θ ∈ [60, 90] y se grafica 90→60
# - Ángulos del usuario:
#     theta = acos(pz/|p|) [deg]
#     phi   = atan2(py, px) en [0,360) [deg]
# - φ_rel para --point (P1/P2/P4/P5): φ_rel = (phi_abs - φ0) mod 360,
#   con φ0 = (90° - az_geo(point→cima)) mod 360.

import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from shw_io import open_shw_bytes, parse_muon_parts, stream_size_hint
from plot_style import apply_scientific_style, finite_percentile, format_angular_axes, style_colorbar, COUNTS_CMAP

# tqdm opcional
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

# Geometría (ajusta coords si las tuyas difieren)
SUMMIT = (4.486552, -75.388975)
POINTS = {
    "P1": (4.492298, -75.381092),
    "P2": (4.494946, -75.388110),
    "P4": (4.476500, -75.386500),
    "P5": (4.488500, -75.379500),
}
MUON_IDS = {"0005", "0006"}  # mu-/mu+

def azimuth_deg(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    th = math.degrees(math.atan2(x, y))
    return (th + 360.0) % 360.0

# Ángulos (tus funciones):
def compute_theta(px, py, pz):
    p_mag = math.sqrt(px*px + py*py + pz*pz)
    if p_mag == 0.0:
        return None
    cos_th = max(-1.0, min(1.0, pz / p_mag))
    return math.degrees(math.acos(cos_th))

def compute_phi(px, py):
    phi = math.degrees(math.atan2(py, px))
    return phi if phi >= 0.0 else phi + 360.0

def solid_angle_per_bin(theta_edges_deg, phi_edges_deg):
    """
    Exact solid angle per (theta, phi) bin.

    For a rectangular bin in spherical coordinates:
        dOmega = dphi * [cos(theta_low) - cos(theta_high)]

    Inputs are in degrees. Output is in sr and has shape:
        (n_theta_bins, n_phi_bins)
    """
    th_rad = np.deg2rad(theta_edges_deg)
    ph_rad = np.deg2rad(phi_edges_deg)

    dphi = np.diff(ph_rad)[None, :]
    dcos = (np.cos(th_rad[:-1]) - np.cos(th_rad[1:]))[:, None]

    return dcos * dphi

def main():
    apply_scientific_style()
    ap = argparse.ArgumentParser(description="Heatmap θ–φ (conteo de muones) rápido y en streaming")
    ap.add_argument("--point", required=True, choices=list(POINTS.keys()), help="Punto de observación (P1/P2/P4/P5)")
    ap.add_argument("--shw", required=True, help="Archivo .shw (ARTI 12 cols)")
    ap.add_argument("--shw-format", choices=["auto", "arti12", "cnf9"], default="auto",
                    help="Formato de entrada. auto detecta ARTI 12 columnas o CNFId energy theta px py pz h bx bz.")
    ap.add_argument("--shw-member", default=None,
                    help="Nombre del miembro dentro de un .tar/.tar.gz. Si se omite, toma el primer .shw.")
    ap.add_argument("--outdir", default=".", help="Directorio de salida")
    ap.add_argument("--bins-theta", type=int, default=60, help="Bins en θ (default 60)")
    ap.add_argument("--bins-phi", type=int, default=40, help="Bins en φ (default 40)")
    # wrap180 activado por defecto, desactívalo con --no-wrap180
    ap.add_argument("--no-wrap180", dest="wrap180", action="store_false",
                    help="No mapear φ_rel a [-180,180) (por defecto SÍ se mapea)")
    ap.set_defaults(wrap180=True)
    ap.add_argument("--phi-min", type=float, default=-50.0, help="Mínimo φ_rel (default -50)")
    ap.add_argument("--phi-max", type=float, default=+50.0, help="Máximo φ_rel (default +50)")
    ap.add_argument("--theta-min", type=float, default=40.0, help="Mínimo θ (default 60)")
    ap.add_argument("--theta-max", type=float, default=90.0, help="Máximo θ (default 90)")
    # sólo muones por defecto; si quieres todos, usa --include-all
    ap.add_argument("--include-all", dest="only_muons", action="store_false",
                    help="Incluir TODAS las partículas (por defecto sólo muones 0005/0006)")
    ap.set_defaults(only_muons=True)
    ap.add_argument("--head", type=int, default=0, help="Si >0, procesa sólo las primeras N coincidencias (debug)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # φ0 del punto
    plat, plon = POINTS[args.point]
    az_geo = azimuth_deg(plat, plon, SUMMIT[0], SUMMIT[1])
    phi0 = (90.0 - az_geo) % 360.0

    # Límites y edges
    th_lo, th_hi = min(args.theta_min, args.theta_max), max(args.theta_min, args.theta_max)
    ph_lo, ph_hi = min(args.phi_min, args.phi_max), max(args.phi_min, args.phi_max)
    th_edges = np.linspace(th_lo, th_hi, args.bins_theta + 1)
    ph_edges = np.linspace(ph_lo, ph_hi, args.bins_phi + 1)
    H = np.zeros((len(th_edges)-1, len(ph_edges)-1), dtype=np.int64)

    # Barra de progreso por bytes
    in_path = Path(args.shw)
    total_bytes = stream_size_hint(in_path)

    taken = 0
    with open_shw_bytes(in_path, member_name=args.shw_member) as f:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"θ–φ ({args.point})")
        for raw in f:
            pbar.update(len(raw))
            s = raw.strip()
            if not s or s.startswith(b"#"):
                continue
            parts = s.split()
            rec = parse_muon_parts(parts, shw_format=args.shw_format, only_muons=args.only_muons)
            if rec is None:
                continue

            th = compute_theta(rec.px, rec.py, rec.pz)
            if th is None or (th < th_lo) or (th > th_hi):
                continue
            ph_abs = compute_phi(rec.px, rec.py)
            ph_rel = (ph_abs - phi0) % 360.0
            if args.wrap180 and ph_rel > 180.0:
                ph_rel -= 360.0
            if (ph_rel < ph_lo) or (ph_rel > ph_hi):
                continue

            i = np.digitize(th, th_edges) - 1
            j = np.digitize(ph_rel, ph_edges) - 1
            if 0 <= i < H.shape[0] and 0 <= j < H.shape[1]:
                H[i, j] += 1
                taken += 1
            if args.head > 0 and taken >= args.head:
                break
        pbar.close()

    # CSV (formato "long")
    th_cent = 0.5*(th_edges[:-1] + th_edges[1:])
    ph_cent = 0.5*(ph_edges[:-1] + ph_edges[1:])
    TH, PH = np.meshgrid(th_cent, ph_cent, indexing="ij")
    import pandas as pd

    # 1) Salida original: conteos crudos por pixel angular.
    df_out = pd.DataFrame({"theta_deg": TH.ravel(), "phi_rel_deg": PH.ravel(), "count": H.ravel()})
    csv_path = outdir / f"theta_phi_counts_{args.point}.csv"
    df_out.to_csv(csv_path, index=False)

    # 2) Salida corregida por jacobiano angular.
    #    Esta corrección NO cambia los conteos. Sólo los expresa por unidad de ángulo sólido.
    #    Para cada pixel:
    #        DeltaOmega = DeltaPhi * [cos(theta_low) - cos(theta_high)]
    #        dN/dOmega  = count / DeltaOmega
    delta_omega = solid_angle_per_bin(th_edges, ph_edges)

    with np.errstate(divide="ignore", invalid="ignore"):
        H_domega = H.astype(float) / delta_omega
        H_domega[~np.isfinite(H_domega)] = np.nan

    df_corr = pd.DataFrame({
        "theta_deg": TH.ravel(),
        "phi_rel_deg": PH.ravel(),
        "count": H.ravel(),
        "delta_omega_sr": delta_omega.ravel(),
        "dN_dOmega_count_per_sr": H_domega.ravel(),
    })
    csv_corr_path = outdir / f"theta_phi_dNdOmega_{args.point}.csv"
    df_corr.to_csv(csv_corr_path, index=False)

    # Plot original: conteos crudos.
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    vmax = finite_percentile(H, 99.5, positive_only=True, fallback=max(float(H.max()), 1.0))
    im = ax.pcolormesh(ph_edges, th_edges, H, shading="flat", cmap=COUNTS_CMAP, vmin=0.0, vmax=vmax, rasterized=True)
    format_angular_axes(ax, args.theta_min, args.theta_max, args.phi_min, args.phi_max)
    ax.set_title(f"Muon counts | {args.point} | N={int(H.sum())}")
    style_colorbar(fig.colorbar(im, ax=ax, shrink=0.92), "Counts")
    png_path = outdir / f"theta_phi_counts_{args.point}.png"
    fig.savefig(png_path)
    plt.close(fig)

    # Plot corregido por ángulo sólido.
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    vmax = finite_percentile(H_domega, 99.5, positive_only=True, fallback=None)
    im = ax.pcolormesh(ph_edges, th_edges, H_domega, shading="flat", cmap=COUNTS_CMAP, vmin=0.0, vmax=vmax, rasterized=True)
    format_angular_axes(ax, args.theta_min, args.theta_max, args.phi_min, args.phi_max)
    ax.set_title(f"Muon intensity proxy | {args.point}")
    style_colorbar(fig.colorbar(im, ax=ax, shrink=0.92), r"Counts sr$^{-1}$")
    png_corr_path = outdir / f"theta_phi_dNdOmega_{args.point}.png"
    fig.savefig(png_corr_path)
    plt.close(fig)

    print(
        f"[OK] Guardado: {png_path.name}, {csv_path.name}, "
        f"{png_corr_path.name}, {csv_corr_path.name} en {outdir} | eventos contados={taken}"
    )

if __name__ == "__main__":
    main()
