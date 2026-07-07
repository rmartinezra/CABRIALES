import os
import math
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Configuración principal
# =========================

HGT_FILES = ["N04W076.hgt", "N04W075.hgt"]  # oeste a este
# Recorte fijo alrededor del Cerro Machín (ajústalo si necesitas más contexto)
BBOX = dict(
    lat_min=4.466944, lat_max=4.500833,
    lon_min=-75.404720, lon_max=-75.372694
)

# Puntos de observación (cuatro seleccionados) y cima
POINTS = {
    "P5": (4.4885,   -75.3795),
    "P1": (4.492298, -75.381092),
    "P4": (4.4765,   -75.3865),     # este mira "más arriba"
    "P2": (4.494946, -75.388110),
}
SUMMIT = (4.486552, -75.388975)  # CIMA

# Parámetros FOV
FOV_HALF_DEG = 60.0
R_MAX_M = 5000.0                 # 5 km
HEIGHT_OFFSET_M = 2.0
THETA_STEP = 0.5
PHI_STEP = 0.5
NSAMPLES = 1000

# Rangos zenitales por punto (P4 mira más arriba)
THETA_RANGE_BY_POINT = {
    "P4": (30.0, 120.0),
    "default": (60.0, 120.0),
}

# Carpeta de salida
OUTDIR = "outputs"


# =========================
# Utilidades geoespaciales
# =========================

R_EARTH = 6371000.0  # m

def meters_to_deg(dx_east_m, dy_north_m, at_lat_deg):
    dlat_deg = (dy_north_m / R_EARTH) * (180.0 / np.pi)
    dlon_deg = (dx_east_m / (R_EARTH * np.cos(np.radians(at_lat_deg)))) * (180.0 / np.pi)
    return dlat_deg, dlon_deg

def deg_to_m(dx_lon_deg, dy_lat_deg, at_lat_deg):
    dy = (dy_lat_deg * np.pi/180.0) * R_EARTH
    dx = (dx_lon_deg * np.pi/180.0) * R_EARTH * np.cos(np.radians(at_lat_deg))
    return dx, dy

def planar_distance_m(lat1, lon1, lat2, lon2):
    dx, dy = deg_to_m(lon2 - lon1, lat2 - lat1, (lat1 + lat2) / 2.0)
    return float(np.hypot(dx, dy))

def azimuth_deg(lat1, lon1, lat2, lon2):
    # bearing geodésico aproximado (suficiente para ~km)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    th = np.degrees(np.arctan2(x, y))
    return (th + 360.0) % 360.0


# =========================
# Lectura de HGT y DEM
# =========================

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
    # Asumimos mosaico horizontal (oeste->este)
    A = load_hgt(files[0])
    B = load_hgt(files[1])
    latsA, lonsA = tile_latlon_vectors(files[0])
    latsB, lonsB = tile_latlon_vectors(files[1])
    assert np.allclose(latsA, latsB, atol=1e-9)
    lats = latsA
    lons = np.concatenate([lonsA, lonsB[1:]])  # evitar columna duplicada
    mosaic = np.concatenate([A, B[:, 1:]], axis=1)
    return mosaic, lats, lons

def crop_dem(mosaic, lats, lons, bbox):
    lat_mask = (lats >= bbox["lat_min"]) & (lats <= bbox["lat_max"])
    lon_mask = (lons >= bbox["lon_min"]) & (lons <= bbox["lon_max"])
    crop = mosaic[np.ix_(lat_mask, lon_mask)]
    crop_lats = lats[lat_mask]
    crop_lons = lons[lon_mask]
    return crop, crop_lats, crop_lons


# =========================
# Interpolación bilineal
# =========================

def make_interp(crop, crop_lats, crop_lons):
    dlat = float(abs(crop_lats[1] - crop_lats[0]))
    dlon = float(abs(crop_lons[1] - crop_lons[0]))
    nlat, nlon = crop.shape

    def interp_elev_vec_nd(lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
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


# =========================
# FOV y proyección
# =========================

def build_fov_relative_with_edges_custom(plat, plon, az_center, interp_elev_vec_nd,
                                         theta_min=60.0, theta_max=120.0,
                                         theta_step=2.0, phi_step=2.0,
                                         max_distance_m=5000.0, n_samples=700,
                                         height_offset=2.0, fov_half=60.0):
    # edges & centers
    theta_edges = np.arange(theta_min, theta_max + theta_step + 1e-6, theta_step)
    phi_edges = np.arange(-fov_half, fov_half + phi_step + 1e-6, phi_step)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2.0
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2.0

    distances = np.linspace(0.0, max_distance_m, n_samples).astype(np.float32)
    z0 = float(interp_elev_vec_nd(np.array([plat]), np.array([plon]))[0])
    TH = np.radians(theta_centers)[:, None]  # (T,1)
    z_los_base = z0 + height_offset + distances[None, :] * np.cos(TH)  # (T,S)

    F = np.zeros((len(theta_centers), len(phi_centers)), dtype=np.uint8)

    for j, dphi in enumerate(phi_centers):
        PH = np.radians(az_center + dphi)
        dxu, dyu = np.sin(PH), np.cos(PH)  # (este, norte) desde acimut N-clockwise
        HORIZ = (distances[None, :] * np.sin(TH)).astype(np.float32)  # (T,S)
        dxe = HORIZ * dxu
        dyn = HORIZ * dyu
        dlat_deg, dlon_deg = meters_to_deg(dxe, dyn, plat)
        lat_path = plat + dlat_deg
        lon_path = plon + dlon_deg
        topo = interp_elev_vec_nd(lat_path, lon_path)  # (T,S)
        valid = ~np.isnan(topo)
        ge_ok = (z_los_base >= topo) | (~valid)
        all_ok = np.all(ge_ok, axis=1) & np.any(valid, axis=1)
        F[:, j] = all_ok.astype(np.uint8)

    return theta_edges, phi_edges, theta_centers, phi_centers, distances, z0, F

def alpha_max_curve(plat, plon, az_center, phi_centers, distances, interp_elev_vec_nd):
    z0 = float(interp_elev_vec_nd(np.array([plat]), np.array([plon]))[0])
    alphas = np.full_like(phi_centers, np.nan, dtype=np.float32)
    for j, dphi in enumerate(phi_centers):
        PH = np.radians(az_center + dphi)
        dxu, dyu = np.sin(PH), np.cos(PH)
        dxe = dxu * distances
        dyn = dyu * distances
        dlat_deg, dlon_deg = meters_to_deg(dxe, dyn, plat)
        lat_path = plat + dlat_deg
        lon_path = plon + dlon_deg
        topo = interp_elev_vec_nd(lat_path, lon_path)  # (S,)
        valid = ~np.isnan(topo)
        if not np.any(valid):
            continue
        dh = topo[valid] - z0
        r = distances[valid]
        r = np.where(r == 0, 1e-6, r)
        alpha = np.arctan2(dh, r)
        alphas[j] = np.nanmax(alpha)
    return alphas  # rad


# =========================
# Gráficas y CSV
# =========================

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def plot_dem_with_fans(crop, crop_lats, crop_lons, az_by_point, outfile_png,
                       fov_half_deg=60.0, margin_m=50.0):
    # Mapa DEM con abanicos ±phi centrados en φ=0 (punto→cima)
    plt.figure(figsize=(8.0, 6.6))
    extent = [crop_lons.min(), crop_lons.max(), crop_lats.min(), crop_lats.max()]
    plt.imshow(crop, extent=extent, origin="upper", aspect="equal")
    plt.colorbar(label="Elevation (m a.s.l.)")
    plt.plot(SUMMIT[1], SUMMIT[0], marker="^", markersize=9, color="orange", label="Cima")
    for name, (plat, plon) in POINTS.items():
        if name not in az_by_point:
            continue
        az0 = az_by_point[name]["azimuth"]
        r_dom = planar_distance_m(plat, plon, SUMMIT[0], SUMMIT[1]) + margin_m
        # construir polígono (usar convención dx=sin, dy=cos)
        rel = np.linspace(-fov_half_deg, +fov_half_deg, 241)
        phis = az0 + rel
        ang = np.radians(phis)
        dxe = np.sin(ang) * r_dom
        dyn = np.cos(ang) * r_dom
        dlat_deg, dlon_deg = meters_to_deg(dxe, dyn, plat)
        arc_lat = plat + dlat_deg
        arc_lon = plon + dlon_deg
        poly_lat = np.concatenate([[plat], arc_lat, [plat]])
        poly_lon = np.concatenate([[plon], arc_lon, [plon]])
        plt.fill(poly_lon, poly_lat, alpha=0.18, linewidth=0.8)
        # bordes (±half)
        for sgn in (-1, +1):
            az_e = az0 + sgn * fov_half_deg
            ang_e = np.radians(az_e)
            dx_e = math.sin(ang_e) * r_dom
            dy_e = math.cos(ang_e) * r_dom
            dlat_e, dlon_e = meters_to_deg(dx_e, dy_e, plat)
            plt.plot([plon, plon + dlon_e], [plat, plat + dlat_e], '-', linewidth=1.1)
        # φ=0 exacto: segmento punto→cima
        plt.plot([plon, SUMMIT[1]], [plat, SUMMIT[0]], '--', linewidth=2.0, color='k')
        plt.plot(plon, plat, 'wo', markersize=6)
        plt.text(plon, plat, f" {name}", color='w', fontsize=9)
    plt.title("DEM + abanicos ±φ centrados en φ=0 (punto→cima)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=180, bbox_inches="tight")
    plt.close()

def plot_fov_png(name, theta_edges, phi_edges, F, theta_boundary, outfile_png):
    PH_e, TH_e = np.meshgrid(phi_edges, theta_edges)
    plt.figure(figsize=(7.6, 5.8))
    plt.pcolormesh(PH_e, TH_e, F, shading="flat")
    plt.gca().invert_yaxis()
    plt.xlabel("Relative azimuth φ (deg) [0 = to summit]")
    plt.ylabel("Zenith θ (deg)")
    plt.title(f"{name} — FOV (R={int(R_MAX_M/1000)} km) y proyección topográfica")
    if theta_boundary is not None:
        phi_cent = (phi_edges[:-1] + phi_edges[1:]) / 2.0
        plt.plot(phi_cent, theta_boundary, linewidth=2.0, label="θ_boundary(φ) = 90° − α_max(φ)")
        plt.legend()
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.axhline(90.0, linestyle=":", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=180, bbox_inches="tight")
    plt.close()

def save_blocked_csv(name, theta_centers, phi_centers, F, outfile_csv):
    """
    Guarda TODA la grilla angular, no sólo las celdas bloqueadas.

    Convención:
      F == 1  -> rayo libre / cielo abierto
      F == 0  -> rayo intercepta topografía / inside-volcano geometry

    Mantengo el nombre histórico blocked_angles_{P}.csv para no romper
    el resto de la cadena, pero ahora el archivo incluye una máscara explícita.
    Esto evita que las etapas siguientes reconstruyan una grilla recortada que
    empieza sólo donde aparece el primer rayo bloqueado.
    """
    ph, th = np.meshgrid(phi_centers, theta_centers)
    inside = (F == 0).astype(np.uint8)
    clear = (F == 1).astype(np.uint8)
    rows = np.column_stack([
        ph.ravel(),
        th.ravel(),
        inside.ravel(),
        clear.ravel(),
    ])
    header = "phi_deg,theta_deg,inside_volcano_geometry,clear_sky_geometry"
    np.savetxt(outfile_csv, rows, delimiter=",", header=header, comments="", fmt=["%.6f", "%.6f", "%d", "%d"])


# =========================
# Pipeline principal
# =========================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # DEM
    mosaic, lats, lons = mosaic_two_hgt(HGT_FILES)
    crop, crop_lats, crop_lons = crop_dem(mosaic, lats, lons, BBOX)
    interp = make_interp(crop, crop_lats, crop_lons)

    # Azimuts hacia la cima y configuración por punto
    az_by_point = {}
    for name, (plat, plon) in POINTS.items():
        az0 = azimuth_deg(plat, plon, SUMMIT[0], SUMMIT[1])
        tmin, tmax = THETA_RANGE_BY_POINT.get(name, THETA_RANGE_BY_POINT["default"])
        az_by_point[name] = dict(azimuth=az0, theta_min=tmin, theta_max=tmax)

    # PNG del DEM + abanicos
    dem_png = os.path.join(OUTDIR, "dem_fans.png")
    plot_dem_with_fans(crop, crop_lats, crop_lons, az_by_point, dem_png,
                       fov_half_deg=FOV_HALF_DEG, margin_m=50.0)

    # Por punto: FOV, proyección, CSV
    for name, (plat, plon) in POINTS.items():
        az0 = az_by_point[name]["azimuth"]
        tmin = az_by_point[name]["theta_min"]
        tmax = az_by_point[name]["theta_max"]
        th_edges, ph_edges, th_cent, ph_cent, distances, z0, F = build_fov_relative_with_edges_custom(
            plat, plon, az0, interp,
            theta_min=tmin, theta_max=tmax, theta_step=THETA_STEP, phi_step=PHI_STEP,
            max_distance_m=R_MAX_M, n_samples=NSAMPLES, height_offset=HEIGHT_OFFSET_M,
            fov_half=FOV_HALF_DEG
        )
        # proyección topográfica
        alphas = alpha_max_curve(plat, plon, az0, ph_cent, distances, interp)
        theta_boundary = 90.0 - np.degrees(alphas)

        # PNG del FOV
        fov_png = os.path.join(OUTDIR, f"fov_{name}.png")
        plot_fov_png(name, th_edges, ph_edges, F, theta_boundary, fov_png)

        # CSV con grilla angular completa + máscara de geometría del volcán
        csv_path = os.path.join(OUTDIR, f"blocked_angles_{name}.csv")
        save_blocked_csv(name, th_cent, ph_cent, F, csv_path)

    # Resumen
    print(f"[OK] Outputs en: {OUTDIR}")
    print(" - dem_fans.png")
    for name in POINTS.keys():
        print(f" - fov_{name}.png")
        print(f" - blocked_angles_{name}.csv")

if __name__ == "__main__":
    main()
