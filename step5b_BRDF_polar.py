import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
import os

# --- Output directory ---
outdir = "figures/crop_analysis"
os.makedirs(outdir, exist_ok=True)

# --- PROSAIL base configuration ---
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02, 'LAI': 3.0,
    'psoil': 0.2, 'tts': 30,
    'LIDFa': -0.35, 'LIDFb': -0.15
}

# --- Spectral bands of interest ---
CHL_BAND = (670, 680)  # chlorophyll absorption
RED_EDGE = (710, 720)  # red edge

# --- Angular bins (simulate 0–180°, reflect to 180–360°) ---
n_az = 100
n_vz = 50
azimuth_half_edges = np.linspace(0, 180, n_az + 1)
zenith_edges = np.linspace(0, 70, n_vz + 1)
azimuth_centers = 0.5 * (azimuth_half_edges[:-1] + azimuth_half_edges[1:])
zenith_centers = 0.5 * (zenith_edges[:-1] + zenith_edges[1:])
AZ_C, VZ_C = np.meshgrid(azimuth_centers, zenith_centers, indexing='ij')

# --- Hotspot parameter values to simulate ---
hspot_values = [0.0, 0.1, 0.5, 1.0]

# --- Spectral index masks ---
prosail = Prosail()
wl = np.array(prosail.wl)
chl_mask = (wl >= CHL_BAND[0]) & (wl <= CHL_BAND[1])
re_mask = (wl >= RED_EDGE[0]) & (wl <= RED_EDGE[1])

# --- Simulate BRDF for ψ = 0–180° ---
def simulate_brdf(hspot):
    chl_refl = np.zeros_like(AZ_C)
    re_refl = np.zeros_like(AZ_C)
    for i in range(AZ_C.shape[0]):
        for j in range(AZ_C.shape[1]):
            config = base_config.copy()
            config['psi'] = AZ_C[i, j]
            config['tto'] = VZ_C[i, j]
            config['hspot'] = hspot
            refl = np.array(prosail.run(config))
            chl_refl[i, j] = np.mean(refl[chl_mask])
            re_refl[i, j] = np.mean(refl[re_mask])
    return chl_refl, re_refl

# --- Extend BRDF to full 0–360° by vertical flip ---
def extend_to_full_polar(data_half):
    return np.vstack([data_half, np.flipud(data_half)])


# --- Full azimuth edge grid for plotting (0–360°) ---
azimuth_full_edges = np.linspace(0, 360, 2 * n_az + 1)
azimuth_full_edges_rad = np.radians(azimuth_full_edges)
THETA, R = np.meshgrid(azimuth_full_edges_rad, zenith_edges, indexing='ij')

# --- Plotting function ---
def plot_polar_brdf(data_full, title, filename):
    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, polar=True)
    pcm = ax.pcolormesh(THETA, R, data_full, shading='auto', cmap='viridis')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 70)
    ax.set_yticks(np.arange(0, 80, 10))
    ax.set_yticklabels([f"{d}°" for d in np.arange(0, 80, 10)])
    ax.set_rlabel_position(135)
    ax.set_title(title, fontsize=12)
    fig.colorbar(pcm, ax=ax, orientation='vertical', label='Reflectance')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=300)
    plt.close()

# --- Run simulation and plotting ---
for hspot in hspot_values:
    chl_half, re_half = simulate_brdf(hspot)
    chl_full = extend_to_full_polar(chl_half)
    re_full = extend_to_full_polar(re_half)
    tag = f"hspot_{hspot:.2f}".replace('.', '_')

    plot_polar_brdf(
        chl_full,
        f"BRDF – Chlorophyll Band (670–680 nm), hspot = {hspot}",
        f"brdf_chl_full_{tag}.jpg"
    )
    plot_polar_brdf(
        re_full,
        f"BRDF – Red Edge Band (710–720 nm), hspot = {hspot}",
        f"brdf_rededge_full_{tag}.jpg"
    )

print(f"✅ Full polar BRDF plots saved to: {outdir}")
