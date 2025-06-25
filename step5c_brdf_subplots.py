import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
import os

# --- Output directory ---
outdir = "figures/crop_analysis"
os.makedirs(outdir, exist_ok=True)

# --- PROSAIL configuration ---
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02, 'LAI': 3.0,
    'psoil': 0.2, 'tts': 30,
    'LIDFa': -0.35, 'LIDFb': -0.15
}

# --- Spectral bands ---
CHL_BAND = (670, 680)
RED_EDGE = (710, 720)

# --- Angular grid (simulate 0–180°, reflect to 360°) ---
n_az = 100
n_vz = 50
azimuth_half_edges = np.linspace(0, 180, n_az + 1)
zenith_edges = np.linspace(0, 70, n_vz + 1)
azimuth_centers = 0.5 * (azimuth_half_edges[:-1] + azimuth_half_edges[1:])
zenith_centers = 0.5 * (zenith_edges[:-1] + zenith_edges[1:])
AZ_C, VZ_C = np.meshgrid(azimuth_centers, zenith_centers, indexing='ij')

# --- hspot values ---
hspot_values = [0.0, 0.1, 0.5, 1.0]

# --- PROSAIL wavelength setup ---
prosail = Prosail()
wl = np.array(prosail.wl)
chl_mask = (wl >= CHL_BAND[0]) & (wl <= CHL_BAND[1])
re_mask = (wl >= RED_EDGE[0]) & (wl <= RED_EDGE[1])

# --- Simulate BRDF over 0–180° azimuths ---
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

# --- Symmetric extension to full 0–360° azimuth domain ---
def extend_to_full_polar(data_half):
    return np.vstack([data_half, np.flipud(data_half)])

# --- Plotting function with dynamic global max ---
def make_subplot_figure(data_list, titles, outfile):
    global_max = max(np.max(d) for d in data_list)
    vmin, vmax = 0, global_max

    n = len(data_list)
    ncols = 2
    nrows = (n + 1) // 2

    azimuth_full_edges = np.linspace(0, 360, 2 * n_az + 1)
    azimuth_full_edges_rad = np.radians(azimuth_full_edges)
    THETA, R = np.meshgrid(azimuth_full_edges_rad, zenith_edges, indexing='ij')

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), subplot_kw={'projection': 'polar'})
    axes = axes.ravel()

    for i, (data, title) in enumerate(zip(data_list, titles)):
        ax = axes[i]
        pcm = ax.pcolormesh(THETA, R, data, shading='auto', cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 70)
        ax.set_yticks(np.arange(0, 80, 10))
        ax.set_yticklabels([f"{d}°" for d in np.arange(0, 80, 10)])
        ax.set_rlabel_position(135)
        ax.set_title(title, fontsize=12)
        fig.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.8, pad=0.1)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, outfile), dpi=300)
    plt.close()

# --- Run simulations and build plots ---
chl_data_list, re_data_list = [], []
chl_titles, re_titles = [], []

for hspot in hspot_values:
    chl_half, re_half = simulate_brdf(hspot)
    chl_full = extend_to_full_polar(chl_half)
    re_full = extend_to_full_polar(re_half)
    tag = f"hspot = {hspot}"
    chl_data_list.append(chl_full)
    re_data_list.append(re_full)
    chl_titles.append(f"Chlorophyll 670–680 nm\n{tag}")
    re_titles.append(f"Red Edge 710–720 nm\n{tag}")

# --- Plot with dynamic vmax ---
make_subplot_figure(chl_data_list, chl_titles, "brdf_chl_dynamic_vmax.jpg")
make_subplot_figure(re_data_list, re_titles, "brdf_rededge_dynamic_vmax.jpg")

print(f"✅ Subplot JPGs saved with shared colorbars scaled to global maxima.")
