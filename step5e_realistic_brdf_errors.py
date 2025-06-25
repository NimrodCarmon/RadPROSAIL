"""
This script compares spectral reflectance profiles from PROSAIL simulations using two LIDF types
(planophile and erectophile) over constrained viewing geometries. It calculates Z-score RMSEs,
plots a heatmap of the angular sensitivity (limited to 0–25° VZA), includes spectral profile
comparisons for tto=0° and tto=15°, and appends a BRDF polar plot (erectophile, hspot=0.3) showing
reflectance in the red-edge band (~720 nm).
"""

import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
import os

# --- Output directory ---
outdir = "figures/crop_analysis"
os.makedirs(outdir, exist_ok=True)

# --- Fixed PROSAIL biochemical + solar settings ---
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02, 'LAI': 3.0,
    'psoil': 0.2, 'tts': 30,
    'hspot': 0.0
}

# --- LIDF configurations ---
lidf_types = {
    'planophile':   {'LIDFa':  1.0,  'LIDFb': 0.0},
    'erectophile':  {'LIDFa': -1.0,  'LIDFb': 0.0},
}
lidf_pair = ('planophile', 'erectophile')

# --- Constrained geometry domain ---
tto_list = np.linspace(0, 40, 10)
psi_list = np.linspace(0, 180, 13)
tto_grid, psi_grid = np.meshgrid(tto_list, psi_list, indexing='ij')

# --- Spectral setup ---
wl_min, wl_max = 400, 2500
prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= wl_min) & (wl <= wl_max)
wl_used = wl[wl_mask]

# --- Red edge index ---
red_edge_wl = 720
red_edge_idx = np.argmin(np.abs(wl - red_edge_wl))

def zscore(spec):
    return (spec - np.mean(spec)) / np.std(spec)

def compute_rmse_grid():
    rmse_grid = np.zeros(tto_grid.shape)
    for i in range(tto_grid.shape[0]):
        for j in range(tto_grid.shape[1]):
            tto = tto_grid[i, j]
            psi = psi_grid[i, j]

            cfg1 = base_config.copy()
            cfg1.update(lidf_types[lidf_pair[0]])
            cfg1['tto'] = tto
            cfg1['psi'] = psi
            r1 = np.array(prosail.run(cfg1))[wl_mask]
            z1 = zscore(r1)

            cfg2 = base_config.copy()
            cfg2.update(lidf_types[lidf_pair[1]])
            cfg2['tto'] = tto
            cfg2['psi'] = psi
            r2 = np.array(prosail.run(cfg2))[wl_mask]
            z2 = zscore(r2)

            rmse = np.sqrt(np.mean((z1 - z2) ** 2))
            rmse_grid[i, j] = rmse

    return rmse_grid

def simulate_brdf(hspot, lidf_key):
    n_az = 100
    n_vz = 50
    azimuth_half_edges = np.linspace(0, 180, n_az + 1)
    zenith_edges = np.linspace(0, 70, n_vz + 1)
    azimuth_centers = 0.5 * (azimuth_half_edges[:-1] + azimuth_half_edges[1:])
    zenith_centers = 0.5 * (zenith_edges[:-1] + zenith_edges[1:])
    AZ_C, VZ_C = np.meshgrid(azimuth_centers, zenith_centers, indexing='ij')

    refl_half = np.zeros_like(AZ_C)
    for i in range(AZ_C.shape[0]):
        for j in range(AZ_C.shape[1]):
            config = base_config.copy()
            config.update(lidf_types[lidf_key])
            config['hspot'] = hspot
            config['psi'] = AZ_C[i, j]
            config['tto'] = VZ_C[i, j]
            r = np.array(prosail.run(config))
            refl_half[i, j] = r[red_edge_idx]

    refl_full = np.vstack([refl_half, np.flipud(refl_half)])
    azimuth_full_edges = np.linspace(0, 360, 2 * n_az + 1)
    azimuth_full_edges_rad = np.radians(azimuth_full_edges)
    THETA, R = np.meshgrid(azimuth_full_edges_rad, zenith_edges, indexing='ij')

    return THETA, R, refl_full

def plot_combined(rmse_grid, theta, r, brdf_data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [1.2, 1]})

    # --- RMSE Heatmap ---
    ax1 = axs[0, 0]
    pcm = ax1.pcolormesh(psi_list, tto_list, rmse_grid, shading='auto', cmap='inferno')
    ax1.set_xlabel("Relative Azimuth (ψ°)")
    ax1.set_ylabel("View Zenith (tto°)")
    ax1.set_title("Z-score RMSE: Planophile vs Erectophile")
    ax1.set_ylim(0, 25)
    fig.colorbar(pcm, ax=ax1, label="Z-RMSE")

    # --- Polar BRDF plot (Upper Right) ---
    ax_polar = fig.add_subplot(2, 2, 2, polar=True)
    pcm = ax_polar.pcolormesh(theta, r, brdf_data, shading='auto', cmap='viridis')
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.set_ylim(0, 70)
    ax_polar.set_yticks(np.arange(0, 80, 10))
    ax_polar.set_yticklabels([f"{d}°" for d in np.arange(0, 80, 10)])
    ax_polar.set_rlabel_position(135)
    ax_polar.set_title("BRDF @ Red Edge (~720nm)", va='bottom')
    fig.colorbar(pcm, ax=ax_polar, orientation='vertical', label='Reflectance')

    # Remove axes box
    for spine in ax_polar.spines.values():
        spine.set_visible(False)

    ax_polar.text(0.5, -0.15, "LIDF: Erectophile\nhspot=0.3", transform=ax_polar.transAxes,
                  ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    # --- Spectral Comparisons ---
    view_angles = [0, 15]
    for ax, tto in zip(axs[1][:2], view_angles):
        psi = 90  # Representative cross-plane
        cfg1 = base_config.copy()
        cfg1.update(lidf_types['planophile'])
        cfg1['tto'] = tto
        cfg1['psi'] = psi
        r1 = np.array(prosail.run(cfg1))[wl_mask]
        z1 = zscore(r1)

        cfg2 = base_config.copy()
        cfg2.update(lidf_types['erectophile'])
        cfg2['tto'] = tto
        cfg2['psi'] = psi
        r2 = np.array(prosail.run(cfg2))[wl_mask]
        z2 = zscore(r2)

        rmse = np.sqrt(np.mean((z1 - z2) ** 2))

        ax.plot(wl_used, z1, label='Planophile')
        ax.plot(wl_used, z2, label='Erectophile')
        ax.set_title(f"tto={tto}°, ψ=90°, RMSE={rmse:.3f}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Z-score Reflectance")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "shape_sensitivity_constrained.jpg"), dpi=300)
    plt.show()

# --- Run ---
rmse_grid = compute_rmse_grid()
theta, r, brdf_data = simulate_brdf(hspot=0.3, lidf_key='erectophile')
plot_combined(rmse_grid, theta, r, brdf_data)

print(f"✅ Saved: shape_sensitivity_constrained.jpg")
