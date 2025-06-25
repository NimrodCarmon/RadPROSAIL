"""
This script explores the effect of low LAI (0–1.0) on spectral reflectance across three soil
conditions (wet, medium, dry) using PROSAIL. The plots follow IEEE style, are vertically stacked,
and use a shared colorbar to encode LAI.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from prosail2 import Prosail
import matplotlib as mpl

# --- IEEE-style plotting settings ---
mpl.rcParams.update({
    "font.family": "serif",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# --- Spectral domain ---
wl_min, wl_max = 400, 2500
prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= wl_min) & (wl <= wl_max)
wl_used = wl[wl_mask]

# --- Base simulation parameters ---
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02, 'tts': 30, 'tto': 0, 'psi': 0,
    'hspot': 0.1, 'LIDFa': 1.0, 'LIDFb': 0.0
}

# --- LAI values to plot ---
lai_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
norm = Normalize(vmin=lai_values.min(), vmax=lai_values.max())
colors = viridis(norm(lai_values))

# --- psoil conditions ---
psoils = [0.0, 0.5, 1.0]
soil_titles = {0.0: "Wet Soil (psoil = 0)", 0.5: "Medium Soil (psoil = 0.5)", 1.0: "Dry Soil (psoil = 1)"}

# --- Simulate ---
results = {}
for psoil in psoils:
    spectra = []
    for lai in lai_values:
        cfg = base_config.copy()
        cfg['LAI'] = lai
        cfg['psoil'] = psoil
        r = np.array(prosail.run(cfg))[wl_mask]
        spectra.append(r)
    results[psoil] = spectra

# --- Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True, sharey=True,
                        gridspec_kw={'right': 0.85})
for ax, psoil in zip(axs, psoils):
    for r, color in zip(results[psoil], colors):
        ax.plot(wl_used, r, color=color)
    ax.set_title(soil_titles[psoil])
    ax.set_ylabel("Reflectance")

axs[2].set_xlabel("Wavelength (nm)")

# --- Shared colorbar ---
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="LAI")

plt.suptitle("Spectral Reflectance for Low LAI (0–1.0) Across Soil Moisture Levels", fontsize=13)
plt.tight_layout(rect=[0, 0, 0.87, 0.95])
plt.savefig("figures/reflectance_low_LAI_trisoil_ieee.jpg", dpi=300)
plt.show()


# --- LAI values from 1 to 5 ---
lai_values_high = np.linspace(1.0, 5.0, 6)
norm_high = Normalize(vmin=lai_values_high.min(), vmax=lai_values_high.max())
colors_high = viridis(norm_high(lai_values_high))

# --- Simulate for LAI 1–5 ---
results_high = {}
for psoil in psoils:
    spectra = []
    for lai in lai_values_high:
        cfg = base_config.copy()
        cfg['LAI'] = lai
        cfg['psoil'] = psoil
        r = np.array(prosail.run(cfg))[wl_mask]
        spectra.append(r)
    results_high[psoil] = spectra

# --- Plotting for LAI 1–5 ---
fig, axs = plt.subplots(3, 1, figsize=(6, 9), sharex=True, sharey=True,
                        gridspec_kw={'right': 0.85})
for ax, psoil in zip(axs, psoils):
    for r, color in zip(results_high[psoil], colors_high):
        ax.plot(wl_used, r, color=color)
    ax.set_title(soil_titles[psoil])
    ax.set_ylabel("Reflectance")

axs[2].set_xlabel("Wavelength (nm)")

# --- Colorbar for LAI 1–5 ---
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
sm_high = plt.cm.ScalarMappable(cmap='viridis', norm=norm_high)
sm_high.set_array([])
fig.colorbar(sm_high, cax=cbar_ax, label="LAI")

plt.suptitle("Spectral Reflectance for LAI 1–5 Across Soil Moisture Levels", fontsize=13)
plt.tight_layout(rect=[0, 0, 0.87, 0.95])
plt.savefig("figures/reflectance_high_LAI_trisoil_ieee.jpg", dpi=300)
plt.show()
