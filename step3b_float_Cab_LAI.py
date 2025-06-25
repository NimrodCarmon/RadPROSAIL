"""
=============================================================
NDVI Spectrum Inversion: Joint LAI & Cab Fit with Debugging
=============================================================

This script estimates LAI and Cab from NDVI_*.csv reflectance spectra 
using PROSAIL, by fitting both parameters for each eSZA. It provides:

1. Reading and preprocessing of all NDVI_*.csv files in ../data/
2. For each spectrum and each eSZA:
   - Interpolates observed spectrum to PROSAIL grid
   - Defines and minimizes a cost function (MSE) between observed and PROSAIL-simulated spectra
   - Both LAI and Cab are optimized (floated) simultaneously
   - Stores optimizer's trial parameter values and model spectra for a user-chosen debug case
3. Results: for all spectra and all eSZA, a scatter plot of estimated (Cab, LAI)
4. Debugging: for one chosen spectrum/eSZA, a plot showing the fit evolution and best-fit spectrum

Written by: [Your Name]
Date: [Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
from scipy.optimize import minimize
import os
import glob

# === DEBUGGING PARAMETERS ===
DEBUG_SINGLE_SPECTRUM = True   # Set to True to enable debug plots
DEBUG_NDVI_INDEX = 0           # Index of NDVI file to debug (0 = first file)
DEBUG_ESZA_INDEX = 0           # Index of eSZA column to debug (0 = first angle)

# === PROSAIL SETUP ===
prosail = Prosail()
base_config = {
    'N': 1.5,        # Structure parameter
    'Cab': 45,       # Chlorophyll content (will be optimized)
    'Car': 8,        # Carotenoids (fixed)
    'Cbrown': 0.2,   # Brown pigment (fixed)
    'Cw': 0.03,      # Water thickness (will be optimized)
    'Cm': 0.02,      # Dry matter (fixed)
    'psoil': 0.2,    # Soil brightness (fixed)
    'hspot': 0.1,    # Hotspot (fixed)
    'tts': 40,       # Solar zenith (deg) -- set to 40 to match 'truth'
    'tto': 30,       # Observer zenith (fixed)
    'psi': 60,       # Azimuth (fixed)
    'LIDFa': -1,     # LAD param a
    'LIDFb': 0       # LAD param b
}
wl_prosail = np.array(prosail.wl)
wl_mask = (wl_prosail >= 350) & (wl_prosail <= 1250)

# === FILE I/O ===
indir = '../data'
ndvi_files = sorted(glob.glob(os.path.join(indir, 'NDVI_*.csv')))
if not ndvi_files:
    raise RuntimeError("No NDVI_*.csv files found in ../data/.")

# === STORAGE FOR RESULTS AND DEBUGGING ===
all_lai, all_cab, all_esza, all_specs = [], [], [], []

debug_iteration_params = []
debug_iteration_spectra = []
debug_obs_interp = None
debug_best_params = None

# === MAIN LOOP: Process all NDVI files and all eSZA columns ===
for file_idx, file in enumerate(ndvi_files):
    df = pd.read_csv(file)
    spec_name = os.path.splitext(os.path.basename(file))[0]
    wavelengths = df['wavelength'].values
    wl_subset_mask = (wavelengths >= 350) & (wavelengths <= 1250)
    esza_cols = [c for c in df.columns if c.startswith('eSZA_')]

    for col_idx, col in enumerate(esza_cols):
        obs_spectrum = df[col].values
        obs_subset = obs_spectrum[wl_subset_mask]
        if np.any(np.isnan(obs_subset)):
            continue

        # Interpolate the observed spectrum to PROSAIL grid for direct comparison
        obs_interp = np.interp(wl_prosail[wl_mask], wavelengths[wl_subset_mask], obs_subset)

        # --- For debugging, monitor optimizer iterations ---
        iteration_params = []
        iteration_spectra = []

        # Cost function: mean squared error between obs and model for LAI & Cab
        def cost(params):
            lai, cab = params
            config = base_config.copy()
            config['LAI'] = float(lai)
            config['Cab'] = float(cab)
            sim = np.array(prosail.run(config))
            sim_subset = sim[wl_mask]
            # For the selected debug spectrum, record parameter trial and spectrum
            if (DEBUG_SINGLE_SPECTRUM and 
                file_idx == DEBUG_NDVI_INDEX and col_idx == DEBUG_ESZA_INDEX):
                iteration_params.append((lai, cab))
                iteration_spectra.append(sim_subset.copy())
            return np.mean((obs_interp - sim_subset) ** 2)

        # Set bounds: LAI (0.1–8), Cab (10–80)
        bounds = [(0.1, 8.0), (10.0, 80.0)]

        # Initial guess: LAI=2, Cab=45
        res = minimize(
            cost,
            x0=[2.0, 45.0],
            bounds=bounds,
            method='L-BFGS-B',
            options={'eps': 0.5, 'maxiter': 80}
        )
        best_lai, best_cab = res.x
        if not res.success:
            print(f"WARNING: Optimization failed for {spec_name} {col}. Falling back to grid search.")
            lai_grid = np.linspace(0.1, 8.0, 20)
            cab_grid = np.linspace(10.0, 80.0, 10)
            mesh_lai, mesh_cab = np.meshgrid(lai_grid, cab_grid)
            params_grid = np.column_stack([mesh_lai.ravel(), mesh_cab.ravel()])
            costs = np.array([cost(params) for params in params_grid])
            best_idx = np.argmin(costs)
            best_lai, best_cab = params_grid[best_idx]
        all_lai.append(best_lai)
        all_cab.append(best_cab)
        all_esza.append(int(col.split('_')[1]))
        all_specs.append(spec_name)

        # Save debug info for the spectrum we're tracking
        if (DEBUG_SINGLE_SPECTRUM and 
            file_idx == DEBUG_NDVI_INDEX and col_idx == DEBUG_ESZA_INDEX):
            debug_iteration_params = iteration_params
            debug_iteration_spectra = iteration_spectra
            debug_obs_interp = obs_interp
            debug_best_params = (best_lai, best_cab)

# === SCATTER PLOT: Estimated Cab (x) vs LAI (y) for all spectra/eSZA ===
fig, ax = plt.subplots(figsize=(7,7))
sc = ax.scatter(all_cab, all_lai, c=all_esza, cmap='viridis', s=30, alpha=0.8)
plt.xlabel('Estimated Cab')
plt.ylabel('Estimated LAI')
plt.title('Estimated LAI vs. Cab for All NDVI Spectra/eSZA')
plt.xlim(10, 80)
plt.ylim(0, 9)
plt.colorbar(sc, label='eSZA (deg)')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('lai_vs_cab_scatter.jpg', dpi=300)
plt.show()
print("✅ Scatter plot saved as lai_vs_cab_scatter.jpg")

# === DEBUGGING PLOT: Show fitting progress for a single spectrum/eSZA ===
if DEBUG_SINGLE_SPECTRUM and debug_obs_interp is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(wl_prosail[wl_mask], debug_obs_interp, label='Observed Spectrum', color='black', linewidth=2)
    nplot = min(15, len(debug_iteration_params))
    for i in np.linspace(0, len(debug_iteration_params)-1, nplot, dtype=int):
        lai_i, cab_i = debug_iteration_params[i]
        plt.plot(
            wl_prosail[wl_mask], 
            debug_iteration_spectra[i], 
            label=f'Sim LAI={lai_i:.2f}, Cab={cab_i:.1f}', 
            alpha=0.45
        )
    # Plot the best-fit spectrum
    config = base_config.copy()
    config['LAI'], config['Cab'] = debug_best_params
    bestfit = np.array(prosail.run(config))[wl_mask]
    plt.plot(wl_prosail[wl_mask], bestfit, color='red', linestyle='--', linewidth=2, 
             label=f'Best fit LAI={debug_best_params[0]:.2f}, Cab={debug_best_params[1]:.1f}')
    plt.title(f"DEBUG: Fit to Observed Spectrum\nFile: {ndvi_files[DEBUG_NDVI_INDEX]}, eSZA_{DEBUG_ESZA_INDEX+1}")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig('debug_optimizer_fit_single_spectrum.jpg', dpi=250)
    plt.show()
    print("✅ Debug plot of optimizer steps saved as debug_optimizer_fit_single_spectrum.jpg")
