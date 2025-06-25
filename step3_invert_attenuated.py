"""
===========================================================
NDVI Spectrum LAI Inversion using PROSAIL and Optimization
===========================================================

This script estimates Leaf Area Index (LAI) from atmospherically-corrected reflectance spectra 
stored in NDVI_*.csv files, using the PROSAIL radiative transfer model and numerical optimization.

**Steps performed:**
1. Loads all NDVI_*.csv files from the '../data' directory.
2. For each spectrum and each effective solar zenith angle (eSZA) column:
   a. Extracts the observed reflectance spectrum for wavelengths 350-1250 nm.
   b. Defines a cost function: mean squared error (MSE) between observed and PROSAIL-simulated spectra.
   c. Uses scipy.optimize.minimize to estimate the LAI that minimizes the cost.
   d. If optimization fails, falls back to a grid search for robust inversion.
3. Collects and plots the LAI vs. eSZA curve for each NDVI spectrum in a subplot.
4. Saves the summary plot to disk.

**Requirements:** 
- PROSAIL2 Python package
- pandas, numpy, matplotlib, scipy
- NDVI_*.csv files with structure: 
    wavelength, eSZA_1, eSZA_2, ..., eSZA_90

Customize the script as needed for your own parameter ranges or spectral regions.

Written by: [Your Name]
Date: [Date]
"""

"""
===========================================================
NDVI Spectrum LAI Inversion using PROSAIL and Optimization
===========================================================

[...amblem as before, see previous script...]
"""
"""
===========================================================
NDVI Spectrum LAI Inversion using PROSAIL and Optimization
===========================================================

[...amblem as before, see previous script...]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
from scipy.optimize import minimize
import os
import glob
import pdb

# --- Debug: Set this to True to plot fitting details for a single spectrum ---
DEBUG_SINGLE_SPECTRUM = True
DEBUG_NDVI_INDEX = 0    # Which NDVI file to debug
DEBUG_ESZA_INDEX = 0    # Which eSZA column to debug

# --- PROSAIL setup ---
prosail = Prosail()
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02,
    'psoil': 0.2, 'hspot': 0.1, 'tts': 45,
    'tto': 30, 'psi': 60, 'LIDFa': -1, 'LIDFb': 0
}
wl_prosail = np.array(prosail.wl)
wl_mask = (wl_prosail >= 350) & (wl_prosail <= 1250)

indir = '../data'
ndvi_files = sorted(glob.glob(os.path.join(indir, 'NDVI_*.csv')))
if not ndvi_files:
    raise RuntimeError("No NDVI_*.csv files found in ../data/.")

all_results = []

# --- Storage for debug info ---
debug_iteration_lais = []
debug_iteration_spectra = []
debug_obs_interp = None
debug_best_lai = None

for file_idx, file in enumerate(ndvi_files):
    
    df = pd.read_csv(file)
    spec_name = os.path.splitext(os.path.basename(file))[0]
    wavelengths = df['wavelength'].values
    wl_subset_mask = (wavelengths >= 350) & (wavelengths <= 1250)
    esza_cols = [c for c in df.columns if c.startswith('eSZA_')]
    lai_curve = []

    for col_idx, col in enumerate(esza_cols):
        obs_spectrum = df[col].values
        obs_subset = obs_spectrum[wl_subset_mask]
        if np.any(np.isnan(obs_subset)):
            lai_curve.append(np.nan)
            continue

        obs_interp = np.interp(wl_prosail[wl_mask], wavelengths[wl_subset_mask], obs_subset)

        # --- For debugging, monitor optimizer iterations ---
        iteration_lais = []
        iteration_spectra = []

        def cost(lai):
            config = base_config.copy()
            config['LAI'] = float(lai[0])
            sim = np.array(prosail.run(config))
            sim_subset = sim[wl_mask]
            # For the selected debug spectrum, store each LAI tried and its spectrum
            if (DEBUG_SINGLE_SPECTRUM and 
                file_idx == DEBUG_NDVI_INDEX and col_idx == DEBUG_ESZA_INDEX):
                iteration_lais.append(float(lai[0]))
                iteration_spectra.append(sim_subset.copy())
            return np.mean((obs_interp - sim_subset) ** 2)

        # Print a cost landscape for reference
        if col_idx == 0:
            test_lais = [0.1, 1.0, 2.0, 4.0, 8.0]
            print(f"\nDEBUG: {spec_name} {col}:")
            for t_lai in test_lais:
                val = cost([t_lai])
                print(f"  LAI={t_lai:.2f} -> cost={val:.5f}")

        # --- Optimize LAI ---
        res = minimize(
            cost,
            x0=[2.0],
            bounds=[(0.1, 8.0)],
            method='L-BFGS-B',
            options={'eps':0.2, 'maxiter':50}
        )
        best_lai = res.x[0]

        # Fallback to grid search if needed
        if not res.success:
            print(f"WARNING: Optimization failed for {spec_name} {col}. Falling back to grid search.")
            lai_grid = np.linspace(0.1, 8.0, 50)
            costs = [cost([l]) for l in lai_grid]
            best_lai = lai_grid[np.argmin(costs)]
        lai_curve.append(best_lai)

        # If debugging this spectrum, store info for later plotting
        if (DEBUG_SINGLE_SPECTRUM and 
            file_idx == DEBUG_NDVI_INDEX and col_idx == DEBUG_ESZA_INDEX):
            debug_iteration_lais = iteration_lais
            debug_iteration_spectra = iteration_spectra
            debug_obs_interp = obs_interp
            debug_best_lai = best_lai

    all_results.append({
        'spec_name': spec_name,
        'esza': [int(c.split('_')[1]) for c in esza_cols],
        'lai': lai_curve
    })

# --- Step 7: Plotting LAI vs. eSZA for All Spectra ---
num_specs = len(all_results)
fig, axes = plt.subplots(1, num_specs, figsize=(6 * num_specs, 4), sharey=True)
if num_specs == 1:
    axes = [axes]

for i, result in enumerate(all_results):
    axes[i].plot(result['esza'], result['lai'], marker='o')
    axes[i].set_title(result['spec_name'])
    axes[i].set_xlabel('Effective SZA (deg)')
    axes[i].set_ylabel('Estimated LAI')
    axes[i].set_ylim(0, 9)
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('lai_vs_esza_all_spectra_mle.jpg', dpi=300)
plt.show()

print("✅ LAI estimation and plotting completed for all NDVI spectra.")

# --- Debug: Visualize optimizer steps for a single spectrum ---
if DEBUG_SINGLE_SPECTRUM and debug_obs_interp is not None:
    plt.figure(figsize=(8, 5))
    # Plot observed measurement
    plt.plot(wl_prosail[wl_mask], debug_obs_interp, label='Observed Spectrum', color='black', linewidth=2)

    # Plot simulated spectra from several optimizer steps (up to 15 for clarity)
    nplot = min(15, len(debug_iteration_lais))
    for i in np.linspace(0, len(debug_iteration_lais)-1, nplot, dtype=int):
        plt.plot(
            wl_prosail[wl_mask], 
            debug_iteration_spectra[i], 
            label=f'Sim LAI={debug_iteration_lais[i]:.2f}', 
            alpha=0.4
        )

    # Plot the best-fit spectrum
    config = base_config.copy()
    config['LAI'] = debug_best_lai
    bestfit = np.array(prosail.run(config))[wl_mask]
    plt.plot(wl_prosail[wl_mask], bestfit, color='red', linestyle='--', linewidth=2, label=f'Best fit LAI={debug_best_lai:.2f}')

    plt.title(f"DEBUG: Fit to Observed Spectrum\nFile {ndvi_files[DEBUG_NDVI_INDEX]}, eSZA_{DEBUG_ESZA_INDEX+1}")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig('debug_optimizer_fit_single_spectrum.jpg', dpi=250)
    plt.show()
    print("✅ Debug plot of optimizer steps saved as debug_optimizer_fit_single_spectrum.jpg")
