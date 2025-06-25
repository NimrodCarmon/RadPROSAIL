import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
from scipy.optimize import least_squares
import os
import glob

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

# --- Load NDVI files ---
indir = '../data'
ndvi_files = sorted(glob.glob(os.path.join(indir, 'NDVI_*.csv')))
if not ndvi_files:
    raise RuntimeError("No NDVI_*.csv files found in ../data/.")

# --- Find spectrum with eSZA=45° for debug plot ---
DEBUG_SINGLE_SPECTRUM = True
DEBUG_NDVI_INDEX = None
DEBUG_ESZA_INDEX = None
target_angle = 45

for file_idx, file in enumerate(ndvi_files):
    df = pd.read_csv(file)
    esza_cols = [c for c in df.columns if c.startswith('eSZA_')]
    for col_idx, col in enumerate(esza_cols):
        angle = int(col.split('_')[1])
        if angle == target_angle:
            DEBUG_NDVI_INDEX = file_idx
            DEBUG_ESZA_INDEX = col_idx
            print(f"🔍 Debug spectrum selected: '{os.path.basename(file)}' with eSZA = {angle}°")
            break
    if DEBUG_NDVI_INDEX is not None:
        break

if DEBUG_NDVI_INDEX is None:
    print(f"⚠️  No spectrum found with eSZA = {target_angle}°. Debug plot will be skipped.")
    DEBUG_SINGLE_SPECTRUM = False

# --- Storage ---
all_results = []
debug_iteration_lais = []
debug_iteration_spectra = []
debug_obs_interp = None
debug_best_lai = None

# --- Process spectra ---
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
        iteration_lais = []
        iteration_spectra = []

        def residuals(lai_array):
            config = base_config.copy()
            config['LAI'] = float(lai_array[0])
            sim = np.array(prosail.run(config))[wl_mask]
            if DEBUG_SINGLE_SPECTRUM and file_idx == DEBUG_NDVI_INDEX and col_idx == DEBUG_ESZA_INDEX:
                iteration_lais.append(float(lai_array[0]))
                iteration_spectra.append(sim.copy())
            return np.diff(sim) - np.diff(obs_interp)

        res = least_squares(
            residuals,
            x0=[2.0],
            bounds=(0.1, 8.0),
            method='trf',
            diff_step=0.5
        )
        best_lai = res.x[0]
        lai_curve.append(best_lai)

        if DEBUG_SINGLE_SPECTRUM and file_idx == DEBUG_NDVI_INDEX and col_idx == DEBUG_ESZA_INDEX:
            debug_iteration_lais = iteration_lais
            debug_iteration_spectra = iteration_spectra
            debug_obs_interp = obs_interp
            debug_best_lai = best_lai

    all_results.append({
        'spec_name': spec_name,
        'esza': [int(c.split('_')[1]) for c in esza_cols],
        'lai': lai_curve
    })

# --- Plot LAI vs. eSZA ---
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
plt.savefig('lai_vs_esza_all_spectra_leastsq.jpg', dpi=300)
plt.show()

print("✅ LAI inversion using least squares completed.")

# --- Debug: Plot optimizer steps for selected spectrum ---
if DEBUG_SINGLE_SPECTRUM and debug_obs_interp is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(wl_prosail[wl_mask], debug_obs_interp, label='Observed Spectrum', color='black', linewidth=2)

    nplot = min(15, len(debug_iteration_lais))
    for i in np.linspace(0, len(debug_iteration_lais) - 1, nplot, dtype=int):
        plt.plot(
            wl_prosail[wl_mask],
            debug_iteration_spectra[i],
            label=f'Sim LAI={debug_iteration_lais[i]:.2f}',
            alpha=0.4
        )

    config = base_config.copy()
    config['LAI'] = debug_best_lai
    best_fit = np.array(prosail.run(config))[wl_mask]
    plt.plot(wl_prosail[wl_mask], best_fit, color='red', linestyle='--', linewidth=2,
             label=f'Best fit LAI={debug_best_lai:.2f}')

    plt.title(f"DEBUG: Fit to Observed Spectrum\nFile {ndvi_files[DEBUG_NDVI_INDEX]}, eSZA_{target_angle}")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig('debug_optimizer_fit_single_spectrum_leastsq.jpg', dpi=250)
    plt.show()
    print("✅ Debug fit plot saved as debug_optimizer_fit_single_spectrum_leastsq.jpg")
