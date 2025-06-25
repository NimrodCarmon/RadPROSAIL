import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
from scipy.optimize import least_squares
import os

# --- Setup ---
prosail = Prosail()
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02,
    'psoil': 0.2, 'hspot': 0.1, 'tts': 45,
    'tto': 30, 'psi': 60, 'LIDFa': -1, 'LIDFb': 0
}
wl = np.array(prosail.wl)
wl_mask = (wl >= 350) & (wl <= 1250)
wl_subset = wl[wl_mask]

# --- Define LAI values ---
lai_values = np.linspace(1, 8, 5)  # [1.0, 2.75, 4.5, 6.25, 8.0]

# --- Simulate spectra ---
spectra = []
param_records = []

for i, lai in enumerate(lai_values):
    config = base_config.copy()
    config['LAI'] = lai
    refl = np.array(prosail.run(config))[wl_mask]

    spectra.append(refl)
    param_records.append({'ID': f'Sim_{i+1}', **config})

# --- Save spectra and parameters ---
spectra_df = pd.DataFrame(spectra, columns=wl_subset)
spectra_df.insert(0, 'ID', [f'Sim_{i+1}' for i in range(len(lai_values))])
spectra_df.to_csv('simulated_reflectance_spectra.csv', index=False)

params_df = pd.DataFrame(param_records)
params_df.to_csv('simulation_parameters.csv', index=False)

# --- Plot spectra colored by LAI ---
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.cm.viridis
norm = plt.Normalize(min(lai_values), max(lai_values))

for refl, lai in zip(spectra, lai_values):
    ax.plot(wl_subset, refl, color=cmap(norm(lai)), linewidth=2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('LAI', rotation=270, labelpad=15)

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Reflectance')
ax.set_title('Simulated PROSAIL Spectra Colored by LAI')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('simulated_spectra_by_lai.jpg', dpi=300)
plt.close()

# --- Inversion with least squares ---
estimated_lais = []
for i, (true_lai, obs) in enumerate(zip(lai_values, spectra)):
    obs = np.array(obs)

    def residuals(lai_array):
        config = base_config.copy()
        config['LAI'] = float(lai_array[0])
        sim = np.array(prosail.run(config))[wl_mask]
        return sim - obs

    res = least_squares(
        residuals,
        x0=[2.0],
        bounds=(0.1, 8.0),
        method='trf',
        diff_step=0.5  # larger step size for better convergence
    )
    est_lai = res.x[0]
    estimated_lais.append(est_lai)

    # Plot observed vs. fitted spectrum
    fitted = np.array(prosail.run({**base_config, 'LAI': est_lai}))[wl_mask]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wl_subset, obs, label='True Spectrum', linewidth=2, color='black')
    ax.plot(wl_subset, fitted, label='Fitted Spectrum', linestyle='--', color='red')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.set_title(f'Sim_{i+1}: True LAI = {true_lai:.2f}, Estimated LAI = {est_lai:.2f}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'spectrum_fit_Sim_{i+1}.jpg', dpi=300)
    plt.close()

# --- Final Report ---
print("✅ LAI inversion and plotting complete.")
print("Estimated vs. True LAI:")
for i, (true_lai, est_lai) in enumerate(zip(lai_values, estimated_lais), 1):
    print(f"  Sim_{i}: True LAI = {true_lai:.2f}, Estimated LAI = {est_lai:.2f}")
