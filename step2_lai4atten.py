import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
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
wl_prosail = np.array(prosail.wl)  # Convert to np.array for masking
wl_mask = (wl_prosail >= 350) & (wl_prosail <= 1250)

# --- Parameters ---
lai_values = np.linspace(0.1, 8.0, 100)
indir = './attenuated_reflectance_spectra'
outdir = './figures'
os.makedirs(outdir, exist_ok=True)
csv_files = sorted(glob.glob(os.path.join(indir, 'atmos_attenuated_reflectance_*.csv')))

# --- Process each file ---
results = []

for file in csv_files:
    df = pd.read_csv(file)
    spec_name = os.path.basename(file).replace('atmos_attenuated_reflectance_', '').replace('.csv', '')
    wavelengths = df.columns[1:].astype(float)
    wl_subset_mask = (wavelengths >= 350) & (wavelengths <= 1250)

    for _, row in df.iterrows():
        angle = row['eSZA (deg)']
        obs_spectrum = row.iloc[1:].values
        obs_subset = obs_spectrum[wl_subset_mask]

        if np.any(np.isnan(obs_subset)):
            continue

        best_lai = None
        best_rmse = np.inf

        for lai in lai_values:
            config = base_config.copy()
            config['LAI'] = lai
            sim = prosail.run(config)
            sim = np.array(sim)  # Ensure it's a NumPy array
            sim_subset = sim[wl_mask]
            rmse = np.sqrt(np.mean((obs_subset - sim_subset) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_lai = lai

        results.append({'Spectrum': spec_name, 'eSZA': angle, 'Estimated_LAI': best_lai, 'RMSE': best_rmse})
        print(f"{spec_name} @ eSZA={angle:.1f}° → LAI={best_lai:.2f}, RMSE={best_rmse:.4f}")

# --- Results ---
results_df = pd.DataFrame(results)
import pdb; pdb.set_trace()
results_df.to_csv(os.path.join(outdir, 'lai_estimates_all_angles.csv'), index=False)

# --- Plot per spectrum ---
for spec in results_df['Spectrum'].unique():
    sub = results_df[results_df['Spectrum'] == spec]

    # LAI vs eSZA
    plt.figure(figsize=(7, 4))
    plt.plot(sub['eSZA'], sub['Estimated_LAI'], marker='o')
    plt.title(f'Estimated LAI vs eSZA — {spec}')
    plt.xlabel('Effective Solar Zenith Angle (°)')
    plt.ylabel('Estimated LAI')
    plt.ylim(0, 9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{outdir}/lai_vs_esza_{spec}.jpg', dpi=300)
    plt.close()

    # RMSE vs eSZA
    plt.figure(figsize=(7, 4))
    plt.plot(sub['eSZA'], sub['RMSE'], marker='o', color='red')
    plt.title(f'RMSE vs eSZA — {spec}')
    plt.xlabel('Effective Solar Zenith Angle (°)')
    plt.ylabel('RMSE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{outdir}/rmse_vs_esza_{spec}.jpg', dpi=300)
    plt.close()

print("\n✅ LAI inversion completed for all spectra and angles. Results saved to:", outdir)
