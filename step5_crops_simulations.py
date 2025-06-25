import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
import os

# --- Output directory ---
outdir = "figures/crop_analysis"
os.makedirs(outdir, exist_ok=True)

# --- Common PROSAIL configuration ---
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02, 'LAI': 3.0,
    'psoil': 0.2, 'hspot': 0.1, 'tts': 45,
    'tto': 30, 'psi': 60
}

# --- Crop-specific leaf angle distributions ---
crop_lidf = {
    'Wheat (Erectophile)':       {'LIDFa': -1.0,  'LIDFb': 0.0},
    'Soybean (Planophile)':      {'LIDFa':  1.0,  'LIDFb': 0.0},
    'Maize (Plagiophile)':       {'LIDFa':  0.0,  'LIDFb': -1.0},
    'Rice (Erectophile)':        {'LIDFa': -0.7,  'LIDFb': 0.0},
    'Sunflower (Extremophile)':  {'LIDFa':  0.0,  'LIDFb': 1.0},
    'Spherical (Generic)':       {'LIDFa': -0.35, 'LIDFb': -0.15}
}

# --- Simulate reflectance spectra ---
prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= 400) & (wl <= 2500)
wl = wl[wl_mask]

spectra = {}
for crop, lidf in crop_lidf.items():
    config = base_config.copy()
    config.update(lidf)
    reflectance = np.array(prosail.run(config))[wl_mask]
    spectra[crop] = reflectance

# --- Plot ---
plt.figure(figsize=(10, 6))
for label, refl in spectra.items():
    plt.plot(wl, refl, label=label)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.title('Reflectance Sensitivity to LIDF Across Crop Types', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=9, loc='upper right', frameon=True)
plt.tight_layout()

# --- Save ---
outfile = os.path.join(outdir, "lidf_crop_reflectance_comparison.jpg")
plt.savefig(outfile, dpi=300)
plt.show()
print(f"✅ Plot saved to {outfile}")

# --- Z-score normalization across wavelengths for each spectrum ---
spectra_zscore = {
    crop: (refl - np.mean(refl)) / np.std(refl)
    for crop, refl in spectra.items()
}

# --- Plot Z-score normalized spectra ---
plt.figure(figsize=(10, 6))
for label, norm_refl in spectra_zscore.items():
    plt.plot(wl, norm_refl, label=label)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Z-score Normalized Reflectance', fontsize=12)
plt.title('Spectral Shape Comparison (Z-score Normalized)', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=9, loc='upper right', frameon=True)
plt.tight_layout()

# --- Save ---
zscore_outfile = os.path.join(outdir, "lidf_crop_reflectance_zscore.jpg")
plt.savefig(zscore_outfile, dpi=300)
plt.show()
print(f"✅ Z-score plot saved to {zscore_outfile}")


# --- N parameter sensitivity demonstration ---
N_values = [1.0, 1.5, 2.0, 2.5, 3.0]  # Low to high structure
N_spectra = {}

for N_val in N_values:
    config = base_config.copy()
    config.update({'N': N_val, 'LIDFa': -0.35, 'LIDFb': -0.15})  # Keep LIDF fixed to spherical
    refl = np.array(prosail.run(config))[wl_mask]
    N_spectra[f"N = {N_val}"] = refl

# --- Plot N variation ---
plt.figure(figsize=(10, 6))
for label, refl in N_spectra.items():
    plt.plot(wl, refl, label=label)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.title('Reflectance Sensitivity to Leaf Structure Parameter (N)', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=9, loc='upper right', frameon=True)
plt.tight_layout()

# --- Save ---
outfile_N = os.path.join(outdir, "N_parameter_reflectance_comparison.jpg")
plt.savefig(outfile_N, dpi=300)
plt.show()
print(f"✅ N-variation plot saved to {outfile_N}")



# --- Z-score normalization across wavelengths for each N-spectrum ---
N_spectra_zscore = {
    label: (refl - np.mean(refl)) / np.std(refl)
    for label, refl in N_spectra.items()
}

# --- Plot Z-score normalized N variations ---
plt.figure(figsize=(10, 6))
for label, norm_refl in N_spectra_zscore.items():
    plt.plot(wl, norm_refl, label=label)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Z-score Normalized Reflectance', fontsize=12)
plt.title('Spectral Shape Sensitivity to Leaf Structure Parameter (N)', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=9, loc='upper right', frameon=True)
plt.tight_layout()

# --- Save ---
outfile_N_zscore = os.path.join(outdir, "N_parameter_reflectance_zscore.jpg")
plt.savefig(outfile_N_zscore, dpi=300)
plt.show()
print(f"✅ Z-score N-variation plot saved to {outfile_N_zscore}")


import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
import os

# --- Setup ---
outdir = "figures/crop_analysis"
os.makedirs(outdir, exist_ok=True)

# --- Base config (fixed leaf/biochemical properties) ---
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02, 'LAI': 3.0,
    'psoil': 0.2, 'LIDFa': -0.35, 'LIDFb': -0.15,
    'tts': 45, 'tto': 45, 'psi': 0  # Backscatter geometry
}

# --- hspot values to test ---
hspot_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
spectra_hspot = {}

prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= 400) & (wl <= 2500)
wl = wl[wl_mask]

# --- Simulate for each hspot ---
for h in hspot_values:
    config = base_config.copy()
    config['hspot'] = h
    refl = np.array(prosail.run(config))[wl_mask]
    spectra_hspot[f"hspot = {h}"] = refl

# --- Plot absolute reflectance ---
plt.figure(figsize=(10, 6))
for label, refl in spectra_hspot.items():
    plt.plot(wl, refl, label=label)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectance', fontsize=12)
plt.title('Reflectance Sensitivity to Hotspot Parameter (hspot)', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=9, loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "hspot_reflectance_comparison.jpg"), dpi=300)
plt.show()

# --- Z-score normalization ---
spectra_hspot_zscore = {
    label: (refl - np.mean(refl)) / np.std(refl)
    for label, refl in spectra_hspot.items()
}

# --- Plot normalized reflectance ---
plt.figure(figsize=(10, 6))
for label, refl_z in spectra_hspot_zscore.items():
    plt.plot(wl, refl_z, label=label)

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Z-score Normalized Reflectance', fontsize=12)
plt.title('Z-score Normalized Reflectance vs hspot Parameter', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=9, loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "hspot_reflectance_zscore.jpg"), dpi=300)
plt.show()
