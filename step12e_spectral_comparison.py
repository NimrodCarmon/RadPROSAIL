import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from prosail2 import Prosail

# Output directory
output_dir = 'figures/reflectance_fractional_error'
os.makedirs(output_dir, exist_ok=True)

# Vegetation configuration
cfg = {
    'N': 1.5,
    'Cab': 50.0,
    'Car': 8.0,
    'Cbrown': 0.3,
    'Cw': 0.02,
    'Cm': 0.01,
    'LAI': 2.0,
    'psoil': 0.5,
    'tto': 30.0,
    'psi': 0.0,
    'hspot': 0.05,
    'LIDFa': -0.5,
    'LIDFb': 0.0
}

# Wavelengths (400–2500 nm)
wavelengths = np.arange(400, 2501)

# SZAs (0 to 45, step 5)
sza_list = np.arange(0, 50, 5)

# Initialize PROSAIL models
prosail_canref = Prosail(use_lut_canopy_model=False)
prosail_lut = Prosail(use_lut_canopy_model=True)

fractional_errors = []

# Compute fractional errors
for tts in sza_list:
    cfg['tts'] = tts
    resv_canref = prosail_canref.run(cfg)
    resv_lut = prosail_lut.run(cfg)
    frac_error = (resv_canref - resv_lut) / resv_lut
    fractional_errors.append(frac_error)

fractional_errors = np.array(fractional_errors)

# Plot with colormap
plt.figure(figsize=(12, 6))
colors = cm.viridis(np.linspace(0, 1, len(sza_list)))

for i, (tts, color) in enumerate(zip(sza_list, colors)):
    plt.plot(wavelengths, fractional_errors[i], color=color, label=f'SZA={tts}°')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Fractional Reflectance Error')
plt.title('Fractional Reflectance Errors (Original vs. LUT-based)')
plt.grid(alpha=0.4, linestyle='--')
plt.xlim(400, 2500)

# Colorbar indicating SZA
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=sza_list.min(), vmax=sza_list.max()))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Solar Zenith Angle (degrees)')

plt.tight_layout()

# Save plot
plot_path = os.path.join(output_dir, 'fractional_reflectance_errors_vs_wavelength_sza_colormap.png')
plt.savefig(plot_path, dpi=300)
plt.close()
