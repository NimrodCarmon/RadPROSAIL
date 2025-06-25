import numpy as np
import matplotlib.pyplot as plt
import os
from prosail2 import Prosail

# Create output directory
output_dir = 'figures/reflectance_comparison'
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

# Solar zenith angles to test
sza_list = [0, 15, 30, 45, 60, 75]

# Containers for directional reflectance
resv_canref_list = []
resv_lut_list = []

# Initialize models explicitly
prosail_canref = Prosail(use_lut_canopy_model=False)
prosail_lut = Prosail(use_lut_canopy_model=True)

for tts in sza_list:
    cfg['tts'] = tts

    # Original implementation reflectance
    resv_canref = prosail_canref.run(cfg)
    resv_canref_list.append(resv_canref)

    # LUT-based implementation reflectance
    resv_lut = prosail_lut.run(cfg)
    resv_lut_list.append(resv_lut)

resv_canref_array = np.array(resv_canref_list)
resv_lut_array = np.array(resv_lut_list)

# Plotting reflectance comparison
plt.figure(figsize=(10, 6))
plt.plot(sza_list, resv_canref_array.mean(axis=1), '-o', label='Original canref (resv)')
plt.plot(sza_list, resv_lut_array.mean(axis=1), '-s', label='LUT-based canopy reflectance (resv)')

plt.xlabel('Solar Zenith Angle (degrees)')
plt.ylabel('Mean Directional Spectral Reflectance (resv)')
plt.title('Comparison of Directional Canopy Reflectance over SZA')
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.tight_layout()

# Save figure
plot_path = os.path.join(output_dir, 'directional_reflectance_comparison_sza.png')
plt.savefig(plot_path, dpi=300)
plt.close()
