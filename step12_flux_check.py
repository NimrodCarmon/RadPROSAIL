import numpy as np
from prosail2 import Prosail

# --- Initialize PROSAIL ---
prosail = Prosail()

# --- Define single config ---
cfg = {
    'N': 1.5,
    'Cab': 50.0,
    'Car': 8.0,
    'Cbrown': 0.3,
    'Cw': 0.02,
    'Cm': 0.01,
    'LAI': 2.0,           # Moderate LAI to allow canopy structure to influence BRDF
    'psoil': 0.5,
    'tts': 60.0,          # High solar zenith angle (grazing illumination)
    'tto': 60.0,          # Observer at same angle as illumination
    'psi': 0.0,           # Aligned with sun (azimuth = 0°)
    'hspot': 0.05,        # Sharp hotspot peak (low structural randomness)
    'LIDFa': -0.5,        # Erectophile leaf distribution to enhance hotspot
    'LIDFb': 0.0
}


# --- Run simulation ---
refl = np.array(prosail.run(cfg))

# --- Output ---
print(f"✅ PROSAIL single run complete. Output shape: {refl.shape}")
print(f"Reflectance range: min={refl.min():.4f}, max={refl.max():.4f}")
