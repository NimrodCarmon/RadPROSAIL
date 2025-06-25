"""
This script samples the PROSAIL parameter space and runs 10,000 simulations,
logging any runtime warnings or invalid results (e.g., NaN, Inf, or exceptions).
Used to verify numerical stability before LUT generation.
"""

import numpy as np
import warnings
from prosail2 import Prosail

# --- Spectral mask ---
wl_min, wl_max = 400, 2500
prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= wl_min) & (wl <= wl_max)

# --- Number of samples ---
n_samples = 10000
n_bands = np.sum(wl_mask)

# --- Parameter sampling ranges ---
def random_config():
    return {
        'N': np.random.uniform(1.0, 2.5),
        'Cab': np.random.uniform(10, 80),
        'Car': np.random.uniform(2, 20),
        'Cbrown': np.random.uniform(0.0, 0.5),
        'Cw': np.random.uniform(0.005, 0.05),
        'Cm': np.random.uniform(0.005, 0.05),
        'LAI': np.random.uniform(0.1, 6.0),
        'psoil': np.random.uniform(0.0, 1.0),
        'tts': np.random.uniform(0, 65),
        'tto': np.random.uniform(0, 35),
        'psi': np.random.uniform(0, 180),
        'hspot': np.random.uniform(0.05, 0.5),
        'LIDFa': np.random.uniform(-2.0, 2.0),
        'LIDFb': 0.0
    }

# --- Run and check ---
n_valid = 0
n_nan = 0
n_warn = 0
invalid_indices = []

print("Running PROSAIL parameter check...")

for i in range(n_samples):
    cfg = random_config()
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            refl = np.array(prosail.run(cfg))[wl_mask]
            if np.any(np.isnan(refl)) or np.any(np.isinf(refl)):
                n_nan += 1
                invalid_indices.append((i, "NaN/Inf", cfg))
            elif wlist:
                n_warn += 1
                invalid_indices.append((i, "Warning", cfg))
            else:
                n_valid += 1
    except Exception as e:
        n_nan += 1
        invalid_indices.append((i, f"Exception: {str(e)}", cfg))

print(f"\nChecked {n_samples} parameter sets.")
print(f"Valid:   {n_valid}")
print(f"Warnings: {n_warn}")
print(f"NaN/Inf: {n_nan}")

if invalid_indices:
    print("\nSample issues:")
    for idx, reason, cfg in invalid_indices[:5]:
        print(f"Index {idx} - {reason} - LAI={cfg['LAI']:.2f}, hspot={cfg['hspot']:.3f}, tts={cfg['tts']:.1f}, tto={cfg['tto']:.1f}")
