"""
Generates a large PROSAIL LUT with spectra using a single CPU core.
Accepts all outputs, then removes NaNs/Infs in a fast cleanup pass.
Suppresses RuntimeWarnings during PROSAIL evaluation.
"""

import numpy as np
from prosail2 import Prosail
import os
import warnings

# --- Config ---
n_samples_target = 1_000_000
output_dir = "lut_output"
os.makedirs(output_dir, exist_ok=True)

# --- Spectral mask (400–2500 nm) ---
prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= 400) & (wl <= 2500)
n_bands = np.sum(wl_mask)

# --- Parameter sampling function ---
def random_config(seed=None):
    rng = np.random.default_rng(seed)
    return {
        'N': rng.uniform(1.0, 2.5),
        'Cab': rng.uniform(10, 80),
        'Car': rng.uniform(2, 20),
        'Cbrown': rng.uniform(0.0, 0.5),
        'Cw': rng.uniform(0.005, 0.05),
        'Cm': rng.uniform(0.005, 0.05),
        'LAI': rng.uniform(0.1, 6.0),
        'psoil': rng.uniform(0.0, 1.0),
        'tts': rng.uniform(0, 65),
        'tto': rng.uniform(0, 35),
        'psi': rng.uniform(0, 180),
        'hspot': rng.uniform(0.05, 0.5),
        'LIDFa': rng.uniform(-1, 1.0),
        'LIDFb': rng.uniform(-1, 1.0)
    }

# --- Main process ---
def main():
    print("Running with a single CPU core...")
    X_raw = []
    Y_raw = []
    rng = np.random.default_rng(42)
    prosail = Prosail()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(n_samples_target):
            cfg = random_config(seed=rng.integers(0, 1e9))
            refl = np.array(prosail.run(cfg))[wl_mask]
            X_raw.append(list(cfg.values()))
            Y_raw.append(refl)

    X_raw = np.array(X_raw)
    Y_raw = np.array(Y_raw)

    # --- Filter NaN/Inf ---
    valid_mask = ~np.any(np.isnan(Y_raw) | np.isinf(Y_raw), axis=1)
    X = X_raw[valid_mask]
    Y = Y_raw[valid_mask]

    print(f"Generated {X.shape[0]} clean spectra with {X.shape[1]} inputs and {Y.shape[1]} bands")

    # --- Save ---
    np.save(os.path.join(output_dir, "X_inputs.npy"), X)
    np.save(os.path.join(output_dir, "Y_reflectance.npy"), Y)
    print(f"✅ Saved to: {output_dir}/X_inputs.npy and Y_reflectance.npy")

if __name__ == "__main__":
    main()
