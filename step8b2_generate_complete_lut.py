"""
Generates a large PROSAIL LUT with spectra using a single CPU core.
Accepts all outputs, then removes NaNs/Infs and values outside [0,1] reflectance.
Suppresses RuntimeWarnings during PROSAIL evaluation.
"""

import numpy as np
from prosail2 import Prosail
import os
import warnings
from tqdm import tqdm  # Added tqdm for progress bar

# --- Config ---
n_samples_target = 1_500_000
output_dir = "lut_output_filtered"
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
        'Cbrown': rng.uniform(0.0, 0.75),
        'Cw': rng.uniform(0.002, 0.08),
        'Cm': rng.uniform(0.002, 0.1),
        'LAI': rng.uniform(0.1, 10),
        'psoil': rng.uniform(0.0, 1.0),
        'tts': rng.uniform(0, 65),
        'tto': rng.uniform(0, 35),
        'psi': rng.uniform(0, 180),
        'hspot': rng.uniform(0.05, 0.75),
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
        for _ in tqdm(range(n_samples_target), desc="Generating spectra"):
            cfg = random_config(seed=rng.integers(0, 1e9))
            refl = np.array(prosail.run(cfg))[wl_mask]
            X_raw.append(list(cfg.values()))
            Y_raw.append(refl)

    X_raw = np.array(X_raw)
    Y_raw = np.array(Y_raw)

    # --- Filter NaN/Inf and out-of-range reflectance ---
    valid_mask = ~np.any(np.isnan(Y_raw) | np.isinf(Y_raw) | (Y_raw < 0) | (Y_raw > 1), axis=1)
    X = X_raw[valid_mask]
    Y = Y_raw[valid_mask]

    print(f"Generated {X.shape[0]} clean spectra with {X.shape[1]} inputs and {Y.shape[1]} bands")

    # --- Save ---
    np.save(os.path.join(output_dir, "X_inputs_filtered.npy"), X)
    np.save(os.path.join(output_dir, "Y_reflectance_filtered.npy"), Y)
    print(f"✅ Saved to: {output_dir}/X_inputs_filtered.npy and Y_reflectance_filtered.npy")

if __name__ == "__main__":
    main()
