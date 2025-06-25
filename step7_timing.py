"""
Benchmark PROSAIL runtime and report the number of output wavelengths after spectral masking.
"""

import numpy as np
import time
from prosail2 import Prosail

# --- Spectral mask ---
wl_min, wl_max = 400, 2500
prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= wl_min) & (wl <= wl_max)
wl_used = wl[wl_mask]

# --- Report spectral resolution ---
n_wavelengths = len(wl_used)
print(f"Number of output wavelengths (400–2500 nm): {n_wavelengths}")

# --- Benchmark ---
n_samples = 1000

def random_config():
    return {
        'N': np.random.uniform(1.0, 2.5),
        'Cab': np.random.uniform(10, 80),
        'Car': np.random.uniform(5, 20),
        'Cbrown': np.random.uniform(0, 0.4),
        'Cw': np.random.uniform(0.005, 0.05),
        'Cm': np.random.uniform(0.005, 0.05),
        'LAI': np.random.uniform(0.1, 6.0),
        'psoil': np.random.uniform(0.0, 1.0),
        'tts': np.random.uniform(0, 60),
        'tto': np.random.uniform(0, 60),
        'psi': np.random.uniform(0, 180),
        'hspot': np.random.uniform(0.01, 0.5),
        'LIDFa': np.random.uniform(-2, 2),
        'LIDFb': 0.0
    }

start = time.time()
for _ in range(n_samples):
    cfg = random_config()
    spectrum = np.array(prosail.run(cfg))[wl_mask]
end = time.time()

duration = end - start
rate = n_samples / duration

print(f"Simulated {n_samples} spectra in {duration:.2f} seconds")
print(f"Rate: {rate:.1f} spectra/second on single CPU core")
