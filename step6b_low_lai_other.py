"""
This script examines how changes in Cab (chlorophyll), Cw (leaf water), Cbrown (dry matter),
and Cm (leaf structure mass) influence spectral reflectance at low LAI (0.5) using PROSAIL.
All variations are visualized in a stacked subplot figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail

# --- Spectral setup ---
wl_min, wl_max = 400, 2500
prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= wl_min) & (wl <= wl_max)
wl_used = wl[wl_mask]

# --- Base configuration ---
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02, 'LAI': 0.5,
    'psoil': 0.5, 'tts': 30, 'tto': 0, 'psi': 0,
    'hspot': 0.1, 'LIDFa': 1.0, 'LIDFb': 0.0
}

# --- Parameter variations to evaluate ---
variations = {
    "Chlorophyll (Cab)": [{'Cab': 20}, {'Cab': 70}],
    "Leaf Water (Cw)": [{'Cw': 0.01}, {'Cw': 0.08}],
    "Dead Matter (Cbrown)": [{'Cbrown': 0.0}, {'Cbrown': 0.4}],
    "Dry Matter (Cm)": [{'Cm': 0.005}, {'Cm': 0.05}]
}

# --- Plotting ---
fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

for ax, (title, param_sets) in zip(axs, variations.items()):
    for p in param_sets:
        cfg = base_config.copy()
        cfg.update(p)
        label = ", ".join(f"{k}={v}" for k, v in p.items())
        r = np.array(prosail.run(cfg))[wl_mask]
        ax.plot(wl_used, r, label=label)
    ax.set_title(title)
    ax.set_ylabel("Reflectance")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=False)

axs[-1].set_xlabel("Wavelength (nm)")

plt.suptitle("Spectral Sensitivity at Low LAI (0.5): Biochemical Parameter Variations", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/prosail_lai05_biochem_with_cm.jpg", dpi=300)
plt.show()
