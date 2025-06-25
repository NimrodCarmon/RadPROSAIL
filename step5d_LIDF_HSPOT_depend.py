import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
from sklearn.metrics.pairwise import cosine_distances
import os

# --- Output directory ---
outdir = "figures/crop_analysis"
os.makedirs(outdir, exist_ok=True)

# --- Fixed PROSAIL biochemical + solar settings ---
base_config = {
    'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
    'Cw': 0.03, 'Cm': 0.02, 'LAI': 3.0,
    'psoil': 0.2, 'tts': 30,  # fixed sun angle
    'hspot': None,  # will set later
}

# --- LIDF configurations ---
lidf_types = {
    'planophile':   {'LIDFa':  1.0,  'LIDFb': 0.0},
    'erectophile':  {'LIDFa': -1.0,  'LIDFb': 0.0},
    'spherical':    {'LIDFa': -0.35, 'LIDFb': -0.15}
}
lidf_pairs = [('planophile', 'erectophile'), ('planophile', 'spherical'), ('erectophile', 'spherical')]

# --- Angular sampling ---
tto_list = np.linspace(0, 70, 12)       # view zenith (radial)
psi_list = np.linspace(0, 180, 13)      # relative azimuth (angular)
tto_grid, psi_grid = np.meshgrid(tto_list, psi_list, indexing='ij')

# --- Spectral settings ---
wl_min, wl_max = 400, 2500
prosail = Prosail()
wl = np.array(prosail.wl)
wl_mask = (wl >= wl_min) & (wl <= wl_max)
wl_used = wl[wl_mask]

# --- Function: Z-score normalize spectrum ---
def zscore(spec):
    return (spec - np.mean(spec)) / np.std(spec)

# --- Function: Compute shape metrics between LIDFs ---
def compute_metrics(hspot):
    shape_metrics = {}

    for (lidf1, lidf2) in lidf_pairs:
        angle_diff = np.zeros(tto_grid.shape)
        rmse_diff = np.zeros(tto_grid.shape)

        for i, tto in enumerate(tto_list):
            for j, psi in enumerate(psi_list):
                s1 = base_config.copy()
                s1.update(lidf_types[lidf1])
                s1['tto'] = tto
                s1['psi'] = psi
                s1['hspot'] = hspot
                r1 = np.array(prosail.run(s1))[wl_mask]
                z1 = zscore(r1)

                s2 = base_config.copy()
                s2.update(lidf_types[lidf2])
                s2['tto'] = tto
                s2['psi'] = psi
                s2['hspot'] = hspot
                r2 = np.array(prosail.run(s2))[wl_mask]
                z2 = zscore(r2)

                angle = cosine_distances(z1.reshape(1, -1), z2.reshape(1, -1))[0, 0]
                rmse = np.sqrt(np.mean((z1 - z2) ** 2))

                angle_diff[i, j] = angle
                rmse_diff[i, j] = rmse

        shape_metrics[(lidf1, lidf2)] = {'angle': angle_diff, 'rmse': rmse_diff}

    return shape_metrics

# --- Plot heatmaps ---
def plot_heatmap(data, title, fname, cmap='inferno'):
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.pcolormesh(psi_list, tto_list, data, shading='auto', cmap=cmap)
    ax.set_xlabel("Relative Azimuth (ψ°)")
    ax.set_ylabel("View Zenith (tto°)")
    ax.set_title(title)
    fig.colorbar(c, ax=ax, label="Difference")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close()

# --- Run for selected hspot values ---
for hspot_val in [0.0, 0.5, 1.0]:
    metrics = compute_metrics(hspot_val)

    for (lidf1, lidf2), result in metrics.items():
        tag = f"{lidf1}_vs_{lidf2}_hspot_{hspot_val}".replace('.', '_')

        plot_heatmap(
            result['angle'],
            f"Spectral Angle: {lidf1} vs {lidf2} (hspot={hspot_val})",
            f"shape_angle_{tag}.jpg"
        )

        plot_heatmap(
            result['rmse'],
            f"Z-RMSE: {lidf1} vs {lidf2} (hspot={hspot_val})",
            f"shape_rmse_{tag}.jpg"
        )

print(f"✅ Spectral shape sensitivity heatmaps saved to: {outdir}")
