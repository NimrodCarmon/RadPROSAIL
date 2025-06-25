import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import time
from scipy.spatial import cKDTree
from tqdm import tqdm

# === CONFIGURATION ===
LUT_DIR = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output_filtered"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"
OUTPUT_DIR = "figures/lut_inversion_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD LUT ===
X = np.load(os.path.join(LUT_DIR, 'X_inputs_filtered.npy'))
Y = np.load(os.path.join(LUT_DIR, 'Y_reflectance_filtered.npy'))
X_params = X[:, :8]  # Parameters: N to psoil
param_names = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'LAI', 'psoil']
print(f"✅ LUT loaded with {X_params.shape[0]} entries.")

# === LOAD OBSERVED WAVELENGTHS ===
ds = xr.open_dataset(OBS_DATA_PATH)
wavelengths_obs = ds['wavelengths'].values
fwhm_obs = ds['fwhm'].values
R_obs_full = ds['reflectance'].values[:, :10, :10]  # 10x10 pixels

# === BUILD CONVOLUTION MATRIX ===
print("✅ Building convolution matrix...")

wl_highres = np.linspace(400, 2500, 2101)
n_bands_obs = len(wavelengths_obs)
n_bands_highres = len(wl_highres)

conv_matrix = np.zeros((n_bands_obs, n_bands_highres))

for i, (wl_c, fwhm) in enumerate(zip(wavelengths_obs, fwhm_obs)):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    window = np.exp(-0.5 * ((wl_highres - wl_c) / sigma) ** 2)
    window /= window.sum()
    conv_matrix[i, :] = window

# === APPLY CONVOLUTION VIA MATRIX MULTIPLICATION (with tqdm) ===
print("✅ Resampling LUT reflectance spectra (progress shown)...")
start_conv = time.time()

Y_resampled = np.empty((Y.shape[0], n_bands_obs))
for i in tqdm(range(Y.shape[0]), desc="Resampling LUT"):
    Y_resampled[i] = conv_matrix @ Y[i]

end_conv = time.time()
print(f"✅ LUT resampled in {end_conv - start_conv:.2f} seconds.")

# === KD-TREE BUILD ===
kdtree = cKDTree(Y_resampled)
print("✅ KD-tree built.")

# === INVERSION ===
param_maps = np.zeros((8, 10, 10))
var_maps = np.zeros((8, 10, 10))

start_inversion = time.time()
print("✅ Starting inversion on 10x10 pixels...")
for i in tqdm(range(10), desc="Pixels"):
    for j in range(10):
        R_obs = R_obs_full[:, i, j]
        _, idxs = kdtree.query(R_obs, k=20)
        closest_params = X_params[idxs]
        best_params = closest_params[0]
        param_maps[:, i, j] = best_params
        var_maps[:, i, j] = np.var(closest_params, axis=0)

elapsed_inversion = time.time() - start_inversion
print(f"✅ Inversion completed in {elapsed_inversion:.2f} seconds for 100 pixels.")

# === SAVE PARAMETER MAPS ===
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx, name in enumerate(param_names):
    im = axes[idx].imshow(param_maps[idx], cmap='viridis')
    axes[idx].set_title(f'{name} estimate')
    plt.colorbar(im, ax=axes[idx])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_estimates_matrix_resample.jpg'), dpi=200)
print("✅ Saved parameter estimates map.")

# === SAVE VARIANCE MAPS ===
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for idx, name in enumerate(param_names):
    im = axes[idx].imshow(var_maps[idx], cmap='magma')
    axes[idx].set_title(f'{name} variance')
    plt.colorbar(im, ax=axes[idx])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_variances_matrix_resample.jpg'), dpi=200)
print("✅ Saved parameter variance map.")
