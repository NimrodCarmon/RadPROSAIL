import numpy as np
import xarray as xr
import os
import time
from sklearn.decomposition import PCA
from tqdm import tqdm

# === CONFIGURATION ===
LUT_DIR = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output_filtered"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"
OUTFILE = "pca_lut_projection.npz"

# === LOAD LUT ===
X = np.load(os.path.join(LUT_DIR, 'X_inputs_filtered.npy'))
Y = np.load(os.path.join(LUT_DIR, 'Y_reflectance_filtered.npy'))
X_params = X[:, :8]
print(f"✅ Loaded LUT: {X.shape[0]} samples, {Y.shape[1]} bands (high-res)")

# === LOAD OBSERVED WAVELENGTHS ===
ds = xr.open_dataset(OBS_DATA_PATH)
wavelengths_obs = ds['wavelengths'].values
fwhm_obs = ds['fwhm'].values
wl_highres = np.linspace(400, 2500, Y.shape[1])

# === BUILD CONVOLUTION MATRIX ===
print("✅ Building convolution matrix...")
n_bands_obs = len(wavelengths_obs)
conv_matrix = np.zeros((n_bands_obs, len(wl_highres)))

for i, (wl_c, fwhm) in enumerate(zip(wavelengths_obs, fwhm_obs)):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    window = np.exp(-0.5 * ((wl_highres - wl_c) / sigma) ** 2)
    window /= window.sum()
    conv_matrix[i, :] = window

# === APPLY CONVOLUTION ===
print("✅ Resampling LUT (matrix method)...")
start = time.time()
Y_resampled = Y @ conv_matrix.T  # shape: (n_samples, n_bands_obs)
print(f"✅ Resampled in {time.time() - start:.2f}s. Shape: {Y_resampled.shape}")

# === PCA COMPRESSION ===
print("✅ Performing PCA...")
pca = PCA(n_components=0.999999999)  # retain >99.9% variance
Y_pca = pca.fit_transform(Y_resampled)

print(f"✅ PCA reduced from {Y_resampled.shape[1]} to {Y_pca.shape[1]} components")

# === SAVE RESULTS ===
np.savez(
    OUTFILE,
    X_inputs=X_params,
    Y_pca=Y_pca,
    pca_mean=pca.mean_,
    pca_components=pca.components_,
    pca_variance=pca.explained_variance_ratio_,
    wavelengths_obs=wavelengths_obs,
    conv_matrix=conv_matrix
)
print(f"✅ Saved PCA LUT archive to {OUTFILE}")
