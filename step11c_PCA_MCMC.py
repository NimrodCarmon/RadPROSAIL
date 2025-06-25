import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import time
import emcee
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

# === CONFIGURATION ===
ARCHIVE_PATH = "pca_lut_projection.npz"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"
OUTPUT_DIR = "figures/lut_mcmc_input_space"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_STEPS = 300
N_WALKERS = 32
PARAM_NAMES = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'LAI', 'psoil']
PARAM_BOUNDS = np.array([
    [1.0, 10,   2,   0.0, 0.002, 0.002, 0.1, 0.0],   # lower bounds
    [2.5, 80,  20,   0.75, 0.08,  0.1,  10.0, 1.0]   # upper bounds
])

# === LOAD PCA LUT EMULATOR ===
print("✅ Loading PCA LUT archive...")
data = np.load(ARCHIVE_PATH)
X_inputs = data['X_inputs']                  # (N, 8)
Y_pca = data['Y_pca']                        # (N, n_pca)
pca_mean = data['pca_mean']                  # (bands,)
pca_components = data['pca_components']      # (n_pca, bands)
wavelengths_obs = data['wavelengths_obs']

# === TRAIN PCA-BASED EMULATOR ===
print("✅ Fitting PCA-based emulator...")
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_inputs, Y_pca)

def emulator(params):
    """Map input parameters → PCA coefficients → reflectance"""
    pca_pred = reg.predict(params[None, :])   # shape (1, n_pca)
    refl_pred = pca_pred @ pca_components + pca_mean
    return refl_pred.ravel()  # shape (bands,)

# === LOAD OBS REFLECTANCE ===
ds = xr.open_dataset(OBS_DATA_PATH)
R_obs_full = ds['reflectance'].values[:, :10, :10]  # shape: (bands, 10, 10)

# === STORAGE ===
param_means = np.zeros((8, 10, 10))
param_stds  = np.zeros((8, 10, 10))
residuals   = np.zeros((10, 10))
spectral_fits = {}

# === START INVERSION ===
print("✅ Starting MCMC in parameter space using PCA emulator...")
start_total = time.time()

for i in tqdm(range(10), desc="Pixels"):
    for j in range(10):
        R_obs = R_obs_full[:, i, j]

        def log_prior(p):
            if np.all((p >= PARAM_BOUNDS[0]) & (p <= PARAM_BOUNDS[1])):
                return 0.0  # uniform
            return -np.inf

        def log_likelihood(p):
            R_sim = emulator(p)
            return -0.5 * np.sum((R_obs - R_sim)**2)

        def log_prob(p):
            lp = log_prior(p)
            return lp + log_likelihood(p) if np.isfinite(lp) else -np.inf

        # === Init walkers ===
        center = np.mean(PARAM_BOUNDS, axis=0)
        scale = (PARAM_BOUNDS[1] - PARAM_BOUNDS[0]) * 0.1
        pos0 = center + scale * np.random.randn(N_WALKERS, len(PARAM_NAMES))

        sampler = emcee.EnsembleSampler(N_WALKERS, len(PARAM_NAMES), log_prob)
        sampler.run_mcmc(pos0, N_STEPS, progress=False)
        samples = sampler.get_chain(discard=N_STEPS//2, flat=True)

        param_means[:, i, j] = samples.mean(axis=0)
        param_stds[:, i, j]  = samples.std(axis=0)

        R_sim = emulator(param_means[:, i, j])
        residuals[i, j] = np.linalg.norm(R_obs - R_sim)

        if (i, j) in [(0, 0), (3, 3), (5, 5), (7, 2)]:
            spectral_fits[(i, j)] = (R_obs, R_sim)

elapsed = time.time() - start_total
print(f"✅ MCMC completed in {elapsed:.2f} seconds for 100 pixels.")

# === PARAMETER POSTERIOR MEANS ===
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for k, name in enumerate(PARAM_NAMES):
    im = axes.flat[k].imshow(param_means[k], cmap='viridis')
    axes.flat[k].set_title(f'{name} mean')
    plt.colorbar(im, ax=axes.flat[k])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_means_mcmc_input.jpg'), dpi=200)

# === PARAMETER POSTERIOR STDs ===
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for k, name in enumerate(PARAM_NAMES):
    im = axes.flat[k].imshow(param_stds[k], cmap='magma')
    axes.flat[k].set_title(f'{name} std')
    plt.colorbar(im, ax=axes.flat[k])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_stds_mcmc_input.jpg'), dpi=200)

# === RESIDUAL MAP ===
plt.figure(figsize=(6, 5))
plt.imshow(residuals, cmap='inferno')
plt.title("Residual Norm ||Obs - Sim||")
plt.colorbar(label="Residual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residual_map_mcmc_input.jpg'), dpi=200)

# === SPECTRAL FITS ===
for (i, j), (obs, fit) in spectral_fits.items():
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths_obs, obs, 'k--', label='Observed')
    plt.plot(wavelengths_obs, fit, 'r-', label='MCMC Fit')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Fit at pixel ({i},{j})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'spectral_fit_pixel_{i}_{j}_mcmc_input.jpg'), dpi=200)
    plt.close()

print("✅ All MCMC output saved.")
