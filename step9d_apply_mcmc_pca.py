import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import matplotlib.pyplot as plt
import emcee
import os

# === CONFIGURATION ===
MODEL_CHECKPOINT = "model_checkpoints/pca_emulator_sobol.pt"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"
OUTPUT_DIR = "figures/pca_inverse"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_IDX = slice(0, 9)
GEOMETRY_PARAMS = {'tts': 30.0, 'tto': 0.0, 'psi': 0.0, 'hspot': 0.1}

# === LOAD DATA ===
ds = xr.open_dataset(OBS_DATA_PATH)
R_obs = ds['reflectance'].values[:, 50, 50]
wavelengths, fwhm = ds['wavelengths'].values, ds['fwhm'].values
wl_highres = np.linspace(400, 2500, 2101)

# === LOAD EMULATOR TO GPU ===
device = torch.device('cuda')
checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False)
X_mean_gpu = torch.tensor(checkpoint['X_mean'], device=device, dtype=torch.float32)
X_std_gpu = torch.tensor(checkpoint['X_std'], device=device, dtype=torch.float32)
pca_mean_gpu = torch.tensor(checkpoint['pca_mean'], device=device, dtype=torch.float32)
pca_components_gpu = torch.tensor(checkpoint['pca_components'], device=device, dtype=torch.float32)

class PCAEmulator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

emulator = PCAEmulator(len(X_mean_gpu), pca_components_gpu.shape[0]).to(device)
emulator.load_state_dict(checkpoint['model_state_dict'])
emulator.eval()

# === GPU-BASED CONVOLUTION FUNCTION ===
def convolve_to_sensor_gpu(r_highres_gpu, wl_highres, wl_sensor, fwhm_sensor):
    device = r_highres_gpu.device
    wl_highres_gpu = torch.tensor(wl_highres, device=device, dtype=torch.float32)
    n_bands = len(wl_sensor)
    refl_sensor_gpu = torch.zeros((r_highres_gpu.shape[0], n_bands), device=device)

    for i, (wl_c, fwhm_val) in enumerate(zip(wl_sensor, fwhm_sensor)):
        sigma = fwhm_val / (2 * np.sqrt(2 * np.log(2)))
        window_gpu = torch.exp(-0.5 * ((wl_highres_gpu - wl_c) / sigma) ** 2)
        window_gpu /= window_gpu.sum()
        refl_sensor_gpu[:, i] = (r_highres_gpu * window_gpu).sum(dim=1)

    return refl_sensor_gpu

# === GPU MODEL FUNCTION ===
def model_gpu(params_tensor):
    batch_size = params_tensor.shape[0]
    geom_params = torch.tensor(
        [GEOMETRY_PARAMS['tts'], GEOMETRY_PARAMS['tto'], GEOMETRY_PARAMS['psi'], GEOMETRY_PARAMS['hspot']],
        device=device, dtype=torch.float32
    ).repeat(batch_size, 1)

    full_params = torch.cat([params_tensor, geom_params], dim=1)
    x_norm = (full_params - X_mean_gpu) / X_std_gpu

    with torch.no_grad():
        pred_scores = emulator(x_norm)
        pred_refl = pred_scores @ pca_components_gpu + pca_mean_gpu

    return pred_refl  # stays on GPU

# === LOG-PROBABILITY (BATCHED, FULL GPU) ===
param_bounds_original = (
    np.array([1.0, 10.0, 2.0, 0.0, 0.005, 0.005, 0.1, 0.0, -2.0]),
    np.array([2.5, 80.0, 20.0, 0.5, 0.05, 0.05, 6.0, 1.0, 2.0])
)
bounds_low_gpu = torch.tensor(param_bounds_original[0], device=device)
bounds_high_gpu = torch.tensor(param_bounds_original[1], device=device)
R_obs_gpu = torch.tensor(R_obs, device=device, dtype=torch.float32)

def log_prob_batch(params_batch):
    params_tensor = torch.tensor(params_batch, dtype=torch.float32, device=device)
    pred_refl_batch_gpu = model_gpu(params_tensor)

    refl_sensor_batch_gpu = convolve_to_sensor_gpu(pred_refl_batch_gpu, wl_highres, wavelengths, fwhm)
    residual_batch_gpu = refl_sensor_batch_gpu - R_obs_gpu

    log_probs_gpu = -0.5 * torch.sum((residual_batch_gpu / 0.001)**2, dim=1)

    out_of_bounds = torch.any((params_tensor < bounds_low_gpu) | (params_tensor > bounds_high_gpu), dim=1)
    log_probs_gpu[out_of_bounds] = -torch.inf

    return log_probs_gpu.cpu().numpy()

# === INITIALIZE MCMC ===
ndim, nwalkers, nsteps = 9, 128, 500
initial_pos = np.random.uniform(param_bounds_original[0], param_bounds_original[1], size=(nwalkers, ndim))

# === RUN EMCEE MCMC ===
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_batch, vectorize=True)
print("✅ Starting MCMC (fully GPU)...")
sampler.run_mcmc(initial_pos, nsteps, progress=True)

# === POSTERIOR ANALYSIS ===
samples = sampler.get_chain(discard=100, flat=True)
best_params = np.median(samples, axis=0)
print("✅ Best Parameters (Median):", best_params)

# === TRACE PLOTS ===
fig, axes = plt.subplots(ndim, figsize=(8, 14), sharex=True)
labels = ['N', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'LAI', 'psoil', 'LIDFa']
for i in range(ndim):
    axes[i].plot(sampler.get_chain()[:,:,i], alpha=0.5)
    axes[i].set_ylabel(labels[i])
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'mcmc_trace.png'), dpi=200)
print("✅ Saved trace plots.")

# === REFLECTANCE PLOT ===
final_pred_refl_gpu = model_gpu(torch.tensor(best_params[None, :], device=device, dtype=torch.float32))
optimized_refl_gpu = convolve_to_sensor_gpu(final_pred_refl_gpu, wl_highres, wavelengths, fwhm)
optimized_refl = optimized_refl_gpu.cpu().numpy().squeeze()

plt.figure(figsize=(10,5))
plt.plot(wavelengths, R_obs, 'k--', label='Observed')
plt.plot(wavelengths, optimized_refl, 'r-', label='MCMC Optimized')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('MCMC Reflectance Optimization')
plt.legend(); plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'optimized_reflectance.png'), dpi=200)
print("✅ Saved reflectance comparison plot.")

# === RESIDUALS PLOT ===
plt.figure(figsize=(10,4))
plt.plot(wavelengths, optimized_refl - R_obs, 'b')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Residual')
plt.title('Residuals (Optimized - Observed)')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residuals.png'), dpi=200)
print("✅ Saved residuals plot.")
