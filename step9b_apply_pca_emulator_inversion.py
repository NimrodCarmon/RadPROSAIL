import numpy as np
import torch
import torch.nn as nn
import scipy.optimize as opt
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === CONFIGURATION ===
DATA_DIR = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output"
MODEL_CHECKPOINT = "model_checkpoints/pca_emulator_sobol.pt"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"

# State parameters indices (9 state parameters)
STATE_IDX = slice(0, 9)

# Geometry parameters (fixed as per LUT)
GEOMETRY_PARAMS = {
    'tts': 30.0,  # Solar zenith angle
    'tto': 0.0,   # View zenith angle
    'psi': 0.0,   # Relative azimuth
    'hspot': 0.1  # Hotspot
}

# === LOAD OBSERVED REFLECTANCE ===
ds = xr.open_dataset(OBS_DATA_PATH)
R_obs = ds['reflectance'].values[:, 50, 50]
wavelengths = ds['wavelengths'].values
fwhm = ds['fwhm'].values
assert np.all(np.isfinite(R_obs)), "Pixel data invalid."

# === LOAD PCA EMULATOR MODEL ===
checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cpu', weights_only=False)
X_mean, X_std = checkpoint['X_mean'], checkpoint['X_std']
pca_mean, pca_components = checkpoint['pca_mean'], checkpoint['pca_components']

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

emulator = PCAEmulator(len(X_mean), pca_components.shape[0])
emulator.load_state_dict(checkpoint['model_state_dict'])
emulator.eval()

# === CONVOLUTION FUNCTION ===
def convolve_to_sensor(r_highres, wl_highres, wl_sensor, fwhm_sensor):
    r_sensor = np.zeros_like(wl_sensor)
    for i, (wl_c, fwhm_val) in enumerate(zip(wl_sensor, fwhm_sensor)):
        sigma = fwhm_val / (2 * np.sqrt(2 * np.log(2)))
        window = np.exp(-0.5 * ((wl_highres - wl_c) / sigma) ** 2)
        window /= window.sum()
        r_sensor[i] = np.sum(r_highres * window)
    return r_sensor

wl_highres = np.linspace(400, 2500, 2101)

# === RESIDUAL FUNCTION FOR INVERSION ===
def residuals_scaled(scaled_state_params):
    state_params = scaled_state_params * X_std[STATE_IDX] + X_mean[STATE_IDX]
    full_params = np.concatenate([
        state_params,
        [GEOMETRY_PARAMS['tts'], GEOMETRY_PARAMS['tto'], GEOMETRY_PARAMS['psi'], GEOMETRY_PARAMS['hspot']]
    ])
    x_norm = (full_params - X_mean) / X_std
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        predicted_scores = emulator(x_tensor).numpy()
    predicted_refl = predicted_scores @ pca_components + pca_mean
    refl_sensor = convolve_to_sensor(predicted_refl.squeeze(), wl_highres, wavelengths, fwhm)
    return refl_sensor - R_obs

# === OPTIMIZATION CONFIGURATION ===
# Bounds in original parameter units
param_bounds_original = (
    np.array([1.0, 10.0, 2.0, 0.0, 0.005, 0.005, 0.1, 0.0, -2.0]),  # Lower
    np.array([2.5, 80.0, 20.0, 0.5, 0.05, 0.05, 6.0, 1.0, 2.0])     # Upper
)

# Explicit scaling of bounds
scaled_bounds = (
    (param_bounds_original[0] - X_mean[STATE_IDX]) / X_std[STATE_IDX],
    (param_bounds_original[1] - X_mean[STATE_IDX]) / X_std[STATE_IDX]
)

# Initial guess (original units)
initial_guess_original = np.array([1.5, 40.0, 8.0, 0.1, 0.01, 0.01, 2.0, 0.5, 0.0])
initial_guess_scaled = (initial_guess_original - X_mean[STATE_IDX]) / X_std[STATE_IDX]

# Optimization options
opt_options = {
    'method': 'trf',
    'xtol': 1e-12,
    'ftol': 1e-10,
    'gtol': 1e-10,
    'x_scale': np.ones_like(initial_guess_scaled) * 0.5,
    'verbose': 2,
    'max_nfev': 2000
}

# === Sensitivity Check (inserted here) ===
scaled_test_params = initial_guess_scaled + np.random.normal(0, 1.0, initial_guess_scaled.shape)
difference = np.linalg.norm(residuals_scaled(scaled_test_params) - residuals_scaled(initial_guess_scaled))
print("✅ Sensitivity Check Norm:", difference)


# === RUN OPTIMIZATION ===
result = opt.least_squares(
    residuals_scaled,
    initial_guess_scaled,
    bounds=scaled_bounds,
    **opt_options
)

optimized_params = result.x * X_std[STATE_IDX] + X_mean[STATE_IDX]

# === DETAILED OPTIMIZATION REPORT ===
print("\n✅ Optimization Report:")
print("----------------------------")
print(f"Success: {'Yes' if result.success else 'No'}")
print(f"Status: {result.status} - {result.message}")
print(f"Final Cost: {result.cost:.6f}")
print(f"Gradient Norm: {np.linalg.norm(result.grad):.6e}")
print(f"Number of Function Evaluations: {result.nfev}")
print(f"Number of Jacobian Evaluations: {result.njev}")
print(f"Number of Function Evaluations: {result.nfev}")
print(f"Number of Jacobian Evaluations: {result.njev}")
print("\nOptimized state parameters:", optimized_params)

# === GENERATE OPTIMIZED REFLECTANCE ===
optimized_refl = residuals_scaled(result.x) + R_obs

# === PLOTS ===
# Observed vs. Optimized reflectance
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, R_obs, 'k--', label="Observed Reflectance")
plt.plot(wavelengths, optimized_refl, 'r-', label="Optimized Reflectance")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Reflectance Inversion Result\n"
          f"Optimization {'Success' if result.success else 'Failure'} | "
          f"Cost: {result.cost:.4f} | "
          f"Grad Norm: {np.linalg.norm(result.grad):.2e}")
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("optimized_reflectance_pca_emulator_detailed.png", dpi=300)
print("✅ Saved reflectance inversion plot with optimization details.")

# Residual plot
plt.figure(figsize=(10, 4))
residuals = optimized_refl - R_obs
plt.plot(wavelengths, residuals, color='blue')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Residual (Optimized - Observed)")
plt.title("Per-band Residuals After Optimization\n"
          f"Mean Residual: {np.mean(residuals):.2e} | Std: {np.std(residuals):.2e}")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("residuals_per_band_detailed.png", dpi=300)
print("✅ Saved residuals plot with optimization details.")
