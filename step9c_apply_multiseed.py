import numpy as np
import torch
import torch.nn as nn
import scipy.optimize as opt
import xarray as xr
import matplotlib.pyplot as plt

# === CONFIGURATION ===
MODEL_CHECKPOINT = "model_checkpoints/pca_emulator_sobol.pt"
OBS_DATA_PATH = "/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc"

STATE_IDX = slice(0, 9)

GEOMETRY_PARAMS = {'tts':30.0, 'tto':0.0, 'psi':0.0, 'hspot':0.1}

# === LOAD OBSERVED DATA ===
ds = xr.open_dataset(OBS_DATA_PATH)
R_obs = ds['reflectance'].values[:,50,50]
wavelengths, fwhm = ds['wavelengths'].values, ds['fwhm'].values

# === LOAD EMULATOR ===
checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cpu', weights_only=False)
X_mean, X_std = checkpoint['X_mean'], checkpoint['X_std']
pca_mean, pca_components = checkpoint['pca_mean'], checkpoint['pca_components']

class PCAEmulator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(),
                                 nn.Linear(128,128), nn.ReLU(),
                                 nn.Linear(128,output_dim))
    def forward(self, x): return self.net(x)

emulator = PCAEmulator(len(X_mean), pca_components.shape[0])
emulator.load_state_dict(checkpoint['model_state_dict'])
emulator.eval()

# === CONVOLVE FUNCTION ===
def convolve_to_sensor(r_highres, wl_highres, wl_sensor, fwhm_sensor):
    r_sensor = np.zeros_like(wl_sensor)
    for i, (wl_c, fwhm_val) in enumerate(zip(wl_sensor,fwhm_sensor)):
        sigma = fwhm_val/(2*np.sqrt(2*np.log(2)))
        window = np.exp(-0.5*((wl_highres-wl_c)/sigma)**2)
        window /= window.sum()
        r_sensor[i] = np.sum(r_highres*window)
    return r_sensor

wl_highres = np.linspace(400,2500,2101)

# === RESIDUAL FUNCTION ===
def residuals(state_params):
    full_params = np.concatenate([
        state_params, [GEOMETRY_PARAMS['tts'],GEOMETRY_PARAMS['tto'],
                       GEOMETRY_PARAMS['psi'],GEOMETRY_PARAMS['hspot']]])
    x_norm = (full_params - X_mean)/X_std
    with torch.no_grad():
        pred_scores = emulator(torch.tensor(x_norm,dtype=torch.float32).unsqueeze(0)).numpy()
    pred_refl = pred_scores @ pca_components + pca_mean
    refl_sensor = convolve_to_sensor(pred_refl.squeeze(),wl_highres,wavelengths,fwhm)
    return refl_sensor - R_obs

# === PARAMETER GRID SEARCH ===
param_bounds = {
    'N': (1.0, 2.5),
    'Cab': (10, 80),
    'Car': (2, 20),
    'Cbrown': (0, 0.5),
    'Cw': (0.005, 0.05),
    'Cm': (0.005, 0.05),
    'LAI': (0.1, 6.0),
    'psoil': (0.0, 1.0),
    'LIDFa': (-1.0, 1.0)
}

# Grid sampling (1000 points)
n_grid = 1000
rng = np.random.default_rng(42)
grid_params = np.vstack([
    rng.uniform(low,high,n_grid) for low,high in param_bounds.values()
]).T

residual_norms = np.array([np.linalg.norm(residuals(p)) for p in grid_params])
best_idxs = residual_norms.argsort()[:5]

initial_guesses = grid_params[best_idxs]

# === OPTIMIZATION ===
results = []
opt_options = {
    'method': 'trf',
    'xtol': 1e-15,  # even tighter
    'ftol': 1e-15,
    'gtol': 1e-15,
    'x_scale': 'jac',
    'verbose': 2,
    'max_nfev': 5000   # significantly higher
}

scaled_bounds = (
    (np.array([low for low,_ in param_bounds.values()]) - X_mean[STATE_IDX])/X_std[STATE_IDX],
    (np.array([high for _,high in param_bounds.values()]) - X_mean[STATE_IDX])/X_std[STATE_IDX]
)

for idx, guess in enumerate(initial_guesses):
    guess_scaled = (guess - X_mean[STATE_IDX]) / X_std[STATE_IDX]

    res = opt.least_squares(
        lambda x: residuals(x * X_std[STATE_IDX] + X_mean[STATE_IDX]),
        guess_scaled, bounds=scaled_bounds, **opt_options)

    optimized = res.x * X_std[STATE_IDX] + X_mean[STATE_IDX]
    cost = np.linalg.norm(res.fun)
    results.append((cost, optimized, res.fun, res.nfev))

    print(f"Seed {idx+1}: Cost={cost:.6f}, Evaluations={res.nfev}")


best_result = min(results, key=lambda x: x[0])
final_params, final_residuals = best_result[1], best_result[2]

# === REPORTING ===
print("✅ Best optimization cost:", best_result[0])
print("✅ Optimized Parameters:", final_params)

# === PLOT ===
plt.figure(figsize=(10,5))
plt.plot(wavelengths,R_obs,'k--',label="Observed")
plt.plot(wavelengths,R_obs+final_residuals,'r-',label="Optimized")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title(f"Optimized Reflectance (Cost: {best_result[0]:.4f})")
plt.legend(); plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("final_optimized_reflectance.png", dpi=300)
print("✅ Saved final optimized reflectance plot.")
