import numpy as np
import torch
import torch.nn as nn
import scipy.optimize as opt
import xarray as xr
import matplotlib.pyplot as plt

# === Load Data ===
ds = xr.open_dataset("/store/carmon/PROSAIL_inversions/data/true_reflectance_t8.nc")
R_obs = ds['reflectance'].values[:, 50, 50]
wavelengths = ds['wavelengths'].values
fwhm = ds['fwhm'].values

assert np.all(np.isfinite(R_obs)), "Pixel data invalid."

# === Emulator ===
class CNNSpectralEmulator(nn.Module):
    def __init__(self, input_dim=14, output_dim=2101):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.smoother = nn.Conv1d(1, 1, kernel_size=15, padding=7, bias=False)
        with torch.no_grad():
            self.smoother.weight.fill_(1/15.0)
            self.smoother.weight.requires_grad = False

    def forward(self, x):
        x = self.encoder(x).unsqueeze(1)
        return self.smoother(x).squeeze(1)

emulator = CNNSpectralEmulator()
emulator.load_state_dict(torch.load("model_checkpoints/emulator_cnn.pt", map_location="cpu"))
emulator.eval()

X_mean_std = np.load("/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output/X_mean_std.npy")
X_mean, X_std = X_mean_std[0], X_mean_std[1]

# === Convolution ===
def convolve_to_sensor(r_highres, wl_highres, wl_sensor, fwhm_sensor):
    r_sensor = np.zeros_like(wl_sensor)
    for i, (wl_c, fwhm) in enumerate(zip(wl_sensor, fwhm_sensor)):
        sigma = fwhm / (2*np.sqrt(2*np.log(2)))
        window = np.exp(-0.5 * ((wl_highres - wl_c) / sigma)**2)
        window /= window.sum()
        r_sensor[i] = np.sum(r_highres * window)
    return r_sensor

wl_highres = np.linspace(400, 2500, 2101)

# === Cost Function ===
def residuals(params):
    x_norm = (params - X_mean) / (X_std + 1e-6)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        refl_highres = emulator(x_tensor).numpy().squeeze()
    refl_sensor = convolve_to_sensor(refl_highres, wl_highres, wavelengths, fwhm)
    return refl_sensor - R_obs

# === Optimization ===
initial_guess = np.array([2.0, 40.0, 0.01, 0.009, 1.0, 0.0, 30.0] + [0.0]*7)
result = opt.least_squares(residuals, initial_guess, method='lm')

optimized_params = result.x
optimized_refl = residuals(optimized_params) + R_obs

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, R_obs, 'k--', label="Observed Reflectance")
plt.plot(wavelengths, optimized_refl, 'r-', label="Optimized Emulator Output")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("optimized_leastsquares.png")
