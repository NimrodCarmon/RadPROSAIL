"""
Evaluate trained emulator model on validation data.
Reports MSE, RMSE, MAE, SAM, saves per-band RMSE and histogram,
and benchmarks inference speed on GPU and CPU (64-core multiprocessing).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from joblib import Parallel, delayed

# --- Model Definition ---
class CNNSpectralEmulator(nn.Module):
    def __init__(self, input_dim=14, output_dim=2101):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.smoother = nn.Conv1d(1, 1, kernel_size=15, padding=7, bias=False)
        with torch.no_grad():
            self.smoother.weight.fill_(1 / 15.0)
            self.smoother.weight.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x = self.smoother(x)
        return x.squeeze(1)

# --- Load Data ---
data_dir = "/store/carmon/PROSAIL_inversions/RadPROSAIL/lut_output"
X = np.load(os.path.join(data_dir, "X_inputs.npy"))
Y = np.load(os.path.join(data_dir, "Y_reflectance.npy"))

mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1) & ~np.any(np.isnan(Y) | np.isinf(Y), axis=1)
X = X[mask]
Y = Y[mask]

# --- Normalize Inputs Consistently ---
mean_std_path = os.path.join(data_dir, "X_mean_std.npy")
if os.path.exists(mean_std_path):
    X_mean, X_std = np.load(mean_std_path)
    print("✅ Loaded X_mean and X_std from disk")
else:
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1e-6
    np.save(mean_std_path, np.stack([X_mean, X_std]))
    print("✅ Computed and saved X_mean and X_std to disk")

X = (X - X_mean) / X_std

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(np.clip(Y, 0, 2), dtype=torch.float32)

# --- Load CPU-only Model ---
model_cpu = CNNSpectralEmulator()
model_cpu.load_state_dict(torch.load("model_checkpoints/emulator_cnn.pt", map_location="cpu"))
model_cpu.eval()

# --- GPU Benchmark ---
def measure_gpu_speed(X_tensor):
    model_gpu = CNNSpectralEmulator()
    model_gpu.load_state_dict(torch.load("model_checkpoints/emulator_cnn.pt", map_location="cuda"))
    model_gpu = model_gpu.to("cuda")
    model_gpu.eval()

    X_sample = X_tensor[:1000].to("cuda")
    with torch.no_grad():
        for _ in range(5):  # Warm-up
            _ = model_gpu(X_sample)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = model_gpu(X_sample)
    torch.cuda.synchronize()
    end = time.time()
    throughput = 1000 / (end - start)
    print(f"Inference speed on GPU: {throughput:.2f} samples/sec")

# --- CPU Parallel Benchmark ---
def batched_inference(model_state_dict, X_chunk_np):
    torch.set_num_threads(1)
    model = CNNSpectralEmulator()
    model.load_state_dict(model_state_dict)
    model.eval()
    with torch.no_grad():
        X_chunk = torch.tensor(X_chunk_np, dtype=torch.float32)
        return model(X_chunk).numpy()

def measure_cpu_parallel(model, X_tensor, n_jobs):
    chunk_size = 1000
    total_samples = chunk_size * n_jobs
    X_sample = X_tensor[:total_samples]
    X_chunks_np = np.array_split(X_sample.numpy(), n_jobs)

    model_state_dict = model.state_dict()

    start = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(batched_inference)(model_state_dict, chunk) for chunk in X_chunks_np
    )
    end = time.time()

    throughput = total_samples / (end - start)
    print(f"Inference speed on CPU (joblib, {n_jobs} processes): {throughput:.2f} samples/sec")

# --- Run Benchmarks ---
if torch.cuda.is_available():
    measure_gpu_speed(X_tensor)

measure_cpu_parallel(model_cpu, X_tensor, n_jobs=64)

# --- Final Inference ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNSpectralEmulator()
model.load_state_dict(torch.load("model_checkpoints/emulator_cnn.pt", map_location=device))
model = model.to(device)
model.eval()

with torch.no_grad():
    preds = model(X_tensor.to(device)).cpu().numpy()
    targets = Y_tensor.numpy()

# --- Metrics ---
def spectral_angle(a, b):
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return np.arccos(np.clip(dot / (norm_a * norm_b + 1e-8), -1.0, 1.0))

mse = np.mean((preds - targets) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(preds - targets))
sam = np.mean(spectral_angle(preds, targets))

print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"SAM:  {sam:.6f} radians")

# --- Per-band RMSE Plot ---
band_rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=0))
wl = np.linspace(400, 2500, 2101)

os.makedirs("figures/emulator", exist_ok=True)
plt.figure(figsize=(10, 4))
plt.plot(wl, band_rmse)
plt.xlabel("Wavelength (nm)")
plt.ylabel("RMSE")
plt.title("Per-band Reflectance RMSE")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figures/emulator/validation_band_rmse.png", dpi=300)
print("✅ Saved per-band RMSE plot")

# --- Histogram Plot ---
plt.figure(figsize=(6, 4))
abs_errors = np.abs(preds - targets).flatten()
plt.hist(abs_errors, bins=100, alpha=0.8)
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Histogram of Absolute Errors")
plt.tight_layout()
plt.savefig("figures/emulator/validation_error_histogram.png", dpi=300)
print("✅ Saved error histogram plot")
