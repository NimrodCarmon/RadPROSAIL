#!/usr/bin/env python3
"""
===========================================================
DEBUG: Single‐Spectrum PROSAIL MCMC Inversion with Progress Bars
===========================================================

This script reads one attenuated‐reflectance CSV (NDVI*.csv), identifies the 40° column,
drops wavelengths where any reflectance column has NaN (water‐vapor or other gaps),
runs a single emcee MCMC inversion on that 40° spectrum with:
  - reduced assumed noise (σ² = 1e-6)
  - increased burn‐in and chain length
  - tqdm progress bars for MCMC and fit sampling
and then plots measured vs. fitted spectra with 90% credible intervals.

Steps:
  1. Locate the first CSV matching NDVI*.csv in INDIR.
  2. Read its “wavelength” and all reflectance columns.
  3. For PROSAIL wavelengths (400–2500 nm), interpolate each reflectance column onto that grid.
     Build a global mask of wavelengths where all columns are finite (drop any band where at least
     one column is NaN).
  4. Parse the column whose header angle is closest to 40°.
  5. Interpolate that 40° reflectance onto PROSAIL’s grid, apply the global mask, and extract the finite subset.
  6. Run emcee’s affine‐invariant sampler on that subset to retrieve medians for:
       N, LAI, Cab, Car, Cbrown, Cw, Cm, with σ² = 1e-6, N_BURN=1000, N_STEPS=3000.
     Display a tqdm progress bar during burn‐in and production.
  7. From posterior samples, simulate PROSAIL fits at each valid wavelength to compute:
       - fit_mean (median reflectance at each wavelength)
       - fit_low (5th percentile)
       - fit_high (95th percentile)
     Display a tqdm progress bar while generating N_SAMPLES_PLOT forward runs.
  8. Plot measured spectrum, fit_mean, and shaded [fit_low, fit_high] interval.
     Save figure to “figures/MCMC/{spec_name}_40deg_mcmc_fit.png”, creating directories if needed.
  9. Print median parameter values.

Usage:
  python step4_MCMC_debug_single_adjusted_progress.py

Requirements:
  - prosail2 (for Prosail)
  - emcee (v3+)
  - numpy, pandas, matplotlib
  - tqdm
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prosail2 import Prosail
import emcee
from tqdm import tqdm

# ----------------------------------------
# USER CONFIGURATION
# ----------------------------------------

INDIR = "../data"                          # directory containing CSVs
FILE_PATTERN = os.path.join(INDIR, "NDVI*.csv")

# Target “truth” angle
TRUTH_ANGLE = 40

# PROSAIL float parameters (must match order)
FLOAT_PARAMS = ["N", "LAI", "Cab", "Car", "Cbrown", "Cw", "Cm"]

# PROSAIL fixed parameters
FIXED_PARAMS = {
    "hspot": 0.1,
    "tts": TRUTH_ANGLE,  # illumination zenith angle = 40° for the “truth” run
    "tto": 30,
    "psi": 60,
    "LIDFa": 0.66,
    "LIDFb": -0.04,
    "psoil": 0.2
}

# Uniform prior bounds for each float parameter
PARAM_BOUNDS = {
    "N":      (1.0, 3.0),
    "LAI":    (0.2, 8.0),
    "Cab":    (10.0, 90.0),
    "Car":    (1.0, 20.0),
    "Cbrown": (0.0, 1.0),
    "Cw":     (0.005, 0.06),
    "Cm":     (0.005, 0.03)
}

# MCMC settings (increased from original)
N_WALKERS = 28
N_BURN = 1000    # increased burn‐in
N_STEPS = 3000   # increased production steps

# Assumed noise variance (very small for simulated noise‐free data)
SIGMA2 = 1e-8

# Number of posterior samples to use when generating fit envelope
N_SAMPLES_PLOT = 200

# Output figure directory
FIG_DIR = "figures/MCMC"


# ----------------------------------------
# HELPER FUNCTIONS FOR MCMC
# ----------------------------------------

def random_param_vector():
    """
    Draw one random vector (length = len(FLOAT_PARAMS)) uniformly within PARAM_BOUNDS.
    """
    return np.array([np.random.uniform(PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1]) 
                     for p in FLOAT_PARAMS])


def param_dict_from_vector(vec):
    """
    Convert a numpy array of shape (len(FLOAT_PARAMS),) into a PROSAIL config dict.
    """
    d = {p: float(v) for p, v in zip(FLOAT_PARAMS, vec)}
    d.update(FIXED_PARAMS)
    return d


def lnprior(theta):
    """
    Uniform prior over PARAM_BOUNDS:
    Returns 0.0 if all parameters are within their bounds; otherwise -np.inf.
    """
    for t, (lo, hi) in zip(theta, [PARAM_BOUNDS[p] for p in FLOAT_PARAMS]):
        if not (lo <= t <= hi):
            return -np.inf
    return 0.0


def lnlike(theta, measured_spec, wl_indices, prosail_obj):
    config = param_dict_from_vector(theta)
    try:
        sim_full = np.array(prosail_obj.run(config))
        sim = sim_full[wl_indices]
    except Exception:
        return -np.inf

    # Compute finite-difference first derivatives
    d_measured = np.gradient(measured_spec)
    d_sim = np.gradient(sim)

    resid = d_measured - d_sim
    return -0.5 * np.sum(resid**2 / SIGMA2 + np.log(2 * np.pi * SIGMA2))



def lnlike_old(theta, measured_spec, wl_indices, prosail_obj):
    """
    Gaussian log‐likelihood with σ^2 = SIGMA2.
    - measured_spec: 1D array of reflectances at PROSAIL wavelengths indexed by wl_indices.
    - wl_indices: 1D array of integer indices into prosail_obj.wl.
    """
    config = param_dict_from_vector(theta)
    try:
        sim_full = np.array(prosail_obj.run(config))
        sim = sim_full[wl_indices]
    except Exception:
        return -np.inf

    resid = measured_spec - sim
    return -0.5 * np.sum(resid * resid / SIGMA2 + np.log(2 * np.pi * SIGMA2))


def lnprob(theta, measured_spec, wl_indices, prosail_obj):
    """
    Log‐posterior = lnprior + lnlike.
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, measured_spec, wl_indices, prosail_obj)


def run_mcmc_on_spectrum(measured_spec, wl_indices):
    """
    Run emcee’s affine‐invariant ensemble sampler on one measured spectrum.
    - measured_spec: 1D array of reflectance at PROSAIL wavelengths (specified by wl_indices).
    - wl_indices: 1D array of integer indices into prosail.wl for valid wavelengths.

    Returns:
      med_dict: {param_name: median_of_posterior}
      samples: array of shape (N_WALKERS * N_STEPS, ndim)
    """
    prosail_obj = Prosail()
    ndim = len(FLOAT_PARAMS)

    # Initialize walker positions uniformly within prior bounds
    p0 = [random_param_vector() for _ in range(N_WALKERS)]

    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim, lnprob, args=(measured_spec, wl_indices, prosail_obj),
        vectorize=False
    )

    # Burn‐in with tqdm progress bar
    print("Burn‐in:")
    sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()

    # Production with tqdm progress bar
    print("Production:")
    sampler.run_mcmc(None, N_STEPS, progress=True)

    flat_samples = sampler.get_chain(flat=True)
    med = np.median(flat_samples, axis=0)
    med_dict = {p: float(m) for p, m in zip(FLOAT_PARAMS, med)}
    return med_dict, flat_samples


# ----------------------------------------
# MAIN: Single‐Spectrum Debug with Plot
# ----------------------------------------

def main():
    # 1) Find the first CSV matching NDVI*.csv
    csv_files = sorted(glob.glob(FILE_PATTERN))
    if not csv_files:
        raise RuntimeError(f"No CSV files found matching {FILE_PATTERN}")
    filepath = csv_files[0]
    spec_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"Using file: {filepath}")

    # 2) Read CSV into DataFrame
    df = pd.read_csv(filepath)
    if "wavelength" not in df.columns:
        raise RuntimeError(f"'wavelength' column missing in {filepath}")
    orig_wls = df["wavelength"].values

    # 3) Identify all reflectance columns
    refl_cols = [c for c in df.columns if c != "wavelength"]
    if not refl_cols:
        raise RuntimeError(f"No reflectance columns in {filepath}")

    # 4) Build global valid‐wavelength mask across all reflectance columns
    prosail = Prosail()
    wl = np.array(prosail.wl)  # full PROSAIL wavelengths
    base_wl_mask = (wl >= 400) & (wl <= 2500)
    orig_mask = (orig_wls >= 400) & (orig_wls <= 2500)

    # Interpolate each reflectance column onto PROSAIL grid, collect into array
    stacked_interps = []
    for col in refl_cols:
        refl = df[col].values
        if not np.all(orig_mask):
            interp_col = np.interp(wl[base_wl_mask], orig_wls[orig_mask], refl[orig_mask])
        else:
            interp_col = np.interp(wl[base_wl_mask], orig_wls, refl)
        stacked_interps.append(interp_col)
    stacked_interps = np.vstack(stacked_interps)  # shape = (n_cols, n_base_wl)

    # valid0[i] = True if all columns are finite at band i
    valid0 = np.all(np.isfinite(stacked_interps), axis=0)

    # Build final PROSAIL mask of length len(wl): True where base_wl_mask & valid0
    global_wl_mask = np.zeros_like(base_wl_mask, dtype=bool)
    base_indices = np.where(base_wl_mask)[0]
    global_wl_mask[base_indices[valid0]] = True
    wl_indices_global = np.where(global_wl_mask)[0]

    if wl_indices_global.size == 0:
        raise RuntimeError("No valid PROSAIL wavelengths remain after dropping NaNs.")
    print(f"Valid PROSAIL wavelengths after dropping NaN bands: {wl_indices_global.size}")

    # 5) Parse column nearest to 40°
    col_to_angle = {}
    for c in refl_cols:
        m = re.search(r"(\d+)", c)
        if not m:
            raise RuntimeError(f"Cannot parse numeric angle from column '{c}'")
        col_to_angle[c] = int(m.group(1))
    col_40 = min(col_to_angle.keys(), key=lambda c: abs(col_to_angle[c] - TRUTH_ANGLE))
    print(f"Selected column for 40° truth: {col_40} (parsed angle={col_to_angle[col_40]}°)")

    refl_40 = df[col_40].values

    # 6) Interpolate 40° reflectance, subset by global mask
    if not np.all(orig_mask):
        full_interp40 = np.interp(wl[base_wl_mask], orig_wls[orig_mask], refl_40[orig_mask])
    else:
        full_interp40 = np.interp(wl[base_wl_mask], orig_wls, refl_40)
    measured40 = full_interp40[valid0]   # drop NaN bands
    wl_inds40 = wl_indices_global         # indices into prosail.wl

    if np.any(np.isnan(measured40)):
        raise RuntimeError("measured40 still contains NaNs after masking. Abort.")

    # 7) Run MCMC on this single 40° spectrum
    print("Running MCMC on single spectrum (40° truth)...")
    true_params, samples = run_mcmc_on_spectrum(measured40, wl_inds40)

    # 8) Print median parameter results
    print("\n--- MCMC Median Parameters (40° truth) ---")
    for p in FLOAT_PARAMS:
        print(f"{p:<8}: {true_params[p]:.4f}")
    print("-------------------------------------------")

    # 9) Compute fitted reflectance confidence intervals
    n_total = samples.shape[0]
    n_plot = min(N_SAMPLES_PLOT, n_total)
    idxs = np.random.choice(n_total, size=n_plot, replace=False)

    n_wl = wl_inds40.size
    fit_specs = np.zeros((n_plot, n_wl))

    print("Generating fit ensemble:")
    for i, idx in enumerate(tqdm(idxs, ncols=80)):
        theta = samples[idx]
        config = param_dict_from_vector(theta)
        sim_full = np.array(Prosail().run(config))
        fit_specs[i, :] = sim_full[wl_inds40]

    fit_mean = np.median(fit_specs, axis=0)
    fit_low = np.percentile(fit_specs, 5, axis=0)
    fit_high = np.percentile(fit_specs, 95, axis=0)

    # 10) Plot measured vs. fitted with confidence intervals
    fig, ax = plt.subplots(figsize=(6, 4))
    wl_plot = wl[wl_indices_global]

    ax.plot(wl_plot, measured40, color="black", linewidth=1.5, label="Measured")
    ax.plot(wl_plot, fit_mean, color="blue", linewidth=1.0, linestyle="--", label="MCMC Mean Fit")
    ax.fill_between(wl_plot, fit_low, fit_high, color="gray", alpha=0.4, label="90% Credible Interval")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"{spec_name} 40° Measured vs. MCMC Fit")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.7)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_filename = os.path.join(FIG_DIR, f"{spec_name}_40deg_mcmc_fit.png")
    plt.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    plt.close(fig)
    print(f"Spectral plot saved to: {fig_filename}")


if __name__ == "__main__":
    main()
