"""
=====================================================
Parallel MCMC Parameter Estimation for PROSAIL
=====================================================
- Generates 5 synthetic reflectance spectra using PROSAIL.
- Runs MCMC (emcee) on each, in parallel, to estimate parameter posteriors.
- For each spectrum: plots measured, mean MCMC fit, and credible interval.
- Plots posterior distributions and summary.
"""

import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
import emcee
import corner
import multiprocessing

# PARAMETERS TO SAMPLE
FLOAT_PARAMS = [
    'N', 'LAI', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm'
]
FIXED_PARAMS = {
    'hspot': 0.1,
    'tts': 45,
    'tto': 30,
    'psi': 60,
    'LIDFa': 0.66,
    'LIDFb': -0.04,
    'psoil': 0.2
}
PARAM_BOUNDS = {
    'N':      (1.0, 3.0),
    'LAI':    (0.2, 8.0),
    'Cab':    (10, 90),
    'Car':    (1, 20),
    'Cbrown': (0, 1.0),
    'Cw':     (0.005, 0.06),
    'Cm':     (0.005, 0.03)
}
# MCMC settings
N_SPECTRA = 5
N_WALKERS = 28
N_BURN = 400
N_STEPS = 1200

def random_param_vector():
    return np.array([
        np.random.uniform(PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1])
        for p in FLOAT_PARAMS
    ])

def param_dict_from_vector(param_vec):
    d = {p: v for p, v in zip(FLOAT_PARAMS, param_vec)}
    d.update(FIXED_PARAMS)
    return d

def lnprior(theta):
    for t, (lo, hi) in zip(theta, [PARAM_BOUNDS[p] for p in FLOAT_PARAMS]):
        if not (lo <= t <= hi):
            return -np.inf
    return 0.0

def lnlike(theta, measured_spectrum, wl_mask):
    config = param_dict_from_vector(theta)
    try:
        sim_spec = np.array(Prosail().run(config))[wl_mask]
    except Exception:
        return -np.inf
    sigma2 = 0.0025
    resid = measured_spectrum - sim_spec
    ll = -0.5 * np.sum(resid**2 / sigma2 + np.log(2 * np.pi * sigma2))
    return ll

def lnprob(theta, measured_spectrum, wl_mask):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, measured_spectrum, wl_mask)

def run_single_spectrum(idx, wl, wl_mask, measured_spectrum, true_params):
    # Each worker runs MCMC on a different spectrum
    ndim = len(FLOAT_PARAMS)
    p0 = [random_param_vector() for _ in range(N_WALKERS)]
    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim, lnprob,
        args=(measured_spectrum, wl_mask)
    )
    print(f"[Spectrum {idx}] Running burn-in...")
    pos, prob, state = sampler.run_mcmc(p0, N_BURN, progress=False)
    sampler.reset()
    print(f"[Spectrum {idx}] Running main MCMC...")
    sampler.run_mcmc(pos, N_STEPS, progress=False)

    samples = sampler.get_chain(flat=True)
    medians = np.median(samples, axis=0)
    estimated_params = param_dict_from_vector(medians)

    # Compute fit distributions
    n_samples_plot = min(400, len(samples))
    fit_specs = []
    for s in samples[np.random.choice(len(samples), n_samples_plot, replace=False)]:
        fit_specs.append(np.array(Prosail().run(param_dict_from_vector(s)))[wl_mask])
    fit_specs = np.array(fit_specs)
    fit_mean = np.mean(fit_specs, axis=0)
    fit_low = np.percentile(fit_specs, 5, axis=0)
    fit_high = np.percentile(fit_specs, 95, axis=0)

    return {
        "true_params": true_params,
        "estimated_params": estimated_params,
        "samples": samples,
        "fit_mean": fit_mean,
        "fit_low": fit_low,
        "fit_high": fit_high,
        "measured_spectrum": measured_spectrum,
        "wl": wl[wl_mask]
    }

def main():
    np.random.seed(42)
    prosail = Prosail()
    wl = np.array(prosail.wl)
    wl_mask = (wl >= 400) & (wl <= 2500)

    spectra_results = []
    param_vectors = [random_param_vector() for _ in range(N_SPECTRA)]
    all_true_params = [param_dict_from_vector(pv) for pv in param_vectors]
    all_measured = [np.array(prosail.run(tp))[wl_mask] for tp in all_true_params]

    # Run in parallel
    print(f"Launching {N_SPECTRA} parallel MCMC jobs ...")
    pool = multiprocessing.get_context("spawn").Pool(processes=min(N_SPECTRA, multiprocessing.cpu_count()))
    results = [
        pool.apply_async(run_single_spectrum, args=(i, wl, wl_mask, all_measured[i], all_true_params[i]))
        for i in range(N_SPECTRA)
    ]
    spectra_results = [r.get() for r in results]
    pool.close()
    pool.join()
    print("All MCMC jobs done.\n")

    # Plot posterior and fit for each spectrum
    for idx, res in enumerate(spectra_results):
        samples = res["samples"]
        medians = np.median(samples, axis=0)
        print(f"\n--- Spectrum {idx+1} ---")
        for i, p in enumerate(FLOAT_PARAMS):
            print(f"{p:<10} True: {res['true_params'][p]:.4f} | Median: {medians[i]:.4f}")

        # Corner plot
        fig = corner.corner(
            samples,
            labels=FLOAT_PARAMS,
            truths=[res['true_params'][p] for p in FLOAT_PARAMS],
            show_titles=True, title_fmt=".2f"
        )
        plt.savefig(f"mcmc_corner_plot_{idx+1}.jpg", dpi=180)
        plt.close(fig)

        # Spectral fit: credible interval
        plt.figure(figsize=(8,5))
        plt.plot(res['wl'], res['measured_spectrum'], label="Measured", color='black', lw=2)
        plt.plot(res['wl'], res['fit_mean'], '--', color='red', label="MCMC Mean")
        plt.fill_between(res['wl'], res['fit_low'], res['fit_high'], color='red', alpha=0.3, label="90% credible")
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title(f"Spectrum {idx+1}: True vs. MCMC Fit")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"mcmc_spectrum_fit_{idx+1}.jpg", dpi=180)
        plt.close()
        print(f"Spectrum {idx+1} plots saved.")

    print("\n✅ All spectra fitted, posteriors and fits plotted.")

if __name__ == "__main__":
    main()
