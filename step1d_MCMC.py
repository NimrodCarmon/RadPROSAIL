"""
=====================================================
MCMC Parameter Estimation for PROSAIL using emcee
=====================================================
- Generates a synthetic reflectance spectrum using PROSAIL with fixed and floated parameters.
- Uses MCMC (emcee) to sample the parameter posterior distribution given the observed spectrum.
- Plots the parameter posteriors and the model fit.

"""

import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
import emcee
import corner

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

# CHAIN SETTINGS
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
    sigma2 = 0.0025  # Variance of noise term (assume very low noise)
    resid = measured_spectrum - sim_spec
    ll = -0.5 * np.sum(resid**2 / sigma2 + np.log(2 * np.pi * sigma2))
    return ll

def lnprob(theta, measured_spectrum, wl_mask):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, measured_spectrum, wl_mask)

def main():
    np.random.seed(42)
    prosail = Prosail()
    wl = np.array(prosail.wl)
    wl_mask = (wl >= 400) & (wl <= 2500)

    # -- Generate synthetic "measured" spectrum
    true_params_vec = random_param_vector()
    true_params = param_dict_from_vector(true_params_vec)
    measured_spectrum = np.array(prosail.run(true_params))[wl_mask]

    # -- MCMC setup
    ndim = len(FLOAT_PARAMS)
    p0 = [random_param_vector() for _ in range(N_WALKERS)]

    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim, lnprob,
        args=(measured_spectrum, wl_mask)
    )

    print("Running burn-in...")
    pos, prob, state = sampler.run_mcmc(p0, N_BURN, progress=True)
    sampler.reset()
    print("Running main MCMC...")
    sampler.run_mcmc(pos, N_STEPS, progress=True)

    # -- Postprocess
    samples = sampler.get_chain(flat=True)
    medians = np.median(samples, axis=0)
    estimated_params = param_dict_from_vector(medians)

    print("\n=== True vs. Median Estimated Parameters ===")
    for i, p in enumerate(FLOAT_PARAMS):
        print(f"{p:<10} True: {true_params[p]:.4f} | Median: {estimated_params[p]:.4f}")

    # -- Plot corner (parameter posterior)
    fig = corner.corner(
        samples,
        labels=FLOAT_PARAMS,
        truths=[true_params[p] for p in FLOAT_PARAMS],
        show_titles=True, title_fmt=".2f"
    )
    plt.savefig("mcmc_corner_plot.jpg", dpi=200)
    plt.show()

    # -- Plot spectral fit
    plt.figure(figsize=(8,5))
    plt.plot(wl[wl_mask], measured_spectrum, label="Measured (True)", color='black', lw=2)
    fit_spec = np.array(prosail.run(estimated_params))[wl_mask]
    plt.plot(wl[wl_mask], fit_spec, '--', label="MCMC Median Fit", color='red')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.title("Spectral Fit: True vs. MCMC Median Model")
    plt.tight_layout()
    plt.savefig("mcmc_spectrum_fit.jpg", dpi=200)
    plt.show()

    print("\n✅ MCMC completed! Plots saved as 'mcmc_corner_plot.jpg' and 'mcmc_spectrum_fit.jpg'.")

if __name__ == "__main__":
    main()
