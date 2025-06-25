"""
============================================================
PROSAIL Multi-Parameter Inversion with Least Squares (TRF)
============================================================

- Generates a synthetic "measured" spectrum with random PROSAIL parameters (except hspot, tts, tto, psi).
- Attempts to retrieve all the free parameters using scipy's least_squares (TRF solver).
- Compares and plots true vs. estimated parameters and spectral fit.

"""

import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
from scipy.optimize import least_squares

# --- Parameters to float ---
FLOAT_PARAMS = [
    'N', 'LAI', 'Cab', 'Car', 'Cbrown', 'Cw', 'Cm', 'psoil', 'LIDFa', 'LIDFb'
]
FIXED_PARAMS = {
    'hspot': 0.1, 'tts': 45, 'tto': 30, 'psi': 60
}
PARAM_BOUNDS = {
    'N':      (1.0, 3.0),
    'LAI':    (0.2, 8.0),
    'Cab':    (10, 90),
    'Car':    (1, 20),
    'Cbrown': (0, 1.0),
    'Cw':     (0.005, 0.06),
    'Cm':     (0.005, 0.03),
    'psoil':  (0.1, 0.6),
    'LIDFa':  (-2, 2),
    'LIDFb':  (-1, 1)
}

def random_param_vector():
    """Generate a random parameter vector within bounds."""
    return np.array([
        np.random.uniform(PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1])
        for p in FLOAT_PARAMS
    ])

def param_dict_from_vector(param_vec):
    """Convert vector to dict with PROSAIL param keys."""
    d = {p: v for p, v in zip(FLOAT_PARAMS, param_vec)}
    d.update(FIXED_PARAMS)
    return d

def main():
    np.random.seed(42)
    true_params_vec = random_param_vector()
    true_params = param_dict_from_vector(true_params_vec)

    prosail = Prosail()
    wl = np.array(prosail.wl)
    wl_mask = (wl >= 400) & (wl <= 2500)
    measured_spectrum = np.array(prosail.run(true_params))[wl_mask]

    # --- Define the residuals function for least_squares ---
    def residuals(param_vec):
        config = param_dict_from_vector(param_vec)
        sim_spec = np.array(prosail.run(config))[wl_mask]
        return measured_spectrum - sim_spec

    bounds_lower = [PARAM_BOUNDS[p][0] for p in FLOAT_PARAMS]
    bounds_upper = [PARAM_BOUNDS[p][1] for p in FLOAT_PARAMS]

    # Initial guess: mid-point of bounds
    x0 = np.array([(lo+hi)/2 for lo, hi in zip(bounds_lower, bounds_upper)])

    # --- Run least squares inversion using trust-region reflective solver ---
    result = least_squares(
        residuals, 
        x0, 
        bounds=(bounds_lower, bounds_upper),
        method='trf',
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        verbose=2, 
        max_nfev=2000
    )

    estimated_vec = result.x
    estimated_params = param_dict_from_vector(estimated_vec)

    # --- Print results and compare ---
    print("\n=== Parameter Inversion Results ===")
    print(f"{'Parameter':<10} {'True':>10} {'Estimated':>15} {'Abs Error':>10}")
    for p in FLOAT_PARAMS:
        print(f"{p:<10} {true_params[p]:>10.4f} {estimated_params[p]:>15.4f} {abs(true_params[p] - estimated_params[p]):>10.4f}")

    # --- Plot per-parameter true vs. estimated ---
    fig, axs = plt.subplots(2, len(FLOAT_PARAMS)//2, figsize=(16,7))
    axs = axs.ravel()
    for i, p in enumerate(FLOAT_PARAMS):
        axs[i].bar(['True', 'Est'], [true_params[p], estimated_params[p]], color=['gray','royalblue'])
        axs[i].set_title(p)
        axs[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("inversion_param_barplots_leastsq.jpg", dpi=200)
    plt.show()

    # --- Plot spectrum fit ---
    plt.figure(figsize=(8,5))
    plt.plot(wl[wl_mask], measured_spectrum, label="Measured (True)", color='black', lw=2)
    plt.plot(wl[wl_mask], np.array(prosail.run(estimated_params))[wl_mask], '--', label="Best-fit", color='red')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.title("Spectral Fit: True vs. Best-fit Model")
    plt.tight_layout()
    plt.savefig("inversion_spectrum_fit_leastsq.jpg", dpi=200)
    plt.show()

    print("\n✅ Done! Plots saved as 'inversion_param_barplots_leastsq.jpg' and 'inversion_spectrum_fit_leastsq.jpg'.")

if __name__ == "__main__":
    main()
