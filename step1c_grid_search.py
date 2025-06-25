"""
============================================
Parallel PROSAIL Inversion with dogbox Scaling, Fixed LIDF and psoil
============================================

- Synthesizes a random PROSAIL spectrum using all major parameters except
  hspot, tts, tto, psi, LIDFa, LIDFb, and psoil (all fixed).
- Coarse parallel LUT/grid search for robust initial guesses.
- Refines the N best grid points using least-squares ('dogbox') and x_scale.
- Plots and prints true vs. estimated parameters and spectral fit.

"""

import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail
from scipy.optimize import least_squares
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

# === PARAMETERS TO FLOAT, FIX, BOUNDS, GRID ===
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
    'psoil': 0.2  # <-- Fixed psoil value, change as desired
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
GRID_POINTS = {
    'N': 2,
    'LAI': 4,
    'Cab': 4,
    'Car': 2,
    'Cbrown': 2,
    'Cw': 2,
    'Cm': 2
}
N_BEST = 5  # Number of best grid points to refine

def random_param_vector():
    return np.array([
        np.random.uniform(PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1])
        for p in FLOAT_PARAMS
    ])

def param_dict_from_vector(param_vec):
    d = {p: v for p, v in zip(FLOAT_PARAMS, param_vec)}
    d.update(FIXED_PARAMS)
    return d

def get_grid_for_param(p):
    return np.linspace(PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1], GRID_POINTS[p])

def evaluate_grid_point(grid_pt, measured_spectrum, wl_mask):
    prosail = Prosail()
    config = param_dict_from_vector(grid_pt)
    sim = np.array(prosail.run(config))[wl_mask]
    cost = np.mean((measured_spectrum - sim) ** 2)
    return cost

def main():
    np.random.seed(42)
    prosail = Prosail()
    wl = np.array(prosail.wl)
    wl_mask = (wl >= 400) & (wl <= 2500)

    # --- Step 1: Generate a "true" parameter set and synthetic spectrum ---
    true_params_vec = random_param_vector()
    true_params = param_dict_from_vector(true_params_vec)
    measured_spectrum = np.array(prosail.run(true_params))[wl_mask]

    # --- Step 2: Build coarse LUT grid ---
    param_grids = [get_grid_for_param(p) for p in FLOAT_PARAMS]
    grid_points = list(itertools.product(*param_grids))
    print(f"Total grid points: {len(grid_points)}")

    # --- Step 3: Parallel grid search (with progress bar if available) ---
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    costs = np.zeros(len(grid_points))
    print("Evaluating LUT in parallel...")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_grid_point, grid_pt, measured_spectrum, wl_mask): idx
                   for idx, grid_pt in enumerate(grid_points)}
        if use_tqdm:
            for future in tqdm(as_completed(futures), total=len(futures), desc="LUT Grid Search"):
                idx = futures[future]
                costs[idx] = future.result()
        else:
            n_done = 0
            n_total = len(futures)
            update_every = max(1, n_total // 20)
            for future in as_completed(futures):
                idx = futures[future]
                costs[idx] = future.result()
                n_done += 1
                if n_done % update_every == 0 or n_done == n_total:
                    pct = 100 * n_done / n_total
                    print(f"Progress: {n_done}/{n_total} ({pct:.1f}%)")

    # --- Step 4: Take N_BEST best grid points as initial points for least squares ---
    best_idxs = np.argpartition(costs, N_BEST)[:N_BEST]
    print("\nTop N grid points:")
    for i, idx in enumerate(best_idxs):
        print(f"{i+1}: cost={costs[idx]:.5e} params={dict(zip(FLOAT_PARAMS, grid_points[idx]))}")

    # --- Step 5: Least squares refinement (dogbox, x_scale for scaling) ---
    bounds_lower = [PARAM_BOUNDS[p][0] for p in FLOAT_PARAMS]
    bounds_upper = [PARAM_BOUNDS[p][1] for p in FLOAT_PARAMS]
    x_scale = 'jac'  # Or: [hi-lo for (lo, hi) in [PARAM_BOUNDS[p] for p in FLOAT_PARAMS]]

    def residuals(param_vec):
        config = param_dict_from_vector(param_vec)
        sim_spec = np.array(prosail.run(config))[wl_mask]
        return measured_spectrum - sim_spec

    best_refined_cost = np.inf
    best_refined_vec = None

    for i, idx in enumerate(best_idxs):
        print(f"\nRefining grid point {i+1}/{N_BEST}...")
        x0 = np.array(grid_points[idx])
        result = least_squares(
            residuals, x0,
            bounds=(bounds_lower, bounds_upper),
            method='dogbox',
            x_scale=x_scale,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=2000,
            verbose=1
        )
        print(f"  Refined cost: {result.cost:.5e}")
        if result.cost < best_refined_cost:
            best_refined_cost = result.cost
            best_refined_vec = result.x

    estimated_vec = best_refined_vec
    estimated_params = param_dict_from_vector(estimated_vec)

    # --- Step 6: Print results and compare ---
    print("\n=== Parameter Inversion Results ===")
    print(f"{'Parameter':<10} {'True':>10} {'Estimated':>15} {'Abs Error':>10}")
    for p in FLOAT_PARAMS:
        print(f"{p:<10} {true_params[p]:>10.4f} {estimated_params[p]:>15.4f} {abs(true_params[p] - estimated_params[p]):>10.4f}")

    # --- Step 7: Plot per-parameter true vs. estimated (robust to # params) ---
    n_param = len(FLOAT_PARAMS)
    ncols = min(4, n_param)
    nrows = int(np.ceil(n_param / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axs = np.array(axs).ravel()
    for i, p in enumerate(FLOAT_PARAMS):
        axs[i].bar(['True', 'Est'], [true_params[p], estimated_params[p]], color=['gray','royalblue'])
        axs[i].set_title(p)
        axs[i].grid(True, alpha=0.3)
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    plt.savefig("inversion_param_barplots_FIXED_LIDF_psoil.jpg", dpi=200)
    plt.show()

    # --- Step 8: Plot spectrum fit ---
    plt.figure(figsize=(8,5))
    plt.plot(wl[wl_mask], measured_spectrum, label="Measured (True)", color='black', lw=2)
    plt.plot(wl[wl_mask], np.array(prosail.run(estimated_params))[wl_mask], '--', label="Best-fit", color='red')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.title("Spectral Fit: True vs. Best-fit Model")
    plt.tight_layout()
    plt.savefig("inversion_spectrum_fit_FIXED_LIDF_psoil.jpg", dpi=200)
    plt.show()

    print("\n✅ Done! Plots saved as 'inversion_param_barplots_FIXED_LIDF_psoil.jpg' and 'inversion_spectrum_fit_FIXED_LIDF_psoil.jpg'.")

if __name__ == "__main__":
    main()
