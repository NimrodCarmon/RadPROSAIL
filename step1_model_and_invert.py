from prosail2 import Prosail
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def generate_dataset(n_samples=100):
    """Generate synthetic reflectance spectra with random LAI values."""
    p = Prosail()
    base_config = {
        'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
        'Cw': 0.03, 'Cm': 0.02,
        'psoil': 0.2, 'hspot': 0.1, 'tts': 45,
        'tto': 30, 'psi': 60, 'LIDFa': -1, 'LIDFb': 0
    }

    LAI_samples = np.random.uniform(0.5, 8.0, size=n_samples)
    spectra = []

    for lai in LAI_samples:
        config = base_config.copy()
        config['LAI'] = lai
        refl = p.run(config)
        spectra.append(refl)

    spectra = np.array(spectra)
    return p.wl, spectra, LAI_samples

def run_regression_and_plot(wl, spectra, LAI_true):
    """Fit linear model and plot predicted vs. true LAI."""
    model = LinearRegression()
    model.fit(spectra, LAI_true)
    LAI_pred = model.predict(spectra)
    r2 = r2_score(LAI_true, LAI_pred)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(LAI_true, LAI_pred, edgecolor='black', alpha=0.7)
    plt.plot([0, 9], [0, 9], 'r--', label='1:1 Line')
    plt.xlabel('True LAI')
    plt.ylabel('Estimated LAI')
    plt.title(f'Linear Model: True vs Estimated LAI\n$R^2$ = {r2:.3f}')
    plt.grid(True)
    plt.axis('square')
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.legend()
    plt.tight_layout()
    plt.savefig('LAI_estimation_scatter.jpg', dpi=300)
    plt.show()

if __name__ == "__main__":
    wl, spectra, LAI_true = generate_dataset(n_samples=100)
    run_regression_and_plot(wl, spectra, LAI_true)
