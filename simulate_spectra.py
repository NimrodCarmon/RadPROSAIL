from prosail2 import Prosail
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def simulate_spectra_variable_LAI():
    p = Prosail()

    base_config = {
        'N': 1.5, 'Cab': 45, 'Car': 8, 'Cbrown': 0.2,
        'Cw': 0.03, 'Cm': 0.02, 'LAI': 2,
        'psoil': 0.2, 'hspot': 0.1, 'tts': 45,
        'tto': 30, 'psi': 60, 'LIDFa': -1, 'LIDFb': 0
    }

    LAI_values = np.arange(0.5, 8.5, 0.5)
    cmap = mcolors.LinearSegmentedColormap.from_list('LAI_colormap', ['#90EE90', '#006400'])

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, LAI in enumerate(LAI_values):
        config = base_config.copy()
        config['LAI'] = LAI
        spectrum = p.run(config)
        color = cmap(i / (len(LAI_values) - 1))
        ax.plot(p.wl, spectrum, color=color, label=f'LAI={LAI:.1f}')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=LAI_values.min(), vmax=LAI_values.max()))
    sm.set_array([])  # Dummy array to avoid warning
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('LAI')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('HDRF')
    ax.set_ylim(0, 0.3)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('Spectra_LAI.jpg')

if __name__ == "__main__":
    simulate_spectra_variable_LAI()
