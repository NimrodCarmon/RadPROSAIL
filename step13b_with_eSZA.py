# driver_with_etts.py
"""
Kernel driver with TOA and slope-corrected solar zenith angles.

This script sets up surface kernels with various convexities, computes both:
- `tts`: Top-of-atmosphere solar zenith angle (assumed by EMIT)
- `etts`: Effective local solar zenith angle (computed from slope and sun geometry)

These values are passed into the PROSAIL driver as part of the config. This sets
up for future correction of direct irradiance entering the canopy using terrain-aware incidence.

Plots include:
- 3D terrain kernel
- Mean reflectance spectrum

Author: Nimrod Carmon
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prosail2 import Prosail
import time

def generate_kernel_geometry(kernel_size, convexity_param):
    """
    Generate synthetic topography and compute slope, aspect, and centered elevation.

    Parameters:
        kernel_size (int): Size of the square kernel (e.g., 5x5)
        convexity_param (float): Surface curvature parameter

    Returns:
        slope (ndarray): Slope angle per cell (degrees)
        aspect (ndarray): Aspect angle per cell (degrees)
        zz (ndarray): Elevation surface (mean-centered for plotting)
    """
    x = np.linspace(-1, 1, kernel_size)
    y = np.linspace(-1, 1, kernel_size)
    xx, yy = np.meshgrid(x, y)
    zz = -convexity_param * (xx**2 + yy**2)

    # Compute slope and aspect from original zz
    dz_dx, dz_dy = np.gradient(zz)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = (np.degrees(np.arctan2(-dz_dy, -dz_dx)) + 360) % 360

    # Normalize for consistent 3D plot perspective
    zz -= zz.mean()

    return slope, aspect, zz


def compute_etts(tts_deg, azimuth_deg, slope_deg, aspect_deg):
    """
    Compute effective solar zenith angle (etts) from terrain orientation.

    Parameters:
        tts_deg (float): TOA solar zenith angle
        azimuth_deg (float): solar azimuth angle (0=N, 180=S)
        slope_deg (ndarray): local slope per cell (deg)
        aspect_deg (ndarray): local aspect per cell (deg)

    Returns:
        etts_deg (ndarray): slope-corrected incidence angle per cell (deg)
    """
    sza_rad = np.radians(tts_deg)
    az_sun_rad = np.radians(azimuth_deg)
    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)

    cos_i = (
        np.cos(sza_rad) * np.cos(slope_rad) +
        np.sin(sza_rad) * np.sin(slope_rad) * np.cos(az_sun_rad - aspect_rad)
    )
    cos_i = np.clip(cos_i, -1, 1)
    return np.degrees(np.arccos(cos_i))

def run_prosail_kernel(kernel_size, convexity_param, tts=30, azimuth=180):
    """
    Run PROSAIL for a kernel, passing TOA and effective SZA for each cell.

    Parameters:
        kernel_size (int): kernel size (e.g. 5)
        convexity_param (float): dome/bowl shape
        tts (float): TOA solar zenith angle (deg)
        azimuth (float): sun azimuth angle (deg)

    Returns:
        mean_refl (ndarray): mean reflectance across kernel
        wavelengths (ndarray): wavelength vector (nm)
        surface (ndarray): elevation values for plotting
    """
    slope, aspect, surface = generate_kernel_geometry(kernel_size, convexity_param)
    etts = compute_etts(tts, azimuth, slope, aspect)

    prosail = Prosail()
    reflectances = []

    for i in range(kernel_size):
        for j in range(kernel_size):
            conf = {
                'N': 1.5,
                'Cab': 40,
                'Car': 8,
                'Cbrown': 0.5,
                'Cw': 0.015,
                'Cm': 0.009,
                'LAI': 2.5,
                'psoil': 0.5,
                'hspot': 0.01,
                'tts': tts,              # TOA solar zenith (used in RT)
                'etts': etts[i, j],      # slope-corrected effective solar zenith
                'tto': 0,
                'psi': 0,
                'LIDFa': -0.35,
                'LIDFb': -0.15
            }
            refl = prosail.run(conf)
            reflectances.append(refl)

    reflectances = np.array(reflectances)
    mean_refl = reflectances.mean(axis=0)
    wavelengths = np.linspace(400, 2500, mean_refl.shape[0])
    return mean_refl, wavelengths, surface, etts


# -----------------------
# Run and plot
# -----------------------

if __name__ == '__main__':
    convexity_params = [0.0, 2.0, 7.0, 15.0]
    kernel_size = 5
    surface_zlims = (-2.0, 0.5)

    fig = plt.figure(figsize=(16, 8))

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap('jet')

    for idx, convexity in enumerate(convexity_params):
        start = time.time()
        refl, wl, surface, etts = run_prosail_kernel(kernel_size, convexity)
        # Compute cosine(etts), globally normalized to [0, 1]
        cos_etts = np.clip(np.cos(np.radians(etts)), 0.0, 1.0)
        facecolors = cmap(norm(cos_etts))  # Apply consistent colormap scaling
        # Surface plot
        ax1 = fig.add_subplot(2, len(convexity_params), idx + 1, projection='3d')
        x = y = np.linspace(-1, 1, kernel_size)
        xx, yy = np.meshgrid(x, y)
        zlim_fixed = (-5, 15)

        ax1.plot_surface(xx, yy, surface, facecolors=facecolors, rstride=1, cstride=1)
        ax1.set_zlim(zlim_fixed)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.set_title(f'Convexity = {convexity}')
        ax1.view_init(elev=50, azim=25)



        # Reflectance plot
        ax2 = fig.add_subplot(2, len(convexity_params), idx + 1 + len(convexity_params))
        ax2.plot(wl, refl)
        ax2.set_ylim(0, 0.5)
        ax2.set_xlim(400, 2500)
        ax2.grid(True)
        ax2.set_title('Mean Reflectance')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Reflectance')

    plt.tight_layout()
    plt.savefig('figures/kernel_with_etts.jpg', dpi=150)
    plt.show()
