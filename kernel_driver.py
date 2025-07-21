# kernel_driver_plot.py
"""
PROSAIL Kernel Driver with Angular Geometry and Reflectance Visualization

This script performs spatially explicit simulations of canopy reflectance using PROSAIL
across kernels with synthetic surface topographies (dome, bowl, flat). It computes
local solar zenith angles per cell, runs PROSAIL for each, and visualizes both the
surface shape and directional reflectance. It also compares each case to the flat
surface by plotting spectral differences.

Author: Nimrod Carmon
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prosail2 import Prosail
import time

def generate_kernel_geometry(kernel_size, convexity_param):
    x = np.linspace(-1, 1, kernel_size)
    y = np.linspace(-1, 1, kernel_size)
    xx, yy = np.meshgrid(x, y)
    zz = -convexity_param * (xx**2 + yy**2)
    dz_dx, dz_dy = np.gradient(zz)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = (np.degrees(np.arctan2(-dz_dy, -dz_dx)) + 360) % 360
    return slope, aspect, zz

def compute_local_tts(global_tts_deg, global_azimuth_deg, slope_deg, aspect_deg):
    gtts = np.radians(global_tts_deg)
    gazim = np.radians(global_azimuth_deg)
    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)
    cos_i = (
        np.cos(gtts) * np.cos(slope_rad) +
        np.sin(gtts) * np.sin(slope_rad) * np.cos(gazim - aspect_rad)
    )
    cos_i = np.clip(cos_i, -1, 1)
    return np.degrees(np.arccos(cos_i))

def run_kernel_prosail(kernel_size, convexity_param, global_tts=30, global_azimuth=180):
    slope, aspect, surface_z = generate_kernel_geometry(kernel_size, convexity_param)
    tts_local = compute_local_tts(global_tts, global_azimuth, slope, aspect)
    
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
                'tts': tts_local[i, j],
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
    return mean_refl, wavelengths, surface_z

def plot_differences_to_flat(reflectance_list, wavelength_vector, convexity_params):
    """
    Plot reflectance spectral difference vs. flat (convexity = 0) kernel.
    """
    idx_flat = convexity_params.index(0.0)
    refl_flat = reflectance_list[idx_flat]

    plt.figure(figsize=(8, 5))
    for i, convexity in enumerate(convexity_params):
        if i == idx_flat:
            continue
        diff = reflectance_list[i] - refl_flat
        plt.plot(wavelength_vector, diff, label=f'Convexity = {convexity:+.1f}')
    
    plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
    plt.title('Reflectance Difference vs Flat Surface (Convexity = 0)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Δ Reflectance')
    plt.ylim(-0.01, 0.01)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/difference_vs_flat.jpg', dpi=150)
    plt.show()

# -------------------------------
# Run simulation and plot results
# -------------------------------

if __name__ == '__main__':
    convexity_params = [-1.5, 0.0, 0.5, 1.5]
    kernel_size = 5
    surface_zlims = (-2.0, 0.5)

    reflectance_list = []
    wavelength_vector = None

    fig = plt.figure(figsize=(16, 8))
    for idx, convexity in enumerate(convexity_params):
        start_time = time.time()
        refl, wl, surface = run_kernel_prosail(kernel_size, convexity)
        elapsed = time.time() - start_time
        print(f"Convexity = {convexity:>4}: kernel simulation took {elapsed:.3f} sec")

        reflectance_list.append(refl)
        if wavelength_vector is None:
            wavelength_vector = wl

        # 3D kernel surface
        ax1 = fig.add_subplot(2, len(convexity_params), idx + 1, projection='3d')
        x = y = np.linspace(-1, 1, kernel_size)
        xx, yy = np.meshgrid(x, y)
        ax1.plot_surface(xx, yy, surface, cmap='terrain')
        ax1.set_zlim(surface_zlims)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.set_title(f'Convexity = {convexity}')
        ax1.view_init(elev=30, azim=120)

        # Reflectance
        ax2 = fig.add_subplot(2, len(convexity_params), idx + 1 + len(convexity_params))
        ax2.plot(wl, refl)
        ax2.set_ylim(0, 0.5)
        ax2.set_xlim(400, 2500)
        ax2.grid(True)
        ax2.set_title('Reflectance')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Reflectance')

    plt.tight_layout()
    plt.savefig('figures/kernel_shapes_and_reflectances.jpg', dpi=150)
    plt.show()

    # Plot differences to flat
    plot_differences_to_flat(reflectance_list, wavelength_vector, convexity_params)
