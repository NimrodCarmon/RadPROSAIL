# brdf_polar_driver.py
"""
BRDF-style Polar Plots of Directional Reflectance using PROSAIL

This script generates polar plots of reflectance for:
1. Chlorophyll absorption band (~680 nm)
2. NIR plateau (~850 nm)

For each kernel convexity (dome, flat, bowl), it computes directional reflectance
as a function of view zenith angle (VZA) and relative azimuth angle (RAA), using
local solar incidence angles derived from kernel shape.

Reflectance is visualized as BRDF-style polar plots.

Author: Nimrod Carmon
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from prosail2 import Prosail

def compute_local_tts(tts_sun, az_sun, slope_deg, aspect_deg):
    """Compute local tts for each surface facet."""
    gtts = np.radians(tts_sun)
    gazim = np.radians(az_sun)
    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)
    cos_i = (
        np.cos(gtts) * np.cos(slope_rad) +
        np.sin(gtts) * np.sin(slope_rad) * np.cos(gazim - aspect_rad)
    )
    cos_i = np.clip(cos_i, -1, 1)
    return np.degrees(np.arccos(cos_i))

def generate_slope_aspect_from_convexity(convexity, kernel_size=5):
    x = np.linspace(-1, 1, kernel_size)
    y = np.linspace(-1, 1, kernel_size)
    xx, yy = np.meshgrid(x, y)
    zz = -convexity * (xx**2 + yy**2)
    dz_dx, dz_dy = np.gradient(zz)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = (np.degrees(np.arctan2(-dz_dy, -dz_dx)) + 360) % 360
    return slope.mean(), aspect.mean()

def run_prosail_directional(tts, tto, psi):
    """Run PROSAIL for a given sun-view geometry."""
    prosail = Prosail()
    conf = {
        'N': 1.5,
        'Cab': 40,
        'Car': 8,
        'Cbrown': 0.5,
        'Cw': 0.015,
        'Cm': 0.009,
        'LAI': 3.5,
        'psoil': 0.5,
        'hspot': 0.01,
        'tts': tts,
        'tto': tto,
        'psi': psi,
        'LIDFa': -0.35,
        'LIDFb': -0.15
    }
    return prosail.run(conf)

def generate_polar_grid(vza_angles, raa_angles, tts_local):
    """Generate a BRDF-like angular grid of reflectance."""
    reflectance_680 = np.zeros((len(vza_angles), len(raa_angles)))
    reflectance_850 = np.zeros((len(vza_angles), len(raa_angles)))

    for i, vza in enumerate(vza_angles):
        for j, raa in enumerate(raa_angles):
            refl = run_prosail_directional(tts=tts_local, tto=vza, psi=raa)
            reflectance_680[i, j] = refl[wl_to_idx(680)]
            reflectance_850[i, j] = refl[wl_to_idx(850)]

    return reflectance_680, reflectance_850

def wl_to_idx(wl_target):
    """Assume 400–2500 nm over N bands — map wavelength to index."""
    wl = np.linspace(400, 2500, 2101)
    return int(np.argmin(np.abs(wl - wl_target)))

def plot_polar_brdf(data, raa_deg, vza_deg, title, cmap='viridis'):
    """Plot polar reflectance map."""
    RAA, VZA = np.meshgrid(np.radians(raa_deg), vza_deg)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
    pcm = ax.pcolormesh(RAA, VZA, data, shading='auto', cmap=cmap)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title)
    fig.colorbar(pcm, ax=ax, shrink=0.8)
    return fig

# -------------------
# MAIN EXECUTION
# -------------------

if __name__ == '__main__':
    convexity_list = [-1.0, 0.0, 1.0]
    vza_angles = np.linspace(0, 60, 20)  # View zenith angle
    raa_angles = np.linspace(0, 180, 36) # Relative azimuth

    for convexity in convexity_list:
        slope, aspect = generate_slope_aspect_from_convexity(convexity)
        tts_local = compute_local_tts(tts_sun=30, az_sun=180, slope_deg=slope, aspect_deg=aspect)

        r680, r850 = generate_polar_grid(vza_angles, raa_angles, tts_local)

        fig1 = plot_polar_brdf(r680, raa_angles, vza_angles, f'Chl Absorption (680 nm)\nConvexity = {convexity}')
        fig2 = plot_polar_brdf(r850, raa_angles, vza_angles, f'NIR Plateau (850 nm)\nConvexity = {convexity}')
        fig1.savefig(f'figures/polar_680_convexity_{convexity:+.1f}.jpg', dpi=150)
        fig2.savefig(f'figures/polar_850_convexity_{convexity:+.1f}.jpg', dpi=150)
        plt.close(fig1)
        plt.close(fig2)
