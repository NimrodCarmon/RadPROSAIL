import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

# --- Find all NDVI files in ../data ---
indir = '../data'
ndvi_files = sorted(glob.glob(os.path.join(indir, 'NDVI_*.csv')))

if not ndvi_files:
    print("No NDVI_*.csv files found in ../data/. Exiting.")
    sys.exit(1)

# --- Let user select file, or default to first ---
print("Found NDVI files:")
for i, f in enumerate(ndvi_files):
    print(f"  [{i}] {os.path.basename(f)}")
    
# Pick file by index or use first
try:
    idx = int(input(f"Select file index to plot [0-{len(ndvi_files)-1}]: ") or 0)
    input_file = ndvi_files[idx]
except Exception as e:
    print(f"Invalid input. Defaulting to {ndvi_files[0]}")
    input_file = ndvi_files[0]

print(f"\nPlotting spectra from: {input_file}\n")

# --- Load and plot spectra ---
df = pd.read_csv(input_file)
wavelengths = df['wavelength'].values
esza_cols = [col for col in df.columns if col.startswith('eSZA_')]

plt.figure(figsize=(9, 6))
for col in esza_cols:
    try:
        angle = int(col.split('_')[1])
    except Exception:
        angle = col
    plt.plot(wavelengths, df[col], label=f'eSZA {angle}°', alpha=0.8)

plt.title(f'All Reflectance Spectra\n{os.path.basename(input_file)}')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.xlim(wavelengths.min(), wavelengths.max())
plt.ylim(0, 1)
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plotname = 'all_spectra_' + os.path.basename(input_file).replace(".csv",".jpg")
plt.savefig(plotname, dpi=200)
plt.show()
print(f"✅ Plot saved as {plotname}")
