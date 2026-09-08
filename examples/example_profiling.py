"""Measure the cost of generating spectra for a short sequence of galaxies.

The calculator is initialized once, as it would be for a catalogue-scale job,
and the script reports per-galaxy and total execution times.
"""

import numpy as np
import astropy.units as u
import time
import os
import sys

# Add parent directory to path to import galacticus_sed_calculator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator import SEDCalculator

# Setup (do this once)
example_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(example_dir)
sedTemplateFilename = os.path.join(parent_dir, "data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5")
galacticus_file = os.path.join(parent_dir, "data/romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5")

print("Initializing SED calculator...")
t0 = time.time()
sedCalc = SEDCalculator(sedTemplateFilename)
init_time = time.time() - t0
print(f"Initialization took {init_time:.4f} s\n")

obs_wavelengths = np.linspace(0.4, 2.5, 1000) * u.micron

# Profile a few galaxies
num_galaxies_to_test = 100
print(f"Profiling {num_galaxies_to_test} galaxies...")
print("-" * 60)

times = []
for galIndex in range(num_galaxies_to_test):
    t_start = time.time()
    
    total_flux = sedCalc.evaluate_total_spectrum(
        galacticus_file, 
        galIndex, 
        obs_wavelengths=obs_wavelengths
    )
    
    t_end = time.time()
    elapsed = t_end - t_start
    times.append(elapsed)
    
    if galIndex < 5 or galIndex == num_galaxies_to_test - 1:
        print(f"Galaxy {galIndex:4d}: {elapsed:.4f} s")

# Calculate statistics
times = np.array(times)
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)
print(f"Number of galaxies: {len(times)}")
print(f"Mean time: {np.mean(times):.4f} s ({np.mean(times)*1000:.1f} ms)")
print(f"Median time: {np.median(times):.4f} s ({np.median(times)*1000:.1f} ms)")
print(f"Std dev: {np.std(times):.4f} s")
print(f"Min time: {np.min(times):.4f} s")
print(f"Max time: {np.max(times):.4f} s")
print(f"\nTotal time for {num_galaxies_to_test} galaxies: {np.sum(times):.2f} s")
print(
    f"\nProjected time for 10,000 galaxies: "
    f"{np.mean(times) * 10000 / 60:.1f} minutes"
)
