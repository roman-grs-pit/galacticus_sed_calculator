"""
Example demonstrating dust attenuation for emission lines.

This example shows how to apply dust attenuation when generating galaxy spectra
using the generalized GB10 (Garn & Best 2010) model and Calzetti attenuation law.

It also demonstrates how to read the dust model configuration back from a
Galacticus catalog that was processed with calculate_catalog_magnitudes.py
using --dust-config.
"""
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import numpy as np

# import SEDfromSFH as sed - have changed structure of repository so this is now imported from galacticus_sed_calculator
from galacticus_sed_calculator import SEDCalculator
from galacticus_sed_calculator import read_dust_model_from_catalog

# Initialize the SED calculator with a template file
sed_template_file = '../data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
galacticus_file = '../data/romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5'
unit = FlatLambdaCDM(H0=67.74, Om0=0.3089)
sedCalc = SEDCalculator(sed_template_file, cosmology=unit)

# Define wavelength range for the spectrum
obs_wavelengths = np.linspace(0.5, 2.5, 1000)*u.micron #wavelengths to produce sed over
high_res_wavelengths = np.linspace(1e4, 2e4, 5000)*u.angstrom

# ---------------------------------------------------------------------------
# Option A: read dust model parameters from the catalog
# (works when the catalog was processed with calculate_catalog_magnitudes.py
#  using --dust-config)
# ---------------------------------------------------------------------------
try:
    dust_model_specs = read_dust_model_from_catalog(galacticus_file)
    print("Dust model loaded from catalog:")
except KeyError:
    # Option B: fall back to manually specified parameters if the catalog has
    # not yet been processed with --dust-config
    print("No dust model found in catalog – using hard-coded parameters.")
    dust_model_specs = {
        'dust_model': 'gb10_generalised',
        'dust_params': {'delta_0': 0.2772142287561473,
                        'delta_z': -1.579233727951697,
                        'delta_M': -0.8180063760711892,
                        'delta_Mz': -0.5287832073987419,
                        'attenuation_scatter': 0.25},
        'dust_law': 'calzetti',
        'random_uniform_index': None,
    }

print(f"  dust_model : {dust_model_specs['dust_model']}")
print(f"  dust_law   : {dust_model_specs['dust_law']}")
print(f"  dust_params: {dust_model_specs['dust_params']}")
if dust_model_specs['random_uniform_index'] is not None:
    print(f"  random_uniform_index: {dust_model_specs['random_uniform_index']}")

# Select a galaxy to analyze
galaxy_index = 1

# Read galaxy properties
galData = sedCalc.read_galacticus_galaxy(galacticus_file, galaxy_index)
print(f"\nGalaxy {galaxy_index}:")
print(f"Redshift: {galData['redshift']:.3f}")

# Generate spectrum WITHOUT dust attenuation
print("\nGenerating spectrum without dust attenuation...")
spectrum_no_dust = sedCalc.evaluate_total_spectrum(galacticus_file, galaxy_index, 
                        obs_wavelengths=obs_wavelengths,
                        use_synphot=False)

# Generate spectrum WITH dust attenuation using the loaded dust_model_specs.
# The behaviour depends on the value of random_uniform_index in dust_model_specs:
#   - If random_uniform_index is an integer (e.g. loaded from catalog or set
#     manually), the scatter is drawn from the pre-stored uniform random numbers
#     in the Galacticus catalog, giving fully reproducible results across runs.
#   - If random_uniform_index is None and attenuation_scatter > 0, a fresh
#     random draw is made each run, so spectra will differ between runs.
print("\nGenerating spectrum with dust attenuation...")
spectrum_with_dust = sedCalc.evaluate_total_spectrum(galacticus_file, galaxy_index,
                        obs_wavelengths=obs_wavelengths,
                        use_synphot=False,
                        **dust_model_specs)

# Extract flux arrays for plotting
wavelengths_for_plot = np.linspace(8000, 30000, 2000) * u.AA
flux_no_dust = spectrum_no_dust(wavelengths_for_plot, flux_unit='flam')
flux_with_dust = spectrum_with_dust(wavelengths_for_plot, flux_unit='flam')

# Plot the comparison
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(wavelengths_for_plot, flux_no_dust, label='No dust', alpha=0.7, linewidth=1)
plt.plot(wavelengths_for_plot, flux_with_dust, label='With dust (GB10)', alpha=0.7, linewidth=1)
plt.xlabel('Observed Wavelength (Å)')
plt.ylabel('Flux (erg/s/cm²/Å)')
plt.title(f'Galaxy Spectrum Comparison (z={galData["redshift"]:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot the ratio (attenuation factor)
plt.subplot(2, 1, 2)
ratio = flux_no_dust / flux_with_dust
plt.plot(wavelengths_for_plot, ratio, color='red', linewidth=1)
plt.xlabel('Observed Wavelength (Å)')
plt.ylabel('Flux Ratio (no dust / with dust)')
plt.title('Dust Attenuation Effect')
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No attenuation')
plt.legend()

plt.tight_layout()
plt.savefig('dust_attenuation_example.png', dpi=150)
print("\nPlot saved as 'dust_attenuation_example.png'")

print("\n" + "=" * 70)
print("Example complete!")
print("\nYou can customize the dust model parameters:")
print("  - delta_0: Constant offset")
print("  - delta_z: Redshift dependence")
print("  - delta_M: Stellar mass dependence")
print("  - delta_Mz: Mass-redshift coupling")
print("  - attenuation_scatter: normal scatter in attenuation (mags)")
print("=" * 70)
