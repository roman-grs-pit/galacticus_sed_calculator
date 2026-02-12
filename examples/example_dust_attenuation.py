"""
Example demonstrating dust attenuation for emission lines.

This example shows how to apply dust attenuation when generating galaxy spectra
using the GB10 (Garn & Best 2010) generalized model and Calzetti attenuation law.
"""
import numpy as np
import astropy.units as u
from galacticus_sed_calculator import SEDCalculator
import matplotlib.pyplot as plt

# Initialize the SED calculator with a template file
sed_template_file = '../data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
galacticus_file = '../data/romanUNIT.hdf5'

calc = SEDCalculator(sed_template_file)

# Define wavelength range for the spectrum
obs_wavelengths = np.linspace(8000, 30000, 2000) * u.AA

# Parameters for the GB10 dust attenuation model
# These values are from the issue example
dust_params = {
    'delta_0': 0.275,      # Constant offset term
    'delta_z': -1.614,     # Redshift coefficient
    'delta_M': -0.834,     # Stellar mass coefficient  
    'delta_Mz': -0.708,    # Mass-redshift coupling coefficient
    'attenuation_scatter': 0.0  # No scatter for this example
}

# Select a galaxy to analyze
galaxy_index = 0

print("=" * 70)
print("Dust Attenuation Example")
print("=" * 70)

# Read galaxy properties
galData = calc.read_galacticus_galaxy(galacticus_file, galaxy_index)
print(f"\nGalaxy {galaxy_index}:")
print(f"  Redshift: {galData['redshift']:.3f}")

# Generate spectrum WITHOUT dust attenuation
print("\nGenerating spectrum without dust attenuation...")
spectrum_no_dust = calc.evaluate_component_spectrum(
    galacticus_file,
    galIndex=galaxy_index,
    component='disk',
    obs_wavelengths=obs_wavelengths,
    include_emission_lines=True,
    minimumLineWavelength=0.0 * u.micron,
    maximumLineWavelength=10.0 * u.micron,
    lineFWHM=50 * u.AA  # Wider lines for visualization
)

# Generate spectrum WITH dust attenuation
print("Generating spectrum with dust attenuation (GB10 model)...")
spectrum_with_dust = calc.evaluate_component_spectrum(
    galacticus_file,
    galIndex=galaxy_index,
    component='disk',
    obs_wavelengths=obs_wavelengths,
    include_emission_lines=True,
    minimumLineWavelength=0.0 * u.micron,
    maximumLineWavelength=10.0 * u.micron,
    dust_model='gb10_generalised',
    dust_params=dust_params,
    dust_law='calzetti',
    lineFWHM=50 * u.AA
)

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
print("  - attenuation_scatter: Log-normal scatter (optional)")
print("=" * 70)
