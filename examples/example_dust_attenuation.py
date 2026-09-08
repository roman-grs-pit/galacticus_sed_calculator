"""Compare emission-line and continuum dust attenuation for one galaxy."""
import numpy as np
import astropy.units as u
from galacticus_sed_calculator import SEDCalculator
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.cosmology import LambdaCDM

# Initialize the SED calculator with a template file
repository_root = Path(__file__).resolve().parents[1]
sed_template_file = repository_root / 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
galacticus_file = repository_root / 'data/romanUNIT.hdf5'

unit_cosmology = LambdaCDM(H0=67.74, Om0=0.3089, Ode0=0.6911)
calc = SEDCalculator(sed_template_file, cosmology=unit_cosmology)

# Define wavelength range for the spectrum
obs_wavelengths = np.linspace(8000, 30000, 2000) * u.AA

# Example parameters for the GB10 dust attenuation model. Scatter is disabled
# to make the result deterministic.
dust_params = {
    'delta_0': 0.275,      # Constant offset term
    'delta_z': -1.614,     # Redshift coefficient
    'delta_M': -0.834,     # Stellar mass coefficient  
    'delta_Mz': -0.708,    # Mass-redshift coupling coefficient
    'attenuation_scatter': 0.0
}

# The continuum model is configured separately from the emission-line model.
# This fixed A_V is illustrative rather than a universal choice.
continuum_dust = {
    'model': 'fixed_av',
    'params': {'A_V': 1.0},
    'law': 'calzetti',
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

# Generate a spectrum with emission-line attenuation only.
print("Generating spectrum with emission-line attenuation...")
spectrum_line_dust = calc.evaluate_component_spectrum(
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

# Add an independently configured attenuation model for the stellar continuum.
print("Generating spectrum with emission-line and continuum attenuation...")
spectrum_all_dust = calc.evaluate_component_spectrum(
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
    continuum_dust=continuum_dust,
    lineFWHM=50 * u.AA,
)

# Extract flux arrays for plotting
wavelengths_for_plot = np.linspace(8000, 30000, 2000) * u.AA
flux_no_dust = spectrum_no_dust(wavelengths_for_plot, flux_unit='flam')
flux_line_dust = spectrum_line_dust(wavelengths_for_plot, flux_unit='flam')
flux_all_dust = spectrum_all_dust(wavelengths_for_plot, flux_unit='flam')

# Plot the comparison
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(wavelengths_for_plot, flux_no_dust, label='No dust', alpha=0.7, linewidth=1)
plt.plot(
    wavelengths_for_plot,
    flux_line_dust,
    label='Emission-line dust',
    alpha=0.7,
    linewidth=1,
)
plt.plot(
    wavelengths_for_plot,
    flux_all_dust,
    label='Emission-line and continuum dust',
    alpha=0.7,
    linewidth=1,
)
plt.xlabel('Observed Wavelength (Å)')
plt.ylabel('Flux (erg/s/cm²/Å)')
plt.title(f'Galaxy Spectrum Comparison (z={galData["redshift"]:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot the attenuation factors. Avoid division warnings at wavelengths where
# both spectra have zero flux.
plt.subplot(2, 1, 2)
line_ratio = np.divide(
    flux_no_dust.value,
    flux_line_dust.value,
    out=np.full(flux_no_dust.shape, np.nan),
    where=flux_line_dust.value > 0,
)
all_ratio = np.divide(
    flux_no_dust.value,
    flux_all_dust.value,
    out=np.full(flux_no_dust.shape, np.nan),
    where=flux_all_dust.value > 0,
)
plt.plot(wavelengths_for_plot, line_ratio, label='Emission-line dust')
plt.plot(wavelengths_for_plot, all_ratio, label='Emission-line and continuum dust')
plt.xlabel('Observed Wavelength (Å)')
plt.ylabel('Flux Ratio (no dust / with dust)')
plt.title('Dust Attenuation Effect')
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No attenuation')
plt.legend()

plt.tight_layout()
output_path = Path(__file__).with_name('dust_attenuation_example.png')
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved as '{output_path}'")

print("\n" + "=" * 70)
print("Example complete!")
print("\nYou can customize the dust model parameters:")
print("  - delta_0: Constant offset")
print("  - delta_z: Redshift dependence")
print("  - delta_M: Stellar mass dependence")
print("  - delta_Mz: Mass-redshift coupling")
print("  - attenuation_scatter: normal scatter in attenuation (mags)")
print("  - continuum_dust: an independently configured continuum model")
print("=" * 70)
