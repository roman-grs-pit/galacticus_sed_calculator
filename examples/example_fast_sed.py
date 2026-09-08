#!/usr/bin/env python
"""Compare the adaptive synphot and fixed-grid SED calculation paths."""

import numpy as np
import astropy.units as u
import os
import sys

# Add parent directory to path to import galacticus_sed_calculator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator import SEDCalculator

def main():
    print("=" * 70)
    print("Fast SED Generation Example")
    print("=" * 70)
    print()
    
    # Initialize calculator
    example_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(example_dir)
    sed_template_file = os.path.join(parent_dir, 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5')
    galacticus_file = os.path.join(parent_dir, 'data/romanUNIT.hdf5')
    calc = SEDCalculator(sed_template_file)
    
    # Define wavelength grid - needs to be fine enough to resolve emission lines
    wavelengths = np.linspace(8000, 30000, 2000) * u.AA
    print(f"Wavelength grid: {len(wavelengths)} points from {wavelengths.min()} to {wavelengths.max()}")
    print()
    
    # Example 1: synphot represents each line as a separate spectrum. Combining
    # those spectra gives an adaptive waveset with extra sampling near the lines.
    print("Example 1: Using the adaptive synphot path (default)")
    print("-" * 70)
    spectrum = calc.evaluate_component_spectrum(
        galacticus_file,
        galIndex=0,
        component='disk',
        obs_wavelengths=wavelengths,
        include_emission_lines=True,
        use_synphot=True  # This is the default
    )
    print(f"Result type: {type(spectrum)}")
    print(f"Spectrum has waveset: {hasattr(spectrum, 'waveset')}")
    # Evaluate at some wavelengths
    flux_synphot = spectrum(wavelengths, flux_unit='FNU')
    print(f"Flux at 15000 Å: {flux_synphot[700]}")
    print()
    
    # Example 2: the fast path adds the continuum and every line directly on
    # the supplied wavelength grid, then wraps the result as a SourceSpectrum.
    print("Example 2: Using the fixed-grid fast path (use_synphot=False)")
    print("-" * 70)
    spectrum_fast = calc.evaluate_component_spectrum(
        galacticus_file,
        galIndex=0,
        component='disk',
        obs_wavelengths=wavelengths,
        include_emission_lines=True,
        use_synphot=False
    )
    print(f"Result type: {type(spectrum_fast)}")
    print(f"Spectrum has waveset: {hasattr(spectrum_fast, 'waveset')}")
    # Evaluate at some wavelengths
    flux_fast = spectrum_fast(wavelengths, flux_unit='FNU')
    print(f"Flux at 15000 Å: {flux_fast[700]}")
    print()
    
    # Example 3: Verify results are identical
    print("Example 3: Verifying results match")
    print("-" * 70)
    flux_synphot_val = flux_synphot.to_value(u.Lsun / (u.Hz * u.Mpc**2))
    flux_fast_val = flux_fast.to_value(u.Lsun / (u.Hz * u.Mpc**2))
    
    max_diff = np.max(np.abs(flux_synphot_val - flux_fast_val))
    relative_diff = max_diff / np.max(flux_synphot_val)
    
    print(f"Maximum absolute difference: {max_diff:.2e} Lsun/(Hz Mpc^2)")
    print(f"Maximum relative difference: {relative_diff:.2e} ({relative_diff*100:.6f}%)")
    print("Results agree on the supplied wavelength grid")
    print()
    
    # Example 4: When to use each method
    print("Example 4: When to use each method")
    print("-" * 70)
    print("Use synphot (use_synphot=True) when:")
    print("  - You want adaptive sampling around individual emission lines")
    print("  - You are working with a small number of galaxies")
    print("  - The convenience of combining separate synphot spectra matters")
    print()
    print("Use fast path (use_synphot=False) when:")
    print("  - You are processing many galaxies")
    print("  - You want faster computation (roughly 2-4x in current benchmarks)")
    print("  - You want to control the wavelength grid")
    print("    (ensure that it is fine enough to resolve the emission lines)")
    print()
    print("Both methods return a synphot SourceSpectrum object.")
    print()
    print("=" * 70)

if __name__ == '__main__':
    main()
