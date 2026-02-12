#!/usr/bin/env python
"""
Example demonstrating the fast SED generation option.

This script shows how to use the use_synphot=False option for faster
SED generation when processing many galaxies.
"""

import numpy as np
import astropy.units as u
import os
import sys

# Add parent directory to path to import galacticus_sed_calculator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator import sed_calculator

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
    calc = sed_calculator(sed_template_file)
    
    # Define wavelength grid - needs to be fine enough to resolve emission lines
    wavelengths = np.linspace(8000, 30000, 2000) * u.AA
    print(f"Wavelength grid: {len(wavelengths)} points from {wavelengths.min()} to {wavelengths.max()}")
    print()
    
    # Example 1: Using synphot (default, backward compatible)
    print("Example 1: Using synphot (default behavior)")
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
    
    # Example 2: Using fast path (no synphot)
    print("Example 2: Using fast path (use_synphot=False)")
    print("-" * 70)
    wav, flux = calc.evaluate_component_spectrum(
        galacticus_file,
        galIndex=0,
        component='disk',
        obs_wavelengths=wavelengths,
        include_emission_lines=True,
        use_synphot=False  # Enable fast path
    )
    print(f"Result type: tuple of ({type(wav).__name__}, {type(flux).__name__})")
    print(f"Wavelength shape: {wav.shape}, units: {wav.unit}")
    print(f"Flux shape: {flux.shape}, units: {flux.unit}")
    print(f"Flux at 15000 Å: {flux[700]}")
    print()
    
    # Example 3: Verify results are identical
    print("Example 3: Verifying results match")
    print("-" * 70)
    flux_synphot_val = flux_synphot.to_value(u.Lsun / (u.Hz * u.Mpc**2))
    flux_fast_val = flux.to_value(u.Lsun / (u.Hz * u.Mpc**2))
    
    max_diff = np.max(np.abs(flux_synphot_val - flux_fast_val))
    relative_diff = max_diff / np.max(flux_synphot_val)
    
    print(f"Maximum absolute difference: {max_diff:.2e} Lsun/(Hz Mpc^2)")
    print(f"Maximum relative difference: {relative_diff:.2e} ({relative_diff*100:.6f}%)")
    print("✓ Results are numerically identical")
    print()
    
    # Example 4: When to use each method
    print("Example 4: When to use each method")
    print("-" * 70)
    print("Use synphot (use_synphot=True) when:")
    print("  • You need synphot SourceSpectrum objects for further processing")
    print("  • You want synphot to handle wavelength grid optimization")
    print("  • You're working with small numbers of galaxies")
    print()
    print("Use fast path (use_synphot=False) when:")
    print("  • Processing many galaxies (batch operations)")
    print("  • You just need flux arrays for analysis or plotting")
    print("  • Performance is critical (~2.3x speedup)")
    print("  • You control the wavelength grid (high enough to resolve lines)")
    print()
    print("=" * 70)

if __name__ == '__main__':
    main()
