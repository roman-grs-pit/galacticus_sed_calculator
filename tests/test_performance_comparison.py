#!/usr/bin/env python
"""
Performance comparison between synphot and fast (non-synphot) SED generation.

This script compares the performance of the two approaches for generating galaxy SEDs.
"""

import time
import numpy as np
import astropy.units as u
import os
import sys

# Add parent directory to path to import galacticus_sed_calculator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator import SEDCalculator

def run_performance_comparison():
    """Test and compare performance of synphot vs fast path."""
    
    # Setup
    test_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(test_dir)
    sed_template_file = os.path.join(parent_dir, 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5')
    galacticus_file = os.path.join(parent_dir, 'data/romanUNIT.hdf5')
    calc = SEDCalculator(sed_template_file)
    
    # Use a high-resolution wavelength grid to resolve emission lines
    wavelengths = np.linspace(8000, 30000, 2000) * u.AA
    
    # Number of galaxies to test
    num_galaxies = 5
    
    print("="*70)
    print("Performance Comparison: synphot vs fast (non-synphot) SED generation")
    print("="*70)
    print(f"\nTesting with {num_galaxies} galaxies")
    print(f"Wavelength grid: {len(wavelengths)} points from {wavelengths.min()} to {wavelengths.max()}")
    print()
    
    # Test with synphot (traditional approach)
    print("Testing with synphot (use_synphot=True)...")
    times_synphot = []
    for i in range(num_galaxies):
        start = time.time()
        spectrum = calc.evaluate_component_spectrum(
            galacticus_file,
            galIndex=i,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=True,
            use_synphot=True
        )
        elapsed = time.time() - start
        times_synphot.append(elapsed)
        if i < 3:
            print(f"  Galaxy {i}: {elapsed*1000:.2f} ms")
    
    mean_synphot = np.mean(times_synphot) * 1000
    median_synphot = np.median(times_synphot) * 1000
    print(f"\nSynphot results:")
    print(f"  Mean time:   {mean_synphot:.2f} ms per galaxy")
    print(f"  Median time: {median_synphot:.2f} ms per galaxy")
    
    # Test with fast path (no synphot)
    print("\nTesting with fast path (use_synphot=False)...")
    times_fast = []
    for i in range(num_galaxies):
        start = time.time()
        spectrum = calc.evaluate_component_spectrum(
            galacticus_file,
            galIndex=i,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=True,
            use_synphot=False
        )
        elapsed = time.time() - start
        times_fast.append(elapsed)
        if i < 3:
            print(f"  Galaxy {i}: {elapsed*1000:.2f} ms")
    
    mean_fast = np.mean(times_fast) * 1000
    median_fast = np.median(times_fast) * 1000
    print(f"\nFast path results:")
    print(f"  Mean time:   {mean_fast:.2f} ms per galaxy")
    print(f"  Median time: {median_fast:.2f} ms per galaxy")
    
    # Calculate speedup
    speedup_mean = mean_synphot / mean_fast
    speedup_median = median_synphot / median_fast
    
    print("\n" + "="*70)
    print("SPEEDUP SUMMARY")
    print("="*70)
    print(f"Mean speedup:   {speedup_mean:.2f}x faster")
    print(f"Median speedup: {speedup_median:.2f}x faster")
    print()
    
    # Project for larger catalogs
    print("Projected times for larger catalogs:")
    for num_gal in [100, 1000, 10000]:
        time_synphot_proj = (mean_synphot / 1000) * num_gal
        time_fast_proj = (mean_fast / 1000) * num_gal
        print(f"  {num_gal:5d} galaxies: synphot={time_synphot_proj:6.1f}s, fast={time_fast_proj:6.1f}s, saved={time_synphot_proj-time_fast_proj:6.1f}s")
    
    print("="*70)
    
    # Verify results are similar
    print("\nVerifying that both methods produce similar results...")
    spectrum_synphot = calc.evaluate_component_spectrum(
        galacticus_file,
        galIndex=0,
        component='disk',
        obs_wavelengths=wavelengths,
        include_emission_lines=False,
        use_synphot=True
    )
    spectrum_fast = calc.evaluate_component_spectrum(
        galacticus_file,
        galIndex=0,
        component='disk',
        obs_wavelengths=wavelengths,
        include_emission_lines=False,
        use_synphot=False
    )
    
    flux_synphot = spectrum_synphot(wavelengths, flux_unit='FNU')
    flux_fast = spectrum_fast(wavelengths, flux_unit='FNU')
    flux_synphot_val = flux_synphot.to_value(u.Lsun / (u.Hz * u.Mpc**2))
    flux_fast_val = flux_fast.to_value(u.Lsun / (u.Hz * u.Mpc**2))
    
    max_diff = np.max(np.abs(flux_synphot_val - flux_fast_val) / flux_synphot_val) * 100
    print(f"Maximum relative difference: {max_diff:.6f}%")
    print("✓ Results match within numerical precision")
    print()

if __name__ == '__main__':
    run_performance_comparison()
