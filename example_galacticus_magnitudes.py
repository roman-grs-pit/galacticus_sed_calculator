#!/usr/bin/env python3
"""
Example showing how to calculate magnitudes for Galacticus galaxies.

This demonstrates integration of synphot magnitude calculation with
the existing sed_calculator class and Galacticus data.
"""

import numpy as np
from synphot import SpectralElement, Observation
from synphot.models import Empirical1D
import astropy.units as u
from SEDfromSFH import sed_calculator


def create_roman_filter(filter_name):
    """
    Create a simplified Roman WFI filter.
    
    In production, this should load actual transmission curves from files.
    
    Parameters
    ----------
    filter_name : str
        Filter name (e.g., 'F158')
        
    Returns
    -------
    bandpass : SpectralElement
        The bandpass filter
    """
    # Simplified filter specifications
    filter_specs = {
        'F062': (6200 * u.AA, 2800 * u.AA),
        'F087': (8690 * u.AA, 2200 * u.AA),
        'F106': (10600 * u.AA, 2600 * u.AA),
        'F129': (12930 * u.AA, 3200 * u.AA),
        'F146': (14640 * u.AA, 3400 * u.AA),
        'F158': (15770 * u.AA, 3900 * u.AA),
        'F184': (18420 * u.AA, 3200 * u.AA),
        'F213': (21250 * u.AA, 3500 * u.AA),
    }
    
    if filter_name not in filter_specs:
        raise ValueError(f"Unknown filter: {filter_name}")
    
    center_wave, width = filter_specs[filter_name]
    wave_min = center_wave - width/2
    wave_max = center_wave + width/2
    
    # Create wavelength array
    wavelengths = np.linspace(wave_min.value * 0.8, wave_max.value * 1.2, 200) * center_wave.unit
    
    # Create transmission (simplified box filter with 80% peak transmission)
    transmission = np.where(
        (wavelengths >= wave_min) & (wavelengths <= wave_max),
        0.8,
        0.0
    )
    
    bandpass = SpectralElement(Empirical1D, points=wavelengths, lookup_table=transmission)
    return bandpass


def calculate_galaxy_magnitudes(sed_calc, filename, gal_index, filters, 
                                magnitude_system='AB', include_agn=True):
    """
    Calculate observed magnitudes for a Galacticus galaxy.
    
    Parameters
    ----------
    sed_calc : sed_calculator
        The SED calculator instance
    filename : str
        Path to Galacticus HDF5 file
    gal_index : int
        Galaxy index in the catalog
    filters : dict
        Dictionary mapping filter names to SpectralElement objects
    magnitude_system : str, optional
        'AB', 'ST', or 'Vega' (default 'AB')
    include_agn : bool, optional
        Whether to include AGN component (default True)
        
    Returns
    -------
    results : dict
        Dictionary with:
        - 'magnitudes': dict of filter_name -> magnitude
        - 'redshift': galaxy redshift
        - 'effective_wavelengths': dict of filter_name -> effective wavelength
    """
    # Get galaxy data
    gal_data = sed_calc.read_galacticus_galaxy(filename, gal_index)
    redshift = gal_data['redshift']
    
    # Get total spectrum
    obs_wavelengths = np.linspace(4000, 23000, 2000) * u.AA
    spectrum = sed_calc.evaluate_total_spectrum(
        filename=filename,
        galIndex=gal_index,
        includeAGN=include_agn,
        obs_wavelengths=obs_wavelengths
    )
    
    # Calculate magnitudes in all filters
    magnitudes = {}
    effective_wavelengths = {}
    
    for filter_name, bandpass in filters.items():
        try:
            # Create observation with taper to handle partial overlap
            obs = Observation(spectrum, bandpass, force='taper')
            
            # Calculate magnitude
            if magnitude_system == 'AB':
                mag = obs.effstim(flux_unit=u.ABmag)
            elif magnitude_system == 'ST':
                mag = obs.effstim(flux_unit=u.STmag)
            elif magnitude_system == 'Vega':
                from synphot import SourceSpectrum
                vega = SourceSpectrum.from_vega()
                mag = obs.effstim(flux_unit='vegamag', vegaspec=vega)
            else:
                raise ValueError(f"Unknown magnitude system: {magnitude_system}")
            
            magnitudes[filter_name] = mag.value
            effective_wavelengths[filter_name] = obs.effective_wavelength().value
            
        except Exception as e:
            print(f"Warning: Failed to calculate {filter_name} magnitude: {e}")
            magnitudes[filter_name] = np.nan
            effective_wavelengths[filter_name] = np.nan
    
    return {
        'magnitudes': magnitudes,
        'redshift': redshift,
        'effective_wavelengths': effective_wavelengths
    }


def calculate_component_magnitudes(sed_calc, filename, gal_index, filters, 
                                   component='disk', magnitude_system='AB'):
    """
    Calculate observed magnitudes for a specific galaxy component.
    
    Parameters
    ----------
    sed_calc : sed_calculator
        The SED calculator instance
    filename : str
        Path to Galacticus HDF5 file
    gal_index : int
        Galaxy index in the catalog
    filters : dict
        Dictionary mapping filter names to SpectralElement objects
    component : str, optional
        'disk', 'spheroid', or 'AGN' (default 'disk')
    magnitude_system : str, optional
        'AB', 'ST', or 'Vega' (default 'AB')
        
    Returns
    -------
    magnitudes : dict
        Dictionary mapping filter names to magnitudes
    """
    # Get component spectrum
    obs_wavelengths = np.linspace(4000, 23000, 2000) * u.AA
    spectrum = sed_calc.evaluate_component_spectrum(
        filename=filename,
        galIndex=gal_index,
        component=component,
        obs_wavelengths=obs_wavelengths
    )
    
    # Calculate magnitudes
    magnitudes = {}
    for filter_name, bandpass in filters.items():
        try:
            obs = Observation(spectrum, bandpass, force='taper')
            
            if magnitude_system == 'AB':
                mag = obs.effstim(flux_unit=u.ABmag)
            elif magnitude_system == 'ST':
                mag = obs.effstim(flux_unit=u.STmag)
            elif magnitude_system == 'Vega':
                from synphot import SourceSpectrum
                vega = SourceSpectrum.from_vega()
                mag = obs.effstim(flux_unit='vegamag', vegaspec=vega)
            
            magnitudes[filter_name] = mag.value
            
        except Exception as e:
            print(f"Warning: Failed to calculate {filter_name} for {component}: {e}")
            magnitudes[filter_name] = np.nan
    
    return magnitudes


def example_single_galaxy():
    """Example: Calculate magnitudes for a single galaxy."""
    print("=" * 70)
    print("EXAMPLE 1: Single Galaxy Magnitudes")
    print("=" * 70)
    
    # Initialize sed_calculator
    print("\n1. Initializing SED calculator...")
    sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
    galacticus_file = 'data/romanUNIT.hdf5'
    
    try:
        calc = sed_calculator(sed_template_file)
        print("   ✓ SED calculator initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        print("   (This is expected if data files are not present)")
        return
    
    # Create Roman filters
    print("\n2. Creating Roman WFI filters...")
    filter_names = ['F087', 'F129', 'F158', 'F184']
    filters = {name: create_roman_filter(name) for name in filter_names}
    print(f"   ✓ Created {len(filters)} filters")
    
    # Calculate magnitudes for galaxy 0
    print("\n3. Calculating magnitudes for galaxy index 0...")
    try:
        results = calculate_galaxy_magnitudes(
            calc, galacticus_file, gal_index=0,
            filters=filters, magnitude_system='AB'
        )
        
        print(f"   Galaxy redshift: z = {results['redshift']:.4f}")
        print("\n   Filter    λ_eff (Å)   AB magnitude")
        print("   " + "-" * 45)
        for filter_name in filter_names:
            mag = results['magnitudes'][filter_name]
            eff_wave = results['effective_wavelengths'][filter_name]
            print(f"   {filter_name:6s}    {eff_wave:8.1f}    {mag:6.3f}")
        
        # Calculate colors
        print("\n   Colors:")
        if not np.isnan(results['magnitudes']['F087']) and not np.isnan(results['magnitudes']['F158']):
            color1 = results['magnitudes']['F087'] - results['magnitudes']['F158']
            print(f"   F087-F158: {color1:+.3f} mag")
        if not np.isnan(results['magnitudes']['F129']) and not np.isnan(results['magnitudes']['F184']):
            color2 = results['magnitudes']['F129'] - results['magnitudes']['F184']
            print(f"   F129-F184: {color2:+.3f} mag")
        
        print("\n✓ Single galaxy calculation complete!")
        
    except Exception as e:
        print(f"   ✗ Calculation failed: {e}")
        print("   (This is expected if data files are not present)")


def example_component_comparison():
    """Example: Compare disk vs spheroid magnitudes."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Component Magnitude Comparison")
    print("=" * 70)
    
    # Initialize
    print("\n1. Initializing...")
    sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
    galacticus_file = 'data/romanUNIT.hdf5'
    
    try:
        calc = sed_calculator(sed_template_file)
        filters = {'F158': create_roman_filter('F158')}
        print("   ✓ Setup complete")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return
    
    # Calculate for both components
    print("\n2. Calculating F158 magnitudes for disk and spheroid...")
    try:
        disk_mags = calculate_component_magnitudes(
            calc, galacticus_file, gal_index=0,
            filters=filters, component='disk'
        )
        
        spheroid_mags = calculate_component_magnitudes(
            calc, galacticus_file, gal_index=0,
            filters=filters, component='spheroid'
        )
        
        print(f"   Disk F158:     {disk_mags['F158']:6.3f} mag")
        print(f"   Spheroid F158: {spheroid_mags['F158']:6.3f} mag")
        
        if not np.isnan(disk_mags['F158']) and not np.isnan(spheroid_mags['F158']):
            if disk_mags['F158'] < spheroid_mags['F158']:
                print(f"   → Disk is brighter by {spheroid_mags['F158'] - disk_mags['F158']:.3f} mag")
            else:
                print(f"   → Spheroid is brighter by {disk_mags['F158'] - spheroid_mags['F158']:.3f} mag")
        
        print("\n✓ Component comparison complete!")
        
    except Exception as e:
        print(f"   ✗ Calculation failed: {e}")


def example_multiple_galaxies():
    """Example: Calculate magnitudes for multiple galaxies."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiple Galaxy Magnitudes")
    print("=" * 70)
    
    # Initialize
    print("\n1. Initializing...")
    sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
    galacticus_file = 'data/romanUNIT.hdf5'
    
    try:
        calc = sed_calculator(sed_template_file)
        filters = {'F158': create_roman_filter('F158')}
        print("   ✓ Setup complete")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return
    
    # Calculate for first 3 galaxies
    print("\n2. Calculating F158 magnitudes for first 3 galaxies...")
    print("   Index    Redshift    F158 mag")
    print("   " + "-" * 40)
    
    try:
        for gal_idx in range(3):
            try:
                results = calculate_galaxy_magnitudes(
                    calc, galacticus_file, gal_idx,
                    filters=filters
                )
                print(f"   {gal_idx:5d}    {results['redshift']:8.4f}    {results['magnitudes']['F158']:8.3f}")
            except Exception as e:
                print(f"   {gal_idx:5d}    Failed: {e}")
        
        print("\n✓ Multiple galaxy calculation complete!")
        
    except Exception as e:
        print(f"   ✗ Calculation failed: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("GALACTICUS MAGNITUDE CALCULATION EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate how to calculate observed magnitudes")
    print("for Galacticus galaxies using synphot and the sed_calculator class.")
    
    # Run examples
    example_single_galaxy()
    example_component_comparison()
    example_multiple_galaxies()
    
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nThese examples show:")
    print("  ✓ How to calculate magnitudes for individual galaxies")
    print("  ✓ How to compare disk vs spheroid contributions")
    print("  ✓ How to process multiple galaxies")
    print("  ✓ Integration with existing sed_calculator class")
    print("\nFor production use:")
    print("  • Replace simplified filters with actual transmission curves")
    print("  • Add error handling and validation")
    print("  • Implement parallel processing for large catalogs")
    print("  • Add tests and documentation")
    print()


if __name__ == '__main__':
    main()
