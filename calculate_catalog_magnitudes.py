#!/usr/bin/env python3
"""
Example script showing how to calculate magnitudes for all galaxies in a Galacticus catalog.

This script demonstrates:
1. Loading Roman WFI bandpasses once (for efficiency)
2. Calculating magnitudes for all galaxies in a catalog
3. Saving results to an output file
"""

import numpy as np
import h5py
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import stpsf
from SEDfromSFH import sed_calculator
import time

# Configuration
SED_TEMPLATE_FILE = "data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5"
GALACTICUS_CATALOG = "data/romanUNIT.hdf5"
OUTPUT_FILE = "galaxy_magnitudes.hdf5"

# Roman WFI filters to calculate
ROMAN_FILTERS = ["F062", "F087", "F106", "F129", "F158", "F184", "F213"]

# Wavelength grid for spectrum calculation
OBS_WAVELENGTHS = np.linspace(4000, 23000, 2000) * u.AA


def load_roman_bandpasses(filter_names):
    """
    Load Roman WFI bandpass filters using stpsf.
    
    This should be done once and the bandpasses reused for all galaxies.
    
    Parameters
    ----------
    filter_names : list of str
        List of Roman filter names (e.g., ['F158', 'F184'])
    
    Returns
    -------
    bandpasses : dict
        Dictionary mapping filter names to synphot SpectralElement objects
    """
    print("Loading Roman WFI bandpass filters...")
    roman = stpsf.WFI()
    bandpasses = {}
    
    for filter_name in filter_names:
        bandpasses[filter_name] = roman._get_synphot_bandpass(filter_name)
        print(f"  Loaded {filter_name}")
    
    return bandpasses


def get_galaxy_count(filename):
    """Get the number of galaxies in the catalog."""
    with h5py.File(filename, 'r') as f:
        # Get the length of the redshift array (or any other galaxy property)
        n_galaxies = len(f['/Lightcone/Output1/nodeData/lightconeRedshiftObserved'][:])
    return n_galaxies


def calculate_catalog_magnitudes(sed_template_file, galacticus_catalog, 
                                 bandpasses, output_file=None,
                                 max_galaxies=None, component='total'):
    """
    Calculate magnitudes for all galaxies in a Galacticus catalog.
    
    Parameters
    ----------
    sed_template_file : str
        Path to SED template HDF5 file
    galacticus_catalog : str
        Path to Galacticus catalog HDF5 file
    bandpasses : dict
        Dictionary of bandpass filters (from load_roman_bandpasses)
    output_file : str, optional
        Path to output HDF5 file. If None, results are not saved.
    max_galaxies : int, optional
        Maximum number of galaxies to process (for testing). 
        If None, processes all galaxies.
    component : str, optional
        Galaxy component to use ('total', 'disk', 'spheroid'). Default is 'total'.
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'magnitudes': 2D array of shape (n_galaxies, n_filters)
        - 'filter_names': list of filter names
        - 'redshifts': array of galaxy redshifts
        - 'galaxy_indices': array of galaxy indices processed
    """
    # Initialize the SED calculator
    print(f"\nInitializing SED calculator with template: {sed_template_file}")
    # Use UNIT cosmology since the catalog was generated with it
    unit_cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    calc = sed_calculator(sed_template_file, cosmology=unit_cosmo)
    
    # Get number of galaxies in catalog
    n_galaxies_total = get_galaxy_count(galacticus_catalog)
    print(f"Found {n_galaxies_total} galaxies in catalog")
    
    # Limit number of galaxies if requested
    if max_galaxies is not None:
        n_galaxies = min(max_galaxies, n_galaxies_total)
        print(f"Processing first {n_galaxies} galaxies (max_galaxies={max_galaxies})")
    else:
        n_galaxies = n_galaxies_total
        print(f"Processing all {n_galaxies} galaxies")
    
    # Get filter names
    filter_names = list(bandpasses.keys())
    n_filters = len(filter_names)
    
    # Initialize arrays to store results
    magnitude_array = np.full((n_galaxies, n_filters), np.nan)
    redshifts = np.zeros(n_galaxies)
    
    # Read redshifts for all galaxies
    print("\nReading galaxy redshifts...")
    with h5py.File(galacticus_catalog, 'r') as f:
        redshifts[:] = f['/Lightcone/Output1/nodeData/lightconeRedshiftObserved'][:n_galaxies]
    
    # Calculate magnitudes for each galaxy
    print(f"\nCalculating magnitudes in {n_filters} filters for {n_galaxies} galaxies...")
    print("Progress: ", end='', flush=True)
    
    start_time = time.time()
    
    for i in range(n_galaxies):
        # Progress indicator
        if (i + 1) % max(1, n_galaxies // 20) == 0:
            progress = (i + 1) / n_galaxies * 100
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_galaxies - i - 1) / rate if rate > 0 else 0
            print(f"{progress:.0f}% ", end='', flush=True)
        
        try:
            # Calculate magnitudes for this galaxy
            mags = calc.calculate_magnitudes(
                galacticus_catalog, 
                galIndex=i,
                bandpasses=bandpasses,
                component=component,
                obs_wavelengths=OBS_WAVELENGTHS
            )
            
            # Store results in array
            for j, filter_name in enumerate(filter_names):
                magnitude_array[i, j] = mags[filter_name]
                
        except Exception as e:
            print(f"\nWarning: Failed to process galaxy {i}: {e}")
            # magnitude_array already initialized with NaN values
    
    print("Done!")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.1f} seconds ({elapsed_time/n_galaxies:.2f} sec/galaxy)")
    
    # Prepare results
    results = {
        'magnitudes': magnitude_array,
        'filter_names': filter_names,
        'redshifts': redshifts,
        'galaxy_indices': np.arange(n_galaxies)
    }
    
    # Save to file if requested
    if output_file is not None:
        print(f"\nSaving results to {output_file}...")
        save_magnitude_catalog(results, output_file)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for j, filter_name in enumerate(filter_names):
        valid_mags = magnitude_array[:, j][~np.isnan(magnitude_array[:, j])]
        if len(valid_mags) > 0:
            print(f"{filter_name:6s}: mean={np.mean(valid_mags):6.2f}, "
                  f"median={np.median(valid_mags):6.2f}, "
                  f"std={np.std(valid_mags):5.2f}, "
                  f"valid={len(valid_mags)}/{n_galaxies}")
        else:
            print(f"{filter_name:6s}: No valid magnitudes")
    
    return results


def save_magnitude_catalog(results, output_file):
    """
    Save magnitude catalog to HDF5 file.
    
    Parameters
    ----------
    results : dict
        Results dictionary from calculate_catalog_magnitudes
    output_file : str
        Path to output HDF5 file
    """
    with h5py.File(output_file, 'w') as f:
        # Save magnitude array
        f.create_dataset('magnitudes', data=results['magnitudes'],
                        compression='gzip', compression_opts=9)
        
        # Save filter names as attributes
        f['magnitudes'].attrs['filter_names'] = results['filter_names']
        f['magnitudes'].attrs['description'] = 'AB magnitudes for each galaxy in each filter'
        
        # Save redshifts
        f.create_dataset('redshifts', data=results['redshifts'],
                        compression='gzip', compression_opts=9)
        
        # Save galaxy indices
        f.create_dataset('galaxy_indices', data=results['galaxy_indices'],
                        compression='gzip', compression_opts=9)
    
    print(f"Saved magnitude catalog with shape {results['magnitudes'].shape}")


def main():
    """Main execution function."""
    print("="*60)
    print("CALCULATE MAGNITUDES FOR GALACTICUS CATALOG")
    print("="*60)
    
    # Step 1: Load bandpass filters (do this once!)
    bandpasses = load_roman_bandpasses(ROMAN_FILTERS)
    
    # Step 2: Calculate magnitudes for all galaxies
    # For testing, you can set max_galaxies to a small number (e.g., 10)
    # For production, set max_galaxies=None to process all galaxies
    results = calculate_catalog_magnitudes(
        sed_template_file=SED_TEMPLATE_FILE,
        galacticus_catalog=GALACTICUS_CATALOG,
        bandpasses=bandpasses,
        output_file=OUTPUT_FILE,
        max_galaxies=10,  # Set to None to process all galaxies
        component='total'
    )
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Total galaxies processed: {len(results['galaxy_indices'])}")
    print(f"Filters: {', '.join(results['filter_names'])}")


if __name__ == '__main__':
    main()
