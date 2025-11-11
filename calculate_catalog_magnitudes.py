#!/usr/bin/env python3
"""
Calculate observed magnitudes for galaxies in a Galacticus catalog.

This script calculates Roman WFI magnitudes for all galaxies in a Galacticus catalog
using the SED calculator and synphot. Magnitudes can be saved directly to the 
Galacticus file or to a separate output file.
"""

import numpy as np
import h5py
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import stpsf
from SEDfromSFH import sed_calculator
import time
import shutil
import os
import argparse
import sys

# Default configuration
DEFAULT_SED_TEMPLATE = "data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5"
DEFAULT_FILTERS = ["F062", "F087", "F106", "F129", "F158", "F184", "F213"]
DEFAULT_WAVELENGTH_MIN = 4000  # Angstroms
DEFAULT_WAVELENGTH_MAX = 23000  # Angstroms
DEFAULT_WAVELENGTH_NPOINTS = 2000


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
                                 max_galaxies=None, component='total',
                                 save_to_input=False, copy_input=True,
                                 check_existing=True, obs_wavelengths=None):
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
        Path to output HDF5 file. If None and save_to_input=False, 
        results are not saved to a separate file.
    max_galaxies : int, optional
        Maximum number of galaxies to process (for testing). 
        If None, processes all galaxies.
    component : str, optional
        Galaxy component to use ('total', 'disk', 'spheroid'). Default is 'total'.
    save_to_input : bool, optional
        If True, save magnitudes to the Galacticus catalog file in the format:
        /Lightcone/Output1/nodeData/apparentMagnitudeRomanWFI:<filter>
        If False, save to a separate output file. Default is False.
    copy_input : bool, optional
        If True and save_to_input=True, copy the input file before modifying.
        The copy will be named with '_with_magnitudes' suffix. Default is True.
    check_existing : bool, optional
        If True, check if magnitudes already exist and skip calculation if they do.
        Default is True.
    obs_wavelengths : Quantity, optional
        Wavelength grid for spectrum calculation. If None, uses default
        np.linspace(4000, 23000, 2000) * u.AA
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'magnitudes': 2D array of shape (n_galaxies, n_filters)
        - 'filter_names': list of filter names
        - 'redshifts': array of galaxy redshifts
        - 'galaxy_indices': array of galaxy indices processed
        - 'output_file': path to file where magnitudes were saved (if applicable)
    """
    # Determine the file to work with
    working_file = galacticus_catalog
    
    if save_to_input:
        if copy_input:
            # Create a copy of the input file
            base, ext = os.path.splitext(galacticus_catalog)
            working_file = f"{base}_with_magnitudes{ext}"
            
            if os.path.exists(working_file):
                print(f"\nWarning: Output file {working_file} already exists")
                response = input("Overwrite? (y/n): ").lower()
                if response != 'y':
                    print("Aborted.")
                    return None
            
            print(f"\nCopying {galacticus_catalog} to {working_file}...")
            shutil.copy2(galacticus_catalog, working_file)
            print("Copy complete.")
        else:
            print(f"\nWarning: Magnitudes will be added directly to {galacticus_catalog}")
            print("The original file will be modified!")
    
    # Get filter names
    filter_names = list(bandpasses.keys())
    n_filters = len(filter_names)
    
    # Check if magnitudes already exist
    if check_existing and save_to_input:
        print("\nChecking for existing magnitude datasets...")
        existing_filters = []
        with h5py.File(working_file, 'r') as f:
            for filter_name in filter_names:
                dataset_path = f'/Lightcone/Output1/nodeData/apparentMagnitudeRomanWFI:{filter_name}'
                if dataset_path in f:
                    existing_filters.append(filter_name)
        
        if existing_filters:
            print(f"Found existing magnitudes for filters: {', '.join(existing_filters)}")
            response = input("Recalculate and overwrite? (y/n): ").lower()
            if response != 'y':
                print("Skipping calculation. Returning None.")
                return None
    
    # Initialize the SED calculator
    print(f"\nInitializing SED calculator with template: {sed_template_file}")
    # Use UNIT cosmology since the catalog was generated with it
    unit_cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    calc = sed_calculator(sed_template_file, cosmology=unit_cosmo)
    
    # Set wavelength grid
    if obs_wavelengths is None:
        obs_wavelengths = np.linspace(4000, 23000, 2000) * u.AA
    
    # Get number of galaxies in catalog
    n_galaxies_total = get_galaxy_count(working_file)
    print(f"Found {n_galaxies_total} galaxies in catalog")
    
    # Limit number of galaxies if requested
    if max_galaxies is not None:
        n_galaxies = min(max_galaxies, n_galaxies_total)
        print(f"Processing first {n_galaxies} galaxies (max_galaxies={max_galaxies})")
    else:
        n_galaxies = n_galaxies_total
        print(f"Processing all {n_galaxies} galaxies")
    
    # Initialize arrays to store results
    magnitude_array = np.full((n_galaxies, n_filters), np.nan)
    redshifts = np.zeros(n_galaxies)
    
    # Read redshifts for all galaxies
    print("\nReading galaxy redshifts...")
    with h5py.File(working_file, 'r') as f:
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
                working_file, 
                galIndex=i,
                bandpasses=bandpasses,
                component=component,
                obs_wavelengths=obs_wavelengths
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
    
    # Save to file
    if save_to_input:
        print(f"\nSaving magnitudes to Galacticus file: {working_file}...")
        save_magnitudes_to_galacticus_file(working_file, results, component)
        results['output_file'] = working_file
    elif output_file is not None:
        print(f"\nSaving results to separate file: {output_file}...")
        save_magnitude_catalog(results, output_file)
        results['output_file'] = output_file
    
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


def save_magnitudes_to_galacticus_file(galacticus_file, results, component='total'):
    """
    Save magnitude data directly to the Galacticus HDF5 file.
    
    Magnitudes are saved to datasets with paths like:
    /Lightcone/Output1/nodeData/apparentMagnitudeRomanWFI:<filter>
    
    Parameters
    ----------
    galacticus_file : str
        Path to Galacticus HDF5 file (will be modified)
    results : dict
        Results dictionary from calculate_catalog_magnitudes
    component : str, optional
        Component type used for magnitude calculation. Default is 'total'.
    """
    filter_names = results['filter_names']
    magnitude_array = results['magnitudes']
    n_galaxies = len(results['galaxy_indices'])
    
    # Determine comment based on component
    if component == 'total':
        comment = "Total AB magnitude (disk + spheroid + emission lines)"
    elif component == 'disk':
        comment = "Disk AB magnitude (disk + emission lines)"
    elif component == 'spheroid':
        comment = "Spheroid AB magnitude (spheroid + emission lines)"
    elif component == 'AGN':
        comment = "AGN AB magnitude (emission lines only)"
    else:
        comment = f"{component} AB magnitude"
    
    with h5py.File(galacticus_file, 'a') as f:
        # Create or access the nodeData group
        node_data_path = '/Lightcone/Output1/nodeData'
        if node_data_path not in f:
            raise ValueError(f"Path {node_data_path} not found in {galacticus_file}")
        
        for j, filter_name in enumerate(filter_names):
            dataset_path = f'{node_data_path}/apparentMagnitudeRomanWFI:{filter_name}'
            
            # Delete existing dataset if it exists
            if dataset_path in f:
                print(f"  Deleting existing dataset: {dataset_path}")
                del f[dataset_path]
            
            # Create new dataset
            print(f"  Creating dataset: {dataset_path}")
            dataset = f.create_dataset(
                dataset_path,
                data=magnitude_array[:, j],
                compression='gzip',
                compression_opts=9
            )
            
            # Add attributes
            dataset.attrs['comment'] = comment.encode('utf-8')
            dataset.attrs['filter'] = filter_name.encode('utf-8')
    
    print(f"Saved {len(filter_names)} magnitude datasets to {galacticus_file}")


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


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate observed magnitudes for galaxies in a Galacticus catalog.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('catalog', 
                       help='Path to Galacticus HDF5 catalog file')
    parser.add_argument('sed_template',
                       help='Path to SED template HDF5 file')
    
    # Optional arguments
    parser.add_argument('-f', '--filters', nargs='+', default=DEFAULT_FILTERS,
                       help='Roman WFI filters to calculate (e.g., F062 F087 F158)')
    parser.add_argument('-n', '--max-galaxies', type=int, default=None,
                       help='Maximum number of galaxies to process (default: all)')
    parser.add_argument('-c', '--component', default='total',
                       choices=['total', 'disk', 'spheroid', 'AGN'],
                       help='Galaxy component to use for magnitude calculation')
    
    # Save options
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument('--save-to-input', action='store_true', default=True,
                           help='Save magnitudes to the input Galacticus file (default)')
    save_group.add_argument('--save-to-file', metavar='OUTPUT',
                           help='Save magnitudes to a separate HDF5 file instead')
    
    parser.add_argument('--no-copy', action='store_true',
                       help='Modify input file directly instead of creating a copy (use with caution!)')
    parser.add_argument('--no-check-existing', action='store_true',
                       help='Skip check for existing magnitudes and overwrite without prompting')
    
    # Wavelength options
    parser.add_argument('--wavelength-min', type=float, default=DEFAULT_WAVELENGTH_MIN,
                       help='Minimum wavelength for spectrum calculation (Angstroms)')
    parser.add_argument('--wavelength-max', type=float, default=DEFAULT_WAVELENGTH_MAX,
                       help='Maximum wavelength for spectrum calculation (Angstroms)')
    parser.add_argument('--wavelength-npoints', type=int, default=DEFAULT_WAVELENGTH_NPOINTS,
                       help='Number of wavelength points for spectrum calculation')
    
    # System options
    parser.add_argument('--magnitude-system', default='AB',
                       choices=['AB', 'ST', 'Vega'],
                       help='Magnitude system to use')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print("="*60)
    print("CALCULATE MAGNITUDES FOR GALACTICUS CATALOG")
    print("="*60)
    print(f"\nInput catalog: {args.catalog}")
    print(f"SED template: {args.sed_template}")
    print(f"Filters: {', '.join(args.filters)}")
    print(f"Component: {args.component}")
    print(f"Magnitude system: {args.magnitude_system}")
    
    if args.max_galaxies:
        print(f"Max galaxies: {args.max_galaxies}")
    else:
        print("Processing all galaxies")
    
    # Check if files exist
    if not os.path.exists(args.catalog):
        print(f"\nError: Catalog file not found: {args.catalog}")
        sys.exit(1)
    if not os.path.exists(args.sed_template):
        print(f"\nError: SED template file not found: {args.sed_template}")
        sys.exit(1)
    
    # Create wavelength grid
    obs_wavelengths = np.linspace(args.wavelength_min, args.wavelength_max, 
                                   args.wavelength_npoints) * u.AA
    
    # Load bandpass filters
    print("\nLoading bandpass filters...")
    bandpasses = load_roman_bandpasses(args.filters)
    
    # Determine save mode
    if args.save_to_file:
        save_to_input = False
        output_file = args.save_to_file
        copy_input = False
        print(f"\nSaving to separate file: {output_file}")
    else:
        save_to_input = True
        output_file = None
        copy_input = not args.no_copy
        if copy_input:
            print("\nSaving to input file (will create a copy with '_with_magnitudes' suffix)")
        else:
            print("\nWARNING: Saving to input file directly (no copy will be made)")
    
    check_existing = not args.no_check_existing
    
    # Calculate magnitudes
    results = calculate_catalog_magnitudes(
        sed_template_file=args.sed_template,
        galacticus_catalog=args.catalog,
        bandpasses=bandpasses,
        output_file=output_file,
        max_galaxies=args.max_galaxies,
        component=args.component,
        save_to_input=save_to_input,
        copy_input=copy_input,
        check_existing=check_existing,
        obs_wavelengths=obs_wavelengths
    )
    
    if results is not None:
        print("\n" + "="*60)
        print("DONE!")
        print("="*60)
        if 'output_file' in results:
            print(f"\nResults saved to: {results['output_file']}")
        print(f"Total galaxies processed: {len(results['galaxy_indices'])}")
        print(f"Filters: {', '.join(results['filter_names'])}")
    else:
        print("\nOperation cancelled or skipped.")


if __name__ == '__main__':
    main()
