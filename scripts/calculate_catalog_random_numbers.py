#!/usr/bin/env python3
"""
Generate and save random numbers for galaxies in a Galacticus catalog.

This script generates uniform random numbers (0 to 1) for each galaxy in a 
Galacticus catalog. These random numbers can be used for various purposes such as:
- Assigning scatter to dust attenuation
- Assigning galaxy inclinations
- Other stochastic properties

The script supports both lightcone and fixed-time formats, with random numbers
generated at each output time for fixed-time catalogs.
"""

import numpy as np
import h5py
from SEDfromSFH import detect_galacticus_format
import shutil
import os
import argparse
import sys


def generate_random_numbers(n_galaxies, n_random=5, seed=None):
    """
    Generate uniform random numbers for galaxies.
    
    Parameters
    ----------
    n_galaxies : int
        Number of galaxies to generate random numbers for
    n_random : int, optional
        Number of random values per galaxy (default: 5)
    seed : int, optional
        Random seed for reproducibility. If None, uses system random state.
        
    Returns
    -------
    random_numbers : ndarray
        Array of shape (n_galaxies, n_random) with uniform random values in [0, 1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.uniform(0.0, 1.0, size=(n_galaxies, n_random))


def calculate_catalog_random_numbers(galacticus_catalog, n_random=5, seed=None,
                                     output_file=None, max_galaxies=None,
                                     save_to_input=False, copy_input=True,
                                     check_existing=True):
    """
    Generate random numbers for all galaxies in a Galacticus catalog.
    
    Parameters
    ----------
    galacticus_catalog : str
        Path to Galacticus catalog HDF5 file
    n_random : int, optional
        Number of random values per galaxy (default: 5)
    seed : int, optional
        Random seed for reproducibility. If None, uses system random state.
    output_file : str, optional
        Path to output HDF5 file. If None and save_to_input=False,
        results are not saved to a separate file.
    max_galaxies : int, optional
        Maximum number of galaxies to process (for testing).
        If None, processes all galaxies.
    save_to_input : bool, optional
        If True, save random numbers to the Galacticus catalog file.
        Saves to /Lightcone/Output1/nodeData/randomUniform or
        /Outputs/OutputN/nodeData/randomUniform depending on format.
        If False, save to a separate output file. Default is False.
    copy_input : bool, optional
        If True and save_to_input=True, copy the input file before modifying.
        The copy will be named with '_with_random' suffix. Default is True.
    check_existing : bool, optional
        If True, check if random numbers already exist and skip calculation if they do.
        Default is True.
        
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'random_numbers': array of random numbers
        - 'n_random': number of random values per galaxy
        - 'seed': random seed used (or None)
        - 'galaxy_indices': array of galaxy indices processed
        - 'output_file': path to file where random numbers were saved (if applicable)
        - 'format_type': detected format ('lightcone' or 'fixed-time')
        - 'base_path': base path used in the HDF5 file
        - 'outputs_processed': list of output paths processed (for fixed-time format)
    """
    # Detect the format of the catalog
    format_type, base_path = detect_galacticus_format(galacticus_catalog)
    print(f"\nDetected format: {format_type}")
    print(f"Base path: {base_path}")
    
    # Determine the file to work with
    working_file = galacticus_catalog
    
    if save_to_input:
        if copy_input:
            # Create a copy of the input file
            base, ext = os.path.splitext(galacticus_catalog)
            working_file = f"{base}_with_random{ext}"
            
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
            print(f"\nWarning: Random numbers will be added directly to {galacticus_catalog}")
            print("The original file will be modified!")
    
    # For fixed-time format, we need to process all outputs
    outputs_to_process = []
    if format_type == 'fixed-time':
        with h5py.File(working_file, 'r') as f:
            outputs = [key for key in f['/Outputs'].keys() if key.startswith('Output')]
            outputs.sort(key=lambda x: int(x.replace('Output', '')))
            outputs_to_process = [f'/Outputs/{output}' for output in outputs]
        print(f"\nFound {len(outputs_to_process)} output times in fixed-time catalog")
    else:
        outputs_to_process = [base_path]
    
    # Check if random numbers already exist
    if check_existing and save_to_input:
        print("\nChecking for existing random number datasets...")
        any_exist = False
        for output_path in outputs_to_process:
            node_data_path = f'{output_path}/nodeData'
            with h5py.File(working_file, 'r') as f:
                random_exists = f'{node_data_path}/randomUniform' in f
                if random_exists:
                    any_exist = True
                    print(f"  Found existing random numbers in {output_path}")
        
        if any_exist:
            response = input("Recalculate and overwrite? (y/n): ").lower()
            if response != 'y':
                print("Skipping calculation. Returning None.")
                return None
    
    # Generate random numbers for each output
    all_random_numbers = []
    total_galaxies = 0
    
    for output_path in outputs_to_process:
        node_data_path = f'{output_path}/nodeData'
        
        # Get the number of galaxies in this output
        print(f"\nProcessing {output_path}...")
        with h5py.File(working_file, 'r') as f:
            # Try to find any dataset in nodeData to determine number of galaxies
            node_data_group = f[node_data_path]
            # Get first dataset to determine size
            first_dataset_name = list(node_data_group.keys())[0]
            n_galaxies_total = len(node_data_group[first_dataset_name])
        
        print(f"  Found {n_galaxies_total} galaxies")
        
        # Limit number of galaxies if requested
        if max_galaxies is not None:
            n_galaxies = min(max_galaxies, n_galaxies_total)
            print(f"  Processing first {n_galaxies} galaxies (max_galaxies={max_galaxies})")
        else:
            n_galaxies = n_galaxies_total
            print(f"  Processing all {n_galaxies} galaxies")
        
        # Generate random numbers for this output
        print(f"  Generating {n_random} random numbers per galaxy...")
        random_numbers = generate_random_numbers(n_galaxies, n_random=n_random, seed=seed)
        
        print(f"  Random number range: [{np.min(random_numbers):.4f}, {np.max(random_numbers):.4f}]")
        
        all_random_numbers.append(random_numbers)
        total_galaxies += n_galaxies
        
        # Save to file if requested
        if save_to_input:
            print(f"  Saving random numbers to {output_path}...")
            save_random_numbers_to_galacticus_file(
                working_file, random_numbers, n_random, seed,
                output_path, node_data_path
            )
    
    # Prepare results
    results = {
        'random_numbers': all_random_numbers[0] if len(all_random_numbers) == 1 else all_random_numbers,
        'n_random': n_random,
        'seed': seed,
        'galaxy_indices': np.arange(n_galaxies if len(all_random_numbers) == 1 else total_galaxies),
        'format_type': format_type,
        'base_path': base_path,
        'outputs_processed': outputs_to_process
    }
    
    if save_to_input:
        results['output_file'] = working_file
    elif output_file is not None:
        print(f"\nSaving results to separate file: {output_file}...")
        save_random_number_catalog(results, output_file)
        results['output_file'] = output_file
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Format type: {format_type}")
    print(f"Number of outputs processed: {len(outputs_to_process)}")
    print(f"Total galaxies: {total_galaxies}")
    print(f"Random numbers per galaxy: {n_random}")
    print(f"Random seed: {seed if seed is not None else 'None (random)'}")
    
    return results


def save_random_numbers_to_galacticus_file(galacticus_file, random_numbers, 
                                          n_random, seed, output_path, node_data_path):
    """
    Save random number data directly to the Galacticus HDF5 file.
    
    Random numbers are saved to datasets like:
    /Lightcone/Output1/nodeData/randomUniform (for lightcone)
    /Outputs/Output1/nodeData/randomUniform (for fixed-time)
    
    Parameters
    ----------
    galacticus_file : str
        Path to Galacticus HDF5 file (will be modified)
    random_numbers : ndarray
        Array of random numbers (n_galaxies, n_random)
    n_random : int
        Number of random values per galaxy
    seed : int or None
        Random seed used (if any)
    output_path : str
        Path to the output group (e.g., '/Lightcone/Output1')
    node_data_path : str
        Path to the nodeData group
    """
    with h5py.File(galacticus_file, 'a') as f:
        # Create or access the nodeData group
        if node_data_path not in f:
            raise ValueError(f"Path {node_data_path} not found in {galacticus_file}")
        
        # Save random numbers
        random_path = f'{node_data_path}/randomUniform'
        if random_path in f:
            print(f"    Deleting existing dataset: {random_path}")
            del f[random_path]
        
        print(f"    Creating dataset: {random_path}")
        random_dataset = f.create_dataset(random_path, data=random_numbers)
        random_dataset.attrs['comment'] = b'Uniform random numbers in [0, 1) for stochastic properties'
        random_dataset.attrs['n_random'] = n_random
        if seed is not None:
            random_dataset.attrs['seed'] = seed
        random_dataset.attrs['description'] = b'Array of shape (n_galaxies, n_random) with uniform random values'
    
    print(f"    Saved random number dataset to {random_path}")


def save_random_number_catalog(results, output_file):
    """
    Save random number catalog to HDF5 file.
    
    Parameters
    ----------
    results : dict
        Results dictionary from calculate_catalog_random_numbers
    output_file : str
        Path to output HDF5 file
    """
    with h5py.File(output_file, 'w') as f:
        # Handle single or multiple outputs
        if isinstance(results['random_numbers'], list):
            # Multiple outputs (fixed-time format)
            for i, (random_numbers, output_path) in enumerate(
                zip(results['random_numbers'], results['outputs_processed'])
            ):
                group_name = f"Output{i+1}"
                grp = f.create_group(group_name)
                grp.create_dataset('randomUniform', data=random_numbers,
                                 compression='gzip', compression_opts=9)
                grp.attrs['original_path'] = output_path
        else:
            # Single output (lightcone format)
            f.create_dataset('randomUniform', data=results['random_numbers'],
                           compression='gzip', compression_opts=9)
        
        # Save metadata as attributes
        f.attrs['n_random'] = results['n_random']
        if results['seed'] is not None:
            f.attrs['seed'] = results['seed']
        f.attrs['format_type'] = results['format_type']
        f.attrs['description'] = 'Uniform random numbers for galaxies in Galacticus catalog'
        
        # Save galaxy indices
        f.create_dataset('galaxy_indices', data=results['galaxy_indices'],
                        compression='gzip', compression_opts=9)
    
    n_galaxies = len(results['galaxy_indices'])
    print(f"Saved random number catalog with {n_galaxies} galaxies")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate and save random numbers for galaxies in a Galacticus catalog.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('catalog',
                       help='Path to Galacticus HDF5 catalog file')
    
    # Optional arguments for random number generation
    parser.add_argument('-n', '--n-random', type=int, default=5,
                       help='Number of random values per galaxy')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: random)')
    
    # Processing options
    parser.add_argument('--max-galaxies', type=int, default=None,
                       help='Maximum number of galaxies to process (default: all)')
    
    # Save options
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument('--save-to-input', action='store_true',
                           help='Save random numbers to the input Galacticus file (default)')
    save_group.add_argument('--save-to-file', metavar='OUTPUT',
                           help='Save random numbers to a separate HDF5 file instead')
    
    parser.add_argument('--no-copy', action='store_true',
                       help='Modify input file directly instead of creating a copy (use with caution!)')
    parser.add_argument('--no-check-existing', action='store_true',
                       help='Skip check for existing random numbers and overwrite without prompting')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print("="*60)
    print("GENERATE RANDOM NUMBERS FOR GALACTICUS CATALOG")
    print("="*60)
    print(f"\nInput catalog: {args.catalog}")
    print(f"Random numbers per galaxy: {args.n_random}")
    print(f"Random seed: {args.seed if args.seed is not None else 'None (random)'}")
    
    if args.max_galaxies:
        print(f"Max galaxies: {args.max_galaxies}")
    else:
        print("Processing all galaxies")
    
    # Check if file exists
    if not os.path.exists(args.catalog):
        print(f"\nError: Catalog file not found: {args.catalog}")
        sys.exit(1)
    
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
            print("\nSaving to input file (will create a copy with '_with_random' suffix)")
        else:
            print("\nWARNING: Saving to input file directly (no copy will be made)")
    
    check_existing = not args.no_check_existing
    
    # Calculate random numbers
    results = calculate_catalog_random_numbers(
        galacticus_catalog=args.catalog,
        n_random=args.n_random,
        seed=args.seed,
        output_file=output_file,
        max_galaxies=args.max_galaxies,
        save_to_input=save_to_input,
        copy_input=copy_input,
        check_existing=check_existing
    )
    
    if results is not None:
        print("\n" + "="*60)
        print("DONE!")
        print("="*60)
        if 'output_file' in results:
            print(f"\nResults saved to: {results['output_file']}")
        print(f"Total galaxies processed: {len(results['galaxy_indices'])}")
    else:
        print("\nOperation cancelled or skipped.")


if __name__ == '__main__':
    main()
