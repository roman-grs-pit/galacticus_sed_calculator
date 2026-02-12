#!/usr/bin/env python3
"""
Calculate RA and Dec coordinates for galaxies in a Galacticus lightcone catalog.

This script converts the native lightcone angular coordinates (theta, phi) to 
astronomical RA and Dec coordinates, with support for repositioning the field 
center and applying a roll angle. Coordinates can be saved directly to the 
Galacticus file or to a separate output file.
"""

import numpy as np
import h5py
import astropy.units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation
from scipy.spatial.transform import Rotation
import shutil
import os
import argparse
import sys

# Add parent directory to path to import galacticus_sed_calculator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator.sed_calculator import detect_galacticus_format


def convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=0.0, roll=0.0):
    """
    Convert lightcone angular coordinates (theta, phi) to RA and Dec.
    
    The lightcone coordinates are centered at theta=0 (looking down the cone).
    This function repositions the field center to the specified (RA0, Dec0) and
    applies an optional roll angle.
    
    Parameters
    ----------
    theta : array-like
        Polar angle in radians (0 at cone center)
    phi : array-like
        Azimuthal angle in radians (-π to π or 0 to 2π)
    ra0 : float, optional
        Right Ascension of the new field center in degrees (default: 0.0)
    dec0 : float, optional
        Declination of the new field center in degrees (default: 0.0)
    roll : float, optional
        Roll angle in degrees for rotation around the line of sight (default: 0.0)
        
    Returns
    -------
    ra : ndarray
        Right Ascension in degrees (0 to 360)
    dec : ndarray
        Declination in degrees (-90 to 90)
        
    Notes
    -----
    The transformation process:
    1. Convert (theta, phi) to Cartesian coordinates on unit sphere
    2. Apply roll rotation around z-axis
    3. Rotate to reposition field center from (RA=0, Dec=90) to (RA0, Dec0)
    4. Convert back to spherical coordinates (RA, Dec)
    
    The original lightcone has theta=0 pointing "up" (Dec=90), and we rotate
    this to point at the desired field center.
    
    Note: Galacticus lightcone angular coordinates are in radians.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    
    # Convert lightcone coordinates to Cartesian
    # theta=0 is at the north pole (z=1), theta increases toward equator
    # Input theta and phi are already in radians (Galacticus native format)
    theta_rad = theta
    phi_rad = phi
    
    # Standard spherical to Cartesian: theta is polar angle from +z axis
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    
    # Stack into (N, 3) array for rotation
    coords = np.stack([x, y, z], axis=-1)
    
    # Apply roll rotation around z-axis (line of sight)
    if roll != 0.0:
        roll_rotation = Rotation.from_euler('z', roll, degrees=True)
        coords = roll_rotation.apply(coords)
    
    # Rotate to reposition field center
    # Original field center is at (RA=0, Dec=90) -> pointing along +z
    # Target field center is at (RA0, Dec0)
    # We need to rotate +z axis to point at (RA0, Dec0)
    
    if ra0 != 0.0 or dec0 != 90.0:
        # Calculate rotation that takes +z to the target direction
        # First rotate around y-axis by (90 - dec0) to get correct declination
        # Then rotate around z-axis by ra0 to get correct right ascension
        dec_rotation = Rotation.from_euler('y', 90.0 - dec0, degrees=True)
        ra_rotation = Rotation.from_euler('z', ra0, degrees=True)
        
        # Combined rotation: apply dec first, then ra
        total_rotation = ra_rotation * dec_rotation
        coords = total_rotation.apply(coords)
    
    # Convert back to spherical coordinates
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # RA: arctan2(y, x), wrapped to [0, 360)
    ra = np.rad2deg(np.arctan2(y, x))
    ra = np.where(ra < 0, ra + 360, ra)
    
    # Dec: arcsin(z), in range [-90, 90]
    dec = np.rad2deg(np.arcsin(np.clip(z, -1, 1)))
    
    return ra, dec


def calculate_catalog_coordinates(galacticus_catalog, ra0=0.0, dec0=0.0, roll=0.0,
                                  output_file=None, max_galaxies=None,
                                  save_to_input=False, copy_input=True,
                                  check_existing=True):
    """
    Calculate RA and Dec coordinates for all galaxies in a Galacticus lightcone catalog.
    
    Parameters
    ----------
    galacticus_catalog : str
        Path to Galacticus lightcone catalog HDF5 file
    ra0 : float, optional
        Right Ascension of field center in degrees (default: 0.0)
    dec0 : float, optional
        Declination of field center in degrees (default: 0.0)
    roll : float, optional
        Roll angle in degrees (default: 0.0)
    output_file : str, optional
        Path to output HDF5 file. If None and save_to_input=False,
        results are not saved to a separate file.
    max_galaxies : int, optional
        Maximum number of galaxies to process (for testing).
        If None, processes all galaxies.
    save_to_input : bool, optional
        If True, save coordinates to the Galacticus catalog file.
        Saves to /Lightcone/Output1/nodeData/rightAscension and declination
        If False, save to a separate output file. Default is False.
    copy_input : bool, optional
        If True and save_to_input=True, copy the input file before modifying.
        The copy will be named with '_with_coordinates' suffix. Default is True.
    check_existing : bool, optional
        If True, check if coordinates already exist and skip calculation if they do.
        Default is True.
        
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'ra': array of Right Ascension values
        - 'dec': array of Declination values
        - 'theta': array of original theta values
        - 'phi': array of original phi values
        - 'ra0': field center RA
        - 'dec0': field center Dec
        - 'roll': roll angle
        - 'galaxy_indices': array of galaxy indices processed
        - 'output_file': path to file where coordinates were saved (if applicable)
        - 'format_type': detected format ('lightcone' or 'fixed-time')
        - 'base_path': base path used in the HDF5 file
    """
    # Detect the format of the catalog
    format_type, base_path = detect_galacticus_format(galacticus_catalog)
    print(f"\nDetected format: {format_type}")
    print(f"Base path: {base_path}")
    
    if format_type != 'lightcone':
        raise ValueError("This script currently only supports lightcone format files. "
                        "Fixed-time format files do not have angular position data.")
    
    # Determine the file to work with
    working_file = galacticus_catalog
    
    if save_to_input:
        if copy_input:
            # Create a copy of the input file
            base, ext = os.path.splitext(galacticus_catalog)
            working_file = f"{base}_with_coordinates{ext}"
            
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
            print(f"\nWarning: Coordinates will be added directly to {galacticus_catalog}")
            print("The original file will be modified!")
    
    # Check if coordinates already exist
    if check_existing and save_to_input:
        print("\nChecking for existing coordinate datasets...")
        node_data_path = f'{base_path}/nodeData'
        with h5py.File(working_file, 'r') as f:
            ra_exists = f'{node_data_path}/rightAscension' in f
            dec_exists = f'{node_data_path}/declination' in f
        
        if ra_exists or dec_exists:
            print("Found existing coordinate datasets")
            response = input("Recalculate and overwrite? (y/n): ").lower()
            if response != 'y':
                print("Skipping calculation. Returning None.")
                return None
    
    # Read lightcone angular coordinates
    print("\nReading lightcone angular coordinates...")
    node_data_path = f'{base_path}/nodeData'
    
    with h5py.File(working_file, 'r') as f:
        theta = f[f'{node_data_path}/lightconeAngularTheta'][:]
        phi = f[f'{node_data_path}/lightconeAngularPhi'][:]
        n_galaxies_total = len(theta)
    
    print(f"Found {n_galaxies_total} galaxies in catalog")
    
    # Limit number of galaxies if requested
    if max_galaxies is not None:
        n_galaxies = min(max_galaxies, n_galaxies_total)
        print(f"Processing first {n_galaxies} galaxies (max_galaxies={max_galaxies})")
        theta = theta[:n_galaxies]
        phi = phi[:n_galaxies]
    else:
        n_galaxies = n_galaxies_total
        print(f"Processing all {n_galaxies} galaxies")
    
    # Convert to RA and Dec
    print(f"\nConverting coordinates with field center at (RA={ra0}°, Dec={dec0}°), roll={roll}°...")
    ra, dec = convert_lightcone_to_radec(theta, phi, ra0=ra0, dec0=dec0, roll=roll)
    
    print(f"RA range: [{np.min(ra):.2f}, {np.max(ra):.2f}] degrees")
    print(f"Dec range: [{np.min(dec):.2f}, {np.max(dec):.2f}] degrees")
    
    # Prepare results
    results = {
        'ra': ra,
        'dec': dec,
        'theta': theta,
        'phi': phi,
        'ra0': ra0,
        'dec0': dec0,
        'roll': roll,
        'galaxy_indices': np.arange(n_galaxies),
        'format_type': format_type,
        'base_path': base_path
    }
    
    # Save to file
    if save_to_input:
        print(f"\nSaving coordinates to Galacticus file: {working_file}...")
        save_coordinates_to_galacticus_file(working_file, results, format_type, base_path)
        results['output_file'] = working_file
    elif output_file is not None:
        print(f"\nSaving results to separate file: {output_file}...")
        save_coordinate_catalog(results, output_file)
        results['output_file'] = output_file
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Number of galaxies: {n_galaxies}")
    print(f"Field center: RA={ra0:.2f}°, Dec={dec0:.2f}°")
    print(f"Roll angle: {roll:.2f}°")
    print(f"RA range: [{np.min(ra):.4f}, {np.max(ra):.4f}]°")
    print(f"Dec range: [{np.min(dec):.4f}, {np.max(dec):.4f}]°")
    print(f"Angular extent: {np.max(theta):.4f}° from field center")
    
    return results


def save_coordinates_to_galacticus_file(galacticus_file, results, 
                                       format_type=None, base_path=None):
    """
    Save coordinate data directly to the Galacticus HDF5 file.
    
    Coordinates are saved to datasets:
    /Lightcone/Output1/nodeData/rightAscension
    /Lightcone/Output1/nodeData/declination
    
    Parameters
    ----------
    galacticus_file : str
        Path to Galacticus HDF5 file (will be modified)
    results : dict
        Results dictionary from calculate_catalog_coordinates
    format_type : str, optional
        Format type ('lightcone' or 'fixed-time'). If None, will be detected.
    base_path : str, optional
        Base path in HDF5 file. If None, will be detected.
    """
    # Detect format if not provided
    if format_type is None or base_path is None:
        format_type, base_path = detect_galacticus_format(galacticus_file)
    
    ra = results['ra']
    dec = results['dec']
    ra0 = results['ra0']
    dec0 = results['dec0']
    roll = results['roll']
    
    with h5py.File(galacticus_file, 'a') as f:
        # Create or access the nodeData group
        node_data_path = f'{base_path}/nodeData'
        if node_data_path not in f:
            raise ValueError(f"Path {node_data_path} not found in {galacticus_file}")
        
        # Save RA
        ra_path = f'{node_data_path}/rightAscension'
        if ra_path in f:
            print(f"  Deleting existing dataset: {ra_path}")
            del f[ra_path]
        
        print(f"  Creating dataset: {ra_path}")
        ra_dataset = f.create_dataset(ra_path, data=ra)
        ra_dataset.attrs['comment'] = b'Right Ascension in degrees (0 to 360)'
        ra_dataset.attrs['units'] = b'degrees'
        ra_dataset.attrs['fieldCenterRA'] = ra0
        ra_dataset.attrs['fieldCenterDec'] = dec0
        ra_dataset.attrs['rollAngle'] = roll
        
        # Save Dec
        dec_path = f'{node_data_path}/declination'
        if dec_path in f:
            print(f"  Deleting existing dataset: {dec_path}")
            del f[dec_path]
        
        print(f"  Creating dataset: {dec_path}")
        dec_dataset = f.create_dataset(dec_path, data=dec)
        dec_dataset.attrs['comment'] = b'Declination in degrees (-90 to 90)'
        dec_dataset.attrs['units'] = b'degrees'
        dec_dataset.attrs['fieldCenterRA'] = ra0
        dec_dataset.attrs['fieldCenterDec'] = dec0
        dec_dataset.attrs['rollAngle'] = roll
    
    print(f"Saved RA and Dec coordinate datasets to {galacticus_file}")


def save_coordinate_catalog(results, output_file):
    """
    Save coordinate catalog to HDF5 file.
    
    Parameters
    ----------
    results : dict
        Results dictionary from calculate_catalog_coordinates
    output_file : str
        Path to output HDF5 file
    """
    with h5py.File(output_file, 'w') as f:
        # Save coordinates
        f.create_dataset('ra', data=results['ra'],
                        compression='gzip', compression_opts=9)
        f.create_dataset('dec', data=results['dec'],
                        compression='gzip', compression_opts=9)
        
        # Save original lightcone coordinates
        f.create_dataset('theta', data=results['theta'],
                        compression='gzip', compression_opts=9)
        f.create_dataset('phi', data=results['phi'],
                        compression='gzip', compression_opts=9)
        
        # Save transformation parameters as attributes
        f.attrs['fieldCenterRA'] = results['ra0']
        f.attrs['fieldCenterDec'] = results['dec0']
        f.attrs['rollAngle'] = results['roll']
        f.attrs['description'] = 'RA and Dec coordinates calculated from Galacticus lightcone angular positions'
        
        # Save galaxy indices
        f.create_dataset('galaxy_indices', data=results['galaxy_indices'],
                        compression='gzip', compression_opts=9)
    
    print(f"Saved coordinate catalog with {len(results['ra'])} galaxies")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate RA and Dec coordinates for galaxies in a Galacticus lightcone catalog.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('catalog',
                       help='Path to Galacticus HDF5 lightcone catalog file')
    
    # Optional arguments for field positioning
    parser.add_argument('--ra0', type=float, default=0.0,
                       help='Right Ascension of field center in degrees')
    parser.add_argument('--dec0', type=float, default=0.0,
                       help='Declination of field center in degrees')
    parser.add_argument('--roll', type=float, default=0.0,
                       help='Roll angle in degrees')
    
    # Processing options
    parser.add_argument('-n', '--max-galaxies', type=int, default=None,
                       help='Maximum number of galaxies to process (default: all)')
    
    # Save options
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument('--save-to-input', action='store_true', default=True,
                           help='Save coordinates to the input Galacticus file (default)')
    save_group.add_argument('--save-to-file', metavar='OUTPUT',
                           help='Save coordinates to a separate HDF5 file instead')
    
    parser.add_argument('--no-copy', action='store_true',
                       help='Modify input file directly instead of creating a copy (use with caution!)')
    parser.add_argument('--no-check-existing', action='store_true',
                       help='Skip check for existing coordinates and overwrite without prompting')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print("="*60)
    print("CALCULATE RA AND DEC FOR GALACTICUS LIGHTCONE CATALOG")
    print("="*60)
    print(f"\nInput catalog: {args.catalog}")
    print(f"Field center: RA={args.ra0}°, Dec={args.dec0}°")
    print(f"Roll angle: {args.roll}°")
    
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
            print("\nSaving to input file (will create a copy with '_with_coordinates' suffix)")
        else:
            print("\nWARNING: Saving to input file directly (no copy will be made)")
    
    check_existing = not args.no_check_existing
    
    # Calculate coordinates
    results = calculate_catalog_coordinates(
        galacticus_catalog=args.catalog,
        ra0=args.ra0,
        dec0=args.dec0,
        roll=args.roll,
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
