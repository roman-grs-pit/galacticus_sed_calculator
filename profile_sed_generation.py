"""
Profiling script for SED generation performance testing.

This script sets up profiling infrastructure to measure the performance of
SED generation and identify bottlenecks such as:
- HDF5 file I/O operations
- Array multiplication and summation
- Other computational operations

Usage:
    python profile_sed_generation.py [--num-galaxies NUM] [--output-file FILE]
"""

import argparse
import time
import cProfile
import pstats
import io
import numpy as np
import astropy.units as u
from SEDfromSFH import sed_calculator


def profile_sed_generation(
    sed_template_filename,
    galacticus_filename,
    num_galaxies=None,
    obs_wavelengths=None,
    detailed_profile=False,
    output_file=None,
    include_emission_lines=True,
    use_synphot=True
):
    """
    Profile SED generation for multiple galaxies.
    
    Parameters
    ----------
    sed_template_filename : str
        Path to SED template HDF5 file
    galacticus_filename : str
        Path to Galacticus galaxy catalog HDF5 file
    num_galaxies : int or None
        Number of galaxies to profile. If None, profiles all galaxies.
    obs_wavelengths : array-like or None
        Wavelengths at which to evaluate spectra. If None, uses default range.
    detailed_profile : bool
        If True, run detailed cProfile profiling. If False, just time individual operations.
    output_file : str or None
        If provided, write profiling results to this file.
        
    Returns
    -------
    results : dict
        Dictionary containing timing results and statistics
    """
    
    # Setup
    if obs_wavelengths is None:
        obs_wavelengths = np.linspace(0.4, 2.5, 1000) * u.micron
    
    print("=" * 80)
    print("SED Generation Profiling")
    print("=" * 80)
    print(f"SED template: {sed_template_filename}")
    print(f"Galaxy catalog: {galacticus_filename}")
    print(f"Wavelength points: {len(obs_wavelengths)}")
    
    # Initialize calculator (time this separately)
    print("\nInitializing SED calculator...")
    t0 = time.time()
    sedCalc = sed_calculator(sed_template_filename)
    init_time = time.time() - t0
    print(f"  Initialization time: {init_time:.4f} s")
    
    # Determine how many galaxies to process
    import h5py
    with h5py.File(galacticus_filename, 'r') as f:
        # Auto-detect format and get redshift dataset path
        from SEDfromSFH import detect_galacticus_format
        format_type, base_path = detect_galacticus_format(galacticus_filename)
        
        if format_type == 'lightcone':
            redshift_path = f'{base_path}/nodeData/lightconeRedshiftObserved'
        else:  # fixed-time
            # For fixed-time, all galaxies have same redshift, so just count disk SFH entries
            redshift_path = f'{base_path}/nodeData/diskStarFormationHistoryMass'
        
        total_galaxies = len(f[redshift_path])
    
    if num_galaxies is None or num_galaxies > total_galaxies:
        num_galaxies = total_galaxies
    
    print(f"\nProcessing {num_galaxies} galaxies (out of {total_galaxies} total)")
    
    # Storage for timing results
    results = {
        'init_time': init_time,
        'num_galaxies': num_galaxies,
        'per_galaxy_times': [],
        'component_times': {
            'total': [],
            'evaluate_spectrum': [],
        }
    }
    
    # Profile each galaxy
    print("\nProfiling individual galaxy SED generation:")
    print("-" * 80)
    
    for galIndex in range(num_galaxies):
        # Time the entire operation
        t_start = time.time()
        
        if detailed_profile and galIndex == 0:
            # Run detailed profiling on first galaxy only
            pr = cProfile.Profile()
            pr.enable()
        
        # Main operation: evaluate total spectrum
        try:
            total_flux = sedCalc.evaluate_total_spectrum(
                galacticus_filename, 
                galIndex, 
                obs_wavelengths=obs_wavelengths,
                include_emission_lines=include_emission_lines,
                use_synphot=use_synphot
            )
            success = True
        except Exception as e:
            print(f"  Galaxy {galIndex}: ERROR - {str(e)}")
            success = False
            total_flux = None
        
        if detailed_profile and galIndex == 0:
            pr.disable()
        
        t_end = time.time()
        elapsed = t_end - t_start
        
        if success:
            results['per_galaxy_times'].append(elapsed)
            
            # Print progress every 100 galaxies or for first 10
            if galIndex < 10 or (galIndex + 1) % 100 == 0:
                print(f"  Galaxy {galIndex:6d}: {elapsed:.4f} s")
        
        # Save detailed profile for first galaxy
        if detailed_profile and galIndex == 0 and success:
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.strip_dirs()
            ps.sort_stats('cumulative')
            ps.print_stats(30)  # Top 30 functions
            results['detailed_profile_first_galaxy'] = s.getvalue()
    
    # Calculate statistics
    if results['per_galaxy_times']:
        times = np.array(results['per_galaxy_times'])
        results['stats'] = {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'total': np.sum(times),
        }
        
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Galaxies processed: {len(times)}")
        print(f"Total time: {results['stats']['total']:.2f} s")
        print(f"Mean time per galaxy: {results['stats']['mean']:.4f} s ({results['stats']['mean']*1000:.2f} ms)")
        print(f"Median time per galaxy: {results['stats']['median']:.4f} s ({results['stats']['median']*1000:.2f} ms)")
        print(f"Std dev: {results['stats']['std']:.4f} s")
        print(f"Min time: {results['stats']['min']:.4f} s")
        print(f"Max time: {results['stats']['max']:.4f} s")
        print(f"\nEstimated time for 10,000 galaxies: {results['stats']['mean'] * 10000 / 60:.1f} minutes")
        print(f"Target time (0.2 s/galaxy): {0.2 * 10000 / 60:.1f} minutes")
        print(f"Current vs target: {results['stats']['mean'] / 0.2:.1f}x slower")
        
        if detailed_profile and 'detailed_profile_first_galaxy' in results:
            print("\n" + "=" * 80)
            print("DETAILED PROFILE (first galaxy)")
            print("=" * 80)
            print(results['detailed_profile_first_galaxy'])
    
    # Write to output file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("SED Generation Profiling Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"SED template: {sed_template_filename}\n")
            f.write(f"Galaxy catalog: {galacticus_filename}\n")
            f.write(f"Wavelength points: {len(obs_wavelengths)}\n")
            f.write(f"Initialization time: {init_time:.4f} s\n\n")
            
            if results['per_galaxy_times']:
                f.write("Summary Statistics\n")
                f.write("-" * 80 + "\n")
                f.write(f"Galaxies processed: {len(times)}\n")
                f.write(f"Total time: {results['stats']['total']:.2f} s\n")
                f.write(f"Mean time per galaxy: {results['stats']['mean']:.4f} s\n")
                f.write(f"Median time per galaxy: {results['stats']['median']:.4f} s\n")
                f.write(f"Std dev: {results['stats']['std']:.4f} s\n")
                f.write(f"Min time: {results['stats']['min']:.4f} s\n")
                f.write(f"Max time: {results['stats']['max']:.4f} s\n\n")
                
                if detailed_profile and 'detailed_profile_first_galaxy' in results:
                    f.write("\nDetailed Profile (first galaxy)\n")
                    f.write("=" * 80 + "\n")
                    f.write(results['detailed_profile_first_galaxy'])
                    f.write("\n")
        
        print(f"\nResults written to: {output_file}")
    
    return results


def profile_sed_components(
    sed_template_filename,
    galacticus_filename,
    galIndex=0,
    obs_wavelengths=None,
    use_synphot=True
):
    """
    Profile individual components of SED generation for a single galaxy.
    
    This breaks down the SED generation into components to identify bottlenecks:
    - File I/O (reading galaxy data)
    - SED template loading
    - SFH to SED calculation
    - Spectrum resampling
    - Emission line addition
    
    Parameters
    ----------
    sed_template_filename : str
        Path to SED template HDF5 file
    galacticus_filename : str
        Path to Galacticus galaxy catalog HDF5 file
    galIndex : int
        Index of galaxy to profile
    obs_wavelengths : array-like or None
        Wavelengths at which to evaluate spectra
        
    Returns
    -------
    component_times : dict
        Dictionary with timing for each component
    """
    
    if obs_wavelengths is None:
        obs_wavelengths = np.linspace(0.4, 2.5, 1000) * u.micron
    
    print("\n" + "=" * 80)
    print(f"COMPONENT-LEVEL PROFILING (Galaxy {galIndex})")
    print("=" * 80)
    
    component_times = {}
    
    # Time: SED calculator initialization
    t0 = time.time()
    sedCalc = sed_calculator(sed_template_filename)
    component_times['init_calculator'] = time.time() - t0
    print(f"Initialize calculator: {component_times['init_calculator']:.4f} s")
    
    # Time: Reading galaxy data from file
    t0 = time.time()
    galData = sedCalc.read_galacticus_galaxy(galacticus_filename, galIndex)
    component_times['read_galaxy_data'] = time.time() - t0
    print(f"Read galaxy data: {component_times['read_galaxy_data']:.4f} s")
    
    # Time: Calculate continuum for disk
    t0 = time.time()
    disk_spectrum = sedCalc.evaluate_component_spectrum(
        galacticus_filename, galIndex, 
        component='disk',
        obs_wavelengths=obs_wavelengths,
        include_emission_lines=False,
        use_synphot=use_synphot
    )
    component_times['disk_continuum'] = time.time() - t0
    print(f"Disk continuum: {component_times['disk_continuum']:.4f} s")
    
    # Time: Calculate continuum for spheroid
    t0 = time.time()
    spheroid_spectrum = sedCalc.evaluate_component_spectrum(
        galacticus_filename, galIndex, 
        component='spheroid',
        obs_wavelengths=obs_wavelengths,
        include_emission_lines=False,
        use_synphot=use_synphot
    )
    component_times['spheroid_continuum'] = time.time() - t0
    print(f"Spheroid continuum: {component_times['spheroid_continuum']:.4f} s")
    
    # Time: Add emission lines for disk
    t0 = time.time()
    disk_spectrum_with_lines = sedCalc.evaluate_component_spectrum(
        galacticus_filename, galIndex, 
        component='disk',
        obs_wavelengths=obs_wavelengths,
        include_emission_lines=True,
        use_synphot=use_synphot
    )
    component_times['disk_with_lines'] = time.time() - t0
    print(f"Disk with emission lines: {component_times['disk_with_lines']:.4f} s")
    component_times['disk_emission_lines_only'] = (
        component_times['disk_with_lines'] - component_times['disk_continuum']
    )

    # Time: Calculate AGN (which is only emission lines for now)
    t0 = time.time()
    AGN_spectrum = sedCalc.evaluate_component_spectrum(
        galacticus_filename, galIndex, 
        component='AGN',
        obs_wavelengths=obs_wavelengths,
        include_emission_lines=True,
        use_synphot=use_synphot
    )
    component_times['AGN_with_lines'] = time.time() - t0
    print(f"AGN with emission lines: {component_times['AGN_with_lines']:.4f} s")
    
    # Time: Full spectrum (disk + spheroid + AGN with lines)
    t0 = time.time()
    total_spectrum = sedCalc.evaluate_total_spectrum(
        galacticus_filename, galIndex, 
        obs_wavelengths=obs_wavelengths,
        use_synphot=use_synphot
    )
    component_times['total_spectrum'] = time.time() - t0
    print(f"Total spectrum (all components): {component_times['total_spectrum']:.4f} s")
    
    print("\n" + "-" * 80)
    print("COMPONENT BREAKDOWN:")
    print("-" * 80)
    total = component_times['total_spectrum']
    for key, val in sorted(component_times.items(), key=lambda x: -x[1]):
        pct = (val / total * 100) if total > 0 else 0
        print(f"{key:30s}: {val:8.4f} s ({pct:5.1f}%)")
    
    return component_times


def main():
    """Main entry point for profiling script."""
    parser = argparse.ArgumentParser(
        description='Profile SED generation performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile first 100 galaxies
  python profile_sed_generation.py --num-galaxies 100
  
  # Profile all galaxies with detailed profiling
  python profile_sed_generation.py --detailed-profile
  
  # Profile and save results to file
  python profile_sed_generation.py --num-galaxies 50 --output-file profile_results.txt
  
  # Profile individual components for one galaxy
  python profile_sed_generation.py --component-profile --galaxy-index 8
        """
    )
    
    parser.add_argument(
        '--sed-template',
        default='./data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5',
        help='Path to SED template file (default: ./data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5)'
    )
    
    parser.add_argument(
        '--galaxy-catalog',
        default='./data/romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5',
        help='Path to galaxy catalog file (default: ./data/romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5)'
    )
    
    parser.add_argument(
        '--num-galaxies',
        type=int,
        default=None,
        help='Number of galaxies to profile (default: all)'
    )
    
    parser.add_argument(
        '--galaxy-index',
        type=int,
        default=8,
        help='Galaxy index for component profiling (default: 8)'
    )
    
    parser.add_argument(
        '--detailed-profile',
        action='store_true',
        help='Enable detailed cProfile profiling (shows function-level breakdown)'
    )
    
    parser.add_argument(
        '--component-profile',
        action='store_true',
        help='Profile individual components (file I/O, computation, etc.) for one galaxy'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file for profiling results (default: print to console only)'
    )
    
    parser.add_argument(
        '--wavelength-min',
        type=float,
        default=0.4,
        help='Minimum wavelength in microns (default: 0.4)'
    )
    
    parser.add_argument(
        '--wavelength-max',
        type=float,
        default=2.5,
        help='Maximum wavelength in microns (default: 2.5)'
    )
    
    parser.add_argument(
        '--wavelength-points',
        type=int,
        default=1000,
        help='Number of wavelength points (default: 1000)'
    )

    parser.add_argument(
        '--no-emission-lines',
        action='store_true',
        help='Disable emission line calculation'
    )

    parser.add_argument(
        '--no-synphot',
        action='store_true',
        help='Use numpy arrays rather than synphot objects to store spectra'
    )
    
    args = parser.parse_args()
    
    # Create wavelength array
    obs_wavelengths = np.linspace(
        args.wavelength_min, 
        args.wavelength_max, 
        args.wavelength_points
    ) * u.micron
    
    if args.component_profile:
        # Profile individual components for one galaxy
        profile_sed_components(
            args.sed_template,
            args.galaxy_catalog,
            galIndex=args.galaxy_index,
            obs_wavelengths=obs_wavelengths,
            use_synphot=not args.no_synphot
        )
    else:
        # Profile multiple galaxies
        results = profile_sed_generation(
            args.sed_template,
            args.galaxy_catalog,
            num_galaxies=args.num_galaxies,
            obs_wavelengths=obs_wavelengths,
            detailed_profile=args.detailed_profile,
            output_file=args.output_file,
            include_emission_lines=not args.no_emission_lines,
            use_synphot=not args.no_synphot
        )


if __name__ == '__main__':
    main()
