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
import time
import shutil
import os
import argparse
import sys
import multiprocessing
import json
import re

# Add parent directory to path to import galacticus_sed_calculator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator import SEDCalculator
from galacticus_sed_calculator.sed_calculator import detect_galacticus_format
from galacticus_sed_calculator.dust_attenuation import (
    dust_attenuation_gb10_generalised,
    apply_dust_attenuation_to_line,
    _calzetti_k_lambda,
)

# Default configuration
DEFAULT_SED_TEMPLATE = "data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5"
DEFAULT_FILTERS = ["F062", "F087", "F106", "F129", "F158", "F184", "F213"]
DEFAULT_WAVELENGTH_MIN = 4000  # Angstroms
DEFAULT_WAVELENGTH_MAX = 23000  # Angstroms
DEFAULT_WAVELENGTH_NPOINTS = 2000

# Module-level state for worker processes (populated by _init_worker)
_worker_state = {}


def _init_worker(sed_template_file, filter_names, cosmology, dust_model=None,
                 dust_params=None, dust_law='calzetti'):
    """
    Initialize per-worker process state.

    Called once per worker process when using multiprocessing.Pool. Loads the
    SEDCalculator and Roman bandpass filters so they are reused across all
    galaxy tasks assigned to that worker.
    """
    global _worker_state
    # stpsf is an optional dependency; import lazily so the module can be
    # imported in environments where stpsf is not installed (e.g. tests).
    import stpsf
    calc = SEDCalculator(sed_template_file, cosmology=cosmology)
    roman = stpsf.WFI()
    bandpasses = {f: roman._get_synphot_bandpass(f) for f in filter_names}
    _worker_state['calc'] = calc
    _worker_state['bandpasses'] = bandpasses
    _worker_state['dust_model'] = dust_model
    _worker_state['dust_params'] = dust_params
    _worker_state['dust_law'] = dust_law


def _process_galaxy_worker(args):
    """
    Worker function: calculate magnitudes for a single galaxy.

    Parameters
    ----------
    args : tuple
        (galaxy_index, working_file, component, obs_wavelengths)

    Returns
    -------
    tuple
        (galaxy_index, magnitudes_dict) where magnitudes_dict maps filter names
        to magnitude values, or None if the calculation failed.
    """
    i, working_file, component, obs_wavelengths = args
    try:
        mags = _worker_state['calc'].calculate_magnitudes(
            working_file,
            galIndex=i,
            bandpasses=_worker_state['bandpasses'],
            component=component,
            obs_wavelengths=obs_wavelengths,
            dust_model=_worker_state['dust_model'],
            dust_params=_worker_state['dust_params'],
            dust_law=_worker_state['dust_law'],
        )
        return i, mags
    except Exception as e:
        print(f"\nWarning: Failed to process galaxy {i}: {type(e).__name__}: {e}")
        return i, None


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
    # stpsf is an optional dependency; import lazily so the module can be
    # imported in environments where stpsf is not installed (e.g. tests).
    import stpsf
    roman = stpsf.WFI()
    bandpasses = {}
    
    for filter_name in filter_names:
        bandpasses[filter_name] = roman._get_synphot_bandpass(filter_name)
        print(f"  Loaded {filter_name}")
    
    return bandpasses


def load_dust_config(config_file):
    """
    Load dust attenuation configuration from a JSON file.

    The JSON file should contain the following keys:
    - ``dust_model``: name of the dust model (e.g. ``'gb10_generalised'``)
    - ``dust_params``: dict of model parameters (e.g. delta_0, delta_z, ...)
    - ``dust_law``: name of the attenuation law (e.g. ``'calzetti'``)

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.

    Returns
    -------
    dust_model : str
    dust_params : dict
    dust_law : str
    """
    with open(config_file, 'r') as fh:
        cfg = json.load(fh)

    required = ('dust_model', 'dust_params', 'dust_law')
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(
            f"Dust config file '{config_file}' is missing required keys: {missing}"
        )

    return cfg['dust_model'], cfg['dust_params'], cfg['dust_law']


def calculate_and_save_dust_attenuated_emission_lines(
    galacticus_catalog,
    dust_model,
    dust_params,
    dust_law,
    max_galaxies=None,
    components=None,
):
    """
    Calculate dust-attenuated emission line luminosities and save them to the
    Galacticus catalog HDF5 file.

    For each ``luminosityEmissionLine<Component>:<lineName>`` dataset a
    corresponding ``dustAttenuatedLuminosityEmissionLine<Component>:<lineName>``
    dataset is created.

    Parameters
    ----------
    galacticus_catalog : str
        Path to the Galacticus HDF5 catalog (modified in-place).
    dust_model : str
        Name of the dust attenuation model (currently only 'gb10_generalised').
    dust_params : dict
        Parameters for the dust model.
    dust_law : str
        Name of the attenuation law (currently only 'calzetti').
    max_galaxies : int, optional
        If set, only the first *max_galaxies* entries are processed.
    components : list of str, optional
        Components to process.  Defaults to ['disk', 'spheroid', 'AGN'].
    """
    if components is None:
        components = ['disk', 'spheroid', 'AGN']

    format_type, base_path = detect_galacticus_format(galacticus_catalog)
    node_data_path = f'{base_path}/nodeData'

    with h5py.File(galacticus_catalog, 'r') as f:
        if format_type == 'lightcone':
            all_redshifts = f[f'{node_data_path}/lightconeRedshiftObserved'][:]
        else:
            raise NotImplementedError(
                "Dust-attenuated emission lines are currently only supported "
                "for lightcone-format catalogs."
            )

        n_galaxies_total = len(all_redshifts)
        n_galaxies = min(max_galaxies, n_galaxies_total) if max_galaxies is not None else n_galaxies_total

        redshifts = all_redshifts[:n_galaxies]

        disk_mass = f[f'{node_data_path}/diskMassStellar'][:n_galaxies]
        spheroid_mass_path = f'{node_data_path}/spheroidMassStellar'
        spheroid_mass = (
            f[spheroid_mass_path][:n_galaxies]
            if spheroid_mass_path in f
            else np.zeros(n_galaxies)
        )
        total_stellar_mass = disk_mass + spheroid_mass

    # Calculate H-alpha attenuation for all galaxies at once (vectorized)
    if dust_model == 'gb10_generalised':
        A_Halpha = dust_attenuation_gb10_generalised(
            total_stellar_mass, redshifts, **dust_params
        )
    else:
        raise ValueError(
            f"Dust model '{dust_model}' not supported for emission line attenuation."
        )

    # Pre-compute ratio k(V)/k(H-alpha) for the Calzetti law so we can
    # vectorise the per-line wavelength scaling later.
    if dust_law == 'calzetti':
        k_Halpha = _calzetti_k_lambda(6562.8)
        k_V = _calzetti_k_lambda(5500.0)
    else:
        raise ValueError(f"Dust law '{dust_law}' is not currently supported.")

    print(f"\nSaving dust-attenuated emission lines ({components})...")
    with h5py.File(galacticus_catalog, 'a') as f:
        # Find all emission-line datasets
        all_keys = list(f[node_data_path].keys())
        emission_keys = [k for k in all_keys if k.startswith('luminosityEmissionLine')]

        for orig_key in emission_keys:
            # Determine which component this key belongs to
            component = None
            for comp in components:
                cap_comp = comp[0].upper() + comp[1:]
                if orig_key.startswith(f'luminosityEmissionLine{cap_comp}:'):
                    component = comp
                    break
            if component is None:
                continue  # not a requested component

            # Extract rest-frame wavelength from the dataset name
            # Names are like balmerAlpha6565 -> wavelength 6565 Å
            line_name = orig_key.split(':')[1]
            match = re.search(r'\d+$', line_name)
            if match is None:
                print(f"  Warning: cannot parse wavelength from '{orig_key}', skipping")
                continue
            line_wavelength_AA = float(match.group())

            # Compute wavelength-dependent attenuation factor
            # A_lambda = A_Halpha * k(lambda) / k(Halpha)
            if dust_law == 'calzetti':
                k_line = _calzetti_k_lambda(line_wavelength_AA)
                # A_V from A_Halpha: A_V = A_Halpha * k_V / k_Halpha
                # A_lambda = A_V * k_line / k_V
                #          = A_Halpha * k_line / k_Halpha
                A_lambda = A_Halpha * k_line / k_Halpha  # shape (n_galaxies,)
            attenuation_factor = 10.0 ** (-0.4 * A_lambda)

            # Read original luminosities and attenuate
            orig_luminosities = f[f'{node_data_path}/{orig_key}'][:n_galaxies]
            attenuated = orig_luminosities * attenuation_factor

            # Save to new dataset
            new_key = f'dustAttenuated{orig_key[0].upper() + orig_key[1:]}'
            new_path = f'{node_data_path}/{new_key}'
            if new_path in f:
                del f[new_path]
            ds = f.create_dataset(new_path, data=attenuated)
            ds.attrs['comment'] = (
                f'Dust-attenuated {orig_key}. '
                f'Applied {dust_model} dust model with {dust_law} attenuation law.'
            ).encode('utf-8')
            ds.attrs['dust_model'] = dust_model.encode('utf-8')
            ds.attrs['dust_law'] = dust_law.encode('utf-8')

    n_saved = len([
        k for k in emission_keys
        if any(k.startswith(f'luminosityEmissionLine{c[0].upper()+c[1:]}:') for c in components)
    ])
    print(f"  Saved {n_saved} dust-attenuated emission line datasets.")


def save_dust_metadata_to_galacticus_file(galacticus_file, dust_model, dust_params, dust_law):
    """
    Save dust model metadata as attributes of a new ``DustModel`` group in the
    Galacticus HDF5 file.

    Parameters
    ----------
    galacticus_file : str
        Path to Galacticus HDF5 file (modified in-place).
    dust_model : str
        Name of the dust model.
    dust_params : dict
        Parameters for the dust model.
    dust_law : str
        Name of the attenuation law.
    """
    with h5py.File(galacticus_file, 'a') as f:
        # Remove existing group if present so we always write fresh metadata
        if 'DustModel' in f:
            del f['DustModel']
        grp = f.create_group('DustModel')
        grp.attrs['dust_model'] = dust_model.encode('utf-8')
        grp.attrs['dust_law'] = dust_law.encode('utf-8')
        for key, value in dust_params.items():
            if isinstance(value, str):
                grp.attrs[key] = value.encode('utf-8')
            else:
                grp.attrs[key] = value
    print(f"Saved DustModel metadata to {galacticus_file}")


def get_galaxy_count(filename):
    """Get the number of galaxies in the catalog."""
    # Detect format and get appropriate path
    format_type, base_path = detect_galacticus_format(filename)
    
    with h5py.File(filename, 'r') as f:
        # Get the length of an array in nodeData
        node_data_path = f'{base_path}/nodeData'
        
        if format_type == 'lightcone':
            # Use lightcone-specific redshift array
            n_galaxies = len(f[f'{node_data_path}/lightconeRedshiftObserved'][:])
        else:
            # For fixed-time, we can use any dataset in nodeData
            # Let's use diskStarFormationHistoryMass
            n_galaxies = len(f[f'{node_data_path}/diskStarFormationHistoryMass'][:])
    
    return n_galaxies


def calculate_catalog_magnitudes(sed_template_file, galacticus_catalog, 
                                 bandpasses, output_file=None,
                                 max_galaxies=None, component='total',
                                 save_to_input=False, copy_input=True,
                                 check_existing=True, obs_wavelengths=None,
                                 n_jobs=1, dust_model=None, dust_params=None,
                                 dust_law='calzetti'):
    """
    Calculate magnitudes for all galaxies in a Galacticus catalog.
    
    Automatically detects whether the catalog is in lightcone or fixed-time format
    and handles paths accordingly.
    
    Parameters
    ----------
    sed_template_file : str
        Path to SED template HDF5 file
    galacticus_catalog : str
        Path to Galacticus catalog HDF5 file (lightcone or fixed-time format)
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
        If True, save magnitudes to the Galacticus catalog file.
        For lightcone: /Lightcone/Output1/nodeData/apparentMagnitudeRomanWFI:<filter>
        or, with dust: /Lightcone/Output1/nodeData/dustAttenuatedApparentMagnitudeRomanWFI:<filter>
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
    n_jobs : int, optional
        Number of parallel worker processes to use. ``1`` (default) runs
        sequentially. ``-1`` uses all available CPU cores. Values greater
        than 1 request that exact number of workers.
    dust_model : str, optional
        Name of the dust attenuation model to apply to emission lines.
        Currently only ``'gb10_generalised'`` is supported. If ``None``
        (default), no dust attenuation is applied (dust-free magnitudes).
    dust_params : dict, optional
        Parameters for the dust model. Required when *dust_model* is not None.
    dust_law : str, optional
        Attenuation law. Currently only ``'calzetti'`` is supported.
        Default is ``'calzetti'``.
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'magnitudes': 2D array of shape (n_galaxies, n_filters)
        - 'filter_names': list of filter names
        - 'redshifts': array of galaxy redshifts
        - 'galaxy_indices': array of galaxy indices processed
        - 'output_file': path to file where magnitudes were saved (if applicable)
        - 'format_type': detected format ('lightcone' or 'fixed-time')
        - 'base_path': base path used in the HDF5 file
        - 'dust_model': dust model used (None if no dust)
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

    # Determine the dataset name prefix depending on whether dust is applied
    magnitude_prefix = 'dustAttenuatedApparentMagnitudeRomanWFI' if dust_model is not None else 'apparentMagnitudeRomanWFI'
    
    # Check if magnitudes already exist
    if check_existing and save_to_input:
        print("\nChecking for existing magnitude datasets...")
        existing_filters = []
        node_data_path = f'{base_path}/nodeData'
        with h5py.File(working_file, 'r') as f:
            for filter_name in filter_names:
                dataset_path = f'{node_data_path}/{magnitude_prefix}:{filter_name}'
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
    calc = SEDCalculator(sed_template_file, cosmology=unit_cosmo)
    
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
    if format_type == 'lightcone':
        # Lightcone: per-galaxy redshifts
        with h5py.File(working_file, 'r') as f:
            redshifts[:] = f[f'{base_path}/nodeData/lightconeRedshiftObserved'][:n_galaxies]
    else:
        # Fixed-time: calculate redshift from outputTime
        from SEDfromSFH import outputTime_to_redshift
        with h5py.File(working_file, 'r') as f:
            outputTime = f[base_path].attrs['outputTime']
            redshift = outputTime_to_redshift(outputTime)
            redshifts[:] = float(redshift)  # All galaxies at same redshift
        print(f"  Fixed-time output at z={redshifts[0]:.4f} (outputTime={outputTime:.4f} Gyr)")
    
    # Calculate magnitudes for each galaxy
    if n_jobs == 0 or (n_jobs < -1):
        raise ValueError(f"n_jobs must be -1 (all CPUs), 1 (sequential), or a positive integer; got {n_jobs}")
    actual_n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    print(f"\nCalculating magnitudes in {n_filters} filters for {n_galaxies} galaxies...")
    if actual_n_jobs > 1:
        print(f"Using {actual_n_jobs} parallel worker processes")
    print("Progress: ", end='', flush=True)

    start_time = time.time()

    if actual_n_jobs == 1:
        # Sequential path (original behaviour)
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
                    obs_wavelengths=obs_wavelengths,
                    dust_model=dust_model,
                    dust_params=dust_params,
                    dust_law=dust_law,
                )

                # Store results in array
                for j, filter_name in enumerate(filter_names):
                    magnitude_array[i, j] = mags[filter_name]

            except Exception as e:
                print(f"\nWarning: Failed to process galaxy {i}: {e}")
                # magnitude_array already initialized with NaN values
    else:
        # Parallel path using multiprocessing.Pool
        cosmology = calc.cosmo
        worker_args = [
            (i, working_file, component, obs_wavelengths)
            for i in range(n_galaxies)
        ]
        with multiprocessing.Pool(
            processes=actual_n_jobs,
            initializer=_init_worker,
            initargs=(sed_template_file, filter_names, cosmology,
                      dust_model, dust_params, dust_law)
        ) as pool:
            n_done = 0
            for i, mags in pool.imap_unordered(_process_galaxy_worker, worker_args):
                n_done += 1
                if n_done % max(1, n_galaxies // 20) == 0:
                    progress = n_done / n_galaxies * 100
                    elapsed = time.time() - start_time
                    rate = n_done / elapsed
                    print(f"{progress:.0f}% ", end='', flush=True)
                if mags is not None:
                    for j, filter_name in enumerate(filter_names):
                        magnitude_array[i, j] = mags[filter_name]
                # If mags is None the row stays as NaN (already initialised)
    
    print("Done!")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.1f} seconds ({elapsed_time/n_galaxies:.2f} sec/galaxy)")
    
    # Prepare results
    results = {
        'magnitudes': magnitude_array,
        'filter_names': filter_names,
        'redshifts': redshifts,
        'galaxy_indices': np.arange(n_galaxies),
        'format_type': format_type,
        'base_path': base_path,
        'dust_model': dust_model,
        'magnitude_prefix': magnitude_prefix,
    }
    
    # Save to file
    if save_to_input:
        print(f"\nSaving magnitudes to Galacticus file: {working_file}...")
        save_magnitudes_to_galacticus_file(working_file, results, component, format_type, base_path)
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


def save_magnitudes_to_galacticus_file(galacticus_file, results, component='total', 
                                       format_type=None, base_path=None):
    """
    Save magnitude data directly to the Galacticus HDF5 file.
    
    Magnitudes are saved to datasets with paths like:
    /Lightcone/Output1/nodeData/apparentMagnitudeRomanWFI:<filter>
    or, when dust attenuation was applied:
    /Lightcone/Output1/nodeData/dustAttenuatedApparentMagnitudeRomanWFI:<filter>
    
    Parameters
    ----------
    galacticus_file : str
        Path to Galacticus HDF5 file (will be modified)
    results : dict
        Results dictionary from calculate_catalog_magnitudes
    component : str, optional
        Component type used for magnitude calculation. Default is 'total'.
    format_type : str, optional
        Format type ('lightcone' or 'fixed-time'). If None, will be detected.
    base_path : str, optional
        Base path in HDF5 file. If None, will be detected.
    """
    # Detect format if not provided
    if format_type is None or base_path is None:
        format_type, base_path = detect_galacticus_format(galacticus_file)
    
    filter_names = results['filter_names']
    magnitude_array = results['magnitudes']
    n_galaxies = len(results['galaxy_indices'])
    magnitude_prefix = results.get('magnitude_prefix', 'apparentMagnitudeRomanWFI')
    dust_model = results.get('dust_model')

    # Determine comment based on component
    dust_suffix = (
        f" Dust attenuation applied using '{dust_model}' model."
        if dust_model is not None
        else ""
    )
    if component == 'total':
        comment = f"Total AB magnitude (disk + spheroid + AGN) including emission lines. Note there is currently no AGN continuum.{dust_suffix}"
    elif component == 'disk':
        comment = f"Disk AB magnitude including emission lines.{dust_suffix}"
    elif component == 'spheroid':
        comment = f"Spheroid AB magnitude including emission lines.{dust_suffix}"
    elif component == 'AGN':
        comment = f"AGN AB magnitude (emission lines only).{dust_suffix}"
    else:
        comment = f"{component} AB magnitude.{dust_suffix}"
    
    with h5py.File(galacticus_file, 'a') as f:
        # Create or access the nodeData group
        node_data_path = f'{base_path}/nodeData'
        if node_data_path not in f:
            raise ValueError(f"Path {node_data_path} not found in {galacticus_file}")
        
        for j, filter_name in enumerate(filter_names):
            dataset_path = f'{node_data_path}/{magnitude_prefix}:{filter_name}'
            
            # Delete existing dataset if it exists
            if dataset_path in f:
                print(f"  Deleting existing dataset: {dataset_path}")
                del f[dataset_path]
            
            # Create new dataset
            print(f"  Creating dataset: {dataset_path}")
            dataset = f.create_dataset(
                dataset_path,
                data=magnitude_array[:, j]
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
    parser.add_argument('-j', '--n-jobs', type=int, default=1,
                       help='Number of parallel worker processes. '
                            '1 = sequential (default). -1 = use all available CPUs.')
    
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
    
    # Dust attenuation options
    parser.add_argument('--dust-config', metavar='DUST_CONFIG_JSON', default=None,
                       help='Path to a JSON file specifying the dust attenuation model. '
                            'The file must contain the keys "dust_model", "dust_params", '
                            'and "dust_law". When provided, dust-attenuated magnitudes are '
                            'saved as dustAttenuatedApparentMagnitudeRomanWFI:<filter> '
                            'alongside dust-attenuated emission line luminosities '
                            '(dustAttenuatedLuminosityEmissionLine*), and a DustModel '
                            'metadata group is written to the output file.')

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
    print(f"Parallel workers: {args.n_jobs} "
          f"({'all CPUs' if args.n_jobs == -1 else 'sequential' if args.n_jobs == 1 else f'{args.n_jobs} workers'})")
    
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

    # Load dust config if provided
    dust_model = None
    dust_params = None
    dust_law = 'calzetti'
    if args.dust_config is not None:
        if not os.path.exists(args.dust_config):
            print(f"\nError: Dust config file not found: {args.dust_config}")
            sys.exit(1)
        dust_model, dust_params, dust_law = load_dust_config(args.dust_config)
        print(f"\nDust attenuation enabled:")
        print(f"  Model: {dust_model}")
        print(f"  Law:   {dust_law}")
        print(f"  Params: {dust_params}")
    
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
        obs_wavelengths=obs_wavelengths,
        n_jobs=args.n_jobs,
        dust_model=dust_model,
        dust_params=dust_params,
        dust_law=dust_law,
    )
    
    if results is not None:
        # When dust is enabled and we saved to an HDF5 file, also save the
        # dust-attenuated emission lines and the DustModel metadata group.
        if dust_model is not None and 'output_file' in results:
            output_hdf5 = results['output_file']
            print(f"\nCalculating dust-attenuated emission lines...")
            calculate_and_save_dust_attenuated_emission_lines(
                output_hdf5,
                dust_model=dust_model,
                dust_params=dust_params,
                dust_law=dust_law,
                max_galaxies=args.max_galaxies,
            )
            save_dust_metadata_to_galacticus_file(output_hdf5, dust_model, dust_params, dust_law)

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
