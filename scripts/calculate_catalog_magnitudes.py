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
from galacticus_sed_calculator.sed_calculator import detect_galacticus_format, outputTime_to_redshift
from galacticus_sed_calculator.dust_attenuation import (
    dust_attenuation_gb10_generalised,
    read_dust_model_from_catalog,
    _calzetti_k_lambda,
)

# Default configuration
DEFAULT_SED_TEMPLATE = "data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5"
DEFAULT_FILTERS = ["F062", "F087", "F106", "F129", "F158", "F184", "F213"]
DEFAULT_WAVELENGTH_MIN = 4000  # Angstroms
DEFAULT_WAVELENGTH_MAX = 24000  # Angstroms
DEFAULT_WAVELENGTH_NPOINTS = 2000

# Module-level state for worker processes (populated by _init_worker)
_worker_state = {}


def _init_worker(sed_template_file, filter_names, cosmology,
                 dust_model=None, dust_params=None, dust_law='calzetti',
                 random_uniform_index=None):
    """
    Initialize per-worker process state.

    Called once per worker process when using multiprocessing.Pool. Loads the
    SEDCalculator and Roman bandpass filters so they are reused across all
    galaxy tasks assigned to that worker.
    """
    global _worker_state
    calc = SEDCalculator(sed_template_file, cosmology=cosmology)
    roman = stpsf.WFI()
    bandpasses = {f: roman._get_synphot_bandpass(f) for f in filter_names}
    _worker_state['calc'] = calc
    _worker_state['bandpasses'] = bandpasses
    _worker_state['dust_model'] = dust_model
    _worker_state['dust_params'] = dust_params
    _worker_state['dust_law'] = dust_law
    _worker_state['random_uniform_index'] = random_uniform_index


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
        (galaxy_index, result_dict) where result_dict contains:
        - 'mags': dict mapping filter names to dust-free magnitude values
        - 'dust_mags': dict mapping filter names to dust-attenuated magnitudes
          (only present when a dust model is configured in _worker_state)
        Returns (galaxy_index, None) if the calculation failed.
    """
    i, working_file, component, obs_wavelengths = args
    try:
        result = {}
        result['mags'] = _worker_state['calc'].calculate_magnitudes(
            working_file,
            galIndex=i,
            bandpasses=_worker_state['bandpasses'],
            component=component,
            obs_wavelengths=obs_wavelengths
        )
        dust_model = _worker_state.get('dust_model')
        if dust_model is not None:
            result['dust_mags'] = _worker_state['calc'].calculate_magnitudes(
                working_file,
                galIndex=i,
                bandpasses=_worker_state['bandpasses'],
                component=component,
                obs_wavelengths=obs_wavelengths,
                dust_model=dust_model,
                dust_params=_worker_state['dust_params'],
                dust_law=_worker_state['dust_law'],
                random_uniform_index=_worker_state.get('random_uniform_index'),
            )
        return i, result
    except Exception as e:
        print(f"\nWarning: Failed to process galaxy {i}: {type(e).__name__}: {e}")
        return i, None


def load_dust_config(config_file):
    """
    Load dust model configuration from a JSON file.

    Parameters
    ----------
    config_file : str
        Path to a JSON file containing the dust model configuration.
        Required keys: 'dust_model', 'dust_params', 'dust_law'.
        Optional key: 'random_uniform_index' (int) — column index into the
        ``nodeData/randomUniform`` dataset for reproducible scatter.

    Returns
    -------
    dust_model : str
        Name of the dust model (e.g. 'gb10_generalised').
    dust_params : dict
        Dictionary of parameters for the dust model.
    dust_law : str
        Name of the attenuation law (e.g. 'calzetti').
    random_uniform_index : int or None
        Column index into ``nodeData/randomUniform`` used to draw reproducible
        per-galaxy scatter values.  ``None`` if not specified in the config.

    Examples
    --------
    Example config file::

        {
            "dust_model": "gb10_generalised",
            "dust_params": {
                "delta_0": 0.275,
                "delta_z": -1.614,
                "delta_M": -0.834,
                "delta_Mz": -0.708,
                "attenuation_scatter": 0.0
            },
            "dust_law": "calzetti"
        }

    To enable reproducible per-galaxy scatter supply a non-zero
    ``attenuation_scatter`` and add ``"random_uniform_index"``::

        {
            "dust_model": "gb10_generalised",
            "dust_params": {
                "delta_0": 0.275,
                "delta_z": -1.614,
                "delta_M": -0.834,
                "delta_Mz": -0.708,
                "attenuation_scatter": 0.3
            },
            "dust_law": "calzetti",
            "random_uniform_index": 0
        }
    """
    with open(config_file, 'r') as f:
        config = json.load(f)

    for key in ('dust_model', 'dust_params', 'dust_law'):
        if key not in config:
            raise ValueError(
                f"Dust config file '{config_file}' is missing required key '{key}'. "
                "Required keys: dust_model, dust_params, dust_law."
            )

    random_uniform_index = config.get('random_uniform_index', None)
    return config['dust_model'], config['dust_params'], config['dust_law'], random_uniform_index


def calculate_dust_attenuated_emission_lines(galacticus_file, base_path, format_type,
                                             dust_model, dust_params, dust_law,
                                             n_galaxies=None,
                                             random_uniform_index=None):
    """
    Calculate dust-attenuated emission line luminosities for all galaxies.

    Reads each ``luminosityEmissionLine*`` dataset from the HDF5 file, applies
    dust attenuation using the specified model, and returns a dictionary of
    attenuated arrays named ``dustAttenuatedLuminosityEmissionLine*``.

    Parameters
    ----------
    galacticus_file : str
        Path to the Galacticus HDF5 file.
    base_path : str
        Base path within the HDF5 file (e.g. '/Lightcone/Output1').
    format_type : str
        Catalog format: 'lightcone' or 'fixed-time'.
    dust_model : str
        Dust attenuation model name. Currently only 'gb10_generalised' is supported.
    dust_params : dict
        Parameters for the dust model.
    dust_law : str
        Attenuation law name. Currently only 'calzetti' is supported.
    n_galaxies : int, optional
        Number of galaxies to process. If None, processes all galaxies.
    random_uniform_index : int, optional
        Column index into the ``nodeData/randomUniform`` dataset (shape
        ``[n_galaxies, n_columns]``) used to draw reproducible per-galaxy
        scatter values.  When provided, the corresponding column is passed
        to the dust model as ``random_uniform``, enabling reproducible
        scatter without re-drawing new random numbers.  Has no effect if
        ``attenuation_scatter`` in ``dust_params`` is 0.

    Returns
    -------
    attenuated_datasets : dict
        Mapping of dataset name (same as the original ``luminosityEmissionLine*``
        name, e.g. ``'luminosityEmissionLineDisk:balmerAlpha6565'``) to a NumPy
        array of dust-attenuated luminosities.  The datasets are intended to be
        written into a ``dustAttenuatedNodeData`` group rather than into
        ``nodeData``, so they intentionally carry the same name as their
        dust-free counterparts.
    """
    node_data_path = f'{base_path}/nodeData'

    with h5py.File(galacticus_file, 'r') as f:
        nd = f[node_data_path]

        # Stellar mass for dust model
        disk_mass = nd['diskMassStellar'][:]
        spheroid_mass = nd['spheroidMassStellar'][:] if 'spheroidMassStellar' in nd else np.zeros_like(disk_mass)
        total_stellar_mass = disk_mass + spheroid_mass

        # Redshifts
        if format_type == 'lightcone':
            redshifts = nd['lightconeRedshiftObserved'][:]
        else:
            outputTime = f[base_path].attrs['outputTime']
            redshift_val = float(outputTime_to_redshift(outputTime))
            redshifts = np.full(len(total_stellar_mass), redshift_val)

        # All emission line dataset names and their data
        emission_line_names = [name for name in nd.keys()
                               if name.startswith('luminosityEmissionLine')]
        emission_line_data = {name: nd[name][:] for name in emission_line_names}

        # Pre-saved random uniform values for reproducible scatter
        random_uniform = None
        if random_uniform_index is not None:
            random_uniform_path = f'{node_data_path}/randomUniform'
            if random_uniform_path in f:
                ru_dataset = f[random_uniform_path]
                if ru_dataset.ndim == 2:
                    if random_uniform_index < ru_dataset.shape[1]:
                        random_uniform = ru_dataset[:, random_uniform_index]
                    else:
                        raise ValueError(
                            f"random_uniform_index={random_uniform_index} is out of bounds. "
                            f"Dataset has {ru_dataset.shape[1]} columns."
                        )
                else:
                    raise ValueError(
                        f"randomUniform dataset has unexpected shape: {ru_dataset.shape}. "
                        "Expected a 2-D array [n_galaxies, n_columns]."
                    )
            else:
                raise ValueError(
                    f"random_uniform_index specified but '{random_uniform_path}' "
                    "not found in file."
                )

    # Limit to n_galaxies if requested
    if n_galaxies is not None:
        total_stellar_mass = total_stellar_mass[:n_galaxies]
        redshifts = redshifts[:n_galaxies]
        emission_line_data = {k: v[:n_galaxies] for k, v in emission_line_data.items()}
        if random_uniform is not None:
            random_uniform = random_uniform[:n_galaxies]

    # Compute per-galaxy H-alpha attenuation
    if dust_model == 'gb10_generalised':
        A_Halpha = dust_attenuation_gb10_generalised(
            total_stellar_mass, redshifts,
            **dust_params,
            random_uniform=random_uniform,
        )
    else:
        raise ValueError(
            f"Dust model '{dust_model}' not supported in "
            "calculate_dust_attenuated_emission_lines. "
            "Currently only 'gb10_generalised' is implemented."
        )

    # k(H-alpha) value used to re-scale to A(lambda)
    HALPHA_WAVELENGTH_AA = 6562.8  # H-alpha rest-frame wavelength in Angstroms
    k_Halpha = _calzetti_k_lambda(HALPHA_WAVELENGTH_AA)

    attenuated_datasets = {}
    for dataset_name, luminosities in emission_line_data.items():
        # Extract rest-frame wavelength (trailing digits in the line name part)
        # e.g. 'luminosityEmissionLineDisk:balmerAlpha6565' -> 6565
        line_name_part = dataset_name.split(':')[-1]
        match = re.search(r'\d+$', line_name_part)
        if match is None:
            continue
        rest_wavelength_AA = float(match.group())

        if dust_law == 'calzetti':
            k_line = _calzetti_k_lambda(rest_wavelength_AA)
            # A(lambda) = A_Halpha * k(lambda) / k(H-alpha)
            A_lambda = A_Halpha * k_line / k_Halpha
        else:
            raise ValueError(
                f"Dust law '{dust_law}' not supported. "
                "Currently only 'calzetti' is implemented."
            )

        attenuation_factor = 10.0 ** (-0.4 * A_lambda)
        attenuated_lum = luminosities * attenuation_factor

        # Keep the same dataset name; the caller writes these into a
        # separate 'dustAttenuatedNodeData' group.
        attenuated_datasets[dataset_name] = attenuated_lum

    return attenuated_datasets


def save_dust_model_metadata(group, dust_model, dust_params, dust_law,
                             random_uniform_index=None):
    """
    Attach dust model metadata as attributes to an HDF5 group.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group to which the attributes will be attached (typically the
        ``dustAttenuatedNodeData`` group).
    dust_model : str
        Name of the dust model.
    dust_params : dict
        Dictionary of dust model parameters.  Stored as a JSON string under the
        ``dust_params`` attribute so that the nested structure is preserved.
    dust_law : str
        Name of the attenuation law.
    random_uniform_index : int or None, optional
        Column index into ``nodeData/randomUniform`` used for reproducible
        per-galaxy scatter.  Stored as the ``random_uniform_index`` attribute
        when not ``None``; the attribute is omitted otherwise.
    """
    group.attrs['dust_model'] = dust_model
    group.attrs['dust_law'] = dust_law
    group.attrs['dust_params'] = json.dumps(dust_params)
    if random_uniform_index is not None:
        group.attrs['random_uniform_index'] = int(random_uniform_index)


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
                                 dust_law='calzetti', random_uniform_index=None):
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
        For fixed-time: /Outputs/Output1/nodeData/apparentMagnitudeRomanWFI:<filter>
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
        Dust attenuation model to use for emission lines. When provided,
        dust-attenuated magnitudes (``dustAttenuatedApparentMagnitudeRomanWFI:<filter>``)
        and emission line luminosities
        (``dustAttenuatedLuminosityEmissionLine*``) are also saved, together
        with a ``DustModel`` metadata group.
        Currently only ``'gb10_generalised'`` is supported. Default is None.
    dust_params : dict, optional
        Parameters for the dust model. Required when ``dust_model`` is not None.
    dust_law : str, optional
        Attenuation law to use. Default is ``'calzetti'``.
    random_uniform_index : int, optional
        Column index into the ``nodeData/randomUniform`` dataset used to draw
        reproducible per-galaxy scatter values.  When provided, the same
        column of pre-saved random numbers is used for both the per-galaxy
        magnitude calculations and the vectorised emission-line attenuation,
        giving consistent, reproducible results.  Has no effect when
        ``dust_model`` is None or ``attenuation_scatter`` in ``dust_params``
        is 0.  Default is None (random scatter drawn fresh each run).
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'magnitudes': 2D array of shape (n_galaxies, n_filters)
        - 'dust_magnitudes': 2D array of shape (n_galaxies, n_filters),
          only present when dust_model is specified
        - 'dust_emission_lines': dict mapping dataset name to attenuated array,
          only present when dust_model is specified
        - 'filter_names': list of filter names
        - 'redshifts': array of galaxy redshifts
        - 'galaxy_indices': array of galaxy indices processed
        - 'output_file': path to file where magnitudes were saved (if applicable)
        - 'format_type': detected format ('lightcone' or 'fixed-time')
        - 'base_path': base path used in the HDF5 file
        - 'dust_model': dust model name (only present when dust_model is specified)
        - 'dust_params': dust model parameters (only present when dust_model is specified)
        - 'dust_law': dust attenuation law (only present when dust_model is specified)
        - 'random_uniform_index': int or None (only present when dust_model is specified)
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
    
    # Check if magnitudes already exist
    if check_existing and save_to_input:
        print("\nChecking for existing magnitude datasets...")
        existing_filters = []
        node_data_path = f'{base_path}/nodeData'
        with h5py.File(working_file, 'r') as f:
            for filter_name in filter_names:
                dataset_path = f'{node_data_path}/apparentMagnitudeRomanWFI:{filter_name}'
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
    if dust_model is not None:
        dust_magnitude_array = np.full((n_galaxies, n_filters), np.nan)
    redshifts = np.zeros(n_galaxies)
    
    # Read redshifts for all galaxies
    print("\nReading galaxy redshifts...")
    if format_type == 'lightcone':
        # Lightcone: per-galaxy redshifts
        with h5py.File(working_file, 'r') as f:
            redshifts[:] = f[f'{base_path}/nodeData/lightconeRedshiftObserved'][:n_galaxies]
    else:
        # Fixed-time: calculate redshift from outputTime
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
    if dust_model is not None:
        print(f"Dust model: {dust_model} (computing dust-attenuated magnitudes too)")
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
                # Calculate dust-free magnitudes for this galaxy
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

                # Calculate dust-attenuated magnitudes if requested
                if dust_model is not None:
                    dust_mags = calc.calculate_magnitudes(
                        working_file,
                        galIndex=i,
                        bandpasses=bandpasses,
                        component=component,
                        obs_wavelengths=obs_wavelengths,
                        dust_model=dust_model,
                        dust_params=dust_params,
                        dust_law=dust_law,
                        random_uniform_index=random_uniform_index,
                    )
                    for j, filter_name in enumerate(filter_names):
                        dust_magnitude_array[i, j] = dust_mags[filter_name]

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
                      dust_model, dust_params, dust_law, random_uniform_index)
        ) as pool:
            n_done = 0
            for i, result in pool.imap_unordered(_process_galaxy_worker, worker_args):
                n_done += 1
                if n_done % max(1, n_galaxies // 20) == 0:
                    progress = n_done / n_galaxies * 100
                    elapsed = time.time() - start_time
                    rate = n_done / elapsed
                    print(f"{progress:.0f}% ", end='', flush=True)
                if result is not None:
                    for j, filter_name in enumerate(filter_names):
                        magnitude_array[i, j] = result['mags'][filter_name]
                    if dust_model is not None and 'dust_mags' in result:
                        for j, filter_name in enumerate(filter_names):
                            dust_magnitude_array[i, j] = result['dust_mags'][filter_name]
                # If result is None the rows stay as NaN (already initialised)
    
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
        'base_path': base_path
    }

    # Add dust results if a dust model was specified
    if dust_model is not None:
        results['dust_magnitudes'] = dust_magnitude_array
        results['dust_model'] = dust_model
        results['dust_params'] = dust_params
        results['dust_law'] = dust_law
        results['random_uniform_index'] = random_uniform_index
        print(f"\nCalculating dust-attenuated emission line luminosities...")
        results['dust_emission_lines'] = calculate_dust_attenuated_emission_lines(
            working_file, base_path, format_type,
            dust_model, dust_params, dust_law,
            n_galaxies=n_galaxies,
            random_uniform_index=random_uniform_index,
        )
        print(f"  Computed {len(results['dust_emission_lines'])} dust-attenuated emission line datasets")
    
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
    
    Dust-free magnitudes are written to:
    /Lightcone/Output1/nodeData/apparentMagnitudeRomanWFI:<filter>
    or
    /Outputs/Output1/nodeData/apparentMagnitudeRomanWFI:<filter>

    When dust results are present, a parallel ``dustAttenuatedNodeData`` group
    is created at the same level as ``nodeData``.  Dust-attenuated magnitudes
    and emission lines are stored there with the **same dataset names** as their
    dust-free counterparts, e.g.::

        /Lightcone/Output1/dustAttenuatedNodeData/apparentMagnitudeRomanWFI:F062
        /Lightcone/Output1/dustAttenuatedNodeData/luminosityEmissionLineDisk:balmerAlpha6565

    The dust model used is recorded as attributes of the
    ``dustAttenuatedNodeData`` group (``dust_model``, ``dust_law``,
    ``dust_params`` stored as a JSON string).
    
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
    
    # Determine comment based on component
    if component == 'total':
        comment = "Total AB magnitude (disk + spheroid + AGN) including emission lines. Note there is currently no AGN continuum."
    elif component == 'disk':
        comment = "Disk AB magnitude including emission lines"
    elif component == 'spheroid':
        comment = "Spheroid AB magnitude including emission lines"
    elif component == 'AGN':
        comment = "AGN AB magnitude (emission lines only)"
    else:
        comment = f"{component} AB magnitude"
    
    with h5py.File(galacticus_file, 'a') as f:
        # Create or access the nodeData group
        node_data_path = f'{base_path}/nodeData'
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
                data=magnitude_array[:, j]
            )
            
            # Add attributes
            dataset.attrs['comment'] = comment.encode('utf-8')
            dataset.attrs['filter'] = filter_name.encode('utf-8')

        # Save dust-attenuated data into a separate dustAttenuatedNodeData group
        if 'dust_magnitudes' in results or 'dust_emission_lines' in results:
            dust_group_path = f'{base_path}/dustAttenuatedNodeData'
            dust_comment = comment.replace(
                "AB magnitude", "dust-attenuated AB magnitude"
            )

            # Create (or overwrite) the dustAttenuatedNodeData group
            if dust_group_path in f:
                del f[dust_group_path]
            dust_group = f.create_group(dust_group_path)

            # Attach dust model metadata to the group
            save_dust_model_metadata(
                dust_group,
                results['dust_model'],
                results['dust_params'],
                results['dust_law'],
                random_uniform_index=results.get('random_uniform_index'),
            )

            # Dust-attenuated magnitudes (same dataset names as dust-free)
            if 'dust_magnitudes' in results:
                dust_magnitude_array = results['dust_magnitudes']
                for j, filter_name in enumerate(filter_names):
                    dust_path = f'{dust_group_path}/apparentMagnitudeRomanWFI:{filter_name}'
                    print(f"  Creating dataset: {dust_path}")
                    ds = f.create_dataset(
                        dust_path, data=dust_magnitude_array[:, j]
                    )
                    ds.attrs['comment'] = dust_comment.encode('utf-8')
                    ds.attrs['filter'] = filter_name.encode('utf-8')

            # Dust-attenuated emission lines (same dataset names as dust-free)
            if 'dust_emission_lines' in results:
                for ds_name, attenuated_lum in results['dust_emission_lines'].items():
                    ds_path = f'{dust_group_path}/{ds_name}'
                    print(f"  Creating dataset: {ds_path}")
                    f.create_dataset(ds_path, data=attenuated_lum)
    
    print(f"Saved {len(filter_names)} magnitude datasets to {galacticus_file}")
    if 'dust_magnitudes' in results:
        print(f"Saved {len(filter_names)} dust-attenuated magnitude datasets to {galacticus_file}")
    if 'dust_emission_lines' in results:
        print(f"Saved {len(results['dust_emission_lines'])} dust-attenuated emission line datasets to {galacticus_file}")


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
    
    # System options
    parser.add_argument('--magnitude-system', default='AB',
                       choices=['AB', 'ST', 'Vega'],
                       help='Magnitude system to use')
    
    # Dust attenuation options
    parser.add_argument('--dust-config', metavar='DUST_CONFIG',
                       help='Path to a JSON file specifying the dust attenuation model.  '
                            'When provided, dust-attenuated magnitudes '
                            '(dustAttenuatedApparentMagnitudeRomanWFI:<filter>) and '
                            'emission line luminosities '
                            '(dustAttenuatedLuminosityEmissionLine*) are also saved, '
                            'together with a DustModel metadata group.  '
                            'Required JSON keys: dust_model, dust_params, dust_law.')
    
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
    random_uniform_index = None
    if args.dust_config:
        if not os.path.exists(args.dust_config):
            print(f"\nError: Dust config file not found: {args.dust_config}")
            sys.exit(1)
        print(f"\nLoading dust config from: {args.dust_config}")
        dust_model, dust_params, dust_law, random_uniform_index = load_dust_config(args.dust_config)
        print(f"  dust_model: {dust_model}")
        print(f"  dust_law:   {dust_law}")
        print(f"  dust_params: {dust_params}")
        if random_uniform_index is not None:
            print(f"  random_uniform_index: {random_uniform_index}")
    
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
        random_uniform_index=random_uniform_index,
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
