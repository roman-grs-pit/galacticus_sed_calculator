#!/usr/bin/env python3
"""
Create a downsampled Galacticus catalog with SEDs.

This script applies user-defined selection cuts (e.g. redshift range, apparent
magnitude limit, or any nodeData property) to a Galacticus HDF5 catalog.
Galaxies that pass the cuts are copied — including all their data and metadata
— to a new HDF5 file.  An observed-frame SED (flux-density spectrum) is then
calculated for each selected galaxy and stored in the output file.

Typical use-case: create a DESI BGS-like sample by selecting galaxies brighter
than a given magnitude threshold in a specified redshift range.

Example
-------
::

    python create_downsampled_catalog.py \\
        input_catalog.hdf5 \\
        sed_template.hdf5 \\
        output_catalog.hdf5 \\
        --redshift-min 0.05 --redshift-max 0.4 \\
        --magnitude-max 20.2 --magnitude-filter F158

Output datasets added to the output catalog
-------------------------------------------
* ``<base_path>/nodeData/observedSEDWavelengths``
  1-D array of wavelengths in Angstroms, shape ``(n_wavelengths,)``.
* ``<base_path>/nodeData/observedSED``
  2-D array of observed-frame flux densities in erg/(s cm² Hz),
  shape ``(n_selected_galaxies, n_wavelengths)``.
"""

import json
import os
import sys
import time
import argparse

import numpy as np
import h5py
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

# Add parent directory to path so the package can be imported when this script
# is run directly from the scripts/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator import SEDCalculator
from galacticus_sed_calculator.sed_calculator import (
    detect_galacticus_format,
    outputTime_to_redshift,
)

# Default wavelength grid
DEFAULT_WAVELENGTH_MIN = 4000    # Angstroms
DEFAULT_WAVELENGTH_MAX = 24000   # Angstroms
DEFAULT_WAVELENGTH_NPOINTS = 2000


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _get_n_galaxies(galacticus_file, base_path):
    """Return the total number of galaxies in the nodeData group."""
    node_data_path = f'{base_path}/nodeData'
    with h5py.File(galacticus_file, 'r') as f:
        nd = f[node_data_path]
        first_key = next(iter(nd.keys()))
        return len(nd[first_key])


def get_redshifts(galacticus_file, base_path, format_type, n_galaxies=None):
    """
    Read per-galaxy redshifts from a Galacticus catalog.

    For lightcone format the per-galaxy observed redshift array is read
    directly.  For fixed-time format a constant redshift derived from the
    output time attribute is returned.

    Parameters
    ----------
    galacticus_file : str
        Path to the Galacticus HDF5 catalog.
    base_path : str
        Base path within the file (e.g. ``'/Lightcone/Output1'``).
    format_type : str
        ``'lightcone'`` or ``'fixed-time'``.
    n_galaxies : int or None
        Number of galaxies; required for fixed-time format if the caller does
        not already know it.  When ``None`` it is inferred from the first
        dataset found in ``nodeData``.

    Returns
    -------
    redshifts : ndarray, shape (n_galaxies,)
    """
    with h5py.File(galacticus_file, 'r') as f:
        if format_type == 'lightcone':
            redshifts = f[f'{base_path}/nodeData/lightconeRedshiftObserved'][:]
        else:
            outputTime = f[base_path].attrs['outputTime']
            redshift_val = float(outputTime_to_redshift(outputTime))
            if n_galaxies is None:
                nd = f[f'{base_path}/nodeData']
                first_key = next(iter(nd.keys()))
                n_galaxies = len(nd[first_key])
            redshifts = np.full(n_galaxies, redshift_val)
    return redshifts


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def filter_galaxies(galacticus_file, base_path, format_type,
                    redshift_min=None, redshift_max=None,
                    magnitude_max=None, magnitude_filter=None,
                    property_cuts=None):
    """
    Apply selection cuts and return indices of galaxies that pass all cuts.

    Parameters
    ----------
    galacticus_file : str
        Path to the Galacticus HDF5 file.
    base_path : str
        Base path within the HDF5 file (e.g. ``'/Lightcone/Output1'``).
    format_type : str
        ``'lightcone'`` or ``'fixed-time'``.
    redshift_min : float or None
        Minimum redshift (inclusive).  ``None`` means no lower bound.
    redshift_max : float or None
        Maximum redshift (inclusive).  ``None`` means no upper bound.
    magnitude_max : float or None
        Apparent magnitude upper limit: galaxies *brighter* than this value
        (i.e. with a smaller magnitude number) pass the cut.  ``None`` means
        no magnitude cut.  Requires ``magnitude_filter``.
    magnitude_filter : str or None
        Name of the Roman WFI filter whose stored magnitude is used for the
        cut (e.g. ``'F158'``).  Required when ``magnitude_max`` is not
        ``None``.
    property_cuts : dict or None
        Extra cuts expressed as a mapping::

            {dataset_name: (min_value, max_value)}

        Each entry selects galaxies whose value of
        ``nodeData/<dataset_name>`` satisfies
        ``min_value <= value <= max_value``.  Either bound may be ``None``
        to mean no bound in that direction.

    Returns
    -------
    selected_indices : ndarray of int
        Indices (within the catalog) of galaxies that pass all cuts.

    Raises
    ------
    ValueError
        If ``magnitude_max`` is set but ``magnitude_filter`` is not, or if a
        requested dataset does not exist in the catalog.
    """
    node_data_path = f'{base_path}/nodeData'

    n_galaxies = _get_n_galaxies(galacticus_file, base_path)
    mask = np.ones(n_galaxies, dtype=bool)

    # ------------------------------------------------------------------
    # Redshift cuts
    # ------------------------------------------------------------------
    if redshift_min is not None or redshift_max is not None:
        redshifts = get_redshifts(
            galacticus_file, base_path, format_type, n_galaxies=n_galaxies
        )
        if redshift_min is not None:
            mask &= redshifts >= redshift_min
        if redshift_max is not None:
            mask &= redshifts <= redshift_max

    # ------------------------------------------------------------------
    # Apparent magnitude cut
    # ------------------------------------------------------------------
    if magnitude_max is not None:
        if magnitude_filter is None:
            raise ValueError(
                "magnitude_filter must be specified when magnitude_max is set."
            )
        mag_dataset = (
            f'{node_data_path}/apparentMagnitudeRomanWFI:{magnitude_filter}'
        )
        with h5py.File(galacticus_file, 'r') as f:
            if mag_dataset not in f:
                raise ValueError(
                    f"Magnitude dataset '{mag_dataset}' not found in "
                    f"'{galacticus_file}'.  Run calculate_catalog_magnitudes.py "
                    "first to compute apparent magnitudes."
                )
            magnitudes = f[mag_dataset][:]
        mask &= magnitudes <= magnitude_max

    # ------------------------------------------------------------------
    # Custom property cuts
    # ------------------------------------------------------------------
    if property_cuts:
        with h5py.File(galacticus_file, 'r') as f:
            for dataset_name, (min_val, max_val) in property_cuts.items():
                dataset_path = f'{node_data_path}/{dataset_name}'
                if dataset_path not in f:
                    raise ValueError(
                        f"Dataset '{dataset_path}' not found in "
                        f"'{galacticus_file}'."
                    )
                values = f[dataset_path][:]
                if min_val is not None:
                    mask &= values >= min_val
                if max_val is not None:
                    mask &= values <= max_val

    return np.where(mask)[0]


def copy_galaxy_data(input_file, output_file, selected_indices, base_path):
    """
    Create a new HDF5 catalog containing only the selected galaxies.

    All datasets inside the ``nodeData`` group are sliced to the selected
    rows.  The ``Parameters`` group and all attributes on the parent groups
    along the path to ``nodeData`` are copied verbatim.  Provenance
    attributes recording the source file and selected indices are added to
    the output group.

    Parameters
    ----------
    input_file : str
        Path to the source Galacticus HDF5 file.
    output_file : str
        Path for the new HDF5 file to create.  The file must not already
        exist.
    selected_indices : array-like of int
        Indices (in the source file) of the galaxies to copy.
    base_path : str
        HDF5 path to the group that owns ``nodeData``
        (e.g. ``'/Lightcone/Output1'`` or ``'/Outputs/Output1'``).
    """
    selected_indices = np.asarray(selected_indices)

    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dst:

        # Copy the Parameters group verbatim (contains cosmology, model config)
        if 'Parameters' in src:
            src.copy('Parameters', dst)

        # Recreate the parent group hierarchy, preserving attributes at each
        # level (e.g. 'outputTime' on the Output group).
        parts = [p for p in base_path.strip('/').split('/') if p]
        current_path = ''
        for part in parts:
            current_path = f'{current_path}/{part}'
            src_group = src[current_path]
            dst_group = dst.require_group(current_path)
            for key, val in src_group.attrs.items():
                dst_group.attrs[key] = val

        # Slice all nodeData datasets to the selected rows
        node_data_path = f'{base_path}/nodeData'
        src_nd = src[node_data_path]
        dst_nd = dst.require_group(node_data_path)

        for ds_name in src_nd.keys():
            src_ds = src_nd[ds_name]
            data = src_ds[:]

            # Slice along the first (galaxy) axis; handle 1-D and N-D arrays
            new_data = data[selected_indices]

            dst_ds = dst_nd.create_dataset(
                ds_name, data=new_data,
                compression='gzip', compression_opts=4,
            )
            # Preserve all dataset attributes (units, comments, etc.)
            for key, val in src_ds.attrs.items():
                dst_ds.attrs[key] = val

        # Record provenance on the output group
        dst[base_path].attrs['downsampledFrom'] = input_file.encode('utf-8')
        dst[base_path].attrs['nSelectedGalaxies'] = len(selected_indices)
        dst[base_path].attrs['selectedIndices'] = selected_indices


def calculate_and_save_seds(input_file, output_file, selected_indices, base_path,
                             sed_template_file, obs_wavelengths,
                             component='total', include_emission_lines=True,
                             dust_model=None, dust_params=None,
                             dust_law='calzetti', random_uniform_index=None,
                             cosmology=None):
    """
    Calculate SEDs for selected galaxies and save them to the output file.

    SEDs are evaluated from the *input* catalog (using the original galaxy
    indices) and written into the *output* catalog.  The output file is
    expected to have already been populated with galaxy data by
    :func:`copy_galaxy_data`.

    Parameters
    ----------
    input_file : str
        Path to the source Galacticus HDF5 file (used by the SED calculator).
    output_file : str
        Path to the output HDF5 file (where SEDs will be written).
    selected_indices : array-like of int
        Original (input-file) indices of the galaxies for which SEDs are
        computed.
    base_path : str
        HDF5 base path (e.g. ``'/Lightcone/Output1'``).
    sed_template_file : str
        Path to the SED template HDF5 file for :class:`SEDCalculator`.
    obs_wavelengths : Quantity
        Wavelength grid on which to evaluate the spectra (astropy Quantity,
        assumed to be in Angstroms).
    component : str, optional
        Galaxy component: ``'total'`` (default) combines disk, spheroid, and
        AGN; ``'disk'`` or ``'spheroid'`` select individual components.
    include_emission_lines : bool, optional
        Whether to include emission lines in the SED.  Default ``True``.
    dust_model : str or None, optional
        Dust attenuation model name passed to :class:`SEDCalculator`.
    dust_params : dict or None, optional
        Parameters for the dust model.
    dust_law : str, optional
        Attenuation law.  Default ``'calzetti'``.
    random_uniform_index : int or None, optional
        Column index into ``nodeData/randomUniform`` for reproducible scatter.
    cosmology : astropy.cosmology or None, optional
        Cosmology object.  Uses the UNIT cosmology by default.

    Returns
    -------
    seds : ndarray, shape (n_selected, n_wavelengths)
        Flux-density values in units of erg/(s cm² Hz).
    wavelengths_AA : ndarray, shape (n_wavelengths,)
        Wavelength grid in Angstroms.
    """
    selected_indices = np.asarray(selected_indices)
    n_selected = len(selected_indices)

    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    calc = SEDCalculator(sed_template_file, cosmology=cosmology)

    wav_AA = obs_wavelengths.to_value(u.AA)
    n_wav = len(wav_AA)

    sed_array = np.full((n_selected, n_wav), np.nan)

    start_time = time.time()
    print(f"\nCalculating SEDs for {n_selected} galaxies...")
    print("Progress: ", end='', flush=True)

    for out_idx, orig_idx in enumerate(selected_indices):
        if (out_idx + 1) % max(1, n_selected // 20) == 0:
            progress = (out_idx + 1) / n_selected * 100
            print(f"{progress:.0f}% ", end='', flush=True)

        try:
            if component == 'total':
                spectrum = calc.evaluate_total_spectrum(
                    input_file, int(orig_idx),
                    obs_wavelengths=obs_wavelengths,
                    include_emission_lines=include_emission_lines,
                    use_synphot=False,
                    dust_model=dust_model,
                    dust_params=dust_params,
                    dust_law=dust_law,
                    random_uniform_index=random_uniform_index,
                )
            else:
                spectrum = calc.evaluate_component_spectrum(
                    input_file, int(orig_idx),
                    component=component,
                    obs_wavelengths=obs_wavelengths,
                    include_emission_lines=include_emission_lines,
                    use_synphot=False,
                    dust_model=dust_model,
                    dust_params=dust_params,
                    dust_law=dust_law,
                    random_uniform_index=random_uniform_index,
                )

            # Evaluate on the requested wavelength grid
            fnu = spectrum(obs_wavelengths, flux_unit='FNU').to_value(
                u.erg / (u.s * u.cm**2 * u.Hz)
            )
            sed_array[out_idx] = fnu

        except Exception as e:
            print(
                f"\nWarning: Failed for galaxy orig_idx={orig_idx}: "
                f"{type(e).__name__}: {e}"
            )

    print("Done!")
    elapsed = time.time() - start_time
    print(
        f"Total SED calculation time: {elapsed:.1f} s "
        f"({elapsed / max(n_selected, 1):.2f} s/galaxy)"
    )

    # ------------------------------------------------------------------
    # Write to output file
    # ------------------------------------------------------------------
    node_data_path = f'{base_path}/nodeData'
    with h5py.File(output_file, 'a') as f:
        nd = f.require_group(node_data_path)

        # Wavelength axis (common for all galaxies in this catalog)
        if 'observedSEDWavelengths' in nd:
            del nd['observedSEDWavelengths']
        wav_ds = nd.create_dataset('observedSEDWavelengths', data=wav_AA)
        wav_ds.attrs['units'] = b'Angstroms'
        wav_ds.attrs['description'] = (
            b'Wavelength grid for observed-frame SEDs'
        )

        # SED flux-density array [n_galaxies x n_wavelengths]
        if 'observedSED' in nd:
            del nd['observedSED']
        sed_ds = nd.create_dataset(
            'observedSED', data=sed_array,
            compression='gzip', compression_opts=4,
        )
        sed_ds.attrs['units'] = b'erg/(s cm^2 Hz)'
        sed_ds.attrs['component'] = component.encode('utf-8')
        sed_ds.attrs['include_emission_lines'] = b'True' if include_emission_lines else b'False'
        if dust_model is not None:
            sed_ds.attrs['dust_model'] = dust_model.encode('utf-8')
            sed_ds.attrs['dust_law'] = dust_law.encode('utf-8')
            if dust_params is not None:
                sed_ds.attrs['dust_params'] = json.dumps(dust_params).encode('utf-8')
        sed_ds.attrs['description'] = (
            b'Observed-frame flux-density SED for each galaxy, '
            b'shape (n_galaxies, n_wavelengths). Units: erg/(s cm^2 Hz).'
        )

    print(f"\nSaved SED datasets to {output_file}")
    print(f"  observedSEDWavelengths: shape ({n_wav},)")
    print(f"  observedSED:            shape ({n_selected}, {n_wav})")

    return sed_array, wav_AA


def create_downsampled_catalog(galacticus_catalog, sed_template_file,
                               output_file,
                               redshift_min=None, redshift_max=None,
                               magnitude_max=None, magnitude_filter=None,
                               property_cuts=None,
                               obs_wavelengths=None,
                               component='total',
                               include_emission_lines=True,
                               dust_model=None, dust_params=None,
                               dust_law='calzetti',
                               random_uniform_index=None,
                               cosmology=None):
    """
    Create a downsampled Galacticus catalog with SEDs.

    Applies selection cuts to the input catalog, copies the matching galaxies
    (including all their data and metadata) to a new HDF5 file, and then
    calculates and stores an observed-frame SED for each selected galaxy.

    Parameters
    ----------
    galacticus_catalog : str
        Path to the source Galacticus HDF5 file.
    sed_template_file : str
        Path to the SED template HDF5 file used by :class:`SEDCalculator`.
    output_file : str
        Path for the output HDF5 file.  The file must not already exist.
    redshift_min : float or None, optional
        Minimum redshift cut (inclusive).  ``None`` means no lower bound.
    redshift_max : float or None, optional
        Maximum redshift cut (inclusive).  ``None`` means no upper bound.
    magnitude_max : float or None, optional
        Keep only galaxies with apparent magnitude ≤ ``magnitude_max``
        (i.e. brighter than this limit).  Requires ``magnitude_filter``.
    magnitude_filter : str or None, optional
        Roman WFI filter name for the magnitude cut (e.g. ``'F158'``).
    property_cuts : dict or None, optional
        Extra cuts: ``{dataset_name: (min_value, max_value)}``.
    obs_wavelengths : Quantity or None, optional
        Wavelength grid for SED calculation.  Defaults to
        ``np.linspace(4000, 24000, 2000) * u.AA``.
    component : str, optional
        Galaxy component for SED calculation (``'total'``, ``'disk'``, or
        ``'spheroid'``).  Default ``'total'``.
    include_emission_lines : bool, optional
        Include emission lines in the SED.  Default ``True``.
    dust_model : str or None, optional
        Dust attenuation model name; passed to :class:`SEDCalculator`.
    dust_params : dict or None, optional
        Parameters for the dust model.
    dust_law : str, optional
        Attenuation law.  Default ``'calzetti'``.
    random_uniform_index : int or None, optional
        Column index into ``nodeData/randomUniform`` for reproducible scatter.
    cosmology : astropy.cosmology or None, optional
        Cosmology to use.  Defaults to
        ``FlatLambdaCDM(H0=67.74, Om0=0.3089)``.

    Returns
    -------
    results : dict
        Summary with keys: ``'selected_indices'``, ``'n_selected'``,
        ``'output_file'``, ``'format_type'``, ``'base_path'``,
        ``'n_wavelengths'``.  When no galaxies pass the cuts,
        ``'output_file'`` is ``None`` and ``'n_wavelengths'`` is absent.
    """
    format_type, base_path = detect_galacticus_format(galacticus_catalog)
    print(f"\nDetected format: {format_type}")
    print(f"Base path: {base_path}")

    if obs_wavelengths is None:
        obs_wavelengths = np.linspace(
            DEFAULT_WAVELENGTH_MIN,
            DEFAULT_WAVELENGTH_MAX,
            DEFAULT_WAVELENGTH_NPOINTS,
        ) * u.AA

    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=67.74, Om0=0.3089)

    # ------------------------------------------------------------------
    # Step 1: Filter galaxies
    # ------------------------------------------------------------------
    print("\nApplying selection cuts...")
    selected_indices = filter_galaxies(
        galacticus_catalog, base_path, format_type,
        redshift_min=redshift_min, redshift_max=redshift_max,
        magnitude_max=magnitude_max, magnitude_filter=magnitude_filter,
        property_cuts=property_cuts,
    )
    n_total = _get_n_galaxies(galacticus_catalog, base_path)
    n_selected = len(selected_indices)
    print(f"Selected {n_selected} / {n_total} galaxies")

    if n_selected == 0:
        print(
            "Warning: No galaxies passed the cuts.  Output file will not be "
            "created."
        )
        return {
            'selected_indices': selected_indices,
            'n_selected': 0,
            'output_file': None,
            'format_type': format_type,
            'base_path': base_path,
        }

    # ------------------------------------------------------------------
    # Step 2: Copy galaxy data to output file
    # ------------------------------------------------------------------
    print(f"\nCopying {n_selected} galaxies to {output_file}...")
    copy_galaxy_data(
        galacticus_catalog, output_file, selected_indices, base_path
    )
    print("Copy complete.")

    # ------------------------------------------------------------------
    # Step 3: Calculate and save SEDs
    # ------------------------------------------------------------------
    sed_array, wav_AA = calculate_and_save_seds(
        galacticus_catalog, output_file, selected_indices, base_path,
        sed_template_file=sed_template_file,
        obs_wavelengths=obs_wavelengths,
        component=component,
        include_emission_lines=include_emission_lines,
        dust_model=dust_model,
        dust_params=dust_params,
        dust_law=dust_law,
        random_uniform_index=random_uniform_index,
        cosmology=cosmology,
    )

    return {
        'selected_indices': selected_indices,
        'n_selected': n_selected,
        'output_file': output_file,
        'format_type': format_type,
        'base_path': base_path,
        'n_wavelengths': len(wav_AA),
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_property_cuts(cut_strings):
    """
    Parse property-cut strings of the form ``'name:min:max'``.

    Either bound may be an empty string to indicate no limit in that
    direction.  For example ``'diskMassStellar:1e9:'`` sets only a lower
    bound.

    Parameters
    ----------
    cut_strings : list of str
        Each element should be ``'dataset_name:min_value:max_value'``.

    Returns
    -------
    property_cuts : dict
        ``{dataset_name: (min_val, max_val)}`` where each bound is a
        ``float`` or ``None``.

    Raises
    ------
    ValueError
        If a string does not have exactly two colons.
    """
    property_cuts = {}
    for s in cut_strings:
        parts = s.split(':')
        if len(parts) != 3:
            raise ValueError(
                f"Property cut must be 'name:min:max', got '{s}'."
            )
        name, lo, hi = parts
        lo_val = float(lo) if lo.strip() else None
        hi_val = float(hi) if hi.strip() else None
        property_cuts[name] = (lo_val, hi_val)
    return property_cuts


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            'Create a downsampled Galacticus catalog with SEDs.\n\n'
            'Applies selection cuts to a Galacticus catalog, copies the '
            'matching galaxies to a new HDF5 file, and calculates an '
            'observed-frame SED for each one.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        'catalog',
        help='Path to the input Galacticus HDF5 catalog file.',
    )
    parser.add_argument(
        'sed_template',
        help='Path to the SED template HDF5 file.',
    )
    parser.add_argument(
        'output',
        help='Path for the output HDF5 catalog file.',
    )

    # Selection cuts
    cut_group = parser.add_argument_group('Selection cuts')
    cut_group.add_argument(
        '--redshift-min', type=float, default=None,
        help='Minimum redshift (inclusive).',
    )
    cut_group.add_argument(
        '--redshift-max', type=float, default=None,
        help='Maximum redshift (inclusive).',
    )
    cut_group.add_argument(
        '--magnitude-max', type=float, default=None,
        help=(
            'Keep only galaxies brighter than this apparent magnitude in '
            'the filter specified by --magnitude-filter.'
        ),
    )
    cut_group.add_argument(
        '--magnitude-filter', default=None,
        help=(
            'Roman WFI filter name for the magnitude cut (e.g. F158).  '
            'Required when --magnitude-max is set.'
        ),
    )
    cut_group.add_argument(
        '--property-cut', metavar='NAME:MIN:MAX',
        action='append', dest='property_cuts',
        help=(
            "Cut on a nodeData dataset, e.g. 'diskMassStellar:1e9:1e11'.  "
            "Either bound may be empty, e.g. 'diskMassStellar:1e9:'.  "
            "Can be repeated for multiple cuts."
        ),
    )

    # SED options
    sed_group = parser.add_argument_group('SED options')
    sed_group.add_argument(
        '--component', default='total',
        choices=['total', 'disk', 'spheroid'],
        help='Galaxy component for SED calculation.  Default: total.',
    )
    sed_group.add_argument(
        '--no-emission-lines', action='store_true',
        help='Exclude emission lines from the computed SED.',
    )
    sed_group.add_argument(
        '--wavelength-min', type=float, default=DEFAULT_WAVELENGTH_MIN,
        help='Minimum wavelength of the SED grid (Angstroms).',
    )
    sed_group.add_argument(
        '--wavelength-max', type=float, default=DEFAULT_WAVELENGTH_MAX,
        help='Maximum wavelength of the SED grid (Angstroms).',
    )
    sed_group.add_argument(
        '--wavelength-npoints', type=int, default=DEFAULT_WAVELENGTH_NPOINTS,
        help='Number of points in the wavelength grid.',
    )
    sed_group.add_argument(
        '--dust-config', metavar='DUST_CONFIG',
        help=(
            'Path to a YAML file specifying dust attenuation (same format '
            'as used by calculate_catalog_magnitudes.py).  Required keys: '
            'dust_model, dust_params, dust_law.'
        ),
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    print('=' * 60)
    print('CREATE DOWNSAMPLED GALACTICUS CATALOG WITH SEDs')
    print('=' * 60)
    print(f'\nInput catalog:  {args.catalog}')
    print(f'SED template:   {args.sed_template}')
    print(f'Output catalog: {args.output}')

    # Validate input files
    for path, label in [
        (args.catalog, 'catalog'),
        (args.sed_template, 'SED template'),
    ]:
        if not os.path.exists(path):
            print(f'\nError: {label} file not found: {path}')
            sys.exit(1)

    if os.path.exists(args.output):
        print(f'\nError: Output file already exists: {args.output}')
        print('Remove it first or choose a different output path.')
        sys.exit(1)

    # Parse property cuts
    property_cuts = None
    if args.property_cuts:
        property_cuts = parse_property_cuts(args.property_cuts)

    # Load dust config if provided
    dust_model = None
    dust_params = None
    dust_law = 'calzetti'
    random_uniform_index = None
    if args.dust_config:
        if not os.path.exists(args.dust_config):
            print(f'\nError: Dust config file not found: {args.dust_config}')
            sys.exit(1)
        # Re-use the loader from calculate_catalog_magnitudes
        from calculate_catalog_magnitudes import load_dust_config
        dust_model, dust_params, dust_law, random_uniform_index = (
            load_dust_config(args.dust_config)
        )
        print(f'\nDust model:  {dust_model}')
        print(f'Dust law:    {dust_law}')
        print(f'Dust params: {dust_params}')

    obs_wavelengths = np.linspace(
        args.wavelength_min, args.wavelength_max, args.wavelength_npoints
    ) * u.AA

    results = create_downsampled_catalog(
        galacticus_catalog=args.catalog,
        sed_template_file=args.sed_template,
        output_file=args.output,
        redshift_min=args.redshift_min,
        redshift_max=args.redshift_max,
        magnitude_max=args.magnitude_max,
        magnitude_filter=args.magnitude_filter,
        property_cuts=property_cuts,
        obs_wavelengths=obs_wavelengths,
        component=args.component,
        include_emission_lines=not args.no_emission_lines,
        dust_model=dust_model,
        dust_params=dust_params,
        dust_law=dust_law,
        random_uniform_index=random_uniform_index,
    )

    if results and results['n_selected'] > 0:
        print('\n' + '=' * 60)
        print('DONE!')
        print('=' * 60)
        print(f"\nSelected galaxies: {results['n_selected']}")
        print(f"Output file:       {results['output_file']}")
    else:
        print('\nNo galaxies selected; no output file was created.')


if __name__ == '__main__':
    main()
