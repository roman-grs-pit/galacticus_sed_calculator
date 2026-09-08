"""Regression tests for the catalog magnitude command."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import h5py
import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import calculate_catalog_magnitudes as magnitude_command


def _make_catalog(path, n_galaxies=2, include_cosmology=True):
    with h5py.File(path, 'w') as handle:
        node_data = handle.require_group('Lightcone/Output1/nodeData')
        node_data.create_dataset(
            'lightconeRedshiftObserved',
            data=np.linspace(0.5, 1.0, n_galaxies),
        )
        if include_cosmology:
            cosmology = handle.require_group('Parameters/cosmologyParameters')
            cosmology.attrs['HubbleConstant'] = 67.74
            cosmology.attrs['OmegaMatter'] = 0.3089
            cosmology.attrs['OmegaDarkEnergy'] = 0.6911
            cosmology.attrs['OmegaBaryon'] = 0.0462
            cosmology.attrs['temperatureCMB'] = 2.72548


class _SuccessfulCalculator:
    magnitude_systems = []

    def __init__(self, _template, cosmology):
        self.cosmo = cosmology

    def calculate_magnitudes(
        self,
        _catalog,
        *,
        bandpasses,
        magnitude_system,
        **_kwargs,
    ):
        self.magnitude_systems.append(magnitude_system)
        return {name: 23.0 for name in bandpasses}


class _FailingCalculator:
    def __init__(self, _template, cosmology):
        self.cosmo = cosmology

    def calculate_magnitudes(self, *_args, **_kwargs):
        raise ValueError('incompatible SED template')


class _NanCalculator:
    def __init__(self, _template, cosmology):
        self.cosmo = cosmology

    def calculate_magnitudes(self, _catalog, *, bandpasses, **_kwargs):
        return {name: np.nan for name in bandpasses}


class TestCatalogMagnitudeCommand(unittest.TestCase):
    def setUp(self):
        _SuccessfulCalculator.magnitude_systems = []

    def test_reads_cosmology_from_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = os.path.join(tmp, 'catalog.hdf5')
            _make_catalog(catalog)
            cosmology = magnitude_command.load_cosmology_from_catalog(catalog)

        self.assertAlmostEqual(cosmology.H0.value, 67.74)
        self.assertAlmostEqual(cosmology.Om0, 0.3089)
        self.assertAlmostEqual(cosmology.Ode0, 0.6911)
        self.assertAlmostEqual(cosmology.Ob0, 0.0462)

    def test_missing_cosmology_has_explanatory_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = os.path.join(tmp, 'catalog.hdf5')
            _make_catalog(catalog, include_cosmology=False)
            with self.assertRaisesRegex(ValueError, 'Cosmology metadata not found'):
                magnitude_command.load_cosmology_from_catalog(catalog)

    def test_magnitude_system_reaches_calculator_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = os.path.join(tmp, 'catalog.hdf5')
            output = os.path.join(tmp, 'magnitudes.hdf5')
            _make_catalog(catalog)

            with patch.object(
                magnitude_command, 'SEDCalculator', _SuccessfulCalculator
            ):
                results = magnitude_command.calculate_catalog_magnitudes(
                    'template.hdf5',
                    catalog,
                    {'F158': object()},
                    output_file=output,
                    magnitude_system='ST',
                    obs_wavelengths=np.linspace(10000, 20000, 10) * u.AA,
                )

            self.assertEqual(_SuccessfulCalculator.magnitude_systems, ['ST', 'ST'])
            self.assertEqual(results['magnitude_system'], 'ST')
            with h5py.File(output, 'r') as handle:
                self.assertEqual(handle['magnitudes'].attrs['magnitude_system'], 'ST')
                self.assertIn('ST magnitudes', handle['magnitudes'].attrs['description'])
                self.assertEqual(
                    handle['magnitudes'].attrs['n_calculation_errors'], 0
                )
                self.assertAlmostEqual(
                    handle['cosmology'].attrs['HubbleConstant'], 67.74
                )
                self.assertEqual(handle['calculation_error_indices'].shape, (0,))

    def test_parallel_worker_uses_requested_magnitude_system(self):
        calculator = _SuccessfulCalculator(
            'template.hdf5',
            FlatLambdaCDM(H0=70, Om0=0.3),
        )
        state = {
            'calc': calculator,
            'bandpasses': {'F158': object()},
            'magnitude_system': 'Vega',
            'dust_model': None,
            'continuum_dust': None,
        }
        with patch.dict(magnitude_command._worker_state, state, clear=True):
            _, result = magnitude_command._process_galaxy_worker(
                (0, 'catalog.hdf5', 'total', np.linspace(1, 2, 2) * u.AA)
            )

        self.assertEqual(result['mags']['F158'], 23.0)
        self.assertEqual(_SuccessfulCalculator.magnitude_systems, ['Vega'])

    def test_unavailable_vega_reference_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = os.path.join(tmp, 'catalog.hdf5')
            output = os.path.join(tmp, 'magnitudes.hdf5')
            _make_catalog(catalog)

            with patch(
                'synphot.SourceSpectrum.from_vega',
                side_effect=OSError('reference unavailable'),
            ):
                with self.assertRaisesRegex(RuntimeError, 'Vega reference'):
                    magnitude_command.calculate_catalog_magnitudes(
                        'template.hdf5',
                        catalog,
                        {'F158': object()},
                        output_file=output,
                        magnitude_system='Vega',
                    )

            self.assertFalse(os.path.exists(output))

    def test_complete_processing_failure_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = os.path.join(tmp, 'catalog.hdf5')
            output = os.path.join(tmp, 'magnitudes.hdf5')
            _make_catalog(catalog)

            with patch.object(magnitude_command, 'SEDCalculator', _FailingCalculator):
                with self.assertRaisesRegex(RuntimeError, 'failed for every galaxy'):
                    magnitude_command.calculate_catalog_magnitudes(
                        'template.hdf5',
                        catalog,
                        {'F158': object()},
                        output_file=output,
                        obs_wavelengths=np.linspace(10000, 20000, 10) * u.AA,
                    )

            self.assertFalse(os.path.exists(output))

    def test_all_nan_result_is_preserved_without_processing_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = os.path.join(tmp, 'catalog.hdf5')
            output = os.path.join(tmp, 'magnitudes.hdf5')
            _make_catalog(catalog)

            with patch.object(magnitude_command, 'SEDCalculator', _NanCalculator):
                results = magnitude_command.calculate_catalog_magnitudes(
                    'template.hdf5',
                    catalog,
                    {'F158': object()},
                    output_file=output,
                    obs_wavelengths=np.linspace(10000, 20000, 10) * u.AA,
                )

            self.assertTrue(os.path.exists(output))
            self.assertTrue(np.isnan(results['magnitudes']).all())
            self.assertEqual(results['calculation_error_indices'].size, 0)

    def test_subset_cannot_be_written_into_full_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = os.path.join(tmp, 'catalog.hdf5')
            copied_catalog = os.path.join(tmp, 'catalog_with_magnitudes.hdf5')
            _make_catalog(catalog, n_galaxies=2)

            with self.assertRaisesRegex(ValueError, 'Use --save-to-file'):
                magnitude_command.calculate_catalog_magnitudes(
                    'template.hdf5',
                    catalog,
                    {'F158': object()},
                    max_galaxies=1,
                    save_to_input=True,
                )

            self.assertFalse(os.path.exists(copied_catalog))


if __name__ == '__main__':
    unittest.main()
