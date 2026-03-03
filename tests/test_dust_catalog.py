"""
Tests for dust attenuation support in calculate_catalog_magnitudes.py.

These tests cover:
- load_dust_config: loading a JSON config file for the dust model
- calculate_dust_attenuated_emission_lines: vectorised emission-line attenuation
- save_dust_model_metadata: writing dust model attributes onto an HDF5 group
- save_magnitudes_to_galacticus_file: saving dust-attenuated datasets into the
  dustAttenuatedNodeData group
"""

import json
import os
import sys
import tempfile
import unittest

import h5py
import numpy as np

# Make the scripts/ directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from calculate_catalog_magnitudes import (
    load_dust_config,
    calculate_dust_attenuated_emission_lines,
    save_dust_model_metadata,
    save_magnitudes_to_galacticus_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUST_PARAMS = {
    'delta_0': 0.275,
    'delta_z': -1.614,
    'delta_M': -0.834,
    'delta_Mz': -0.708,
    'attenuation_scatter': 0.0,
}
DUST_MODEL = 'gb10_generalised'
DUST_LAW = 'calzetti'


def _make_minimal_lightcone_hdf5(path, n_gals=5):
    """Create a minimal lightcone-format HDF5 file for testing."""
    rng = np.random.default_rng(42)
    with h5py.File(path, 'w') as f:
        nd = f.require_group('Lightcone/Output1/nodeData')

        nd.create_dataset('lightconeRedshiftObserved',
                          data=rng.uniform(0.5, 2.0, n_gals))
        nd.create_dataset('diskMassStellar',
                          data=rng.uniform(1e9, 1e11, n_gals))
        nd.create_dataset('spheroidMassStellar',
                          data=rng.uniform(0, 5e10, n_gals))

        # Two simple emission line datasets (disk + AGN)
        for comp, wav in [('Disk', 6565), ('AGN', 5008)]:
            name = f'luminosityEmissionLine{comp}:balmerAlpha{wav}'
            nd.create_dataset(name, data=rng.uniform(1e40, 1e42, n_gals))


# ---------------------------------------------------------------------------
# Test: load_dust_config
# ---------------------------------------------------------------------------

class TestLoadDustConfig(unittest.TestCase):
    """Tests for load_dust_config."""

    def _write_config(self, path, content):
        with open(path, 'w') as f:
            json.dump(content, f)

    def test_valid_config_returns_correct_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, 'dust.json')
            self._write_config(cfg_path, {
                'dust_model': DUST_MODEL,
                'dust_params': DUST_PARAMS,
                'dust_law': DUST_LAW,
            })
            dust_model, dust_params, dust_law = load_dust_config(cfg_path)

        self.assertEqual(dust_model, DUST_MODEL)
        self.assertEqual(dust_law, DUST_LAW)
        self.assertEqual(dust_params['delta_0'], 0.275)
        self.assertAlmostEqual(dust_params['delta_z'], -1.614)

    def test_missing_key_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, 'dust.json')
            # dust_law is missing
            self._write_config(cfg_path, {
                'dust_model': DUST_MODEL,
                'dust_params': DUST_PARAMS,
            })
            with self.assertRaises(ValueError) as ctx:
                load_dust_config(cfg_path)
            self.assertIn('dust_law', str(ctx.exception))

    def test_nonexistent_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_dust_config('/tmp/does_not_exist_12345.json')


# ---------------------------------------------------------------------------
# Test: calculate_dust_attenuated_emission_lines
# ---------------------------------------------------------------------------

class TestCalculateDustAttenuatedEmissionLines(unittest.TestCase):
    """Tests for calculate_dust_attenuated_emission_lines."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.hdf5_path = os.path.join(self.tmp_dir, 'catalog.hdf5')
        _make_minimal_lightcone_hdf5(self.hdf5_path, n_gals=5)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_dict_with_same_names_as_intrinsic(self):
        """Returned keys should match the original luminosityEmissionLine* names."""
        result = calculate_dust_attenuated_emission_lines(
            self.hdf5_path,
            base_path='/Lightcone/Output1',
            format_type='lightcone',
            dust_model=DUST_MODEL,
            dust_params=DUST_PARAMS,
            dust_law=DUST_LAW,
        )
        # Keys must be the same as the original dataset names (no prefix added)
        self.assertIn('luminosityEmissionLineDisk:balmerAlpha6565', result)
        self.assertIn('luminosityEmissionLineAGN:balmerAlpha5008', result)
        # Must NOT have the old dustAttenuated* prefix
        for key in result:
            self.assertFalse(key.startswith('dustAttenuated'),
                             f"Key should not have dustAttenuated prefix: {key}")

    def test_attenuated_luminosity_is_less_than_intrinsic(self):
        """Dust attenuation can only reduce luminosity."""
        result = calculate_dust_attenuated_emission_lines(
            self.hdf5_path,
            base_path='/Lightcone/Output1',
            format_type='lightcone',
            dust_model=DUST_MODEL,
            dust_params=DUST_PARAMS,
            dust_law=DUST_LAW,
        )
        with h5py.File(self.hdf5_path, 'r') as f:
            intrinsic = f['Lightcone/Output1/nodeData/'
                          'luminosityEmissionLineDisk:balmerAlpha6565'][:]
        attenuated = result['luminosityEmissionLineDisk:balmerAlpha6565']
        # Dust can only reduce flux: attenuated < intrinsic (strictly, because
        # GB10 gives positive A_Halpha for realistic stellar masses)
        np.testing.assert_array_less(attenuated, intrinsic)

    def test_output_shape_matches_input(self):
        result = calculate_dust_attenuated_emission_lines(
            self.hdf5_path,
            base_path='/Lightcone/Output1',
            format_type='lightcone',
            dust_model=DUST_MODEL,
            dust_params=DUST_PARAMS,
            dust_law=DUST_LAW,
        )
        for arr in result.values():
            self.assertEqual(arr.shape, (5,))

    def test_n_galaxies_limits_output_length(self):
        result = calculate_dust_attenuated_emission_lines(
            self.hdf5_path,
            base_path='/Lightcone/Output1',
            format_type='lightcone',
            dust_model=DUST_MODEL,
            dust_params=DUST_PARAMS,
            dust_law=DUST_LAW,
            n_galaxies=3,
        )
        for arr in result.values():
            self.assertEqual(arr.shape, (3,))

    def test_unsupported_dust_model_raises(self):
        with self.assertRaises(ValueError):
            calculate_dust_attenuated_emission_lines(
                self.hdf5_path,
                base_path='/Lightcone/Output1',
                format_type='lightcone',
                dust_model='unknown_model',
                dust_params=DUST_PARAMS,
                dust_law=DUST_LAW,
            )


# ---------------------------------------------------------------------------
# Test: save_dust_model_metadata
# ---------------------------------------------------------------------------

class TestSaveDustModelMetadata(unittest.TestCase):
    """Tests for save_dust_model_metadata (writes to an h5py.Group)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.hdf5_path = os.path.join(self.tmp_dir, 'catalog.hdf5')
        with h5py.File(self.hdf5_path, 'w') as f:
            f.create_group('TestGroup')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_dust_model_attributes_are_correct(self):
        with h5py.File(self.hdf5_path, 'a') as f:
            grp = f['TestGroup']
            save_dust_model_metadata(grp, DUST_MODEL, DUST_PARAMS, DUST_LAW)
            self.assertEqual(grp.attrs['dust_model'], DUST_MODEL)
            self.assertEqual(grp.attrs['dust_law'], DUST_LAW)

    def test_dust_params_stored_as_json_string(self):
        """dust_params should be stored as a JSON string to preserve nesting."""
        with h5py.File(self.hdf5_path, 'a') as f:
            grp = f['TestGroup']
            save_dust_model_metadata(grp, DUST_MODEL, DUST_PARAMS, DUST_LAW)
            stored = json.loads(grp.attrs['dust_params'])
        self.assertAlmostEqual(stored['delta_0'], 0.275)
        self.assertAlmostEqual(stored['delta_z'], -1.614)

    def test_overwrite_replaces_attributes(self):
        new_params = dict(DUST_PARAMS)
        new_params['delta_0'] = 999.0
        with h5py.File(self.hdf5_path, 'a') as f:
            grp = f['TestGroup']
            save_dust_model_metadata(grp, DUST_MODEL, DUST_PARAMS, DUST_LAW)
            save_dust_model_metadata(grp, DUST_MODEL, new_params, DUST_LAW)
            stored = json.loads(grp.attrs['dust_params'])
        self.assertAlmostEqual(stored['delta_0'], 999.0)


# ---------------------------------------------------------------------------
# Test: save_magnitudes_to_galacticus_file (dust datasets)
# ---------------------------------------------------------------------------

class TestSaveMagnitudesWithDust(unittest.TestCase):
    """Tests that dust-attenuated datasets are written to dustAttenuatedNodeData."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.hdf5_path = os.path.join(self.tmp_dir, 'catalog.hdf5')
        _make_minimal_lightcone_hdf5(self.hdf5_path, n_gals=5)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_results(self, with_dust=True, n_gals=5, filter_names=None):
        if filter_names is None:
            filter_names = ['F062', 'F158']
        n_filters = len(filter_names)
        results = {
            'magnitudes': np.full((n_gals, n_filters), 24.0),
            'filter_names': filter_names,
            'redshifts': np.ones(n_gals),
            'galaxy_indices': np.arange(n_gals),
            'format_type': 'lightcone',
            'base_path': '/Lightcone/Output1',
        }
        if with_dust:
            results['dust_magnitudes'] = np.full((n_gals, n_filters), 25.0)
            results['dust_model'] = DUST_MODEL
            results['dust_params'] = DUST_PARAMS
            results['dust_law'] = DUST_LAW
            # Dust emission line datasets use the SAME names as the dust-free
            # versions (they will be stored in dustAttenuatedNodeData group)
            results['dust_emission_lines'] = {
                'luminosityEmissionLineDisk:balmerAlpha6565':
                    np.full(n_gals, 1e41),
            }
        return results

    def test_dust_magnitude_datasets_in_dust_group(self):
        """Dust-attenuated magnitudes must be in dustAttenuatedNodeData, not nodeData."""
        results = self._make_results(with_dust=True)
        save_magnitudes_to_galacticus_file(
            self.hdf5_path, results, component='total',
            format_type='lightcone', base_path='/Lightcone/Output1',
        )
        with h5py.File(self.hdf5_path, 'r') as f:
            dust_nd = f['Lightcone/Output1/dustAttenuatedNodeData']
            self.assertIn('apparentMagnitudeRomanWFI:F062', dust_nd)
            self.assertIn('apparentMagnitudeRomanWFI:F158', dust_nd)
            # Must NOT exist as dustAttenuated* in the regular nodeData group
            nd = f['Lightcone/Output1/nodeData']
            self.assertNotIn('dustAttenuatedApparentMagnitudeRomanWFI:F062', nd)

    def test_dust_magnitude_values_are_correct(self):
        results = self._make_results(with_dust=True)
        save_magnitudes_to_galacticus_file(
            self.hdf5_path, results, component='total',
            format_type='lightcone', base_path='/Lightcone/Output1',
        )
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f['Lightcone/Output1/dustAttenuatedNodeData/'
                      'apparentMagnitudeRomanWFI:F062'][:]
        np.testing.assert_array_almost_equal(data, np.full(5, 25.0))

    def test_dust_emission_line_in_dust_group(self):
        """Dust-attenuated emission lines must be in dustAttenuatedNodeData."""
        results = self._make_results(with_dust=True)
        save_magnitudes_to_galacticus_file(
            self.hdf5_path, results, component='total',
            format_type='lightcone', base_path='/Lightcone/Output1',
        )
        with h5py.File(self.hdf5_path, 'r') as f:
            dust_nd = f['Lightcone/Output1/dustAttenuatedNodeData']
            self.assertIn(
                'luminosityEmissionLineDisk:balmerAlpha6565', dust_nd
            )

    def test_dust_model_metadata_on_dust_group(self):
        """Dust model attributes must live on dustAttenuatedNodeData, not a top-level DustModel group."""
        results = self._make_results(with_dust=True)
        save_magnitudes_to_galacticus_file(
            self.hdf5_path, results, component='total',
            format_type='lightcone', base_path='/Lightcone/Output1',
        )
        with h5py.File(self.hdf5_path, 'r') as f:
            # No top-level DustModel group
            self.assertNotIn('DustModel', f)
            # Metadata on the dustAttenuatedNodeData group
            grp = f['Lightcone/Output1/dustAttenuatedNodeData']
            self.assertEqual(grp.attrs['dust_model'], DUST_MODEL)
            stored = json.loads(grp.attrs['dust_params'])
            self.assertAlmostEqual(stored['delta_0'], 0.275)

    def test_no_dust_group_when_dust_not_requested(self):
        results = self._make_results(with_dust=False)
        save_magnitudes_to_galacticus_file(
            self.hdf5_path, results, component='total',
            format_type='lightcone', base_path='/Lightcone/Output1',
        )
        with h5py.File(self.hdf5_path, 'r') as f:
            self.assertNotIn('dustAttenuatedNodeData',
                             f['Lightcone/Output1'])
            self.assertNotIn('DustModel', f)


if __name__ == '__main__':
    unittest.main()
