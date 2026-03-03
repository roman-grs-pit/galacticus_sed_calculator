"""
Tests for dust attenuation support in calculate_catalog_magnitudes.py.
"""
import unittest
import tempfile
import os
import sys
import json
import shutil
import h5py
import numpy as np

# Add parent directory to path to import the scripts module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from scripts.calculate_catalog_magnitudes import (
    load_dust_config,
    save_dust_metadata_to_galacticus_file,
    calculate_and_save_dust_attenuated_emission_lines,
    save_magnitudes_to_galacticus_file,
)


GALACTICUS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'romanUNIT.hdf5')

EXAMPLE_DUST_CONFIG = {
    'dust_model': 'gb10_generalised',
    'dust_params': {
        'delta_0': 0.275,
        'delta_z': -1.614,
        'delta_M': -0.834,
        'delta_Mz': -0.708,
        'attenuation_scatter': 0.0,
    },
    'dust_law': 'calzetti',
}


def _make_copy(src):
    """Return a temporary writable copy of *src*."""
    tmp = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
    tmp.close()
    shutil.copy2(src, tmp.name)
    return tmp.name


class TestLoadDustConfig(unittest.TestCase):
    """Tests for load_dust_config()."""

    def _write_config(self, cfg):
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump(cfg, tmp)
        tmp.close()
        return tmp.name

    def test_loads_valid_config(self):
        path = self._write_config(EXAMPLE_DUST_CONFIG)
        try:
            dust_model, dust_params, dust_law = load_dust_config(path)
            self.assertEqual(dust_model, 'gb10_generalised')
            self.assertEqual(dust_law, 'calzetti')
            self.assertAlmostEqual(dust_params['delta_0'], 0.275)
            self.assertAlmostEqual(dust_params['delta_z'], -1.614)
        finally:
            os.unlink(path)

    def test_missing_key_raises(self):
        bad_cfg = {'dust_model': 'gb10_generalised', 'dust_law': 'calzetti'}
        path = self._write_config(bad_cfg)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_dust_config(path)
            self.assertIn('dust_params', str(ctx.exception))
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        import uuid
        non_existent = os.path.join(tempfile.gettempdir(), f'no_such_file_{uuid.uuid4()}.json')
        with self.assertRaises(FileNotFoundError):
            load_dust_config(non_existent)


class TestSaveDustMetadata(unittest.TestCase):
    """Tests for save_dust_metadata_to_galacticus_file()."""

    def setUp(self):
        self.tmp_file = _make_copy(GALACTICUS_FILE)

    def tearDown(self):
        if os.path.exists(self.tmp_file):
            os.unlink(self.tmp_file)

    def test_creates_dust_model_group(self):
        save_dust_metadata_to_galacticus_file(
            self.tmp_file,
            dust_model='gb10_generalised',
            dust_params=EXAMPLE_DUST_CONFIG['dust_params'],
            dust_law='calzetti',
        )
        with h5py.File(self.tmp_file, 'r') as f:
            self.assertIn('DustModel', f)
            grp = f['DustModel']
            # h5py may return str or bytes depending on version/file
            dust_model_attr = grp.attrs['dust_model']
            if isinstance(dust_model_attr, bytes):
                dust_model_attr = dust_model_attr.decode()
            self.assertEqual(dust_model_attr, 'gb10_generalised')
            dust_law_attr = grp.attrs['dust_law']
            if isinstance(dust_law_attr, bytes):
                dust_law_attr = dust_law_attr.decode()
            self.assertEqual(dust_law_attr, 'calzetti')
            self.assertAlmostEqual(grp.attrs['delta_0'], 0.275, places=6)
            self.assertAlmostEqual(grp.attrs['delta_z'], -1.614, places=6)

    def test_overwrites_existing_dust_model_group(self):
        # Write once
        save_dust_metadata_to_galacticus_file(
            self.tmp_file, 'gb10_generalised', {'delta_0': 0.1}, 'calzetti'
        )
        # Write again with different params
        save_dust_metadata_to_galacticus_file(
            self.tmp_file, 'gb10_generalised', {'delta_0': 0.5}, 'calzetti'
        )
        with h5py.File(self.tmp_file, 'r') as f:
            self.assertAlmostEqual(f['DustModel'].attrs['delta_0'], 0.5, places=6)


class TestCalculateAndSaveDustAttenuatedEmissionLines(unittest.TestCase):
    """Tests for calculate_and_save_dust_attenuated_emission_lines()."""

    def setUp(self):
        self.tmp_file = _make_copy(GALACTICUS_FILE)

    def tearDown(self):
        if os.path.exists(self.tmp_file):
            os.unlink(self.tmp_file)

    def _original_emission_keys(self):
        with h5py.File(self.tmp_file, 'r') as f:
            return [
                k for k in f['Lightcone/Output1/nodeData'].keys()
                if k.startswith('luminosityEmissionLine')
            ]

    def test_creates_dust_attenuated_datasets(self):
        calculate_and_save_dust_attenuated_emission_lines(
            self.tmp_file,
            dust_model='gb10_generalised',
            dust_params=EXAMPLE_DUST_CONFIG['dust_params'],
            dust_law='calzetti',
        )
        orig_keys = self._original_emission_keys()
        with h5py.File(self.tmp_file, 'r') as f:
            nd = f['Lightcone/Output1/nodeData']
            for key in orig_keys:
                new_key = 'dustAttenuated' + key[0].upper() + key[1:]
                self.assertIn(new_key, nd, f"Missing dust-attenuated dataset: {new_key}")

    def test_attenuated_luminosities_leq_original(self):
        """Dust attenuation should reduce luminosities (or leave them equal)."""
        calculate_and_save_dust_attenuated_emission_lines(
            self.tmp_file,
            dust_model='gb10_generalised',
            dust_params=EXAMPLE_DUST_CONFIG['dust_params'],
            dust_law='calzetti',
        )
        with h5py.File(self.tmp_file, 'r') as f:
            nd = f['Lightcone/Output1/nodeData']
            for key in nd.keys():
                if not key.startswith('dustAttenuatedLuminosityEmissionLine'):
                    continue
                orig_key = key[len('dustAttenuated'):]
                orig_key = orig_key[0].lower() + orig_key[1:]
                orig = nd[orig_key][:]
                atten = nd[key][:]
                # Every attenuated value should be <= the original
                self.assertTrue(
                    np.all(atten <= orig + 1e-30),
                    f"{key}: some attenuated values exceed original",
                )

    def test_dataset_attributes_set(self):
        calculate_and_save_dust_attenuated_emission_lines(
            self.tmp_file,
            dust_model='gb10_generalised',
            dust_params=EXAMPLE_DUST_CONFIG['dust_params'],
            dust_law='calzetti',
        )
        with h5py.File(self.tmp_file, 'r') as f:
            nd = f['Lightcone/Output1/nodeData']
            # Find one dust-attenuated key
            dust_keys = [k for k in nd.keys() if k.startswith('dustAttenuatedLuminosityEmissionLine')]
            self.assertTrue(len(dust_keys) > 0)
            ds = nd[dust_keys[0]]
            self.assertIn('dust_model', ds.attrs)
            self.assertIn('dust_law', ds.attrs)

    def test_max_galaxies_respected(self):
        """Only the first N galaxies should be written when max_galaxies is set."""
        max_g = 2
        calculate_and_save_dust_attenuated_emission_lines(
            self.tmp_file,
            dust_model='gb10_generalised',
            dust_params=EXAMPLE_DUST_CONFIG['dust_params'],
            dust_law='calzetti',
            max_galaxies=max_g,
        )
        with h5py.File(self.tmp_file, 'r') as f:
            nd = f['Lightcone/Output1/nodeData']
            dust_keys = [k for k in nd.keys() if k.startswith('dustAttenuatedLuminosityEmissionLine')]
            self.assertTrue(len(dust_keys) > 0)
            self.assertEqual(len(nd[dust_keys[0]]), max_g)


class TestSaveMagnitudesToGalacticusFile(unittest.TestCase):
    """Tests for save_magnitudes_to_galacticus_file() dust prefix behaviour."""

    def setUp(self):
        self.tmp_file = _make_copy(GALACTICUS_FILE)

    def tearDown(self):
        if os.path.exists(self.tmp_file):
            os.unlink(self.tmp_file)

    def _make_results(self, prefix, n=5, filters=('F062', 'F158')):
        return {
            'magnitudes': np.full((n, len(filters)), 22.5),
            'filter_names': list(filters),
            'galaxy_indices': np.arange(n),
            'magnitude_prefix': prefix,
            'dust_model': None if prefix == 'apparentMagnitudeRomanWFI' else 'gb10_generalised',
        }

    def test_saves_with_dust_free_prefix(self):
        results = self._make_results('apparentMagnitudeRomanWFI')
        save_magnitudes_to_galacticus_file(self.tmp_file, results)
        with h5py.File(self.tmp_file, 'r') as f:
            nd = f['Lightcone/Output1/nodeData']
            self.assertIn('apparentMagnitudeRomanWFI:F062', nd)
            self.assertIn('apparentMagnitudeRomanWFI:F158', nd)

    def test_saves_with_dust_attenuated_prefix(self):
        results = self._make_results('dustAttenuatedApparentMagnitudeRomanWFI')
        save_magnitudes_to_galacticus_file(self.tmp_file, results)
        with h5py.File(self.tmp_file, 'r') as f:
            nd = f['Lightcone/Output1/nodeData']
            self.assertIn('dustAttenuatedApparentMagnitudeRomanWFI:F062', nd)
            self.assertIn('dustAttenuatedApparentMagnitudeRomanWFI:F158', nd)
            # Dust-free key should NOT be present
            self.assertNotIn('apparentMagnitudeRomanWFI:F062', nd)


if __name__ == '__main__':
    unittest.main()
