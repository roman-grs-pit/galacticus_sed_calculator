"""
Tests for create_downsampled_catalog.py.

Covers:
- filter_galaxies: redshift, magnitude, and property cuts
- copy_galaxy_data: correct slicing, attribute preservation, provenance
- parse_property_cuts: CLI helper parsing
- _get_n_galaxies: utility
- calculate_and_save_seds: SED output shape and dataset creation
- create_downsampled_catalog: integration smoke-test (no real SED template)
"""

import os
import shutil
import sys
import tempfile
import unittest

import astropy.units as u
import h5py
import numpy as np

# Make the scripts/ directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from create_downsampled_catalog import (
    filter_galaxies,
    copy_galaxy_data,
    parse_property_cuts,
    _get_n_galaxies,
    get_redshifts,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_lightcone_catalog(path, n_gals=20, seed=0):
    """
    Create a minimal lightcone-format HDF5 file for testing.

    Returns
    -------
    redshifts : ndarray
    mags_f158 : ndarray
    disk_masses : ndarray
    """
    rng = np.random.default_rng(seed)
    redshifts = rng.uniform(0.1, 2.0, n_gals)
    mags_f158 = rng.uniform(20.0, 28.0, n_gals)
    disk_masses = rng.uniform(1e9, 1e11, n_gals)
    spheroid_masses = rng.uniform(0.0, 5e10, n_gals)

    with h5py.File(path, 'w') as f:
        params = f.require_group('Parameters')
        params.attrs['version'] = b'test'
        params.attrs['cosmology'] = b'FlatLambdaCDM'

        nd = f.require_group('Lightcone/Output1/nodeData')
        f['Lightcone/Output1'].attrs['description'] = b'test lightcone'

        nd.create_dataset('lightconeRedshiftObserved', data=redshifts)
        nd.create_dataset('diskMassStellar', data=disk_masses)
        nd.create_dataset('spheroidMassStellar', data=spheroid_masses)
        ds = nd.create_dataset(
            'apparentMagnitudeRomanWFI:F158', data=mags_f158
        )
        ds.attrs['filter'] = b'F158'
        ds.attrs['comment'] = b'AB magnitude'

    return redshifts, mags_f158, disk_masses


def _make_fixed_time_catalog(path, n_gals=10, output_time=7.0, seed=1):
    """Create a minimal fixed-time-format HDF5 file for testing."""
    rng = np.random.default_rng(seed)
    disk_masses = rng.uniform(1e9, 1e11, n_gals)

    with h5py.File(path, 'w') as f:
        params = f.require_group('Parameters')
        params.attrs['version'] = b'test'

        nd = f.require_group('Outputs/Output1/nodeData')
        f['Outputs/Output1'].attrs['outputTime'] = output_time

        nd.create_dataset('diskMassStellar', data=disk_masses)
        nd.create_dataset('spheroidMassStellar',
                          data=rng.uniform(0, 5e10, n_gals))

    return disk_masses


# ---------------------------------------------------------------------------
# Tests for _get_n_galaxies
# ---------------------------------------------------------------------------

class TestGetNGalaxies(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.catalog = os.path.join(self.tmp_dir, 'catalog.hdf5')
        _make_lightcone_catalog(self.catalog, n_gals=15)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_correct_count(self):
        n = _get_n_galaxies(self.catalog, '/Lightcone/Output1')
        self.assertEqual(n, 15)


# ---------------------------------------------------------------------------
# Tests for get_redshifts
# ---------------------------------------------------------------------------

class TestGetRedshifts(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.lc_catalog = os.path.join(self.tmp_dir, 'lc.hdf5')
        self.ft_catalog = os.path.join(self.tmp_dir, 'ft.hdf5')
        self.redshifts, _, _ = _make_lightcone_catalog(
            self.lc_catalog, n_gals=10
        )
        _make_fixed_time_catalog(self.ft_catalog, n_gals=8, output_time=7.0)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_lightcone_returns_per_galaxy_redshifts(self):
        z = get_redshifts(
            self.lc_catalog, '/Lightcone/Output1', 'lightcone'
        )
        self.assertEqual(z.shape, (10,))
        np.testing.assert_array_equal(z, self.redshifts)

    def test_fixed_time_returns_constant_array(self):
        z = get_redshifts(
            self.ft_catalog, '/Outputs/Output1', 'fixed-time', n_galaxies=8
        )
        self.assertEqual(z.shape, (8,))
        # All values must be the same (same output time)
        self.assertEqual(len(np.unique(z)), 1)
        # Redshift should be > 0 for a reasonable output time
        self.assertGreater(z[0], 0.0)


# ---------------------------------------------------------------------------
# Tests for filter_galaxies
# ---------------------------------------------------------------------------

class TestFilterGalaxies(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.catalog = os.path.join(self.tmp_dir, 'catalog.hdf5')
        self.redshifts, self.mags, self.masses = _make_lightcone_catalog(
            self.catalog, n_gals=30, seed=42
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_cuts_returns_all_galaxies(self):
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone'
        )
        self.assertEqual(len(idx), 30)

    def test_redshift_min_cut(self):
        z_min = 1.0
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
            redshift_min=z_min,
        )
        self.assertGreater(len(idx), 0)
        self.assertTrue(np.all(self.redshifts[idx] >= z_min))

    def test_redshift_max_cut(self):
        z_max = 1.0
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
            redshift_max=z_max,
        )
        self.assertGreater(len(idx), 0)
        self.assertTrue(np.all(self.redshifts[idx] <= z_max))

    def test_redshift_range_cut(self):
        z_min, z_max = 0.5, 1.5
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
            redshift_min=z_min, redshift_max=z_max,
        )
        self.assertTrue(np.all(self.redshifts[idx] >= z_min))
        self.assertTrue(np.all(self.redshifts[idx] <= z_max))

    def test_magnitude_max_cut(self):
        mag_max = 25.0
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
            magnitude_max=mag_max, magnitude_filter='F158',
        )
        self.assertGreater(len(idx), 0)
        self.assertTrue(np.all(self.mags[idx] <= mag_max))

    def test_magnitude_max_without_filter_raises(self):
        with self.assertRaises(ValueError):
            filter_galaxies(
                self.catalog, '/Lightcone/Output1', 'lightcone',
                magnitude_max=25.0, magnitude_filter=None,
            )

    def test_missing_magnitude_dataset_raises(self):
        with self.assertRaises(ValueError):
            filter_galaxies(
                self.catalog, '/Lightcone/Output1', 'lightcone',
                magnitude_max=25.0, magnitude_filter='F184',
            )

    def test_property_cut_lower_bound(self):
        mass_min = 5e9
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
            property_cuts={'diskMassStellar': (mass_min, None)},
        )
        self.assertTrue(np.all(self.masses[idx] >= mass_min))

    def test_property_cut_both_bounds(self):
        mass_min, mass_max = 5e9, 5e10
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
            property_cuts={'diskMassStellar': (mass_min, mass_max)},
        )
        self.assertTrue(np.all(self.masses[idx] >= mass_min))
        self.assertTrue(np.all(self.masses[idx] <= mass_max))

    def test_combined_redshift_and_magnitude_cuts(self):
        z_max = 1.5
        mag_max = 26.0
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
            redshift_max=z_max,
            magnitude_max=mag_max, magnitude_filter='F158',
        )
        self.assertTrue(np.all(self.redshifts[idx] <= z_max))
        self.assertTrue(np.all(self.mags[idx] <= mag_max))

    def test_impossible_cut_returns_empty(self):
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
            redshift_min=999.0,
        )
        self.assertEqual(len(idx), 0)

    def test_returns_ndarray(self):
        idx = filter_galaxies(
            self.catalog, '/Lightcone/Output1', 'lightcone',
        )
        self.assertIsInstance(idx, np.ndarray)

    def test_property_cut_missing_dataset_raises(self):
        with self.assertRaises(ValueError):
            filter_galaxies(
                self.catalog, '/Lightcone/Output1', 'lightcone',
                property_cuts={'nonExistentDataset': (0, 1)},
            )


# ---------------------------------------------------------------------------
# Tests for copy_galaxy_data
# ---------------------------------------------------------------------------

class TestCopyGalaxyData(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp_dir, 'src.hdf5')
        self.dst = os.path.join(self.tmp_dir, 'dst.hdf5')
        self.redshifts, self.mags, self.masses = _make_lightcone_catalog(
            self.src, n_gals=10, seed=7
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_correct_number_of_galaxies_copied(self):
        selected = np.array([0, 2, 4, 6])
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            n = len(f['Lightcone/Output1/nodeData/diskMassStellar'][:])
        self.assertEqual(n, 4)

    def test_copied_values_match_source(self):
        selected = np.array([1, 3, 5])
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.src, 'r') as src:
            src_mass = src['Lightcone/Output1/nodeData/diskMassStellar'][:][selected]
        with h5py.File(self.dst, 'r') as dst:
            dst_mass = dst['Lightcone/Output1/nodeData/diskMassStellar'][:]
        np.testing.assert_array_equal(src_mass, dst_mass)

    def test_all_nodedata_datasets_present(self):
        selected = np.arange(5)
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.src, 'r') as src:
            src_keys = set(src['Lightcone/Output1/nodeData'].keys())
        with h5py.File(self.dst, 'r') as dst:
            dst_keys = set(dst['Lightcone/Output1/nodeData'].keys())
        self.assertEqual(src_keys, dst_keys)

    def test_parameters_group_is_copied(self):
        selected = np.arange(3)
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            self.assertIn('Parameters', f)

    def test_parameters_attributes_preserved(self):
        selected = np.arange(3)
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            version = f['Parameters'].attrs['version']
            # h5py may return str or bytes depending on version
            if isinstance(version, bytes):
                version = version.decode('utf-8')
            self.assertEqual(version, 'test')

    def test_group_attributes_preserved(self):
        selected = np.array([0, 1])
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            self.assertIn('description', f['Lightcone/Output1'].attrs)

    def test_dataset_attributes_preserved(self):
        selected = np.array([0, 1])
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            attr = f[
                'Lightcone/Output1/nodeData/apparentMagnitudeRomanWFI:F158'
            ].attrs.get('filter')
        # h5py may return str or bytes depending on version
        if isinstance(attr, bytes):
            attr = attr.decode('utf-8')
        self.assertEqual(attr, 'F158')

    def test_provenance_attributes_added(self):
        selected = np.array([0, 1, 2])
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            attrs = f['Lightcone/Output1'].attrs
            self.assertIn('downsampledFrom', attrs)
            self.assertEqual(attrs['nSelectedGalaxies'], 3)
            np.testing.assert_array_equal(attrs['selectedIndices'], selected)

    def test_provenance_downsampledfrom_contains_input_path(self):
        selected = np.array([0])
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            source = f['Lightcone/Output1'].attrs['downsampledFrom']
        # Stored as bytes; decode for comparison
        if isinstance(source, bytes):
            source = source.decode('utf-8')
        self.assertIn(os.path.basename(self.src), source)

    def test_redshift_values_match_after_copy(self):
        selected = np.array([2, 5, 8])
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            dst_z = f[
                'Lightcone/Output1/nodeData/lightconeRedshiftObserved'
            ][:]
        np.testing.assert_array_almost_equal(
            dst_z, self.redshifts[selected]
        )

    def test_copy_single_galaxy(self):
        copy_galaxy_data(self.src, self.dst, [4], '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            n = len(f['Lightcone/Output1/nodeData/diskMassStellar'][:])
        self.assertEqual(n, 1)

    def test_copy_all_galaxies(self):
        selected = np.arange(10)
        copy_galaxy_data(self.src, self.dst, selected, '/Lightcone/Output1')
        with h5py.File(self.dst, 'r') as f:
            n = len(f['Lightcone/Output1/nodeData/diskMassStellar'][:])
        self.assertEqual(n, 10)


# ---------------------------------------------------------------------------
# Tests for parse_property_cuts
# ---------------------------------------------------------------------------

class TestParsePropertyCuts(unittest.TestCase):

    def test_both_bounds(self):
        cuts = parse_property_cuts(['diskMassStellar:1e9:1e11'])
        self.assertIn('diskMassStellar', cuts)
        lo, hi = cuts['diskMassStellar']
        self.assertAlmostEqual(lo, 1e9)
        self.assertAlmostEqual(hi, 1e11)

    def test_lower_bound_only(self):
        cuts = parse_property_cuts(['diskMassStellar:1e9:'])
        lo, hi = cuts['diskMassStellar']
        self.assertAlmostEqual(lo, 1e9)
        self.assertIsNone(hi)

    def test_upper_bound_only(self):
        cuts = parse_property_cuts(['diskMassStellar::1e11'])
        lo, hi = cuts['diskMassStellar']
        self.assertIsNone(lo)
        self.assertAlmostEqual(hi, 1e11)

    def test_both_bounds_none(self):
        cuts = parse_property_cuts(['diskMassStellar::'])
        lo, hi = cuts['diskMassStellar']
        self.assertIsNone(lo)
        self.assertIsNone(hi)

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            parse_property_cuts(['diskMassStellar:1e9'])  # only one colon

    def test_multiple_cuts(self):
        cuts = parse_property_cuts([
            'diskMassStellar:1e9:1e11',
            'spheroidMassStellar::5e10',
        ])
        self.assertEqual(len(cuts), 2)
        self.assertIn('diskMassStellar', cuts)
        self.assertIn('spheroidMassStellar', cuts)

    def test_dataset_name_with_colon(self):
        """Dataset names like 'apparentMagnitudeRomanWFI:F158' contain colons.
        The format is name:min:max where only the FIRST two colons are
        separators.  This test verifies that extra colons in the name are
        handled gracefully — parse_property_cuts splits on exactly 3 parts."""
        # With 4 parts after split this should raise ValueError
        with self.assertRaises(ValueError):
            parse_property_cuts(['a:b:c:d'])


# ---------------------------------------------------------------------------
# Tests for calculate_and_save_seds  (using a mock SED catalog)
# ---------------------------------------------------------------------------

class TestCalculateAndSaveSeds(unittest.TestCase):
    """
    Test calculate_and_save_seds using the real test data files.

    These tests are skipped when the test data HDF5 files are not present.
    """

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(test_dir)
        cls.sed_template = os.path.join(
            parent_dir,
            'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5',
        )
        cls.galacticus_file = os.path.join(
            parent_dir, 'data/romanUNIT.hdf5'
        )
        cls.has_test_data = (
            os.path.exists(cls.sed_template)
            and os.path.exists(cls.galacticus_file)
        )

    def setUp(self):
        if not self.has_test_data:
            self.skipTest('Test data files not available')
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_sed_array_shape_matches_selection(self):
        """SEDs saved to the output file must have the right shape."""
        from create_downsampled_catalog import calculate_and_save_seds

        output_file = os.path.join(self.tmp_dir, 'seds.hdf5')

        # Copy a tiny subset of the catalog so the output file exists
        from galacticus_sed_calculator.sed_calculator import (
            detect_galacticus_format,
        )
        format_type, base_path = detect_galacticus_format(
            self.galacticus_file
        )
        selected = np.array([0, 1])  # just 2 galaxies for speed

        copy_galaxy_data(
            self.galacticus_file, output_file, selected, base_path
        )

        n_wav = 50
        obs_wav = np.linspace(8000, 14000, n_wav) * u.AA

        seds, wav_AA = calculate_and_save_seds(
            self.galacticus_file, output_file, selected, base_path,
            sed_template_file=self.sed_template,
            obs_wavelengths=obs_wav,
            component='total',
            include_emission_lines=False,  # faster, no need for exact result
        )

        self.assertEqual(seds.shape, (2, n_wav))
        self.assertEqual(wav_AA.shape, (n_wav,))

    def test_sed_datasets_created_in_output(self):
        """observedSED and observedSEDWavelengths must appear in the output."""
        from create_downsampled_catalog import calculate_and_save_seds
        from galacticus_sed_calculator.sed_calculator import (
            detect_galacticus_format,
        )

        output_file = os.path.join(self.tmp_dir, 'seds2.hdf5')
        format_type, base_path = detect_galacticus_format(
            self.galacticus_file
        )
        selected = np.array([0])

        copy_galaxy_data(
            self.galacticus_file, output_file, selected, base_path
        )

        obs_wav = np.linspace(8000, 14000, 30) * u.AA
        calculate_and_save_seds(
            self.galacticus_file, output_file, selected, base_path,
            sed_template_file=self.sed_template,
            obs_wavelengths=obs_wav,
            component='total',
            include_emission_lines=False,
        )

        with h5py.File(output_file, 'r') as f:
            nd = f[f'{base_path}/nodeData']
            self.assertIn('observedSED', nd)
            self.assertIn('observedSEDWavelengths', nd)

    def test_sed_wavelengths_match_input_grid(self):
        """Saved wavelength grid must equal the requested obs_wavelengths."""
        from create_downsampled_catalog import calculate_and_save_seds
        from galacticus_sed_calculator.sed_calculator import (
            detect_galacticus_format,
        )

        output_file = os.path.join(self.tmp_dir, 'seds3.hdf5')
        format_type, base_path = detect_galacticus_format(
            self.galacticus_file
        )
        selected = np.array([0])

        copy_galaxy_data(
            self.galacticus_file, output_file, selected, base_path
        )

        n_wav = 40
        obs_wav = np.linspace(8000, 14000, n_wav) * u.AA
        _, wav_AA = calculate_and_save_seds(
            self.galacticus_file, output_file, selected, base_path,
            sed_template_file=self.sed_template,
            obs_wavelengths=obs_wav,
            component='total',
            include_emission_lines=False,
        )

        with h5py.File(output_file, 'r') as f:
            stored_wav = f[f'{base_path}/nodeData/observedSEDWavelengths'][:]

        np.testing.assert_array_almost_equal(stored_wav, wav_AA)

    def test_sed_attributes_set_correctly(self):
        """Check units and component attributes on the saved dataset."""
        from create_downsampled_catalog import calculate_and_save_seds
        from galacticus_sed_calculator.sed_calculator import (
            detect_galacticus_format,
        )

        output_file = os.path.join(self.tmp_dir, 'seds4.hdf5')
        format_type, base_path = detect_galacticus_format(
            self.galacticus_file
        )
        selected = np.array([0])

        copy_galaxy_data(
            self.galacticus_file, output_file, selected, base_path
        )

        obs_wav = np.linspace(8000, 14000, 20) * u.AA
        calculate_and_save_seds(
            self.galacticus_file, output_file, selected, base_path,
            sed_template_file=self.sed_template,
            obs_wavelengths=obs_wav,
            component='disk',
            include_emission_lines=False,
        )

        with h5py.File(output_file, 'r') as f:
            ds = f[f'{base_path}/nodeData/observedSED']
            units_attr = ds.attrs['units']
            component_attr = ds.attrs['component']
            # h5py may return str or bytes depending on version
            if isinstance(units_attr, bytes):
                units_attr = units_attr.decode('utf-8')
            if isinstance(component_attr, bytes):
                component_attr = component_attr.decode('utf-8')
            self.assertEqual(units_attr, 'erg/(s cm^2 Hz)')
            self.assertEqual(component_attr, 'disk')


# ---------------------------------------------------------------------------
# Integration test for create_downsampled_catalog
# ---------------------------------------------------------------------------

class TestCreateDownsampledCatalogIntegration(unittest.TestCase):
    """Integration test using real test data."""

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(test_dir)
        cls.sed_template = os.path.join(
            parent_dir,
            'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5',
        )
        cls.galacticus_file = os.path.join(
            parent_dir, 'data/romanUNIT.hdf5'
        )
        cls.has_test_data = (
            os.path.exists(cls.sed_template)
            and os.path.exists(cls.galacticus_file)
        )

    def setUp(self):
        if not self.has_test_data:
            self.skipTest('Test data files not available')
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _get_catalog_redshifts(self):
        """Return the redshift array from the real catalog."""
        from galacticus_sed_calculator.sed_calculator import (
            detect_galacticus_format,
        )
        fmt, bp = detect_galacticus_format(self.galacticus_file)
        return get_redshifts(self.galacticus_file, bp, fmt)

    def test_redshift_cut_integration(self):
        """create_downsampled_catalog selects the right galaxies and saves SEDs."""
        from create_downsampled_catalog import create_downsampled_catalog

        output_file = os.path.join(self.tmp_dir, 'out.hdf5')
        all_z = self._get_catalog_redshifts()
        z_min, z_max = 0.1, 0.5

        # Use a narrow wavelength range to keep the test fast
        obs_wav = np.linspace(8000, 14000, 30) * u.AA

        results = create_downsampled_catalog(
            galacticus_catalog=self.galacticus_file,
            sed_template_file=self.sed_template,
            output_file=output_file,
            redshift_min=z_min,
            redshift_max=z_max,
            obs_wavelengths=obs_wav,
            component='total',
            include_emission_lines=False,
        )

        expected_n = int(np.sum((all_z >= z_min) & (all_z <= z_max)))
        self.assertEqual(results['n_selected'], expected_n)
        self.assertTrue(os.path.exists(output_file))

        from galacticus_sed_calculator.sed_calculator import (
            detect_galacticus_format,
        )
        _, base_path = detect_galacticus_format(self.galacticus_file)

        with h5py.File(output_file, 'r') as f:
            nd = f[f'{base_path}/nodeData']
            self.assertIn('observedSED', nd)
            self.assertIn('observedSEDWavelengths', nd)
            sed_shape = nd['observedSED'].shape
            self.assertEqual(sed_shape, (expected_n, 30))

    def test_no_galaxies_pass_cuts(self):
        """When no galaxies pass the cuts the result reports n_selected=0."""
        from create_downsampled_catalog import create_downsampled_catalog

        output_file = os.path.join(self.tmp_dir, 'empty.hdf5')

        results = create_downsampled_catalog(
            galacticus_catalog=self.galacticus_file,
            sed_template_file=self.sed_template,
            output_file=output_file,
            redshift_min=999.0,
            obs_wavelengths=np.linspace(8000, 14000, 10) * u.AA,
        )

        self.assertEqual(results['n_selected'], 0)
        self.assertIsNone(results['output_file'])
        self.assertFalse(os.path.exists(output_file))


if __name__ == '__main__':
    unittest.main()
