"""
Tests for SED calculator compatibility validation.
"""
import unittest
import tempfile
import os
import h5py
import numpy as np
from SEDfromSFH import sed_calculator


class TestSEDTemplateParameterExtraction(unittest.TestCase):
    """Test the extraction of SFH parameters from SED template."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
        self.galacticus_file = 'data/romanUNIT.hdf5'
        self.calc = sed_calculator(self.sed_template_file)
    
    def test_get_sed_template_parameters(self):
        """Test that SED template parameters are correctly extracted."""
        params = self.calc.get_sed_template_parameters()
        
        # Check that all required parameters are present
        required_keys = ['ageMinimum', 'ageMaximum', 'countAges', 
                        'metallicityMinimum', 'metallicityMaximum', 'countMetallicities']
        for key in required_keys:
            self.assertIn(key, params, f"Missing required parameter: {key}")
        
        # Check that counts are integers
        self.assertIsInstance(params['countAges'], (int, np.integer))
        self.assertIsInstance(params['countMetallicities'], (int, np.integer))
        
        # Check that values are positive
        self.assertGreater(params['ageMinimum'], 0)
        self.assertGreater(params['ageMaximum'], params['ageMinimum'])
        self.assertGreater(params['metallicityMinimum'], 0)
        self.assertGreater(params['metallicityMaximum'], params['metallicityMinimum'])
        self.assertGreater(params['countAges'], 0)
        self.assertGreater(params['countMetallicities'], 0)
    
    def test_extracted_parameters_match_expected(self):
        """Test that extracted parameters match expected values for test data."""
        params = self.calc.get_sed_template_parameters()
        
        # These values should match the test data files
        self.assertEqual(params['countAges'], 50)
        self.assertEqual(params['countMetallicities'], 11)
        self.assertAlmostEqual(params['ageMinimum'], 0.001, places=6)
        self.assertAlmostEqual(params['metallicityMinimum'], 0.0001, places=6)
        self.assertAlmostEqual(params['metallicityMaximum'], 10.0, places=6)


class TestSFHCompatibilityValidation(unittest.TestCase):
    """Test the SFH compatibility validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
        self.galacticus_file = 'data/romanUNIT.hdf5'
        self.calc = sed_calculator(self.sed_template_file)
    
    def test_compatible_files_pass_validation(self):
        """Test that compatible SED template and SFH pass validation."""
        # Should not raise an exception
        self.calc.validate_sfh_compatibility(self.galacticus_file)
    
    def test_incompatible_count_ages_fails(self):
        """Test that incompatible countAges is detected."""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create a file with mismatched countAges
            with h5py.File(self.galacticus_file, 'r') as src:
                with h5py.File(tmp_filename, 'w') as dst:
                    src.copy('/Parameters', dst)
                    dst['/Parameters/starFormationHistory'].attrs['countAges'] = 40
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                self.calc.validate_sfh_compatibility(tmp_filename)
            
            self.assertIn('countAges mismatch', str(context.exception))
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_incompatible_count_metallicities_fails(self):
        """Test that incompatible countMetallicities is detected."""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create a file with mismatched countMetallicities
            with h5py.File(self.galacticus_file, 'r') as src:
                with h5py.File(tmp_filename, 'w') as dst:
                    src.copy('/Parameters', dst)
                    dst['/Parameters/starFormationHistory'].attrs['countMetallicities'] = 8
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                self.calc.validate_sfh_compatibility(tmp_filename)
            
            self.assertIn('countMetallicities mismatch', str(context.exception))
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_incompatible_metallicity_bounds_fails(self):
        """Test that incompatible metallicity bounds are detected."""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create a file with mismatched metallicityMaximum
            with h5py.File(self.galacticus_file, 'r') as src:
                with h5py.File(tmp_filename, 'w') as dst:
                    src.copy('/Parameters', dst)
                    dst['/Parameters/starFormationHistory'].attrs['metallicityMaximum'] = 5.0
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                self.calc.validate_sfh_compatibility(tmp_filename)
            
            self.assertIn('metallicityMaximum mismatch', str(context.exception))
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_validation_caching(self):
        """Test that validation results are cached."""
        # First validation
        self.calc.validate_sfh_compatibility(self.galacticus_file)
        self.assertIn(self.galacticus_file, self.calc._validated_files)
        
        # Second validation should use cache (no exception even if we clear the file)
        self.calc.validate_sfh_compatibility(self.galacticus_file)
        
        # Cache should still contain the file
        self.assertIn(self.galacticus_file, self.calc._validated_files)
    
    def test_missing_sfh_parameters_fails(self):
        """Test that missing SFH parameters are detected."""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create a file without SFH parameters
            with h5py.File(tmp_filename, 'w') as dst:
                dst.create_group('/Parameters')
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                self.calc.validate_sfh_compatibility(tmp_filename)
            
            self.assertIn('No starFormationHistory parameters', str(context.exception))
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)


class TestIntegrationWithEvaluateComponentSpectrum(unittest.TestCase):
    """Test integration of validation with evaluate_component_spectrum."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
        self.galacticus_file = 'data/romanUNIT.hdf5'
        self.calc = sed_calculator(self.sed_template_file)
    
    def test_evaluate_component_spectrum_validates_compatibility(self):
        """Test that evaluate_component_spectrum validates compatibility."""
        # Should successfully create spectrum for compatible files
        spectrum = self.calc.evaluate_component_spectrum(
            self.galacticus_file, galIndex=0, component='disk'
        )
        
        # Spectrum should be created
        self.assertIsNotNone(spectrum)
        self.assertTrue(hasattr(spectrum, 'waveset'))
    
    def test_evaluate_component_spectrum_fails_for_incompatible(self):
        """Test that evaluate_component_spectrum fails for incompatible files."""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create a file with mismatched parameters but valid structure
            with h5py.File(self.galacticus_file, 'r') as src:
                with h5py.File(tmp_filename, 'w') as dst:
                    # Copy everything
                    for key in src.keys():
                        src.copy(key, dst)
                    # Modify SFH parameters
                    dst['/Parameters/starFormationHistory'].attrs['countAges'] = 40
            
            # Should raise ValueError when trying to evaluate spectrum
            with self.assertRaises(ValueError) as context:
                self.calc.evaluate_component_spectrum(tmp_filename, galIndex=0, component='disk')
            
            self.assertIn('incompatible', str(context.exception))
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)


class TestCalculateMagnitudes(unittest.TestCase):
    """Test the calculate_magnitudes method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
        self.galacticus_file = 'data/romanUNIT.hdf5'
        self.calc = sed_calculator(self.sed_template_file)
    
    def test_calculate_magnitudes_method_exists(self):
        """Test that calculate_magnitudes method exists and has correct signature."""
        self.assertTrue(hasattr(self.calc, 'calculate_magnitudes'))
        
        # Check method is callable
        self.assertTrue(callable(getattr(self.calc, 'calculate_magnitudes')))
    
    def test_calculate_magnitudes_invalid_component(self):
        """Test that calculate_magnitudes raises error for invalid component."""
        # Create a dummy bandpass
        from synphot import SpectralElement
        from synphot.models import Empirical1D
        import astropy.units as u
        
        wavelengths = np.linspace(10000, 20000, 100) * u.AA
        transmission = np.ones(100)
        bandpass = SpectralElement(Empirical1D, points=wavelengths, lookup_table=transmission)
        bandpasses = {'test': bandpass}
        
        # Test with invalid component
        with self.assertRaises(ValueError) as context:
            self.calc.calculate_magnitudes(
                self.galacticus_file,
                galIndex=0,
                bandpasses=bandpasses,
                component='invalid_component'
            )
        
        self.assertIn('Invalid component', str(context.exception))
    
    def test_calculate_magnitudes_invalid_magnitude_system(self):
        """Test that calculate_magnitudes raises error for invalid magnitude system."""
        # Create a dummy bandpass
        from synphot import SpectralElement
        from synphot.models import Empirical1D
        import astropy.units as u
        
        wavelengths = np.linspace(10000, 20000, 100) * u.AA
        transmission = np.ones(100)
        bandpass = SpectralElement(Empirical1D, points=wavelengths, lookup_table=transmission)
        bandpasses = {'test': bandpass}
        
        # Test with invalid magnitude system
        with self.assertRaises(ValueError) as context:
            self.calc.calculate_magnitudes(
                self.galacticus_file,
                galIndex=0,
                bandpasses=bandpasses,
                magnitude_system='invalid_system'
            )
        
        self.assertIn('Invalid magnitude_system', str(context.exception))


class TestFormatDetection(unittest.TestCase):
    """Test auto-detection of Galacticus file formats."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
        self.lightcone_file = 'data/romanUNIT.hdf5'
    
    def test_detect_lightcone_format(self):
        """Test detection of lightcone format."""
        from SEDfromSFH import detect_galacticus_format
        
        format_type, base_path = detect_galacticus_format(self.lightcone_file)
        
        self.assertEqual(format_type, 'lightcone')
        self.assertEqual(base_path, '/Lightcone/Output1')
    
    def test_detect_fixed_time_format(self):
        """Test detection of fixed-time format."""
        from SEDfromSFH import detect_galacticus_format
        from astropy.cosmology import Planck15
        
        # Create a minimal fixed-time format file
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            with h5py.File(tmp_filename, 'w') as f:
                outputs = f.create_group('/Outputs')
                output1 = outputs.create_group('Output1')
                output1.attrs['outputTime'] = 4.5  # Gyr
            
            format_type, base_path = detect_galacticus_format(tmp_filename)
            
            self.assertEqual(format_type, 'fixed-time')
            self.assertEqual(base_path, '/Outputs/Output1')
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_detect_format_raises_on_invalid_file(self):
        """Test that format detection fails gracefully on invalid files."""
        from SEDfromSFH import detect_galacticus_format
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create a file with neither Lightcone nor Outputs
            with h5py.File(tmp_filename, 'w') as f:
                f.create_group('/SomeOtherStructure')
            
            with self.assertRaises(ValueError) as context:
                detect_galacticus_format(tmp_filename)
            
            self.assertIn('Cannot determine format', str(context.exception))
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)


class TestTimeConversion(unittest.TestCase):
    """Test time conversion utilities for fixed-time format."""
    
    def test_outputTime_to_redshift(self):
        """Test conversion from age of universe to redshift."""
        from SEDfromSFH import outputTime_to_redshift
        from astropy.cosmology import Planck15
        
        # Test at a known redshift
        z_expected = 1.0
        output_time = Planck15.age(z_expected).to_value('Gyr')
        z_calculated = outputTime_to_redshift(output_time, Planck15)
        
        self.assertAlmostEqual(float(z_calculated), z_expected, places=4)
    
    def test_age_of_universe_to_lookback_time(self):
        """Test conversion from age of universe to lookback time."""
        from SEDfromSFH import age_of_universe_to_lookback_time
        
        outputTime = 10.0  # Gyr
        ages = np.array([2.0, 5.0, 8.0, 10.0])
        
        lookback_times = age_of_universe_to_lookback_time(ages, outputTime)
        expected = np.array([8.0, 5.0, 2.0, 0.0])
        
        np.testing.assert_array_almost_equal(lookback_times, expected)
    
    def test_age_of_universe_to_lookback_time_handles_negative(self):
        """Test that negative lookback times are clamped to zero."""
        from SEDfromSFH import age_of_universe_to_lookback_time
        
        outputTime = 10.0
        ages = np.array([5.0, 10.0, 10.1])  # Last one would give negative lookback
        
        lookback_times = age_of_universe_to_lookback_time(ages, outputTime)
        
        # All should be non-negative
        self.assertTrue(np.all(lookback_times >= 0))
        # Last one should be clamped to 0
        self.assertAlmostEqual(lookback_times[-1], 0.0)


class TestFixedTimeFormat(unittest.TestCase):
    """Test reading and processing fixed-time format files."""
    
    def setUp(self):
        """Set up test fixtures including a fixed-time format test file."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
        self.lightcone_file = 'data/romanUNIT.hdf5'
        self.calc = sed_calculator(self.sed_template_file)
        
        # Create a fixed-time format test file
        self.fixed_time_file = None
        self._create_fixed_time_test_file()
    
    def _create_fixed_time_test_file(self):
        """Create a minimal fixed-time format test file."""
        from astropy.cosmology import Planck15
        
        # Read sample galaxy from lightcone
        with h5py.File(self.lightcone_file, 'r') as src:
            galIndex = 0
            redshift = src['/Lightcone/Output1/nodeData/lightconeRedshiftObserved'][galIndex]
            diskSFH_raw = src['/Lightcone/Output1/nodeData/diskStarFormationHistoryMass'][galIndex]
            diskSFH_times_lookback = src['/Lightcone/Output1/nodeData/diskStarFormationHistoryTimes'][galIndex]
            spheroidSFH_raw = src['/Lightcone/Output1/nodeData/spheroidStarFormationHistoryMass'][galIndex]
            spheroidSFH_times_lookback = src['/Lightcone/Output1/nodeData/spheroidStarFormationHistoryTimes'][galIndex]
            
            outputTime = Planck15.age(redshift).to_value('Gyr')
            diskSFH_times_age = outputTime - diskSFH_times_lookback
            spheroidSFH_times_age = outputTime - spheroidSFH_times_lookback
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            self.fixed_time_file = tmp.name
        
        with h5py.File(self.fixed_time_file, 'w') as dst:
            output_group = dst.create_group('/Outputs/Output1')
            output_group.attrs['outputTime'] = outputTime
            
            node_data = output_group.create_group('nodeData')
            
            with h5py.File(self.lightcone_file, 'r') as src:
                src_diskSFH = src['/Lightcone/Output1/nodeData/diskStarFormationHistoryMass']
                src_spheroidSFH = src['/Lightcone/Output1/nodeData/spheroidStarFormationHistoryMass']
                
                diskSFH_dataset = node_data.create_dataset(
                    'diskStarFormationHistoryMass',
                    shape=(1,),
                    dtype=src_diskSFH.dtype
                )
                diskSFH_dataset[0] = src_diskSFH[galIndex]
                diskSFH_dataset.attrs['time'] = diskSFH_times_age
                diskSFH_dataset.attrs['metallicity'] = np.array([
                    0.0001, 0.000316228, 0.001, 0.00316228, 0.01, 0.0316228, 
                    0.1, 0.316228, 1.0, 3.16228, 10.0, 1.7976931348623157e+308
                ])
                diskSFH_dataset.attrs['unitsInSI'] = 1.98841586e+40
                
                spheroidSFH_dataset = node_data.create_dataset(
                    'spheroidStarFormationHistoryMass',
                    shape=(1,),
                    dtype=src_spheroidSFH.dtype
                )
                spheroidSFH_dataset[0] = src_spheroidSFH[galIndex]
                spheroidSFH_dataset.attrs['time'] = spheroidSFH_times_age
                spheroidSFH_dataset.attrs['metallicity'] = diskSFH_dataset.attrs['metallicity']
                spheroidSFH_dataset.attrs['unitsInSI'] = 1.98841586e+40
                
                src.copy('/Parameters', dst)
    
    def tearDown(self):
        """Clean up test files."""
        if self.fixed_time_file and os.path.exists(self.fixed_time_file):
            os.unlink(self.fixed_time_file)
    
    def test_read_fixed_time_galaxy(self):
        """Test reading galaxy data from fixed-time format."""
        galData = self.calc.read_galacticus_galaxy(self.fixed_time_file, galIndex=0)
        
        # Check that required keys are present
        self.assertIn('redshift', galData)
        self.assertIn('diskSFH', galData)
        self.assertIn('diskSFH_times', galData)
        self.assertIn('spheroidSFH', galData)
        self.assertIn('spheroidSFH_times', galData)
        self.assertIn('format_type', galData)
        self.assertIn('outputTime', galData)
        
        # Check format type
        self.assertEqual(galData['format_type'], 'fixed-time')
        
        # Check data shapes
        self.assertEqual(galData['diskSFH'].ndim, 2)
        self.assertEqual(galData['spheroidSFH'].ndim, 2)
        self.assertEqual(galData['diskSFH_times'].ndim, 1)
        self.assertEqual(galData['spheroidSFH_times'].ndim, 1)
    
    def test_fixed_time_matches_lightcone(self):
        """Test that fixed-time and lightcone formats produce consistent results."""
        # Read from both formats
        galData_lc = self.calc.read_galacticus_galaxy(self.lightcone_file, galIndex=0)
        galData_ft = self.calc.read_galacticus_galaxy(self.fixed_time_file, galIndex=0)
        
        # Redshifts should match (within numerical precision)
        self.assertAlmostEqual(
            float(galData_ft['redshift']), 
            galData_lc['redshift'], 
            places=4
        )
        
        # SFH data should match exactly (we copied it)
        np.testing.assert_array_equal(galData_ft['diskSFH'], galData_lc['diskSFH'])
        np.testing.assert_array_equal(galData_ft['spheroidSFH'], galData_lc['spheroidSFH'])
        
        # Times should match
        np.testing.assert_array_almost_equal(
            galData_ft['diskSFH_times'], 
            galData_lc['diskSFH_times'],
            decimal=4
        )
    
    def test_fixed_time_spectrum_calculation(self):
        """Test that spectrum calculation works with fixed-time format."""
        import astropy.units as u
        
        galData = self.calc.read_galacticus_galaxy(self.fixed_time_file, galIndex=0)
        
        # Calculate flux
        wavelengths = np.linspace(10000, 20000, 100) * u.AA
        Fnu, wav = self.calc.calculate_continuum_Fnu(
            galData['diskSFH'],
            galData['redshift'],
            wavelengths=wavelengths
        )
        
        # Check output
        self.assertEqual(Fnu.shape, (100,))
        self.assertEqual(wav.shape, (100,))
        self.assertTrue(np.all(np.isfinite(Fnu.value)))
        self.assertTrue(np.all(Fnu.value >= 0))
    
    def test_format_caching(self):
        """Test that format detection is cached."""
        # First read
        self.calc.read_galacticus_galaxy(self.fixed_time_file, galIndex=0)
        self.assertIn(self.fixed_time_file, self.calc._file_formats)
        
        # Second read should use cache
        format_info = self.calc._file_formats[self.fixed_time_file]
        self.assertEqual(format_info[0], 'fixed-time')
        self.assertEqual(format_info[1], '/Outputs/Output1')


if __name__ == '__main__':
    unittest.main()
