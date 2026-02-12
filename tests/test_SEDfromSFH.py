"""
Tests for SED calculator compatibility validation.
"""
import unittest
import tempfile
import os
import sys
import h5py
import numpy as np

# Add parent directory to path to import galacticus_sed_calculator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator import sed_calculator


class TestSEDTemplateParameterExtraction(unittest.TestCase):
    """Test the extraction of SFH parameters from SED template."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Get path to data directory relative to this test file
        test_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(test_dir)
        self.sed_template_file = os.path.join(parent_dir, 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5')
        self.galacticus_file = os.path.join(parent_dir, 'data/romanUNIT.hdf5')
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
        # Get path to data directory relative to this test file
        test_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(test_dir)
        self.sed_template_file = os.path.join(parent_dir, 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5')
        self.galacticus_file = os.path.join(parent_dir, 'data/romanUNIT.hdf5')
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
        self.sed_template_file = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
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
        self.sed_template_file = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
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
    
    def test_fixed_time_format_parameter_validation(self):
        """Test that validation works with fixed-time format parameter names."""
        import tempfile
        
        # Create a fixed-time SED template
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_sed:
            tmp_sed_filename = tmp_sed.name
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_gal:
            tmp_gal_filename = tmp_gal.name
        
        try:
            # Create fixed-time SED template
            with h5py.File(self.sed_template_file, 'r') as src:
                ages = src['ages'][:]
                sedTemplate = src['sedTemplate'][:]
                metallicity = src['metallicity'][:]
                wavelength = src['wavelength'][:]
                
                outputTime = ages[0]
                times = outputTime - ages
            
            with h5py.File(tmp_sed_filename, 'w') as dst:
                dst.create_dataset('time', data=times)
                dst.create_dataset('sedTemplate', data=np.flip(sedTemplate, axis=1))
                dst.create_dataset('metallicity', data=metallicity)
                dst.create_dataset('wavelength', data=wavelength)
            
            # Create fixed-time galaxy file with matching parameters and time array
            with h5py.File(tmp_gal_filename, 'w') as dst:
                sfh_group = dst.create_group('/Parameters/starFormationHistory')
                # Fixed-time format uses different attribute names
                sfh_group.attrs['countMetallicities'] = 11
                sfh_group.attrs['countTimeStepsMaximum'] = len(times)  # Match actual time array length
                sfh_group.attrs['metallicityMaximum'] = 10.0
                sfh_group.attrs['metallicityMinimum'] = 0.0001
                sfh_group.attrs['timeStepMinimum'] = 0.001
                
                # Create minimal Outputs structure
                outputs = dst.create_group('/Outputs')
                output1 = outputs.create_group('Output1')
                output1.attrs['outputTime'] = outputTime
                node_data = output1.create_group('nodeData')
                
                # Add SFH dataset with time attribute that matches SED template
                diskSFH = node_data.create_dataset('diskStarFormationHistoryMass',
                                                   shape=(1,), dtype=h5py.vlen_dtype(np.dtype('float64')))
                diskSFH[0] = np.zeros((11, len(times))).flatten()
                diskSFH.attrs['time'] = times
            
            # Should not raise an exception with fixed-time parameter names and matching times
            calc = sed_calculator(tmp_sed_filename)
            calc.validate_sfh_compatibility(tmp_gal_filename)
            
        finally:
            if os.path.exists(tmp_sed_filename):
                os.unlink(tmp_sed_filename)
            if os.path.exists(tmp_gal_filename):
                os.unlink(tmp_gal_filename)


class TestSEDTemplateFormats(unittest.TestCase):
    """Test support for different SED template formats."""
    
    def test_lightcone_sed_template(self):
        """Test that lightcone SED templates (with /ages) load correctly."""
        sed_template_file = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
        calc = sed_calculator(sed_template_file)
        
        # Check format was detected correctly
        self.assertEqual(calc.sedTemplateFormat, 'lightcone')
        
        # Check that sedAges was loaded
        self.assertTrue(hasattr(calc, 'sedAges'))
        self.assertGreater(len(calc.sedAges), 0)
        
        # Check parameters extraction
        params = calc.get_sed_template_parameters()
        self.assertIn('sedTemplateFormat', params)
        self.assertEqual(params['sedTemplateFormat'], 'lightcone')
        self.assertGreater(params['countAges'], 0)
    
    def test_fixed_time_sed_template(self):
        """Test that fixed-time SED templates (with /time) load correctly."""
        import tempfile
        
        # Create a test fixed-time SED template
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Load lightcone template to convert
            lightcone_sed = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
            with h5py.File(lightcone_sed, 'r') as src:
                ages = src['ages'][:]
                sedTemplate = src['sedTemplate'][:]
                metallicity = src['metallicity'][:]
                wavelength = src['wavelength'][:]
                
                # Convert to cosmic time (ascending)
                outputTime = ages[0]
                times = outputTime - ages
            
            # Create fixed-time template
            with h5py.File(tmp_filename, 'w') as dst:
                dst.create_dataset('time', data=times)
                dst.create_dataset('sedTemplate', data=np.flip(sedTemplate, axis=1))
                dst.create_dataset('metallicity', data=metallicity)
                dst.create_dataset('wavelength', data=wavelength)
            
            # Load and test
            calc = sed_calculator(tmp_filename)
            
            # Check format was detected correctly
            self.assertEqual(calc.sedTemplateFormat, 'fixed-time')
            
            # Check that sedTime was loaded
            self.assertTrue(hasattr(calc, 'sedTime'))
            self.assertGreater(len(calc.sedTime), 0)
            
            # Check that sedTime is in ascending order (cosmic time)
            self.assertTrue(np.all(np.diff(calc.sedTime) >= 0))
            
            # Check parameters extraction
            params = calc.get_sed_template_parameters()
            self.assertIn('sedTemplateFormat', params)
            self.assertEqual(params['sedTemplateFormat'], 'fixed-time')
            self.assertGreater(params['countAges'], 0)
            
            # For fixed-time format: countAges should equal len(sedTime)
            # (because t=0 edge is implicit, not stored)
            self.assertEqual(params['countAges'], len(calc.sedTime))
            
            # ageMinimum should be 0 (implicit lower bound)
            self.assertEqual(params['ageMinimum'], 0.0)
            
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_sed_template_missing_time_data(self):
        """Test that SED template without ages or time raises error."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create an invalid SED template without ages or time
            with h5py.File(tmp_filename, 'w') as dst:
                dst.create_dataset('sedTemplate', data=np.zeros((12, 51, 369)))
                dst.create_dataset('metallicity', data=np.zeros(12))
                dst.create_dataset('wavelength', data=np.zeros(369))
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                sed_calculator(tmp_filename)
            
            self.assertIn('ages', str(context.exception))
            self.assertIn('time', str(context.exception))
            
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)


class TestFormatValidation(unittest.TestCase):
    """Test strict format validation between SED templates and galaxy files."""
    
    def test_lightcone_template_with_lightcone_galaxy_passes(self):
        """Test that lightcone template with lightcone galaxy passes validation."""
        import tempfile
        
        # Use existing lightcone SED template
        sed_template_file = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
        calc = sed_calculator(sed_template_file)
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create lightcone galaxy file
            with h5py.File(tmp_filename, 'w') as f:
                params = f.create_group('/Parameters')
                sfh = params.create_group('starFormationHistory')
                sfh.attrs['ageMinimum'] = 0.001
                sfh.attrs['countAges'] = 50
                sfh.attrs['metallicityMinimum'] = 0.0001
                sfh.attrs['metallicityMaximum'] = 10.0
                sfh.attrs['countMetallicities'] = 11
                
                lightcone = f.create_group('/Lightcone')
                output1 = lightcone.create_group('Output1')
                node_data = output1.create_group('nodeData')
            
            # Should pass
            calc.validate_sfh_compatibility(tmp_filename)
            
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_lightcone_template_with_fixed_time_galaxy_fails(self):
        """Test that lightcone template with fixed-time galaxy fails validation."""
        import tempfile
        
        sed_template_file = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
        calc = sed_calculator(sed_template_file)
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_filename = tmp.name
        
        try:
            # Create fixed-time galaxy file
            with h5py.File(tmp_filename, 'w') as f:
                params = f.create_group('/Parameters')
                sfh = params.create_group('starFormationHistory')
                sfh.attrs['timeStepMinimum'] = 0.001
                sfh.attrs['countTimeStepsMaximum'] = 50
                sfh.attrs['metallicityMinimum'] = 0.0001
                sfh.attrs['metallicityMaximum'] = 10.0
                sfh.attrs['countMetallicities'] = 11
                
                outputs = f.create_group('/Outputs')
                output1 = outputs.create_group('Output1')
                output1.attrs['outputTime'] = 3.5
                node_data = output1.create_group('nodeData')
            
            # Should fail with format mismatch
            with self.assertRaises(ValueError) as context:
                calc.validate_sfh_compatibility(tmp_filename)
            
            self.assertIn('does not match', str(context.exception))
            self.assertIn('lightcone', str(context.exception))
            self.assertIn('fixed-time', str(context.exception))
            
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)
    
    def test_fixed_time_template_with_matching_time_array_passes(self):
        """Test that fixed-time template with matching time array passes validation."""
        import tempfile
        
        # Create fixed-time SED template
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_sed:
            tmp_sed_filename = tmp_sed.name
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_gal:
            tmp_gal_filename = tmp_gal.name
        
        try:
            # Load lightcone template and convert to fixed-time
            with h5py.File('data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5', 'r') as src:
                ages = src['ages'][:]
                sedTemplate = src['sedTemplate'][:]
                metallicity = src['metallicity'][:]
                wavelength = src['wavelength'][:]
                
                outputTime = ages[0]
                times = outputTime - ages
            
            with h5py.File(tmp_sed_filename, 'w') as dst:
                dst.create_dataset('time', data=times)
                dst.create_dataset('sedTemplate', data=np.flip(sedTemplate, axis=1))
                dst.create_dataset('metallicity', data=metallicity)
                dst.create_dataset('wavelength', data=wavelength)
            
            # Create fixed-time galaxy file with MATCHING time array
            with h5py.File(tmp_gal_filename, 'w') as f:
                params = f.create_group('/Parameters')
                sfh = params.create_group('starFormationHistory')
                sfh.attrs['timeStepMinimum'] = 0.001
                sfh.attrs['countTimeStepsMaximum'] = len(times)
                sfh.attrs['metallicityMinimum'] = 0.0001
                sfh.attrs['metallicityMaximum'] = 10.0
                sfh.attrs['countMetallicities'] = 11
                
                outputs = f.create_group('/Outputs')
                output1 = outputs.create_group('Output1')
                output1.attrs['outputTime'] = outputTime
                node_data = output1.create_group('nodeData')
                
                diskSFH = node_data.create_dataset('diskStarFormationHistoryMass',
                                                   shape=(1,), dtype=h5py.vlen_dtype(np.dtype('float64')))
                diskSFH[0] = np.zeros((11, len(times))).flatten()
                diskSFH.attrs['time'] = times  # Same times as SED template
            
            # Should pass
            calc = sed_calculator(tmp_sed_filename)
            calc.validate_sfh_compatibility(tmp_gal_filename)
            
        finally:
            if os.path.exists(tmp_sed_filename):
                os.unlink(tmp_sed_filename)
            if os.path.exists(tmp_gal_filename):
                os.unlink(tmp_gal_filename)
    
    def test_fixed_time_template_with_mismatched_time_array_fails(self):
        """Test that fixed-time template with mismatched time array fails validation."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_sed:
            tmp_sed_filename = tmp_sed.name
        
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_gal:
            tmp_gal_filename = tmp_gal.name
        
        try:
            # Load lightcone template and convert to fixed-time
            with h5py.File('data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5', 'r') as src:
                ages = src['ages'][:]
                sedTemplate = src['sedTemplate'][:]
                metallicity = src['metallicity'][:]
                wavelength = src['wavelength'][:]
                
                outputTime = ages[0]
                times = outputTime - ages
            
            with h5py.File(tmp_sed_filename, 'w') as dst:
                dst.create_dataset('time', data=times)
                dst.create_dataset('sedTemplate', data=np.flip(sedTemplate, axis=1))
                dst.create_dataset('metallicity', data=metallicity)
                dst.create_dataset('wavelength', data=wavelength)
            
            # Create fixed-time galaxy file with DIFFERENT time array
            different_times = times * 1.05  # 5% different - should fail
            
            with h5py.File(tmp_gal_filename, 'w') as f:
                params = f.create_group('/Parameters')
                sfh = params.create_group('starFormationHistory')
                sfh.attrs['timeStepMinimum'] = 0.001
                sfh.attrs['countTimeStepsMaximum'] = len(different_times)
                sfh.attrs['metallicityMinimum'] = 0.0001
                sfh.attrs['metallicityMaximum'] = 10.0
                sfh.attrs['countMetallicities'] = 11
                
                outputs = f.create_group('/Outputs')
                output1 = outputs.create_group('Output1')
                output1.attrs['outputTime'] = outputTime * 1.05
                node_data = output1.create_group('nodeData')
                
                diskSFH = node_data.create_dataset('diskStarFormationHistoryMass',
                                                   shape=(1,), dtype=h5py.vlen_dtype(np.dtype('float64')))
                diskSFH[0] = np.zeros((11, len(different_times))).flatten()
                diskSFH.attrs['time'] = different_times  # Different times
            
            # Should fail with time array mismatch
            calc = sed_calculator(tmp_sed_filename)
            with self.assertRaises(ValueError) as context:
                calc.validate_sfh_compatibility(tmp_gal_filename)
            
            self.assertIn('time arrays do not match', str(context.exception))
            
        finally:
            if os.path.exists(tmp_sed_filename):
                os.unlink(tmp_sed_filename)
            if os.path.exists(tmp_gal_filename):
                os.unlink(tmp_gal_filename)


class TestFastSEDGeneration(unittest.TestCase):
    """Test the fast SED generation without synphot."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
        self.galacticus_file = 'data/romanUNIT.hdf5'
        self.calc = sed_calculator(self.sed_template_file)
    
    def test_use_synphot_parameter_exists(self):
        """Test that use_synphot parameter is accepted."""
        # Should work with use_synphot=True (default)
        import astropy.units as u
        wavelengths = np.linspace(10000, 20000, 100) * u.AA
        spectrum = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths,
            use_synphot=True
        )
        self.assertIsNotNone(spectrum)
    
    def test_use_synphot_false_returns_tuple(self):
        """Test that use_synphot=False returns a tuple."""
        import astropy.units as u
        wavelengths = np.linspace(10000, 20000, 100) * u.AA
        result = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths,
            use_synphot=False
        )
        # Should return a tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        # Check that both elements are quantities with proper units
        wav, flux = result
        self.assertIsInstance(wav, u.Quantity)
        self.assertIsInstance(flux, u.Quantity)
    
    def test_use_synphot_false_requires_wavelengths(self):
        """Test that use_synphot=False requires obs_wavelengths."""
        with self.assertRaises(ValueError) as context:
            self.calc.evaluate_component_spectrum(
                self.galacticus_file,
                galIndex=0,
                component='disk',
                obs_wavelengths=None,
                use_synphot=False
            )
        self.assertIn('obs_wavelengths must be provided', str(context.exception))
    
    def test_synphot_and_fast_paths_produce_similar_results(self):
        """Test that synphot and fast paths produce similar continuum results."""
        import astropy.units as u
        
        # Use a fine wavelength grid to resolve any differences
        wavelengths = np.linspace(10000, 20000, 500) * u.AA
        
        # Get spectrum with synphot
        spectrum_synphot = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=False,  # Test continuum only first
            use_synphot=True
        )
        
        # Get spectrum with fast path
        wav_fast, flux_fast = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=False,  # Test continuum only first
            use_synphot=False
        )
        
        # Evaluate synphot spectrum at the same wavelengths
        flux_synphot = spectrum_synphot(wavelengths, flux_unit='FNU')
        
        # Convert to same units for comparison
        flux_synphot_val = flux_synphot.to_value(u.Lsun / (u.Hz * u.Mpc**2))
        flux_fast_val = flux_fast.to_value(u.Lsun / (u.Hz * u.Mpc**2))
        
        # Should be very similar (within numerical precision)
        np.testing.assert_allclose(flux_fast_val, flux_synphot_val, rtol=1e-5)
    
    def test_fast_path_with_emission_lines(self):
        """Test that fast path works with emission lines."""
        import astropy.units as u
        
        # Use a fine wavelength grid to resolve emission lines
        wavelengths = np.linspace(8000, 30000, 2000) * u.AA
        
        # Get spectrum with emission lines
        wav_fast, flux_fast = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=True,
            use_synphot=False
        )
        
        # Check that we got results
        self.assertEqual(len(wav_fast), len(wavelengths))
        self.assertEqual(len(flux_fast), len(wavelengths))
        
        # Flux should be non-negative
        self.assertTrue(np.all(flux_fast.value >= 0))
    
    def test_gaussian_helper_functions(self):
        """Test the Gaussian emission line helper functions."""
        from SEDfromSFH import gaussian_emission_line, gaussian_from_fwhm
        import astropy.units as u
        
        # Test parameters
        wavelengths = np.linspace(10000, 11000, 1000) * u.AA
        lambda0 = 10500 * u.AA
        fwhm = 10 * u.AA
        total_flux = 1e-16 * u.erg / (u.s * u.cm**2)
        
        # Test gaussian_from_fwhm
        line_flux = gaussian_from_fwhm(wavelengths, lambda0, fwhm, total_flux)
        
        # Check units
        self.assertEqual(line_flux.unit, u.erg / (u.s * u.cm**2 * u.AA))
        
        # Check that line is centered at lambda0
        center_idx = np.argmax(line_flux.value)
        self.assertAlmostEqual(wavelengths[center_idx].value, lambda0.value, delta=1.0)
        
        # Check that integrated flux is approximately correct
        # Integrate using trapezoid rule
        integrated_flux = np.trapezoid(line_flux.value, wavelengths.value) * (u.erg / (u.s * u.cm**2))
        expected_flux = total_flux.to(u.erg / (u.s * u.cm**2))
        
        # Should be within 1% (numerical integration error)
        self.assertAlmostEqual(
            integrated_flux.value / expected_flux.value, 
            1.0, 
            delta=0.01
        )
    
    def test_line_metadata_caching(self):
        """Test that emission line metadata is properly cached."""
        import astropy.units as u
        
        wavelengths = np.linspace(10000, 20000, 100) * u.AA
        
        # First call should populate cache
        self.assertNotIn((self.galacticus_file, 'disk'), self.calc._line_metadata_cache)
        
        wav1, flux1 = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=True,
            use_synphot=False
        )
        
        # Cache should now be populated
        self.assertIn((self.galacticus_file, 'disk'), self.calc._line_metadata_cache)
        
        # Get cached data
        lineNames, lineWavelengths, hdf5_paths = self.calc._line_metadata_cache[(self.galacticus_file, 'disk')]
        
        # Verify cache contents are arrays
        self.assertIsInstance(lineNames, np.ndarray)
        self.assertIsInstance(lineWavelengths, np.ndarray)
        self.assertIsInstance(hdf5_paths, np.ndarray)
        self.assertEqual(len(lineNames), len(lineWavelengths))
        self.assertEqual(len(lineNames), len(hdf5_paths))
        
        # Second call should reuse cache (same object reference)
        cached_data = self.calc._line_metadata_cache[(self.galacticus_file, 'disk')]
        
        wav2, flux2 = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=1,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=True,
            use_synphot=False
        )
        
        # Cache should still contain same object
        self.assertIs(self.calc._line_metadata_cache[(self.galacticus_file, 'disk')], cached_data)
    
    def test_fast_path_backward_compatibility(self):
        """Test that default behavior (use_synphot=True) is unchanged."""
        import astropy.units as u
        
        wavelengths = np.linspace(10000, 20000, 100) * u.AA
        
        # Call without specifying use_synphot (should default to True)
        spectrum = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths
        )
        
        # Should return a SourceSpectrum object
        from synphot import SourceSpectrum
        self.assertIsInstance(spectrum, SourceSpectrum)


if __name__ == '__main__':
    unittest.main()
