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


if __name__ == '__main__':
    unittest.main()
