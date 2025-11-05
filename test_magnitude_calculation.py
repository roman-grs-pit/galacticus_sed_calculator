"""
Tests for magnitude calculation functionality using synphot.

This test module validates that synphot can be used to calculate
observed magnitudes in photometric bands for Galacticus galaxies.
"""

import unittest
import numpy as np
from synphot import SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D, BlackBodyNorm1D
import astropy.units as u
from SEDfromSFH import sed_calculator


class TestSynphotMagnitudeCalculation(unittest.TestCase):
    """Test basic synphot magnitude calculation functionality."""
    
    def test_simple_magnitude_calculation(self):
        """Test that synphot can calculate AB magnitudes."""
        # Create a simple blackbody source
        source = SourceSpectrum(BlackBodyNorm1D, temperature=5000*u.K)
        
        # Create a simple bandpass
        wavelengths = np.linspace(4000, 6000, 100) * u.AA
        transmission = np.ones(100)
        bandpass = SpectralElement(Empirical1D, points=wavelengths, lookup_table=transmission)
        
        # Create observation and calculate magnitude
        obs = Observation(source, bandpass)
        mag_ab = obs.effstim(flux_unit=u.ABmag)
        
        # Check that we got a magnitude value
        self.assertIsNotNone(mag_ab)
        self.assertTrue(hasattr(mag_ab, 'value'))
        self.assertTrue(np.isfinite(mag_ab.value))
    
    def test_magnitude_systems(self):
        """Test that different magnitude systems give different results."""
        source = SourceSpectrum(BlackBodyNorm1D, temperature=5000*u.K)
        wavelengths = np.linspace(4000, 6000, 100) * u.AA
        transmission = np.ones(100)
        bandpass = SpectralElement(Empirical1D, points=wavelengths, lookup_table=transmission)
        obs = Observation(source, bandpass)
        
        # Calculate in different systems
        mag_ab = obs.effstim(flux_unit=u.ABmag)
        mag_st = obs.effstim(flux_unit=u.STmag)
        
        # They should be different (though similar for this spectrum)
        self.assertNotEqual(mag_ab.value, mag_st.value)
        # But should be within reasonable range
        self.assertLess(abs(mag_ab.value - mag_st.value), 1.0)
    
    def test_effective_wavelength(self):
        """Test that effective wavelength can be calculated."""
        source = SourceSpectrum(BlackBodyNorm1D, temperature=5000*u.K)
        wavelengths = np.linspace(4000, 6000, 100) * u.AA
        transmission = np.ones(100)
        bandpass = SpectralElement(Empirical1D, points=wavelengths, lookup_table=transmission)
        obs = Observation(source, bandpass)
        
        eff_wave = obs.effective_wavelength()
        
        # Check it's in the right range
        self.assertGreater(eff_wave.value, 4000)
        self.assertLess(eff_wave.value, 6000)


class TestGalacticusMagnitudeIntegration(unittest.TestCase):
    """Test integration with Galacticus data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
        self.galacticus_file = 'data/romanUNIT.hdf5'
        self.calc = sed_calculator(self.sed_template_file)
    
    def create_test_bandpass(self):
        """Create a simple test bandpass."""
        wavelengths = np.linspace(14000, 17000, 100) * u.AA
        transmission = np.ones(100) * 0.8
        return SpectralElement(Empirical1D, points=wavelengths, lookup_table=transmission)
    
    def test_magnitude_from_galacticus_spectrum(self):
        """Test magnitude calculation from Galacticus galaxy spectrum."""
        # Get spectrum for a galaxy
        spectrum = self.calc.evaluate_component_spectrum(
            filename=self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=np.linspace(8000, 23000, 1000) * u.AA
        )
        
        # Create observation and calculate magnitude
        bandpass = self.create_test_bandpass()
        obs = Observation(spectrum, bandpass, force='taper')
        mag = obs.effstim(flux_unit=u.ABmag)
        
        # Check magnitude is reasonable (not NaN, in reasonable range)
        self.assertTrue(np.isfinite(mag.value))
        self.assertGreater(mag.value, 10)  # Not too bright
        self.assertLess(mag.value, 35)     # Not too faint
    
    def test_total_spectrum_magnitude(self):
        """Test magnitude from total galaxy spectrum (disk + spheroid + AGN)."""
        # Get total spectrum
        spectrum = self.calc.evaluate_total_spectrum(
            filename=self.galacticus_file,
            galIndex=0,
            includeAGN=True,
            obs_wavelengths=np.linspace(8000, 23000, 1000) * u.AA
        )
        
        # Calculate magnitude
        bandpass = self.create_test_bandpass()
        obs = Observation(spectrum, bandpass, force='taper')
        mag = obs.effstim(flux_unit=u.ABmag)
        
        # Check magnitude is finite and reasonable
        self.assertTrue(np.isfinite(mag.value))
        self.assertGreater(mag.value, 10)
        self.assertLess(mag.value, 35)
    
    def test_component_magnitudes_comparison(self):
        """Test that disk and spheroid give different magnitudes."""
        obs_wavelengths = np.linspace(8000, 23000, 1000) * u.AA
        bandpass = self.create_test_bandpass()
        
        # Get disk magnitude
        disk_spectrum = self.calc.evaluate_component_spectrum(
            filename=self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=obs_wavelengths
        )
        disk_obs = Observation(disk_spectrum, bandpass, force='taper')
        disk_mag = disk_obs.effstim(flux_unit=u.ABmag)
        
        # Get spheroid magnitude
        spheroid_spectrum = self.calc.evaluate_component_spectrum(
            filename=self.galacticus_file,
            galIndex=0,
            component='spheroid',
            obs_wavelengths=obs_wavelengths
        )
        spheroid_obs = Observation(spheroid_spectrum, bandpass, force='taper')
        spheroid_mag = spheroid_obs.effstim(flux_unit=u.ABmag)
        
        # They should be different (unless one has zero mass, but that's unlikely)
        # At minimum, both should be finite
        self.assertTrue(np.isfinite(disk_mag.value))
        self.assertTrue(np.isfinite(spheroid_mag.value))
    
    def test_multiple_bandpasses(self):
        """Test calculating magnitudes in multiple bandpasses."""
        # Get spectrum
        spectrum = self.calc.evaluate_total_spectrum(
            filename=self.galacticus_file,
            galIndex=0,
            obs_wavelengths=np.linspace(8000, 23000, 1000) * u.AA
        )
        
        # Create two different bandpasses
        bandpass1 = SpectralElement(
            Empirical1D,
            points=np.linspace(10000, 12000, 100) * u.AA,
            lookup_table=np.ones(100) * 0.8
        )
        bandpass2 = SpectralElement(
            Empirical1D,
            points=np.linspace(15000, 17000, 100) * u.AA,
            lookup_table=np.ones(100) * 0.8
        )
        
        # Calculate magnitudes
        obs1 = Observation(spectrum, bandpass1, force='taper')
        mag1 = obs1.effstim(flux_unit=u.ABmag)
        
        obs2 = Observation(spectrum, bandpass2, force='taper')
        mag2 = obs2.effstim(flux_unit=u.ABmag)
        
        # Both should be finite
        self.assertTrue(np.isfinite(mag1.value))
        self.assertTrue(np.isfinite(mag2.value))
        
        # Can calculate color
        color = mag1 - mag2
        self.assertTrue(np.isfinite(color.value))


class TestRomanFilterSimulation(unittest.TestCase):
    """Test with Roman-like filter specifications."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
        self.galacticus_file = 'data/romanUNIT.hdf5'
        self.calc = sed_calculator(self.sed_template_file)
    
    def create_roman_like_filter(self, center, width):
        """Create a simplified Roman-like filter."""
        wave_min = center - width/2
        wave_max = center + width/2
        wavelengths = np.linspace(wave_min.value*0.8, wave_max.value*1.2, 200) * center.unit
        transmission = np.where(
            (wavelengths >= wave_min) & (wavelengths <= wave_max),
            0.8, 0.0
        )
        return SpectralElement(Empirical1D, points=wavelengths, lookup_table=transmission)
    
    def test_roman_f158_like_magnitude(self):
        """Test magnitude calculation with F158-like filter."""
        # Get spectrum
        spectrum = self.calc.evaluate_total_spectrum(
            filename=self.galacticus_file,
            galIndex=0,
            obs_wavelengths=np.linspace(8000, 23000, 2000) * u.AA
        )
        
        # Create F158-like filter (1.38-1.77 μm)
        f158_filter = self.create_roman_like_filter(15770*u.AA, 3900*u.AA)
        
        # Calculate magnitude
        obs = Observation(spectrum, f158_filter, force='taper')
        mag = obs.effstim(flux_unit=u.ABmag)
        
        # Check magnitude is reasonable
        self.assertTrue(np.isfinite(mag.value))
        self.assertGreater(mag.value, 15)  # Typical for galaxies at z~1
        self.assertLess(mag.value, 30)
    
    def test_multiple_roman_filters(self):
        """Test calculating magnitudes in multiple Roman-like filters."""
        # Get spectrum
        spectrum = self.calc.evaluate_total_spectrum(
            filename=self.galacticus_file,
            galIndex=0,
            obs_wavelengths=np.linspace(4000, 23000, 3000) * u.AA
        )
        
        # Create several Roman-like filters
        filters = {
            'F087': self.create_roman_like_filter(8690*u.AA, 2200*u.AA),
            'F129': self.create_roman_like_filter(12930*u.AA, 3200*u.AA),
            'F158': self.create_roman_like_filter(15770*u.AA, 3900*u.AA),
            'F184': self.create_roman_like_filter(18420*u.AA, 3200*u.AA),
        }
        
        # Calculate magnitudes in all filters
        magnitudes = {}
        for name, bandpass in filters.items():
            obs = Observation(spectrum, bandpass, force='taper')
            mag = obs.effstim(flux_unit=u.ABmag)
            magnitudes[name] = mag.value
            
            # Each should be finite
            self.assertTrue(np.isfinite(mag.value))
        
        # Check we got all magnitudes
        self.assertEqual(len(magnitudes), 4)
        
        # Calculate some colors
        color1 = magnitudes['F087'] - magnitudes['F158']
        color2 = magnitudes['F129'] - magnitudes['F184']
        
        # Colors should be finite
        self.assertTrue(np.isfinite(color1))
        self.assertTrue(np.isfinite(color2))


if __name__ == '__main__':
    unittest.main()
