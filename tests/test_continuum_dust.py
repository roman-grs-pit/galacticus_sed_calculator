"""
Tests for continuum dust attenuation.
"""

import os
import sys
import unittest

import astropy.units as u
import numpy as np

# Add parent directory to path to import galacticus_sed_calculator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from galacticus_sed_calculator import SEDCalculator
from galacticus_sed_calculator.dust_attenuation import (
    apply_dust_attenuation_to_continuum,
    calzetti_attenuation_law,
    dust_attenuation_garnBest10,
    dust_attenuation_gb10_generalised,
    normalize_continuum_dust,
)


CONTINUUM_DUST = {
    'model': 'fixed_av',
    'params': {'A_V': 1.0},
    'law': 'calzetti',
}


class TestContinuumDustHelpers(unittest.TestCase):
    """Test standalone continuum dust helper functions."""

    def test_calzetti_handles_integer_wavelength_arrays(self):
        int_wavelengths = np.array([5500, 10000])
        float_wavelengths = int_wavelengths.astype(float)

        A_int = calzetti_attenuation_law(int_wavelengths, A_V=1.0)
        A_float = calzetti_attenuation_law(float_wavelengths, A_V=1.0)

        np.testing.assert_allclose(A_int, A_float)
        self.assertGreater(A_int[0], 0.0)

    def test_apply_continuum_dust_matches_calzetti_factor(self):
        wavelengths = np.array([1500.0, 5500.0, 10000.0])
        flux = np.ones_like(wavelengths)

        attenuated = apply_dust_attenuation_to_continuum(
            flux, wavelengths, A_V=1.0, dust_law='calzetti'
        )
        expected = flux * 10.0 ** (
            -0.4 * calzetti_attenuation_law(wavelengths, A_V=1.0)
        )

        np.testing.assert_allclose(attenuated, expected)
        self.assertLess(attenuated[0], attenuated[-1])

    def test_normalize_scalar_as_fixed_av(self):
        normalized = normalize_continuum_dust(1.0)
        self.assertEqual(normalized['model'], 'fixed_av')
        self.assertEqual(normalized['law'], 'calzetti')
        self.assertAlmostEqual(normalized['params']['A_V'], 1.0)

    def test_unknown_continuum_dust_model_raises(self):
        with self.assertRaises(ValueError):
            normalize_continuum_dust({'model': 'not_yet_supported'})


class TestDustRandomUniformScatter(unittest.TestCase):
    """Test robust handling of stored uniform deviates for dust scatter."""

    def test_garn_best_scatter_clips_uniform_endpoints(self):
        attenuation = dust_attenuation_garnBest10(
            np.array([1.0e10, 1.0e10]),
            attenuation_scatter=1.0,
            random_uniform=np.array([0.0, 1.0]),
        )

        self.assertTrue(np.all(np.isfinite(attenuation)))
        self.assertGreater(attenuation[1], attenuation[0])
        self.assertLess(attenuation[1], 5.0)

    def test_generalised_scatter_clips_uniform_endpoints(self):
        attenuation = dust_attenuation_gb10_generalised(
            np.array([1.0e10, 1.0e10]),
            np.array([1.0, 1.0]),
            attenuation_scatter=1.0,
            random_uniform=np.array([0.0, 1.0]),
        )

        self.assertTrue(np.all(np.isfinite(attenuation)))
        self.assertEqual(attenuation[0], 0.0)
        self.assertGreater(attenuation[1], attenuation[0])
        self.assertLess(attenuation[1], 5.0)


class TestContinuumDustSEDIntegration(unittest.TestCase):
    """Test continuum dust through SEDCalculator."""

    def setUp(self):
        self.sed_template_file = (
            'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'
        )
        self.galacticus_file = 'data/romanUNIT.hdf5'
        self.calc = SEDCalculator(self.sed_template_file)

    def test_calculate_continuum_fnu_with_fixed_av_is_fainter(self):
        galData = self.calc.read_galacticus_galaxy(self.galacticus_file, 0)
        wavelengths = np.linspace(10000, 20000, 100) * u.AA

        Fnu_no_dust, _ = self.calc.calculate_continuum_Fnu(
            galData['diskSFH'],
            galData['redshift'],
            wavelengths=wavelengths,
        )
        Fnu_with_dust, _ = self.calc.calculate_continuum_Fnu(
            galData['diskSFH'],
            galData['redshift'],
            wavelengths=wavelengths,
            continuum_dust=CONTINUUM_DUST,
        )

        valid = Fnu_no_dust.value > 0
        self.assertTrue(np.any(valid))
        self.assertTrue(np.all(Fnu_with_dust.value[valid] <= Fnu_no_dust.value[valid]))
        self.assertTrue(np.any(Fnu_with_dust.value[valid] < Fnu_no_dust.value[valid]))

    def test_evaluate_component_spectrum_continuum_dust(self):
        wavelengths = np.linspace(10000, 20000, 200) * u.AA

        spectrum_no_dust = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=False,
            use_synphot=False,
        )
        spectrum_with_dust = self.calc.evaluate_component_spectrum(
            self.galacticus_file,
            galIndex=0,
            component='disk',
            obs_wavelengths=wavelengths,
            include_emission_lines=False,
            use_synphot=False,
            continuum_dust=1.0,
        )

        flux_no_dust = spectrum_no_dust(wavelengths, flux_unit='FNU')
        flux_with_dust = spectrum_with_dust(wavelengths, flux_unit='FNU')
        valid = flux_no_dust.value > 0

        self.assertTrue(np.all(flux_with_dust.value[valid] <= flux_no_dust.value[valid]))
        self.assertTrue(np.any(flux_with_dust.value[valid] < flux_no_dust.value[valid]))


if __name__ == '__main__':
    unittest.main()
