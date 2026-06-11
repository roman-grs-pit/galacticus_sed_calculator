"""
Tests for explanatory magnitude-failure warnings.
"""

import os
import tempfile
import unittest

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import h5py
import numpy as np

from galacticus_sed_calculator.sed_calculator import SEDCalculator


class FakeBandpass:
    """Minimal stand-in for a synphot bandpass with a wavelength set."""

    def __init__(self, waveset):
        self.waveset = waveset


def _make_lightcone_file(path, redshift=0.5, disk_mass=1.0, spheroid_mass=1.0):
    with h5py.File(path, 'w') as f:
        nd = f.require_group('Lightcone/Output1/nodeData')
        nd.create_dataset('lightconeRedshiftObserved', data=[redshift])
        nd.create_dataset('diskMassStellar', data=[disk_mass])
        nd.create_dataset('spheroidMassStellar', data=[spheroid_mass])


def _make_calculator():
    calc = object.__new__(SEDCalculator)
    calc.sedWavelength = np.linspace(1000.0, 2000.0, 10)
    calc._file_formats = {}
    calc.cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    return calc


class TestMagnitudeFailureMessages(unittest.TestCase):
    def test_zero_stellar_mass_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'catalog.hdf5')
            _make_lightcone_file(path, disk_mass=0.0, spheroid_mass=0.0)
            calc = _make_calculator()
            message = calc._magnitude_failure_message(
                path,
                0,
                'F062',
                FakeBandpass(np.linspace(5000, 7000, 5) * u.AA),
                np.linspace(4000, 10000, 20) * u.AA,
                ValueError('Integrated flux is <= 0'),
            )
        self.assertIn('setting magnitude to NaN', message)
        self.assertIn('zero disk+spheroid stellar mass', message)

    def test_rest_frame_template_coverage_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'catalog.hdf5')
            _make_lightcone_file(
                path, redshift=6.0, disk_mass=1.0, spheroid_mass=0.0
            )
            calc = _make_calculator()
            message = calc._magnitude_failure_message(
                path,
                0,
                'F062',
                FakeBandpass(np.linspace(5000, 6000, 5) * u.AA),
                np.linspace(4000, 10000, 20) * u.AA,
                ValueError('Integrated flux is <= 0'),
            )
        self.assertIn('at z=6.00', message)
        self.assertIn('outside the SED template range', message)


if __name__ == '__main__':
    unittest.main()
