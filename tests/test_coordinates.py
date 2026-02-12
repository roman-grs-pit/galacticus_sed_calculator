"""
Tests for coordinate transformation functionality.

Note: Galacticus lightcone angular coordinates (theta, phi) are in radians.
Tests have been updated to reflect this.
"""
import unittest
import numpy as np
import h5py
import tempfile
import os
import sys

# Add parent directory to path to import from scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from calculate_catalog_coordinates import convert_lightcone_to_radec


class TestCoordinateTransformation(unittest.TestCase):
    """Test coordinate conversion from lightcone to RA/Dec."""
    
    def test_identity_transformation(self):
        """Test that default parameters give expected results."""
        # At theta=0, phi=anything, we should be at Dec=90 (north pole)
        theta = np.array([0.0])
        phi = np.array([0.0])
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # At theta=0, we should be at the field center
        self.assertAlmostEqual(dec[0], 90.0, places=5)
    
    def test_theta_90_at_equator(self):
        """Test that theta=π/2 gives points at the equator when Dec0=90."""
        theta = np.array([np.pi/2])  # 90 degrees in radians
        phi = np.array([0.0])
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # At theta=π/2 from north pole, we should be at equator
        self.assertAlmostEqual(dec[0], 0.0, places=5)
        self.assertAlmostEqual(ra[0], 0.0, places=5)
    
    def test_phi_rotation(self):
        """Test that phi rotates around the field center."""
        theta_deg = 10.0
        theta = np.array([np.deg2rad(theta_deg)] * 4)
        phi = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])  # 0, 90, 180, 270 deg in radians
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # All should be at same declination (small circle around pole)
        np.testing.assert_array_almost_equal(dec, np.full(4, 90.0 - theta_deg), decimal=5)
        
        # RA should differ by 90 degrees
        expected_ra = np.array([0.0, 90.0, 180.0, 270.0])
        np.testing.assert_array_almost_equal(ra, expected_ra, decimal=4)
    
    def test_field_center_repositioning(self):
        """Test repositioning field center to different RA/Dec."""
        # Center of field (theta=0) should map to new field center
        theta = np.array([0.0])
        phi = np.array([0.0])
        
        # Reposition to RA=100, Dec=30
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=100.0, dec0=30.0, roll=0.0)
        
        self.assertAlmostEqual(ra[0], 100.0, places=4)
        self.assertAlmostEqual(dec[0], 30.0, places=4)
    
    def test_roll_angle(self):
        """Test that roll angle rotates the field."""
        # Point at phi=0, theta=10 degrees
        theta = np.array([np.deg2rad(10.0)])
        phi = np.array([0.0])
        
        # Without roll
        ra1, dec1 = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # With 90 degree roll
        ra2, dec2 = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=90.0)
        
        # Declination should be the same (roll doesn't change distance from center)
        self.assertAlmostEqual(dec1[0], dec2[0], places=4)
        
        # RA should differ by ~90 degrees
        ra_diff = (ra2[0] - ra1[0]) % 360
        self.assertAlmostEqual(ra_diff, 90.0, places=1)
    
    def test_array_shapes(self):
        """Test that function handles arrays correctly."""
        n = 100
        theta = np.random.uniform(0, np.deg2rad(5), n)  # 0 to 5 degrees in radians
        phi = np.random.uniform(0, 2*np.pi, n)  # 0 to 2π radians
        
        ra, dec = convert_lightcone_to_radec(theta, phi)
        
        self.assertEqual(ra.shape, (n,))
        self.assertEqual(dec.shape, (n,))
    
    def test_ra_range(self):
        """Test that RA is in [0, 360) range."""
        theta = np.random.uniform(0, np.deg2rad(10), 1000)  # 0 to 10 degrees in radians
        phi = np.random.uniform(0, 2*np.pi, 1000)  # 0 to 2π radians
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=180.0, dec0=0.0)
        
        self.assertTrue(np.all(ra >= 0))
        self.assertTrue(np.all(ra < 360))
    
    def test_dec_range(self):
        """Test that Dec is in [-90, 90] range."""
        theta = np.random.uniform(0, np.pi/2, 1000)  # 0 to 90 degrees in radians
        phi = np.random.uniform(0, 2*np.pi, 1000)  # 0 to 2π radians
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=0.0)
        
        self.assertTrue(np.all(dec >= -90))
        self.assertTrue(np.all(dec <= 90))
    
    def test_small_angles_precision(self):
        """Test precision for small angles near field center."""
        # Very small theta values in radians
        theta = np.array([0.001, 0.01, 0.1])  # radians (not degrees)
        phi = np.array([np.deg2rad(45.0)] * 3)  # 45 degrees in radians
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # All should be very close to Dec=90
        # theta=0.1 radians ≈ 5.7 degrees, so dec should be > 84
        self.assertTrue(np.all(dec > 84.0))
    
    def test_reversibility_concept(self):
        """Test that small perturbations from field center behave as expected."""
        # Create a small cone around field center
        theta = np.array([np.deg2rad(1.0)] * 4)  # 1 degree in radians
        phi = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])  # 0, 90, 180, 270 deg
        
        ra0, dec0 = 45.0, 30.0
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=ra0, dec0=dec0, roll=0.0)
        
        # Average position should be close to field center
        mean_ra = np.mean(ra)
        mean_dec = np.mean(dec)
        
        # Should be close to field center
        self.assertAlmostEqual(mean_ra, ra0, delta=0.5)
        self.assertAlmostEqual(mean_dec, dec0, delta=0.5)


class TestCoordinateSymmetry(unittest.TestCase):
    """Test symmetry properties of coordinate transformations."""
    
    def test_symmetry_around_poles(self):
        """Test that transformation is symmetric around field center."""
        # Points at equal theta but different phi
        theta = np.array([np.deg2rad(5.0)] * 4)  # 5 degrees in radians
        phi = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])  # 0, 90, 180, 270 deg
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # All should have same declination (circle of constant latitude)
        std_dec = np.std(dec)
        self.assertLess(std_dec, 1e-10)
    
    def test_opposite_rolls_symmetry(self):
        """Test that opposite roll angles produce expected symmetry."""
        theta = np.array([np.deg2rad(10.0)])
        phi = np.array([np.deg2rad(45.0)])
        
        ra1, dec1 = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=30.0)
        ra2, dec2 = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=-30.0)
        
        # Declinations should be the same
        self.assertAlmostEqual(dec1[0], dec2[0], places=10)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_theta_180(self):
        """Test theta=π (opposite side of sphere)."""
        theta = np.array([np.pi])  # 180 degrees in radians
        phi = np.array([0.0])
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # Should be at south pole
        self.assertAlmostEqual(dec[0], -90.0, places=5)
    
    def test_empty_arrays(self):
        """Test with empty input arrays."""
        theta = np.array([])
        phi = np.array([])
        
        ra, dec = convert_lightcone_to_radec(theta, phi)
        
        self.assertEqual(len(ra), 0)
        self.assertEqual(len(dec), 0)
    
    def test_single_point(self):
        """Test with single point."""
        theta = np.array([np.deg2rad(5.0)])
        phi = np.array([np.deg2rad(120.0)])
        
        ra, dec = convert_lightcone_to_radec(theta, phi)
        
        self.assertEqual(len(ra), 1)
        self.assertEqual(len(dec), 1)
        self.assertTrue(0 <= ra[0] < 360)
        self.assertTrue(-90 <= dec[0] <= 90)


if __name__ == '__main__':
    unittest.main()
