"""
Tests for coordinate transformation functionality.
"""
import unittest
import numpy as np
import h5py
import tempfile
import os
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
        """Test that theta=90 gives points at the equator when Dec0=90."""
        theta = np.array([90.0])
        phi = np.array([0.0])
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # At theta=90 from north pole, we should be at equator
        self.assertAlmostEqual(dec[0], 0.0, places=5)
        self.assertAlmostEqual(ra[0], 0.0, places=5)
    
    def test_phi_rotation(self):
        """Test that phi rotates around the field center."""
        theta = np.array([10.0, 10.0, 10.0, 10.0])
        phi = np.array([0.0, 90.0, 180.0, 270.0])
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # All should be at same declination (small circle around pole)
        np.testing.assert_array_almost_equal(dec, np.full(4, 90.0 - 10.0), decimal=5)
        
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
        # Point at phi=0, theta=10
        theta = np.array([10.0])
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
        theta = np.random.uniform(0, 5, n)
        phi = np.random.uniform(0, 360, n)
        
        ra, dec = convert_lightcone_to_radec(theta, phi)
        
        self.assertEqual(ra.shape, (n,))
        self.assertEqual(dec.shape, (n,))
    
    def test_ra_range(self):
        """Test that RA is in [0, 360) range."""
        theta = np.random.uniform(0, 10, 1000)
        phi = np.random.uniform(0, 360, 1000)
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=180.0, dec0=0.0)
        
        self.assertTrue(np.all(ra >= 0))
        self.assertTrue(np.all(ra < 360))
    
    def test_dec_range(self):
        """Test that Dec is in [-90, 90] range."""
        theta = np.random.uniform(0, 90, 1000)
        phi = np.random.uniform(0, 360, 1000)
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=0.0)
        
        self.assertTrue(np.all(dec >= -90))
        self.assertTrue(np.all(dec <= 90))
    
    def test_small_angles_precision(self):
        """Test precision for small angles near field center."""
        # Very small theta values
        theta = np.array([0.001, 0.01, 0.1])
        phi = np.array([45.0, 45.0, 45.0])
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # All should be very close to Dec=90
        # theta=0.1 degrees means we're 0.1 degrees from pole, so dec should be > 89.9
        self.assertTrue(np.all(dec > 89.8))
    
    def test_reversibility_concept(self):
        """Test that small perturbations from field center behave as expected."""
        # Create a small cone around field center
        theta = np.array([1.0, 1.0, 1.0, 1.0])
        phi = np.array([0.0, 90.0, 180.0, 270.0])
        
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
        theta = np.array([5.0, 5.0, 5.0, 5.0])
        phi = np.array([0.0, 90.0, 180.0, 270.0])
        
        ra, dec = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=0.0)
        
        # All should have same declination (circle of constant latitude)
        std_dec = np.std(dec)
        self.assertLess(std_dec, 1e-10)
    
    def test_opposite_rolls_symmetry(self):
        """Test that opposite roll angles produce expected symmetry."""
        theta = np.array([10.0])
        phi = np.array([45.0])
        
        ra1, dec1 = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=30.0)
        ra2, dec2 = convert_lightcone_to_radec(theta, phi, ra0=0.0, dec0=90.0, roll=-30.0)
        
        # Declinations should be the same
        self.assertAlmostEqual(dec1[0], dec2[0], places=10)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_theta_180(self):
        """Test theta=180 (opposite side of sphere)."""
        theta = np.array([180.0])
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
        theta = np.array([5.0])
        phi = np.array([120.0])
        
        ra, dec = convert_lightcone_to_radec(theta, phi)
        
        self.assertEqual(len(ra), 1)
        self.assertEqual(len(dec), 1)
        self.assertTrue(0 <= ra[0] < 360)
        self.assertTrue(-90 <= dec[0] <= 90)


if __name__ == '__main__':
    unittest.main()
