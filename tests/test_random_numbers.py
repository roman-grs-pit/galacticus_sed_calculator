"""
Tests for random number generation functionality.
"""
import unittest
import numpy as np
import h5py
import tempfile
import os
import sys

# Add parent directory to path to import from scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from calculate_catalog_random_numbers import (
    generate_random_numbers,
    calculate_catalog_random_numbers,
    save_random_numbers_to_galacticus_file,
    save_random_number_catalog
)


class TestRandomNumberGeneration(unittest.TestCase):
    """Test random number generation."""
    
    def test_basic_generation(self):
        """Test basic random number generation."""
        n_galaxies = 100
        n_random = 5
        
        random_numbers = generate_random_numbers(n_galaxies, n_random)
        
        # Check shape
        self.assertEqual(random_numbers.shape, (n_galaxies, n_random))
        
        # Check range [0, 1)
        self.assertTrue(np.all(random_numbers >= 0.0))
        self.assertTrue(np.all(random_numbers < 1.0))
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same random numbers."""
        n_galaxies = 50
        n_random = 5
        seed = 42
        
        random1 = generate_random_numbers(n_galaxies, n_random, seed=seed)
        random2 = generate_random_numbers(n_galaxies, n_random, seed=seed)
        
        np.testing.assert_array_equal(random1, random2)
    
    def test_different_seeds_differ(self):
        """Test that different seeds produce different random numbers."""
        n_galaxies = 50
        n_random = 5
        
        random1 = generate_random_numbers(n_galaxies, n_random, seed=42)
        random2 = generate_random_numbers(n_galaxies, n_random, seed=43)
        
        # Should be different (with very high probability)
        self.assertFalse(np.array_equal(random1, random2))
    
    def test_custom_n_random(self):
        """Test custom number of random values."""
        n_galaxies = 100
        n_random = 10
        
        random_numbers = generate_random_numbers(n_galaxies, n_random)
        
        self.assertEqual(random_numbers.shape, (n_galaxies, n_random))
    
    def test_distribution_uniformity(self):
        """Test that distribution is approximately uniform."""
        n_galaxies = 10000
        n_random = 5
        
        random_numbers = generate_random_numbers(n_galaxies, n_random, seed=42)
        
        # Check mean is close to 0.5
        mean = np.mean(random_numbers)
        self.assertAlmostEqual(mean, 0.5, delta=0.05)
        
        # Check values are spread across the range
        self.assertTrue(np.min(random_numbers) < 0.1)
        self.assertTrue(np.max(random_numbers) > 0.9)


class TestSaveToGalacticusFile(unittest.TestCase):
    """Test saving random numbers to Galacticus files."""
    
    def setUp(self):
        """Create a temporary HDF5 file with Galacticus structure."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.hdf5', delete=False)
        self.temp_file.close()
        
        # Create a mock Galacticus lightcone file
        np.random.seed(123)  # Use seed for deterministic test data
        with h5py.File(self.temp_file.name, 'w') as f:
            # Create lightcone structure
            lightcone = f.create_group('Lightcone')
            output1 = lightcone.create_group('Output1')
            node_data = output1.create_group('nodeData')
            
            # Add some dummy data
            n_galaxies = 100
            node_data.create_dataset('galaxyIndex', data=np.arange(n_galaxies))
            node_data.create_dataset('lightconeAngularTheta', 
                                    data=np.random.uniform(0, 0.1, n_galaxies))
            node_data.create_dataset('lightconeAngularPhi',
                                    data=np.random.uniform(0, 2*np.pi, n_galaxies))
    
    def tearDown(self):
        """Remove temporary file."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_save_random_numbers(self):
        """Test saving random numbers to Galacticus file."""
        n_galaxies = 100
        n_random = 5
        random_numbers = generate_random_numbers(n_galaxies, n_random, seed=42)
        
        output_path = '/Lightcone/Output1'
        node_data_path = f'{output_path}/nodeData'
        
        save_random_numbers_to_galacticus_file(
            self.temp_file.name, random_numbers, n_random, 42,
            output_path, node_data_path
        )
        
        # Check that data was saved
        with h5py.File(self.temp_file.name, 'r') as f:
            random_path = f'{node_data_path}/randomUniform'
            self.assertTrue(random_path in f)
            
            saved_data = f[random_path][:]
            np.testing.assert_array_equal(saved_data, random_numbers)
            
            # Check attributes
            self.assertEqual(f[random_path].attrs['n_random'], n_random)
            self.assertEqual(f[random_path].attrs['seed'], 42)
    
    def test_overwrite_existing(self):
        """Test that existing random numbers are overwritten."""
        n_galaxies = 100
        n_random = 5
        
        output_path = '/Lightcone/Output1'
        node_data_path = f'{output_path}/nodeData'
        
        # Save first set
        random1 = generate_random_numbers(n_galaxies, n_random, seed=42)
        save_random_numbers_to_galacticus_file(
            self.temp_file.name, random1, n_random, 42,
            output_path, node_data_path
        )
        
        # Save second set
        random2 = generate_random_numbers(n_galaxies, n_random, seed=43)
        save_random_numbers_to_galacticus_file(
            self.temp_file.name, random2, n_random, 43,
            output_path, node_data_path
        )
        
        # Check that second set is saved
        with h5py.File(self.temp_file.name, 'r') as f:
            random_path = f'{node_data_path}/randomUniform'
            saved_data = f[random_path][:]
            np.testing.assert_array_equal(saved_data, random2)
            self.assertEqual(f[random_path].attrs['seed'], 43)


class TestSaveToSeparateFile(unittest.TestCase):
    """Test saving random numbers to a separate catalog file."""
    
    def setUp(self):
        """Set up test."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.hdf5', delete=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Remove temporary file."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_save_lightcone_format(self):
        """Test saving random numbers for lightcone format."""
        n_galaxies = 100
        n_random = 5
        random_numbers = generate_random_numbers(n_galaxies, n_random, seed=42)
        
        results = {
            'random_numbers': random_numbers,
            'n_random': n_random,
            'seed': 42,
            'galaxy_indices': np.arange(n_galaxies),
            'format_type': 'lightcone',
            'base_path': '/Lightcone/Output1',
            'outputs_processed': ['/Lightcone/Output1']
        }
        
        save_random_number_catalog(results, self.temp_file.name)
        
        # Check saved data
        with h5py.File(self.temp_file.name, 'r') as f:
            self.assertTrue('randomUniform' in f)
            saved_data = f['randomUniform'][:]
            np.testing.assert_array_equal(saved_data, random_numbers)
            
            # Check attributes
            self.assertEqual(f.attrs['n_random'], n_random)
            self.assertEqual(f.attrs['seed'], 42)
            self.assertEqual(f.attrs['format_type'], 'lightcone')
    
    def test_save_fixed_time_format(self):
        """Test saving random numbers for fixed-time format with multiple outputs."""
        n_galaxies = 50
        n_random = 5
        
        # Multiple outputs
        random1 = generate_random_numbers(n_galaxies, n_random, seed=42)
        random2 = generate_random_numbers(n_galaxies, n_random, seed=42)
        
        results = {
            'random_numbers': [random1, random2],
            'n_random': n_random,
            'seed': 42,
            'galaxy_indices': np.arange(n_galaxies * 2),
            'format_type': 'fixed-time',
            'base_path': '/Outputs/Output1',
            'outputs_processed': ['/Outputs/Output1', '/Outputs/Output2']
        }
        
        save_random_number_catalog(results, self.temp_file.name)
        
        # Check saved data
        with h5py.File(self.temp_file.name, 'r') as f:
            self.assertTrue('Output1' in f)
            self.assertTrue('Output2' in f)
            
            saved_data1 = f['Output1/randomUniform'][:]
            saved_data2 = f['Output2/randomUniform'][:]
            
            np.testing.assert_array_equal(saved_data1, random1)
            np.testing.assert_array_equal(saved_data2, random2)
            
            # Check attributes
            self.assertEqual(f.attrs['n_random'], n_random)
            self.assertEqual(f.attrs['format_type'], 'fixed-time')


class TestFixedTimeFormat(unittest.TestCase):
    """Test handling of fixed-time format catalogs."""
    
    def setUp(self):
        """Create a temporary HDF5 file with fixed-time structure."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.hdf5', delete=False)
        self.temp_file.close()
        
        # Create a mock Galacticus fixed-time file
        np.random.seed(456)  # Use seed for deterministic test data
        with h5py.File(self.temp_file.name, 'w') as f:
            outputs = f.create_group('Outputs')
            
            # Create multiple outputs
            for i in [1, 2, 3]:
                output = outputs.create_group(f'Output{i}')
                node_data = output.create_group('nodeData')
                
                # Add some dummy data
                n_galaxies = 50
                node_data.create_dataset('galaxyIndex', data=np.arange(n_galaxies))
                node_data.create_dataset('diskStellarMass', 
                                        data=np.random.uniform(1e9, 1e11, n_galaxies))
                output.attrs['outputTime'] = 13.5 - i * 0.5
    
    def tearDown(self):
        """Remove temporary file."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_process_multiple_outputs(self):
        """Test processing multiple outputs in fixed-time format."""
        # This will process the file without saving (just return results)
        results = calculate_catalog_random_numbers(
            self.temp_file.name,
            n_random=5,
            seed=42,
            save_to_input=False,
            copy_input=False
        )
        
        self.assertIsNotNone(results)
        self.assertEqual(results['format_type'], 'fixed-time')
        self.assertEqual(len(results['outputs_processed']), 3)
        
        # Should have random numbers for all three outputs
        self.assertTrue(isinstance(results['random_numbers'], list))
        self.assertEqual(len(results['random_numbers']), 3)
        
        # Each output should have correct shape
        for random_numbers in results['random_numbers']:
            self.assertEqual(random_numbers.shape, (50, 5))


if __name__ == '__main__':
    unittest.main()
