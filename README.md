# galacticus_sed_calculator
Classes and methods to generate spectra for galaxies simulated with Galacticus, based on a saved star formation history.

## Installation

Install the package in editable mode for development:

```bash
pip install -e .
```

Or install with optional dependencies:

```bash
# For running the scripts
pip install -e ".[scripts]"

# For development and testing
pip install -e ".[dev]"
```

## Repository Structure

```
galacticus_sed_calculator/
├── galacticus_sed_calculator/ # Main package directory
│   ├── __init__.py            # Package initialization
│   └── sed_calculator.py      # SED calculator module
├── scripts/                   # Command-line utilities
│   ├── calculate_catalog_coordinates.py
│   ├── calculate_catalog_magnitudes.py
│   └── calculate_catalog_random_numbers.py
├── tests/                     # Unit tests
├── examples/                  # Example scripts and notebooks
├── profiling/                 # Performance profiling tools
├── docs/                      # Documentation
└── data/                      # Test data files
```

## Features

- **Automatic format detection**: Supports both lightcone and fixed-time Galacticus output formats
- **SED calculation**: Generate rest-frame and observed-frame SEDs from star formation histories
- **Spectrum generation**: Create full spectra including continuum and emission lines
- **Magnitude calculation**: Calculate observed magnitudes in arbitrary bandpasses
- **Coordinate transformation**: Convert lightcone angular positions to RA/Dec with field repositioning
- **Random number generation**: Generate uniform random numbers for stochastic galaxy properties
- **Cross-validation**: Compare results from different Galacticus output formats
- **Performance profiling**: Tools to analyze and optimize SED generation performance

## Supported Galacticus Output Formats

### Lightcone Format
- Path structure: `/Lightcone/Output1/nodeData/...`
- Each galaxy has unique redshift
- Star formation histories stored with per-galaxy lookback times (stellar ages)
- Time and metallicity bins vary per galaxy

### Fixed-Time Format (New!)
- Path structure: `/Outputs/Output1/nodeData/...` (or Output2, Output3, etc.)
- All galaxies output at same cosmic time
- Star formation histories stored with age-of-universe times
- Single `outputTime` attribute defines observation epoch
- Time and metallicity bins stored as dataset attributes (uniform across galaxies)

The format is automatically detected - no configuration needed!

## Usage

For usage examples, see [exampleUsage.ipynb](examples/exampleUsage.ipynb).

### Basic Example

```python
from galacticus_sed_calculator import SEDCalculator
import numpy as np
import astropy.units as u

# Initialize calculator with SED template
calc = SEDCalculator('sed_template.hdf5')

# Works with both lightcone and fixed-time formats!
galData = calc.read_galacticus_galaxy('galacticus_output.hdf5', galIndex=0)

# Calculate flux density
wavelengths = np.linspace(10000, 20000, 100) * u.AA
Fnu, wav = calc.calculate_continuum_Fnu(
    galData['diskSFH'], 
    galData['redshift'],
    wavelengths=wavelengths
)
```

### Fast SED Generation (2.3x Speedup)

For improved performance when generating many spectra, use the `use_synphot=False` option:

```python
import numpy as np
import astropy.units as u

# Use a high-resolution wavelength grid to resolve emission lines
wavelengths = np.linspace(8000, 30000, 2000) * u.AA

# Fast path: returns (wavelength, flux_density) tuple instead of SourceSpectrum
wav, flux = calc.evaluate_component_spectrum(
    'galacticus_output.hdf5',
    galIndex=0,
    component='disk',
    obs_wavelengths=wavelengths,
    include_emission_lines=True,
    use_synphot=False  # Enable fast path
)

# flux is in units of Lsun / (Hz Mpc^2)
# This is ~2.3x faster than the default synphot approach
```

**Note**: When `use_synphot=False`:
- You must provide `obs_wavelengths` 
- Returns numpy arrays instead of synphot SourceSpectrum objects
- Results are numerically identical to the synphot approach
- Best for batch processing many galaxies

### Calculate Magnitudes

```python
# Generate full spectrum (continuum + emission lines)
spectrum = calc.evaluate_component_spectrum(
    'galacticus_output.hdf5',
    galIndex=0,
    component='disk'
)

# Calculate magnitudes in specific bandpasses
import stpsf
roman = stpsf.WFI()
bandpasses = {
    'F158': roman._get_synphot_bandpass('F158'),
    'F184': roman._get_synphot_bandpass('F184')
}

magnitudes = calc.calculate_magnitudes(
    'galacticus_output.hdf5',
    galIndex=0,
    bandpasses=bandpasses,
    component='total'
)
```

### Calculate RA and Dec Coordinates

For lightcone catalogs, convert the angular positions (theta, phi) to astronomical RA and Dec coordinates:

**Note**: Galacticus lightcone coordinates (`lightconeAngularTheta` and `lightconeAngularPhi`) are stored in **radians**.

```bash
# Basic usage - saves to new file with '_with_coordinates' suffix
python scripts/calculate_catalog_coordinates.py galacticus_lightcone.hdf5 \
    --ra0 150 --dec0 30 --roll 0

# Save to separate file instead
python scripts/calculate_catalog_coordinates.py galacticus_lightcone.hdf5 \
    --ra0 150 --dec0 30 --roll 0 \
    --save-to-file coordinates.hdf5

# Reposition field center to different location
python scripts/calculate_catalog_coordinates.py galacticus_lightcone.hdf5 \
    --ra0 45.5 --dec0 -12.3 --roll 15
```

This adds `rightAscension` and `declination` datasets to the catalog with the following features:
- Field center repositioning: Place the cone center at any (RA, Dec) on the sky
- Roll angle: Rotate the field around the line of sight
- Preserves original theta/phi values (in radians)
- Records transformation parameters in dataset attributes
- Outputs RA/Dec in degrees

The transformation process:
1. Reads (theta, phi) polar coordinates in radians from Galacticus
2. Converts to Cartesian on unit sphere
3. Applies roll rotation around line of sight
4. Rotates to reposition field center from (RA=0°, Dec=90°) to (RA0, Dec0)
5. Converts back to (RA, Dec) spherical coordinates in degrees

### Generate Random Numbers for Galaxies

Generate uniform random numbers [0, 1) for each galaxy, useful for stochastic properties like dust attenuation scatter or galaxy inclinations. Works with both lightcone and fixed-time formats:

```bash
# Basic usage - saves to new file with '_with_random' suffix
python scripts/calculate_catalog_random_numbers.py galacticus_catalog.hdf5

# Generate 10 random numbers per galaxy with specific seed
python scripts/calculate_catalog_random_numbers.py galacticus_catalog.hdf5 \
    --n-random 10 --seed 42

# Save to separate file instead
python scripts/calculate_catalog_random_numbers.py galacticus_catalog.hdf5 \
    --save-to-file random_numbers.hdf5

# Test with limited galaxies
python scripts/calculate_catalog_random_numbers.py galacticus_catalog.hdf5 \
    --max-galaxies 100
```

This adds `randomUniform` datasets to the catalog with the following features:
- Generates 5 random numbers per galaxy by default (configurable with `--n-random`)
- Random numbers uniformly distributed in [0, 1)
- Optional seed for reproducibility (`--seed`)
- For fixed-time catalogs: generates random numbers at each output time
- Saves to a copy of the input file by default (with `_with_random` suffix)
- Can save to a separate file with `--save-to-file`
- Records seed and number of random values in dataset attributes
- Dataset shape: (n_galaxies, n_random)

Example usage in Python:
```python
import h5py

with h5py.File('galacticus_catalog_with_random.hdf5', 'r') as f:
    # For lightcone format
    random_numbers = f['/Lightcone/Output1/nodeData/randomUniform'][:]
    # random_numbers has shape (n_galaxies, 5) by default
    
    # For fixed-time format
    random_numbers = f['/Outputs/Output1/nodeData/randomUniform'][:]
    # Each output has its own random numbers
```

## Testing

Run tests with unittest:
```bash
python -m unittest tests.test_SEDfromSFH
python -m unittest tests.test_coordinates
python -m unittest tests.test_random_numbers
```

## Performance Profiling

The repository includes comprehensive profiling tools to analyze SED generation performance. See [docs/PROFILING.md](docs/PROFILING.md) for detailed documentation.

Quick start:
```bash
# Profile 100 galaxies
python profiling/profile_sed_generation.py --num-galaxies 100

# Analyze component-level performance
python profiling/profile_sed_generation.py --component-profile --galaxy-index 8

# Detailed function-level profiling
python profiling/profile_sed_generation.py --num-galaxies 50 --detailed-profile --output-file results.txt
```

Current performance: ~130 ms per galaxy (faster than 0.2 s/galaxy target).

### Fast SED Generation Option

For even better performance, use `use_synphot=False` in `evaluate_component_spectrum()`:
```bash
# Compare performance between synphot and fast path
python tests/test_performance_comparison.py
```

This approach is **2.3x faster** (10.4 ms vs 24.2 ms per galaxy) while producing numerically identical results.
Use this option when generating spectra for many galaxies where you need wavelength arrays rather than synphot SourceSpectrum objects.

## Implementation Details

The fixed-time format support includes:
- Automatic conversion from age-of-universe to lookback times
- Cosmological redshift calculation from output time
- Format-specific path handling
- Caching of detected formats for performance

Both formats produce identical results for the same galaxy data. 
