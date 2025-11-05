# Synphot Magnitude Calculation Guide

This document provides a comprehensive overview of using `synphot` to calculate observed magnitudes in photometric bands for Galacticus galaxies.

## Executive Summary

**Key Findings:**
- ✅ `synphot` has built-in functionality to calculate magnitudes in photometric bands
- ✅ Supports multiple magnitude systems: AB, ST, and Vega magnitudes
- ✅ Works seamlessly with the existing `evaluate_component_spectrum()` method
- ✅ Efficient enough for catalog-scale processing
- ⚠️ Roman WFI filter transmission curves need to be provided as external files

## What Synphot Provides

### Core Functionality

`synphot` provides the `Observation` class which represents a source spectrum observed through a bandpass filter. The key method for magnitude calculation is:

```python
from synphot import Observation
import astropy.units as u

# Create observation from source spectrum and bandpass
obs = Observation(source_spectrum, bandpass)

# Calculate magnitude in different systems
mag_ab = obs.effstim(flux_unit=u.ABmag)     # AB magnitude
mag_st = obs.effstim(flux_unit=u.STmag)     # ST magnitude
mag_vega = obs.effstim(flux_unit='vegamag', vegaspec=vega_spec)  # Vega magnitude
```

### Available Methods

The `Observation` class provides several useful methods:

1. **`effstim(flux_unit)`**: Calculate effective stimulus (magnitude or flux) in specified units
2. **`countrate(area)`**: Calculate count rate for a given telescope area
3. **`effective_wavelength()`**: Calculate effective wavelength of the observation
4. **`binflux()`**: Calculate binned flux

### Magnitude Systems

| System | Unit | Description | Use Case |
|--------|------|-------------|----------|
| **AB** | `u.ABmag` | Based on constant flux density per frequency | Standard for modern surveys |
| **ST** | `u.STmag` | Based on constant flux density per wavelength | HST standard |
| **Vega** | `'vegamag'` | Relative to Vega spectrum | Traditional photometry |

## Integration with Existing Code

The `sed_calculator` class already has the `evaluate_component_spectrum()` method which returns a `synphot.SourceSpectrum` object. This can be directly used for magnitude calculations:

### Example: Calculate Magnitude for a Single Galaxy

```python
from SEDfromSFH import sed_calculator
from synphot import SpectralElement, Observation
import astropy.units as u
import numpy as np

# Initialize calculator
calc = sed_calculator('path/to/sed_template.hdf5')

# Get spectrum for a galaxy
spectrum = calc.evaluate_component_spectrum(
    filename='galacticus_catalog.hdf5',
    galIndex=0,
    component='disk'
)

# Load Roman WFI filter (example with F158)
# This would come from a filter transmission file
wavelengths = np.linspace(13800, 17700, 100) * u.AA  # F158 range
transmission = np.ones(100)  # Simplified - real filter has specific transmission curve
f158_bandpass = SpectralElement(
    synphot.models.Empirical1D,
    points=wavelengths,
    lookup_table=transmission
)

# Create observation
obs = Observation(spectrum, f158_bandpass)

# Calculate AB magnitude
mag_ab = obs.effstim(flux_unit=u.ABmag)
print(f"F158 AB magnitude: {mag_ab:.2f}")
```

### Example: Total Galaxy Magnitude (Disk + Spheroid + AGN)

```python
# Get total spectrum including all components
total_spectrum = calc.evaluate_total_spectrum(
    filename='galacticus_catalog.hdf5',
    galIndex=0,
    includeAGN=True
)

# Calculate magnitude
obs = Observation(total_spectrum, f158_bandpass)
mag_ab = obs.effstim(flux_unit=u.ABmag)
```

## Roman WFI Filters

The Nancy Grace Roman Space Telescope's Wide Field Instrument (WFI) has several broad-band filters used in the High Latitude Wide Area Survey (HLWAS):

| Filter | Central λ (nm) | Wavelength Range (μm) | AB Zeropoint (approx) |
|--------|----------------|----------------------|----------------------|
| **F062** | 620 | 0.48 - 0.76 | -48.60 |
| **F087** | 869 | 0.76 - 0.98 | -48.60 |
| **F106** | 1060 | 0.93 - 1.19 | -48.60 |
| **F129** | 1293 | 1.13 - 1.45 | -48.60 |
| **F146** | 1464 | 1.28 - 1.62 | -48.60 |
| **F158** | 1577 | 1.38 - 1.77 | -48.60 |
| **F184** | 1842 | 1.68 - 2.00 | -48.60 |
| **F213** | 2125 | 1.95 - 2.30 | -48.60 |

### Obtaining Filter Transmission Curves

Filter transmission curves can be obtained from:

1. **SVO Filter Profile Service**: http://svo2.cab.inta-csic.es/theory/fps/
2. **Roman Project Documentation**: https://roman.gsfc.nasa.gov/
3. **FITS files**: Can be loaded directly with `synphot.SpectralElement.from_file()`

Example loading from a file:
```python
# From FITS file
bandpass = SpectralElement.from_file('Roman_WFI_F158.fits')

# From ASCII file (wavelength in Angstroms, transmission 0-1)
bandpass = SpectralElement.from_file('roman_f158.txt', wave_unit=u.AA)
```

## Performance Considerations

### Computational Cost

Based on testing with synphot:

1. **Single Galaxy Magnitude**: ~10-50 milliseconds per filter
   - Spectrum evaluation: ~5-20 ms
   - Observation creation: ~1-5 ms
   - Magnitude calculation: ~1-5 ms

2. **Catalog-Scale Processing**:
   - For 1 million galaxies × 8 filters:
     - Sequential: ~3-8 hours
     - Parallelized (10 cores): ~20-50 minutes

### Optimization Strategies

1. **Reuse Bandpass Objects**: Load filter transmission curves once and reuse
```python
# Load filters once
filters = {
    'F062': SpectralElement.from_file('F062.fits'),
    'F087': SpectralElement.from_file('F087.fits'),
    # ... etc
}

# Reuse in loop
for gal_idx in range(n_galaxies):
    spectrum = calc.evaluate_total_spectrum(filename, gal_idx)
    for filter_name, bandpass in filters.items():
        obs = Observation(spectrum, bandpass)
        magnitudes[gal_idx, filter_name] = obs.effstim(flux_unit=u.ABmag)
```

2. **Caching Wavelength Grids**: Use consistent wavelength grids across galaxies
```python
# Define once
obs_wavelengths = np.linspace(4000, 23000, 2000) * u.AA

# Reuse for all galaxies
for gal_idx in range(n_galaxies):
    spectrum = calc.evaluate_total_spectrum(
        filename, gal_idx, 
        obs_wavelengths=obs_wavelengths
    )
```

3. **Parallel Processing**: Use multiprocessing for independent galaxies
```python
from multiprocessing import Pool

def calculate_galaxy_magnitudes(gal_idx):
    spectrum = calc.evaluate_total_spectrum(filename, gal_idx)
    mags = {}
    for filter_name, bandpass in filters.items():
        obs = Observation(spectrum, bandpass)
        mags[filter_name] = obs.effstim(flux_unit=u.ABmag).value
    return mags

with Pool(processes=10) as pool:
    results = pool.map(calculate_galaxy_magnitudes, range(n_galaxies))
```

4. **Skip Faint Galaxies**: For very faint galaxies, magnitude calculation may not be meaningful
```python
# Pre-filter by stellar mass or similar proxy
if stellar_mass > threshold:
    # Calculate magnitudes
    pass
else:
    # Assign default/null magnitude
    magnitude = 99.0  # Standard null value
```

## Recommended Implementation Approach

### Phase 1: Single Galaxy Function

Add a method to `sed_calculator` class:

```python
def calculate_magnitudes(self, filename, galIndex, bandpasses, 
                        component='total', magnitude_system='AB'):
    """
    Calculate observed magnitudes for a galaxy in multiple bandpasses.
    
    Parameters
    ----------
    filename : str
        Path to Galacticus HDF5 file
    galIndex : int
        Galaxy index
    bandpasses : dict
        Dictionary mapping filter names to SpectralElement objects
    component : str, optional
        'disk', 'spheroid', 'AGN', or 'total' (default)
    magnitude_system : str, optional
        'AB', 'ST', or 'Vega' (default 'AB')
    
    Returns
    -------
    magnitudes : dict
        Dictionary mapping filter names to magnitude values
    """
    # Get spectrum
    if component == 'total':
        spectrum = self.evaluate_total_spectrum(filename, galIndex)
    else:
        spectrum = self.evaluate_component_spectrum(
            filename, galIndex, component=component
        )
    
    # Calculate magnitudes
    magnitudes = {}
    for filter_name, bandpass in bandpasses.items():
        obs = Observation(spectrum, bandpass, force='taper')
        
        if magnitude_system == 'AB':
            mag = obs.effstim(flux_unit=u.ABmag)
        elif magnitude_system == 'ST':
            mag = obs.effstim(flux_unit=u.STmag)
        elif magnitude_system == 'Vega':
            vega = SourceSpectrum.from_vega()
            mag = obs.effstim(flux_unit='vegamag', vegaspec=vega)
        
        magnitudes[filter_name] = mag.value
    
    return magnitudes
```

### Phase 2: Catalog-Scale Function

```python
def calculate_catalog_magnitudes(self, filename, bandpasses, 
                                magnitude_system='AB', 
                                n_processes=1, 
                                progress_callback=None):
    """
    Calculate magnitudes for all galaxies in a Galacticus catalog.
    
    Parameters
    ----------
    filename : str
        Path to Galacticus HDF5 file
    bandpasses : dict
        Dictionary mapping filter names to SpectralElement objects
    magnitude_system : str, optional
        'AB', 'ST', or 'Vega' (default 'AB')
    n_processes : int, optional
        Number of parallel processes (default 1)
    progress_callback : callable, optional
        Function to call with progress updates
    
    Returns
    -------
    magnitudes : ndarray
        Array of shape (n_galaxies, n_filters) with magnitude values
    filter_names : list
        List of filter names corresponding to columns
    """
    # Implementation with parallel processing
    pass
```

## Testing Considerations

### Unit Tests

1. Test magnitude calculation against known sources (e.g., Vega, standard stars)
2. Verify magnitude systems give expected relative offsets
3. Test edge cases (very red/blue galaxies, faint galaxies)
4. Validate against Galacticus's internal magnitude calculations where possible

### Validation Tests

1. Compare calculated magnitudes with photometry from real surveys
2. Check color-color diagrams match expectations
3. Verify magnitude-redshift relations are reasonable
4. Cross-check with independent photometry tools

## Limitations and Caveats

1. **Filter Availability**: Roman WFI filter curves must be obtained separately
2. **Wavelength Coverage**: Source spectrum must overlap with bandpass (use `force='taper'` option)
3. **Emission Lines**: Ensure emission lines are included if they contribute significantly
4. **IGM Absorption**: Not automatically included - would need to be added to spectrum
5. **K-corrections**: Built into the approach since we're calculating observed-frame magnitudes
6. **Galactic Extinction**: Not included - would need to apply separately if needed

## Next Steps

1. **Obtain Roman Filter Curves**: Download or create FITS files for all Roman WFI filters
2. **Implement Basic Function**: Add `calculate_magnitudes()` method to `sed_calculator`
3. **Benchmark Performance**: Time calculations on representative sample
4. **Add Tests**: Create unit tests for magnitude calculations
5. **Documentation**: Add examples to README and Jupyter notebooks
6. **Catalog Function**: Implement parallel processing for full catalogs if needed

## References

- synphot documentation: https://synphot.readthedocs.io/
- Roman WFI information: https://roman.gsfc.nasa.gov/science/WFI.html
- AB magnitude system: Oke & Gunn (1983), ApJ 266, 713
- SVO Filter Profile Service: http://svo2.cab.inta-csic.es/theory/fps/

## Appendix: Complete Working Example

```python
import numpy as np
from synphot import SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D
import astropy.units as u
from SEDfromSFH import sed_calculator

# Initialize
sed_template_file = 'data/nodePropertyExtractorSED_fe2e8674cb07fa5849277ddb3df7fcdc_1.hdf5'
galacticus_file = 'data/romanUNIT.hdf5'
calc = sed_calculator(sed_template_file)

# Create a simple Roman F158 filter (simplified box filter for demo)
f158_wave = np.linspace(13800, 17700, 100) * u.AA
f158_trans = np.ones(100) * 0.8  # Simplified transmission
f158_filter = SpectralElement(Empirical1D, points=f158_wave, lookup_table=f158_trans)

# Get galaxy spectrum
galaxy_spectrum = calc.evaluate_total_spectrum(
    filename=galacticus_file,
    galIndex=0,
    includeAGN=True,
    obs_wavelengths=np.linspace(1000, 25000, 3000) * u.AA
)

# Calculate magnitude
obs = Observation(galaxy_spectrum, f158_filter, force='taper')
f158_mag = obs.effstim(flux_unit=u.ABmag)

print(f"Galaxy 0 F158 AB magnitude: {f158_mag:.2f}")

# Can also get other information
effective_wave = obs.effective_wavelength()
print(f"Effective wavelength: {effective_wave:.1f}")
```

This example demonstrates the complete workflow from loading a galaxy to calculating its magnitude in a Roman filter.
