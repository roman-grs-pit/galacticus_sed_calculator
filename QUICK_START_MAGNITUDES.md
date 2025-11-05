# Quick Start: Magnitude Calculation

This is a quick reference for calculating observed magnitudes using synphot with Galacticus data.

## Installation

```bash
pip install synphot stsynphot h5py astropy
```

## Basic Example

```python
from SEDfromSFH import sed_calculator
from synphot import SpectralElement, Observation
import astropy.units as u
import numpy as np

# 1. Initialize calculator
calc = sed_calculator('path/to/sed_template.hdf5')

# 2. Get galaxy spectrum
spectrum = calc.evaluate_total_spectrum(
    filename='galacticus_catalog.hdf5',
    galIndex=0,
    obs_wavelengths=np.linspace(4000, 23000, 2000) * u.AA
)

# 3. Load bandpass filter (example: create simple filter)
wavelengths = np.linspace(14000, 17000, 200) * u.AA
transmission = np.ones(200) * 0.8  # 80% transmission
bandpass = SpectralElement(
    synphot.models.Empirical1D,
    points=wavelengths,
    lookup_table=transmission
)

# 4. Calculate magnitude
obs = Observation(spectrum, bandpass, force='taper')
magnitude = obs.effstim(flux_unit=u.ABmag)

print(f"AB magnitude: {magnitude:.2f}")
```

## Loading Real Filter Files

```python
# From FITS file
bandpass = SpectralElement.from_file('Roman_WFI_F158.fits')

# From ASCII file (wavelength, transmission columns)
bandpass = SpectralElement.from_file('filter.txt', wave_unit=u.AA)
```

## Multiple Filters

```python
# Define filters
filters = {
    'F087': SpectralElement.from_file('F087.fits'),
    'F129': SpectralElement.from_file('F129.fits'),
    'F158': SpectralElement.from_file('F158.fits'),
}

# Calculate magnitudes
spectrum = calc.evaluate_total_spectrum(filename, galIndex)
magnitudes = {}
for name, bandpass in filters.items():
    obs = Observation(spectrum, bandpass, force='taper')
    magnitudes[name] = obs.effstim(flux_unit=u.ABmag).value
```

## Magnitude Systems

```python
# AB magnitude (default for modern surveys)
mag_ab = obs.effstim(flux_unit=u.ABmag)

# ST magnitude (HST standard)
mag_st = obs.effstim(flux_unit=u.STmag)

# Vega magnitude (requires Vega spectrum)
from synphot import SourceSpectrum
vega = SourceSpectrum.from_vega()
mag_vega = obs.effstim(flux_unit='vegamag', vegaspec=vega)
```

## Component Magnitudes

```python
# Disk only
disk_spectrum = calc.evaluate_component_spectrum(
    filename, galIndex, component='disk'
)

# Spheroid only
spheroid_spectrum = calc.evaluate_component_spectrum(
    filename, galIndex, component='spheroid'
)

# Total (disk + spheroid + AGN)
total_spectrum = calc.evaluate_total_spectrum(
    filename, galIndex, includeAGN=True
)
```

## Performance Tips

1. **Reuse bandpass objects** - Load once, use many times
2. **Use consistent wavelength grids** - Define once for all galaxies
3. **Parallel processing** - Use multiprocessing for large catalogs
4. **Pre-filter galaxies** - Skip very faint galaxies if not needed

## Roman WFI Filters

Filter data can be obtained from:
- SVO Filter Profile Service: http://svo2.cab.inta-csic.es/theory/fps/
- Roman Project: https://roman.gsfc.nasa.gov/

Roman WFI broad-band filters:
- F062 (0.48-0.76 μm)
- F087 (0.76-0.98 μm)
- F106 (0.93-1.19 μm)
- F129 (1.13-1.45 μm)
- F146 (1.28-1.62 μm) - prism
- F158 (1.38-1.77 μm)
- F184 (1.68-2.00 μm)
- F213 (1.95-2.30 μm)

## More Information

- **MAGNITUDE_CALCULATION_SUMMARY.md** - Quick summary and recommendations
- **SYNPHOT_MAGNITUDE_GUIDE.md** - Comprehensive technical guide
- **demo_magnitude_calculation.py** - Run `python3 demo_magnitude_calculation.py`
- **example_galacticus_magnitudes.py** - Run `python3 example_galacticus_magnitudes.py`
- **test_magnitude_calculation.py** - Run `pytest test_magnitude_calculation.py`

## Typical Performance

- Single galaxy, single filter: ~10-15 ms
- Single galaxy, 8 Roman filters: ~100 ms
- 1,000 galaxies (sequential): ~2 minutes
- 1,000 galaxies (10 cores): ~13 seconds
