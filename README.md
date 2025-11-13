# galacticus_sed_calculator
Classes and methods to generate spectra for galaxies simulated with Galacticus, based on a saved star formation history.

## Features

- **Automatic format detection**: Supports both lightcone and fixed-time Galacticus output formats
- **SED calculation**: Generate rest-frame and observed-frame SEDs from star formation histories
- **Spectrum generation**: Create full spectra including continuum and emission lines
- **Magnitude calculation**: Calculate observed magnitudes in arbitrary bandpasses
- **Cross-validation**: Compare results from different Galacticus output formats

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

For usage examples, see [exampleUsage.ipynb](exampleUsage.ipynb).

### Basic Example

```python
from SEDfromSFH import sed_calculator
import numpy as np
import astropy.units as u

# Initialize calculator with SED template
calc = sed_calculator('sed_template.hdf5')

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

## Testing

Run tests with pytest:
```bash
pytest test_SEDfromSFH.py
```

## Implementation Details

The fixed-time format support includes:
- Automatic conversion from age-of-universe to lookback times
- Cosmological redshift calculation from output time
- Format-specific path handling
- Caching of detected formats for performance

Both formats produce identical results for the same galaxy data. 
