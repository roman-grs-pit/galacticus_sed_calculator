# Dust Attenuation Example

This example demonstrates how to apply dust attenuation to galaxy spectra using the `galacticus_sed_calculator` package.

## Overview

The dust attenuation feature allows you to model the effects of dust on emission line fluxes based on galaxy properties such as stellar mass and redshift, and to apply simple continuum attenuation. This implementation includes:

1. **GB10 Generalized Model**: A parametric model based on Garn & Best (2010) that calculates dust attenuation at H-alpha wavelength as a function of galaxy stellar mass and redshift.

2. **Calzetti Attenuation Law**: Describes how dust attenuation varies with wavelength for starburst galaxies.

3. **Fixed-A_V Continuum Model**: Applies a single Calzetti-law `A_V` to the continuum of every galaxy, e.g. `A_V=1.0`.

## Usage

Run the example:
```bash
python example_dust_attenuation.py
```

This will generate a comparison plot showing galaxy spectra with and without dust attenuation.

## Model Parameters

The GB10 model uses the following parameters:

```python
dust_params = {
    'delta_0': 0.275,      # Constant offset term
    'delta_z': -1.614,     # Redshift coefficient
    'delta_M': -0.834,     # Stellar mass coefficient  
    'delta_Mz': -0.708,    # Mass-redshift coupling coefficient
    'attenuation_scatter': 0.0  # Optional log-normal scatter
}
```

The attenuation at H-alpha is calculated as:
```
A(H-alpha) = delta_0 + delta_z * z + delta_M * log10(M*/M_sun) + delta_Mz * z * log10(M*/M_sun)
```

## Customization

You can customize the dust model by:
- Adjusting the `delta_*` parameters to match your calibration
- Adding scatter with `attenuation_scatter`
- Using different galaxy indices to see the effect on different galaxies
- Passing `continuum_dust={'model': 'fixed_av', 'params': {'A_V': 1.0}, 'law': 'calzetti'}` to attenuate the continuum

## References

- Garn, T., & Best, P. N. (2010). "The dust attenuation law in distant galaxies: evidence for variation with spectral type." MNRAS, 409, 421.
- Calzetti, D., et al. (2000). "The Dust Content and Opacity of Actively Star-forming Galaxies." ApJ, 533, 682.
