# galacticus_sed_calculator
Classes and methods to generate spectra for galaxies simulated with Galacticus, based on a saved star formation history.

Currently work in progress, but how to use it is hopefully demonstrated in [exampleUsage.ipynb](exampleUsage.ipynb).

## Magnitude Calculation

This repository now includes comprehensive documentation on calculating observed magnitudes in photometric bands using `synphot`:

- **[MAGNITUDE_CALCULATION_SUMMARY.md](MAGNITUDE_CALCULATION_SUMMARY.md)** - Quick summary of findings and recommendations
- **[SYNPHOT_MAGNITUDE_GUIDE.md](SYNPHOT_MAGNITUDE_GUIDE.md)** - Detailed technical guide with implementation recommendations
- **[demo_magnitude_calculation.py](demo_magnitude_calculation.py)** - Standalone demonstrations of synphot capabilities
- **[example_galacticus_magnitudes.py](example_galacticus_magnitudes.py)** - Examples showing integration with Galacticus data

These documents show how to calculate observed magnitudes (AB, ST, Vega) in Roman WFI filters for individual galaxies or entire catalogs. 
