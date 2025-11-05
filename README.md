# galacticus_sed_calculator
Classes and methods to generate spectra for galaxies simulated with Galacticus, based on a saved star formation history.

Currently work in progress, but how to use it is hopefully demonstrated in [exampleUsage.ipynb](exampleUsage.ipynb).

## Magnitude Calculation

This repository now includes comprehensive documentation on calculating observed magnitudes in photometric bands using `synphot`:

- **[QUICK_START_MAGNITUDES.md](QUICK_START_MAGNITUDES.md)** - ⭐ Start here! Quick reference with code examples
- **[MAGNITUDE_CALCULATION_SUMMARY.md](MAGNITUDE_CALCULATION_SUMMARY.md)** - Executive summary of findings and recommendations
- **[SYNPHOT_MAGNITUDE_GUIDE.md](SYNPHOT_MAGNITUDE_GUIDE.md)** - Comprehensive technical guide with implementation details
- **[demo_magnitude_calculation.py](demo_magnitude_calculation.py)** - Standalone demonstrations of synphot capabilities
- **[example_galacticus_magnitudes.py](example_galacticus_magnitudes.py)** - Examples showing integration with Galacticus data
- **[test_magnitude_calculation.py](test_magnitude_calculation.py)** - Unit tests (run with `pytest`)

These documents show how to calculate observed magnitudes (AB, ST, Vega) in Roman WFI filters for individual galaxies or entire catalogs. 
