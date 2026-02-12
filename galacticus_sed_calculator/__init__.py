"""
galacticus_sed_calculator

A Python package for generating spectra for galaxies simulated with Galacticus,
based on saved star formation histories.
"""

from .sed_calculator import SEDCalculator

# Backward compatibility alias
sed_calculator = SEDCalculator

__version__ = "0.1.0"
__all__ = ["SEDCalculator", "sed_calculator"]
