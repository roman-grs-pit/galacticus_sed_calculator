"""
galacticus_sed_calculator

A Python package for generating spectra for galaxies simulated with Galacticus,
based on saved star formation histories.
"""

from .sed_calculator import sed_calculator

__version__ = "0.1.0"
__all__ = ["sed_calculator"]
