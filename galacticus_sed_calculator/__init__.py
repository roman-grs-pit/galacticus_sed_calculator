"""
galacticus_sed_calculator

A Python package for generating spectra for galaxies simulated with Galacticus,
based on saved star formation histories.
"""

from .sed_calculator import SEDCalculator
from .dust_attenuation import (
    dust_attenuation_gb10_generalised,
    calzetti_attenuation_law,
    apply_dust_attenuation_to_continuum,
    apply_continuum_dust_model,
    normalize_continuum_dust,
    apply_dust_attenuation_to_line,
    read_dust_model_from_catalog,
)

__version__ = "0.1.0"
