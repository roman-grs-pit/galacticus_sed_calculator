"""
galacticus_sed_calculator

A Python package for generating spectra for galaxies simulated with Galacticus,
based on saved star formation histories.
"""

from .dust_attenuation import (
    dust_attenuation_gb10_generalised,
    calzetti_attenuation_law,
    apply_dust_attenuation_to_continuum,
    apply_continuum_dust_model,
    normalize_emission_line_dust,
    normalize_continuum_dust,
    apply_dust_attenuation_to_line,
    read_dust_model_from_catalog,
)

__version__ = "0.1.0"

__all__ = [
    "SEDCalculator",
    "dust_attenuation_gb10_generalised",
    "calzetti_attenuation_law",
    "apply_dust_attenuation_to_continuum",
    "apply_continuum_dust_model",
    "normalize_emission_line_dust",
    "normalize_continuum_dust",
    "apply_dust_attenuation_to_line",
    "read_dust_model_from_catalog",
]


def __getattr__(name):
    if name == "SEDCalculator":
        from .sed_calculator import SEDCalculator

        return SEDCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
