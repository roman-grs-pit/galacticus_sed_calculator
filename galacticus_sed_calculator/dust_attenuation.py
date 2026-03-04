"""
Dust attenuation models for galaxy emission lines and continuum.

This module provides functions to calculate dust attenuation for emission lines
in galaxy spectra. It includes parametric models for predicting attenuation based
on galaxy properties (e.g., GB10 model) and attenuation curves that describe how
dust extinction varies with wavelength (e.g., Calzetti law).
"""

import json

import h5py
import numpy as np
import astropy.units as u
import astropy.constants as const
from typing import Union, List, Optional, Tuple, Callable


def dust_attenuation_garnBest10(
    Mstar: np.ndarray, 
    attenuation_scatter: float = 0.0,
    random_uniform: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate H-alpha dust attenuation following Garn & Best (2010).
    
    This is the mean relationship between stellar mass and H-alpha dust 
    attenuation from https://ui.adsabs.harvard.edu/abs/2010MNRAS.409..421G
    
    Parameters
    ----------
    Mstar : np.ndarray
        Stellar mass in solar masses
    attenuation_scatter : float, optional
        Scatter in the attenuation relation (in magnitudes). Default is 0.0.
        When non-zero, adds Gaussian scatter to the mean relation.
    random_uniform : np.ndarray, optional
        Pre-saved random uniform(0,1) numbers for each galaxy. If provided,
        these are used to generate reproducible scatter via inverse CDF.
        If None and scatter > 0, uses np.random.randn().
        
    Returns
    -------
    np.ndarray
        H-alpha attenuation in magnitudes (A_Halpha)
    """
    X = np.log10(Mstar / 1e10)
    A_Halpha = 0.91 + 0.77 * X + 0.11 * X**2 - 0.09 * X**3
    
    # Add scatter if requested
    if attenuation_scatter > 0:
        if random_uniform is not None:
            # Convert uniform(0,1) to standard normal via inverse CDF
            from scipy.stats import norm
            random_normal = norm.ppf(random_uniform)
            A_Halpha = A_Halpha + random_normal * attenuation_scatter
        else:
            A_Halpha = A_Halpha + np.random.randn(len(X)) * attenuation_scatter
    
    return A_Halpha


def dust_attenuation_gb10_generalised(
    Mstar: np.ndarray,
    z: np.ndarray,
    *,
    delta_0: float = 0.0,
    delta_M: float = 0.0,
    delta_z: float = 0.0,
    delta_Mz: float = 0.0,
    z_pivot: float = 1.0,
    attenuation_scatter: float = 0.0,
    rng: np.random.Generator | None = None,
    random_uniform: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    H-alpha dust attenuation based on Garn & Best (2010), with a flexible
    separable correction in stellar mass and redshift.

    Parameters
    ----------
    Mstar : np.ndarray
        Stellar mass in solar masses.
    z : np.ndarray
        Redshift.
    delta_0 : float, optional
        Global additive offset in magnitudes.
    delta_M : float, optional
        Mass-dependent tilt (per dex in Mstar).
    delta_z : float, optional
        Redshift-dependent offset.
    delta_Mz : float, optional
        Cross-term: mass-dependent redshift evolution.
    z_pivot : float, optional
        Pivot redshift for the redshift coordinate.
    attenuation_scatter : float, optional
        Gaussian scatter in magnitudes added after all corrections.
    rng : np.random.Generator, optional
        Random number generator for reproducibility. Ignored if random_uniform is provided.
    random_uniform : np.ndarray, optional
        Pre-saved random uniform(0,1) numbers for each galaxy. If provided,
        these are used to generate reproducible scatter via inverse CDF.
        If None and scatter > 0, uses rng or np.random.default_rng().

    Returns
    -------
    np.ndarray
        H-alpha attenuation in magnitudes (A_Halpha), clipped to >= 0.
    """

    # --- GB10 baseline ---
    X = np.atleast_1d(np.log10(Mstar / 1e10))
    A_gb10 = 0.91 + 0.77 * X + 0.11 * X**2 - 0.09 * X**3

    # --- Redshift coordinate ---
    u = np.log(1+z) - np.log(1+z_pivot)

    # --- Modified attenuation ---
    A_Halpha = (
        A_gb10
        + delta_0
        + delta_M * X
        + delta_z * u
        + delta_Mz * X * u
    )

    # --- Optional scatter ---
    if attenuation_scatter > 0:
        if random_uniform is not None:
            # Convert uniform(0,1) to standard normal via inverse CDF
            from scipy.stats import norm
            random_normal = norm.ppf(random_uniform)
            A_Halpha = A_Halpha + random_normal * attenuation_scatter
        else:
            if rng is None:
                rng = np.random.default_rng()
            A_Halpha = A_Halpha + rng.normal(0.0, attenuation_scatter, size=len(X))

    # --- Enforce physical floor ---
    A_Halpha = np.maximum(A_Halpha, 0.0)

    return A_Halpha


def calzetti_attenuation_law(wavelength, A_V=1.0):
    """
    Calculate wavelength-dependent dust attenuation using the Calzetti attenuation law.
    
    The Calzetti law (Calzetti et al. 2000) describes how dust attenuation varies with
    wavelength for starburst galaxies. It provides the ratio k(lambda) which relates
    the attenuation at a given wavelength to the V-band attenuation A_V.
    
    The attenuation at wavelength lambda is given by:
        A(lambda) = A_V * k(lambda) / k(V)
    
    where k(lambda) is the attenuation curve defined piecewise for different wavelength ranges.
    
    Parameters
    ----------
    wavelength : float, array-like, or Quantity
        Wavelength(s) at which to calculate the attenuation. If given as a Quantity,
        should have units of length (e.g., Angstroms, microns). If unitless, assumed
        to be in Angstroms.
    A_V : float, optional
        V-band attenuation in magnitudes. Default is 1.0 mag. This sets the overall
        normalization of the attenuation curve.
    
    Returns
    -------
    A_lambda : float or array-like
        Dust attenuation in magnitudes at the specified wavelength(s). Returns the
        same shape as the input wavelength array.
    
    Notes
    -----
    The Calzetti law is defined for wavelengths between 0.12 and 2.2 microns (1200-22000 Å).
    For wavelengths outside this range, the function extrapolates using the edge values.
    
    The k(lambda) curve is defined as:
        - For 0.63 μm <= lambda <= 2.20 μm:
          k(lambda) = 2.659 * (-1.857 + 1.040/lambda_μm) + R_V
        - For 0.12 μm <= lambda < 0.63 μm:
          k(lambda) = 2.659 * (-2.156 + 1.509/lambda_μm - 0.198/lambda_μm^2 + 0.011/lambda_μm^3) + R_V
    
    where R_V = 4.05 (ratio of total to selective extinction) for starburst galaxies.
    
    References
    ----------
    Calzetti, D., et al. (2000). "The Dust Content and Opacity of Actively Star-forming Galaxies."
    ApJ, 533, 682.
    
    Examples
    --------
    >>> # Calculate attenuation at H-alpha (6563 Å)
    >>> wavelength = 6563  # Angstroms
    >>> A_Ha = calzetti_attenuation_law(wavelength, A_V=1.0)
    
    >>> # Calculate for multiple wavelengths
    >>> wavelengths = np.array([4000, 5000, 6563, 10000])  # Angstroms
    >>> A_lambda = calzetti_attenuation_law(wavelengths, A_V=1.5)
    
    >>> # Using astropy units
    >>> import astropy.units as u
    >>> wavelength = 0.6563 * u.micron
    >>> A_Ha = calzetti_attenuation_law(wavelength, A_V=1.0)
    """
    # Handle astropy units - convert to Angstroms if necessary
    if hasattr(wavelength, 'unit'):
        wavelength_AA = wavelength.to(u.AA).value
    else:
        wavelength_AA = np.atleast_1d(wavelength)
    
    # Convert wavelength from Angstroms to microns for the Calzetti formula
    wavelength_micron = wavelength_AA / 10000.0
    
    # Calzetti R_V value for starburst galaxies
    R_V = 4.05
    
    # Calculate k(lambda) using the Calzetti law
    # The law is defined piecewise for different wavelength ranges
    k_lambda = np.zeros_like(wavelength_micron)
    
    # For wavelengths >= 0.63 microns (red/IR)
    mask_red = wavelength_micron >= 0.63
    k_lambda[mask_red] = 2.659 * (-1.857 + 1.040 / wavelength_micron[mask_red]) + R_V
    
    # For wavelengths < 0.63 microns (blue/UV)
    mask_blue = wavelength_micron < 0.63
    k_lambda[mask_blue] = 2.659 * (-2.156 + 
                                     1.509 / wavelength_micron[mask_blue] - 
                                     0.198 / wavelength_micron[mask_blue]**2 + 
                                     0.011 / wavelength_micron[mask_blue]**3) + R_V
    
    # k(V) at V-band (5500 Å = 0.55 microns)
    k_V = 2.659 * (-2.156 + 1.509/0.55 - 0.198/0.55**2 + 0.011/0.55**3) + R_V
    
    # Calculate A(lambda) = A_V * k(lambda) / k(V)
    A_lambda = A_V * k_lambda / k_V
    
    # Return scalar if input was scalar
    if isinstance(wavelength, (int, float)) or (hasattr(wavelength, 'isscalar') and wavelength.isscalar):
        return float(A_lambda[0]) if isinstance(A_lambda, np.ndarray) else A_lambda
    return A_lambda


def apply_dust_attenuation_to_line(line_flux, line_wavelength, A_Halpha, dust_law='calzetti'):
    """
    Apply dust attenuation to an emission line flux.
    
    This function calculates the attenuated flux of an emission line given its intrinsic
    flux, wavelength, and the amount of attenuation at H-alpha. The wavelength-dependent
    attenuation is calculated using a specified dust law (e.g., Calzetti).
    
    Parameters
    ----------
    line_flux : float or array-like or Quantity
        Intrinsic (unattenuated) flux of the emission line. If a Quantity, should have
        units of flux (e.g., erg/(s cm^2)).
    line_wavelength : float or Quantity
        Wavelength of the emission line. If a Quantity, should have units of length.
        If unitless, assumed to be in Angstroms.
    A_Halpha : float or array-like
        Dust attenuation at H-alpha wavelength in magnitudes.
    dust_law : str, optional
        Name of the dust attenuation law to use. Currently only 'calzetti' is supported.
        Default is 'calzetti'.
    
    Returns
    -------
    attenuated_flux : float or array-like or Quantity
        Attenuated flux of the emission line, in the same units as line_flux.
    
    Notes
    -----
    The attenuation is applied using the relation:
        F_obs = F_int * 10^(-0.4 * A(lambda))
    
    where F_obs is the observed (attenuated) flux, F_int is the intrinsic flux,
    and A(lambda) is the wavelength-dependent attenuation.
    
    The attenuation at the line wavelength is calculated from A(H-alpha) using
    the specified dust law to determine the wavelength dependence.
    
    Examples
    --------
    >>> # Attenuate a single emission line
    >>> import astropy.units as u
    >>> line_flux = 1e-15 * u.erg / (u.s * u.cm**2)
    >>> line_wavelength = 6563 * u.AA  # H-alpha
    >>> A_Halpha = 1.0  # mag
    >>> attenuated_flux = apply_dust_attenuation_to_line(line_flux, line_wavelength, A_Halpha)
    """
    # H-alpha wavelength (rest frame)
    Halpha_wavelength_AA = 6562.8
    
    # Calculate A(lambda) at the line wavelength using the specified dust law
    if dust_law == 'calzetti':
        # First get A_V from A(H-alpha)
        # A(H-alpha) = A_V * k(H-alpha) / k(V)
        # So A_V = A(H-alpha) * k(V) / k(H-alpha)
        
        # Get k values at H-alpha and V-band
        k_Halpha = _calzetti_k_lambda(Halpha_wavelength_AA)
        k_V = _calzetti_k_lambda(5500.0)  # V-band at 5500 Å
        
        # Calculate A_V from A(H-alpha)
        A_V = A_Halpha * k_V / k_Halpha
        
        # Now calculate A(lambda) at the line wavelength
        A_lambda = calzetti_attenuation_law(line_wavelength, A_V=A_V)
    else:
        raise ValueError(f"Dust law '{dust_law}' not supported. Currently only 'calzetti' is implemented.")
    
    # Apply the attenuation: F_obs = F_int * 10^(-0.4 * A)
    attenuation_factor = 10**(-0.4 * A_lambda)
    attenuated_flux = line_flux * attenuation_factor
    
    return attenuated_flux


def _calzetti_k_lambda(wavelength_AA):
    """
    Calculate the k(lambda) value for the Calzetti attenuation law. See (for example)
    equation 1.20 from https://ned.ipac.caltech.edu/level5/Sept12/Calzetti/paper.pdf
    
    This is a helper function used internally to compute the Calzetti k(lambda)
    value at a specific wavelength.
    
    Parameters
    ----------
    wavelength_AA : float
        Wavelength in Angstroms.
    
    Returns
    -------
    k_lambda : float
        The k(lambda) value at the specified wavelength.
    """
    wavelength_micron = wavelength_AA / 10000.0
    R_V = 4.05
    
    if wavelength_micron >= 0.63:
        k_lambda = 2.659 * (-1.857 + 1.040 / wavelength_micron) + R_V
    else:
        k_lambda = 2.659 * (-2.156 + 
                            1.509 / wavelength_micron - 
                            0.198 / wavelength_micron**2 + 
                            0.011 / wavelength_micron**3) + R_V
    
    return k_lambda


def read_dust_model_from_catalog(galacticus_file, base_path=None):
    """
    Read dust model configuration from a Galacticus HDF5 catalog.

    Reads the ``dust_model``, ``dust_params``, and ``dust_law`` attributes
    that were written to the ``dustAttenuatedNodeData`` group by
    ``calculate_catalog_magnitudes.py --dust-config``.

    Parameters
    ----------
    galacticus_file : str
        Path to the Galacticus HDF5 file.
    base_path : str, optional
        Base path within the file (e.g. ``'/Lightcone/Output1'``).  If
        ``None``, the format is auto-detected and the first output group is
        used.

    Returns
    -------
    dust_model : str
        Name of the dust model (e.g. ``'gb10_generalised'``).
    dust_params : dict
        Dictionary of dust model parameters.
    dust_law : str
        Name of the attenuation law (e.g. ``'calzetti'``).

    Raises
    ------
    KeyError
        If the ``dustAttenuatedNodeData`` group does not exist in the file
        (i.e. no dust-attenuated quantities have been saved yet).

    Examples
    --------
    >>> dust_model, dust_params, dust_law = read_dust_model_from_catalog(
    ...     'catalog.hdf5'
    ... )
    >>> print(dust_model)
    gb10_generalised
    >>> print(dust_params['delta_0'])
    0.275
    """
    if base_path is None:
        # Import here to avoid a circular import (sed_calculator imports from
        # this module; we only need detect_galacticus_format at call time).
        from .sed_calculator import detect_galacticus_format
        _, base_path = detect_galacticus_format(galacticus_file)

    dust_group_path = f'{base_path}/dustAttenuatedNodeData'

    with h5py.File(galacticus_file, 'r') as f:
        if dust_group_path not in f:
            raise KeyError(
                f"No 'dustAttenuatedNodeData' group found at '{dust_group_path}' "
                f"in '{galacticus_file}'. Run calculate_catalog_magnitudes.py "
                "with --dust-config to generate dust-attenuated quantities first."
            )
        grp = f[dust_group_path]
        dust_model = grp.attrs['dust_model']
        dust_law = grp.attrs['dust_law']
        dust_params = json.loads(grp.attrs['dust_params'])

    return dust_model, dust_params, dust_law
