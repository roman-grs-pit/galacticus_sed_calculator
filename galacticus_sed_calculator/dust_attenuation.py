"""
Dust attenuation models for galaxy emission lines and continuum.

This module provides functions to calculate dust attenuation for emission lines
in galaxy spectra. It includes parametric models for predicting attenuation based
on galaxy properties (e.g., GB10 model) and attenuation curves that describe how
dust extinction varies with wavelength (e.g., Calzetti law).
"""

import numpy as np
import astropy.units as u
import astropy.constants as const


def dust_attenuation_gb10_generalised(stellar_mass, redshift, delta_0, delta_z, delta_M, delta_Mz, attenuation_scatter=0.0):
    """
    Calculate dust attenuation at H-alpha wavelength using the generalized GB10 model.
    
    This function implements a generalized version of the Garn & Best (2010) dust attenuation
    model, which predicts the amount of dust attenuation at the H-alpha emission line wavelength
    based on galaxy stellar mass and redshift.
    
    The model follows the parametric form:
        A(H-alpha) = delta_0 + delta_z * z + delta_M * log10(M*/M_sun) + delta_Mz * z * log10(M*/M_sun)
    
    Optionally, a log-normal scatter can be added to the attenuation values.
    
    Parameters
    ----------
    stellar_mass : float or array-like
        Stellar mass of the galaxy in solar masses (M_sun). Can be a scalar or array.
    redshift : float or array-like
        Redshift of the galaxy. Can be a scalar or array.
    delta_0 : float
        Constant offset term in the attenuation relation.
    delta_z : float
        Coefficient for the redshift dependence.
    delta_M : float
        Coefficient for the stellar mass dependence.
    delta_Mz : float
        Coefficient for the coupled mass-redshift dependence.
    attenuation_scatter : float, optional
        Log-normal scatter to add to the attenuation values (in magnitudes).
        If 0 (default), no scatter is added. If non-zero, random scatter is drawn
        from a normal distribution with this standard deviation.
    
    Returns
    -------
    A_Halpha : float or array-like
        Dust attenuation at H-alpha wavelength in magnitudes. Returns the same shape
        as the input stellar_mass and redshift arrays.
    
    Notes
    -----
    The original Garn & Best (2010) model was calibrated using SDSS galaxies and
    provides a simple parametric description of how dust attenuation correlates with
    galaxy properties. The generalized version allows for flexible parameterization
    through the delta coefficients.
    
    References
    ----------
    Garn, T., & Best, P. N. (2010). "The dust attenuation law in distant galaxies: 
    evidence for variation with spectral type." MNRAS, 409, 421.
    
    Examples
    --------
    >>> # Calculate attenuation for a single galaxy
    >>> stellar_mass = 1e10  # M_sun
    >>> redshift = 0.1
    >>> A_Ha = dust_attenuation_gb10_generalised(stellar_mass, redshift, 
    ...                                           delta_0=0.275, delta_z=-1.614, 
    ...                                           delta_M=-0.834, delta_Mz=-0.708)
    
    >>> # Calculate for multiple galaxies with scatter
    >>> masses = np.array([1e9, 1e10, 1e11])
    >>> redshifts = np.array([0.1, 0.5, 1.0])
    >>> A_Ha = dust_attenuation_gb10_generalised(masses, redshifts,
    ...                                           delta_0=0.275, delta_z=-1.614,
    ...                                           delta_M=-0.834, delta_Mz=-0.708,
    ...                                           attenuation_scatter=0.25)
    """
    # Convert inputs to numpy arrays for consistent handling
    stellar_mass = np.atleast_1d(stellar_mass)
    redshift = np.atleast_1d(redshift)
    
    # Calculate log10 of stellar mass
    log_stellar_mass = np.log10(stellar_mass)
    
    # Calculate mean attenuation using the parametric model
    A_Halpha = (delta_0 + 
                delta_z * redshift + 
                delta_M * log_stellar_mass + 
                delta_Mz * redshift * log_stellar_mass)
    
    # Add scatter if specified
    if attenuation_scatter > 0:
        # Draw random values from a normal distribution
        scatter = np.random.normal(0, attenuation_scatter, size=A_Halpha.shape)
        A_Halpha += scatter
    
    # Ensure attenuation is non-negative
    A_Halpha = np.maximum(A_Halpha, 0.0)
    
    # Return scalar if input was scalar
    if A_Halpha.size == 1:
        return float(A_Halpha[0])
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
    Calculate the k(lambda) value for the Calzetti attenuation law.
    
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
