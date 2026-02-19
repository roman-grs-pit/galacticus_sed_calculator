import numpy as np
import h5py 
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck15 
import re
import synphot
from synphot.models import Empirical1D, GaussianFlux1D
from synphot import units, SourceSpectrum
from .dust_attenuation import (
    dust_attenuation_gb10_generalised,
    apply_dust_attenuation_to_line
)
#from specutils import Spectrum
#from specutils.manipulation import FluxConservingResampler


def calculate_sed_from_sfh_and_template(sedTemplate, starFormationHistory):
    # don't check for compatibility of sedTemplate and SFH metallicities and ages, as this then requires always passing a bunch of extra stuff to this function. Accept a number of age bins in the sedTemplate to skip, which can be 'auto', but as best practice should be calculated from the sedTemplate ages to see that everything seems OK
    NZ, Nage, Nwav = sedTemplate.shape
    assert starFormationHistory.shape[0] == NZ, "SFH must have the same number of metallicity bins as the SED template"
    assert starFormationHistory.shape[1] <= Nage, "SFH must have the same or fewer age bins than the SED template"
    skipAgeBins = Nage - starFormationHistory.shape[1] # skip the early bins which cover time periods before the beginning of the Universe
    SED = np.sum(starFormationHistory[:, :, np.newaxis]*sedTemplate[:,skipAgeBins:], axis=(0, 1))
    return SED

def resample_sed(wavelength_old, Lnu_old, wavelength_new, extrapolateWithZeros=False):
    # here just using linear interpolation. Could use something "flux conserving" for cases where the resolution is being decreased (e.g. https://specutils.readthedocs.io/en/stable/api/specutils.manipulation.FluxConservingResampler.html) but these don't work well for the case of increasing resolution, so we would then need a switch, and it all gets rather complicated...
    if not extrapolateWithZeros:
        if wavelength_new.min() < wavelength_old.min() or wavelength_new.max() > wavelength_old.max():
            raise ValueError("New wavelength grid must be within the range of the old wavelength grid.")
    return np.interp(wavelength_new, wavelength_old, Lnu_old, left=0.0, right=0.0)

def redshift_sed(rest_frame_wavelength, rest_frame_sed, redshift):
    # Calculate the redshifted SED from the observed frame SED. 
    ''' Note that the notion of an "observed frame SED for a galaxy at non-zero redshift" has caused me confusion, so we should probably avoid using it. In partiuclar, it is not clear to me when one should re-scale the SED by 1+z in going from rest frame SED -> observed frame SED -> observed Flux. Given the flux is what we observe, it is probably best just to think in terms of that'''
    redshifted_wavelength = rest_frame_wavelength * (1 + redshift)
    redshifted_sed = rest_frame_sed * (1 + redshift) # Jacobian: d.nu_rest / d.nu_obs = 1 + z
    return redshifted_wavelength, redshifted_sed

def flux_density_from_sed(rest_frame_wavelength, rest_frame_sed, redshift, cosmo=Planck15):
    # note that I am assuming SEDs are dL/d.nu and fluxes are dF/d.nu, 
    redshifted_wavelength = rest_frame_wavelength * (1 + redshift)
    # eq.6 https://ned.ipac.caltech.edu/level5/Sept02/Hogg/Hogg2.html
    flux_density = (1+redshift) * rest_frame_sed / (4*np.pi*cosmo.luminosity_distance(redshift).to_value('Mpc')**2) 
    return redshifted_wavelength, flux_density

def process_wavelength_array(wavelengths):
    """
    Ensure input wavelengths have astropy units, assuming Angstroms if unspecified.

    Parameters
    ----------
    wavelengths : array-like or Quantity
        Array of wavelengths. If unitless, they are assumed to be in Angstroms.

    Returns
    -------
    wavelengths : Quantity
        Wavelengths with units converted to Angstroms.
    """
    wavelengths = np.asanyarray(wavelengths)
    if not isinstance(wavelengths, u.Quantity):
        wavelengths = wavelengths * u.AA
    return wavelengths.to(u.AA)

def detect_galacticus_format(filename):
    """
    Auto-detect whether a Galacticus file uses lightcone or fixed-time output format.
    
    Parameters
    ----------
    filename : str
        Path to the Galacticus HDF5 file
        
    Returns
    -------
    format_type : str
        Either 'lightcone' or 'fixed-time'
    base_path : str
        Base path to nodeData (e.g., '/Lightcone/Output1' or '/Outputs/Output1')
    """
    with h5py.File(filename, 'r') as f:
        if 'Lightcone' in f:
            return 'lightcone', '/Lightcone/Output1'
        elif 'Outputs' in f:
            # Find first output (usually Output1, but could be Output2, etc.)
            outputs = [key for key in f['/Outputs'].keys() if key.startswith('Output')]
            if outputs:
                # Sort to get the first output numerically
                outputs.sort(key=lambda x: int(x.replace('Output', '')))
                return 'fixed-time', f'/Outputs/{outputs[0]}'
            else:
                raise ValueError("Found /Outputs group but no Output* subgroups")
        else:
            raise ValueError("Cannot determine format: neither /Lightcone nor /Outputs found in file")

def outputTime_to_redshift(outputTime, cosmo=Planck15):
    """
    Convert age of universe (outputTime) to redshift.
    
    Parameters
    ----------
    outputTime : float
        Age of universe in Gyr
    cosmo : astropy.cosmology
        Cosmology object (default: Planck15)
        
    Returns
    -------
    redshift : float
        Redshift corresponding to the output time
    """
    from astropy.cosmology import z_at_value
    return z_at_value(cosmo.age, outputTime * u.Gyr)

def age_of_universe_to_lookback_time(times, outputTime):
    """
    Convert age-of-universe times to lookback times (stellar ages).
    
    In fixed-time output format, times are stored as age of universe.
    We need lookback times for SED calculation: lookback_time = outputTime - age_of_universe
    
    Parameters
    ----------
    times : array-like
        Ages of universe in Gyr (time bin edges)
    outputTime : float
        Age of universe at output time in Gyr
        
    Returns
    -------
    lookback_times : ndarray
        Lookback times in Gyr
    """
    times = np.asarray(times)
    lookback_times = outputTime - times
    # Ensure non-negative (can have small numerical errors)
    lookback_times = np.maximum(lookback_times, 0.0)
    return lookback_times

# Default configuration for lightcone format
# Note: This is now auto-detected and adjusted based on file format
galacticus_sed_config = {
    'diskSedPath': '/Lightcone/Output1/nodeData/diskStellarSED:identity',
    'spheroidSedPath': '/Lightcone/Output1/nodeData/spheroidStellarSED:identity',
    'diskSedWavelengths': '/Lightcone/Output1/nodeData/diskStellarSED:identityColumnValues',
    'spheroidSedWavelengths': '/Lightcone/Output1/nodeData/spheroidStellarSED:identityColumnValues',
    'redshift': '/Lightcone/Output1/nodeData/lightconeRedshiftObserved',
    'diskSFH': '/Lightcone/Output1/nodeData/diskStarFormationHistoryMass',
    'diskSFH_times': '/Lightcone/Output1/nodeData/diskStarFormationHistoryTimes',
    'spheroidSFH': '/Lightcone/Output1/nodeData/spheroidStarFormationHistoryMass',
    'spheroidSFH_times': '/Lightcone/Output1/nodeData/spheroidStarFormationHistoryTimes',
}

def getLineNames(f, component='disk', verbose=False):
    """ Get the emission line names for a given component from the Galacticus output file."""
    if component=='AGN':
        for key in f['/Parameters/nodePropertyExtractor'].attrs:
            val = f['/Parameters/nodePropertyExtractor'].attrs[key]
            if val == b'lmnstyEmssnLineAGN':
                line_group = f'/Parameters/nodePropertyExtractor/' + key
                if verbose:
                    print(f'Found luminosityEmissionLine nodePropertyExtractor for {component}: {key}')
    else:
        for key in f['/Parameters/nodePropertyExtractor'].attrs:
            val = f['/Parameters/nodePropertyExtractor'].attrs[key]
            if val == b'luminosityEmissionLine':
                line_group = f'/Parameters/nodePropertyExtractor/' + key
                if f[line_group].attrs['component'] == component.encode('utf-8'):
                    if verbose:
                        print(f'Found luminosityEmissionLine nodePropertyExtractor for {component}: {key}')
                    break
        else:
            raise ValueError("No 'luminosityEmissionLine' nodePropertyExtractor found.")
    lineNames = f[line_group].attrs['lineNames']
    return lineNames

def getLineProperties(fname, component='disk', galIndex=None, hdf5_base_path=None):
    """
    Get the emission line names, wavelengths, and luminosities for a given component from the Galacticus output file.
    
    Parameters:
    - fname: str, path to the Galacticus output HDF5 file.
    - component: str, 'disk', 'AGN', or 'spheroid' to specify which component's emission lines to retrieve.
    - galIndex: int or None, index of the galaxy to retrieve line luminosities for. If None, retrieves all galaxies.
    - hdf5_base_path: str or None, base path for the HDF5 file structure where emission line data is stored.
      If None, will auto-detect based on file format (lightcone vs fixed-time).
    
    Returns:
    - lineNames: np.ndarray of emission line names.
    - lineWavelengths: np.ndarray of wavelengths corresponding to the emission lines.
    - lineLuminosities: np.ndarray of luminosities for the emission lines.
    """
    # Auto-detect base path if not provided
    if hdf5_base_path is None:
        format_type, base_path = detect_galacticus_format(fname)
        hdf5_base_path = f'{base_path}/nodeData/luminosityEmissionLine'
    
    with h5py.File(fname, 'r') as f:
        lineNamesBytes = getLineNames(f, component=component)
        # Convert to regular strings
        lineNames = np.char.decode(lineNamesBytes, encoding='utf-8')
        # Extract the numeric suffix as wavelengths
        lineWavelengths = np.array([int(re.search(r'\d+$', name).group()) for name in lineNames])
        # Build HDF5 paths
        def firstLetterCapitalize(s):
            return s[0].upper() + s[1:]
        hdf5_paths = np.array([f"{hdf5_base_path}{firstLetterCapitalize(component)}:{name}" for name in lineNames])
        # get the line luminosities
        def get_data(f, path, galIndex=None):
            return f[path][:] if galIndex is None else f[path][galIndex]
        lineLuminosities = np.array([get_data(f, hdf5_path, galIndex)  for hdf5_path in hdf5_paths])
    return lineNames, lineWavelengths, lineLuminosities

def minFlux(value):
    """
    Accept either a float (assumed to be erg / (s cm^2)),
    or an astropy Quantity in any valid flux unit.
    Returns an astropy Quantity in erg/(s cm^2).
    """

    default_unit = u.erg / (u.s * u.cm**2)

    # Case 1: astropy Quantity
    if isinstance(value, u.Quantity):
        try:
            return value.to(default_unit)
        except u.UnitConversionError:
            raise ValueError(f"minFlux: cannot convert unit {value.unit} to erg/(s cm^2)")

    # Case 2: plain number
    if isinstance(value, (int, float)):
        return value * default_unit

    # Case 3: unsupported
    raise TypeError("minFlux requires a number or an astropy Quantity.")

def gaussian_emission_line(wav, lambda0, sigma, fLine):
    """
    Generate a Gaussian emission line profile.
    
    Parameters
    ----------
    wav : array-like or Quantity
        Wavelength array. If Quantity, should be in Angstroms.
    lambda0 : float or Quantity
        Central wavelength of the emission line. If Quantity, should be in Angstroms.
    sigma : float or Quantity
        Standard deviation of the Gaussian. If Quantity, should be in Angstroms.
    fLine : float or Quantity
        Total integrated flux of the emission line. If Quantity, should be in erg/(s cm^2).
    
    Returns
    -------
    line_flux : array-like
        Flux density array matching the wavelength array, with astropy units erg/(s cm^2 AA).
    """
    # Handle units
    if isinstance(wav, u.Quantity):
        wav_val = wav.to_value(u.AA)
    else:
        wav_val = np.asarray(wav)
    
    if isinstance(lambda0, u.Quantity):
        lambda0_val = lambda0.to_value(u.AA)
    else:
        lambda0_val = float(lambda0)
    
    if isinstance(sigma, u.Quantity):
        sigma_val = sigma.to_value(u.AA)
    else:
        sigma_val = float(sigma)
    
    if isinstance(fLine, u.Quantity):
        fLine_val = fLine.to_value(u.erg / (u.s * u.cm**2))
    else:
        fLine_val = float(fLine)
    
    # Calculate Gaussian profile
    norm = fLine_val / (sigma_val * np.sqrt(2 * np.pi))
    line_flux = norm * np.exp(-0.5 * ((wav_val - lambda0_val) / sigma_val)**2)
    
    # Return with units
    return line_flux * u.erg / (u.s * u.cm**2 * u.AA)

def gaussian_from_fwhm(wav, lambda0, fwhm, fLine):
    """
    Generate a Gaussian emission line profile from FWHM.
    
    Parameters
    ----------
    wav : array-like or Quantity
        Wavelength array. If Quantity, should be in Angstroms.
    lambda0 : float or Quantity
        Central wavelength of the emission line. If Quantity, should be in Angstroms.
    fwhm : float or Quantity
        Full width at half maximum of the Gaussian. If Quantity, should be in Angstroms.
    fLine : float or Quantity
        Total integrated flux of the emission line. If Quantity, should be in erg/(s cm^2).
    
    Returns
    -------
    line_flux : array-like
        Flux density array matching the wavelength array, with astropy units erg/(s cm^2 AA).
    """
    # Convert FWHM to sigma
    if isinstance(fwhm, u.Quantity):
        fwhm_val = fwhm.to_value(u.AA)
    else:
        fwhm_val = float(fwhm)
    
    sigma_val = fwhm_val / (2 * np.sqrt(2 * np.log(2)))
    
    # Create sigma with units if input had units
    if isinstance(fwhm, u.Quantity):
        sigma = sigma_val * u.AA
    else:
        sigma = sigma_val
    
    return gaussian_emission_line(wav, lambda0, sigma, fLine)

class SEDCalculator:
    def __init__(self, sedTemplateFilename, config=galacticus_sed_config, cosmology=Planck15):
        self.sedTemplateFilename = sedTemplateFilename
        self.load_sed_template()
        self.config = config
        self.cosmo = cosmology
        # Cache for validated files to avoid repeated validation
        self._validated_files = set()
        # Cache for detected file formats
        self._file_formats = {}
        # Cache for emission line names and wavelengths (same for all galaxies)
        # Key: (filename, component), Value: (lineNames, lineWavelengths, hdf5_paths)
        self._line_metadata_cache = {}
    
    def _get_config_for_format(self, format_type, base_path):
        """
        Generate config dictionary for a specific format type.
        
        Parameters
        ----------
        format_type : str
            'lightcone' or 'fixed-time'
        base_path : str
            Base path like '/Lightcone/Output1' or '/Outputs/Output1'
            
        Returns
        -------
        config : dict
            Configuration dictionary with format-specific paths
        """
        node_data_path = f'{base_path}/nodeData'
        
        config = {
            'format_type': format_type,
            'base_path': base_path,
            'diskSedPath': f'{node_data_path}/diskStellarSED:identity',
            'spheroidSedPath': f'{node_data_path}/spheroidStellarSED:identity',
            'diskSedWavelengths': f'{node_data_path}/diskStellarSED:identityColumnValues',
            'spheroidSedWavelengths': f'{node_data_path}/spheroidStellarSED:identityColumnValues',
            'diskSFH': f'{node_data_path}/diskStarFormationHistoryMass',
            'spheroidSFH': f'{node_data_path}/spheroidStarFormationHistoryMass',
        }
        
        if format_type == 'lightcone':
            config['redshift'] = f'{node_data_path}/lightconeRedshiftObserved'
            config['diskSFH_times'] = f'{node_data_path}/diskStarFormationHistoryTimes'
            config['spheroidSFH_times'] = f'{node_data_path}/spheroidStarFormationHistoryTimes'
        # For fixed-time format, redshift and times are handled differently (not in config)
        
        return config
    
    def _get_line_metadata(self, filename, component):
        """
        Get cached emission line metadata (names, wavelengths, HDF5 paths).
        
        This method caches the line names and wavelengths which are the same for all 
        galaxies in a file, avoiding repeated file reads and string processing.
        
        Parameters
        ----------
        filename : str
            Path to Galacticus HDF5 file
        component : str
            Component name ('disk', 'spheroid', or 'AGN')
            
        Returns
        -------
        lineNames : np.ndarray
            Array of emission line names
        lineWavelengths : np.ndarray
            Array of rest-frame wavelengths (Angstroms)
        hdf5_paths : np.ndarray
            Array of HDF5 paths to line luminosity datasets
        """
        cache_key = (filename, component)
        
        # Return cached data if available
        if cache_key in self._line_metadata_cache:
            return self._line_metadata_cache[cache_key]
        
        # Otherwise, compute and cache
        # Auto-detect base path
        format_type, base_path = detect_galacticus_format(filename)
        hdf5_base_path = f'{base_path}/nodeData/luminosityEmissionLine'
        
        with h5py.File(filename, 'r') as f:
            lineNamesBytes = getLineNames(f, component=component)
            # Convert to regular strings
            lineNames = np.char.decode(lineNamesBytes, encoding='utf-8')
            # Extract the numeric suffix as wavelengths
            lineWavelengths = np.array([int(re.search(r'\d+$', name).group()) for name in lineNames])
            # Build HDF5 paths
            def firstLetterCapitalize(s):
                return s[0].upper() + s[1:]
            hdf5_paths = np.array([f"{hdf5_base_path}{firstLetterCapitalize(component)}:{name}" for name in lineNames])
        
        # Cache the result
        self._line_metadata_cache[cache_key] = (lineNames, lineWavelengths, hdf5_paths)
        
        return lineNames, lineWavelengths, hdf5_paths

    def load_sed_template(self):
        """
        Load the SED template from the given filename.
        
        Supports both lightcone and fixed-time SED template formats:
        - Lightcone: uses /ages (stellar ages/lookback times)
        - Fixed-time: uses /time (cosmic time)
        """
        with h5py.File(self.sedTemplateFilename, 'r') as f:
            self.sedTemplate = f['sedTemplate'][:]
            self.sedMetallicity = f['metallicity'][:]
            self.sedWavelength = f['wavelength'][:]
            
            # Detect SED template format
            if 'ages' in f:
                # Lightcone format: stellar ages (lookback times)
                self.sedAges = f['ages'][:]
                self.sedTemplateFormat = 'lightcone'
            elif 'time' in f:
                # Fixed-time format: cosmic times
                self.sedTime = f['time'][:]
                self.sedTemplateFormat = 'fixed-time'
                # For backward compatibility, also store as sedAges
                # (will be converted to lookback times when used with galaxy data)
                self.sedAges = self.sedTime.copy()
            else:
                raise ValueError("SED template must contain either 'ages' (lightcone) or 'time' (fixed-time) dataset")
    
    def get_sed_template_parameters(self):
        """
        Extract the star formation history parameters from the SED template.
        
        The SED template stores bin maximum values in sedAges (for lightcone) or 
        sedTime (for fixed-time). This function reconstructs the parameters that 
        would have been used to generate these bins.
        
        For lightcone templates: sedAges contains lookback times in descending order
        For fixed-time templates: sedTime contains cosmic times in ascending order
        
        Returns
        -------
        dict
            Dictionary containing:
            - ageMinimum: minimum age/time bin boundary (Gyr)
            - ageMaximum: maximum age/time bin boundary (Gyr)
            - countAges: number of age/time bins (excluding the zero-age bin)
            - metallicityMinimum: minimum metallicity bin boundary (Solar units)
            - metallicityMaximum: maximum metallicity bin boundary (Solar units)
            - countMetallicities: number of metallicity bins (excluding the infinity bin)
            - sedTemplateFormat: 'lightcone' or 'fixed-time'
        """
        # Handle both lightcone (ages in descending order) and fixed-time (time in ascending order)
        if self.sedTemplateFormat == 'lightcone':
            # Ages are stored in descending order (lookback time)
            # The last non-zero age is the minimum
            # The first age is the maximum (age of universe)
            # There's always an additional bin at age=0
            
            # Find the index of the zero-age bin
            zero_age_idx = np.where(self.sedAges < 1e-10)[0]
            if len(zero_age_idx) > 0:
                # Exclude the zero-age bin from the count
                non_zero_ages = self.sedAges[:zero_age_idx[0]]
                countAges = len(non_zero_ages)
                ageMaximum = non_zero_ages[0]
                ageMinimum = non_zero_ages[-1]
            else:
                # No explicit zero bin, but there should be
                countAges = len(self.sedAges) - 1
                ageMaximum = self.sedAges[0]
                ageMinimum = self.sedAges[-1]
        else:  # fixed-time
            # Times are in ascending order (cosmic time)
            # For fixed-time: times are bin MAXIMA, starting from the first bin
            # The minimum is implicitly t=0 (not stored in the array)
            # So if we have N time values, we have N bins:
            #   Bin 0: [0, sedTime[0]]
            #   Bin 1: [sedTime[0], sedTime[1]]
            #   ...
            #   Bin N-1: [sedTime[N-2], sedTime[N-1]]
            if len(self.sedTime) > 0:
                countAges = len(self.sedTime)  # Number of bins = number of time values
                ageMinimum = 0.0  # Implicit minimum (t=0, not stored)
                ageMaximum = self.sedTime[-1]  # Actually outputTime
            else:
                raise ValueError("SED template has no time bins")
        
        # Metallicity: bins are logarithmically spaced
        # The first bin starts at the first value (which represents the max of the first bin)
        # Since bins are [0, sedMetallicity[0]], [sedMetallicity[0], sedMetallicity[1]], ...
        # The minimum is the first value (metallicityMinimum parameter in Galacticus)
        # The last finite value before infinity is metallicityMaximum
        # There's always an additional bin extending to infinity
        
        # Find values that are essentially infinity (> 1e10 is a reasonable threshold)
        infinity_threshold = 1e10
        finite_mask = self.sedMetallicity < infinity_threshold
        finite_metallicities = self.sedMetallicity[finite_mask]
        
        if len(finite_metallicities) < len(self.sedMetallicity):
            # There's an infinity bin
            countMetallicities = len(finite_metallicities)
            metallicityMinimum = finite_metallicities[0]
            metallicityMaximum = finite_metallicities[-1]
        else:
            # No infinity bin, which would be unusual
            countMetallicities = len(self.sedMetallicity) - 1
            metallicityMinimum = self.sedMetallicity[0]
            metallicityMaximum = self.sedMetallicity[-1]
        
        return {
            'ageMinimum': ageMinimum,
            'ageMaximum': ageMaximum,
            'countAges': countAges,
            'metallicityMinimum': metallicityMinimum,
            'metallicityMaximum': metallicityMaximum,
            'countMetallicities': countMetallicities,
            'sedTemplateFormat': self.sedTemplateFormat
        }
    
    def validate_sfh_compatibility(self, filename):
        """
        Validate that the star formation history parameters in the Galacticus file
        are compatible with the SED template.
        
        Parameters
        ----------
        filename : str
            Path to the Galacticus HDF5 file
            
        Raises
        ------
        ValueError
            If the SFH parameters don't match the SED template parameters
            
        Notes
        -----
        Results are cached, so repeated calls with the same filename will not
        re-validate the file.
        """
        # Check if we've already validated this file
        if filename in self._validated_files:
            return
        
        # Get parameters from SED template
        sed_params = self.get_sed_template_parameters()
        
        # Detect galaxy file format
        galaxy_format, base_path = detect_galacticus_format(filename)
        
        # OPTION 1: Strict format matching
        # SED template format must match galaxy file format
        if sed_params.get('sedTemplateFormat') != galaxy_format:
            raise ValueError(
                f"SED template format ('{sed_params.get('sedTemplateFormat')}') does not match "
                f"galaxy file format ('{galaxy_format}'). Lightcone SED templates can only be used "
                f"with lightcone galaxy data, and fixed-time SED templates can only be used with "
                f"fixed-time galaxy data, because the time binning algorithms are different."
            )
        
        # Read parameters from Galacticus file
        with h5py.File(filename, 'r') as f:
            if '/Parameters/starFormationHistory' not in f:
                raise ValueError("No starFormationHistory parameters found in Galacticus file")
            
            sfh_group = f['/Parameters/starFormationHistory']
            if galaxy_format == 'lightcone':
                sfh_params = {
                    'ageMinimum': sfh_group.attrs['ageMinimum'],
                    'countAges': sfh_group.attrs['countAges'],
                    'metallicityMinimum': sfh_group.attrs['metallicityMinimum'],
                    'metallicityMaximum': sfh_group.attrs['metallicityMaximum'],
                    'countMetallicities': sfh_group.attrs['countMetallicities']
                }
            else:  # fixed-time
                sfh_params = {
                    'ageMinimum': sfh_group.attrs['timeStepMinimum'],
                    'countAges': sfh_group.attrs['countTimeStepsMaximum'],
                    'metallicityMinimum': sfh_group.attrs['metallicityMinimum'],
                    'metallicityMaximum': sfh_group.attrs['metallicityMaximum'],
                    'countMetallicities': sfh_group.attrs['countMetallicities']
                }
                
                # OPTION 2: For fixed-time, validate that time arrays match
                # Get time array from galaxy file (from SFH dataset attribute)
                node_data_path = f"{base_path}/nodeData"
                if node_data_path in f:
                    disk_sfh_path = f"{node_data_path}/diskStarFormationHistoryMass"
                    if disk_sfh_path in f and 'time' in f[disk_sfh_path].attrs:
                        galaxy_times = f[disk_sfh_path].attrs['time']
                        
                        # Compare with SED template times
                        if hasattr(self, 'sedTime'):
                            if len(galaxy_times) != len(self.sedTime):
                                raise ValueError(
                                    f"Fixed-time: time array length mismatch. "
                                    f"Galaxy file has {len(galaxy_times)} time bins, "
                                    f"SED template has {len(self.sedTime)} time bins."
                                )
                            
                            # Check that time values match (within tolerance)
                            time_tol = 1e-5  # Relative tolerance for time comparison
                            if not np.allclose(galaxy_times, self.sedTime, rtol=time_tol):
                                max_diff = np.max(np.abs(galaxy_times - self.sedTime))
                                # Avoid divide by zero - use absolute comparison for small values
                                nonzero_mask = np.abs(self.sedTime) > 1e-10
                                if np.any(nonzero_mask):
                                    max_rel_diff = np.max(np.abs((galaxy_times[nonzero_mask] - self.sedTime[nonzero_mask]) / self.sedTime[nonzero_mask]))
                                else:
                                    max_rel_diff = 0.0
                                raise ValueError(
                                    f"Fixed-time: time arrays do not match. "
                                    f"Max absolute difference: {max_diff:.6e} Gyr, "
                                    f"Max relative difference: {max_rel_diff:.6e}. "
                                    f"The SED template and galaxy file must have been generated "
                                    f"with the same Galacticus time binning parameters."
                                )
        
        # Compare parameters
        errors = []
        
        # Check counts
        if sfh_params['countAges'] != sed_params['countAges']:
            errors.append(f"countAges mismatch: SFH={sfh_params['countAges']}, SED template={sed_params['countAges']}")
        
        if sfh_params['countMetallicities'] != sed_params['countMetallicities']:
            errors.append(f"countMetallicities mismatch: SFH={sfh_params['countMetallicities']}, SED template={sed_params['countMetallicities']}")
        
        # Check boundaries with some tolerance for floating point comparison
        rel_tol = 1e-6
        
        # For lightcone format, validate ageMinimum
        # For fixed-time format, ageMinimum is actually timeStepMinimum (minimum bin width)
        # which doesn't directly appear in the SED template, but we've already validated
        # the time arrays match above for fixed-time
        if galaxy_format == 'lightcone':
            if not np.isclose(sfh_params['ageMinimum'], sed_params['ageMinimum'], rtol=rel_tol):
                errors.append(f"ageMinimum mismatch: SFH={sfh_params['ageMinimum']}, SED template={sed_params['ageMinimum']}")
        
        if not np.isclose(sfh_params['metallicityMinimum'], sed_params['metallicityMinimum'], rtol=rel_tol):
            errors.append(f"metallicityMinimum mismatch: SFH={sfh_params['metallicityMinimum']}, SED template={sed_params['metallicityMinimum']}")
        
        if not np.isclose(sfh_params['metallicityMaximum'], sed_params['metallicityMaximum'], rtol=rel_tol):
            errors.append(f"metallicityMaximum mismatch: SFH={sfh_params['metallicityMaximum']}, SED template={sed_params['metallicityMaximum']}")
        
        if errors:
            error_msg = "SED template and star formation history are incompatible:\n  " + "\n  ".join(errors)
            raise ValueError(error_msg)
        
        # Cache this file as validated
        self._validated_files.add(filename)

    def calculate_rest_frame_sed(self, starFormationHistory):
        return calculate_sed_from_sfh_and_template(self.sedTemplate, starFormationHistory)
    
    def calculate_observed_frame_sed(self, starFormationHistory, redshift, wavelengths, extrapolateWithZeros=False):
        # Note, deprecated, let's not think about "observed frame SEDs"!
        rest_frame_sed = self.calculate_rest_frame_sed(starFormationHistory)
        rest_frame_wavelength = self.sedWavelength
        redshifted_wavelength, redshifted_sed = redshift_sed(rest_frame_wavelength, rest_frame_sed, redshift)
        observed_sed = resample_sed(redshifted_wavelength, redshifted_sed, wavelengths, extrapolateWithZeros=extrapolateWithZeros)
        return observed_sed
    
    def calculate_continuum_Fnu(self, starFormationHistory, redshift, wavelengths=None, extrapolateWithZeros=False):
        # Calculate the flux density from the SED given the star formation history and redshift
        # first, get rest-frame SED from SFH
        rest_frame_sed = self.calculate_rest_frame_sed(starFormationHistory)
        rest_frame_wavelength = self.sedWavelength
        # convert the SED to an observed flux (including redshifting)
        redshifted_wavelength, flux_density = flux_density_from_sed(rest_frame_wavelength, rest_frame_sed, redshift, cosmo=self.cosmo)
        if wavelengths is None:
            return flux_density*u.Lsun/(u.Hz * u.Mpc**2), redshifted_wavelength*u.AA
        else:
            # ensure wavelengths are in Angstroms
            wavelengths = process_wavelength_array(wavelengths)
        # and finally, resample to the desired wavelength grid
        observed_Fnu = resample_sed(redshifted_wavelength, flux_density, wavelengths.to_value(u.AA), extrapolateWithZeros=extrapolateWithZeros)
        return observed_Fnu*u.Lsun/(u.Hz * u.Mpc**2), wavelengths
    
    def read_galacticus_galaxy(self, filename, galIndex):
        """
        Read a Galacticus catalog file (HDF5) and get properties of one galaxy.
        
        This method automatically detects the file format (lightcone vs fixed-time)
        and reads the appropriate data structure.
        
        Parameters
        ----------
        filename : str
            Path to Galacticus HDF5 file
        galIndex : int
            Index of the galaxy to read
            
        Returns
        -------
        galData : dict
            Dictionary containing galaxy properties:
            - redshift: Galaxy redshift
            - diskSFH: Star formation history for disk (2D array: metallicity x time)
            - diskSFH_times: Time bins for disk SFH (lookback times in Gyr)
            - spheroidSFH: Star formation history for spheroid
            - spheroidSFH_times: Time bins for spheroid SFH (lookback times in Gyr)
        """
        # Detect format if not already cached
        if filename not in self._file_formats:
            format_type, base_path = detect_galacticus_format(filename)
            self._file_formats[filename] = (format_type, base_path)
        else:
            format_type, base_path = self._file_formats[filename]
        
        # Get format-specific config
        config = self._get_config_for_format(format_type, base_path)
        
        with h5py.File(filename, 'r') as f:
            if format_type == 'lightcone':
                # Lightcone format: per-galaxy redshift and times
                self.galData = {
                    'redshift': f[config['redshift']][galIndex],
                    'diskSFH': np.array([list(x) for x in f[config['diskSFH']][galIndex]], dtype=float),
                    'diskSFH_times': f[config['diskSFH_times']][galIndex],
                    'spheroidSFH': np.array([list(x) for x in f[config['spheroidSFH']][galIndex]], dtype=float),
                    'spheroidSFH_times': f[config['spheroidSFH_times']][galIndex],
                }
            
            elif format_type == 'fixed-time':
                # Fixed-time format: single output time, times from attributes
                output_group = f[base_path]
                outputTime = output_group.attrs['outputTime']
                
                # Convert outputTime to redshift
                redshift = outputTime_to_redshift(outputTime, self.cosmo)
                
                # Read SFH datasets
                diskSFH_dataset = f[config['diskSFH']]
                spheroidSFH_dataset = f[config['spheroidSFH']]
                
                # Get time and metallicity bins from dataset attributes
                # Times are age-of-universe bin edges
                disk_times_age_of_universe = diskSFH_dataset.attrs['time']
                spheroid_times_age_of_universe = spheroidSFH_dataset.attrs['time']
                
                # Convert to lookback times (stellar ages)
                # Note: SFH bins represent [time[i-1], time[i]), so we take bin centers
                # Actually, the documentation says times are the maximum time for each bin
                # So we convert directly
                disk_times_lookback = age_of_universe_to_lookback_time(
                    disk_times_age_of_universe, outputTime
                )
                spheroid_times_lookback = age_of_universe_to_lookback_time(
                    spheroid_times_age_of_universe, outputTime
                )
                
                # Read SFH data for this galaxy
                # Note: Data is stored as [metallicity x time] just like lightcone
                diskSFH = np.array([list(x) for x in diskSFH_dataset[galIndex]], dtype=float)
                spheroidSFH = np.array([list(x) for x in spheroidSFH_dataset[galIndex]], dtype=float)
                
                self.galData = {
                    'redshift': redshift,
                    'diskSFH': diskSFH,
                    'diskSFH_times': disk_times_lookback,
                    'spheroidSFH': spheroidSFH,
                    'spheroidSFH_times': spheroid_times_lookback,
                    'format_type': format_type,
                    'outputTime': outputTime,
                }
            
            else:
                raise ValueError(f"Unknown format type: {format_type}")
        
        return self.galData

    def calculate_sed_for_galacticus_galaxy(self, filename, galIndex, wavelengths):
        self.read_galacticus_galaxy(filename, galIndex, checkData=False)
        observed_sed = self.calculate_observed_frame_sed(
            self.galData['diskSFH'], 
            self.galData['redshift'], 
            wavelengths, 
            extrapolateWithZeros=False
        )
        return observed_sed

    def evaluate_component_spectrum(self, filename, galIndex, component='disk', obs_wavelengths=None, include_emission_lines=True, lineFWHM=10*u.AA, minimumLineFlux=0, minimumLineWavelength = 0.9*u.micron, maximumLineWavelength = 2.03*u.micron, use_synphot=True, dust_model=None, dust_params=None, dust_law='calzetti', random_uniform_index=None):
        """
        Evaluate the spectrum of a specified galaxy component.

        This method calculates the spectrum of a specified component (disk, spheroid, or AGN) of a Galacticus galaxy.
        It reads the galaxy data from the specified file, extracts the continuum (for disk and spheroid components),
        and adds emission lines to produce the total spectrum.

        Parameters
        ----------
        filename : str
            The path to the Galacticus HDF5 file containing galaxy data.
        galIndex : int
            The index of the galaxy within the Galacticus file to evaluate.
        component : str, optional
            The name of the galaxy component to evaluate. Must be one of 'disk', 'spheroid', or 'AGN'.
            Default is 'disk'.
        obs_wavelengths :  array-like or Quantity, optional
            The wavelengths at which to evaluate the continuum spectrum. Can be supplied with astropy units,
            or if not, will be assumed to be in Angstroms. When `use_synphot=True`, the final
            wavelength array used for the spectrum may differ because synphot uses variable wavelength
            resolution to capture narrow emission lines. When `use_synphot=False`, this parameter is required
            and the spectrum will be evaluated exactly on this grid. Default is None.
        include_emission_lines : bool, optional
            Whether to include emission lines in the spectrum. If False, only the continuum is returned.
            Default is True.
        lineFWHM : Quantity, optional
            The full width at half maximum (FWHM) of the emission lines, specified as an astropy Quantity
            (or else assumed to be in Angstroms).
            Default is 10 Angstroms.
        minimumLineFlux : float or Quantity, optional
            The minimum line flux for an emission line to be included in the spectrum (strictly speaking
            it is the maximum line flux for an emission line to not be included).
            Can be supplied as a float (assumed to be in erg / (s cm^2)) or as an astropy Quantity in any valid
            flux unit.
            Default is 0 (include all lines with non-zero flux).
        minimumLineWavelength : Quantity, optional
            The minimum observed wavelength for an emission line to be included in the spectrum.
            Default is 0.9 micron.
        maximumLineWavelength : Quantity, optional
            The maximum observed wavelength for an emission line to be included in the spectrum.
            Default is 2.03 micron.
        use_synphot : bool, optional
            Whether to use synphot for spectrum generation. When True (default), returns a synphot.SourceSpectrum
            object. When False, uses direct numpy operations for faster performance and returns a tuple of
            (wavelength, flux_density) arrays. Default is True for backward compatibility.
        dust_model : str, optional
            Name of the dust attenuation model to use for emission lines. Currently supported:
            - 'gb10_generalised': Generalized Garn & Best (2010) model
            - None: No dust attenuation applied (default)
        dust_params : dict, optional
            Dictionary of parameters for the dust model. Required if dust_model is not None.
            For 'gb10_generalised' model, expected parameters are:
            - 'delta_0': Constant offset term
            - 'delta_z': Redshift coefficient
            - 'delta_M': Stellar mass coefficient
            - 'delta_Mz': Mass-redshift coupling coefficient
            - 'attenuation_scatter': Log-normal scatter (optional, default 0.0)
            Example: {'delta_0': 0.275, 'delta_z': -1.614, 'delta_M': -0.834, 
                      'delta_Mz': -0.708, 'attenuation_scatter': 0.25}
        dust_law : str, optional
            Name of the dust attenuation law describing wavelength dependence.
            Currently only 'calzetti' is supported. Default is 'calzetti'.
        random_uniform_index : int, optional
            Index to select from the randomUniform dataset in the HDF5 file.
            If provided, the random number at randomUniform[galIndex, random_uniform_index]
            will be used for dust attenuation scatter. If None, scatter is generated
            using numpy's random number generator. Default is None.
            Example: random_uniform_index=2 uses the 3rd random number for each galaxy.

        Returns
        -------
        component_spectrum : synphot.SourceSpectrum
            Returns a synphot SourceSpectrum object. The spectrum includes both the continuum
            and emission lines (if `include_emission_lines` is True).
            - `component_spectrum.waveset` returns a wavelength array with astropy units.
            - `component_spectrum(wav, flux_unit='FNU')` returns the Fnu flux at the wavelengths `wav`, also with astropy units.

        Raises
        ------
        ValueError
            If the star formation history parameters in the Galacticus file are incompatible
            with the SED template, or if `use_synphot=False` but `obs_wavelengths=None`.

        Notes
        -----
        - For 'disk' and 'spheroid' components, the continuum is calculated based on the star formation history (SFH) and `self.sedTemplate`.
        - For the 'AGN' component, the continuum is set to zero, and only emission lines are included (if `include_emission_lines` is True).
        - Emission lines are modeled as Gaussian profiles with the specified `lineFWHM`.
        - When `use_synphot=True`, the method relies on the synphot library for spectrum calculations.
        - When `use_synphot=False`, direct numpy operations are used for faster performance (~2-4 speedup), though the final result is still
          cast as a synphot.SourceSpectrum object.
        - This method validates that the SFH binning in the Galacticus file matches the SED template binning.
        """
        valid_components = ['disk', 'spheroid', 'AGN']
        if component not in valid_components:
            raise ValueError(f"Invalid component '{component}'. Must be one of {valid_components}.")
        
        # When not using synphot, obs_wavelengths must be provided
        if not use_synphot and obs_wavelengths is None:
            raise ValueError("obs_wavelengths must be provided when use_synphot=False")
        
        # Validate SFH compatibility before processing
        self.validate_sfh_compatibility(filename)
        
        galData = self.read_galacticus_galaxy(filename, galIndex)
        redshift = galData['redshift']
        
        if component in ['disk', 'spheroid']:
            SFH = galData[f'{component}SFH']
            # Handle galaxies with empty SFH (e.g. due to having no spheroid)
            if SFH.size == 0:
                # Create zero continuum flux
                if obs_wavelengths is None:
                    obs_wavelengths = np.linspace(8000, 30000, 1000) * u.AA
                if use_synphot:
                    continuum_flux = SourceSpectrum(Empirical1D, points=obs_wavelengths, 
                                                lookup_table=np.zeros(len(obs_wavelengths)))
                else:
                    continuum_wav = process_wavelength_array(obs_wavelengths)
                    continuum_Fnu = np.zeros(len(continuum_wav)) * u.erg / (u.s * u.Hz * u.cm**2)
            else:
                Fnu, wav = self.calculate_continuum_Fnu(SFH, redshift, obs_wavelengths, extrapolateWithZeros=True)
                if use_synphot:
                    continuum_flux = SourceSpectrum(Empirical1D, points=wav, lookup_table=Fnu)
                else:
                    continuum_wav = wav
                    continuum_Fnu = Fnu
        elif component == 'AGN':
            # zero continuum flux
            if obs_wavelengths is None:
                # need some default to create the SourceSpectrum
                obs_wavelengths = np.linspace(8000, 30000, 1000) * u.AA  # default range in Angstroms
            if use_synphot:
                continuum_flux = SourceSpectrum(Empirical1D, points=obs_wavelengths, lookup_table=np.zeros(len(obs_wavelengths)))
            else:
                continuum_wav = process_wavelength_array(obs_wavelengths)
                continuum_Fnu = np.zeros(len(continuum_wav)) * u.erg / (u.s * u.Hz * u.cm**2)

        # now loop over the emission lines adding them to the continuum
        if use_synphot:
            total_flux = continuum_flux
        else:
            # For non-synphot path, we'll accumulate flux in Fnu units
            total_Fnu = continuum_Fnu.copy()
        # Calculate dust attenuation if a dust model is specified
        A_Halpha = None
        if dust_model is not None:
            if dust_params is None:
                raise ValueError(f"dust_params must be provided when dust_model='{dust_model}'")
            
            if dust_model == 'gb10_generalised':
                # Read stellar mass from the Galacticus file
                with h5py.File(filename, 'r') as f:
                    # Detect format to get proper paths
                    if filename not in self._file_formats:
                        format_type, base_path = detect_galacticus_format(filename)
                        self._file_formats[filename] = (format_type, base_path)
                    else:
                        format_type, base_path = self._file_formats[filename]
                    
                    # Construct paths based on format
                    if format_type == 'lightcone':
                        disk_mass_path = f'{base_path}/nodeData/diskMassStellar'
                        spheroid_mass_path = f'{base_path}/nodeData/spheroidMassStellar'
                        random_uniform_path = f'{base_path}/nodeData/randomUniform'
                    else:  # fixed-time
                        disk_mass_path = f'{base_path}/nodeData/diskMassStellar'
                        spheroid_mass_path = f'{base_path}/nodeData/spheroidMassStellar'
                        random_uniform_path = f'{base_path}/nodeData/randomUniform'
                    
                    # Read stellar masses (in solar masses)
                    disk_mass = f[disk_mass_path][galIndex] if disk_mass_path in f else 0.0
                    spheroid_mass = f[spheroid_mass_path][galIndex] if spheroid_mass_path in f else 0.0
                    total_stellar_mass = disk_mass + spheroid_mass
                    
                    # Read random uniform value if requested
                    random_uniform_value = None
                    if random_uniform_index is not None:
                        if random_uniform_path in f:
                            random_uniform_dataset = f[random_uniform_path]
                            # Check if the index is valid
                            if random_uniform_dataset.ndim == 2:
                                if random_uniform_index < random_uniform_dataset.shape[1]:
                                    random_uniform_value = random_uniform_dataset[galIndex, random_uniform_index]
                                else:
                                    raise ValueError(f"random_uniform_index={random_uniform_index} is out of bounds. "
                                                   f"Dataset has {random_uniform_dataset.shape[1]} random numbers per galaxy.")
                            else:
                                raise ValueError(f"randomUniform dataset has unexpected shape: {random_uniform_dataset.shape}. "
                                               f"Expected 2D array (Ngal x Nrand).")
                        else:
                            raise ValueError(f"random_uniform_index specified but {random_uniform_path} not found in file.")
                
                # Prepare dust_params with random_uniform if available
                dust_params_copy = dust_params.copy()
                if random_uniform_value is not None:
                    dust_params_copy['random_uniform'] = np.array([random_uniform_value])
                
                # Calculate dust attenuation at H-alpha
                A_Halpha = dust_attenuation_gb10_generalised(
                    total_stellar_mass, redshift, **dust_params_copy
                )
            else:
                raise ValueError(f"Dust model '{dust_model}' not supported. Currently only 'gb10_generalised' is implemented.")
        
        if include_emission_lines:
            minimumLineFlux = minFlux(minimumLineFlux)
            # Get cached line metadata (names and wavelengths are same for all galaxies)
            lineNames, lineRestWavelengths, hdf5_paths = self._get_line_metadata(filename, component)
            
            # Read only the luminosities for this specific galaxy
            with h5py.File(filename, 'r') as f:
                lineLuminosities = np.array([f[path][galIndex] for path in hdf5_paths])
            
            for lineName, lineRestWavelength, lineLuminosity in zip(lineNames, lineRestWavelengths, lineLuminosities):
                lineRestWavelength = lineRestWavelength * u.AA
                lineWavelength = lineRestWavelength * (1 + redshift)
                if (lineWavelength < minimumLineWavelength) or (lineWavelength > maximumLineWavelength):
                    continue
                lineFlux = lineLuminosity * (u.erg/u.s) / (4 * np.pi * (self.cosmo.luminosity_distance(redshift).to(u.cm))**2)
                
                # Apply dust attenuation if specified
                if A_Halpha is not None:
                    lineFlux = apply_dust_attenuation_to_line(lineFlux, lineRestWavelength, A_Halpha, dust_law=dust_law)
                
                if lineFlux <= minimumLineFlux:
                    continue
                if use_synphot:
                    line_flux = SourceSpectrum(GaussianFlux1D, total_flux=lineFlux, mean=lineWavelength, fwhm=lineFWHM)
                    total_flux += line_flux
                else:
                    # Add Gaussian emission line directly to the flux array
                    # gaussian_from_fwhm returns flux per Angstrom
                    line_flux_per_AA = gaussian_from_fwhm(continuum_wav, lineWavelength, lineFWHM, lineFlux)
                    # Convert from F_lambda (erg/(s cm^2 AA)) to F_nu (erg/(s cm^2 Hz))
                    # F_nu = F_lambda * lambda^2 / c
                    line_Fnu = line_flux_per_AA * continuum_wav**2 / const.c
                    total_Fnu += line_Fnu.to('erg/(s cm^2 Hz)')
        
        if not use_synphot:
            total_flux = SourceSpectrum(Empirical1D, points=continuum_wav, lookup_table=total_Fnu)
        component_spectrum = total_flux
        return component_spectrum
    
    def evaluate_total_spectrum(self, filename, galIndex, includeAGN=True, obs_wavelengths=np.linspace(8000, 30000, 1000)*u.AA, lineFWHM=10*u.AA, include_emission_lines=True, minimumLineFlux=0, minimumLineWavelength = 0.9*u.micron, maximumLineWavelength = 2.03*u.micron, use_synphot=True, dust_model=None, dust_params=None, dust_law='calzetti', random_uniform_index=None):
        components=['disk','spheroid']
        if includeAGN:
            components.append('AGN')    
        for i,component in enumerate(components):
            spectrum = self.evaluate_component_spectrum(filename, galIndex, component=component, obs_wavelengths=obs_wavelengths, lineFWHM=lineFWHM, include_emission_lines=include_emission_lines, minimumLineFlux=minimumLineFlux, minimumLineWavelength=minimumLineWavelength, maximumLineWavelength=maximumLineWavelength, use_synphot=use_synphot, dust_model=dust_model, dust_params=dust_params, dust_law=dust_law, random_uniform_index=random_uniform_index)
            if i==0:
                total_spectrum = spectrum
            else:
                total_spectrum += spectrum
        return total_spectrum
    
    def calculate_magnitudes(self, filename, galIndex, bandpasses, component='total', 
                            magnitude_system='AB', obs_wavelengths=np.linspace(8000, 30000, 1000)*u.AA,
                            includeAGN=True, lineFWHM=10*u.AA):
        """
        Calculate observed magnitudes for a galaxy in multiple bandpasses.
        
        This method calculates magnitudes by generating a galaxy spectrum and passing it
        through the specified bandpass filters. For catalog-scale processing, bandpass
        objects should be loaded once and reused across galaxies for efficiency.
        
        Parameters
        ----------
        filename : str
            Path to Galacticus HDF5 file
        galIndex : int
            Galaxy index in the catalog
        bandpasses : dict
            Dictionary mapping filter names to synphot SpectralElement objects.
            Example: {'F158': bandpass_f158, 'F184': bandpass_f184}
        component : str, optional
            Component to calculate magnitudes for. Must be one of:
            - 'disk': disk component only
            - 'spheroid': spheroid component only  
            - 'total': combined disk + spheroid (and optionally AGN)
            Default is 'total'.
        magnitude_system : str, optional
            Magnitude system to use. Must be one of:
            - 'AB': AB magnitude system (default)
            - 'ST': ST magnitude system
            - 'Vega': Vega magnitude system
            Default is 'AB'.
        obs_wavelengths : array-like or Quantity, optional
            Wavelengths at which to evaluate the spectrum. Default is 
            np.linspace(8000, 30000, 1000)*u.AA
        includeAGN : bool, optional
            Whether to include AGN component when component='total'. 
            Default is True.
        lineFWHM : Quantity, optional
            Full width at half maximum of emission lines.
            Default is 10 Angstroms.
        
        Returns
        -------
        magnitudes : dict
            Dictionary mapping filter names to magnitude values (float).
            Returns NaN for filters where the calculation fails.
        
        Examples
        --------
        >>> # Load bandpasses once for reuse
        >>> import stpsf
        >>> roman = stpsf.WFI()
        >>> bandpasses = {
        ...     'F158': roman._get_synphot_bandpass('F158'),
        ...     'F184': roman._get_synphot_bandpass('F184')
        ... }
        >>> 
        >>> # Calculate magnitudes for a galaxy
        >>> calc = sed_calculator('sed_template.hdf5')
        >>> mags = calc.calculate_magnitudes('catalog.hdf5', galIndex=0, 
        ...                                    bandpasses=bandpasses)
        >>> print(mags)
        {'F158': 23.45, 'F184': 23.12}
        
        Notes
        -----
        For best performance when processing large catalogs:
        1. Load bandpass objects once before the loop
        2. Reuse the same sed_calculator instance for all galaxies
        3. Use the same obs_wavelengths for all galaxies
        """
        from synphot import Observation
        
        # Get the spectrum for the specified component
        if component == 'total':
            spectrum = self.evaluate_total_spectrum(filename, galIndex, 
                                                    includeAGN=includeAGN,
                                                    obs_wavelengths=obs_wavelengths,
                                                    lineFWHM=lineFWHM)
        elif component in ['disk', 'spheroid', 'AGN']:
            spectrum = self.evaluate_component_spectrum(filename, galIndex,
                                                        component=component,
                                                        obs_wavelengths=obs_wavelengths,
                                                        lineFWHM=lineFWHM)
        else:
            raise ValueError(f"Invalid component '{component}'. Must be 'disk', 'spheroid', 'AGN', or 'total'.")
        
        # Calculate magnitudes in each bandpass
        magnitudes = {}
        for filter_name, bandpass in bandpasses.items():
            try:
                # Create observation by passing spectrum through bandpass
                obs = Observation(spectrum, bandpass, force='taper')
                
                # Calculate magnitude in the specified system
                if magnitude_system == 'AB':
                    mag = obs.effstim(flux_unit=u.ABmag)
                elif magnitude_system == 'ST':
                    mag = obs.effstim(flux_unit=u.STmag)
                elif magnitude_system == 'Vega':
                    # Vega magnitudes require a Vega spectrum
                    vega = SourceSpectrum.from_vega()
                    mag = obs.effstim(flux_unit='vegamag', vegaspec=vega)
                else:
                    raise ValueError(f"Invalid magnitude_system '{magnitude_system}'. Must be 'AB', 'ST', or 'Vega'.")
                
                magnitudes[filter_name] = mag.value
                
            except Exception as e:
                # If magnitude calculation fails for a filter, store NaN
                print(f"Warning: Failed to calculate {filter_name} magnitude for galaxy {galIndex}: {e}")
                magnitudes[filter_name] = np.nan
        
        return magnitudes
