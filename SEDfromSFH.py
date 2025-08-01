import numpy as np
import h5py 
import dust_model.galacticus as galacticus
import astropy.units as u
from astropy.cosmology import Planck15 
import re
import synphot
from synphot.models import Empirical1D, GaussianFlux1D
from synphot import units, SourceSpectrum
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

def getLineProperties(fname, component='disk', galIndex=None, hdf5_base_path = '/Lightcone/Output1/nodeData/luminosityEmissionLine'):
    """
    Get the emission line names, wavelengths, and luminosities for a given component from the Galacticus output file.
    
    Parameters:
    - fname: str, path to the Galacticus output HDF5 file.
    - component: str, 'disk', 'AGN', or 'spheroid' to specify which component's emission lines to retrieve.
    - galIndex: int or None, index of the galaxy to retrieve line luminosities for. If None, retrieves all galaxies.
    - hdf5_base_path: str, base path for the HDF5 file structure where emission line data is stored.
    
    Returns:
    - lineNames: np.ndarray of emission line names.
    - lineWavelengths: np.ndarray of wavelengths corresponding to the emission lines.
    - lineLuminosities: np.ndarray of luminosities for the emission lines.
    """
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

class sed_calculator:
    def __init__(self, sedTemplateFilename, config=galacticus_sed_config, cosmology=Planck15):
        self.sedTemplateFilename = sedTemplateFilename
        self.load_sed_template()
        self.config = config
        self.cosmo = cosmology

    def load_sed_template(self):
        # Load the SED template from the given filename
        with h5py.File(self.sedTemplateFilename, 'r') as f:
            self.sedTemplate = f['sedTemplate'][:]
            self.sedAges = f['ages'][:]
            self.sedMetallicity = f['metallicity'][:]
            self.sedWavelength = f['wavelength'][:]

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
        # read a Galacticus catalog file (hdf5) and get properties of one galaxy
        with h5py.File(filename, 'r') as f:
            self.galData = {
                'redshift': f[self.config['redshift']][galIndex],
                'diskSFH': np.array([list(x) for x in f[self.config['diskSFH']][galIndex]], dtype=float),
                'diskSFH_times': f[self.config['diskSFH_times']][galIndex],
                'spheroidSFH': np.array([list(x) for x in f[self.config['spheroidSFH']][galIndex]], dtype=float),
                'spheroidSFH_times': f[self.config['spheroidSFH_times']][galIndex],
            }
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

    def evaluate_component_spectrum(self, filename, galIndex, component='disk', obs_wavelengths=None, include_emission_lines=True, lineFWHM=10*u.AA):
        """
        Evaluate the spectrum of a specified galaxy component and return it as a synphot Spectrum1D object.

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
            or if not, will be assumed to be in Angstroms. Note that the final
            wavelength array used for the spectrum may differ because synphot uses variable wavelength
            resolution to capture narrow emission lines. Default is None.
        include_emission_lines : bool, optional
            Whether to include emission lines in the spectrum. If False, only the continuum is returned.
            Default is True.
        lineFWHM : Quantity, optional
            The full width at half maximum (FWHM) of the emission lines, specified as an astropy Quantity
            (or else assumed to be in Angstroms).
            Default is 10 Angstroms.

        Returns
        -------
        component_spectrum : synphot.SourceSpectrum
            The evaluated spectrum as a synphot SourceSpectrum object. The spectrum includes both the continuum
            and emission lines (if `include_emission_lines` is True).
            - `component_spectrum.waveset` returns a wavelength array with astropy units.
            - `component_spectrum(wav, flux_unit='FNU')` returns the Fnu flux at the wavelengths `wav`, also with astropy units.

        Notes
        -----
        - For 'disk' and 'spheroid' components, the continuum is calculated based on the star formation history (SFH) and `self.sedTemplate`.
        - For the 'AGN' component, the continuum is set to zero, and only emission lines are included (if `include_emission_lines` is True).
        - Emission lines are modeled as Gaussian profiles with the specified `lineFWHM`.
        - The method relies on the synphot library for spectrum calculations and assumes the Galacticus data is structured correctly.
        """
        valid_components = ['disk', 'spheroid', 'AGN']
        if component not in valid_components:
            raise ValueError(f"Invalid component '{component}'. Must be one of {valid_components}.")
        galData = self.read_galacticus_galaxy(filename, galIndex)
        redshift = galData['redshift']
        if component in ['disk', 'spheroid']:
            SFH = galData[f'{component}SFH']
            Fnu, wav = self.calculate_continuum_Fnu(SFH, redshift, obs_wavelengths, extrapolateWithZeros=True)
            continuum_flux = SourceSpectrum(Empirical1D, points=wav, lookup_table=Fnu)
        elif component == 'AGN':
            # zero continuum flux
            if obs_wavelengths is None:
                # need some default to create the SourceSpectrum
                obs_wavelengths = np.linspace(8000, 30000, 1000) * u.AA  # default range in Angstroms
            continuum_flux = SourceSpectrum(Empirical1D, points=obs_wavelengths, lookup_table=np.zeros(len(obs_wavelengths)))

        # now loop over the emission lines adding them to the continuum
        total_flux = continuum_flux
        if include_emission_lines:
            lineNames, lineRestWavelengths, lineLuminosities = getLineProperties(filename, component=component, galIndex=galIndex)
            for lineName, lineRestWavelength, lineLuminosity in zip(lineNames, lineRestWavelengths, lineLuminosities):
                lineFlux = lineLuminosity * (u.erg/u.s) / (4 * np.pi * (self.cosmo.luminosity_distance(redshift).to(u.cm))**2)
                lineWavelength = (lineRestWavelength * u.AA) * (1 + redshift)
                line_flux = SourceSpectrum(GaussianFlux1D, total_flux=lineFlux, mean=lineWavelength, fwhm=lineFWHM)
                total_flux += line_flux
        component_spectrum = total_flux
        return component_spectrum
    
    def evaluate_total_spectrum(self, filename, galIndex, includeAGN=True, obs_wavelengths=np.linspace(8000, 30000, 1000)*u.AA, lineFWHM=10*u.AA):
        components=['disk','spheroid']
        if includeAGN:
            components.append('AGN')    
        for i,component in enumerate(components):
            spectrum = self.evaluate_component_spectrum(filename, galIndex, component=component, obs_wavelengths=obs_wavelengths, lineFWHM=lineFWHM)
            if i==0:
                total_spectrum = spectrum
            else:
                total_spectrum += spectrum
        return total_spectrum
