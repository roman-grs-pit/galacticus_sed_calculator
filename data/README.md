# Example data

This directory contains small Galacticus products used by the examples and
tests in this repository. They are included so that the main SED and magnitude
workflows can be tried without first generating a Galacticus lightcone.

## Galacticus lightcone catalogs

- `romanUNIT.hdf5` is a five-galaxy example lightcone containing the saved star
  formation histories and other galaxy properties required by the SED
  calculator.
- `romanUNIT_with_magnitudes.hdf5` is a copy of the same example after Roman
  WFI magnitudes have been calculated.
- `romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5` is a larger example
  containing 1,186 galaxies, saved star formation histories, Roman WFI
  magnitudes, and sky coordinates.

These are Galacticus-generated files rather than original UNIT simulation
outputs. Each file retains the Galacticus model parameters and the Galacticus
and Galacticus-datasets Git hashes under its `/Parameters` and `/Version`
groups. The catalogs were generated from UNIT merger trees for work on Roman
Galaxy Redshift Survey mock catalogs.

Users of these examples should cite Galacticus and the UNIT simulations as
appropriate for their application:

- Benson, A. J. (2012), "Galacticus: A semi-analytic model of galaxy
  formation", *New Astronomy*, 17, 175-197,
  <https://doi.org/10.1016/j.newast.2011.07.004>.
- Chuang, C.-H. et al. (2019), "UNIT project: Universe N-body simulations for
  the Investigation of Theoretical models from galaxy surveys", *Monthly
  Notices of the Royal Astronomical Society*, 487, 48-59,
  <https://doi.org/10.1093/mnras/stz1233>.

The Galacticus source code is available at
<https://github.com/galacticusorg/galacticus>.

## SED templates

- `nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5`
- `nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.003.hdf5`

These files were generated with the Galacticus `nodePropertyExtractorSED`
component. They contain a tabulated mapping from stellar age and metallicity to
spectral luminosity. The age grid of an SED template must match the saved star
formation histories in the Galacticus catalog; the calculator reports an
incompatibility when it does not.

The `ageMinimum0.001` template is compatible with the example lightcone
catalogs in this directory. The `ageMinimum0.003` template is retained for
testing the compatibility checks.

## Dust configurations

The YAML files under `dustModel/` are example dust-attenuation configurations.
The emission-line model parameters were calibrated against the Sobral et al.
(2013) H-alpha luminosity functions for a particular Galacticus calibration;
the comments in each file link to the relevant paper and repository discussion.
They are examples rather than universal dust-model parameters.

## Observational diagnostic tables

The repository also contains binned COSMOS-Web/COSMOS2025 F150W reference data
under
`galacticus_sed_calculator/diagnostics/observational_data/f158/`. These tables
were calculated from `COSMOSWeb_mastercatalog_v1.1.fits`; they do not contain the
original object catalog. The accompanying JSON file records the source catalog,
selection, survey area, columns, and binning used to construct them.

When using these tables in scientific work, cite Shuntov, M. et al. (2025),
"COSMOS2025: The COSMOS-Web galaxy catalog of photometry, morphology,
redshifts, and physical parameters from JWST, HST, and ground-based imaging",
*Astronomy & Astrophysics*, 704, A339,
<https://doi.org/10.1051/0004-6361/202555799>. The catalog and its documentation
are available at <https://cosmos2025.iap.fr/>.
