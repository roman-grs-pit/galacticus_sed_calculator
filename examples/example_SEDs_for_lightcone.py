"""Prepare a dust-attenuated spectrum for one lightcone galaxy.

This is a compact starting point for generating spectra to inject into grism
simulations. The cosmology, sky position, observed redshift, and dust model are
all read from the Galacticus catalogue.
"""

from pathlib import Path

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import LambdaCDM

from galacticus_sed_calculator import SEDCalculator, read_dust_model_from_catalog


def read_cosmology_from_catalog(catalog):
    """Construct an Astropy cosmology from Galacticus file metadata."""
    with h5py.File(catalog, "r") as handle:
        attributes = handle["/Parameters/cosmologyParameters"].attrs
        return LambdaCDM(
            H0=float(attributes["HubbleConstant"]),
            Om0=float(attributes["OmegaMatter"]),
            Ode0=float(attributes["OmegaDarkEnergy"]),
            Ob0=float(attributes["OmegaBaryon"]),
            Tcmb0=float(attributes["temperatureCMB"]) * u.K,
        )


repository_root = Path(__file__).resolve().parents[1]
catalog = (
    repository_root
    / "data/romanUNIT-d1_4sqDeg_SFH_withMags_with_coordinates.hdf5"
)
template = (
    repository_root
    / "data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5"
)

calculator = SEDCalculator(template, cosmology=read_cosmology_from_catalog(catalog))
dust_configuration = read_dust_model_from_catalog(catalog)

# This galaxy has H-alpha within the Roman grism wavelength range.
galaxy_index = 593
observed_wavelength = np.linspace(8000, 25000, 3000) * u.AA

with h5py.File(catalog, "r") as handle:
    node_data = handle["/Lightcone/Output1/nodeData"]
    right_ascension = node_data["rightAscension"][galaxy_index]
    declination = node_data["declination"][galaxy_index]
    output_redshift = node_data["redshift"][galaxy_index]
    observed_redshift = node_data["lightconeRedshiftObserved"][galaxy_index]

print(f"Galaxy index: {galaxy_index}")
print(f"Right ascension: {right_ascension:.6f} deg")
print(f"Declination: {declination:.6f} deg")
print(f"Observed redshift: {observed_redshift:.6f}")
print(f"Output-epoch redshift: {output_redshift:.6f}")
print(
    "Use lightconeRedshiftObserved for the measurable redshift: unlike "
    "nodeData/redshift, it includes the line-of-sight peculiar velocity."
)

component_spectra = {
    component: calculator.evaluate_component_spectrum(
        catalog,
        galIndex=galaxy_index,
        component=component,
        obs_wavelengths=observed_wavelength,
        **dust_configuration,
    )
    for component in ("disk", "spheroid")
}
total_spectrum = calculator.evaluate_total_spectrum(
    catalog,
    galIndex=galaxy_index,
    obs_wavelengths=observed_wavelength,
    **dust_configuration,
)

figure, axis = plt.subplots(figsize=(8, 4.5))
for component, spectrum in component_spectra.items():
    axis.plot(
        observed_wavelength.to_value(u.micron),
        spectrum(observed_wavelength, flux_unit="flam"),
        label=component,
        alpha=0.8,
    )
axis.plot(
    observed_wavelength.to_value(u.micron),
    total_spectrum(observed_wavelength, flux_unit="flam"),
    color="black",
    label="total (including AGN lines)",
    linewidth=1.5,
)
axis.set(
    xlabel="Observed wavelength [micron]",
    ylabel="Flux density [FLAM]",
    title=f"Galacticus galaxy {galaxy_index}, z={observed_redshift:.3f}",
)
axis.set_yscale("log")
axis.legend()
figure.tight_layout()

output_path = Path(__file__).with_name("sed_components_example.png")
figure.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
