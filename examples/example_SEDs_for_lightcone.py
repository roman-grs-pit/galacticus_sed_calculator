"""Calculate and plot spectra for one galaxy in a Galacticus lightcone."""

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import LambdaCDM

from galacticus_sed_calculator import SEDCalculator


repository_root = Path(__file__).resolve().parents[1]
catalog = repository_root / "data/romanUNIT.hdf5"
template = (
    repository_root
    / "data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5"
)
unit_cosmology = LambdaCDM(H0=67.74, Om0=0.3089, Ode0=0.6911)

calculator = SEDCalculator(template, cosmology=unit_cosmology)
galaxy_index = 0
galaxy = calculator.read_galacticus_galaxy(catalog, galaxy_index)
observed_wavelength = np.linspace(8000, 25000, 3000) * u.AA

component_spectra = {
    component: calculator.evaluate_component_spectrum(
        catalog,
        galIndex=galaxy_index,
        component=component,
        obs_wavelengths=observed_wavelength,
    )
    for component in ("disk", "spheroid", "AGN")
}
total_spectrum = calculator.evaluate_total_spectrum(
    catalog,
    galIndex=galaxy_index,
    obs_wavelengths=observed_wavelength,
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
    label="total",
    linewidth=1.5,
)
axis.set(
    xlabel="Observed wavelength [micron]",
    ylabel="Flux density [FLAM]",
    title=f"Galacticus galaxy {galaxy_index}, z={galaxy['redshift']:.3f}",
)
axis.set_yscale("log")
axis.legend()
figure.tight_layout()

output_path = Path(__file__).with_name("sed_components_example.png")
figure.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
