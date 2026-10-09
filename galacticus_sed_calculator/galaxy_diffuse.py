"""Map Galacticus galaxies to disk-dust transfer, keeping source light separate.

This is a diffuse-only building block, not the complete clouds_diffuse model.
Local stellar/nebular attenuation can be applied before this layer. It never
applies an empirical line model or interprets one as local attenuation.
"""

from dataclasses import asdict, dataclass, field
from enum import IntFlag

import astropy.units as u
import numpy as np


class DiffuseStatus(IntFlag):
    """Stable schema-v1 bitmask; zero means no flagged conditions."""

    IGNORED_SPHEROID_METALS = 1
    DISK_METALS_SANITIZED = 2
    SPHEROID_METALS_SANITIZED = 4
    LOW_OPTICAL_DEPTH_BLEND = 8
    OPTICAL_DEPTH_CLIPPED = 16
    OPTICAL_DEPTH_EXTRAPOLATED = 32
    SCALE_RATIO_CLIPPED = 64
    WAVELENGTH_ENDPOINT_ROUNDOFF = 128
    EXTRAPOLATION_POSITIVE_SLOPE = 256


def _finite_scalar(name, value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


@dataclass(frozen=True)
class DiffuseDustParameters:
    dust_to_metals_ratio: float = 0.334
    cloud_fraction: float = 0.25
    negative_metal_tolerance_msun: float = 0.0

    def __post_init__(self):
        for name in ("dust_to_metals_ratio", "cloud_fraction", "negative_metal_tolerance_msun"):
            value = _finite_scalar(name, getattr(self, name))
            if value < 0 or (name != "negative_metal_tolerance_msun" and value > 1):
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class DiffuseGalaxyInputs:
    """Physical inputs, with scale radii in Mpc and metal masses in Msun.

    Disk radius is the exponential radial scale; spheroid radius is the
    Hernquist scale. Neither is a half-light radius. Missing/invalid radii are
    allowed only when irrelevant to the requested light and dust medium.
    Explicit inclination takes precedence over the stored orientation deviate.
    """

    disk_metal_mass_msun: float
    spheroid_metal_mass_msun: float
    disk_scale_radius_mpc: float | None = None
    spheroid_scale_radius_mpc: float | None = None
    orientation_uniform: float | None = None
    inclination_degrees: float | None = None
    catalog_units_in_si: dict = field(default_factory=dict)
    catalog_profiles: dict = field(default_factory=dict)


def read_diffuse_galaxy_inputs(node_data, galaxy_index, *, inclination_degrees=None):
    """Read one row from an open lightcone or fixed-time ``nodeData`` group.

    Honors each field's ``unitsInSI`` rather than assuming a particular solar
    mass constant. These standard Galacticus field names denote scale radii.
    Missing unit metadata is an error; missing radii are deferred until their
    relevance is known. Never draw a new orientation or reuse GB10's column 0.
    """
    if not isinstance(galaxy_index, (int, np.integer)) or galaxy_index < 0:
        raise ValueError("galaxy_index must be a nonnegative integer")
    units = {}

    def read(name, target_si, optional=False):
        if name not in node_data:
            if optional:
                return None
            raise ValueError(f"Missing required diffuse-dust field: {name}")
        ds = node_data[name]
        if ds.ndim != 1 or galaxy_index >= len(ds):
            raise ValueError(f"Invalid row/shape for {name}")
        if "unitsInSI" not in ds.attrs:
            raise ValueError(f"Missing unitsInSI for {name}; units must be explicit")
        factor = _finite_scalar(f"{name}.unitsInSI", ds.attrs["unitsInSI"])
        if factor <= 0:
            raise ValueError(f"{name}.unitsInSI must be positive")
        units[name] = factor
        return float(ds[galaxy_index]) * factor / target_si

    disk_metals = read("diskAbundancesGasMetals", u.Msun.to(u.kg))
    spheroid_metals = read("spheroidAbundancesGasMetals", u.Msun.to(u.kg))
    disk_radius = read("diskRadius", u.Mpc.to(u.m), optional=True)
    spheroid_radius = read("spheroidRadius", u.Mpc.to(u.m), optional=True)
    orientation = None
    if inclination_degrees is None:
        if "randomUniform" not in node_data:
            raise ValueError("Diffuse dust requires randomUniform[:,1] or an explicit inclination_degrees")
        ds = node_data["randomUniform"]
        if ds.ndim != 2 or ds.shape[1] < 2 or galaxy_index >= ds.shape[0]:
            raise ValueError("randomUniform must have shape (Ngal, Nrand>=2) and contain the requested row")
        orientation = float(ds[galaxy_index, 1])
    profiles = {}
    for component in ('Disk', 'Spheroid'):
        group = node_data.file.get(f'Parameters/component{component}')
        attribute = f'massDistribution{component}'
        if group is not None and attribute in group.attrs:
            name = group.attrs[attribute]
            profiles[component.lower()] = name.decode() if isinstance(name, bytes) else str(name)
    return DiffuseGalaxyInputs(disk_metals, spheroid_metals, disk_radius, spheroid_radius,
                              orientation, inclination_degrees, units, profiles)


@dataclass(frozen=True)
class ComponentLight:
    """Rest-frame input light, intrinsic or already locally attenuated.

    Stellar Lnu is in Lsun/Hz, integrated line luminosities in erg/s, and line
    wavelengths in microns. Lines are not broadened here. Empty line arrays
    mean no requested lines, not an assumption about unrequested wavelengths.
    """

    stellar_lnu: np.ndarray
    line_wavelength_micron: np.ndarray = field(default_factory=lambda: np.empty(0))
    line_luminosity: np.ndarray = field(default_factory=lambda: np.empty(0))
    line_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentDiffuseResult:
    input_light: ComponentLight
    stellar_lnu: np.ndarray
    line_luminosity: np.ndarray
    stellar_transmission: np.ndarray
    line_transmission: np.ndarray
    flags: frozenset[str]
    coordinates: dict


@dataclass(frozen=True)
class GalaxyDiffuseResult:
    wavelength_micron: np.ndarray
    components: dict[str, ComponentDiffuseResult]
    diagnostics: dict
    provenance: dict

    @property
    def total_stellar_lnu(self):
        return sum((c.stellar_lnu for c in self.components.values()),
                   np.zeros_like(self.wavelength_micron))


class BensonGalaxyDust:
    """Galaxy adapter for an already-open Benson provider; does not own it.

    Apply once per galaxy, with separate component light. The medium is shared
    by stellar and nebular sources and is always disk-distributed. A future
    geometry/backend needs its own adapter, not forced Benson coordinates.
    """

    def __init__(self, provider, parameters=None):
        self.provider = provider
        self.parameters = parameters if parameters is not None else DiffuseDustParameters()
        if not isinstance(self.parameters, DiffuseDustParameters):
            raise TypeError("parameters must be DiffuseDustParameters")

    @property
    def provenance(self):
        return {
            "adapter": "benson_disk_dust", "schema_version": 1,
            "parameters": asdict(self.parameters), "provider": self.provider.provenance,
            "radius_convention": "exponential_disk_and_hernquist_scale_radii",
            "input_units": {"metal_mass": "Msun", "scale_radius": "Mpc", "wavelength": "micron"},
            "orientation": {"random_uniform_column": 1, "rule": "cos(i)=u",
                            "gb10_scatter_column": 0, "reserved_position_angle_column": 2},
            "status_bits": {flag.name.lower(): int(flag) for flag in DiffuseStatus},
            "composition": "diffuse_only_on_supplied_light; no implicit local or empirical attenuation",
        }

    def _coordinates(self, galaxy):
        flags = set()
        masses = {}
        for component in ("disk", "spheroid"):
            name = f"{component}_metal_mass_msun"
            original = _finite_scalar(name, getattr(galaxy, name))
            if original < -self.parameters.negative_metal_tolerance_msun:
                raise ValueError(f"{name}={original:g} is below the explicit negative-metal tolerance")
            masses[component] = max(original, 0.0)
            if original < 0:
                flags.add(f"{component}_metals_sanitized")
        if masses["spheroid"] > 0:
            flags.add("ignored_spheroid_metals")
        if galaxy.inclination_degrees is not None:
            inclination = _finite_scalar("inclination_degrees", galaxy.inclination_degrees)
            if not 0 <= inclination <= 90:
                raise ValueError("inclination_degrees must be in [0,90]")
            orientation_source = "explicit"
        else:
            if galaxy.orientation_uniform is None:
                raise ValueError("Supply orientation_uniform from randomUniform[:,1] or inclination_degrees")
            uniform = _finite_scalar("orientation_uniform", galaxy.orientation_uniform)
            if not 0 <= uniform <= 1:
                raise ValueError("orientation_uniform must be in [0,1]")
            inclination = float(np.degrees(np.arccos(uniform)))
            orientation_source = "randomUniform[:,1]"
        dust_mass = ((1 - self.parameters.cloud_fraction) * self.parameters.dust_to_metals_ratio
                     * masses["disk"])
        tau = 0.0
        if dust_mass > 0:
            if galaxy.catalog_profiles.get('disk', 'exponentialDisk') != 'exponentialDisk':
                raise ValueError("Benson disk dust requires exponentialDisk scale-radius semantics")
            radius = _finite_scalar("disk_scale_radius_mpc", galaxy.disk_scale_radius_mpc)
            if radius <= 0:
                raise ValueError("Nonzero diffuse dust requires a positive disk scale radius")
            opacity = self.provider.provenance["opacity_v_cm2_per_g_dust"]
            tau = opacity * dust_mass * u.Msun.to(u.g) / (2*np.pi*(radius*u.Mpc.to(u.cm))**2)
            if not np.isfinite(tau):
                raise ValueError("Nonfinite optical depth from dust mass and disk radius")
        total_metals = masses["disk"] + masses["spheroid"]
        diagnostics = {
            "inclination_degrees": inclination, "orientation_source": orientation_source,
            "disk_optical_depth_v": tau, "diffuse_disk_dust_mass_msun": dust_mass,
            "ignored_spheroid_metal_mass_msun": masses["spheroid"],
            "ignored_spheroid_metal_fraction": masses["spheroid"]/total_metals if total_metals else 0.0,
            "spheroid_scale_ratio": None,
            "original_inputs": asdict(galaxy),
            "disk_metal_mass_msun_used": masses["disk"],
            "spheroid_metal_mass_msun_used": masses["spheroid"],
        }
        return diagnostics, flags

    def apply(self, wavelength_micron, galaxy, components):
        """Attenuate separately supplied disk/spheroid light in the rest frame.

        Zero-luminosity sources do not require a source radius. All requested
        wavelengths must be covered, including those of zero-luminosity lines.
        Nonzero dust still requires a valid disk radius even with no disk light.
        """
        if not components or not set(components).issubset({"disk", "spheroid"}):
            raise ValueError("Supply disk and/or spheroid components; AGN transfer is unsupported")
        wave = np.asarray(wavelength_micron, dtype=float)
        if wave.ndim != 1 or wave.size == 0:
            raise ValueError("wavelength_micron must be a nonempty one-dimensional array")
        diagnostics, flags = self._coordinates(galaxy)
        outputs = {}
        for source, light in components.items():
            stellar = np.asarray(light.stellar_lnu, dtype=float)
            line_wave = np.asarray(light.line_wavelength_micron, dtype=float)
            lines = np.asarray(light.line_luminosity, dtype=float)
            if stellar.shape != wave.shape or line_wave.ndim != 1 or lines.shape != line_wave.shape:
                raise ValueError(f"Incompatible stellar/line array shapes for {source}")
            if light.line_names and len(light.line_names) != lines.size:
                raise ValueError("line_names must match the line arrays")
            if not np.all(np.isfinite(stellar)) or not np.all(np.isfinite(lines)):
                raise ValueError(f"Nonfinite {source} light")
            if np.any(stellar < 0) or np.any(lines < 0):
                raise ValueError(f"Negative {source} light is not supported")
            active = bool(np.any(stellar != 0) or np.any(lines != 0))
            ratio = None
            tau = diagnostics["disk_optical_depth_v"]
            if source == "spheroid" and active and tau > 0:
                if galaxy.catalog_profiles.get('spheroid', 'hernquist') != 'hernquist':
                    raise ValueError("Benson spheroid sources require hernquist scale-radius semantics")
                radius = _finite_scalar("spheroid_scale_radius_mpc", galaxy.spheroid_scale_radius_mpc)
                if radius <= 0:
                    raise ValueError("Nonzero spheroid light through dust requires a positive spheroid scale radius")
                ratio = radius / galaxy.disk_scale_radius_mpc
                diagnostics["spheroid_scale_ratio"] = ratio
            # Evaluate continuum and lines together so they receive the same
            # transfer law. For an absent source, tau=0 only validates coverage.
            result = self.provider.evaluate(
                np.concatenate((wave, line_wave)), source=source,
                inclination_degrees=diagnostics["inclination_degrees"],
                optical_depth_v=tau if active else 0.0, spheroid_scale_ratio=ratio,
            )
            stellar_t, line_t = result.transmission[:wave.size], result.transmission[wave.size:]
            flags.update(result.flags)
            outputs[source] = ComponentDiffuseResult(
                light, stellar*stellar_t, lines*line_t, stellar_t, line_t,
                result.flags, {**result.coordinates, "source_has_light": active},
            )
        status = DiffuseStatus(0)
        for flag in flags:
            status |= DiffuseStatus[flag.upper()]
        diagnostics.update(flags=sorted(flags), status=int(status))
        return GalaxyDiffuseResult(wave.copy(), outputs, diagnostics, self.provenance)
