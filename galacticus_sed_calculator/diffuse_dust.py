"""Component-specific diffuse transfer from the canonical Benson (2018) atlas.

This module consumes optical depth and dimensionless geometry, not galaxy
catalogs or SEDs. All wavelengths are REST-FRAME microns. It performs no
downloads and does not change the existing continuum/emission-line models.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray


BENSON2018_SHA256 = "2a04540da7c3748279a9f1656bc891dfcd282249a87e0840a53a898c529ec051"
BENSON2018_FILENAME = (
    "compendium_exp_sech_Hernquist_hd0.137_hz0.137_"
    "dustD03Rv4.0_Attenuations.hdf5"
)
BENSON2018_URL = "https://zenodo.org/records/6335545"
BENSON2018_SIZE = 583_657_496


@dataclass(frozen=True)
class VerifiedBensonAtlas:
    """Picklable receipt returned by :func:`verify_benson_atlas`.

Pass this receipt to workers to avoid hashing 557 MiB in every process.
The provider checks the file's identity, size and modification time before
opening it. Treat the file as immutable for the lifetime of all providers.
"""

    path: Path
    sha256: str
    file_signature: tuple[int, int, int, int]


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def verify_benson_atlas(path: str | os.PathLike) -> VerifiedBensonAtlas:
    """Verify the canonical local artifact without copying or downloading it."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Benson atlas not found: {path}. Download {BENSON2018_FILENAME} "
            f"from {BENSON2018_URL} and supply its local path. "
            "The download is approximately 557 MiB."
        )
    before = _file_signature(path)
    if before[2] != BENSON2018_SIZE:
        raise ValueError(f"Wrong Benson atlas size: expected {BENSON2018_SIZE} bytes")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if _file_signature(path) != before:
        raise ValueError("Benson atlas changed during checksum verification")
    if digest.hexdigest() != BENSON2018_SHA256:
        raise ValueError("Benson atlas SHA-256 mismatch; supply the canonical R_V=4 file")
    return VerifiedBensonAtlas(path, digest.hexdigest(), before)


@dataclass(frozen=True)
class DiffuseTransferResult:
    """Transmission matching the input wavelength shape, plus diagnostics.

Flags are for this evaluation (one source in one galaxy), not per wavelength.
The original and used coordinates are both retained when clipping is selected.
No stellar/nebular distinction is imposed: equal source geometries receive
equal transfer. ``transmission`` may exceed one due to directional scattering.
"""

    transmission: NDArray[np.float64]
    flags: frozenset[str]
    coordinates: dict[str, float | str | None]


def _scalar(name: str, value: float) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite scalar")
    return float(array)


def _bracket(axis: NDArray, value: float, *, logarithmic: bool):
    """One exact node or two adjacent nodes, with interpolation weights."""
    index = int(np.searchsorted(axis, value))
    if index < len(axis) and axis[index] == value:
        return slice(index, index + 1), np.array([1.0])
    if index == 0 or index == len(axis):
        raise ValueError("Internal interpolation coordinate outside table")
    lo, hi = axis[index - 1:index + 1]
    if logarithmic:
        lo, hi, value = np.log([lo, hi, value])
    weight = (value - lo) / (hi - lo)
    return slice(index - 1, index + 1), np.array([1.0 - weight, weight])


class Benson2018DiffuseProvider:
    """Lazy, bounded-memory interpolation of disk-dust transfer functions.

    ``evaluate`` accepts already calculated physical coordinates. Conversion
    from gas/metal masses, radii and stored orientations belongs to the caller.
    A future emulator can supply the same result type and its own geometry
    inputs; callers must not assume that every backend uses this atlas's axes.

    ``high_tau_policy`` is ``error`` (default), ``clip`` or
    ``atlas_extrapolation``. ``scale_ratio_policy`` is ``error`` or ``clip``.
    Wavelength bounds always raise, apart from 1e-7 relative endpoint roundoff
    (the published final wavelength is slightly below 3 microns).

    Open one provider per worker, preferably with a parent-verified receipt.
    Use as a context manager; do not share an open HDF5 handle across a fork.
    """

    def __init__(
        self,
        atlas: str | os.PathLike | VerifiedBensonAtlas,
        *,
        high_tau_policy: Literal["error", "clip", "atlas_extrapolation"] = "error",
        scale_ratio_policy: Literal["error", "clip"] = "error",
    ):
        if high_tau_policy not in {"error", "clip", "atlas_extrapolation"}:
            raise ValueError("Unknown high_tau_policy")
        if scale_ratio_policy not in {"error", "clip"}:
            raise ValueError("Unknown scale_ratio_policy")
        receipt = atlas if isinstance(atlas, VerifiedBensonAtlas) else verify_benson_atlas(atlas)
        if receipt.sha256 != BENSON2018_SHA256:
            raise ValueError("Receipt does not identify the canonical Benson atlas")
        if _file_signature(receipt.path) != receipt.file_signature:
            raise ValueError("Benson atlas changed since verification; verify it again")
        self.high_tau_policy = high_tau_policy
        self.scale_ratio_policy = scale_ratio_policy
        self._pid = os.getpid()
        self._file = h5py.File(receipt.path, "r")
        try:
            self._validate_schema()
        except Exception:
            self.close()
            raise

    def _validate_schema(self):
        axes = {}
        for name in ("wavelength", "inclination", "opticalDepth", "spheroidScaleRadial"):
            if name not in self._file or not isinstance(self._file[name], h5py.Dataset):
                raise ValueError(f"Missing atlas axis: {name}")
            axis = np.asarray(self._file[name], dtype=float)
            if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)):
                raise ValueError(f"Invalid atlas axis: {name}")
            if not np.all(np.diff(axis) > 0):
                raise ValueError(f"Atlas axis must be strictly increasing: {name}")
            axes[name] = axis
        self._wavelength = axes["wavelength"]
        self._inclination = axes["inclination"]
        self._tau = axes["opticalDepth"]
        self._ratio = axes["spheroidScaleRadial"]
        if (self._wavelength[0] <= 0 or self._ratio[0] <= 0
                or self._tau[0] != 0 or len(self._tau) < 3
                or self._inclination[0] != 0 or self._inclination[-1] != 90):
            raise ValueError("Atlas axes do not have the expected physical boundaries")
        w, i, t, r = map(len, (self._wavelength, self._inclination, self._tau, self._ratio))
        shapes = {
            "attenuationDisk": (w, i, t),
            "attenuationSpheroid": (w, i, t, r),
            "attenuationUncertaintyDisk": (w, i, t),
            "attenuationUncertaintySpheroid": (w, i, t, r),
            "extrapolationCoefficientsDisk": (2, w, i),
            "extrapolationCoefficientsSpheroid": (2, w, i, r),
        }
        for name, shape in shapes.items():
            if (name not in self._file or not isinstance(self._file[name], h5py.Dataset)
                    or self._file[name].shape != shape
                    or not np.issubdtype(self._file[name].dtype, np.number)):
                raise ValueError(f"Invalid atlas dataset/shape: {name}; expected {shape}")
        expected = {"opacity": 32062.2129019, "diskScaleVertical": 0.137,
                    "dustScaleVertical": 0.137, "diskCutOff": 10, "spheroidCutOff": 10}
        for name, value in expected.items():
            if name not in self._file.attrs or not np.isclose(
                float(self._file.attrs[name]), value, rtol=1e-10, atol=0
            ):
                raise ValueError(f"Unexpected atlas attribute: {name}")
        for name, expected_text in (("dustDescription", "R_V=4.0"),
                                    ("diskStructureVertical", "sechSquared")):
            value = self._file.attrs.get(name, "")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if expected_text not in str(value):
                raise ValueError(f"Unexpected atlas attribute: {name}")

    @property
    def provenance(self) -> dict:
        """Scientific identity and numerical policies, without private paths."""
        return {
            "backend": "benson2018_rv4", "schema_version": 1,
            "filename": BENSON2018_FILENAME, "sha256": BENSON2018_SHA256,
            "source": BENSON2018_URL, "doi": "10.5281/zenodo.6335545",
            "reference": "https://arxiv.org/abs/1810.01449", "license": "CC-BY-4.0",
            "opacity_v_cm2_per_g_dust": 32062.2129019,
            "target": "attenuation_magnitudes",
            "coordinates": ["log_wavelength", "inclination_degrees", "log_tau_v", "log_scale_ratio"],
            "low_tau_policy": "linear_transmission_to_unity",
            "high_tau_policy": self.high_tau_policy,
            "scale_ratio_policy": self.scale_ratio_policy,
            "wavelength_policy": "error", "endpoint_roundoff_rtol": 1e-7,
        }

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        return {name: (float(axis[0]), float(axis[-1])) for name, axis in (
            ("wavelength_micron", self._wavelength),
            ("inclination_degrees", self._inclination),
            ("optical_depth_v", self._tau),
            ("spheroid_scale_ratio", self._ratio),
        )}

    def close(self):
        """Close the HDF5 handle; repeated calls are harmless."""
        self._file.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        raise TypeError("Pass a VerifiedBensonAtlas to workers, not an open provider")

    def _check_open(self):
        if self._pid != os.getpid():
            raise RuntimeError("Open a new Benson provider in each process; do not inherit HDF5 handles")
        if not self._file.id.valid:
            raise RuntimeError("Benson provider is closed")

    def evaluate(
        self,
        wavelength_micron: ArrayLike,
        *,
        source: Literal["disk", "spheroid"],
        inclination_degrees: float,
        optical_depth_v: float,
        spheroid_scale_ratio: float | None = None,
    ) -> DiffuseTransferResult:
        """Return transfer for one source geometry at any-shaped wavelengths.

        ``optical_depth_v`` is the full central face-on V-band disk optical
        depth. ``spheroid_scale_ratio`` is the Hernquist source scale radius
        divided by the disk radial scale length (not a half-light radius).
        It is required only for a dusty spheroid-source query.
        """
        self._check_open()
        if source not in {"disk", "spheroid"}:
            raise ValueError("source must be 'disk' or 'spheroid'; central/AGN sources are unsupported")
        wavelength = np.asarray(wavelength_micron, dtype=float)
        if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("Rest-frame wavelength_micron must be finite and positive")
        lower, upper = self.bounds["wavelength_micron"]
        if np.any(wavelength < lower * (1 - 1e-7)) or np.any(wavelength > upper * (1 + 1e-7)):
            raise ValueError(f"Rest-frame wavelength outside atlas coverage [{lower}, {upper}] micron")
        flags = set()
        if np.any((wavelength < lower) | (wavelength > upper)):
            flags.add("wavelength_endpoint_roundoff")
        wave = np.clip(wavelength.ravel(), lower, upper)
        inclination = _scalar("inclination_degrees", inclination_degrees)
        tau = _scalar("optical_depth_v", optical_depth_v)
        if not 0 <= inclination <= 90:
            raise ValueError("inclination_degrees must lie in [0, 90]")
        if tau < 0:
            raise ValueError("optical_depth_v must be nonnegative")
        coordinates = {"source": source, "inclination_degrees": inclination,
                       "optical_depth_v": tau, "optical_depth_v_used": tau,
                       "spheroid_scale_ratio": None, "spheroid_scale_ratio_used": None}
        if tau == 0:
            return DiffuseTransferResult(np.ones(wavelength.shape), frozenset(flags), coordinates)
        ratio = None
        if source == "spheroid":
            if spheroid_scale_ratio is None:
                raise ValueError("Dusty spheroid sources require spheroid_scale_ratio")
            ratio = _scalar("spheroid_scale_ratio", spheroid_scale_ratio)
            if ratio <= 0:
                raise ValueError("spheroid_scale_ratio must be positive")
            coordinates["spheroid_scale_ratio"] = ratio
            if ratio < self._ratio[0] or ratio > self._ratio[-1]:
                if self.scale_ratio_policy == "error":
                    raise ValueError("spheroid_scale_ratio outside atlas coverage")
                ratio = float(np.clip(ratio, self._ratio[0], self._ratio[-1]))
                flags.add("scale_ratio_clipped")
            coordinates["spheroid_scale_ratio_used"] = ratio
        extrapolate = tau > self._tau[-1] and self.high_tau_policy == "atlas_extrapolation"
        if tau > self._tau[-1]:
            if self.high_tau_policy == "error":
                raise ValueError(f"optical_depth_v exceeds {self._tau[-1]}; choose an explicit high_tau_policy")
            if extrapolate:
                flags.add("optical_depth_extrapolated")
            else:
                tau = float(self._tau[-1])
                flags.add("optical_depth_clipped")
        coordinates["optical_depth_v_used"] = tau
        suffix = "Disk" if source == "disk" else "Spheroid"
        if extrapolate:
            c0, c1 = self._interpolate(wave, inclination, None, ratio,
                                     "extrapolationCoefficients" + suffix)
            if np.any(c1 > 0):
                flags.add("extrapolation_positive_slope")
            with np.errstate(over="ignore", under="ignore"):
                transmission = np.exp(c0 + c1 * np.log(tau))
            if not np.all(np.isfinite(transmission)):
                raise ValueError("Atlas extrapolation produced nonfinite transmission")
        else:
            sampled_tau = max(tau, float(self._tau[1]))
            attenuation = self._interpolate(wave, inclination, sampled_tau, ratio,
                                            "attenuation" + suffix)
            transmission = 10.0 ** (-0.4 * attenuation)
            if tau < self._tau[1]:
                transmission = 1.0 + (tau / self._tau[1]) * (transmission - 1.0)
                flags.add("low_optical_depth_blend")
        return DiffuseTransferResult(transmission.reshape(wavelength.shape),
                                     frozenset(flags), coordinates)

    def _interpolate(self, wavelength, inclination, tau, ratio, dataset_name):
        """Read only wavelength nodes and neighbouring geometry hyperslabs.

        Even for the whole spectrum the temporary spheroid block has at most
        n_wavelength * 2 * 2 * 2 values, instead of the 268 MiB full tensor.
        """
        if not len(wavelength):
            return np.empty((2, 0)) if tau is None else np.empty(0)
        high = np.searchsorted(self._wavelength, wavelength, side="left")
        high = np.minimum(high, len(self._wavelength) - 1)
        low = np.maximum(high - 1, 0)
        low = np.where(self._wavelength[high] == wavelength, high, low)
        nodes = np.unique(np.concatenate((low, high)))
        brackets = [_bracket(self._inclination, inclination, logarithmic=False)]
        if tau is not None:
            # Positive tau coordinates only; never take log(0).
            selection, weights = _bracket(self._tau[1:], tau, logarithmic=True)
            brackets.append((slice(selection.start + 1, selection.stop + 1), weights))
        if ratio is not None:
            brackets.append(_bracket(self._ratio, ratio, logarithmic=True))
        selection = (nodes,) + tuple(item[0] for item in brackets)
        if tau is None:
            selection = (slice(None),) + selection
        values = np.asarray(self._file[dataset_name][selection], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite values in selected atlas cells: {dataset_name}")
        if tau is not None:
            if np.any(values <= 0):
                raise ValueError("Attenuation interpolation requires positive table transmission")
            values = -2.5 * np.log10(values)
        for _, weights in reversed(brackets):
            values = np.sum(values * weights, axis=-1)
        lo = np.searchsorted(nodes, low)
        hi = np.searchsorted(nodes, high)
        fraction = np.zeros(wavelength.shape)
        different = low != high
        fraction[different] = (
            np.log(wavelength[different] / self._wavelength[low[different]])
            / np.log(self._wavelength[high[different]] / self._wavelength[low[different]])
        )
        return values[..., lo] * (1 - fraction) + values[..., hi] * fraction
