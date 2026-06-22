from __future__ import annotations

import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path
import re
import warnings
from collections.abc import Iterable
from typing import Any

import h5py
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).with_name("romanUNIT-d1_100_0.2Deg.hdf5")
DEFAULT_OBSERVATION_DIR = Path(__file__).with_name("observational_data") / "f158"
DEG2_PER_STERADIAN = (180.0 / np.pi) ** 2


@dataclass(frozen=True)
class ObservationCatalog:
    label: str
    path: Path
    area_deg2: float
    redshift: np.ndarray
    magnitude: np.ndarray
    metadata: dict[str, Any]


def expand_inputs(
    inputs: list[Path],
    globs: list[str],
    *,
    skip_missing: bool,
    default_input: Path = DEFAULT_INPUT,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    if not inputs and not globs and default_input.exists():
        inputs = [default_input]
    paths = [path.expanduser().resolve() for path in inputs]
    for pattern in globs:
        matches = [Path(match).expanduser().resolve() for match in glob.glob(pattern)]
        if not matches:
            raise FileNotFoundError(f"No files matched --input-glob {pattern!r}")
        paths.extend(matches)
    unique = list(dict.fromkeys(paths))
    missing = [path for path in unique if not path.exists()]
    if missing:
        if not skip_missing:
            raise FileNotFoundError(f"Input file(s) do not exist: {missing}")
        unique = [path for path in unique if path.exists()]
    skipped = [(path, "file does not exist") for path in missing]
    return sorted(unique), skipped


def lightcone_output_names(handle: h5py.File) -> list[str]:
    return sorted(handle["/Lightcone"].keys(), key=lambda name: int(re.sub(r"\D", "", name) or 0))


def covered_unit_ids(paths: list[Path]) -> set[int]:
    covered: set[int] = set()
    for path in paths:
        match = re.search(r"-d(\d+)_(\d+)_", path.name)
        if match is None:
            continue
        start, end = (int(match.group(1)), int(match.group(2)))
        covered.update(range(start, end + 1))
    return covered


def format_unit_id_ranges(values: Iterable[int], *, limit: int = 20) -> str:
    ids = sorted(values)
    if not ids:
        return "none"
    ranges: list[str] = []
    start = previous = ids[0]
    for value in ids[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
    if len(ranges) > limit:
        return ", ".join(ranges[:limit]) + f", ... ({len(ranges)} ranges total)"
    return ", ".join(ranges)


def infer_unit_scale(paths: list[Path], total_units: int) -> float:
    processed = 0
    for path in paths:
        match = re.search(r"-d(\d+)_(\d+)_", path.name)
        if match is None:
            continue
        start, end = (int(match.group(1)), int(match.group(2)))
        processed += end - start + 1
    if processed <= 0:
        warnings.warn("Could not infer UNIT realization coverage from filenames; using scale=1.", stacklevel=2)
        return 1.0
    return float(total_units) / float(processed)


def resolve_unit_scale(value: str, paths: list[Path], total_units: int) -> float:
    if value.strip().lower() == "auto":
        return infer_unit_scale(paths, total_units)
    scale = float(value)
    if scale <= 0.0:
        raise ValueError("--unit-realization-scale must be positive")
    return scale


def warn_for_incomplete_unit_coverage(paths: list[Path], total_units: int) -> None:
    covered = covered_unit_ids(paths)
    if not covered:
        return
    missing = sorted(set(range(1, total_units + 1)) - covered)
    if missing:
        warnings.warn(
            f"Usable files cover {len(covered)} / {total_units} UNIT ids; "
            f"missing {format_unit_id_ranges(missing)}",
            stacklevel=2,
        )


def write_skip_report(
    output_dir: Path,
    skipped: list[tuple[Path, str]],
    paths: list[Path],
    total_units: int,
) -> None:
    covered = covered_unit_ids(paths)
    missing_ids = sorted(set(range(1, total_units + 1)) - covered) if covered else []
    if not skipped and not missing_ids:
        return
    report_path = output_dir / "skipped_lightcone_files.txt"
    with report_path.open("w") as handle:
        handle.write(f"usable files: {len(paths)}\n")
        handle.write(f"skipped files: {len(skipped)}\n")
        if covered:
            handle.write(f"covered UNIT ids: {len(covered)} / {total_units}\n")
            handle.write(f"missing UNIT ids: {format_unit_id_ranges(missing_ids)}\n")
        handle.write("\nSKIPPED FILES\n")
        for path, reason in skipped:
            handle.write(f"{path}\n  {reason}\n")
    print(f"Skip/missing-file report: {report_path}", flush=True)


def cosmology_from_file(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        group = handle["/Parameters/cosmologyParameters"]
        return {
            "H0": float(group.attrs["HubbleConstant"]),
            "OmegaMatter": float(group.attrs["OmegaMatter"]),
            "OmegaDarkEnergy": float(group.attrs["OmegaDarkEnergy"]),
        }


def comoving_distance_mpc(z: np.ndarray, cosmology: dict[str, float]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    flat = z.ravel()
    if flat.size == 0:
        return np.asarray([], dtype=float).reshape(z.shape)
    z_max = float(np.nanmax(flat))
    if z_max <= 0.0:
        return np.zeros_like(z)
    n_grid = max(4096, int(np.ceil(z_max * 2048)))
    grid = np.linspace(0.0, z_max, n_grid)
    e_z = np.sqrt(
        cosmology["OmegaMatter"] * (1.0 + grid) ** 3
        + cosmology["OmegaDarkEnergy"]
    )
    integrand = 1.0 / e_z
    cumulative = np.empty_like(grid)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))
    c_km_s = 299792.458
    distance = (c_km_s / cosmology["H0"]) * np.interp(flat, grid, cumulative)
    return distance.reshape(z.shape)


def luminosity_distance_mpc(z: np.ndarray, cosmology: dict[str, float]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return (1.0 + z) * comoving_distance_mpc(z, cosmology)


def distance_modulus(z: np.ndarray, cosmology: dict[str, float]) -> np.ndarray:
    d_l_mpc = luminosity_distance_mpc(z, cosmology)
    result = np.full_like(d_l_mpc, np.nan, dtype=float)
    positive = d_l_mpc > 0.0
    result[positive] = 5.0 * np.log10(d_l_mpc[positive]) + 25.0
    return result


def pseudo_absolute_magnitude(
    apparent_magnitude: np.ndarray,
    redshift: np.ndarray,
    cosmology: dict[str, float],
) -> np.ndarray:
    redshift = np.asarray(redshift, dtype=float)
    apparent_magnitude = np.asarray(apparent_magnitude, dtype=float)
    return apparent_magnitude - distance_modulus(redshift, cosmology) + 2.5 * np.log10(1.0 + redshift)


def volume_per_square_degree(z_min: float, z_max: float, cosmology: dict[str, float]) -> float:
    chi_min, chi_max = comoving_distance_mpc(np.asarray([z_min, z_max]), cosmology)
    return float((chi_max**3 - chi_min**3) / (3.0 * DEG2_PER_STERADIAN))


def centers_from_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def weights_from_node_data(
    node_data: h5py.Group,
    unit_scale: float,
    angular_weight_mode: str,
) -> np.ndarray:
    angular_weight = np.asarray(node_data["angularWeight"][...], dtype=float)
    if angular_weight_mode == "area":
        if np.any(angular_weight <= 0.0):
            raise ValueError(f"{node_data.name}/angularWeight contains non-positive areas")
        weights = unit_scale / angular_weight
    else:
        weights = angular_weight * unit_scale
    if "nodeSubsamplingWeight" in node_data:
        weights = weights * np.asarray(node_data["nodeSubsamplingWeight"][...], dtype=float)
    return weights


def validate_f158_file(
    path: Path,
    *,
    magnitude_group: str,
    magnitude_dataset: str,
    redshift_dataset: str,
) -> None:
    with h5py.File(path, "r") as handle:
        output_names = lightcone_output_names(handle)
        if not output_names:
            raise RuntimeError("no /Lightcone outputs")
        for output_name in output_names:
            output_group = handle[f"/Lightcone/{output_name}"]
            node_data = output_group["nodeData"]
            _ = node_data[redshift_dataset].shape
            _ = node_data["angularWeight"].shape
            mag_group = output_group[magnitude_group]
            _ = mag_group[magnitude_dataset].shape


def filter_usable_f158_paths(
    paths: list[Path],
    *,
    magnitude_group: str,
    magnitude_dataset: str,
    redshift_dataset: str,
    skip_bad_files: bool,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    usable: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for path in paths:
        try:
            validate_f158_file(
                path,
                magnitude_group=magnitude_group,
                magnitude_dataset=magnitude_dataset,
                redshift_dataset=redshift_dataset,
            )
            usable.append(path)
        except Exception as exc:
            if not skip_bad_files:
                raise RuntimeError(f"Failed validation for {path}") from exc
            skipped.append((path, repr(exc)))
            print(f"WARNING: skipping {path}: {exc}", flush=True)
    if not usable:
        raise RuntimeError("No usable input files remain after validation")
    return usable, skipped


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_limits(
    *values: np.ndarray | Iterable[float] | None,
    margin_fraction: float = 0.08,
    min_margin: float = 0.15,
) -> tuple[float, float] | None:
    arrays = []
    for value in values:
        if value is None:
            continue
        array = np.asarray(value, dtype=float).ravel()
        finite = array[np.isfinite(array)]
        if finite.size:
            arrays.append(finite)
    if not arrays:
        return None
    finite = np.concatenate(arrays)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper <= lower:
        lower -= min_margin
        upper += min_margin
        return lower, upper
    margin = max(margin_fraction * (upper - lower), min_margin)
    return lower - margin, upper + margin


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".txt"}:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, delim_whitespace=True, comment="#")
    return pd.read_csv(path, delim_whitespace=True, comment="#")


def _load_metadata(path: Path, default_label: str, default_area_deg2: float | None) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    metadata: dict[str, Any] = {"source": default_label}
    if sidecar.exists():
        with sidecar.open() as handle:
            metadata.update(json.load(handle))
    if "area_deg2" not in metadata and default_area_deg2 is not None:
        metadata["area_deg2"] = float(default_area_deg2)
    return metadata


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {column.lower(): column for column in frame.columns}


def _first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = _column_lookup(frame)
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _apply_common_observation_cuts(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lookup = _column_lookup(result)
    for column_name, allowed in [
        ("use_phot", {1, True}),
        ("star_flag", {0, False}),
        ("flag_star", {0, False}),
        ("warn_flag", {0, False}),
        ("flag_combined", {0, False}),
        ("lp_type", {0, False}),
        ("type", {0, False}),
    ]:
        if column_name in lookup:
            column = lookup[column_name]
            values = result[column]
            result = result.loc[values.isin(allowed)]
    return result


def load_observation_catalog(
    path: Path,
    *,
    label: str,
    default_area_deg2: float | None,
    redshift_column: str | None = None,
    magnitude_column: str | None = None,
) -> ObservationCatalog:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = _apply_common_observation_cuts(_read_table(path))
    metadata = _load_metadata(path, label, default_area_deg2)
    area_deg2 = metadata.get("area_deg2")
    if area_deg2 is None:
        raise ValueError(f"No area_deg2 supplied for {path}; pass an area argument or add a .meta.json sidecar")
    area_deg2 = float(area_deg2)
    if area_deg2 <= 0.0:
        raise ValueError(f"Non-positive area_deg2={area_deg2} for {path}")

    redshift_col = redshift_column or metadata.get("redshift_column")
    if redshift_col is None:
        redshift_col = _first_column(
            frame,
            [
                "redshift",
                "z",
                "z_best",
                "zbest",
                "z_peak",
                "z_phot",
                "zphot",
                "zfinal",
                "zpdf_med",
                "lp_zBEST",
                "ez_z_phot",
            ],
        )
    if redshift_col is None or redshift_col not in frame:
        raise ValueError(f"Could not identify redshift column in {path}")

    magnitude_col = magnitude_column or metadata.get("magnitude_column")
    if magnitude_col is None:
        magnitude_col = _first_column(
            frame,
            [
                "magnitude",
                "mag",
                "f150w_mag",
                "f_F150W_mag",
                "F150W_mag",
                "mag_model_f150w",
                "mag_auto_f150w",
                "mag_aper_f150w",
                "f160w_mag",
                "f_F160W_mag",
                "F160W_mag",
                "UVISTA_H_MAG_AUTO",
                "H_MAG_AUTO",
                "H_mag",
            ],
        )
    if magnitude_col is not None and magnitude_col in frame:
        magnitude = pd.to_numeric(frame[magnitude_col], errors="coerce").to_numpy(dtype=float)
    else:
        flux_col = _first_column(
            frame,
            [
                "f_F150W",
                "F150W_FLUX",
                "flux_f150w",
                "flux_model_f150w",
                "flux_auto_f150w",
                "f_F160W",
                "F160W_FLUX",
                "flux_f160w",
            ],
        )
        if flux_col is None:
            raise ValueError(f"Could not identify magnitude or F150W/F160W flux column in {path}")
        flux = pd.to_numeric(frame[flux_col], errors="coerce").to_numpy(dtype=float)
        magnitude = np.full_like(flux, np.nan, dtype=float)
        positive = flux > 0.0
        magnitude[positive] = 25.0 - 2.5 * np.log10(flux[positive])
        magnitude_col = f"{flux_col} converted with AB zeropoint 25"

    redshift = pd.to_numeric(frame[redshift_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(redshift) & np.isfinite(magnitude) & (redshift > 0.0) & (magnitude > 0.0) & (magnitude < 90.0)
    metadata.update(
        {
            "source": metadata.get("source", label),
            "area_deg2": area_deg2,
            "redshift_column": redshift_col,
            "magnitude_column": magnitude_col,
            "valid_rows": int(np.count_nonzero(valid)),
        }
    )
    return ObservationCatalog(
        label=str(metadata.get("source", label)),
        path=path,
        area_deg2=area_deg2,
        redshift=redshift[valid],
        magnitude=magnitude[valid],
        metadata=metadata,
    )


def maybe_load_observation_catalog(
    path: Path | None,
    default_path: Path,
    *,
    label: str,
    default_area_deg2: float | None,
    redshift_column: str | None = None,
    magnitude_column: str | None = None,
) -> ObservationCatalog | None:
    chosen = path.expanduser() if path is not None else default_path
    if not chosen.exists():
        if path is not None:
            raise FileNotFoundError(chosen)
        print(f"Observation catalog not found for {label}: {chosen}", flush=True)
        return None
    catalog = load_observation_catalog(
        chosen,
        label=label,
        default_area_deg2=default_area_deg2,
        redshift_column=redshift_column,
        magnitude_column=magnitude_column,
    )
    print(
        f"Loaded {catalog.label}: {catalog.metadata['valid_rows']} usable rows, "
        f"area={catalog.area_deg2:.4g} deg^2",
        flush=True,
    )
    return catalog


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
