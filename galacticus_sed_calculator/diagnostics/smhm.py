from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    centers_from_edges,
    lightcone_output_names,
    resolve_unit_scale,
    weights_from_node_data,
    write_csv,
)


@dataclass(frozen=True)
class SMHMDiagnosticsConfig:
    halo_mass_dataset: str = "basicMass"
    disk_stellar_mass_dataset: str = "diskMassStellar"
    spheroid_stellar_mass_dataset: str = "spheroidMassStellar"
    redshift_dataset: str = "lightconeRedshiftObserved"
    z_centers: tuple[float, ...] = (0.0, 1.0, 2.0)
    z_half_width: float = 0.10
    output_redshift_tolerance: float | None = None
    log10_halo_mass_min: float = 9.5
    log10_halo_mass_max: float = 15.0
    log10_halo_mass_bin_width: float = 0.25
    angular_weight_mode: str = "area"
    unit_realization_scale: str | float = "auto"
    unit_realization_total: int = 10000
    galaxy_selection: str = "all"
    fill_missing_node_weights_with_median: bool = False
    dpi: int = 220


def _mass_edges(config: SMHMDiagnosticsConfig) -> np.ndarray:
    if config.log10_halo_mass_bin_width <= 0.0:
        raise ValueError("SMHM halo-mass bin width must be positive")
    if config.log10_halo_mass_max <= config.log10_halo_mass_min:
        raise ValueError("SMHM halo-mass maximum must be larger than minimum")
    n_bins = int(np.ceil((config.log10_halo_mass_max - config.log10_halo_mass_min) / config.log10_halo_mass_bin_width))
    return config.log10_halo_mass_min + config.log10_halo_mass_bin_width * np.arange(n_bins + 1, dtype=float)


def _output_names(handle: h5py.File, group_name: str) -> list[str]:
    return sorted(handle[group_name].keys(), key=lambda name: int("".join(ch for ch in name if ch.isdigit()) or 0))


def _fixed_output_redshift(output_group: h5py.Group) -> float:
    for attr_name in ["outputRedshift", "redshift"]:
        if attr_name in output_group.attrs:
            return float(output_group.attrs[attr_name])
    if "outputExpansionFactor" in output_group.attrs:
        expansion_factor = float(output_group.attrs["outputExpansionFactor"])
        if expansion_factor > 0.0:
            return 1.0 / expansion_factor - 1.0
    node_data = output_group["nodeData"]
    if "redshift" in node_data:
        redshift = np.asarray(node_data["redshift"][...], dtype=float)
        finite = redshift[np.isfinite(redshift)]
        if finite.size:
            return float(np.median(finite))
    raise KeyError(f"Could not determine redshift for {output_group.name}")


def _read_tree_weights(
    output_group: h5py.Group,
    n_nodes: int,
    *,
    fill_missing_with_median: bool,
) -> np.ndarray:
    required = ["mergerTreeStartIndex", "mergerTreeCount", "mergerTreeWeight"]
    if not all(name in output_group for name in required):
        return np.ones(n_nodes, dtype=float)
    tree_weights = np.asarray(output_group["mergerTreeWeight"][...], dtype=float)
    tree_counts = np.asarray(output_group["mergerTreeCount"][...], dtype=int)
    tree_starts = np.asarray(output_group["mergerTreeStartIndex"][...], dtype=int)
    node_weights = np.zeros(n_nodes, dtype=float)
    assigned = np.zeros(n_nodes, dtype=bool)
    for start, count, weight in zip(tree_starts, tree_counts, tree_weights, strict=True):
        stop = int(start) + int(count)
        if start < 0 or stop > n_nodes:
            raise ValueError(f"Merger-tree node range [{start}:{stop}] is outside nodeData length {n_nodes}")
        node_weights[start:stop] = weight
        assigned[start:stop] = True
    if np.any(~assigned):
        missing_count = int(np.count_nonzero(~assigned))
        if not fill_missing_with_median:
            raise ValueError("Some nodes were not assigned a merger-tree weight")
        finite_weights = tree_weights[np.isfinite(tree_weights)]
        if finite_weights.size == 0:
            raise ValueError("Some nodes were not assigned a merger-tree weight and no finite median weight is available")
        fill_value = float(np.median(finite_weights))
        node_weights[~assigned] = fill_value
        warnings.warn(
            f"Filled {missing_count} unassigned node weight(s) in {output_group.name} "
            f"with median merger-tree weight {fill_value:.6g}",
            RuntimeWarning,
            stacklevel=2,
        )
    return node_weights


def _fixed_output_weights(output_group: h5py.Group, config: SMHMDiagnosticsConfig, n_nodes: int) -> np.ndarray:
    node_data = output_group["nodeData"]
    weights = _read_tree_weights(
        output_group,
        n_nodes,
        fill_missing_with_median=config.fill_missing_node_weights_with_median,
    )
    if "nodeSubsamplingWeight" in node_data:
        weights = weights * np.asarray(node_data["nodeSubsamplingWeight"][...], dtype=float)
    return weights


def _stellar_mass(node_data: h5py.Group, config: SMHMDiagnosticsConfig) -> np.ndarray:
    disk = np.asarray(node_data[config.disk_stellar_mass_dataset][...], dtype=float)
    if config.spheroid_stellar_mass_dataset in node_data:
        spheroid = np.asarray(node_data[config.spheroid_stellar_mass_dataset][...], dtype=float)
    else:
        spheroid = np.zeros_like(disk)
    return disk + spheroid


def _selection_mask(node_data: h5py.Group, selection: str, size: int) -> np.ndarray:
    if selection == "all":
        return np.ones(size, dtype=bool)
    if "nodeIsIsolated" not in node_data:
        raise KeyError("nodeData/nodeIsIsolated is required for central/satellite SMHM selections")
    is_central = np.asarray(node_data["nodeIsIsolated"][...], dtype=int) == 1
    if selection == "centrals":
        return is_central
    if selection == "satellites":
        return ~is_central
    raise ValueError(f"Unknown SMHM galaxy selection: {selection}")


def _empty_accumulator(n_cases: int, n_bins: int) -> dict[str, np.ndarray]:
    shape = (n_cases, n_bins)
    return {
        "sum_w": np.zeros(shape, dtype=float),
        "sum_w2": np.zeros(shape, dtype=float),
        "sum_w_y": np.zeros(shape, dtype=float),
        "sum_w_y2": np.zeros(shape, dtype=float),
        "raw_count": np.zeros(shape, dtype=np.int64),
        "matched_outputs": np.zeros(n_cases, dtype=np.int64),
        "matched_redshift_min": np.full(n_cases, np.nan, dtype=float),
        "matched_redshift_max": np.full(n_cases, np.nan, dtype=float),
    }


def _record_matched_output(accumulator: dict[str, np.ndarray], case_index: int, redshift: float) -> None:
    accumulator["matched_outputs"][case_index] += 1
    current_min = accumulator["matched_redshift_min"][case_index]
    current_max = accumulator["matched_redshift_max"][case_index]
    accumulator["matched_redshift_min"][case_index] = redshift if not np.isfinite(current_min) else min(current_min, redshift)
    accumulator["matched_redshift_max"][case_index] = redshift if not np.isfinite(current_max) else max(current_max, redshift)


def _accumulate_values(
    accumulator: dict[str, np.ndarray],
    case_index: int,
    log10_halo_mass: np.ndarray,
    log10_stellar_mass: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> None:
    valid = (
        np.isfinite(log10_halo_mass)
        & np.isfinite(log10_stellar_mass)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        return
    x = log10_halo_mass[valid]
    y = log10_stellar_mass[valid]
    w = weights[valid]
    counts, _ = np.histogram(x, bins=edges)
    sum_w, _ = np.histogram(x, bins=edges, weights=w)
    sum_w2, _ = np.histogram(x, bins=edges, weights=w**2)
    sum_w_y, _ = np.histogram(x, bins=edges, weights=w * y)
    sum_w_y2, _ = np.histogram(x, bins=edges, weights=w * y**2)
    accumulator["raw_count"][case_index] += counts.astype(np.int64)
    accumulator["sum_w"][case_index] += sum_w
    accumulator["sum_w2"][case_index] += sum_w2
    accumulator["sum_w_y"][case_index] += sum_w_y
    accumulator["sum_w_y2"][case_index] += sum_w_y2


def _process_node_data(
    accumulator: dict[str, np.ndarray],
    case_index: int,
    node_data: h5py.Group,
    weights: np.ndarray,
    edges: np.ndarray,
    config: SMHMDiagnosticsConfig,
) -> None:
    halo_mass = np.asarray(node_data[config.halo_mass_dataset][...], dtype=float)
    stellar_mass = _stellar_mass(node_data, config)
    selected = _selection_mask(node_data, config.galaxy_selection, halo_mass.size)
    valid = selected & (halo_mass > 0.0) & (stellar_mass > 0.0)
    log10_halo_mass = np.full(halo_mass.size, np.nan, dtype=float)
    log10_stellar_mass = np.full(stellar_mass.size, np.nan, dtype=float)
    log10_halo_mass[valid] = np.log10(halo_mass[valid])
    log10_stellar_mass[valid] = np.log10(stellar_mass[valid])
    _accumulate_values(accumulator, case_index, log10_halo_mass, log10_stellar_mass, weights, edges)


def _validate_file(path: Path, config: SMHMDiagnosticsConfig) -> None:
    with h5py.File(path, "r") as handle:
        if "/Lightcone" not in handle and "/Outputs" not in handle:
            raise RuntimeError("file has neither /Lightcone nor /Outputs")
        group_name = "/Lightcone" if "/Lightcone" in handle else "/Outputs"
        names = lightcone_output_names(handle) if group_name == "/Lightcone" else _output_names(handle, group_name)
        if not names:
            raise RuntimeError(f"no outputs in {group_name}")
        output_group = handle[f"{group_name}/{names[0]}"]
        node_data = output_group["nodeData"]
        _ = node_data[config.halo_mass_dataset].shape
        _ = node_data[config.disk_stellar_mass_dataset].shape
        if config.galaxy_selection != "all":
            _ = node_data["nodeIsIsolated"].shape
        if group_name == "/Lightcone":
            _ = node_data[config.redshift_dataset].shape
            _ = node_data["angularWeight"].shape
        else:
            _ = _fixed_output_redshift(output_group)


def _filter_usable_paths(
    paths: list[Path],
    config: SMHMDiagnosticsConfig,
    *,
    skip_bad_files: bool,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    usable: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for path in paths:
        try:
            _validate_file(path, config)
            usable.append(path)
        except Exception as exc:
            if not skip_bad_files:
                raise RuntimeError(f"Failed SMHM validation for {path}") from exc
            skipped.append((path, repr(exc)))
            print(f"WARNING: skipping {path}: {exc}", flush=True)
    if not usable:
        raise RuntimeError("No usable input files remain after SMHM validation")
    return usable, skipped


def _accumulate(
    paths: list[Path],
    *,
    config: SMHMDiagnosticsConfig,
    edges: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    accumulator = _empty_accumulator(len(config.z_centers), edges.size - 1)
    input_kind_counts = {"lightcone": 0, "outputs": 0}
    output_tolerance = config.output_redshift_tolerance
    if output_tolerance is None:
        output_tolerance = config.z_half_width
    for file_index, path in enumerate(paths, start=1):
        with h5py.File(path, "r") as handle:
            print(f"[SMHM {file_index}/{len(paths)}] {path}", flush=True)
            if "/Lightcone" in handle:
                input_kind_counts["lightcone"] += 1
                for output_name in lightcone_output_names(handle):
                    output_group = handle[f"/Lightcone/{output_name}"]
                    node_data = output_group["nodeData"]
                    redshift = np.asarray(node_data[config.redshift_dataset][...], dtype=float)
                    weights = weights_from_node_data(node_data, float(config.unit_realization_scale), config.angular_weight_mode)
                    for case_index, z_center in enumerate(config.z_centers):
                        z_min = max(0.0, z_center - config.z_half_width)
                        z_max = z_center + config.z_half_width
                        z_selection = np.isfinite(redshift) & (redshift >= z_min) & (redshift < z_max)
                        if not np.any(z_selection):
                            continue
                        _record_matched_output(accumulator, case_index, float(np.nanmedian(redshift[z_selection])))
                        sliced_weights = np.zeros_like(weights)
                        sliced_weights[z_selection] = weights[z_selection]
                        _process_node_data(accumulator, case_index, node_data, sliced_weights, edges, config)
            if "/Outputs" in handle:
                input_kind_counts["outputs"] += 1
                for output_name in _output_names(handle, "/Outputs"):
                    output_group = handle[f"/Outputs/{output_name}"]
                    output_redshift = _fixed_output_redshift(output_group)
                    distances = np.abs(np.asarray(config.z_centers, dtype=float) - output_redshift)
                    case_index = int(np.argmin(distances))
                    if distances[case_index] > float(output_tolerance):
                        continue
                    node_data = output_group["nodeData"]
                    first_dataset = next(iter(node_data.values()))
                    weights = _fixed_output_weights(output_group, config, first_dataset.shape[0])
                    _record_matched_output(accumulator, case_index, output_redshift)
                    _process_node_data(accumulator, case_index, node_data, weights, edges, config)
    return accumulator, input_kind_counts


def _rows_from_accumulator(
    accumulator: dict[str, np.ndarray],
    config: SMHMDiagnosticsConfig,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    centers = centers_from_edges(edges)
    rows: list[dict[str, Any]] = []
    for case_index, z_center in enumerate(config.z_centers):
        z_min = max(0.0, z_center - config.z_half_width)
        z_max = z_center + config.z_half_width
        for bin_index, halo_center in enumerate(centers):
            sum_w = float(accumulator["sum_w"][case_index, bin_index])
            sum_w2 = float(accumulator["sum_w2"][case_index, bin_index])
            sum_w_y = float(accumulator["sum_w_y"][case_index, bin_index])
            sum_w_y2 = float(accumulator["sum_w_y2"][case_index, bin_index])
            raw_count = int(accumulator["raw_count"][case_index, bin_index])
            mean = sum_w_y / sum_w if sum_w > 0.0 else np.nan
            variance = max(sum_w_y2 / sum_w - mean**2, 0.0) if sum_w > 0.0 and np.isfinite(mean) else np.nan
            std = np.sqrt(variance) if np.isfinite(variance) else np.nan
            n_eff = sum_w**2 / sum_w2 if sum_w2 > 0.0 else np.nan
            stderr = std / np.sqrt(n_eff) if np.isfinite(std) and np.isfinite(n_eff) and n_eff > 0.0 else np.nan
            rows.append(
                {
                    "sample_label": f"z{z_center:g}",
                    "target_redshift": float(z_center),
                    "z_min": float(z_min),
                    "z_max": float(z_max),
                    "matched_output_redshift_min": float(accumulator["matched_redshift_min"][case_index]),
                    "matched_output_redshift_max": float(accumulator["matched_redshift_max"][case_index]),
                    "matched_outputs": int(accumulator["matched_outputs"][case_index]),
                    "bin_index": int(bin_index),
                    "log10_halo_mass_min": float(edges[bin_index]),
                    "log10_halo_mass_center": float(halo_center),
                    "log10_halo_mass_max": float(edges[bin_index + 1]),
                    "raw_count": raw_count,
                    "sum_weight": sum_w,
                    "effective_count": n_eff,
                    "mean_log10_stellar_mass": mean,
                    "std_log10_stellar_mass": std,
                    "stderr_mean_log10_stellar_mass": stderr,
                }
            )
    return rows


def _plot_smhm(frame: pd.DataFrame, output_path: Path, *, config: SMHMDiagnosticsConfig) -> None:
    n_cases = len(config.z_centers)
    ncols = min(3, n_cases)
    nrows = int(np.ceil(n_cases / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, z_center in zip(axes_flat, config.z_centers, strict=False):
        sample_label = f"z{z_center:g}"
        rows = frame.loc[frame["sample_label"] == sample_label].sort_values("bin_index")
        valid = rows["sum_weight"].to_numpy(dtype=float) > 0.0
        x = rows["log10_halo_mass_center"].to_numpy(dtype=float)
        y = rows["mean_log10_stellar_mass"].to_numpy(dtype=float)
        yerr = rows["stderr_mean_log10_stellar_mass"].to_numpy(dtype=float)
        axis.plot(x[valid], y[valid], color="#1f77b4", lw=2.0, marker="o", ms=3.0, label=r"$\langle\log M_\star\rangle$")
        finite_err = valid & np.isfinite(yerr)
        if np.any(finite_err):
            axis.fill_between(x[finite_err], y[finite_err] - yerr[finite_err], y[finite_err] + yerr[finite_err], color="#1f77b4", alpha=0.18, linewidth=0, label="standard error")
        diagonal_min = min(config.log10_halo_mass_min, np.nanmin(y[valid]) if np.any(valid) else config.log10_halo_mass_min)
        diagonal_max = max(config.log10_halo_mass_max, np.nanmax(y[valid]) if np.any(valid) else config.log10_halo_mass_max)
        axis.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], color="0.55", lw=1.0, ls=":", label=r"$M_\star=M_{\rm halo}$")
        matched = rows["matched_outputs"].iloc[0] if not rows.empty else 0
        z_match_min = rows["matched_output_redshift_min"].iloc[0] if not rows.empty else np.nan
        z_match_max = rows["matched_output_redshift_max"].iloc[0] if not rows.empty else np.nan
        if np.isfinite(z_match_min) and np.isfinite(z_match_max):
            subtitle = f"matched z={z_match_min:.3g}-{z_match_max:.3g}; outputs={matched:g}"
        else:
            subtitle = "no matched data"
        axis.set_title(f"SMHM {sample_label}\n{subtitle}", fontsize=10)
        axis.set_xlabel(r"$\log_{10}(M_{\rm halo}/M_\odot)$")
        axis.set_ylabel(r"$\langle\log_{10}(M_\star/M_\odot)\rangle$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=7, loc="best")
    for axis in axes_flat[n_cases:]:
        axis.axis("off")
    finite_y = frame.loc[frame["sum_weight"] > 0.0, "mean_log10_stellar_mass"].to_numpy(dtype=float)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size:
        lower = float(np.nanmin(finite_y))
        upper = float(np.nanmax(finite_y))
        margin = max(0.25, 0.08 * (upper - lower if upper > lower else 1.0))
        for axis in axes_flat[:n_cases]:
            axis.set_ylim(lower - margin, upper + margin)
    for axis in axes_flat[:n_cases]:
        axis.set_xlim(config.log10_halo_mass_min, config.log10_halo_mass_max)
    fig.suptitle(f"Stellar mass-halo mass relation ({config.galaxy_selection})", fontsize=13)
    fig.savefig(output_path, dpi=config.dpi)
    plt.close(fig)


def run_smhm_diagnostics(
    paths: list[Path],
    *,
    output_dir: Path,
    data_dir: Path,
    config: SMHMDiagnosticsConfig,
    skip_bad_files: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    usable_paths, skipped = _filter_usable_paths(paths, config, skip_bad_files=skip_bad_files)
    unit_scale = (
        resolve_unit_scale(config.unit_realization_scale, usable_paths, config.unit_realization_total)
        if isinstance(config.unit_realization_scale, str)
        else float(config.unit_realization_scale)
    )
    config = replace(config, unit_realization_scale=unit_scale)
    edges = _mass_edges(config)
    accumulator, input_kind_counts = _accumulate(usable_paths, config=config, edges=edges)
    rows = _rows_from_accumulator(accumulator, config, edges)
    csv_name = "stellar_mass_halo_mass_relation.csv"
    plot_name = "stellar_mass_halo_mass_relation.png"
    write_csv(data_dir / csv_name, rows)
    frame = pd.DataFrame(rows)
    _plot_smhm(frame, output_dir / plot_name, config=config)
    return {
        "usable_files": len(usable_paths),
        "skipped_files": [(str(path), reason) for path, reason in skipped],
        "unit_realization_scale": unit_scale,
        "input_kind_counts": input_kind_counts,
        "outputs": {
            "data": [csv_name],
            "plots": [plot_name],
        },
    }
