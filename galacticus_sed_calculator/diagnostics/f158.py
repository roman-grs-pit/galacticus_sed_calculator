from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "galacticus_diagnostics_mplconfig"))

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .common import (
    DEFAULT_OBSERVATION_DIR,
    centers_from_edges,
    filter_usable_f158_paths,
    finite_limits,
    lightcone_output_names,
    maybe_load_observation_catalog,
    resolve_unit_scale,
    weights_from_node_data,
    write_csv,
)


MAGNITUDE_DATASET = "apparentMagnitudeRomanWFI:F158"
ARCMIN2_PER_DEG2 = 3600.0


@dataclass(frozen=True)
class F158DiagnosticsConfig:
    magnitude_group: str = "dustAttenuatedNodeData"
    magnitude_limit: float = 26.0
    redshift_dataset: str = "lightconeRedshiftObserved"
    halo_mass_dataset: str = "basicMass"
    z_min: float = 0.0
    z_max: float = 8.0
    dz: float = 0.05
    coarse_dz: float = 0.10
    survey_area_deg2: float = 16.0
    apparent_magnitude_min: float = 16.0
    apparent_magnitude_max: float = 30.0
    apparent_magnitude_bin_width: float = 0.25
    z_centers: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    z_half_width: float = 0.25
    angular_weight_mode: str = "area"
    unit_realization_scale: str | float = "auto"
    unit_realization_total: int = 10000
    halo_scatter_max_points: int = 200000
    halo_sample_seed: int = 12345
    halo_summary_dz: float = 0.25
    halo_mass_resolution: float | None = None
    plot_individual_files: bool = False
    individual_file_alpha: float = 0.06
    dpi: int = 220
    cosmos_web_catalog: Path | None = None
    cosmos_web_area_deg2: float = 0.54
    cosmos_web_redshift_column: str | None = None
    cosmos_web_magnitude_column: str | None = None
    three_dhst_catalog: Path | None = None
    three_dhst_area_deg2: float = 0.249
    three_dhst_redshift_column: str | None = None
    three_dhst_magnitude_column: str | None = None
    cosmos2020_catalog: Path | None = None
    cosmos2020_area_deg2: float = 2.0
    cosmos2020_redshift_column: str | None = None
    cosmos2020_magnitude_column: str | None = None


def _edges(z_min: float, z_max: float, dz: float) -> np.ndarray:
    if dz <= 0.0:
        raise ValueError("redshift bin width must be positive")
    if z_max <= z_min:
        raise ValueError("z_max must be greater than z_min")
    n_bins = int(np.ceil((z_max - z_min) / dz))
    edges = z_min + dz * np.arange(n_bins + 1, dtype=float)
    edges[-1] = z_max
    return edges


def _magnitude_edges(m_min: float, m_max: float, dm: float) -> np.ndarray:
    if dm <= 0.0:
        raise ValueError("apparent magnitude bin width must be positive")
    if m_max <= m_min:
        raise ValueError("apparent magnitude max must be greater than min")
    n_bins = int(np.ceil((m_max - m_min) / dm))
    edges = m_min + dm * np.arange(n_bins + 1, dtype=float)
    edges[-1] = m_max
    return edges


def _case_definitions(z_centers: tuple[float, ...], z_half_width: float, magnitude_edges: np.ndarray) -> list[dict[str, Any]]:
    if z_half_width <= 0.0:
        raise ValueError("z_half_width must be positive")
    centers = centers_from_edges(magnitude_edges)
    return [
        {
            "sample_label": f"z{z:.2f}",
            "target_redshift": float(z),
            "z_min": max(0.0, float(z) - z_half_width),
            "z_max": float(z) + z_half_width,
            "centers": centers,
            "edges": magnitude_edges,
        }
        for z in z_centers
    ]


def _empty_nz_accumulator(edges: np.ndarray) -> dict[str, np.ndarray | int]:
    return {
        "weighted_counts": np.zeros(edges.size - 1, dtype=float),
        "weighted_variance": np.zeros(edges.size - 1, dtype=float),
        "raw_counts": np.zeros(edges.size - 1, dtype=int),
        "total_rows": 0,
        "selected_rows": 0,
    }


def _empty_magnitude_accumulators(cases: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {
        index: {
            "all_counts": np.zeros(len(case["centers"]), dtype=float),
            "all_variance": np.zeros(len(case["centers"]), dtype=float),
            "selected_counts": np.zeros(len(case["centers"]), dtype=float),
            "selected_variance": np.zeros(len(case["centers"]), dtype=float),
            "all_raw": np.zeros(len(case["centers"]), dtype=int),
            "selected_raw": np.zeros(len(case["centers"]), dtype=int),
            "raw_rows_in_redshift_window": 0,
        }
        for index, case in enumerate(cases)
    }


def _covered_unit_ids(paths: list[Path]) -> set[int]:
    covered: set[int] = set()
    import re

    for path in paths:
        match = re.search(r"-d(\d+)_(\d+)_", path.name)
        if match is None:
            continue
        start, end = (int(match.group(1)), int(match.group(2)))
        covered.update(range(start, end + 1))
    return covered


def _individual_file_scale_factor(path: Path, paths: list[Path]) -> float:
    covered_all = _covered_unit_ids(paths)
    covered_one = _covered_unit_ids([path])
    if covered_all and covered_one:
        return float(len(covered_all)) / float(len(covered_one))
    return float(max(len(paths), 1))


def _accumulate_magnitude_counts(
    accumulator: dict[str, Any],
    values: np.ndarray,
    weights: np.ndarray,
    selected: np.ndarray,
    edges: np.ndarray,
) -> None:
    all_counts, _ = np.histogram(values, bins=edges, weights=weights)
    all_var, _ = np.histogram(values, bins=edges, weights=weights**2)
    all_raw, _ = np.histogram(values, bins=edges)
    selected_counts, _ = np.histogram(values[selected], bins=edges, weights=weights[selected])
    selected_var, _ = np.histogram(values[selected], bins=edges, weights=weights[selected] ** 2)
    selected_raw, _ = np.histogram(values[selected], bins=edges)
    accumulator["all_counts"] += all_counts
    accumulator["all_variance"] += all_var
    accumulator["selected_counts"] += selected_counts
    accumulator["selected_variance"] += selected_var
    accumulator["all_raw"] += all_raw.astype(int)
    accumulator["selected_raw"] += selected_raw.astype(int)


def _add_scaled_magnitude_counts(
    accumulator: dict[str, Any],
    source: dict[str, Any],
    scale_factor: float,
) -> None:
    accumulator["raw_rows_in_redshift_window"] += int(source["raw_rows_in_redshift_window"])
    accumulator["all_counts"] += source["all_counts"] * scale_factor
    accumulator["all_variance"] += source["all_variance"] * scale_factor**2
    accumulator["selected_counts"] += source["selected_counts"] * scale_factor
    accumulator["selected_variance"] += source["selected_variance"] * scale_factor**2
    accumulator["all_raw"] += source["all_raw"].astype(int)
    accumulator["selected_raw"] += source["selected_raw"].astype(int)


def _rows_from_nz(
    edges: np.ndarray,
    accum: dict[str, np.ndarray | int],
    *,
    survey_area_deg2: float,
) -> list[dict[str, Any]]:
    centers = centers_from_edges(edges)
    widths = np.diff(edges)
    weighted_counts = np.asarray(accum["weighted_counts"], dtype=float)
    weighted_variance = np.asarray(accum["weighted_variance"], dtype=float)
    raw_counts = np.asarray(accum["raw_counts"], dtype=int)
    dndz = weighted_counts / widths
    dndz_std = np.sqrt(np.clip(weighted_variance, 0.0, None)) / widths
    cumulative = np.cumsum(weighted_counts)
    rows: list[dict[str, Any]] = []
    for index, center in enumerate(centers):
        rows.append(
            {
                "bin_index": index,
                "z_min": float(edges[index]),
                "z_center": float(center),
                "z_max": float(edges[index + 1]),
                "dz": float(widths[index]),
                "weighted_surface_density_deg2": float(weighted_counts[index]),
                "weighted_surface_density_deg2_std": float(np.sqrt(max(weighted_variance[index], 0.0))),
                "dN_dz_deg2": float(dndz[index]),
                "dN_dz_deg2_std": float(dndz_std[index]),
                "cumulative_surface_density_deg2": float(cumulative[index]),
                "expected_count_in_survey": float(weighted_counts[index] * survey_area_deg2),
                "raw_selected_count": int(raw_counts[index]),
            }
        )
    return rows


def _rows_from_apparent_magnitude_distributions(
    cases: list[dict[str, Any]],
    accumulators: dict[int, dict[str, Any]],
    *,
    magnitude_limit: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        acc = accumulators[index]
        edges = np.asarray(case["edges"], dtype=float)
        centers = np.asarray(case["centers"], dtype=float)
        dmag = np.diff(edges)
        for bin_index, center in enumerate(centers):
            for selection_name, counts_key, variance_key, raw_key in [
                ("all", "all_counts", "all_variance", "all_raw"),
                ("magnitude_limited", "selected_counts", "selected_variance", "selected_raw"),
            ]:
                counts = float(acc[counts_key][bin_index])
                sigma_counts = float(np.sqrt(max(acc[variance_key][bin_index], 0.0)))
                surface_density = counts / dmag[bin_index]
                surface_density_std = sigma_counts / dmag[bin_index]
                rows.append(
                    {
                        "selection": selection_name,
                        "sample_label": case["sample_label"],
                        "target_redshift": float(case["target_redshift"]),
                        "z_min": float(case["z_min"]),
                        "z_max": float(case["z_max"]),
                        "magnitude_limit": float(magnitude_limit),
                        "complete_for_magnitude_limit": bool(center <= magnitude_limit),
                        "bin_index": int(bin_index),
                        "apparent_magnitude_min": float(edges[bin_index]),
                        "apparent_magnitude_center": float(center),
                        "apparent_magnitude_max": float(edges[bin_index + 1]),
                        "weighted_surface_density_deg2": counts,
                        "raw_count": int(acc[raw_key][bin_index]),
                        "raw_rows_in_redshift_window": int(acc["raw_rows_in_redshift_window"]),
                        "surface_density_deg2_mag": float(surface_density),
                        "surface_density_deg2_mag_std": float(surface_density_std),
                        "log10_surface_density_deg2_mag": (
                            np.log10(surface_density) if surface_density > 0.0 else np.nan
                        ),
                        "log10_surface_density_deg2_mag_std": (
                            surface_density_std / (surface_density * np.log(10.0))
                            if surface_density > 0.0
                            else np.nan
                        ),
                    }
                )
    return rows


def _observation_nz_rows(catalogs: list[Any], edges: np.ndarray, magnitude_limit: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    centers = centers_from_edges(edges)
    widths = np.diff(edges)
    for catalog in catalogs:
        selected = (
            np.isfinite(catalog.redshift)
            & np.isfinite(catalog.magnitude)
            & (catalog.redshift >= edges[0])
            & (catalog.redshift < edges[-1])
            & (catalog.magnitude < magnitude_limit)
        )
        counts, _ = np.histogram(catalog.redshift[selected], bins=edges)
        cumulative = np.cumsum(counts / catalog.area_deg2)
        for index, center in enumerate(centers):
            surface_density = counts[index] / catalog.area_deg2
            rows.append(
                {
                    "source": catalog.label,
                    "bin_index": index,
                    "z_min": float(edges[index]),
                    "z_center": float(center),
                    "z_max": float(edges[index + 1]),
                    "dz": float(widths[index]),
                    "count": int(counts[index]),
                    "area_deg2": float(catalog.area_deg2),
                    "weighted_surface_density_deg2": float(surface_density),
                    "dN_dz_deg2": float(surface_density / widths[index]),
                    "dN_dz_deg2_std": float(np.sqrt(counts[index]) / (catalog.area_deg2 * widths[index])),
                    "cumulative_surface_density_deg2": float(cumulative[index]),
                }
            )
    return rows


def _observation_apparent_magnitude_rows(
    catalogs: list[Any],
    cases: list[dict[str, Any]],
    *,
    magnitude_limit: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for catalog in catalogs:
        finite = np.isfinite(catalog.redshift) & np.isfinite(catalog.magnitude)
        for case in cases:
            edges = np.asarray(case["edges"], dtype=float)
            centers = np.asarray(case["centers"], dtype=float)
            dmag = np.diff(edges)
            selected = finite & (catalog.redshift >= case["z_min"]) & (catalog.redshift < case["z_max"])
            counts, _ = np.histogram(catalog.magnitude[selected], bins=edges)
            for bin_index, center in enumerate(centers):
                surface_density = counts[bin_index] / (catalog.area_deg2 * dmag[bin_index])
                surface_density_std = np.sqrt(counts[bin_index]) / (catalog.area_deg2 * dmag[bin_index])
                rows.append(
                    {
                        "source": catalog.label,
                        "selection": "catalog_available",
                        "sample_label": case["sample_label"],
                        "target_redshift": float(case["target_redshift"]),
                        "z_min": float(case["z_min"]),
                        "z_max": float(case["z_max"]),
                        "area_deg2": float(catalog.area_deg2),
                        "magnitude_limit": float(magnitude_limit),
                        "complete_for_magnitude_limit": bool(center <= magnitude_limit),
                        "bin_index": int(bin_index),
                        "apparent_magnitude_min": float(edges[bin_index]),
                        "apparent_magnitude_center": float(center),
                        "apparent_magnitude_max": float(edges[bin_index + 1]),
                        "count": int(counts[bin_index]),
                        "surface_density_deg2_mag": float(surface_density),
                        "surface_density_deg2_mag_std": float(surface_density_std),
                        "log10_surface_density_deg2_mag": (
                            np.log10(surface_density) if surface_density > 0.0 else np.nan
                        ),
                        "log10_surface_density_deg2_mag_std": (
                            surface_density_std / (surface_density * np.log(10.0))
                            if surface_density > 0.0
                            else np.nan
                        ),
                    }
                )
    return rows


def _halo_summary_rows(sample: pd.DataFrame, *, z_min: float, z_max: float, dz: float) -> list[dict[str, Any]]:
    if sample.empty:
        return []
    edges = _edges(z_min, z_max, dz)
    rows: list[dict[str, Any]] = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        values = sample.loc[(sample["redshift"] >= lo) & (sample["redshift"] < hi), "log10_halo_mass"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        p16, p50, p84 = np.percentile(values, [16.0, 50.0, 84.0])
        rows.append(
            {
                "bin_index": index,
                "z_min": float(lo),
                "z_center": float(0.5 * (lo + hi)),
                "z_max": float(hi),
                "sample_count": int(values.size),
                "log10_halo_mass_p16": float(p16),
                "log10_halo_mass_median": float(p50),
                "log10_halo_mass_p84": float(p84),
                "log10_halo_mass_min": float(np.min(values)),
            }
        )
    return rows


def _load_observation_catalogs(config: F158DiagnosticsConfig) -> list[Any]:
    return [
        catalog
        for catalog in [
            maybe_load_observation_catalog(
                config.three_dhst_catalog,
                DEFAULT_OBSERVATION_DIR / "3dhst_f160w_catalog.csv",
                label="3D-HST/CANDELS F160W",
                default_area_deg2=config.three_dhst_area_deg2,
                redshift_column=config.three_dhst_redshift_column,
                magnitude_column=config.three_dhst_magnitude_column,
            ),
            maybe_load_observation_catalog(
                config.cosmos_web_catalog,
                DEFAULT_OBSERVATION_DIR / "cosmos2025_f150w_catalog.csv",
                label="COSMOS-Web/COSMOS2025 F150W",
                default_area_deg2=config.cosmos_web_area_deg2,
                redshift_column=config.cosmos_web_redshift_column,
                magnitude_column=config.cosmos_web_magnitude_column,
            ),
            maybe_load_observation_catalog(
                config.cosmos2020_catalog,
                DEFAULT_OBSERVATION_DIR / "cosmos2020_h_catalog.csv",
                label="COSMOS2020 H",
                default_area_deg2=config.cosmos2020_area_deg2,
                redshift_column=config.cosmos2020_redshift_column,
                magnitude_column=config.cosmos2020_magnitude_column,
            ),
        ]
        if catalog is not None
    ]


def _accumulate(
    paths: list[Path],
    *,
    config: F158DiagnosticsConfig,
    fine_edges: np.ndarray,
    coarse_edges: np.ndarray,
    magnitude_cases: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, np.ndarray | int]], dict[int, dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    edge_sets = {"fine": fine_edges, "coarse": coarse_edges}
    nz_accumulators = {name: _empty_nz_accumulator(edges) for name, edges in edge_sets.items()}
    mag_accumulators = _empty_magnitude_accumulators(magnitude_cases)
    individual_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.halo_sample_seed)
    halo_sample_z = np.asarray([], dtype=float)
    halo_sample_log_mass = np.asarray([], dtype=float)
    halo_sample_priority = np.asarray([], dtype=float)

    for file_index, path in enumerate(paths, start=1):
        file_nz_accumulators = {name: _empty_nz_accumulator(edges) for name, edges in edge_sets.items()} if config.plot_individual_files else {}
        file_scale_factor = _individual_file_scale_factor(path, paths) if config.plot_individual_files else 1.0
        with h5py.File(path, "r") as handle:
            print(f"[{file_index}/{len(paths)}] {path}", flush=True)
            for output_name in lightcone_output_names(handle):
                output_group = handle[f"/Lightcone/{output_name}"]
                node_data = output_group["nodeData"]
                redshift = np.asarray(node_data[config.redshift_dataset][...], dtype=float)
                magnitude = np.asarray(output_group[config.magnitude_group][MAGNITUDE_DATASET][...], dtype=float)
                weights = weights_from_node_data(node_data, config.unit_realization_scale, config.angular_weight_mode)
                halo_mass = np.asarray(node_data[config.halo_mass_dataset][...], dtype=float)
                base_valid = np.isfinite(redshift) & np.isfinite(magnitude) & np.isfinite(weights)
                selected_valid = (
                    base_valid
                    & (redshift >= config.z_min)
                    & (redshift < config.z_max)
                    & (magnitude < config.magnitude_limit)
                )
                for accum in nz_accumulators.values():
                    accum["total_rows"] = int(accum["total_rows"]) + int(redshift.size)
                    accum["selected_rows"] = int(accum["selected_rows"]) + int(np.count_nonzero(selected_valid))
                halo_valid = selected_valid & np.isfinite(halo_mass) & (halo_mass > 0.0)
                if np.any(halo_valid) and config.halo_scatter_max_points > 0:
                    candidate_z = redshift[halo_valid]
                    candidate_log_mass = np.log10(halo_mass[halo_valid])
                    candidate_priority = rng.random(candidate_z.size)
                    halo_sample_z = np.concatenate([halo_sample_z, candidate_z])
                    halo_sample_log_mass = np.concatenate([halo_sample_log_mass, candidate_log_mass])
                    halo_sample_priority = np.concatenate([halo_sample_priority, candidate_priority])
                    if halo_sample_z.size > config.halo_scatter_max_points:
                        keep = np.argpartition(halo_sample_priority, config.halo_scatter_max_points - 1)[: config.halo_scatter_max_points]
                        halo_sample_z = halo_sample_z[keep]
                        halo_sample_log_mass = halo_sample_log_mass[keep]
                        halo_sample_priority = halo_sample_priority[keep]
                for name, edges in edge_sets.items():
                    edge_valid = selected_valid & (redshift >= edges[0]) & (redshift < edges[-1])
                    if not np.any(edge_valid):
                        continue
                    counts, _ = np.histogram(redshift[edge_valid], bins=edges, weights=weights[edge_valid])
                    variance, _ = np.histogram(redshift[edge_valid], bins=edges, weights=weights[edge_valid] ** 2)
                    raw, _ = np.histogram(redshift[edge_valid], bins=edges)
                    nz_accumulators[name]["weighted_counts"] = np.asarray(nz_accumulators[name]["weighted_counts"], dtype=float) + counts
                    nz_accumulators[name]["weighted_variance"] = np.asarray(nz_accumulators[name]["weighted_variance"], dtype=float) + variance
                    nz_accumulators[name]["raw_counts"] = np.asarray(nz_accumulators[name]["raw_counts"], dtype=int) + raw.astype(int)
                    if config.plot_individual_files:
                        file_nz_accumulators[name]["weighted_counts"] = np.asarray(file_nz_accumulators[name]["weighted_counts"], dtype=float) + counts * file_scale_factor
                        file_nz_accumulators[name]["weighted_variance"] = np.asarray(file_nz_accumulators[name]["weighted_variance"], dtype=float) + variance * file_scale_factor**2
                        file_nz_accumulators[name]["raw_counts"] = np.asarray(file_nz_accumulators[name]["raw_counts"], dtype=int) + raw.astype(int)
                        file_nz_accumulators[name]["total_rows"] = int(file_nz_accumulators[name]["total_rows"]) + int(redshift.size)
                        file_nz_accumulators[name]["selected_rows"] = int(file_nz_accumulators[name]["selected_rows"]) + int(np.count_nonzero(edge_valid))
                for index, case in enumerate(magnitude_cases):
                    z_selection = base_valid & (redshift >= case["z_min"]) & (redshift < case["z_max"])
                    if not np.any(z_selection):
                        continue
                    mag_accumulators[index]["raw_rows_in_redshift_window"] += int(np.count_nonzero(z_selection))
                    values = magnitude[z_selection]
                    selected = values < config.magnitude_limit
                    _accumulate_magnitude_counts(
                        mag_accumulators[index],
                        values,
                        weights[z_selection],
                        selected,
                        np.asarray(case["edges"], dtype=float),
                    )
        if config.plot_individual_files:
            for edge_name, edges in edge_sets.items():
                rows = _rows_from_nz(edges, file_nz_accumulators[edge_name], survey_area_deg2=1.0)
                for row in rows:
                    row["file"] = str(path)
                    row["file_name"] = path.name
                    row["edge_set"] = edge_name
                    row["individual_file_scale_factor"] = float(file_scale_factor)
                individual_rows.extend(rows)

    halo_sample = pd.DataFrame({"redshift": halo_sample_z, "log10_halo_mass": halo_sample_log_mass}).sort_values(
        "redshift",
        ignore_index=True,
    )
    return nz_accumulators, mag_accumulators, halo_sample, pd.DataFrame(individual_rows)


def _plot_nz(
    model_frame: pd.DataFrame,
    individual_frame: pd.DataFrame,
    obs_frame: pd.DataFrame,
    halo_sample: pd.DataFrame,
    halo_summary: pd.DataFrame,
    output_path: Path,
    *,
    config: F158DiagnosticsConfig,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 10.2), sharex=True, constrained_layout=True)
    model_color = "#1f77b4"
    if not individual_frame.empty:
        first = True
        for _, group in individual_frame.groupby("file", sort=False):
            group = group.sort_values("z_center")
            label = "individual files" if first else "_nolegend_"
            axes[0].plot(
                group["z_center"],
                group["dN_dz_deg2"] / ARCMIN2_PER_DEG2,
                color=model_color,
                lw=0.7,
                alpha=config.individual_file_alpha,
                label=label,
                zorder=1,
            )
            axes[1].plot(
                group["z_center"],
                group["cumulative_surface_density_deg2"] / ARCMIN2_PER_DEG2,
                color=model_color,
                lw=0.7,
                alpha=config.individual_file_alpha,
                label=label,
                zorder=1,
            )
            first = False
    x = model_frame["z_center"].to_numpy(dtype=float)
    y = model_frame["dN_dz_deg2"].to_numpy(dtype=float) / ARCMIN2_PER_DEG2
    yerr = model_frame["dN_dz_deg2_std"].to_numpy(dtype=float) / ARCMIN2_PER_DEG2
    axes[0].plot(x, y, color=model_color, lw=2.0, label="Roman UNIT lightcone", zorder=3)
    finite = np.isfinite(y) & np.isfinite(yerr)
    axes[0].fill_between(x[finite], y[finite] - yerr[finite], y[finite] + yerr[finite], color=model_color, alpha=0.18)
    cumulative_deg2 = model_frame["cumulative_surface_density_deg2"].to_numpy(dtype=float)
    cumulative = cumulative_deg2 / ARCMIN2_PER_DEG2
    axes[1].plot(x, cumulative, color=model_color, lw=2.0, label="Roman UNIT lightcone", zorder=3)
    colors = {"3D-HST/CANDELS F160W": "0.10", "COSMOS-Web/COSMOS2025 F150W": "0.35", "COSMOS2020 H": "#d95f02"}
    if not obs_frame.empty:
        for source, group in obs_frame.groupby("source", sort=False):
            group = group.sort_values("z_center")
            color = colors.get(str(source), "0.35")
            axes[0].plot(group["z_center"], group["dN_dz_deg2"] / ARCMIN2_PER_DEG2, lw=1.4, color=color, alpha=0.85, label=str(source))
            axes[1].plot(group["z_center"], group["cumulative_surface_density_deg2"] / ARCMIN2_PER_DEG2, lw=1.4, color=color, alpha=0.85, label=str(source))
    axes[0].set_ylabel(r"$dN/dz/d\Omega\ [\mathrm{arcmin}^{-2}]$")
    axes[1].set_ylabel(r"$N(<z)/d\Omega\ [\mathrm{arcmin}^{-2}]$")
    axes[1].set_xlabel("redshift")
    axes[0].set_title(f"F158-selected redshift distribution, mF158 < {config.magnitude_limit:g}")
    for axis in axes[:2]:
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].text(
        0.98,
        0.05,
        f"Roman mock expectation for {config.survey_area_deg2:g} deg2 = "
        f"{cumulative_deg2[-1] * config.survey_area_deg2:,.0f} objects",
        transform=axes[1].transAxes,
        fontsize=8,
        color="0.25",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    if not halo_sample.empty:
        axes[2].scatter(
            halo_sample["redshift"],
            halo_sample["log10_halo_mass"],
            s=1.0,
            alpha=0.08,
            color=model_color,
            linewidths=0,
            rasterized=True,
        )
    if not halo_summary.empty:
        summary = halo_summary.sort_values("z_center")
        axes[2].fill_between(
            summary["z_center"],
            summary["log10_halo_mass_p16"],
            summary["log10_halo_mass_p84"],
            color=model_color,
            alpha=0.16,
            linewidth=0,
            label="16-84 percentile",
        )
        axes[2].plot(summary["z_center"], summary["log10_halo_mass_median"], color=model_color, lw=2.0, label="median")
        axes[2].plot(summary["z_center"], summary["log10_halo_mass_min"], color="0.25", lw=1.1, ls=":", label="sample minimum")
    if config.halo_mass_resolution is not None and config.halo_mass_resolution > 0.0:
        axes[2].axhline(np.log10(config.halo_mass_resolution), color="#d95f02", lw=1.4, ls="--", label=f"resolution ref. {config.halo_mass_resolution:.2g} Msun")
    axes[2].set_ylabel(r"$\log_{10}(M_{\rm halo}/M_\odot)$")
    axes[2].set_xlabel("redshift")
    axes[2].set_title(f"Selected-galaxy halo masses from nodeData/{config.halo_mass_dataset}")
    axes[2].grid(alpha=0.22)
    handles, labels = axes[2].get_legend_handles_labels()
    if not halo_sample.empty:
        handles.insert(0, Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=model_color, markeredgecolor="none", markersize=5, alpha=0.55))
        labels.insert(0, "selected galaxies")
    axes[2].legend(handles, labels, frameon=False, fontsize=8, loc="best")
    limits = finite_limits(y, obs_frame["dN_dz_deg2"].to_numpy(dtype=float) / ARCMIN2_PER_DEG2 if not obs_frame.empty else None)
    if limits is not None:
        axes[0].set_ylim(bottom=max(0.0, limits[0]), top=limits[1])
    axes[2].set_xlim(config.z_min, config.z_max)
    fig.savefig(output_path, dpi=config.dpi)
    plt.close(fig)


def _plot_apparent_magnitude_distributions(
    model_frame: pd.DataFrame,
    obs_frame: pd.DataFrame,
    cases: list[dict[str, Any]],
    output_path: Path,
    *,
    config: F158DiagnosticsConfig,
) -> None:
    ncols = 3 if len(cases) > 3 else len(cases)
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.1 * ncols, 3.8 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    obs_colors = {"3D-HST/CANDELS F160W": "0.10", "COSMOS-Web/COSMOS2025 F150W": "0.20", "COSMOS2020 H": "#d95f02"}
    for axis, case in zip(axes_flat, cases, strict=False):
        label = case["sample_label"]
        subset = model_frame.loc[model_frame["sample_label"] == label].copy()
        y_values: list[np.ndarray] = []
        for selection, color, linestyle, plot_label in [
            ("all", "0.55", "--", "all model galaxies"),
            ("magnitude_limited", "#1f77b4", "-", "model mF158 < limit"),
        ]:
            rows = subset.loc[subset["selection"] == selection].sort_values("bin_index")
            x = rows["apparent_magnitude_center"].to_numpy(dtype=float)
            density = rows["surface_density_deg2_mag"].to_numpy(dtype=float)
            y = np.full_like(density, np.nan, dtype=float)
            positive = density > 0.0
            y[positive] = np.log10(density[positive])
            axis.plot(x, y, color=color, ls=linestyle, lw=2.0, marker="s" if selection == "magnitude_limited" else None, ms=3.0, label=plot_label, zorder=3)
            sigma = rows["log10_surface_density_deg2_mag_std"].to_numpy(dtype=float)
            finite = positive & np.isfinite(sigma)
            if selection == "magnitude_limited" and np.any(finite):
                axis.fill_between(x[finite], y[finite] - sigma[finite], y[finite] + sigma[finite], color=color, alpha=0.16, linewidth=0)
            y_values.append(y[np.isfinite(y)])
        axis.axvline(config.magnitude_limit, color="0.35", ls=":", lw=1.1, label=f"m={config.magnitude_limit:g}")
        axis.axvspan(config.magnitude_limit, case["edges"][-1], color="0.8", alpha=0.18, linewidth=0)
        if not obs_frame.empty:
            obs_subset = obs_frame.loc[obs_frame["sample_label"] == label].copy()
            for source, source_group in obs_subset.groupby("source", sort=False):
                color = obs_colors.get(str(source), "0.25")
                observed = source_group.loc[source_group["surface_density_deg2_mag"] > 0.0].sort_values("bin_index")
                if not observed.empty:
                    axis.errorbar(
                        observed["apparent_magnitude_center"],
                        observed["log10_surface_density_deg2_mag"],
                        yerr=observed["log10_surface_density_deg2_mag_std"],
                        fmt="o",
                        ms=3.4,
                        capsize=2.0,
                        color=color,
                        ecolor=color,
                        alpha=0.82,
                        label=str(source),
                    )
                    y_values.append(observed["log10_surface_density_deg2_mag"].to_numpy(dtype=float))
        limits = finite_limits(*y_values)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.invert_xaxis()
        axis.grid(alpha=0.22)
        axis.set_title(f"F158 apparent magnitudes {label} ({case['z_min']:.2f}<z<{case['z_max']:.2f})")
        axis.set_xlabel("observed apparent magnitude")
        axis.set_ylabel(r"$\log_{10} dN/dm/d\Omega\ [\mathrm{deg}^{-2}\,\mathrm{mag}^{-1}]$")
        axis.legend(frameon=False, fontsize=7)
    for axis in axes_flat[len(cases) :]:
        axis.axis("off")
    fig.savefig(output_path, dpi=config.dpi)
    plt.close(fig)


def run_f158_diagnostics(
    paths: list[Path],
    *,
    output_dir: Path,
    data_dir: Path,
    config: F158DiagnosticsConfig,
    skip_bad_files: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    usable_paths, skipped = filter_usable_f158_paths(
        paths,
        magnitude_group=config.magnitude_group,
        magnitude_dataset=MAGNITUDE_DATASET,
        redshift_dataset=config.redshift_dataset,
        skip_bad_files=skip_bad_files,
    )
    unit_scale = (
        resolve_unit_scale(config.unit_realization_scale, usable_paths, config.unit_realization_total)
        if isinstance(config.unit_realization_scale, str)
        else float(config.unit_realization_scale)
    )
    config = replace(config, unit_realization_scale=unit_scale)
    fine_edges = _edges(config.z_min, config.z_max, config.dz)
    coarse_edges = _edges(config.z_min, config.z_max, config.coarse_dz)
    magnitude_edges = _magnitude_edges(
        config.apparent_magnitude_min,
        config.apparent_magnitude_max,
        config.apparent_magnitude_bin_width,
    )
    magnitude_cases = _case_definitions(config.z_centers, config.z_half_width, magnitude_edges)
    nz_accumulators, mag_accumulators, halo_sample, individual_nz_frame = _accumulate(
        usable_paths,
        config=config,
        fine_edges=fine_edges,
        coarse_edges=coarse_edges,
        magnitude_cases=magnitude_cases,
    )
    fine_rows = _rows_from_nz(fine_edges, nz_accumulators["fine"], survey_area_deg2=config.survey_area_deg2)
    coarse_rows = _rows_from_nz(coarse_edges, nz_accumulators["coarse"], survey_area_deg2=config.survey_area_deg2)
    apparent_rows = _rows_from_apparent_magnitude_distributions(
        magnitude_cases,
        mag_accumulators,
        magnitude_limit=config.magnitude_limit,
    )
    data_outputs = [
        f"f158_nz_dz{config.dz:g}.csv",
        f"f158_nz_dz{config.coarse_dz:g}.csv",
        "f158_apparent_magnitude_distributions.csv",
    ]
    write_csv(data_dir / f"f158_nz_dz{config.dz:g}.csv", fine_rows)
    write_csv(data_dir / f"f158_nz_dz{config.coarse_dz:g}.csv", coarse_rows)
    write_csv(data_dir / "f158_apparent_magnitude_distributions.csv", apparent_rows)
    if config.plot_individual_files and not individual_nz_frame.empty:
        individual_nz_frame.to_csv(data_dir / "f158_nz_individual_files.csv", index=False)
        data_outputs.append("f158_nz_individual_files.csv")
    catalogs = _load_observation_catalogs(config)
    obs_nz_rows = _observation_nz_rows(catalogs, fine_edges, config.magnitude_limit) if catalogs else []
    obs_apparent_rows = (
        _observation_apparent_magnitude_rows(catalogs, magnitude_cases, magnitude_limit=config.magnitude_limit)
        if catalogs
        else []
    )
    if obs_nz_rows:
        write_csv(data_dir / "observed_f158_analog_nz.csv", obs_nz_rows)
        data_outputs.append("observed_f158_analog_nz.csv")
    if obs_apparent_rows:
        write_csv(data_dir / "observed_f158_analog_apparent_magnitude_distributions.csv", obs_apparent_rows)
        data_outputs.append("observed_f158_analog_apparent_magnitude_distributions.csv")
    halo_sample.to_csv(data_dir / "f158_selected_halo_mass_sample.csv", index=False)
    data_outputs.append("f158_selected_halo_mass_sample.csv")
    halo_summary_rows = _halo_summary_rows(
        halo_sample,
        z_min=config.z_min,
        z_max=config.z_max,
        dz=config.halo_summary_dz,
    )
    if halo_summary_rows:
        write_csv(data_dir / "f158_selected_halo_mass_summary.csv", halo_summary_rows)
        data_outputs.append("f158_selected_halo_mass_summary.csv")
    model_nz_frame = pd.DataFrame(fine_rows)
    obs_nz_frame = pd.DataFrame(obs_nz_rows)
    halo_summary_frame = pd.DataFrame(halo_summary_rows)
    individual_fine_nz_frame = pd.DataFrame()
    if config.plot_individual_files and not individual_nz_frame.empty:
        individual_fine_nz_frame = individual_nz_frame.loc[individual_nz_frame["edge_set"] == "fine"].copy()
    _plot_nz(
        model_nz_frame,
        individual_fine_nz_frame,
        obs_nz_frame,
        halo_sample,
        halo_summary_frame,
        output_dir / "f158_nz.png",
        config=config,
    )
    _plot_apparent_magnitude_distributions(
        pd.DataFrame(apparent_rows),
        pd.DataFrame(obs_apparent_rows),
        magnitude_cases,
        output_dir / "f158_apparent_magnitude_distributions.png",
        config=config,
    )
    return {
        "usable_files": len(usable_paths),
        "skipped_files": [(str(path), reason) for path, reason in skipped],
        "total_rows": int(nz_accumulators["fine"]["total_rows"]),
        "selected_rows": int(nz_accumulators["fine"]["selected_rows"]),
        "observation_catalogs": [catalog.metadata for catalog in catalogs],
        "outputs": {
            "plots": ["f158_nz.png", "f158_apparent_magnitude_distributions.png"],
            "data": data_outputs,
        },
    }
