#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "galacticus_diagnostics_mplconfig"))

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments.
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from galacticus_sed_calculator.diagnostics.common import (  # noqa: E402
    expand_inputs,
    format_unit_id_ranges,
    covered_unit_ids,
)
from galacticus_sed_calculator.diagnostics.emission_line_lfs import (  # noqa: E402
    run_emission_line_diagnostics,
)
from galacticus_sed_calculator.diagnostics.f158 import (  # noqa: E402
    F158DiagnosticsConfig,
    run_f158_diagnostics,
)
from galacticus_sed_calculator.diagnostics.smhm import (  # noqa: E402
    SMHMDiagnosticsConfig,
    run_smhm_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate standard QA diagnostics for Galacticus Roman mock lightcone catalogs. "
            "PNG files are written directly to --output-dir; CSVs and provenance are stored "
            "under --output-dir/data and --output-dir/metadata."
        )
    )
    parser.add_argument("inputs", nargs="*", type=Path, default=[], help="Input HDF5 files.")
    parser.add_argument("--input-glob", action="append", default=[], help="Additional quoted glob of HDF5 files.")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics"))
    parser.add_argument("--skip-bad-files", action="store_true")
    parser.add_argument("--no-f158", action="store_true", help="Do not make F158 n(z)/magnitude diagnostics.")
    parser.add_argument("--no-emission-lines", action="store_true", help="Do not make emission-line LF diagnostics.")
    parser.add_argument("--no-smhm", action="store_true", help="Do not make stellar mass-halo mass diagnostics.")
    parser.add_argument("--unit-realization-scale", default="auto")
    parser.add_argument("--unit-realization-total", type=int, default=10000)
    parser.add_argument("--redshift-dataset", default="lightconeRedshiftObserved")
    parser.add_argument(
        "--angular-weight-mode",
        choices=["area", "surface-density"],
        default="area",
    )
    parser.add_argument("--magnitude-group", choices=["nodeData", "dustAttenuatedNodeData"], default="dustAttenuatedNodeData")
    parser.add_argument("--magnitude-limit", type=float, default=26.0)
    parser.add_argument("--survey-area-deg2", type=float, default=16.0)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=8.0)
    parser.add_argument("--dz", type=float, default=0.05)
    parser.add_argument("--coarse-dz", type=float, default=0.10)
    parser.add_argument("--apparent-magnitude-min", type=float, default=16.0)
    parser.add_argument("--apparent-magnitude-max", type=float, default=30.0)
    parser.add_argument("--apparent-magnitude-bin-width", type=float, default=0.25)
    parser.add_argument("--z-centers", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    parser.add_argument("--z-half-width", type=float, default=0.25)
    parser.add_argument("--halo-mass-dataset", default="basicMass")
    parser.add_argument("--halo-scatter-max-points", type=int, default=200000)
    parser.add_argument("--halo-sample-seed", type=int, default=12345)
    parser.add_argument("--halo-summary-dz", type=float, default=0.25)
    parser.add_argument("--halo-mass-resolution", type=float, default=None)
    parser.add_argument("--smhm-z-centers", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    parser.add_argument("--smhm-z-half-width", type=float, default=0.10)
    parser.add_argument("--smhm-output-redshift-tolerance", type=float, default=None)
    parser.add_argument("--smhm-log10-halo-mass-min", type=float, default=9.5)
    parser.add_argument("--smhm-log10-halo-mass-max", type=float, default=15.0)
    parser.add_argument("--smhm-log10-halo-mass-bin-width", type=float, default=0.25)
    parser.add_argument("--smhm-disk-stellar-mass-dataset", default="diskMassStellar")
    parser.add_argument("--smhm-spheroid-stellar-mass-dataset", default="spheroidMassStellar")
    parser.add_argument("--smhm-selection", choices=["all", "centrals", "satellites"], default="all")
    parser.add_argument("--smhm-fill-missing-node-weights-with-median", action="store_true")
    parser.add_argument("--plot-individual-files", action="store_true")
    parser.add_argument("--individual-file-alpha", type=float, default=0.06)
    parser.add_argument("--cosmos-web-catalog", type=Path, default=None)
    parser.add_argument("--cosmos-web-area-deg2", type=float, default=0.54)
    parser.add_argument("--cosmos-web-redshift-column", default=None)
    parser.add_argument("--cosmos-web-magnitude-column", default=None)
    parser.add_argument("--three-dhst-catalog", type=Path, default=None)
    parser.add_argument("--three-dhst-area-deg2", type=float, default=0.249)
    parser.add_argument("--three-dhst-redshift-column", default=None)
    parser.add_argument("--three-dhst-magnitude-column", default=None)
    parser.add_argument("--cosmos2020-catalog", type=Path, default=None)
    parser.add_argument("--cosmos2020-area-deg2", type=float, default=2.0)
    parser.add_argument("--cosmos2020-redshift-column", default=None)
    parser.add_argument("--cosmos2020-magnitude-column", default=None)
    parser.add_argument(
        "--no-packaged-observation-reference",
        action="store_true",
        help="Do not use the bundled binned COSMOS-Web F150W reference curves.",
    )
    parser.add_argument("--include-agn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--halpha-target-hdf5", type=Path, default=None)
    parser.add_argument("--point-redshift-half-width", type=float, default=0.10)
    parser.add_argument("--min-log10-phi", type=float, default=-8.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_catalog_manifest(path: Path, paths: list[Path], total_units: int) -> None:
    import csv
    import re

    covered = covered_unit_ids(paths)
    missing = sorted(set(range(1, total_units + 1)) - covered) if covered else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "file_name", "unit_start", "unit_end"])
        writer.writeheader()
        for catalog_path in paths:
            match = re.search(r"-d(\d+)_(\d+)_", catalog_path.name)
            start, end = (match.group(1), match.group(2)) if match else ("", "")
            writer.writerow(
                {
                    "file": str(catalog_path),
                    "file_name": catalog_path.name,
                    "unit_start": start,
                    "unit_end": end,
                }
            )
    if missing:
        missing_path = path.with_name("missing_unit_ids.txt")
        missing_path.write_text(format_unit_id_ranges(missing) + "\n")


def _write_combined_skip_report(path: Path, results: dict[str, Any]) -> None:
    skipped: list[tuple[str, str, str]] = []
    for section, result in results.items():
        for file_name, reason in result.get("skipped_files", []):
            skipped.append((section, file_name, reason))
    if not skipped:
        return
    with path.open("w") as handle:
        handle.write(f"skipped files: {len(skipped)}\n\n")
        for section, file_name, reason in skipped:
            handle.write(f"[{section}] {file_name}\n  {reason}\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    data_dir = output_dir / "data"
    metadata_dir = output_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    paths, missing = expand_inputs(args.inputs, args.input_glob, skip_missing=args.skip_bad_files)
    if not paths:
        raise FileNotFoundError("No input catalogs were supplied or matched.")
    if missing:
        with (data_dir / "missing_input_files.txt").open("w") as handle:
            for path, reason in missing:
                handle.write(f"{path}\n  {reason}\n")

    command = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    (metadata_dir / "command.txt").write_text(command + "\n")
    _write_catalog_manifest(metadata_dir / "catalog_manifest.csv", paths, args.unit_realization_total)

    results: dict[str, Any] = {}
    if not args.no_f158:
        f158_config = F158DiagnosticsConfig(
            magnitude_group=args.magnitude_group,
            magnitude_limit=args.magnitude_limit,
            redshift_dataset=args.redshift_dataset,
            halo_mass_dataset=args.halo_mass_dataset,
            z_min=args.z_min,
            z_max=args.z_max,
            dz=args.dz,
            coarse_dz=args.coarse_dz,
            survey_area_deg2=args.survey_area_deg2,
            apparent_magnitude_min=args.apparent_magnitude_min,
            apparent_magnitude_max=args.apparent_magnitude_max,
            apparent_magnitude_bin_width=args.apparent_magnitude_bin_width,
            z_centers=tuple(args.z_centers),
            z_half_width=args.z_half_width,
            angular_weight_mode=args.angular_weight_mode,
            unit_realization_scale=args.unit_realization_scale,
            unit_realization_total=args.unit_realization_total,
            halo_scatter_max_points=args.halo_scatter_max_points,
            halo_sample_seed=args.halo_sample_seed,
            halo_summary_dz=args.halo_summary_dz,
            halo_mass_resolution=args.halo_mass_resolution,
            plot_individual_files=args.plot_individual_files,
            individual_file_alpha=args.individual_file_alpha,
            dpi=args.dpi,
            cosmos_web_catalog=args.cosmos_web_catalog,
            cosmos_web_area_deg2=args.cosmos_web_area_deg2,
            cosmos_web_redshift_column=args.cosmos_web_redshift_column,
            cosmos_web_magnitude_column=args.cosmos_web_magnitude_column,
            three_dhst_catalog=args.three_dhst_catalog,
            three_dhst_area_deg2=args.three_dhst_area_deg2,
            three_dhst_redshift_column=args.three_dhst_redshift_column,
            three_dhst_magnitude_column=args.three_dhst_magnitude_column,
            cosmos2020_catalog=args.cosmos2020_catalog,
            cosmos2020_area_deg2=args.cosmos2020_area_deg2,
            cosmos2020_redshift_column=args.cosmos2020_redshift_column,
            cosmos2020_magnitude_column=args.cosmos2020_magnitude_column,
            packaged_observation_reference=not args.no_packaged_observation_reference,
        )
        results["f158"] = run_f158_diagnostics(
            paths,
            output_dir=output_dir,
            data_dir=data_dir,
            config=f158_config,
            skip_bad_files=args.skip_bad_files,
        )
    if not args.no_emission_lines:
        results["emission_lines"] = run_emission_line_diagnostics(
            paths,
            output_dir=output_dir,
            data_dir=data_dir,
            unit_realization_scale=args.unit_realization_scale,
            unit_realization_total=args.unit_realization_total,
            redshift_dataset=args.redshift_dataset,
            angular_weight_mode=args.angular_weight_mode,
            include_agn=args.include_agn,
            skip_bad_files=args.skip_bad_files,
            min_log10_phi=args.min_log10_phi,
            halpha_target_hdf5=args.halpha_target_hdf5,
            point_redshift_half_width=args.point_redshift_half_width,
            dpi=args.dpi,
        )
    if not args.no_smhm:
        smhm_config = SMHMDiagnosticsConfig(
            halo_mass_dataset=args.halo_mass_dataset,
            disk_stellar_mass_dataset=args.smhm_disk_stellar_mass_dataset,
            spheroid_stellar_mass_dataset=args.smhm_spheroid_stellar_mass_dataset,
            redshift_dataset=args.redshift_dataset,
            z_centers=tuple(args.smhm_z_centers),
            z_half_width=args.smhm_z_half_width,
            output_redshift_tolerance=args.smhm_output_redshift_tolerance,
            log10_halo_mass_min=args.smhm_log10_halo_mass_min,
            log10_halo_mass_max=args.smhm_log10_halo_mass_max,
            log10_halo_mass_bin_width=args.smhm_log10_halo_mass_bin_width,
            angular_weight_mode=args.angular_weight_mode,
            unit_realization_scale=args.unit_realization_scale,
            unit_realization_total=args.unit_realization_total,
            galaxy_selection=args.smhm_selection,
            fill_missing_node_weights_with_median=args.smhm_fill_missing_node_weights_with_median,
            dpi=args.dpi,
        )
        results["smhm"] = run_smhm_diagnostics(
            paths,
            output_dir=output_dir,
            data_dir=data_dir,
            config=smhm_config,
            skip_bad_files=args.skip_bad_files,
        )

    config_dump = {
        "inputs": [str(path) for path in paths],
        "arguments": _json_safe(vars(args)),
        "results": _json_safe(results),
    }
    with (metadata_dir / "diagnostics_config.yaml").open("w") as handle:
        if yaml is not None:
            yaml.safe_dump(config_dump, handle, sort_keys=True)
        else:
            json.dump(config_dump, handle, indent=2, sort_keys=True)
            handle.write("\n")
    with (metadata_dir / "diagnostics_metadata.json").open("w") as handle:
        json.dump(config_dump, handle, indent=2, sort_keys=True)
    _write_combined_skip_report(data_dir / "skipped_lightcone_files.txt", results)
    print(output_dir)
    print(data_dir)
    print(metadata_dir)


if __name__ == "__main__":
    main()
