from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
import re
import warnings
from collections.abc import Iterable
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "galacticus_diagnostics_mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(__file__).with_name("romanUNIT-d1_100_0.2Deg.hdf5")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("emission_line_lfs")

DEG2_PER_STERADIAN = (180.0 / np.pi) ** 2

LINE_GROUPS = {
    "halpha_sobral": {
        "label": "Halpha Sobral",
        "lines": ["balmerAlpha6565"],
        "target": "halpha_sobral",
    },
    "hbeta_oiii_khostovan": {
        "label": "Hbeta+[OIII] Khostovan",
        "lines": ["balmerBeta4863", "oxygenIII4960", "oxygenIII5008"],
        "target": "khostovan_hbeta_oiii",
    },
    "oii_khostovan": {
        "label": "[OII] Khostovan",
        "lines": ["oxygenII3727", "oxygenII3730"],
        "target": "khostovan_oii",
    },
    "oii_comparat": {
        "label": "[OII] Comparat",
        "lines": ["oxygenII3727", "oxygenII3730"],
        "target": "comparat_oii",
    },
}

STANDARD_PLOT_NAMES = {
    "halpha_sobral": "emission_line_lfs_halpha.png",
    "hbeta_oiii_khostovan": "emission_line_lfs_oiii_hbeta.png",
    "oii_khostovan": "emission_line_lfs_oii_khostovan.png",
    "oii_comparat": "emission_line_lfs_oii_comparat.png",
}

COMPONENTS_NO_AGN = ["Disk", "Spheroid"]
COMPONENTS_WITH_AGN = ["Disk", "Spheroid", "AGN"]

# The Sobral centers are read from the same reduced Galacticus file used by the
# calibration plots when available. These fallback centers keep the script useful
# if that campaign artifact is not present.
HALPHA_FALLBACK_CASES = {
    "z1": {"target_redshift": 0.40, "z_min": 0.30, "z_max": 0.50, "centers": [40.1, 40.2, 40.3, 40.4, 40.5, 40.6, 40.7, 40.8, 40.9, 41.0, 41.1, 41.2, 41.3, 41.4, 41.5, 41.6, 41.8, 42.1]},
    "z2": {"target_redshift": 0.84, "z_min": 0.74, "z_max": 0.94, "centers": [41.3, 41.45, 41.6, 41.75, 41.9, 42.05, 42.2, 42.35, 42.5]},
    "z3": {"target_redshift": 1.47, "z_min": 1.37, "z_max": 1.57, "centers": [41.7, 41.8, 41.9, 42.0, 42.1, 42.2, 42.3, 42.4, 42.5, 42.6, 42.7, 42.8, 43.0]},
    "z4": {"target_redshift": 2.23, "z_min": 2.10, "z_max": 2.30, "centers": [41.6, 41.75, 41.9, 42.0, 42.1, 42.2, 42.3, 42.4, 42.5, 42.6, 42.7, 42.8, 42.9, 43.0, 43.2]},
}

HALPHA_TARGETS = {
    "z1": {
        "log10_phi": [-1.66, -1.70, -1.81, -1.93, -1.96, -2.03, -2.12, -2.27, -2.29, -2.42, -2.46, -2.57, -2.69, -2.73, -2.88, -3.03, -3.56, -3.71],
        "log10_phi_sigma": [0.041899948208964824, 0.04189994820896487, 0.04189994820896492, 0.05299194139370686, 0.07595704300860859, 0.0759570430086085, 0.10000445651846165, 0.08784232783962358, 0.10000445651846175, 0.11244987756673408, 0.12518518971141407, 0.1515526542986074, 0.23834802917049033, 0.20807412921501356, 0.2540158859886384, 0.5379697550971432, 0.971054913931064, 1.7930342085838926],
    },
    "z2": {
        "log10_phi": [-1.93, -2.02, -2.18, -2.43, -2.73, -3.01, -3.27, -3.79, -4.13],
        "log10_phi_sigma": [0.031060439614246805, 0.03106043961424687, 0.04189994820896488, 0.06434230029007926, 0.20807412921501356, 0.20807412921501364, 0.27004869407470694, 1.1066404885920722, 13.619199476439892],
    },
    "z3": {
        "log10_phi": [-2.13, -2.25, -2.34, -2.47, -2.62, -2.73, -2.91, -3.18, -3.55, -3.81, -4.22, -4.55, -4.86],
        "log10_phi_sigma": [0.11244987756673412, 0.10000445651846168, 0.06434230029007915, 0.05299194139370664, 0.05299194139370661, 0.04189994820896491, 0.08784232783962355, 0.12518518971141412, 0.22303681631519043, 0.3559915597056739, 0.6075054178347827, 1.1066404885920726, 1.1066404885920722],
    },
    "z4": {
        "log10_phi": [-1.93, -2.07, -2.19, -2.31, -2.41, -2.50, -2.59, -2.73, -2.88, -3.09, -3.33, -3.67, -4.01, -4.22, -4.63],
        "log10_phi_sigma": [0.23834802917049036, 0.19345203445520576, 0.07595704300860832, 0.05299194139370667, 0.052991941393706624, 0.04189994820896487, 0.052991941393706596, 0.06434230029007926, 0.1651987871198594, 0.20807412921501353, 0.2864549542357638, 0.45242104095921176, 0.9710549139310627, 1.6443695977824069, 0.6820142228606556],
    },
}

COMPARAT_OII_CASES = [
    (0.100, 0.240, [40.625, 40.875, 41.125, 41.375, 41.625, 41.875, 42.125]),
    (0.240, 0.400, [42.125, 42.375, 42.625]),
    (0.500, 0.695, [41.125, 41.375, 41.625, 41.875, 42.125, 42.375, 42.625, 42.875, 43.125]),
    (0.695, 0.880, [40.875, 41.125, 41.375, 41.625, 41.875, 42.125, 42.375, 42.625, 42.875, 43.125]),
    (0.880, 1.090, [41.125, 41.375, 41.625, 41.875, 42.125, 42.375, 42.625, 42.875, 43.125, 43.375]),
    (1.090, 1.340, [41.875, 42.125, 42.375, 42.625, 42.875, 43.125, 43.375]),
    (1.340, 1.650, [42.375, 42.625, 42.875, 43.125, 43.625]),
]

COMPARAT_OII_TARGET_ROWS = [
    (0.1, 0.24, 40.5, 40.75, 40.625, 0.009407974698846633, 0.004808445958441802),
    (0.1, 0.24, 40.75, 41.0, 40.875, 0.0054156867329375824, 0.006039321173723157),
    (0.1, 0.24, 41.0, 41.25, 41.125, 0.0104503094431532, 0.11646640501051132),
    (0.1, 0.24, 41.25, 41.5, 41.375, 0.0009127229870358616, 0.00031172936987260207),
    (0.1, 0.24, 41.5, 41.75, 41.625, 0.0003455237051126245, 0.00013835435494532076),
    (0.1, 0.24, 41.75, 42.0, 41.875, 5.394244896375752e-05, 1.424352411907424e-05),
    (0.1, 0.24, 42.0, 42.25, 42.125, 9.635864527236619e-06, 8.335239875435857e-06),
    (0.24, 0.4, 42.0, 42.25, 42.125, 5.966271733815524e-05, 2.847366115920076e-05),
    (0.24, 0.4, 42.25, 42.5, 42.375, 3.221279960579221e-06, 1.4087467457173704e-06),
    (0.24, 0.4, 42.5, 42.75, 42.625, 3.980285897323328e-07, 4.212320369518159e-07),
    (0.5, 0.695, 41.0, 41.25, 41.125, 0.0036056267446948303, 0.00023840519246212444),
    (0.5, 0.695, 41.25, 41.5, 41.375, 0.0020526541754825726, 9.701201580701669e-05),
    (0.5, 0.695, 41.5, 41.75, 41.625, 0.0007895832415677095, 5.122114716813447e-05),
    (0.5, 0.695, 41.75, 42.0, 41.875, 0.0003950307531201811, 5.421822851234774e-05),
    (0.5, 0.695, 42.0, 42.25, 42.125, 0.00013965072522332893, 2.1656214678148722e-05),
    (0.5, 0.695, 42.25, 42.5, 42.375, 5.106054780361592e-05, 1.4880249732651428e-05),
    (0.5, 0.695, 42.5, 42.75, 42.625, 7.908936747642984e-06, 2.2822820039030995e-06),
    (0.5, 0.695, 42.75, 43.0, 42.875, 1.9471490407950797e-06, 9.531340972401034e-07),
    (0.5, 0.695, 43.0, 43.25, 43.125, 1.4755662802217956e-06, 1.91964456376003e-06),
    (0.695, 0.88, 40.75, 41.0, 40.875, 0.008919037630916043, 0.0006308891438848719),
    (0.695, 0.88, 41.0, 41.25, 41.125, 0.007599046590187872, 0.0005210296973017121),
    (0.695, 0.88, 41.25, 41.5, 41.375, 0.0029098767083512284, 0.00021333622006553383),
    (0.695, 0.88, 41.5, 41.75, 41.625, 0.0014690040915166657, 0.00017361033496895645),
    (0.695, 0.88, 41.75, 42.0, 41.875, 0.00048168663599647494, 4.712435665195147e-05),
    (0.695, 0.88, 42.0, 42.25, 42.125, 0.0001773568622565417, 1.3707509422488111e-05),
    (0.695, 0.88, 42.25, 42.5, 42.375, 6.535327721838865e-05, 9.118097074782572e-06),
    (0.695, 0.88, 42.5, 42.75, 42.625, 2.409070815063799e-05, 4.8971675841526705e-06),
    (0.695, 0.88, 42.75, 43.0, 42.875, 2.9118891804606997e-06, 1.1893688348103816e-06),
    (0.695, 0.88, 43.0, 43.25, 43.125, 3.577705478613474e-07, 2.5176486770724477e-07),
    (0.88, 1.09, 41.0, 41.25, 41.125, 0.008725507348748667, 0.0007209279625227571),
    (0.88, 1.09, 41.25, 41.5, 41.375, 0.004603406756484013, 0.00034051608433914495),
    (0.88, 1.09, 41.5, 41.75, 41.625, 0.0017964211526011857, 0.00015398744979432),
    (0.88, 1.09, 41.75, 42.0, 41.875, 0.0004769980507983302, 4.5168165145017085e-05),
    (0.88, 1.09, 42.0, 42.25, 42.125, 0.00017316075018608667, 1.73695937259821e-05),
    (0.88, 1.09, 42.25, 42.5, 42.375, 9.916642950277742e-05, 1.2708012410219177e-05),
    (0.88, 1.09, 42.5, 42.75, 42.625, 3.074308123920519e-05, 4.757049507638488e-06),
    (0.88, 1.09, 42.75, 43.0, 42.875, 1.059026506893828e-05, 4.533473998374513e-06),
    (0.88, 1.09, 43.0, 43.25, 43.125, 7.154548638410591e-07, 4.788864645808512e-07),
    (0.88, 1.09, 43.25, 43.5, 43.375, 4.9148361056608294e-08, 7.059698575489387e-08),
    (1.09, 1.34, 41.75, 42.0, 41.875, 0.001347451930016421, 0.0001408495330874631),
    (1.09, 1.34, 42.0, 42.25, 42.125, 0.0006653544074621484, 8.666016402910637e-05),
    (1.09, 1.34, 42.25, 42.5, 42.375, 0.00020109246216303895, 2.5967623259473417e-05),
    (1.09, 1.34, 42.5, 42.75, 42.625, 8.516702801735969e-05, 1.2358751149972428e-05),
    (1.09, 1.34, 42.75, 43.0, 42.875, 2.9772183656133103e-05, 6.580399299569928e-06),
    (1.09, 1.34, 43.0, 43.25, 43.125, 2.153318544971774e-06, 8.310987650717356e-07),
    (1.09, 1.34, 43.25, 43.5, 43.375, 3.73684278115385e-08, 7.844759458910501e-08),
    (1.34, 1.65, 42.25, 42.5, 42.375, 0.00037131787936794014, 4.9278714041900814e-05),
    (1.34, 1.65, 42.5, 42.75, 42.625, 0.0003262626351883699, 5.938207860053729e-05),
    (1.34, 1.65, 42.75, 43.0, 42.875, 0.00010472801033051467, 3.017867889728435e-05),
    (1.34, 1.65, 43.0, 43.25, 43.125, 1.4892156738475339e-05, 7.911448538167599e-06),
    (1.34, 1.65, 43.5, 43.75, 43.625, 2.0310398953024006e-07, 2.644327618044589e-07),
]


def finite_limits_from_values(
    *values: np.ndarray | Iterable[float] | None,
    margin_fraction: float = 0.08,
    min_margin: float = 0.15,
) -> tuple[float, float] | None:
    finite_arrays = []
    for value in values:
        if value is None:
            continue
        array = np.asarray(value, dtype=float).ravel()
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_arrays.append(finite)
    if not finite_arrays:
        return None
    finite = np.concatenate(finite_arrays)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper <= lower:
        lower -= min_margin
        upper += min_margin
    else:
        margin = max(margin_fraction * (upper - lower), min_margin)
        lower -= margin
        upper += margin
    return lower, upper


def set_ylim_from_values(
    axis,
    *values: np.ndarray | Iterable[float] | None,
    margin_fraction: float = 0.08,
    min_margin: float = 0.15,
) -> tuple[float, float] | None:
    limits = finite_limits_from_values(*values, margin_fraction=margin_fraction, min_margin=min_margin)
    if limits is not None:
        axis.set_ylim(*limits)
    return limits


def _lf_dataframe(table: list[tuple[float, float, float, int, float, float, float, float]]) -> pd.DataFrame:
    data = pd.DataFrame(
        table,
        columns=[
            "redshift",
            "log10_luminosity_erg_s",
            "half_width_dex",
            "n_emitters",
            "log10_phi_observed",
            "log10_phi_final",
            "log10_phi_final_error",
            "volume_1e5_mpc3",
        ],
    )
    data["z_min"] = data["redshift"]
    data["z_max"] = data["redshift"]
    data["log10_luminosity_min"] = data["log10_luminosity_erg_s"] - data["half_width_dex"]
    data["log10_luminosity_max"] = data["log10_luminosity_erg_s"] + data["half_width_dex"]
    data["phi_mpc3_dex"] = 10.0 ** data["log10_phi_final"]
    data["phi_lower_mpc3_dex"] = 10.0 ** (data["log10_phi_final"] - data["log10_phi_final_error"])
    data["phi_upper_mpc3_dex"] = 10.0 ** (data["log10_phi_final"] + data["log10_phi_final_error"])
    data["phi_err_lower_mpc3_dex"] = data["phi_mpc3_dex"] - data["phi_lower_mpc3_dex"]
    data["phi_err_upper_mpc3_dex"] = data["phi_upper_mpc3_dex"] - data["phi_mpc3_dex"]
    return data


def _load_khostovan_lf(line_set: str) -> pd.DataFrame:
    if line_set == "hbeta_oiii":
        return _lf_dataframe(
            [
                (0.84, 41.10, 0.10, 703, -1.97, -1.82, 0.02, 3.25),
                (0.84, 41.30, 0.10, 465, -2.15, -2.04, 0.03, 3.25),
                (0.84, 41.50, 0.10, 262, -2.39, -2.35, 0.04, 3.25),
                (0.84, 41.70, 0.10, 128, -2.71, -2.61, 0.06, 3.25),
                (0.84, 41.90, 0.10, 68, -2.98, -2.94, 0.08, 3.25),
                (0.84, 42.10, 0.10, 28, -3.37, -3.17, 0.13, 3.25),
                (0.84, 42.30, 0.10, 12, -3.73, -3.52, 0.20, 3.25),
                (0.84, 42.50, 0.10, 3, -4.34, -4.12, 0.39, 3.25),
                (1.42, 41.95, 0.15, 284, -2.63, -2.49, 0.03, 4.06),
                (1.42, 42.25, 0.15, 73, -3.22, -3.14, 0.07, 4.06),
                (1.42, 42.55, 0.15, 12, -4.01, -3.89, 0.19, 4.06),
                (1.42, 42.85, 0.15, 2, -4.78, -4.64, 0.48, 4.06),
                (2.23, 42.60, 0.075, 84, -3.27, -3.08, 0.06, 10.46),
                (2.23, 42.75, 0.075, 70, -3.36, -3.14, 0.07, 10.69),
                (2.23, 42.90, 0.075, 22, -3.86, -3.65, 0.13, 10.69),
                (2.23, 43.05, 0.075, 5, -4.51, -4.26, 0.29, 10.69),
                (3.24, 42.65, 0.075, 70, -3.33, -3.17, 0.07, 9.99),
                (3.24, 42.80, 0.075, 52, -3.48, -3.26, 0.09, 10.48),
                (3.24, 42.95, 0.075, 25, -3.80, -3.55, 0.13, 10.48),
                (3.24, 43.10, 0.075, 6, -4.42, -4.17, 0.27, 10.48),
            ]
        )
    if line_set == "oii":
        return _lf_dataframe(
            [
                (1.47, 41.65, 0.075, 590, -2.24, -2.08, 0.02, 6.80),
                (1.47, 41.80, 0.075, 425, -2.38, -2.28, 0.03, 6.80),
                (1.47, 41.95, 0.075, 257, -2.60, -2.46, 0.04, 6.80),
                (1.47, 42.10, 0.075, 127, -2.90, -2.69, 0.06, 6.80),
                (1.47, 42.25, 0.075, 42, -3.39, -3.05, 0.10, 6.80),
                (1.47, 42.40, 0.075, 19, -3.73, -3.55, 0.15, 6.80),
                (1.47, 42.55, 0.075, 6, -4.23, -4.23, 0.28, 6.80),
                (2.25, 42.45, 0.10, 92, -3.14, -2.77, 0.05, 6.29),
                (2.25, 42.65, 0.10, 37, -3.53, -3.15, 0.08, 6.29),
                (2.25, 42.85, 0.10, 3, -4.62, -4.46, 0.35, 6.29),
                (3.34, 43.05, 0.050, 12, -4.12, -3.86, 0.17, 15.88),
                (3.34, 43.15, 0.075, 7, -4.37, -3.92, 0.24, 16.52),
                (3.34, 43.30, 0.075, 2, -5.22, -4.87, 0.48, 16.52),
            ]
        )
    raise ValueError(f"Unknown line_set={line_set!r}")


def _load_comparat_lf() -> pd.DataFrame:
    data = pd.DataFrame(
        COMPARAT_OII_TARGET_ROWS,
        columns=["z_min", "z_max", "logL_min", "logL_max", "logL_plot", "phi", "e_phi"],
    )
    data["z_label"] = data["z_min"].map(lambda value: f"{value:.3f}") + "-" + data["z_max"].map(
        lambda value: f"{value:.3f}"
    )
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate dust-attenuated emission-line luminosity functions from "
            "Galacticus UNIT lightcone HDF5 files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[],
        help="Input HDF5 files. Shell globs may also be passed via --input-glob.",
    )
    parser.add_argument("--input-glob", action="append", default=[], help="Additional quoted glob of HDF5 files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--unit-realization-scale",
        default="auto",
        help=(
            "Density boost for partial UNIT coverage. Use 'auto' to infer "
            "10000 / sum(d_end-d_start+1) from romanUNIT-dSTART_END filenames, "
            "or pass a numeric value."
        ),
    )
    parser.add_argument("--unit-realization-total", type=int, default=10000)
    parser.add_argument("--redshift-dataset", default="lightconeRedshiftObserved")
    parser.add_argument(
        "--angular-weight-mode",
        choices=["area", "surface-density"],
        default="area",
        help=(
            "Interpret nodeData/angularWeight. 'area' treats it as the effective sky area in deg^2 "
            "represented by one object and weights by 1/angularWeight. 'surface-density' uses the "
            "stored values directly as deg^-2 weights."
        ),
    )
    parser.add_argument("--include-agn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--skip-bad-files",
        action="store_true",
        help=(
            "Skip unreadable/incomplete HDF5 files after printing a warning and writing "
            "a skipped_lightcone_files.txt report. UNIT scale is inferred from usable files only."
        ),
    )
    parser.add_argument("--min-log10-phi", type=float, default=-8.0)
    parser.add_argument(
        "--halpha-target-hdf5",
        type=Path,
        default=None,
        help="Optional local HDF5 file with /analyses Sobral H-alpha target data; defaults to built-in targets.",
    )
    parser.add_argument("--point-redshift-half-width", type=float, default=0.10)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _expand_inputs(inputs: list[Path], globs: list[str], *, skip_missing: bool) -> tuple[list[Path], list[tuple[Path, str]]]:
    if not inputs and not globs and DEFAULT_INPUT.exists():
        inputs = [DEFAULT_INPUT]
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


def _covered_unit_ids(paths: list[Path]) -> set[int]:
    covered: set[int] = set()
    for path in paths:
        match = re.search(r"-d(\d+)_(\d+)_", path.name)
        if match is None:
            continue
        start, end = (int(match.group(1)), int(match.group(2)))
        covered.update(range(start, end + 1))
    return covered


def _format_unit_id_ranges(values: Iterable[int], *, limit: int = 20) -> str:
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


def _infer_unit_scale(paths: list[Path], total_units: int) -> float:
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


def _resolve_unit_scale(value: str, paths: list[Path], total_units: int) -> float:
    if value.strip().lower() == "auto":
        return _infer_unit_scale(paths, total_units)
    scale = float(value)
    if scale <= 0.0:
        raise ValueError("--unit-realization-scale must be positive")
    return scale


def _cosmology_from_file(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        group = handle["/Parameters/cosmologyParameters"]
        return {
            "H0": float(group.attrs["HubbleConstant"]),
            "OmegaMatter": float(group.attrs["OmegaMatter"]),
            "OmegaDarkEnergy": float(group.attrs["OmegaDarkEnergy"]),
        }


def _comoving_distance_mpc(z: np.ndarray, cosmology: dict[str, float]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    flat = z.ravel()
    z_max = float(np.max(flat))
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


def _volume_per_square_degree(z_min: float, z_max: float, cosmology: dict[str, float]) -> float:
    chi_min, chi_max = _comoving_distance_mpc(np.asarray([z_min, z_max]), cosmology)
    return (chi_max**3 - chi_min**3) / (3.0 * DEG2_PER_STERADIAN)


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def _centers_half_widths_to_edges(centers: np.ndarray, half_widths: np.ndarray) -> np.ndarray:
    return np.round(np.concatenate([[centers[0] - half_widths[0]], centers + half_widths]), decimals=6)


def _built_in_halpha_target_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for sample_label, metadata in HALPHA_FALLBACK_CASES.items():
        centers = np.asarray(metadata["centers"], dtype=float)
        target = HALPHA_TARGETS[sample_label]
        cases.append(
            {
                "observable": "halpha_sobral",
                "sample_label": sample_label,
                "target_redshift": metadata["target_redshift"],
                "z_min": metadata["z_min"],
                "z_max": metadata["z_max"],
                "centers": centers,
                "edges": _centers_to_edges(centers),
                "target": pd.DataFrame(
                    {
                        "x": centers,
                        "log10_phi": np.asarray(target["log10_phi"], dtype=float),
                        "log10_phi_sigma": np.asarray(target["log10_phi_sigma"], dtype=float),
                    }
                ),
            }
        )
    return cases


def _halpha_target_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return _built_in_halpha_target_cases()

    cases: list[dict[str, Any]] = []
    if path.exists():
        with h5py.File(path, "r") as handle:
            for sample_label, metadata in HALPHA_FALLBACK_CASES.items():
                analysis = f"luminosityFunctionHalphaSobral2013HiZELS{sample_label.upper()}"
                group_path = f"/analyses/{analysis}"
                if group_path not in handle:
                    continue
                group = handle[group_path]
                centers = np.log10(np.asarray(group["luminosity"][...], dtype=float))
                target_phi = np.asarray(group["luminosityFunctionTarget"][...], dtype=float) * np.log(10.0)
                covariance = np.asarray(group["luminosityFunctionCovarianceTarget"][...], dtype=float)
                target_sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None)) * np.log(10.0)
                cases.append(
                    {
                        "observable": "halpha_sobral",
                        "sample_label": sample_label,
                        "target_redshift": metadata["target_redshift"],
                        "z_min": metadata["z_min"],
                        "z_max": metadata["z_max"],
                        "centers": centers,
                        "edges": _centers_to_edges(centers),
                        "target": pd.DataFrame(
                            {
                                "x": centers,
                                "log10_phi": np.log10(target_phi),
                                "log10_phi_sigma": target_sigma / (target_phi * np.log(10.0)),
                            }
                        ),
                    }
                )
    if cases:
        return cases

    warnings.warn(f"No Sobral target HDF5 found at {path}; using built-in H-alpha targets.", stacklevel=2)
    return _built_in_halpha_target_cases()


def _khostovan_cases(line_set: str, observable: str, half_width_default: float) -> list[dict[str, Any]]:
    target = _load_khostovan_lf(line_set)
    cases: list[dict[str, Any]] = []
    for redshift, group in target.groupby("redshift", sort=True):
        group = group.sort_values("log10_luminosity_erg_s")
        centers = group["log10_luminosity_erg_s"].to_numpy(dtype=float)
        half_widths = group["half_width_dex"].to_numpy(dtype=float)
        z = float(redshift)
        z_min = max(0.0, z - half_width_default)
        z_max = z + half_width_default
        cases.append(
            {
                "observable": observable,
                "sample_label": f"z{z:.2f}",
                "target_redshift": z,
                "z_min": z_min,
                "z_max": z_max,
                "centers": centers,
                "edges": _centers_half_widths_to_edges(centers, half_widths),
                "target": pd.DataFrame(
                    {
                        "x": centers,
                        "xerr": half_widths,
                        "log10_phi": group["log10_phi_final"].to_numpy(dtype=float),
                        "log10_phi_sigma": group["log10_phi_final_error"].to_numpy(dtype=float),
                    }
                ),
            }
        )
    return cases


def _comparat_cases() -> list[dict[str, Any]]:
    target = _load_comparat_lf()
    cases: list[dict[str, Any]] = []
    for z_min, z_max, centers_text in COMPARAT_OII_CASES:
        centers = np.asarray(centers_text, dtype=float)
        target_rows = target.loc[np.isclose(target["z_min"], z_min) & np.isclose(target["z_max"], z_max)].copy()
        if not target_rows.empty:
            target_rows = target_rows.sort_values("logL_plot")
            phi = target_rows["phi"].to_numpy(dtype=float)
            sigma = target_rows["e_phi"].to_numpy(dtype=float)
            target_frame = pd.DataFrame(
                {
                    "x": target_rows["logL_plot"].to_numpy(dtype=float),
                    "xerr": 0.5
                    * (
                        target_rows["logL_max"].to_numpy(dtype=float)
                        - target_rows["logL_min"].to_numpy(dtype=float)
                    ),
                    "log10_phi": np.log10(phi),
                    "log10_phi_sigma": sigma / (phi * np.log(10.0)),
                }
            )
        else:
            target_frame = pd.DataFrame()
        cases.append(
            {
                "observable": "oii_comparat",
                "sample_label": f"z{z_min:.3f}_{z_max:.3f}",
                "target_redshift": 0.5 * (z_min + z_max),
                "z_min": z_min,
                "z_max": z_max,
                "centers": centers,
                "edges": _centers_to_edges(centers),
                "target": target_frame,
            }
        )
    return cases


def _case_definitions(args: argparse.Namespace) -> list[dict[str, Any]]:
    return (
        _halpha_target_cases(args.halpha_target_hdf5.expanduser().resolve() if args.halpha_target_hdf5 else None)
        + _khostovan_cases("hbeta_oiii", "hbeta_oiii_khostovan", args.point_redshift_half_width)
        + _khostovan_cases("oii", "oii_khostovan", args.point_redshift_half_width)
        + _comparat_cases()
    )


def _lightcone_output_names(handle: h5py.File) -> list[str]:
    return sorted(handle["/Lightcone"].keys(), key=lambda name: int(re.sub(r"\D", "", name) or 0))


def _line_luminosity(output_group: h5py.Group, observable: str, include_agn: bool) -> np.ndarray:
    data = output_group["dustAttenuatedNodeData"]
    components = COMPONENTS_WITH_AGN if include_agn else COMPONENTS_NO_AGN
    total = None
    for component in components:
        for line in LINE_GROUPS[observable]["lines"]:
            dataset = f"luminosityEmissionLine{component}:{line}"
            if dataset not in data:
                if component == "AGN" and include_agn:
                    continue
                raise KeyError(f"{output_group.name}/dustAttenuatedNodeData/{dataset} is missing")
            values = np.asarray(data[dataset][...], dtype=float)
            total = values.copy() if total is None else total + values
    if total is None:
        raise RuntimeError(f"No luminosity datasets were summed for {observable}")
    return total


def _validate_lightcone_file(
    path: Path,
    cases: list[dict[str, Any]],
    *,
    redshift_dataset: str,
    include_agn: bool,
) -> None:
    observables = sorted({str(case["observable"]) for case in cases})
    components = COMPONENTS_WITH_AGN if include_agn else COMPONENTS_NO_AGN
    with h5py.File(path, "r") as handle:
        output_names = _lightcone_output_names(handle)
        if not output_names:
            raise RuntimeError("no /Lightcone outputs")
        for output_name in output_names:
            output_group = handle[f"/Lightcone/{output_name}"]
            nd = output_group["nodeData"]
            _ = nd[redshift_dataset].shape
            _ = nd["angularWeight"].shape
            data = output_group["dustAttenuatedNodeData"]
            for observable in observables:
                for component in components:
                    for line in LINE_GROUPS[observable]["lines"]:
                        dataset = f"luminosityEmissionLine{component}:{line}"
                        if dataset not in data:
                            if component == "AGN" and include_agn:
                                continue
                            raise KeyError(f"{output_group.name}/dustAttenuatedNodeData/{dataset} is missing")
                        _ = data[dataset].shape


def _filter_usable_paths(
    paths: list[Path],
    cases: list[dict[str, Any]],
    *,
    redshift_dataset: str,
    include_agn: bool,
    skip_bad_files: bool,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    usable: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for path in paths:
        try:
            _validate_lightcone_file(
                path,
                cases,
                redshift_dataset=redshift_dataset,
                include_agn=include_agn,
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


def _write_skip_report(
    output_dir: Path,
    skipped: list[tuple[Path, str]],
    paths: list[Path],
    total_units: int,
) -> None:
    covered = _covered_unit_ids(paths)
    missing_ids = sorted(set(range(1, total_units + 1)) - covered) if covered else []
    if not skipped and not missing_ids:
        return
    report_path = output_dir / "skipped_lightcone_files.txt"
    with report_path.open("w") as handle:
        handle.write(f"usable files: {len(paths)}\n")
        handle.write(f"skipped files: {len(skipped)}\n")
        if covered:
            handle.write(f"covered UNIT ids: {len(covered)} / {total_units}\n")
            handle.write(f"missing UNIT ids: {_format_unit_id_ranges(missing_ids)}\n")
        handle.write("\nSKIPPED FILES\n")
        for path, reason in skipped:
            handle.write(f"{path}\n  {reason}\n")
    print(f"Skip/missing-file report: {report_path}", flush=True)


def _accumulate_histograms(
    paths: list[Path],
    cases: list[dict[str, Any]],
    *,
    unit_scale: float,
    redshift_dataset: str,
    angular_weight_mode: str,
    include_agn: bool,
) -> dict[int, dict[str, Any]]:
    accumulators = {
        index: {
            "weighted_counts": np.zeros(len(case["centers"]), dtype=float),
            "weighted_counts_variance": np.zeros(len(case["centers"]), dtype=float),
            "raw_rows": 0,
        }
        for index, case in enumerate(cases)
    }
    for file_index, path in enumerate(paths, start=1):
        try:
            with h5py.File(path, "r") as handle:
                print(f"[{file_index}/{len(paths)}] {path}", flush=True)
                for output_name in _lightcone_output_names(handle):
                    output_group = handle[f"/Lightcone/{output_name}"]
                    nd = output_group["nodeData"]
                    redshift = np.asarray(nd[redshift_dataset][...], dtype=float)
                    angular_weight = np.asarray(nd["angularWeight"][...], dtype=float)
                    if angular_weight_mode == "area":
                        if np.any(angular_weight <= 0.0):
                            raise ValueError(f"{output_group.name}/nodeData/angularWeight contains non-positive areas")
                        weights = unit_scale / angular_weight
                    else:
                        weights = angular_weight * unit_scale
                    if "nodeSubsamplingWeight" in nd:
                        weights = weights * np.asarray(nd["nodeSubsamplingWeight"][...], dtype=float)
                    luminosity_cache: dict[str, np.ndarray] = {}
                    for index, case in enumerate(cases):
                        selection = (
                            np.isfinite(redshift)
                            & (redshift >= float(case["z_min"]))
                            & (redshift < float(case["z_max"]))
                        )
                        if not np.any(selection):
                            continue
                        observable = str(case["observable"])
                        if observable not in luminosity_cache:
                            luminosity_cache[observable] = _line_luminosity(output_group, observable, include_agn)
                        luminosity = luminosity_cache[observable]
                        valid = selection & np.isfinite(luminosity) & np.isfinite(weights) & (luminosity > 0.0)
                        if not np.any(valid):
                            continue
                        log_luminosity = np.log10(luminosity[valid])
                        w = weights[valid]
                        counts, _ = np.histogram(log_luminosity, bins=case["edges"], weights=w)
                        variance, _ = np.histogram(log_luminosity, bins=case["edges"], weights=w**2)
                        accumulators[index]["weighted_counts"] += counts
                        accumulators[index]["weighted_counts_variance"] += variance
                        accumulators[index]["raw_rows"] += int(np.count_nonzero(valid))
        except Exception as exc:
            raise RuntimeError(f"Failed while reading {path}") from exc
    return accumulators


def _rows_from_accumulators(
    cases: list[dict[str, Any]],
    accumulators: dict[int, dict[str, Any]],
    cosmology: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        acc = accumulators[index]
        centers = np.asarray(case["centers"], dtype=float)
        edges = np.asarray(case["edges"], dtype=float)
        dlog10 = np.diff(edges)
        volume_per_deg2 = _volume_per_square_degree(float(case["z_min"]), float(case["z_max"]), cosmology)
        phi = acc["weighted_counts"] / (volume_per_deg2 * dlog10)
        phi_std = np.sqrt(np.clip(acc["weighted_counts_variance"], 0.0, None)) / (volume_per_deg2 * dlog10)
        for bin_index, center in enumerate(centers):
            value = float(phi[bin_index])
            sigma = float(phi_std[bin_index])
            rows.append(
                {
                    "observable": case["observable"],
                    "observable_label": LINE_GROUPS[str(case["observable"])]["label"],
                    "sample_label": case["sample_label"],
                    "target_redshift": float(case["target_redshift"]),
                    "z_min": float(case["z_min"]),
                    "z_max": float(case["z_max"]),
                    "volume_mpc3_per_deg2": float(volume_per_deg2),
                    "bin_index": int(bin_index),
                    "log10_luminosity_min": float(edges[bin_index]),
                    "log10_luminosity_center": float(center),
                    "log10_luminosity_max": float(edges[bin_index + 1]),
                    "weighted_surface_density_deg2": float(acc["weighted_counts"][bin_index]),
                    "raw_rows_in_redshift_window": int(acc["raw_rows"]),
                    "phi_mpc3_dex": value,
                    "phi_mpc3_dex_shot_noise_std": sigma,
                    "log10_phi": np.log10(value) if value > 0.0 else np.nan,
                    "log10_phi_shot_noise_std": sigma / (value * np.log(10.0)) if value > 0.0 else np.nan,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_observable(
    frame: pd.DataFrame,
    cases: list[dict[str, Any]],
    observable: str,
    output_path: Path,
    *,
    min_log10_phi: float,
    dpi: int,
) -> None:
    observable_cases = [case for case in cases if case["observable"] == observable]
    if not observable_cases:
        return
    ncols = 2 if len(observable_cases) > 1 else 1
    nrows = int(np.ceil(len(observable_cases) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.8 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for axis, case in zip(axes_flat, observable_cases, strict=False):
        rows = frame.loc[
            (frame["observable"] == observable)
            & (frame["sample_label"].astype(str) == str(case["sample_label"]))
        ].sort_values("bin_index")
        x = rows["log10_luminosity_center"].to_numpy(dtype=float)
        phi = rows["phi_mpc3_dex"].to_numpy(dtype=float)
        y = np.full_like(phi, np.nan, dtype=float)
        positive = phi > 0.0
        y[positive] = np.log10(np.maximum(phi[positive], 10.0**min_log10_phi))
        sigma = rows["log10_phi_shot_noise_std"].to_numpy(dtype=float)
        axis.plot(x, y, color="#1f77b4", lw=2.0, marker="s", ms=3.5, label="UNIT lightcone")
        finite_sigma = np.isfinite(sigma) & positive
        if np.any(finite_sigma):
            axis.fill_between(
                x[finite_sigma],
                y[finite_sigma] - sigma[finite_sigma],
                y[finite_sigma] + sigma[finite_sigma],
                color="#1f77b4",
                alpha=0.18,
                linewidth=0.0,
            )
        if not np.any(positive):
            axis.text(
                0.5,
                0.12,
                "no UNIT counts in plotted luminosity bins",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=8,
                color="0.35",
            )

        y_values = [y[np.isfinite(y)]]
        target = case["target"]
        if isinstance(target, pd.DataFrame) and not target.empty:
            xerr = target["xerr"].to_numpy(dtype=float) if "xerr" in target.columns else None
            yerr = target["log10_phi_sigma"].to_numpy(dtype=float)
            axis.errorbar(
                target["x"].to_numpy(dtype=float),
                target["log10_phi"].to_numpy(dtype=float),
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                color="0.12",
                ecolor="0.35",
                capsize=2.0,
                ms=4.0,
                label="target",
                zorder=5,
            )
            y_values.append(target["log10_phi"].to_numpy(dtype=float))

        set_ylim_from_values(axis, *y_values)
        axis.set_title(
            f"{LINE_GROUPS[observable]['label']} {case['sample_label']} "
            f"({case['z_min']:.2f}<z<{case['z_max']:.2f})"
        )
        axis.set_xlabel(r"$\log_{10}(L/\mathrm{erg}\ \mathrm{s}^{-1})$")
        axis.set_ylabel(r"$\log_{10}\Phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)

    for axis in axes_flat[len(observable_cases) :]:
        axis.axis("off")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def run_emission_line_diagnostics(
    paths: list[Path],
    *,
    output_dir: Path,
    data_dir: Path,
    unit_realization_scale: str = "auto",
    unit_realization_total: int = 10000,
    redshift_dataset: str = "lightconeRedshiftObserved",
    angular_weight_mode: str = "area",
    include_agn: bool = False,
    skip_bad_files: bool = False,
    min_log10_phi: float = -8.0,
    halpha_target_hdf5: Path | None = None,
    point_redshift_half_width: float = 0.10,
    dpi: int = 220,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        halpha_target_hdf5=halpha_target_hdf5,
        point_redshift_half_width=point_redshift_half_width,
    )
    cases = _case_definitions(args)
    usable_paths, skipped = _filter_usable_paths(
        paths,
        cases,
        redshift_dataset=redshift_dataset,
        include_agn=include_agn,
        skip_bad_files=skip_bad_files,
    )
    if skipped:
        _write_skip_report(data_dir, skipped, usable_paths, unit_realization_total)
    unit_scale = _resolve_unit_scale(unit_realization_scale, usable_paths, unit_realization_total)
    cosmology = _cosmology_from_file(usable_paths[0])
    if skipped:
        print(f"WARNING: skipped {len(skipped)} bad/missing lightcone file(s)")
    print(f"Reading {len(usable_paths)} usable lightcone file(s) for emission-line LFs")
    print(f"Using UNIT realization scale = {unit_scale:.6g}")
    accumulators = _accumulate_histograms(
        usable_paths,
        cases,
        unit_scale=unit_scale,
        redshift_dataset=redshift_dataset,
        angular_weight_mode=angular_weight_mode,
        include_agn=include_agn,
    )
    rows = _rows_from_accumulators(cases, accumulators, cosmology)
    csv_path = data_dir / "emission_line_lfs.csv"
    _write_csv(csv_path, rows)
    frame = pd.DataFrame(rows)
    plot_outputs: list[str] = []
    for observable in LINE_GROUPS:
        filename = STANDARD_PLOT_NAMES.get(observable, f"emission_line_lfs_{observable}.png")
        path = output_dir / filename
        _plot_observable(frame, cases, observable, path, min_log10_phi=min_log10_phi, dpi=dpi)
        plot_outputs.append(filename)
        print(path)
    print(csv_path)
    return {
        "usable_files": len(usable_paths),
        "skipped_files": [(str(path), reason) for path, reason in skipped],
        "unit_realization_scale": unit_scale,
        "include_agn": include_agn,
        "outputs": {
            "plots": plot_outputs,
            "data": ["emission_line_lfs.csv"],
        },
    }


def main() -> None:
    args = parse_args()
    paths, skipped = _expand_inputs(args.inputs, args.input_glob, skip_missing=args.skip_bad_files)
    cases = _case_definitions(args)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, bad_files = _filter_usable_paths(
        paths,
        cases,
        redshift_dataset=args.redshift_dataset,
        include_agn=args.include_agn,
        skip_bad_files=args.skip_bad_files,
    )
    skipped.extend(bad_files)
    _write_skip_report(output_dir, skipped, paths, args.unit_realization_total)

    unit_scale = _resolve_unit_scale(args.unit_realization_scale, paths, args.unit_realization_total)
    cosmology = _cosmology_from_file(paths[0])

    if skipped:
        print(f"WARNING: skipped {len(skipped)} bad/missing lightcone file(s)")
    print(f"Reading {len(paths)} usable lightcone file(s)")
    print(f"Using UNIT realization scale = {unit_scale:.6g}")
    print(
        "Cosmology: "
        f"H0={cosmology['H0']:.4g}, "
        f"OmegaMatter={cosmology['OmegaMatter']:.4g}, "
        f"OmegaDarkEnergy={cosmology['OmegaDarkEnergy']:.4g}"
    )
    accumulators = _accumulate_histograms(
        paths,
        cases,
        unit_scale=unit_scale,
        redshift_dataset=args.redshift_dataset,
        angular_weight_mode=args.angular_weight_mode,
        include_agn=args.include_agn,
    )
    rows = _rows_from_accumulators(cases, accumulators, cosmology)
    csv_path = output_dir / "unit_lightcone_emission_line_lfs.csv"
    _write_csv(csv_path, rows)

    frame = pd.DataFrame(rows)
    frame.attrs["unit_realization_scale"] = unit_scale
    for observable in LINE_GROUPS:
        path = output_dir / f"unit_lightcone_{observable}.png"
        _plot_observable(frame, cases, observable, path, min_log10_phi=args.min_log10_phi, dpi=args.dpi)
        print(path)
    print(csv_path)


if __name__ == "__main__":
    main()
