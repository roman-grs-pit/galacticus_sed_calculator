"""Evaluate diffuse disk-dust transfer, with optional lightweight diagnostics.

Run from the repository root after installing the package:
    python examples/example_benson_diffuse.py /path/to/atlas.hdf5
    python examples/example_benson_diffuse.py /path/to/atlas.hdf5 --benchmark 100 --check-interpolation

This example does not require SSP templates, a galaxy catalog, or network access.
"""

import argparse
import json
import resource
import sys
import time

import h5py
import numpy as np

from galacticus_sed_calculator.diffuse_dust import (
    Benson2018DiffuseProvider,
    verify_benson_atlas,
)


def held_out_checks(provider, path):
    """Omit a few native inclination/size nodes and predict from neighbours.

    These checks probe coordinate choices at doubled native spacing. They
    compare noisy table entries, not predictions against independent RT truth.
    Inclination tests include 86 and 88 degrees; size tests include compact
    spheroids. No optical-depth or wavelength nodes are withheld here.
    """
    with h5py.File(path, "r") as f:
        wave = f["wavelength"][:]
        tau = f["opticalDepth"][:]
        inc = f["inclination"][:]
        ratio = f["spheroidScaleRadial"][:]
    wave = wave[[np.argmin(abs(wave - v)) for v in (0.15, 0.55, 2.0)]]
    tau = tau[[np.argmin(abs(tau - v)) for v in (0.1, 1., 10.)]]
    residuals = {"inclination_degrees_disk": [], "inclination_degrees_spheroid": [],
                 "log_spheroid_scale_ratio": []}
    def attenuation(source, t, i, r):
        result = provider.evaluate(wave, source=source, optical_depth_v=t,
                                   inclination_degrees=i, spheroid_scale_ratio=r)
        return -2.5*np.log10(result.transmission)
    for depth in tau:
        for source in ("disk", "spheroid"):
            for r in ([None] if source == "disk" else [0.01, 0.1, 1., 10.]):
                for j in (1, 15, 30, 43, 44):
                    weight = (inc[j] - inc[j-1]) / (inc[j+1] - inc[j-1])
                    prediction = ((1-weight)*attenuation(source, depth, inc[j-1], r)
                                  + weight*attenuation(source, depth, inc[j+1], r))
                    truth = attenuation(source, depth, inc[j], r)
                    residuals["inclination_degrees_" + source].extend(prediction - truth)
        for j in (1, 10, 25, 40, 49):
            weight = np.log(ratio[j]/ratio[j-1]) / np.log(ratio[j+1]/ratio[j-1])
            for angle in (0., 60., 88.):
                prediction = ((1-weight)*attenuation("spheroid", depth, angle, ratio[j-1])
                              + weight*attenuation("spheroid", depth, angle, ratio[j+1]))
                truth = attenuation("spheroid", depth, angle, ratio[j])
                residuals["log_spheroid_scale_ratio"].extend(prediction - truth)
    return {key: {"cells": len(values), "mean_delta_A_mag": float(np.mean(values)),
                  "p95_abs_delta_A_mag": float(np.percentile(np.abs(values), 95)),
                  "max_abs_delta_A_mag": float(np.max(np.abs(values)))}
            for key, values in residuals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", help="Local canonical Benson R_V=4 HDF5 file")
    parser.add_argument("--tau", type=float, default=1.)
    parser.add_argument("--inclination", type=float, default=60.)
    parser.add_argument("--scale-ratio", type=float, default=0.1)
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="Time N 369-wavelength queries per source")
    parser.add_argument("--check-interpolation", action="store_true")
    args = parser.parse_args()
    if args.benchmark < 0:
        parser.error("--benchmark must be nonnegative")
    start = time.perf_counter()
    receipt = verify_benson_atlas(args.atlas)
    report = {"checksum_seconds": time.perf_counter() - start}
    with Benson2018DiffuseProvider(receipt) as model:
        report["provenance"] = model.provenance
        report["bounds"] = model.bounds
        report["wavelength_micron"] = [0.15, 0.55, 2., 3.]
        for source in ("disk", "spheroid"):
            result = model.evaluate(report["wavelength_micron"], source=source,
                                    inclination_degrees=args.inclination, optical_depth_v=args.tau,
                                    spheroid_scale_ratio=args.scale_ratio)
            report[source] = {"transmission": result.transmission.tolist(),
                              "flags": sorted(result.flags), "coordinates": result.coordinates}
        if args.benchmark:
            wave = np.geomspace(0.1, 3., 369)
            report["benchmark"] = {"queries_per_source": args.benchmark, "wavelengths": len(wave)}
            for source in ("disk", "spheroid"):
                start = time.perf_counter()
                for j in range(args.benchmark):
                    model.evaluate(wave, source=source, inclination_degrees=(j*7.3) % 90,
                                   optical_depth_v=0.01*10**(4*(j % 101)/100),
                                   spheroid_scale_ratio=0.001*10**(5*(j % 97)/96))
                report["benchmark"][source + "_milliseconds_per_query"] = (
                    1000*(time.perf_counter()-start)/args.benchmark)
        if args.check_interpolation:
            report["held_out_node_checks"] = held_out_checks(model, receipt.path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["process_peak_rss_mib"] = peak / (1024**2 if sys.platform == "darwin" else 1024)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
