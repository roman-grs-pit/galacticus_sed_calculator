"""Reproduce the diffuse-transfer figures in docs/BENSON_DIFFUSE.md.

Requires matplotlib (available in the diagnostics extra) and the external
canonical atlas. Run with the package installed or PYTHONPATH=.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

from galacticus_sed_calculator.diffuse_dust import (
    Benson2018DiffuseProvider,
    verify_benson_atlas,
)


COLORS = ["#0072B2", "#D55E00", "#009E73"]
CREDIT = "Benson (2018) / Zenodo 6335545, R_V = 4 atlas (CC BY 4.0) | Diffuse dust only"


def transmission(provider, wave, source="disk", inclination=60, tau=1, ratio=0.1):
    return provider.evaluate(
        wave, source=source, inclination_degrees=inclination,
        optical_depth_v=tau, spheroid_scale_ratio=ratio,
    ).transmission


def finish(fig, axes, path, title, subtitle):
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=17, weight="bold")
    fig.text(0.08, 0.895, subtitle, fontsize=11, color="#444444")
    fig.text(0.08, 0.025, CREDIT, fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.77, wspace=0.27)
    fig.savefig(path, dpi=160, facecolor="white", metadata={"Description": CREDIT})
    plt.close(fig)


def plot_spectra(provider, output):
    wave = np.geomspace(0.1, 3, 500)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, source in zip(axes, ("disk", "spheroid")):
        for color, angle in zip(COLORS, (0, 60, 85)):
            values = transmission(provider, wave, source, inclination=angle)
            ax.plot(wave, values, color=color, label=rf"$i={angle}^\circ$")
        ax.axhline(1, color="#777777", lw=1, ls=":")
        ax.set(xscale="log", xlim=(0.1, 3), title=f"{source.capitalize()} light through disk dust",
               xlabel=r"Rest-frame wavelength $\lambda$ [$\mu$m]")
        ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 3])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.legend(frameon=False, loc="lower right")
    axes[0].set(ylabel=r"Transmission $T = L_{\rm emergent}/L_{\rm intrinsic}$")
    # Autoscaling retains values above unity: directional scattering is not clipped.
    axes[0].set_ylim(bottom=0)
    finish(fig, axes, output / "benson_diffuse_spectra.png",
           "One dust disk, two source populations",
           r"$\tau_V=1$; spheroid scale ratio $r_{\rm sph}/r_{\rm disk}=0.1$; "
           r"$i=0^\circ$ is face-on. Dotted line: no attenuation.")


def plot_geometry(provider, output):
    angles = np.linspace(0, 90, 181)
    depths = np.geomspace(0.001, 100, 181)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    cases = [("disk", 0.1, "#222222", "Disk sources")]
    cases += [("spheroid", ratio, color, rf"Spheroid: $r_{{\rm sph}}/r_{{\rm disk}}={ratio:g}$")
              for ratio, color in zip((0.01, 0.1, 1), COLORS)]
    for source, ratio, color, label in cases:
        axes[0].plot(angles, [transmission(provider, 0.44, source, a, 1, ratio)
                              for a in angles], color=color, label=label)
        axes[1].plot(depths, [transmission(provider, 0.44, source, 60, t, ratio)
                              for t in depths], color=color, label=label)
    axes[0].set(xlim=(0, 90), xlabel="Inclination [degrees]",
                ylabel="Transmission at 0.44 µm", title=r"Change viewing angle ($\tau_V=1$)")
    axes[0].set_xticks([0, 30, 60, 90])
    axes[1].set(xscale="log", xlim=(0.001, 100), xlabel=r"Central face-on optical depth $\tau_V$",
                title=r"Change dust optical depth ($i=60^\circ$)")
    for ax in axes:
        ax.axhline(1, color="#777777", lw=1, ls=":")
    axes[0].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    finish(fig, axes, output / "benson_diffuse_geometry.png",
           "Attenuation depends on where the light originates",
           "Rest-frame 0.44 µm (a monochromatic B-band proxy, not a filter integral). "
           "Dust remains in the disk in every curve.")


def plot_interpolation(provider, atlas, output):
    # Read genuine atlas nodes directly, independently of the provider curve.
    with h5py.File(atlas, "r") as f:
        wave = f["wavelength"][:]
        depth = f["opticalDepth"][:]
        angle = f["inclination"][:]
        j = np.argmin(abs(angle - 60))
        k = np.argmin(abs(depth - 1))
        w = np.argmin(abs(wave - 0.44))
        wavelength_nodes = np.flatnonzero((wave >= 0.16) & (wave <= 0.32))
        depth_nodes = np.flatnonzero((depth >= 0.1) & (depth <= 10))
        raw_wave = f["attenuationDisk"][wavelength_nodes, j, k]
        raw_depth = f["attenuationDisk"][w, j, depth_nodes]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, nodes, raw, varying in zip(
        axes, (wave[wavelength_nodes], depth[depth_nodes]),
        (raw_wave, raw_depth), ("wavelength", "optical depth"),
    ):
        query = np.geomspace(nodes[0], nodes[-1], 500)
        if varying == "wavelength":
            values = transmission(provider, query, inclination=angle[j], tau=depth[k])
        else:
            values = np.array([transmission(provider, wave[w], inclination=angle[j], tau=t)
                               for t in query])
        ax.plot(query, -2.5*np.log10(values), color=COLORS[0], lw=2, label="Provider interpolation")
        ax.scatter(nodes, -2.5*np.log10(raw), color="#222222", s=23,
                   zorder=3, label="Native atlas nodes")
        ax.set(xscale="log", xlim=(nodes[0], nodes[-1]), ylabel=r"Attenuation $A=-2.5\log_{10}T$ [mag]")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set(xlabel=r"Rest-frame wavelength $\lambda$ [$\mu$m]",
                title=rf"Wavelength slice: $\tau_V={depth[k]:.3g}$")
    axes[0].set_xticks([0.18, 0.22, 0.26, 0.30])
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[1].set(xlabel=r"Central face-on optical depth $\tau_V$",
                title=rf"Optical-depth slice: $\lambda={wave[w]:.3f}$ µm")
    finish(fig, axes, output / "benson_diffuse_interpolation.png",
           "How the provider fills gaps between atlas nodes",
           rf"Disk light, $i={angle[j]:g}^\circ$. "
           r"Interpolate $A$ against $\log\lambda$ and $\log\tau_V$, then return $T=10^{-0.4A}$.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", help="Local canonical Benson R_V=4 HDF5 atlas")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    receipt = verify_benson_atlas(args.atlas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}), \
            Benson2018DiffuseProvider(receipt) as provider:
        plot_spectra(provider, args.output_dir)
        plot_geometry(provider, args.output_dir)
        plot_interpolation(provider, receipt.path, args.output_dir)
    print(f"Wrote three diffuse-transfer figures to {args.output_dir}")


if __name__ == "__main__":
    main()
