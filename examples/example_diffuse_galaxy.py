"""Plot a diffuse-only galaxy SED and optionally smoke-test catalog rows.

Requires the canonical external atlas, a compatible SED template/catalog, and
matplotlib. Sampling is deterministic and unweighted, not a population study.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from galacticus_sed_calculator import SEDCalculator
from galacticus_sed_calculator.diffuse_dust import Benson2018DiffuseProvider, verify_benson_atlas
from galacticus_sed_calculator.galaxy_diffuse import BensonGalaxyDust, DiffuseDustParameters
from galacticus_sed_calculator.sed_calculator import detect_galacticus_format


def plot_result(result, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    wave = result.wavelength_micron
    for (source, comp), color in zip(result.components.items(), ['#0072B2', '#D55E00']):
        axes[0].plot(wave, comp.input_light.stellar_lnu, color=color, ls='--', alpha=.65,
                     label=f'{source.capitalize()}: intrinsic')
        axes[0].plot(wave, comp.stellar_lnu, color=color, label=f'{source.capitalize()}: diffuse')
        axes[1].plot(wave, comp.stellar_transmission, color=color, label=source.capitalize())
        selected = comp.input_light.line_luminosity > 0
        axes[1].scatter(comp.input_light.line_wavelength_micron[selected], comp.line_transmission[selected],
                        color=color, edgecolor='white', s=45, zorder=3)
    axes[0].set(xscale='log', yscale='log', ylabel=r'Stellar $L_\nu$ [$L_\odot$ Hz$^{-1}$]',
                title='Separate component continua')
    axes[1].set(xscale='log', ylabel='Diffuse transmission', title='Lines share the component transfer', ylim=(0, None))
    axes[1].axhline(1, color='gray', ls=':', lw=1)
    for ax in axes:
        ax.set(xlabel='Rest-frame wavelength [µm]', xlim=(wave[0], wave[-1]))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=9)
    d = result.diagnostics
    fig.suptitle('From catalog properties to diffuse-attenuated component light',
                 x=.08, y=.98, ha='left', fontsize=16, weight='bold')
    fig.text(.08, .90, f"Galaxy row {d['galaxy_index']}; z={d['redshift']:.2f}; "
             rf"$i={d['inclination_degrees']:.1f}^\circ$; $\tau_V={d['disk_optical_depth_v']:.3f}$. "
             'Dots: star-forming emission lines.', fontsize=11)
    fig.text(.08, .025, 'Diffuse only: no birth clouds, local nebular screen or GB10. '
             'Benson (2018) / Zenodo 6335545 atlas (CC BY 4.0).', fontsize=8, color='#555555')
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.77, wspace=.28)
    fig.savefig(output/'benson_diffuse_galaxy.png', dpi=160, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas')
    parser.add_argument('--catalog', default='data/romanUNIT.hdf5')
    parser.add_argument('--template', default='data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5')
    parser.add_argument('--galaxy-index', type=int, default=0)
    parser.add_argument('--sample-size', type=int, default=0,
                        help='Also evaluate evenly spaced catalog rows; report failures, never silently sanitize')
    parser.add_argument('--negative-metal-tolerance-msun', type=float, default=0.)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error('--sample-size must be nonnegative')
    calc = SEDCalculator(args.template)
    wave = np.geomspace(.12, 2., 369)
    receipt = verify_benson_atlas(args.atlas)
    report = {}
    with Benson2018DiffuseProvider(receipt) as provider:
        model = BensonGalaxyDust(provider, DiffuseDustParameters(
            negative_metal_tolerance_msun=args.negative_metal_tolerance_msun))
        result = calc.calculate_diffuse_galaxy_seds(args.catalog, args.galaxy_index, wave, model)
        report.update(provenance=result.provenance, example_diagnostics=result.diagnostics)
        if args.sample_size:
            _, base = detect_galacticus_format(args.catalog)
            with h5py.File(args.catalog) as f:
                count = len(f[base+'/nodeData/diskAbundancesGasMetals'])
            indices = np.unique(np.linspace(0, count-1, min(count, args.sample_size), dtype=int))
            flags, failures = Counter(), []
            start = time.perf_counter()
            for index in indices:
                try:
                    sample = calc.calculate_diffuse_galaxy_seds(args.catalog, int(index), wave, model)
                    flags.update(sample.diagnostics['flags'])
                except ValueError as exc:
                    failures.append({'galaxy_index': int(index), 'error': str(exc)})
            seconds = time.perf_counter()-start
            report['sample'] = {'rows': len(indices), 'successful': len(indices)-len(failures),
                                'seconds': seconds, 'milliseconds_per_attempt': 1000*seconds/len(indices),
                                'flags_among_successes': dict(flags), 'failures': failures,
                                'sampling': 'evenly spaced row indices; unweighted; one warm process'}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_before_plot_mib'] = peak/(1024**2 if sys.platform == 'darwin' else 1024)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_result(result, args.output_dir)
    with (args.output_dir/'benson_diffuse_galaxy.json').open('w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    summary = {'peak_rss_before_plot_mib': report['peak_rss_before_plot_mib']}
    if 'sample' in report:
        summary['sample'] = {k: v for k, v in report['sample'].items() if k != 'failures'}
        summary['sample']['failed'] = len(report['sample']['failures'])
    print(json.dumps(summary, indent=2))
    print(f'Wrote figure and diagnostic report to {args.output_dir}')


if __name__ == '__main__':
    main()
