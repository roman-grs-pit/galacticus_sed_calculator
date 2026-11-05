"""Serial/parallel equivalence, streaming, and compatibility of catalog SEDs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import astropy.units as u
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import generate_catalog_seds as script
import create_downsampled_catalog as legacy
from test_downsampled_catalog import _make_lightcone_catalog, _add_dust_attenuated_node_data

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT/'data/nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'


@pytest.fixture
def source(tmp_path):
    path = tmp_path/'source.hdf5'
    shutil.copy2(ROOT/'data/romanUNIT.hdf5', path)
    with h5py.File(path, 'r') as f:
        count = len(f['Lightcone/Output1/nodeData/lightconeRedshiftObserved'])
    _add_dust_attenuated_node_data(path, count)
    with h5py.File(path, 'a') as f:
        nd = f['Lightcone/Output1/nodeData']
        if 'randomUniform' in nd:
            del nd['randomUniform']
        nd['randomUniform'] = np.random.default_rng(42).uniform(size=(count, 2))
        g = f['Lightcone/Output1/dustAttenuatedNodeData']
        params = json.loads(g.attrs['dust_params'])
        params['attenuation_scatter'] = 0.3
        g.attrs['dust_params'] = json.dumps(params)
        g.attrs['random_uniform_index'] = 0
        g.attrs['continuum_dust'] = json.dumps({'model': 'fixed_av', 'params': {'A_V': 1.2}, 'law': 'calzetti'})
    return str(path)


@pytest.mark.parametrize('unit', ['fnu', 'flam'])
def test_parallel_matches_serial_with_dust_and_row_order(source, tmp_path, unit):
    selected = np.array([0, 2, 4])
    outputs = []
    wave = np.linspace(8000, 20000, 301)*u.AA
    for jobs in (1, 2):
        output = str(tmp_path/f'{jobs}.hdf5')
        script.copy_galaxy_data(source, output, selected, '/Lightcone/Output1')
        array, actual_wave = script.calculate_and_save_seds(
            source, output, selected, '/Lightcone/Output1', str(TEMPLATE), wave,
            flux_unit=unit, n_jobs=jobs, batch_size=2, return_array=False)
        assert array is None
        np.testing.assert_array_equal(actual_wave, wave.value)
        outputs.append(output)
    with h5py.File(outputs[0]) as serial, h5py.File(outputs[1]) as parallel:
        for group in ('nodeData', 'dustAttenuatedNodeData'):
            path = 'Lightcone/Output1/'+group+'/observedSED'
            np.testing.assert_array_equal(serial[path][...], parallel[path][...])
            assert np.all(np.isfinite(parallel[path][...]))
            assert np.any(parallel[path][...] > 0)
            assert serial[path].attrs['flux_unit'] == parallel[path].attrs['flux_unit']
        assert parallel['Lightcone/Output1'].attrs['sedGenerationComplete']
        np.testing.assert_array_equal(parallel['Lightcone/Output1'].attrs['selectedIndices'], selected)
    # Independent existing array implementation, not just the new serial path.
    calc = script.SEDCalculator(str(TEMPLATE), cosmology=script.FlatLambdaCDM(H0=67.74, Om0=0.3089))
    expected = script._compute_sed_array(source, selected, wave, calc, 'total', True, flux_unit=unit, quiet=True)
    with h5py.File(outputs[0]) as f:
        np.testing.assert_array_equal(expected, f['Lightcone/Output1/nodeData/observedSED'][...])


def test_worker_failure_leaves_incomplete_marker(source, tmp_path):
    output = str(tmp_path/'failed.hdf5')
    script.copy_galaxy_data(source, output, [0], '/Lightcone/Output1')
    with pytest.raises(RuntimeError, match='source row 0'):
        script.calculate_and_save_seds(source, output, [0], '/Lightcone/Output1', str(TEMPLATE),
                                      np.linspace(8000, 18000, 30)*u.AA,
                                      component='invalid', n_jobs=2, return_array=False)
    with h5py.File(output) as f:
        assert not f['Lightcone/Output1'].attrs['sedGenerationComplete']


def test_all_galaxies_and_legacy_alias(tmp_path):
    assert legacy.create_downsampled_catalog is script.generate_catalog_seds
    source = str(tmp_path/'many.hdf5')
    _make_lightcone_catalog(source, n_gals=9000)
    rows = script.filter_galaxies(source, '/Lightcone/Output1', 'lightcone')
    assert len(rows) == 9000
    output = str(tmp_path/'copy.hdf5')
    script.copy_galaxy_data(source, output, rows, '/Lightcone/Output1')
    with h5py.File(output) as f:
        path = f['Lightcone/Output1'].attrs['selectedIndicesDataset']
        np.testing.assert_array_equal(f[path][...], rows)
    with pytest.raises(FileExistsError):
        script.copy_galaxy_data(source, output, rows, '/Lightcone/Output1')


@pytest.mark.parametrize('name', ['generate_catalog_seds.py', 'create_downsampled_catalog.py'])
def test_cli_help(name):
    result = subprocess.run([sys.executable, str(ROOT/'scripts'/name), '--help'], capture_output=True, text=True, check=True)
    assert '--n-jobs' in result.stdout
    assert '--batch-size' in result.stdout


@pytest.mark.parametrize('jobs,batch', [(0, 32), (1, 0), (-1, 32)])
def test_invalid_worker_options(jobs, batch):
    with pytest.raises(ValueError):
        script.generate_catalog_seds('missing.hdf5', 'missing.hdf5', 'unused.hdf5', n_jobs=jobs, batch_size=batch)


@pytest.mark.parametrize('name', ['generate_catalog_seds.py', 'create_downsampled_catalog.py'])
def test_cli_parallel_generation(name, tmp_path):
    output = tmp_path/'cli.hdf5'
    subprocess.run([
        sys.executable, str(ROOT/'scripts'/name), str(ROOT/'data/romanUNIT.hdf5'),
        str(TEMPLATE), str(output), '--n-jobs', '2', '--batch-size', '2',
        '--wavelength-min', '8000', '--wavelength-max', '18000',
        '--wavelength-npoints', '51', '--no-emission-lines',
    ], check=True, capture_output=True, text=True)
    with h5py.File(output) as f:
        assert f['Lightcone/Output1'].attrs['sedGenerationComplete']
        assert f['Lightcone/Output1/nodeData/observedSED'].shape == (5, 51)


@pytest.mark.parametrize('jobs', [1, 2])
def test_dust_only_matches_default(source, tmp_path, jobs):
    wave = np.linspace(8000, 20000, 51)*u.AA
    both = tmp_path/'both.hdf5'
    only = tmp_path/'only.hdf5'
    script.generate_catalog_seds(source, str(TEMPLATE), str(both),
                                 obs_wavelengths=wave, batch_size=2)
    script.generate_catalog_seds(source, str(TEMPLATE), str(only),
                                 obs_wavelengths=wave, batch_size=2, n_jobs=jobs,
                                 dust_attenuated_only=True)
    with h5py.File(both) as a, h5py.File(only) as b:
        base = 'Lightcone/Output1'
        for name in ('observedSED', 'observedSEDWavelengths'):
            assert name not in b[base+'/nodeData']
        np.testing.assert_array_equal(
            a[base+'/dustAttenuatedNodeData/observedSED'][...],
            b[base+'/dustAttenuatedNodeData/observedSED'][...])
        assert b[base].attrs['sedGenerationComplete']


def test_dust_only_skips_free_calculation(monkeypatch):
    calls = []
    def compute(*args, **kwargs):
        calls.append(kwargs)
        return np.ones((2, 3))
    monkeypatch.setattr(script, '_compute_sed_array', compute)
    state = (None, 'source', None, 'total', True, 'flam',
             {'dust_model': 'example'}, True)
    start, free, dusty = script._calculate_batch((0, [0, 1]), state)
    assert free is None
    assert len(calls) == 1
    assert calls[0]['dust_model'] == 'example'
    assert dusty.shape == (2, 3)


def test_dust_only_requires_metadata_before_copy(tmp_path):
    output = tmp_path/'absent.hdf5'
    with pytest.raises(ValueError, match='requires catalog dust metadata'):
        script.generate_catalog_seds(str(ROOT/'data/romanUNIT.hdf5'),
                                     str(TEMPLATE), str(output),
                                     dust_attenuated_only=True)
    assert not output.exists()
