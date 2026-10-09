"""Galaxy physics and component-light integration, without external downloads."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import astropy.units as u
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galacticus_sed_calculator import SEDCalculator
from galacticus_sed_calculator.diffuse_dust import Benson2018DiffuseProvider
from galacticus_sed_calculator.galaxy_diffuse import (
    BensonGalaxyDust, ComponentLight, DiffuseDustParameters, DiffuseGalaxyInputs,
    DiffuseStatus, read_diffuse_galaxy_inputs,
)
from test_diffuse_dust import atlas  # Reuse the small, checksum-verified analytic atlas fixture.


@pytest.fixture
def model(atlas):
    with Benson2018DiffuseProvider(atlas[0]) as provider:
        yield BensonGalaxyDust(provider)


@pytest.fixture
def galaxy():
    return DiffuseGalaxyInputs(1e7, 1e6, 0.003, 0.0003, orientation_uniform=0.5)


def evaluate(model, galaxy, source="disk", light=None):
    return model.apply([0.3, 1.], galaxy,
                       {source: light if light is not None else ComponentLight(np.ones(2))})


def test_optical_depth_units_scalings_and_ignored_medium(model, galaxy):
    result = evaluate(model, galaxy)
    expected = ((32062.2129019*u.cm**2/u.g) * (1e7*0.334*0.75*u.Msun)
                / (2*np.pi*(0.003*u.Mpc)**2)).decompose().value
    assert result.diagnostics['disk_optical_depth_v'] == pytest.approx(expected)
    assert result.diagnostics['ignored_spheroid_metal_fraction'] == pytest.approx(1/11)
    assert result.diagnostics['status'] & DiffuseStatus.IGNORED_SPHEROID_METALS
    for changed, factor in [(replace(galaxy, disk_metal_mass_msun=2e7), 2),
                            (replace(galaxy, disk_scale_radius_mpc=0.006), 0.25)]:
        assert evaluate(model, changed).diagnostics['disk_optical_depth_v'] == pytest.approx(factor*expected)
    half_dust = BensonGalaxyDust(model.provider, DiffuseDustParameters(dust_to_metals_ratio=0.167))
    assert evaluate(half_dust, galaxy).diagnostics['disk_optical_depth_v'] == pytest.approx(expected/2)
    json.dumps(result.provenance, allow_nan=False)
    json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize("uniform,angle", [(0, 90), (0.5, 60), (1, 0)])
def test_orientation_endpoints(model, galaxy, uniform, angle):
    result = evaluate(model, replace(galaxy, orientation_uniform=uniform))
    assert result.diagnostics['inclination_degrees'] == pytest.approx(angle)


def test_explicit_orientation_overrides_unused_random(model, galaxy):
    result = evaluate(model, replace(galaxy, inclination_degrees=17, orientation_uniform=np.nan))
    assert result.diagnostics['inclination_degrees'] == 17
    assert result.diagnostics['orientation_source'] == 'explicit'


@pytest.mark.parametrize("changed,match", [
    ({"orientation_uniform": None}, 'Supply orientation'),
    ({"orientation_uniform": 1.1}, 'in \\[0,1\\]'),
    ({"orientation_uniform": np.nan}, 'finite'),
    ({"inclination_degrees": -1}, 'in \\[0,90\\]'),
    ({"disk_metal_mass_msun": -1}, 'negative-metal tolerance'),
    ({"spheroid_metal_mass_msun": -1e12}, 'negative-metal tolerance'),
    ({"disk_metal_mass_msun": np.inf}, 'finite'),
    ({"disk_scale_radius_mpc": 0}, 'positive disk'),
    ({"disk_scale_radius_mpc": None}, 'finite'),
    ({"disk_scale_radius_mpc": np.nan}, 'finite'),
])
def test_invalid_galaxy_inputs(model, galaxy, changed, match):
    with pytest.raises(ValueError, match=match):
        evaluate(model, replace(galaxy, **changed))


@pytest.mark.parametrize("params", [dict(cloud_fraction=1.1), dict(dust_to_metals_ratio=-1),
                                   dict(negative_metal_tolerance_msun=-1), dict(cloud_fraction=np.nan)])
def test_invalid_parameters(params):
    with pytest.raises(ValueError):
        DiffuseDustParameters(**params)


def test_bounded_sanitation_is_opt_in_and_records_original(model, galaxy):
    bad = replace(galaxy, disk_metal_mass_msun=-1e-8, spheroid_metal_mass_msun=-1e-9)
    with pytest.raises(ValueError):
        evaluate(model, bad)
    permissive = BensonGalaxyDust(model.provider, DiffuseDustParameters(negative_metal_tolerance_msun=1e-8))
    result = evaluate(permissive, bad)
    assert result.diagnostics['status'] == int(DiffuseStatus.DISK_METALS_SANITIZED | DiffuseStatus.SPHEROID_METALS_SANITIZED)
    assert result.diagnostics['original_inputs']['disk_metal_mass_msun'] == -1e-8
    assert result.diagnostics['disk_metal_mass_msun_used'] == 0
    np.testing.assert_array_equal(result.components['disk'].stellar_lnu, [1, 1])
    with pytest.raises(ValueError):
        evaluate(permissive, replace(bad, disk_metal_mass_msun=-1e-7))


@pytest.mark.parametrize("params", [DiffuseDustParameters(), DiffuseDustParameters(cloud_fraction=1),
                                   DiffuseDustParameters(dust_to_metals_ratio=0)])
def test_dust_free_needs_no_radii_but_records_spheroid_metals(model, galaxy, params):
    dust_free = replace(galaxy, disk_metal_mass_msun=0, disk_scale_radius_mpc=None,
                        spheroid_scale_radius_mpc=None)
    result = evaluate(BensonGalaxyDust(model.provider, params), dust_free, 'spheroid')
    assert result.diagnostics['disk_optical_depth_v'] == 0
    assert 'ignored_spheroid_metals' in result.diagnostics['flags']
    np.testing.assert_array_equal(result.components['spheroid'].stellar_lnu, [1, 1])
    if params != DiffuseDustParameters():
        assert evaluate(BensonGalaxyDust(model.provider, params), galaxy).diagnostics['disk_optical_depth_v'] == 0


def test_source_geometry_only_required_for_nonzero_spheroid_light(model, galaxy):
    missing = replace(galaxy, spheroid_scale_radius_mpc=None)
    evaluate(model, missing, 'disk')
    result = evaluate(model, missing, 'spheroid', ComponentLight(np.zeros(2)))
    assert not result.components['spheroid'].coordinates['source_has_light']
    assert result.diagnostics['spheroid_scale_ratio'] is None
    with pytest.raises(ValueError, match='spheroid_scale_radius'):
        evaluate(model, missing, 'spheroid')
    # A line-only source still requires a size; no disk light does not remove disk dust.
    with pytest.raises(ValueError, match='spheroid_scale_radius'):
        evaluate(model, missing, 'spheroid', ComponentLight(np.zeros(2), np.array([0.3]), np.array([1.])))
    with pytest.raises(ValueError, match='positive disk'):
        evaluate(model, replace(galaxy, disk_scale_radius_mpc=0), light=ComponentLight(np.zeros(2)))


def test_equal_stellar_nebular_transfer_and_separate_component_sum(model, galaxy):
    light = ComponentLight(np.array([2., 3.]), np.array([0.3, 1.]), np.array([7., 11.]), ('a', 'b'))
    result = model.apply([0.3, 1.], galaxy, {'disk': light, 'spheroid': light})
    for comp in result.components.values():
        np.testing.assert_array_equal(comp.stellar_transmission, comp.line_transmission)
        np.testing.assert_allclose(comp.stellar_lnu, light.stellar_lnu*comp.stellar_transmission)
        np.testing.assert_allclose(comp.line_luminosity, light.line_luminosity*comp.line_transmission)
    np.testing.assert_array_equal(light.stellar_lnu, [2, 3])  # No mutation of inputs.
    np.testing.assert_array_equal(result.total_stellar_lnu,
                                  result.components['disk'].stellar_lnu + result.components['spheroid'].stellar_lnu)
    assert result.diagnostics['spheroid_scale_ratio'] == pytest.approx(0.1)
    # Feed light after an arbitrary local screen; it is not applied a second time.
    local = ComponentLight(light.stellar_lnu*0.5, light.line_wavelength_micron, light.line_luminosity*0.25)
    after_local = evaluate(model, galaxy, light=local).components['disk']
    np.testing.assert_allclose(after_local.stellar_lnu, result.components['disk'].stellar_lnu*0.5)
    np.testing.assert_allclose(after_local.line_luminosity, result.components['disk'].line_luminosity*0.25)


def test_provider_flags_propagated(model, galaxy):
    model.provider.high_tau_policy = 'clip'
    model.provider.scale_ratio_policy = 'clip'
    result = evaluate(model, replace(galaxy, disk_metal_mass_msun=1e12, spheroid_scale_radius_mpc=1), 'spheroid')
    assert result.diagnostics['status'] & DiffuseStatus.OPTICAL_DEPTH_CLIPPED
    assert result.diagnostics['status'] & DiffuseStatus.SCALE_RATIO_CLIPPED
    assert result.components['spheroid'].coordinates['optical_depth_v_used'] == 10
    assert result.diagnostics['disk_optical_depth_v'] > 10


def test_declared_incompatible_profiles_rejected_only_when_relevant(model, galaxy):
    with pytest.raises(ValueError, match='exponentialDisk'):
        evaluate(model, replace(galaxy, catalog_profiles={'disk': 'other'}))
    different_spheroid = replace(galaxy, catalog_profiles={'spheroid': 'other'})
    evaluate(model, different_spheroid, 'disk')
    with pytest.raises(ValueError, match='hernquist'):
        evaluate(model, different_spheroid, 'spheroid')
    evaluate(model, different_spheroid, 'spheroid', ComponentLight(np.zeros(2)))


@pytest.mark.parametrize("light", [ComponentLight(np.ones(3)), ComponentLight(np.array([-1, 2])),
                                  ComponentLight(np.array([np.nan, 2])),
                                  ComponentLight(np.ones(2), np.ones(2), np.ones(3))])
def test_bad_light_shapes_and_values(model, galaxy, light):
    with pytest.raises(ValueError):
        evaluate(model, galaxy, light=light)


def test_agn_and_out_of_domain_requests_rejected(model, galaxy):
    with pytest.raises(ValueError, match='AGN'):
        evaluate(model, galaxy, 'AGN')
    with pytest.raises(ValueError, match='coverage'):
        evaluate(model, galaxy, light=ComponentLight(np.ones(2), np.array([4.]), np.zeros(1)))


@pytest.mark.parametrize("base", ['Lightcone/Output1', 'Outputs/Output2'])
def test_catalog_reader_units_orientation_and_missing_fields(tmp_path, base):
    with h5py.File(tmp_path/'catalog.hdf5', 'w') as f:
        node = f.create_group(base+'/nodeData')
        # Deliberately use kg and kpc, not the usual Galacticus Msun and Mpc.
        for name, value, factor in [('diskAbundancesGasMetals', 1e7*u.Msun.to(u.kg), 1),
                                    ('spheroidAbundancesGasMetals', 0, 1),
                                    ('diskRadius', 3, u.kpc.to(u.m))]:
            node[name] = [value]
            node[name].attrs['unitsInSI'] = factor
        node['randomUniform'] = [[0.9, 0.25, 0.7]]
        inputs = read_diffuse_galaxy_inputs(node, 0)
        assert inputs.disk_metal_mass_msun == pytest.approx(1e7)
        assert inputs.disk_scale_radius_mpc == pytest.approx(0.003)
        assert inputs.spheroid_scale_radius_mpc is None
        assert inputs.orientation_uniform == 0.25
        del node['randomUniform']
        assert read_diffuse_galaxy_inputs(node, 0, inclination_degrees=12).inclination_degrees == 12
        with pytest.raises(ValueError, match='randomUniform'):
            read_diffuse_galaxy_inputs(node, 0)
        del node['diskRadius'].attrs['unitsInSI']
        with pytest.raises(ValueError, match='unitsInSI'):
            read_diffuse_galaxy_inputs(node, 0, inclination_degrees=12)


DATA = Path(__file__).resolve().parents[1] / 'data'
TEMPLATE = DATA / 'nodePropertyExtractorSED_Nt50_NZ11_ageMinimum0.001.hdf5'


@pytest.mark.parametrize("fixed_time", [False, True])
def test_calculator_rest_frame_integration(model, tmp_path, fixed_time):
    catalog, template = DATA/'romanUNIT.hdf5', TEMPLATE
    if fixed_time:
        catalog, template = tmp_path/'fixed.hdf5', tmp_path/'template.hdf5'
        with h5py.File(TEMPLATE) as src, h5py.File(template, 'w') as dst:
            output_time = 10.0
            times = output_time * (1 - src['ages'][:] / src['ages'][0])
            for key in ('sedTemplate', 'metallicity', 'wavelength'):
                src.copy(key, dst)
            dst['time'] = times
        with h5py.File(DATA/'romanUNIT.hdf5') as src, h5py.File(catalog, 'w') as dst:
            src.copy('Parameters', dst)
            sfh = dst['Parameters/starFormationHistory']
            sfh.attrs['timeStepMinimum'] = sfh.attrs['ageMinimum']
            sfh.attrs['countTimeStepsMaximum'] = len(times)
            output = dst.create_group('Outputs/Output2')
            output.attrs['outputTime'] = output_time
            src.copy('Lightcone/Output1/nodeData', output)
            for comp in ('disk', 'spheroid'):
                output[f'nodeData/{comp}StarFormationHistoryMass'].attrs['time'] = times
    calc = SEDCalculator(template)
    wave = np.array([0.3, 0.6565, 1.0])
    result = calc.calculate_diffuse_galaxy_seds(catalog, 0, wave*u.micron, model)
    assert set(result.components) == {'disk', 'spheroid'}
    galaxy = calc.read_galacticus_galaxy(catalog, 0)
    for source, component in result.components.items():
        expected = np.interp(wave*1e4, calc.sedWavelength, calc.calculate_rest_frame_sed(galaxy[source+'SFH']))
        np.testing.assert_allclose(component.input_light.stellar_lnu, expected)
        assert 'balmerAlpha6565' in component.input_light.line_names
        index = component.input_light.line_names.index('balmerAlpha6565')
        assert component.line_transmission[index] == component.stellar_transmission[1]
    # Underlying template extends well beyond the atlas: an in-range request is valid.
    assert calc.sedWavelength[-1] > 30000
    no_lines = calc.calculate_diffuse_galaxy_seds(catalog, 0, wave, model, include_emission_lines=False)
    assert all(c.line_luminosity.size == 0 for c in no_lines.components.values())
    with pytest.raises(ValueError, match='coverage'):
        calc.calculate_diffuse_galaxy_seds(catalog, 0, [0.5, 3.5], model)


@pytest.mark.skipif(not os.environ.get('BENSON2018_ATLAS'), reason='Supply canonical atlas')
def test_canonical_catalog_smoke():
    calc = SEDCalculator(TEMPLATE)
    with Benson2018DiffuseProvider(os.environ['BENSON2018_ATLAS']) as provider:
        model = BensonGalaxyDust(provider)
        for index in range(5):
            result = calc.calculate_diffuse_galaxy_seds(DATA/'romanUNIT.hdf5', index, np.geomspace(.12, 2, 100), model)
            assert np.all(np.isfinite(result.total_stellar_lnu))
            assert result.diagnostics['status'] >= 0


def test_line_units_and_exact_line_center_range(model, tmp_path):
    catalog = tmp_path/'line_units.hdf5'
    with h5py.File(DATA/'romanUNIT.hdf5') as src, h5py.File(catalog, 'w') as dst:
        for key in ('Parameters', 'Lightcone'):
            src.copy(key, dst)
    calc = SEDCalculator(TEMPLATE)
    original = calc.calculate_diffuse_galaxy_seds(catalog, 0, [0.6565], model)
    for source in ('disk', 'spheroid'):
        assert original.components[source].input_light.line_names == ('balmerAlpha6565',)
    with h5py.File(catalog, 'r+') as f:
        line = f['Lightcone/Output1/nodeData/luminosityEmissionLineDisk:balmerAlpha6565']
        line[:] = line[:] * 1e-7
        line.attrs['unitsInSI'] = 1.0  # Now stored in watts, not erg/s.
    converted = calc.calculate_diffuse_galaxy_seds(catalog, 0, [0.6565], model)
    np.testing.assert_allclose(converted.components['disk'].line_luminosity,
                               original.components['disk'].line_luminosity)
    with h5py.File(catalog, 'r+') as f:
        del f['Lightcone/Output1/nodeData/luminosityEmissionLineDisk:balmerAlpha6565'].attrs['unitsInSI']
    with pytest.raises(ValueError, match='Missing unitsInSI for emission-line'):
        calc.calculate_diffuse_galaxy_seds(catalog, 0, [0.6565], model)
