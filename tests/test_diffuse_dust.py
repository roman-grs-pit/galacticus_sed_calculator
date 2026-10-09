"""Offline analytic tests, plus opt-in tests of the canonical external atlas."""

import hashlib
import multiprocessing
import os
import pickle
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galacticus_sed_calculator import diffuse_dust as dust


def analytic_attenuation(wavelength, inclination, tau, ratio=None):
    # Cross terms make this genuinely multilinear in transformed coordinates.
    x, y, z = np.log(wavelength), inclination / 90, np.log(tau)
    result = 0.4 + 0.12*x + 0.21*y + 0.17*z + 0.04*x*z - 0.02*y*z
    if ratio is not None:
        r = np.log(ratio)
        result = result + 0.15*r + 0.01*x*r + 0.02*y*z*r
    return result


@pytest.fixture
def atlas(tmp_path, monkeypatch):
    """Tiny fixture replaces ONLY the canonical fingerprint in these tests."""
    path = tmp_path / "atlas.hdf5"
    w = np.array([0.1, 0.3, 1.0, 3.0])
    i = np.array([0., 30., 60., 90.])
    t = np.array([0., 0.01, 0.1, 1., 10.])
    r = np.array([0.01, 0.1, 1., 10.])
    disk = np.ones((len(w), len(i), len(t)))
    sph = np.ones((*disk.shape, len(r)))
    disk[:, :, 1:] = 10 ** (-0.4 * analytic_attenuation(
        w[:, None, None], i[None, :, None], t[None, None, 1:]))
    sph[:, :, 1:, :] = 10 ** (-0.4 * analytic_attenuation(
        w[:, None, None, None], i[None, :, None, None],
        t[None, None, 1:, None], r[None, None, None, :]))
    with h5py.File(path, "w") as f:
        for name, value in zip(("wavelength", "inclination", "opticalDepth", "spheroidScaleRadial"), (w, i, t, r)):
            f[name] = value
        f["attenuationDisk"] = disk
        f["attenuationSpheroid"] = sph
        f["attenuationUncertaintyDisk"] = np.zeros_like(disk)
        f["attenuationUncertaintySpheroid"] = np.zeros_like(sph)
        # Coefficient interpolation has an independently calculable answer.
        c0 = -1 + 0.1*np.log(w[:, None]) + 0.2*i[None, :]/90
        c1 = np.broadcast_to(-0.7, c0.shape)
        f["extrapolationCoefficientsDisk"] = np.stack((c0, c1))
        f["extrapolationCoefficientsSpheroid"] = np.stack((
            c0[:, :, None] + 0.1*np.log(r[None, None, :]),
            np.broadcast_to(c1[:, :, None], sph.shape[:2] + (len(r),)),
        ))
        f.attrs.update(opacity=32062.2129019, diskScaleVertical="0.137",
                       dustScaleVertical="0.137", diskCutOff="10", spheroidCutOff="10",
                       dustDescription="Weingartner & Draine (2001) R_V=4.0",
                       diskStructureVertical="sechSquared")
    def accept_fixture():
        monkeypatch.setattr(dust, "BENSON2018_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
        monkeypatch.setattr(dust, "BENSON2018_SIZE", path.stat().st_size)
    accept_fixture()
    return path, accept_fixture


@pytest.mark.parametrize("source", ["disk", "spheroid"])
def test_multilinear_attenuation_and_unsorted_wavelength_shapes(atlas, source):
    wave = np.array([[2.1, 0.17], [0.4, 0.17]])
    ratio = 0.037
    with dust.Benson2018DiffuseProvider(atlas[0]) as model:
        result = model.evaluate(wave, source=source, inclination_degrees=47,
                                optical_depth_v=0.37, spheroid_scale_ratio=ratio)
    expected = 10 ** (-0.4*analytic_attenuation(wave, 47, 0.37, ratio if source == "spheroid" else None))
    np.testing.assert_allclose(result.transmission, expected, rtol=2e-15)
    assert not result.flags


@pytest.mark.parametrize("source", ["disk", "spheroid"])
def test_exact_grid_nodes_and_scattering_above_unity(atlas, source):
    with dust.Benson2018DiffuseProvider(atlas[0]) as model, h5py.File(atlas[0]) as f:
        for angle in f["inclination"][:]:
            for depth in f["opticalDepth"][1:]:
                result = model.evaluate(f["wavelength"][:], source=source,
                                        inclination_degrees=angle, optical_depth_v=depth,
                                        spheroid_scale_ratio=0.01)
                expected = 10 ** (-0.4*analytic_attenuation(
                    f["wavelength"][:], angle, depth, 0.01 if source == "spheroid" else None))
                np.testing.assert_allclose(result.transmission, expected, rtol=2e-15)
        bright = model.evaluate(0.1, source=source, inclination_degrees=0,
                                optical_depth_v=0.01, spheroid_scale_ratio=0.01)
        assert bright.transmission > 1
        assert bright.transmission.shape == ()


def test_zero_and_low_tau(atlas):
    with dust.Benson2018DiffuseProvider(atlas[0]) as model:
        zero = model.evaluate([0.2, 0.6], source="spheroid", inclination_degrees=90,
                              optical_depth_v=0)  # No source size required.
        np.testing.assert_array_equal(zero.transmission, [1, 1])
        edge = model.evaluate([0.2, 0.6], source="disk", inclination_degrees=20,
                              optical_depth_v=0.01)
        low = model.evaluate([0.2, 0.6], source="disk", inclination_degrees=20,
                             optical_depth_v=0.0025)
        np.testing.assert_allclose(low.transmission, 1 + 0.25*(edge.transmission - 1))
        assert low.flags == {"low_optical_depth_blend"}


def test_strict_and_clipped_bounds(atlas):
    with dust.Benson2018DiffuseProvider(atlas[0]) as model:
        with pytest.raises(ValueError, match="high_tau_policy"):
            model.evaluate(0.5, source="disk", inclination_degrees=20, optical_depth_v=100)
        with pytest.raises(ValueError, match="scale_ratio outside"):
            model.evaluate(0.5, source="spheroid", inclination_degrees=20,
                           optical_depth_v=1, spheroid_scale_ratio=20)
    with dust.Benson2018DiffuseProvider(atlas[0], high_tau_policy="clip", scale_ratio_policy="clip") as model:
        clipped = model.evaluate(0.5, source="spheroid", inclination_degrees=20,
                                 optical_depth_v=100, spheroid_scale_ratio=20)
        edge = model.evaluate(0.5, source="spheroid", inclination_degrees=20,
                              optical_depth_v=10, spheroid_scale_ratio=10)
        np.testing.assert_array_equal(clipped.transmission, edge.transmission)
        assert clipped.flags == {"optical_depth_clipped", "scale_ratio_clipped"}
        assert clipped.coordinates["optical_depth_v"] == 100
        assert clipped.coordinates["optical_depth_v_used"] == 10
        assert clipped.coordinates["spheroid_scale_ratio"] == 20
        assert clipped.coordinates["spheroid_scale_ratio_used"] == 10


@pytest.mark.parametrize("source", ["disk", "spheroid"])
def test_atlas_extrapolation(atlas, source):
    wave = np.array([0.15, 0.7, 2.5])
    with dust.Benson2018DiffuseProvider(atlas[0], high_tau_policy="atlas_extrapolation") as model:
        result = model.evaluate(wave, source=source, inclination_degrees=45,
                                optical_depth_v=100, spheroid_scale_ratio=0.07)
    c0 = -1 + 0.1*np.log(wave) + 0.1
    if source == "spheroid":
        c0 += 0.1*np.log(0.07)
    np.testing.assert_allclose(result.transmission, np.exp(c0 - 0.7*np.log(100)), rtol=2e-15)
    assert result.flags == {"optical_depth_extrapolated"}


def test_positive_extrapolation_slope_is_flagged(atlas):
    path, accept = atlas
    with h5py.File(path, "r+") as f:
        f["extrapolationCoefficientsDisk"][1] = 0.5
    accept()
    with dust.Benson2018DiffuseProvider(path, high_tau_policy="atlas_extrapolation") as model:
        result = model.evaluate(0.5, source="disk", inclination_degrees=40, optical_depth_v=100)
        assert "extrapolation_positive_slope" in result.flags
        assert result.transmission > 1  # Do not silently clip the supplied fit.


@pytest.mark.parametrize("kwargs,match", [
    ({"source": "agn"}, "unsupported"),
    ({"inclination_degrees": -1}, "inclination_degrees"),
    ({"inclination_degrees": 91}, "inclination_degrees"),
    ({"inclination_degrees": [1, 2]}, "scalar"),
    ({"optical_depth_v": -1}, "nonnegative"),
    ({"optical_depth_v": np.nan}, "finite"),
    ({"source": "spheroid"}, "require spheroid_scale_ratio"),
    ({"source": "spheroid", "spheroid_scale_ratio": 0}, "positive"),
    ({"source": "spheroid", "spheroid_scale_ratio": np.inf}, "finite"),
    ({"wavelength_micron": 3.1}, "coverage"),
    ({"wavelength_micron": 0.09}, "coverage"),
    ({"wavelength_micron": np.nan}, "finite"),
    ({"wavelength_micron": 0}, "positive"),
])
def test_invalid_queries(atlas, kwargs, match):
    query = dict(wavelength_micron=0.5, source="disk", inclination_degrees=30, optical_depth_v=1)
    query.update(kwargs)
    with dust.Benson2018DiffuseProvider(atlas[0]) as model, pytest.raises(ValueError, match=match):
        model.evaluate(**query)


def test_endpoints_empty_wavelengths_and_lifecycle(atlas):
    model = dust.Benson2018DiffuseProvider(atlas[0])
    rounded = model.evaluate(3*(1+1e-8), source="disk", inclination_degrees=0, optical_depth_v=1)
    assert rounded.flags == {"wavelength_endpoint_roundoff"}
    assert model.evaluate([], source="disk", inclination_degrees=0, optical_depth_v=1).transmission.shape == (0,)
    assert str(atlas[0]) not in str(model.provenance)
    with pytest.raises(TypeError, match="workers"):
        pickle.dumps(model)
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.evaluate(0.5, source="disk", inclination_degrees=0, optical_depth_v=0)


def test_worker_receipt_avoids_rehash_and_checks_mutation(atlas, monkeypatch):
    receipt = pickle.loads(pickle.dumps(dust.verify_benson_atlas(atlas[0])))
    def no_rehash(*args):
        pytest.fail("Worker rehashed the atlas")
    monkeypatch.setattr(dust, "verify_benson_atlas", no_rehash)
    with dust.Benson2018DiffuseProvider(receipt) as model:
        model._pid -= 1
        with pytest.raises(RuntimeError, match="each process"):
            model.evaluate(0.5, source="disk", inclination_degrees=0, optical_depth_v=1)
    stat = atlas[0].stat()
    os.utime(atlas[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="changed since"):
        dust.Benson2018DiffuseProvider(receipt)


def test_checksum_and_missing_file_errors(atlas, monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="zenodo.org"):
        dust.verify_benson_atlas(tmp_path / "missing.hdf5")
    monkeypatch.setattr(dust, "BENSON2018_SHA256", "0"*64)
    with pytest.raises(ValueError, match="SHA-256"):
        dust.verify_benson_atlas(atlas[0])
    monkeypatch.setattr(dust, "BENSON2018_SIZE", 2)
    with pytest.raises(ValueError, match="size"):
        dust.verify_benson_atlas(atlas[0])


@pytest.mark.parametrize("mutation,match", [
    ("missing", "Missing atlas axis"),
    ("order", "strictly increasing"),
    ("shape", "dataset/shape"),
    ("opacity", "attribute: opacity"),
    ("grain", "attribute: dustDescription"),
])
def test_schema_validation(atlas, mutation, match):
    path, accept = atlas
    with h5py.File(path, "r+") as f:
        if mutation == "missing":
            del f["wavelength"]
        elif mutation == "order":
            f["opticalDepth"][2] = f["opticalDepth"][1]
        elif mutation == "shape":
            del f["attenuationSpheroid"]
            f["attenuationSpheroid"] = np.ones((2, 2))
        elif mutation == "opacity":
            f.attrs["opacity"] = 1
        else:
            f.attrs["dustDescription"] = "R_V=3.1"
    accept()
    with pytest.raises(ValueError, match=match):
        dust.Benson2018DiffuseProvider(path)


@pytest.mark.parametrize("value", [0., -1., np.nan])
def test_bad_selected_transmission_is_not_silently_floored(atlas, value):
    path, accept = atlas
    with h5py.File(path, "r+") as f:
        f["attenuationDisk"][0, 0, 1] = value
    accept()
    with dust.Benson2018DiffuseProvider(path) as model:
        with pytest.raises(ValueError, match="positive|Nonfinite"):
            model.evaluate(0.1, source="disk", inclination_degrees=0, optical_depth_v=0.01)
        # An invalid, zero-weight neighbour must not poison an exact-node query.
        result = model.evaluate(0.3, source="disk", inclination_degrees=0, optical_depth_v=0.01)
        assert np.isfinite(result.transmission)


def test_hyperslab_reads_are_small_and_never_load_uncertainties(atlas, monkeypatch):
    original = h5py.Dataset.__getitem__
    reads = []
    def record(dataset, selection):
        value = original(dataset, selection)
        if "attenuation" in dataset.name or "extrapolation" in dataset.name:
            reads.append((dataset.name, np.asarray(value).size))
        return value
    monkeypatch.setattr(h5py.Dataset, "__getitem__", record)
    with dust.Benson2018DiffuseProvider(atlas[0]) as model:
        model.evaluate(np.geomspace(0.1, 3, 100), source="spheroid", inclination_degrees=45,
                       optical_depth_v=0.3, spheroid_scale_ratio=0.3)
    assert len(reads) == 1
    assert reads[0][0] == "/attenuationSpheroid"
    assert reads[0][1] == 4*2*2*2


@pytest.mark.skipif(not os.environ.get("BENSON2018_ATLAS"), reason="Set BENSON2018_ATLAS for canonical integration checks")
def test_canonical_atlas():
    path = os.environ["BENSON2018_ATLAS"]
    with dust.Benson2018DiffuseProvider(path) as model, h5py.File(path) as f:
        wave, angle, tau, ratio = (f["wavelength"][100], f["inclination"][10],
                                   f["opticalDepth"][20], f["spheroidScaleRadial"][25])
        disk = model.evaluate(wave, source="disk", inclination_degrees=angle, optical_depth_v=tau)
        sph = model.evaluate(wave, source="spheroid", inclination_degrees=angle,
                             optical_depth_v=tau, spheroid_scale_ratio=ratio)
        assert disk.transmission == pytest.approx(0.8420483727300836, rel=1e-14)
        assert sph.transmission == pytest.approx(0.6898254998576951, rel=1e-14)
        # Exercise actual corners at low and high depths without full-table copies.
        for ratio in [0.001, 100.]:
            result = model.evaluate([0.01, 0.55, 3.0], source="spheroid", inclination_degrees=90,
                                    optical_depth_v=10000, spheroid_scale_ratio=ratio)
            assert np.all(np.isfinite(result.transmission))


def _canonical_worker(arguments):
    receipt, source = arguments
    with dust.Benson2018DiffuseProvider(receipt) as model:
        return model.evaluate([0.15, 0.55, 2.0], source=source, inclination_degrees=63,
                              optical_depth_v=1.7, spheroid_scale_ratio=0.23).transmission


@pytest.mark.skipif(not os.environ.get("BENSON2018_ATLAS"), reason="Set BENSON2018_ATLAS for canonical multiprocessing check")
def test_canonical_spawned_workers():
    receipt = dust.verify_benson_atlas(os.environ["BENSON2018_ATLAS"])
    requests = [(receipt, source) for source in ("disk", "spheroid")]
    expected = [_canonical_worker(request) for request in requests]
    with multiprocessing.get_context("spawn").Pool(2) as pool:
        actual = pool.map(_canonical_worker, requests)
    np.testing.assert_array_equal(actual, expected)
