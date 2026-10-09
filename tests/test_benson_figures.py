"""Opt-in reproducibility smoke test for the scientific documentation figures."""

import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.environ.get("BENSON2018_ATLAS"), reason="Set BENSON2018_ATLAS to reproduce figures")
def test_canonical_figures(tmp_path):
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("Plotting requires optional matplotlib dependency")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "figures"
    subprocess.run(
        [sys.executable, str(root / "examples/plot_benson_diffuse.py"),
         os.environ["BENSON2018_ATLAS"], "--output-dir", str(output)],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=120,
        env=dict(os.environ, PYTHONPATH=str(root), MPLCONFIGDIR=str(tmp_path / "mpl")),
    )
    expected = {f"benson_diffuse_{name}.png" for name in ("spectra", "geometry", "interpolation")}
    assert {path.name for path in output.iterdir()} == expected
    for path in output.iterdir():
        # PNG signature and IHDR dimensions; no additional imaging dependency.
        with path.open("rb") as stream:
            header = stream.read(24)
        assert header[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", header[16:24]) == (1760, 768)
