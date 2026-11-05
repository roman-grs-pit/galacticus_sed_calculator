"""Read line luminosities through legacy and modern HDF5 extractor metadata."""

import h5py
import numpy as np
import pytest

from galacticus_sed_calculator.sed_calculator import getLineNames, getLineProperties


@pytest.mark.parametrize('agn', ['lmnstyEmssnLineAGN', 'luminosityEmissionLineAGN'])
@pytest.mark.parametrize('metadata_bytes', [False, True])
@pytest.mark.parametrize('line_bytes', [False, True])
@pytest.mark.parametrize('single_line', [False, True])
def test_names_and_luminosities(tmp_path, agn, metadata_bytes, line_bytes, single_line):
    """Metadata representations must not change component selection or luminosities."""
    def encode(value):
        return np.bytes_(value) if metadata_bytes else value

    lines = ['balmerAlpha6565'] if single_line else ['balmerAlpha6565', 'oxygenIII5008']
    line_values = [np.bytes_(line) if line_bytes else line for line in lines]
    path = tmp_path / 'catalog.hdf5'
    with h5py.File(path, 'w') as f:
        root = f.create_group('Parameters/nodePropertyExtractor')
        nd = f.create_group('Lightcone/Output1/nodeData')
        # An unrelated extractor must not be mistaken for an emission-line one.
        root.attrs['nodePropertyExtractorUnrelated'] = encode('massStellar')
        for i, component in enumerate(('disk', 'spheroid', 'AGN')):
            key = f'nodePropertyExtractor{i}'
            root.attrs[key] = encode(agn if component == 'AGN' else 'luminosityEmissionLine')
            group = root.create_group(key)
            if component != 'AGN':
                group.attrs['component'] = encode(component)
            group.attrs['lineNames'] = line_values[0] if single_line else line_values
            for j, line in enumerate(lines):
                name = f'luminosityEmissionLine{component[0].upper() + component[1:]}:{line}'
                nd[name] = [10.0 * i + j + 1, 10.0 * i + j + 101]

    for i, component in enumerate(('disk', 'spheroid', 'AGN')):
        names, wavelengths, luminosities = getLineProperties(path, component, galIndex=1)
        np.testing.assert_array_equal(names, lines)
        np.testing.assert_array_equal(wavelengths, [6565] if single_line else [6565, 5008])
        np.testing.assert_array_equal(luminosities, [10.0 * i + j + 101 for j in range(len(lines))])


@pytest.mark.parametrize('component', ['disk', 'spheroid', 'AGN'])
def test_missing_component_has_explanatory_error(tmp_path, component):
    with h5py.File(tmp_path / 'missing.hdf5', 'w') as f:
        root = f.create_group('Parameters/nodePropertyExtractor')
        root.attrs['unrelated'] = 'massStellar'
        with pytest.raises(ValueError, match=f'component {component!r}'):
            getLineNames(f, component)
