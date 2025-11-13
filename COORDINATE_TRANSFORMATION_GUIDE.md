# RA and Dec Coordinate Transformation - Usage Examples

This document provides detailed examples of how to use the new RA and Dec coordinate transformation functionality.

## Overview

The `calculate_catalog_coordinates.py` script converts Galacticus lightcone angular coordinates (theta, phi) to astronomical Right Ascension (RA) and Declination (Dec) coordinates. This allows you to:

- Reposition the field center to any location on the sky
- Apply a roll angle to rotate the field around the line of sight
- Save coordinates to the catalog or to a separate file
- Maintain all transformation parameters as metadata

## Quick Start

### Basic Usage

Convert coordinates with default field center at north pole:

```bash
python calculate_catalog_coordinates.py data/galacticus_lightcone.hdf5
```

This creates a new file `data/galacticus_lightcone_with_coordinates.hdf5` with RA and Dec datasets added.

### Reposition Field Center

Place the field center at a specific location on the sky:

```bash
python calculate_catalog_coordinates.py data/galacticus_lightcone.hdf5 \
    --ra0 150.5 --dec0 30.2
```

### Apply Roll Angle

Rotate the field by 45 degrees around the line of sight:

```bash
python calculate_catalog_coordinates.py data/galacticus_lightcone.hdf5 \
    --ra0 150.5 --dec0 30.2 --roll 45
```

### Save to Separate File

Save coordinates to a standalone file instead of modifying the catalog:

```bash
python calculate_catalog_coordinates.py data/galacticus_lightcone.hdf5 \
    --ra0 150 --dec0 30 \
    --save-to-file coordinates.hdf5
```

## Python API

You can also use the coordinate transformation directly in Python:

```python
from calculate_catalog_coordinates import convert_lightcone_to_radec
import numpy as np

# Your lightcone angular coordinates
theta = np.array([0.1, 0.5, 1.0])  # degrees from field center
phi = np.array([0, 90, 180])        # azimuthal angle

# Convert to RA/Dec with field center at (150°, 30°)
ra, dec = convert_lightcone_to_radec(
    theta, phi,
    ra0=150.0,   # Field center RA
    dec0=30.0,   # Field center Dec
    roll=0.0     # Roll angle
)

print(f"RA: {ra}")
print(f"Dec: {dec}")
```

### Process Entire Catalog

```python
from calculate_catalog_coordinates import calculate_catalog_coordinates

results = calculate_catalog_coordinates(
    galacticus_catalog='data/galacticus_lightcone.hdf5',
    ra0=150.0,
    dec0=30.0,
    roll=0.0,
    save_to_input=True,
    copy_input=True
)

# Results contain:
# - 'ra': array of RA values
# - 'dec': array of Dec values
# - 'theta': original theta values
# - 'phi': original phi values
# - 'ra0', 'dec0', 'roll': transformation parameters
```

## Coordinate System Details

### Original Lightcone Coordinates

- **theta**: Polar angle from field center (0 to ~few degrees)
- **phi**: Azimuthal angle (0 to 360 degrees)
- **Field center**: Originally at theta=0, pointing along +z axis

### Transformed Coordinates

- **RA**: Right Ascension (0 to 360 degrees)
- **Dec**: Declination (-90 to 90 degrees)
- **Field center**: Can be placed anywhere on the sky

### Transformation Process

1. Convert (theta, phi) to Cartesian coordinates on unit sphere
2. Apply roll rotation around z-axis (line of sight)
3. Rotate to move field center from (RA=0°, Dec=90°) to (RA0, Dec0)
4. Convert back to spherical (RA, Dec) coordinates

## Output Format

### When Saving to Catalog

Two new datasets are added to `/Lightcone/Output1/nodeData/`:

- `rightAscension`: RA in degrees (0 to 360)
- `declination`: Dec in degrees (-90 to 90)

Each dataset includes attributes:
- `comment`: Description of the coordinate
- `units`: "degrees"
- `fieldCenterRA`: RA of field center
- `fieldCenterDec`: Dec of field center
- `rollAngle`: Roll angle applied

### When Saving to Separate File

The output HDF5 file contains:

**Datasets:**
- `ra`: Right Ascension array
- `dec`: Declination array
- `theta`: Original theta values
- `phi`: Original phi values
- `galaxy_indices`: Galaxy indices processed

**Attributes:**
- `fieldCenterRA`: Field center RA
- `fieldCenterDec`: Field center Dec
- `rollAngle`: Roll angle applied
- `description`: Text description

## Examples

### Example 1: Roman WFI Survey Field

Position a 4 square degree field at a typical Roman survey location:

```bash
python calculate_catalog_coordinates.py data/romanUNIT.hdf5 \
    --ra0 270.0 --dec0 -30.0 --roll 0
```

### Example 2: Multiple Fields

Process the same catalog with different field positions:

```bash
# Field 1
python calculate_catalog_coordinates.py data/romanUNIT.hdf5 \
    --ra0 150 --dec0 2 \
    --save-to-file field1_coords.hdf5

# Field 2
python calculate_catalog_coordinates.py data/romanUNIT.hdf5 \
    --ra0 180 --dec0 -15 \
    --save-to-file field2_coords.hdf5
```

### Example 3: Testing with Limited Galaxies

Process only the first 100 galaxies for testing:

```bash
python calculate_catalog_coordinates.py data/romanUNIT.hdf5 \
    --ra0 100 --dec0 25 \
    --max-galaxies 100
```

## Visualization

After calculating coordinates, you can visualize them:

```python
import matplotlib.pyplot as plt
import h5py

with h5py.File('galacticus_lightcone_with_coordinates.hdf5', 'r') as f:
    ra = f['/Lightcone/Output1/nodeData/rightAscension'][:]
    dec = f['/Lightcone/Output1/nodeData/declination'][:]

plt.figure(figsize=(10, 8))
plt.scatter(ra, dec, s=1, alpha=0.5)
plt.xlabel('RA (degrees)')
plt.ylabel('Dec (degrees)')
plt.title('Galaxy Positions on Sky')
plt.grid(True, alpha=0.3)
plt.show()
```

## Notes

- The script currently only supports lightcone format files (not fixed-time format)
- Original theta/phi values are preserved in the catalog
- All transformation parameters are saved as metadata
- The transformation is reversible (you can always go back to theta/phi)
- RA is always in range [0, 360), Dec in range [-90, 90]

## Testing

Run the test suite:

```bash
python -m unittest test_coordinates
```

All 15 tests should pass, covering:
- Coordinate transformations
- Field center repositioning
- Roll angle application
- Edge cases and boundary conditions
- Symmetry properties
