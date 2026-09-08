# Tested environment

The installation and catalog-magnitude quick start were tested from a clean
Python 3.10 environment on 2026-09-08 with these direct dependencies:

| Package | Version |
| --- | ---: |
| Python | 3.10.21 |
| NumPy | 2.2.6 |
| h5py | 3.16.0 |
| Astropy | 6.1.7 |
| SciPy | 1.15.3 |
| synphot | 1.7.0 |
| PyYAML | 6.0.3 |
| STPSF | 2.2.0 |

These versions are a record of a known-working environment, not mandatory
pins. The package does not yet specify minimum versions for its scientific
Python dependencies because the oldest compatible combinations have not been
systematically tested. Continuous integration tests current compatible
releases on Python 3.10 through 3.13.
