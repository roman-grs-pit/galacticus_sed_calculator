#!/usr/bin/env python3
"""Compatibility entry point for generate_catalog_seds.py."""
import sys
import generate_catalog_seds as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    # Preserve imports of public and historical helper functions.
    sys.modules[__name__] = _implementation
