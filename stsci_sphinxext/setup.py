#!/usr/bin/env python
from __future__ import division # confidence high

try:
    from pytools import stsci_distutils_hack
except ImportError:
    import stsci_distutils_hack
stsci_distutils_hack.run(pytools_version = "3.0")
