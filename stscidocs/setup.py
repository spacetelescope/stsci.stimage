#!/usr/bin/env python

from __future__ import division # confidence high

try :
    import pytools.stsci_distutils_hack
except ImportError, e:
    print ""
    print e
    print ""
    print "You should install stsci_python first.  If you already did, something is wrong"
    print "with your PYTHONPATH"
    print ""
    import sys
    sys.exit(1)

pytools.stsci_distutils_hack.run(pytools_version = "3.0")
