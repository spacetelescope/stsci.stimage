from __future__ import division # confidence high

import sys
import distutils
import distutils.core

try:
    import numpy
except:
    raise ImportError("NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH")

numpyinc = numpy.get_include()

ext = [ distutils.core.Extension('opuscoords.GCcoords',['src/GCcoords_module.c','src/gc_coords_pkg.c'],
                 include_dirs = [numpyinc])
    ]

pkg = "opuscoords"

setupargs = {
    'version' : 		"1.0.0",
    'description' : 	"Python Tools for OPUS Coordinate Conversions",
    'author' : 		"Warren J. Hack",
    'author_email' : 	"help@stsci.edu",
    'license' : 		"http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'platforms' : 	["Linux","Solaris","Mac OS X"],
    'scripts':        [],
    'data_files' : 	[('opuscoords',['SP_LICENSE'])],
    'package_dir' :     { 'opuscoords' : 'lib/opuscoords', },
    'ext_modules' :   ext,
    }

