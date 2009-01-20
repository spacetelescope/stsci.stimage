import sys
import distutils
import distutils.core

try:
    import numpy
except:
    raise ImportError("NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH")

pythoninc = distutils.sysconfig.get_python_inc()
numpyinc = numpy.get_include()

ext = [ distutils.core.Extension('opuscoords.GCcoords',['src/GCcoords_module.c','src/gc_coords_pkg.c'],
                 include_dirs = [pythoninc,numpyinc])
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
      'data_files' : 	[(pkg,['lib/SP_LICENSE'])],
      'ext_modules' :   ext,
    }

