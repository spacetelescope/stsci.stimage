from distutils.core import setup, Extension
from distutils import sysconfig
import sys

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'final', 0):
    raise SystemExit, "Python 2.3 or later required to build imagestats."

import numpy

if numpy.__version__ < "1.0.2":
    raise SystemExit, "Numpy 1.0.2 or later required to build imagestats."

pythoninc = sysconfig.get_python_inc()
numpyinc = numpy.get_include()

IMAGE_EXTENSIONS = [
    Extension('image._combine', \
              ["image/src/_combinemodule.c"],
              include_dirs = [numpyinc,pythoninc]+[numpy.get_numarray_include()]
          )
]


