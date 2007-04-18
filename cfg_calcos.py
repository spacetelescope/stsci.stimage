from distutils.core import setup, Extension
from distutils import sysconfig
import sys

try:
    import numpy
    import numpy.numarray as nn
except ImportError:
    print "Numarray was not found. It may not be installed or it may not be on your PYTHONPATH.\n"

if numpy.__version__ < "1.0.2":
    raise SystemExit, "Numpy 1.0.2 or later required to build calcos."

"""
if sys.platform != 'win32':
    calcos_libraries = ['m']
else:
    calcos_libraries = []
"""

pythoninc = sysconfig.get_python_inc()
numpyinc = numpy.get_include()
numpynumarrayinc = nn.get_numarray_include_dirs()

CALCOS_EXTENSIONS = [Extension("calcos.ccos", ["src/ccos.c"],
           define_macros = [('NUMPY', '1')],
           include_dirs = [pythoninc, numpyinc, numpynumarrayinc])]


