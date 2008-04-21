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

NDIMAGE_EXTENSIONS = [

    Extension("ndimage._nd_image", \
              ["ndimage/src/nd_image.c",
               "ndimage/src/ni_filters.c",
               "ndimage/src/ni_fourier.c",
               "ndimage/src/ni_interpolation.c",
               "ndimage/src/ni_measure.c",
               "ndimage/src/ni_morphology.c",
               "ndimage/src/ni_support.c"],
               include_dirs=['ndimage/src']+[pythoninc,numpyinc]
               ),

    Extension('ndimage._segment', \
              ['ndimage/src/segment/Segmenter_EXT.c',
               'ndimage/src/segment/Segmenter_IMPL.c'],
               include_dirs=['ndimage/src']+[pythoninc,numpyinc],
               depends = ['ndimage/src/segment/ndImage_Segmenter_structs.h']
               ),

    Extension('ndimage._register', \
              ['ndimage/src/register/Register_EXT.c',
               'ndimage/src/register/Register_IMPL.c'],
               include_dirs=['ndimage/src']+[pythoninc,numpyinc]

           )
]


