from distutils.core import setup, Extension
import numarray
from numarray.numarrayext import NumarrayExtension
import sys

if not hasattr(sys, 'version_info') or sys.version_info < (2,2,0,'alpha',0):
    raise SystemExit, "Python 2.2 or later required to build numarray."

setup(name = "arrdriz",
      version = "0.5",
      description = "",
      packages=[""],
      package_dir={"":""},
      ext_modules=[NumarrayExtension("arrdriz",['arrdrizmodule.c','tdriz.c','tblot.c','drutil.c','doblot.c',
                                                'drcall.c','inter2d.c','bieval.c'],
                   include_dirs=["/usr/ra/f2c"],
                   library_dirs=["/usr/ra/f2c"],
                   libraries=['f2c','m'])]
      )
