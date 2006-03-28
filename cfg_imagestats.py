from distutils import sysconfig
import sys

if not hasattr(sys, 'version_info') or sys.version_info < (2,2,0,'alpha',0):
    raise SystemExit, "Python 2.2 or later required to build imagestats."
try:
    import numarray
    from numarray.numarrayext import NumarrayExtension
except:
    raise ImportError("Numarray was not found. It may not be installed or it may not be on your PYTHONPATH. Imagestats requires numarray v 1.1 or later.\n")

if numarray.__version__ < "1.1":
    raise SystemExit, "Numarray 1.1 or later required to build imagestats."

if sys.platform != 'win32':
    imagestats_libraries = ['m']
else:
    imagestats_libraries = []

pythoninc = sysconfig.get_python_inc()


IMAGESTATS_EXTENSIONS = [NumarrayExtension('imagestats.buildHistogram', \
                        ['imagestats/src/buildHistogram.c'],
                        include_dirs = [pythoninc],
                        libraries = imagestats_libraries),
                        NumarrayExtension('imagestats.computeMean', \
                        ['imagestats/src/computeMean.c'],
                        include_dirs = [pythoninc],
                        libraries = imagestats_libraries)
                        ]


