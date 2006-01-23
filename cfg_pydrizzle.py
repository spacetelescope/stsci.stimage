from distutils import sysconfig
from distutils.core import Extension
import sys, os.path, string
f2c_inc_dir = []
f2c_lib_dir = []

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later required to build pydrizzle."

try:
    import numarray
    from numarray.numarrayext import NumarrayExtension
except:
    raise ImportError("Numarray was not found. It may not be installed or it may not be on your PYTHONPATH. Pydrizzle requires numarray v 1.1 or later.\n")
    
if numarray.__version__ < "1.1":
    raise SystemExit, "Numarray 1.1 or later required to build pydrizzle."

pythoninc = sysconfig.get_python_inc()
pydrizzle_inc_dirs = []
pydrizzle_lib_dirs = []
pydrizzle_inc_dirs.append(pythoninc)



def getF2CDirs(args):
    """ Defines the location of the F2C include and library directories. """
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--with-f2c=<f2c-dir> "
    for a in args:
        if string.find(a, '--with-f2c=') != -1:
            f2cdir = os.path.abspath(string.split(a, '=')[1])
            sys.argv.remove(a)
            if os.path.exists(os.path.join(f2cdir, 'f2c.h')):
                f2c_inc_dir.append(f2cdir)
            elif os.path.exists(os.path.join(f2cdir, 'include','f2c.h')):
                f2c_inc_dir.append(os.path.join(f2cdir, 'include'))
            else:
                raise SystemExit, "f2c.h needed to build pydrizzle."
            if os.path.exists(os.path.join(f2cdir, 'libf2c.a')) or \
                   os.path.exists(os.path.join(f2cdir, 'libf2c.so')):
                f2c_lib_dir.append(f2cdir)
            elif os.path.exists(os.path.join(f2cdir, 'lib','libf2c.a')) or \
                     os.path.eists(os.path.join(f2cdir, 'lib','libf2c.so')):
                f2c_lib_dir.append(os.path.join(f2cdir, 'lib'))
            else:
                raise SystemExit, "libf2c needed to build pydrizzle."

args = sys.argv[:]
getF2CDirs(args)


PYDRIZZLE_EXTENSIONS = [NumarrayExtension("pydrizzle.arrdriz", \
                ['pydrizzle/src/arrdrizmodule.c', \
                'pydrizzle/src/tdriz.c','pydrizzle/src/tblot.c', \
                'pydrizzle/src/drutil.c','pydrizzle/src/doblot.c', \
                'pydrizzle/src/drcall.c', 'pydrizzle/src/inter2d.c', \
                'pydrizzle/src/bieval.c'],
                include_dirs = [pythoninc] + f2c_inc_dir,
                library_dirs = f2c_lib_dir,
                libraries=['f2c','m'])]

