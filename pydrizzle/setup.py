from distutils.core import setup, Extension
import numarray
from numarray.numarrayext import NumarrayExtension
import sys

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later required to build imagestats."

def dolocal():
    """Adds a command line option --local=<install-dir> which is an abbreviation for
    'put all of pydrizzle in <install-dir>/pydrizzle'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir = a.split("=")[1]
            sys.argv.extend([
                "--install-lib="+dir,
                ])
            sys.argv.remove(a)

def getF2CDirs():
    """ Defines the location of the F2C include and library directories. """

    platform = sys.platform[:5]
    if platform == 'linux':
        f2c_include  = "/data/chulak1/local/include"
        f2c_lib      = "/data/chulak1/local/lib"
    elif platform == 'sunos':
        f2c_include  = "/usr/ra/f2c"
        f2c_lib      = "/usr/ra/f2c"
    else:
        print 'ERROR: Unsupported platform: ',sys.platform
        print 'ERROR: No supported version of F2C available!'
        raise ValueError

    return f2c_include,f2c_lib

def getExtensions(f2cdirs):
    ext = [NumarrayExtension("arrdriz",['src/arrdrizmodule.c','src/tdriz.c','src/tblot.c',
                                'src/drutil.c','src/doblot.c','src/drcall.c',
                                'src/inter2d.c','src/bieval.c'],
                   include_dirs=[f2cdirs[0]],
                   library_dirs=[f2cdirs[1]],
                   libraries=['f2c','m'])]

    return ext


def dosetup(ext):
    r = setup(name = "pydrizzle",
              version = "5.0.17",
              description = "Geometrically correct and combine images using Drizzle",
              author = "Warren Hack",
              author_email = "help@stsci.edu",
              license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris"],
              packages=['pydrizzle','pydrizzle/traits102'],
              package_dir={'pydrizzle':'lib','pydrizzle/traits102':'traits102'},
              ext_modules=ext)
    return r


def main():
    args = sys.argv
    dolocal()
    f2cdirs = getF2CDirs()
    ext = getExtensions(f2cdirs)
    dosetup(ext)


if __name__ == "__main__":
    main()
