from distutils.core import setup, Extension
import numarray
from numarray.numarrayext import NumarrayExtension
import sys, string, os

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
            
def setF2CLibnames():
    """ Defines the name(s) of the F2C libraries to be used."""
    platform = sys.platform[:5]
    if platform.find('win') > -1:
        #  Windows Visual C 6.0 libraries
        f2c_libnames = ['libf77','libi77']
    else:
        # Unix based libraries: linux, Mac OS X, Solaris
        f2c_libnames = ['f2c','m']
    
    return f2c_libnames
    
def getF2CDirs(args,f2clibs):
    """ Defines the location of the F2C include and library directories. """
    platform = sys.platform[:5]
    if platform.find('win') ==  -1:
        f2c_libname = 'libf2c.a'
    else:
        f2c_libname = f2clibs[0]+'.lib'
	
    # Define default values for include and lib directories
    f2c_include = f2c_lib = None
    
    for a in args:
        if string.find(a, '--with-f2c=') != -1:
            f2cdir = string.split(a, '=')[1]
            sys.argv.remove(a)
            if os.path.isfile(os.path.join(f2cdir, 'f2c.h')):
                f2c_include = f2cdir
            elif os.path.isfile(os.path.join(f2cdir, 'include','f2c.h')):
                f2c_include = os.path.join(f2cdir, 'include','f2c.h')
            else:
                print "File f2c.h not found.\n"
                sys.exit(1)
            if os.path.isfile(os.path.join(f2cdir, f2c_libname)):
                f2c_lib = f2cdir
            elif os.path.isfile(os.path.join(f2cdir, 'include',f2c_libname)):
                f2c_include = os.path.join(f2cdir, 'lib',f2c_libname)
            else:
                print "Library %s not found.\n"%(f2c_libname)
                sys.exit(1)

    return f2c_include,f2c_lib

def getExtensions(f2cdirs,f2clibs):
    """ 
    F2PY Usage:
    Commands for compiling the arrdriz sharable using F2PY
    can be found in the 'compile_f2py' script and run using:
    status = os.system('pydrizzle/compile_f2py')
    ext = [NumarrayExtension("pydrizzle/arrdriz")]
    """
    ext = [NumarrayExtension("arrdriz",['src/arrdrizmodule.c','src/tdriz.c','src/tblot.c',
                                'src/drutil.c','src/doblot.c','src/drcall.c',
                                'src/inter2d.c','src/bieval.c'],
                   include_dirs=[f2cdirs[0]],
                   library_dirs=[f2cdirs[1]],
                   libraries=f2clibs)]
                             
    return ext


def dosetup(ext):
    r = setup(name = "pydrizzle",
              version = "5.0.17",
              description = "Geometrically correct and combine images using Drizzle",
              author = "Warren Hack",
              author_email = "help@stsci.edu",
              license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris","Mac OS X","Win32"],
              packages=['pydrizzle','pydrizzle/traits102'],
              package_dir={'pydrizzle':'lib','pydrizzle/traits102':'traits102'},
              ext_modules=ext)
    return r


def main():
    args = sys.argv
    dolocal()
    f2clibs = setF2CLibnames()
    f2cdirs = getF2CDirs(args,f2clibs)
    ext = getExtensions(f2cdirs,f2clibs)
    dosetup(ext)


if __name__ == "__main__":
    main()

