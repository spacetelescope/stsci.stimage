from distutils.core import setup, Extension
import numarray
from numarray.numarrayext import NumarrayExtension
import sys, string, os.path, shutil

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later required to build pydrizzle."

if numarray.__version__ < "1.1":
    raise SystemExit, "Numarray 1.1 or later required to build pydrizzle."
ver = sys.version_info
python_exec = 'python' + str(ver[0]) + '.' + str(ver[1])

def dolocal():
    """Adds a command line option --local=<install-dir> which is an abbreviation for
    'put all of pydrizzle in <install-dir>/pydrizzle'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir =  os.path.abspath(a.split("=")[1])
            sys.argv.extend([
                "--install-lib="+dir,
                "--install-data="+os.path.join(dir,"pydrizzle")
                ])
            sys.argv.remove(a)

def getF2CDirs(args):
    """ Defines the location of the F2C include and library directories. """
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--with-f2c=<f2c-dir> "
    for a in args:
        if string.find(a, '--with-f2c=') != -1:
            f2cdir = string.split(a, '=')[1]
            sys.argv.remove(a)
            if os.path.exists(os.path.join(f2cdir, 'f2c.h')):
                f2c_include = f2cdir
            elif os.path.exists(os.path.join(f2cdir, 'include','f2c.h')):
                f2c_include = os.path.join(f2cdir, 'include')
            else:
                raise SystemExit, "f2c.h needed to build pydrizzle."
            if os.path.exists(os.path.join(f2cdir, 'libf2c.a')):
                f2c_lib = f2cdir
            elif os.path.exists(os.path.join(f2cdir, 'lib','libf2c.a')):
                f2c_lib = os.path.join(f2cdir, 'lib')
            else:
                raise SystemExit, "libf2c needed to build pydrizzle."
    return f2c_include,f2c_lib

def getDataDir(args):
    for a in args:
        if string.find(a, '--home=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = os.path.join(dir, 'lib/python/pydrizzle')
        elif string.find(a, '--prefix=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = os.path.join(dir, 'lib', python_exec, 'site-packages/pydrizzle')
        elif a.startswith('--install-data='):
            dir = string.split(a, '=')[1]
            data_dir = dir
        else:
            data_dir = os.path.join(sys.prefix, 'lib', python_exec, 'site-packages/pydrizzle')
    return data_dir


def getExtensions(f2cdirs,args):
    numarrayIncludeDir = './'
    for a in args:
        if a.startswith('--home='):
            numarrayIncludeDir = os.path.abspath(os.path.join(a.split('=')[1], 'include', 'python', 'numarray'))
        elif a.startswith('--prefix='):
            numarrayIncludeDir = os.path.abspath(os.path.join(a.split('=')[1], 'include','python2.3', 'numarray'))
        elif a.startswith('--local='):
            numarrayIncludeDir = os.path.abspath(a.split('=')[1])

    ext = [NumarrayExtension("pydrizzle/arrdriz",['src/arrdrizmodule.c','src/tdriz.c','src/tblot.c',
                                'src/drutil.c','src/doblot.c','src/drcall.c',
                                'src/inter2d.c','src/bieval.c'],
                   include_dirs=[numarrayIncludeDir, f2cdirs[0]],
                   library_dirs=[f2cdirs[1]],
                   libraries=['f2c','m'])]

    return ext


def dosetup(data_dir, ext):
    r = setup(name = "pydrizzle",
              version = "5.2.6",
              description = "Geometrically correct and combine images using Drizzle",
              author = "Warren Hack",
              author_email = "help@stsci.edu",
              license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris", "MacOS X"],
              packages=['pydrizzle','pydrizzle/traits102'],
              package_dir={'pydrizzle':'lib','pydrizzle/traits102':'traits102'},
              data_files = [(data_dir,['lib/LICENSE.txt'])],
              ext_modules=ext)
    return r

def copy_doc(data_dir, args):
    if 'install' in args:
        doc_dir = os.path.join(data_dir,'doc')
        if os.path.exists(doc_dir):
            try:
                shutil.rmtree(doc_dir)
            except:
                print "Error removing doc directory\n"
        shutil.copytree('doc', doc_dir)


def main():
    args = sys.argv
    dolocal()
    data_dir = getDataDir(args)
    f2cdirs = getF2CDirs(args)
    ext = getExtensions(f2cdirs,args)
    dosetup(data_dir, ext)
    copy_doc(data_dir, args)

if __name__ == "__main__":
    main()
