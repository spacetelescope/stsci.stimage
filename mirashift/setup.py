from distutils.core import setup, Extension
from distutils import sysconfig
from distutils.command.install_data import install_data
import sys, os.path

package_name = 'mirashift'
if not hasattr(sys, 'version_info') or sys.version_info < (2,2,0,'alpha',0):
    raise SystemExit, "Python 2.2 or later required to build %s."%(package_name)

#pythonlib = sysconfig.get_python_lib(plat_specific=1)
pythoninc = sysconfig.get_python_inc()
try:
    import numarray
    from numarray.numarrayext import NumarrayExtension
except:
    raise ImportError("Numarray was not found. It may not be installed or it may not be on your PYTHONPATH\n")

if numarray.__version__ < "1.1":
    raise SystemExit, "Numarray 1.1 or later required to build %s."%(package_name)

if sys.platform != 'win32':
    mirashift_libs = ['m']
else:
    mirashift_libs = []


args = sys.argv[:]
for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
        sys.argv.extend([
                "--install-lib="+dir,
                ])
        #remove --local from both sys.argv and args
        args.remove(a)
        sys.argv.remove(a)

class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)


def getExtensions(args):
    ext = [NumarrayExtension("mirashift.chainMoments",['src/chainMoments.c','src/expandArray.c'],
                             include_dirs=[pythoninc],
                             libraries=mirashift_libs),
           NumarrayExtension("mirashift.expandArray", ['src/expandArray.c'],
                             include_dirs=[pythoninc],
                             libraries=mirashift_libs)]

    return ext


def dosetup(ext):
    r = setup(name = package_name,
              version = "0.1alpha",
              description = "Compute offsets between images",
              author = "Warren Hack, Nadezhda Dencheva",
              author_email = "help@stsci.edu",
              license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris","Mac OS X", "Windows"],
              packages=[package_name],
              package_dir={package_name:'lib'},
              cmdclass = {'install_data':smart_install_data},
              data_files = [(package_name,['lib/LICENSE.txt'])],
              ext_modules=ext)
    return r


if __name__ == "__main__":
    ext = getExtensions(args)
    dosetup(ext)


