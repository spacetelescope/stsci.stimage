from distutils.core import setup
import sys, os.path, string, shutil
from distutils import sysconfig
from distutils.command.install_data import install_data

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'final',0):
    raise SystemExit, "Python 2.3 or later required to build multidrizzle."

ver = sysconfig.get_python_version()
pythonver = 'python' + ver
pythonlib = sysconfig.get_python_lib(plat_specific=1)

args = sys.argv[:]

import version
version.__set_svn_version__(fullInfo=False)

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

setup(name = "multidrizzle",
      version = "3.2.0",
      description = "Automated process for HST image combination and cosmic-ray rejection",
      author = "Warren Hack, Christopher Hanley, Ivo Busko, Robert Jedrzejewski, and Anton Koekemoer",
      author_email = "help@stsci.edu",
      license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
      platforms = ["Linux","Solaris","Mac OS X", "Windows"],
      packages=['multidrizzle'],
      package_dir={'multidrizzle':'lib'},
      cmdclass = {'install_data':smart_install_data},
      data_files = [('multidrizzle',['lib/LICENSE.txt'])],
      )



