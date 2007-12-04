from distutils.core import setup
import sys, os.path
from distutils.command.install_data import install_data

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
    raise SystemExit, "Python 2.3 or later required to build nictools."

args = sys.argv[:]

for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
        sys.argv.extend([
                "--install-lib="+dir,
                "--install-scripts=%s" % dir])
        args.remove(a)
        sys.argv.remove(a)

class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

setup(name = "nictools",
      version = "1.0.0",
      description = "Python Tools for NICMOS Data",
      author = "Vicki Laidler, David Grumm",
      author_email = "help@stsci.edu",
      license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
      platforms = ["Linux","Solaris","Mac OS X"],
      packages = ['nictools'],
      package_dir={'nictools':'lib'},
      cmdclass = {'install_data':smart_install_data},
      data_files = [('nictools',['lib/SP_LICENSE'])],
      )



