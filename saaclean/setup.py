from distutils.core import setup
import sys, os.path 
from distutils import sysconfig
from distutils.command.install_data import install_data

pythonlib = sysconfig.get_python_lib()
ver = sysconfig.get_python_version()
pythonver = 'python' + ver


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


if __name__ == "__main__":
    setup(name = "saaclean",
          version = "0.5",
          description = "Estimates and removes persistent CR signal due to a prior SAA passage.",
          author = "Vicki Laidler",
          author_email = "help@stsci.edu",
          platforms = ["Linux","Solaris","Mac OS X", "Windows"],
          packages=['saaclean'],
          package_dir={'saaclean':'lib'},
          cmdclass = {'install_data':smart_install_data},
          data_files = [('saaclean',['lib/SP_LICENSE'])],
          )


