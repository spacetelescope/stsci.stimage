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
    setup(name = "puftcorr",
          version = "0.1",
          description = "Estimates and removes 'Mr. Staypuft' signal from a NICMOS exposure.",
          author = "Howard Bushouse",
          author_email = "help@stsci.edu",
          license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
          platforms = ["Linux","Solaris","Mac OS X"],
          packages=['puftcorr'],
          package_dir={'puftcorr':'lib'},
          cmdclass = {'install_data':smart_install_data},
          data_files = [('puftcorr',['lib/LICENSE.txt'])],
          )

	

