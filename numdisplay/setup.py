#!/usr/bin/env python

import sys, os.path
from distutils.core import setup
from distutils.sysconfig import *
#from distutils.command.install import install
from distutils.command.install_data import install_data

pythonlib = get_python_lib(plat_specific=1)
ver = get_python_version()
pythonver = 'python' + ver
data_dir = os.path.join(pythonlib, 'numdisplay')

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

if __name__ == '__main__' :

    setup(
        name="Numdisplay",
        version="1.2",
        description="Package for displaying numpy arrays in DS9",
        author="Warren Hack",
        maintainer_email="help@stsci.edu",
        url="",
        license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
        platforms = ["any"],
        packages = ['numdisplay'],
        package_dir={'numdisplay':''},
        cmdclass = {'install_data':smart_install_data},
        data_files = [('numdisplay',['imtoolrc', 'LICENSE.txt'])]
        )
