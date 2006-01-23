#!/usr/bin/env python

import sys, os.path 
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install import install

pythonlib = get_python_lib(plat_specific=1)
ver = get_python_version()
pythonver = 'python' + ver
data_dir = os.path.join(pythonlib, 'numdisplay')

args = sys.argv[:]

for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
        sys.argv.append('--install-lib=%s' % dir)
        data_dir = os.path.join(dir, 'numdisplay')
        #remove --local from both sys.argv and args
        args.remove(a)
        sys.argv.remove(a)
    elif a.startswith('--home='):
        data_dir = os.path.join(os.path.abspath(a.split('=')[1]), 'lib', 'python', 'numdisplay')
        args.remove(a)
    elif a.startswith('--prefix='):
        data_dir = os.path.join(os.path.abspath(a.split('=')[1]), 'lib', pythonver, 'site-packages', 'numdisplay')
        args.remove(a)
    elif a.startswith('--install-data='):
        data_dir = os.path.abspath(a.split('=')[1])
        args.remove(a)
    elif a.startswith('bdist_wininst'):
        install.INSTALL_SCHEMES['nt']['data'] = install.INSTALL_SCHEMES['nt']['purelib']
        args.remove(a)

if __name__ == '__main__' :

    setup(
        name="Numdisplay",
        version="1.0",
        description="Package for displaying numarray arrays in DS9",
        author="Warren Hack",
        maintainer_email="help@stsci.edu",
        url="",
        license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
     platforms = ["any"],
        packages = ['numdisplay'],
	package_dir={'numdisplay':''},
	data_files = [(data_dir, ['imtoolrc'])]
        )

