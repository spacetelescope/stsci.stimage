#!/usr/bin/env python

import os 
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install import install


py_bin = parse_makefile(get_makefile_filename())['BINDIR']
ver = sys.version_info
python_exec = 'python' + str(ver[0])+'.'+str(ver[1])
params = string.join(sys.argv[2:])
                     
if __name__ == '__main__' :

    setup(
        name="Numdisplay",
        version="0.1",
        description="",
        author="Warren Hack",
        maintainer_email="help@stsci.edu",
        url="",
        packages = ['numdisplay'],
	package_dir={'numdisplay':''},
	extra_package_dir = ['numdisplay'],
	data_files = [('numdisplay', ['imtoolrc'])]
        )

