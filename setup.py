#!/usr/bin/env python

import os 
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install import install


py_bin = parse_makefile(get_makefile_filename())['BINDIR']
ver = sys.version_info
python_exec = 'python' + str(ver[0])+'.'+str(ver[1])
params = string.join(sys.argv[2:])
packages = ['pyraf', 'numdisplay']
topdir = os.getcwd()

if __name__ == '__main__' :

    setup(
        name="Numarray, PyFITS, PyRAF",
        version="1.1",
        description="",
        author="Science Software Branch, STScI",
        author_email="help@stsci.edu",
        url="http://www.stsci.edu/resources/software_hardware/index_html?category=Data_Analysis",
        py_modules = ['pyfits', 'readgeis', 'fitsdiff']
        )
    for p in packages:
        os.chdir(topdir)
        os.chdir(p)
        os.system(python_exec + " setup.py install " + params)





