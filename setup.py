#!/usr/bin/env python

import os, os.path
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install import install
import shutil

py_bin = parse_makefile(get_makefile_filename())['BINDIR']
ver = sys.version_info
python_exec = 'python' + str(ver[0])+'.'+str(ver[1])
args = sys.argv[2:]
f2cparams = []
params = []
moduleparams = []
for a in args:
    if a.startswith('--with-f2c='):
        f2cparams.append('--with-f2c='+os.path.abspath((string.split(a, '=')[1])))
        sys.argv.remove(a)
    elif a.startswith('--local='):
	f2cparams.append('--local='+os.path.abspath(string.split(a, '=')[1]))
	params.append('--local='+os.path.abspath(string.split(a, '=')[1]))
	sys.argv.append('--install-lib=%s' % os.path.abspath((string.split(a,"=")[1])))
	sys.argv.append('--install-data=%s' % os.path.abspath((string.split(a,"=")[1])))

	sys.argv.remove(a)
    else:
	f2cparams.append(a)
	params.append(a)

f2cpar = string.join(f2cparams)
par = string.join(params)
f2cpackages = ['pydrizzle']
packages = ['pyraf', 'numdisplay', 'imagestats', 'multidrizzle', 'saaclean', 'pyfits', 'pytools']
topdir = os.getcwd()


def dosetup():
    r = setup(
        name="STScI Python Software",
        version="2.2",
        description="",
        author="Science Software Branch, STScI",
        maintainer_email="help@stsci.edu",
        url="http://www.stsci.edu/resources/software_hardware/index_html?category=Data_Analysis",
        )
    for p in packages:
        os.chdir(topdir)
        os.chdir(p)
        os.system(python_exec + " setup.py install " + par)
    for p in f2cpackages:
	print "Installing %s\n", p
        os.chdir(topdir)
        os.chdir(p)
        os.system(python_exec + " setup.py install " + f2cpar)

    return r


def main():
    args = sys.argv
    if '--help' in args:
	for p in (packages + f2cpackages):
	    os.chdir(topdir)
            os.chdir(p)
            os.system(python_exec + " setup.py --help ")
    else:
        dosetup()



if __name__ == '__main__' :
    main()


	




