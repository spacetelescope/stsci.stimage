#!/usr/bin/env python

import os, os.path, sys
from distutils.core import setup
from distutils.sysconfig import *
#from distutils.command.install import install

from cfg_pyraf import PYRAF_DATA_FILES, PYRAF_SCRIPTS, PYRAF_EXTENSIONS, PYRAF_CLCACHE
from cfg_pydrizzle import PYDRIZZLE_EXTENSIONS
from cfg_modules import PYFITS_MODULES, PYTOOLS_MODULES
from cfg_imagestats import IMAGESTATS_EXTENSIONS

#py_includes = get_python_inc(plat_specific=1)
py_libs =  get_python_lib(plat_specific=1)
ver = get_python_version()
pythonver = 'python' + ver

args = sys.argv[2:]
data_dir = py_libs

PACKAGES = ['pyraf','numdisplay', 'imagestats', 'multidrizzle', 'saaclean', 'pydrizzle', 'pydrizzle.traits102']



PACKAGE_DIRS = {'pyraf':'pyraf/lib','numdisplay':'numdisplay', 'imagestats':'imagestats/lib', 'multidrizzle':'multidrizzle/lib', 'saaclean':'saaclean/lib', 'pydrizzle':'pydrizzle/lib', 'pydrizzle.traits102':'pydrizzle/traits102'}

PYMODULES = PYFITS_MODULES + PYTOOLS_MODULES

for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
	sys.argv.append('--install-lib=%s' % dir)
        sys.argv.append('--install-scripts=%s' % os.path.join(dir, 'pyraf'))
        data_dir = os.path.abspath(string.split(a,"=")[1])
	sys.argv.remove(a)
    elif a.startswith('--home='):
        data_dir = os.path.join(os.path.abspath(string.split(a, '=')[1]), 'lib', 'python')
        args.remove(a)
    elif a.startswith('--prefix='):
        data_dir = os.path.join(os.path.abspath(string.split(a, '=')[1]), 'lib', pythonver, 'site-packages')
        args.remove(a)
    elif a.startswith('--install-data='):
        data_dir = os.path.abspath(string.split(a, '=')[1])
        args.remove(a)
    elif a.startswith('--clean_dist'):
        for f in PYMODULES:
            print "cleaning distribution ..."
            file = f + '.py'
            try:
                os.unlink(file)
            except OSError: pass
        sys.argv.remove(a)
        sys.exit(0)

    else:
        print "Invalid argument  %s", a

PYRAF_DATA_DIR = os.path.join(data_dir, 'pyraf')
PYRAF_CLCACHE_DIR = os.path.join(data_dir, 'pyraf', 'clcache')

NUMDISPLAY_DATA_DIR = os.path.join(data_dir, 'numdisplay')
NUMDISPLAY_DATA_FILES = ['numdisplay/imtoolrc']

SAACLEAN_DATA_FILES = ['saaclean/lib/SP_LICENSE']
SAACLEAN_DATA_DIR = os.path.join(data_dir, 'saaclean')

DATA_FILES = [(PYRAF_DATA_DIR, PYRAF_DATA_FILES), (PYRAF_CLCACHE_DIR, PYRAF_CLCACHE), (NUMDISPLAY_DATA_DIR, NUMDISPLAY_DATA_FILES), (SAACLEAN_DATA_DIR, SAACLEAN_DATA_FILES)]
EXTENSIONS = PYRAF_EXTENSIONS + PYDRIZZLE_EXTENSIONS + IMAGESTATS_EXTENSIONS

if sys.platform == 'win32':
    PACKAGES.remove('pyraf')
    del(PACKAGE_DIRS['pyraf'])
    #remove pyraf's data files
    DATA_FILES = [(NUMDISPLAY_DATA_DIR, NUMDISPLAY_DATA_FILES),(SAACLEAN_DATA_DIR, SAACLEAN_DATA_FILES)]
    EXTENSIONS = PYDRIZZLE_EXTENSIONS + IMAGESTATS_EXTENSIONS


setup(name="STScI Python Software",
      version="2.2",
      description="",
      author="Science Software Branch, STScI",
      maintainer_email="help@stsci.edu",
      url="http://www.stsci.edu/resources/software_hardware/index_html?category=Data_Analysis",
      packages = PACKAGES,
      py_modules = PYMODULES,
      package_dir = PACKAGE_DIRS,
      data_files = DATA_FILES,
      scripts = ['pyraf/lib/pyraf'],
      ext_modules = EXTENSIONS,
      )






