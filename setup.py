#!/usr/bin/env python

import os, os.path, sys
import glob
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install_data import install_data

from cfg_pyraf import PYRAF_DATA_FILES, PYRAF_SCRIPTS, PYRAF_EXTENSIONS, PYRAF_CLCACHE
from cfg_pydrizzle import PYDRIZZLE_EXTENSIONS
from cfg_imagestats import IMAGESTATS_EXTENSIONS
from cfg_calcos import CALCOS_EXTENSIONS

#py_includes = get_python_inc(plat_specific=1)
py_libs =  get_python_lib(plat_specific=1)
ver = get_python_version()
pythonver = 'python' + ver

args = sys.argv[2:]
#data_dir = py_libs


PACKAGES = ['calcos','numdisplay', 'imagestats',
            'multidrizzle', 'pydrizzle', 'pydrizzle.traits102',
            'pytools', 'nictools', 'stistools', 'wfpc2tools']

#The normal directory structure is {packagename:packagename/lib.}
PACKAGE_DIRS = {}
for p in PACKAGES:
    PACKAGE_DIRS[p]="%s/lib"%p
#Exceptions are allowed; put them here.
PACKAGE_DIRS['numdisplay']='numdisplay'
PACKAGE_DIRS['pydrizzle.traits102']='pydrizzle/traits102'


for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
        sys.argv.extend([
                "--install-lib="+dir,
                "--install-scripts=%s" % os.path.join(dir,"pyraf"),
                ])
        sys.argv.remove(a)
        args.remove(a)


class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)




IMAGESTATS_DATA_DIR = os.path.join('imagestats')
IMAGESTATS_DATA_FILES = ['imagestats/lib/LICENSE.txt']

MULTIDRIZZLE_DATA_DIR = os.path.join('multidrizzle')
MULTIDRIZZLE_DATA_FILES = ['multidrizzle/lib/LICENSE.txt']

NUMDISPLAY_DATA_DIR = os.path.join('numdisplay')
NUMDISPLAY_DATA_FILES = ['numdisplay/imtoolrc', 'numdisplay/LICENSE.txt']

PYDRIZZLE_DATA_DIR = os.path.join('pydrizzle')
PYDRIZZLE_DATA_FILES = ['pydrizzle/lib/LICENSE.txt']

NICTOOLS_DATA_FILES = ['nictools/lib/SP_LICENSE']
NICTOOLS_DATA_DIR = os.path.join('nictools')


DATA_FILES = [ (NUMDISPLAY_DATA_DIR, NUMDISPLAY_DATA_FILES),
               (NICTOOLS_DATA_DIR, NICTOOLS_DATA_FILES),
               (IMAGESTATS_DATA_DIR, IMAGESTATS_DATA_FILES),
               (MULTIDRIZZLE_DATA_DIR, MULTIDRIZZLE_DATA_FILES),
               (PYDRIZZLE_DATA_DIR, PYDRIZZLE_DATA_FILES)  ]

EXTENSIONS = PYDRIZZLE_EXTENSIONS + IMAGESTATS_EXTENSIONS

SCRIPTS = None

#The following packages are part of the stsci_python module distribution,
#but reside in their own repositories. This setup.py will look for
#them before attempting to build them.
#
#Only try to build PyRAF if the pyraf directory exists locally
# and we're not on a windwos platform
if os.path.exists(os.path.join('pyraf')) and sys.platform != 'win32':
    PACKAGES.append('pyraf')
    PACKAGE_DIRS['pyraf']='pyraf/lib'
    PYRAF_DATA_DIR = os.path.join('pyraf')
    PYRAF_CLCACHE_DIR = os.path.join('pyraf', 'clcache')
    EXTENSIONS = EXTENSIONS + PYRAF_EXTENSIONS
    DATA_FILES.extend([(PYRAF_DATA_DIR, PYRAF_DATA_FILES), (PYRAF_CLCACHE_DIR, PYRAF_CLCACHE)])
    SCRIPTS = ['pyraf/lib/pyraf']

#Only install pyfits if the pyfits directory exists locally
if os.path.exists(os.path.join('pyfits')):
    PACKAGES.append('pyfits')
    PACKAGE_DIRS['pyfits']='pyfits/lib'

# install pysynphot if the pysynphot directory exists
if os.path.exists(os.path.join('pysynphot')):
	PACKAGES.append('pysynphot')
	PACKAGE_DIRS['pysynphot']='pysynphot/lib'
	DATA_FILES.append( ( "pysynphot", glob.glob(os.path.join('pysynphot','test','etctest_base_class.py')) ) )
	PYSYNPHOT_DATA_DIR = os.path.join('pysynphot','data')
	DATA_FILES.append( ( PYSYNPHOT_DATA_DIR, glob.glob(os.path.join('pysynphot', 'data', 'generic', '*')) ) )
	DATA_FILES.append( ( PYSYNPHOT_DATA_DIR, glob.glob(os.path.join('pysynphot', 'data', 'wavecat', '*')) ) )

setup(name="STScI Python Software",
      version="2.5",
      description="",
      author="Science Software Branch, STScI",
      maintainer_email="help@stsci.edu",
      url="http://www.stsci.edu/resources/software_hardware/index_html?category=Data_Analysis",
      packages = PACKAGES,
      package_dir = PACKAGE_DIRS,
      cmdclass = {'install_data':smart_install_data},
      data_files = DATA_FILES,
      scripts = SCRIPTS,
      ext_modules = EXTENSIONS,
      )






