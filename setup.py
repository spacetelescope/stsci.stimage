#!/usr/bin/env python

import os, os.path, sys
import glob
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install_data import install_data

from cfg_pyraf import PYRAF_DATA_FILES, PYRAF_SCRIPTS, PYRAF_EXTENSIONS, PYRAF_CLCACHE
from cfg_pydrizzle import PYDRIZZLE_EXTENSIONS
from cfg_imagestats import IMAGESTATS_EXTENSIONS
from cfg_imagemanip import IMAGEMANIP_EXTENSIONS
from cfg_calcos import CALCOS_EXTENSIONS
from cfg_ndimage import NDIMAGE_EXTENSIONS

# PACKAGES is the list of all packages that we want to install.  If you
# want it, make sure it is listed here and in PACKAGE_DIRS

PACKAGES = ['calcos','numdisplay', 'imagestats', 'imagemanip',
    'multidrizzle', 'pydrizzle', 'pydrizzle.traits102',
    'pydrizzle.distortion','pytools', 'nictools', 
    'stistools', 'wfpc2tools','ndimage']


# uninstall_packages is a list of packages that we want to remove when we do
# an install, but that we are not installing.  If an old package has been
# removed from the distribution, list it here.
#
uninstall_packages = [ ] 

#The normal directory structure is {packagename:packagename/lib.}
#
# PACKAGE_DIRS[x] is the relative directory where we can find the source code 
# for # package x
#
PACKAGE_DIRS = {}
for p in PACKAGES:
    PACKAGE_DIRS[p]="%s/lib"%p

#Exceptions are allowed; put them here.
PACKAGE_DIRS['numdisplay']='numdisplay'
PACKAGE_DIRS['pydrizzle.traits102']='pydrizzle/traits102'
PACKAGE_DIRS['pydrizzle.distortion']='pydrizzle/lib/distortion'
PACKAGE_DIRS['ndimage']='ndimage'
args = sys.argv[2:]

# set this to check version numbers of imported modules after an install
check_versions = 1

for a in args :
    print a
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

IMAGEMANIP_DATA_DIR = os.path.join('imagemanip')
IMAGEMANIP_DATA_FILES = ['imagemanip/lib/LICENSE.txt']

MULTIDRIZZLE_DATA_DIR = os.path.join('multidrizzle')
MULTIDRIZZLE_DATA_FILES = ['multidrizzle/lib/LICENSE.txt']

NUMDISPLAY_DATA_DIR = os.path.join('numdisplay')
NUMDISPLAY_DATA_FILES = ['numdisplay/imtoolrc', 'numdisplay/LICENSE.txt']

PYDRIZZLE_DATA_DIR = os.path.join('pydrizzle')
PYDRIZZLE_DATA_FILES = ['pydrizzle/lib/LICENSE.txt']

NICTOOLS_DATA_DIR = os.path.join('nictools')
NICTOOLS_DATA_FILES = ['nictools/lib/SP_LICENSE']

NDIMAGE_DATA_DIR = os.path.join('ndimage')
NDIMAGE_DATA_DIR = os.path.join('ndimage/tests')
NDIMAGE_DATA_FILES = ['ndimage/LICENSE.txt',"ndimagesvn/tests/slice112.raw","ndimage/tests/test_ndimage.py","ndimage/tests/test_segment.py"]

DATA_FILES = [ (NUMDISPLAY_DATA_DIR, NUMDISPLAY_DATA_FILES),
               (NICTOOLS_DATA_DIR, NICTOOLS_DATA_FILES),
               (IMAGESTATS_DATA_DIR, IMAGESTATS_DATA_FILES),
               (MULTIDRIZZLE_DATA_DIR, MULTIDRIZZLE_DATA_FILES),
               (PYDRIZZLE_DATA_DIR, PYDRIZZLE_DATA_FILES),
               (IMAGEMANIP_DATA_DIR, IMAGEMANIP_DATA_FILES),
               (NDIMAGE_DATA_DIR, NDIMAGE_DATA_FILES)]

EXTENSIONS = PYDRIZZLE_EXTENSIONS + IMAGESTATS_EXTENSIONS + IMAGEMANIP_EXTENSIONS + CALCOS_EXTENSIONS + NDIMAGE_EXTENSIONS

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


#
#
#
def uninstall_modules( modules ) :
    # bug: this function should observe the -q / -v flags
    print "Begin uninstall"

    # To uninstall a module, you need to know what directory it is in.
    # To find that information, you can import it and look at __path__
    # That will be a list of things that you can recursively remove to delete the
    # module.

    # BUT:  If you import something, it finds the module in the current directory.
    # That is bad, because we have everything here with us. To fix that, purge the 
    # current directory from sys.path before we start looking for modules.

    cwd = os.getcwd()
    l = [ ]
    for x in sys.path :
        if x == cwd or x == '' or x == '.' :
            continue
        l.append(x)
    sys.path = l

    # Now we walk the list of modules that we want to uninstall and
    # remove each one.

    import shutil

    # I want to make sure that we remove all copies of the module that are
    # in the search path.  If we don't, we might remove the first one, only
    # to find another one behind it somewhere.  So, keep trying until we
    # can't find anything more to delete.  
    #
    # If we can't remove one of the modules, we also can't remove anything
    # that it shadows.  removed_list is a list of everything we tried to
    # remove; if it is still there in a second pass, we ignore it so we
    # don't get in to an infinite loop.

    removed_list = [ ]
    any_removed = 1
    while any_removed :
        any_removed=0
        for x in modules :
            try :
                mod = __import__( x )
            except :
                # modules that don't exist are ok
                continue

            for dir in mod.__path__ :
                if not dir in removed_list :
                    removed_list.append(dir)
                    try :
                        shutil.rmtree(dir, 0)
                    except :
                        print "    remove",dir,"failed"
                    any_removed=1
    print "End uninstall"

if sys.argv[1] == "uninstall" :
    # Uninstall any old copies of packages that we don't want around.  That is
    # a list of all the packages we are installing, plus a list of whatever
    # other things we might be wanting to outdate.  (i.e. was in the last release
    # but should not be in this one.  It would be better for distutils to have a 
    # real uninstall, but I am not going to implement one right now.)
    # I am assuming that "module" and "package" are the same thing, though I am 
    # not sure that is strictly true.
    uninstall_modules( PACKAGES + uninstall_packages )
    sys.exit(0)

setup(name="STScI Python Software",
      version="2.7dev",
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

# don't check versions until it works in the automatic build
check_versions = 0
if check_versions :
    # if installing everything, we can also check the module version numbers now
    import testpk
    testpk.testpk()

