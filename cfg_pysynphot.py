from distutils.core import Extension
import sys, os.path, glob, string, commands

#Define the datafiles that are part of this package
DATA_FILES = glob.glob(os.path.join('data', 'generic', '*'))
WAVECAT_FILES = glob.glob(os.path.join('data', 'wavecat', '*'))
DATA_FILES_DIR = os.path.join('pysynphot', 'data')
testfiles = glob.glob(os.path.join('test','etctest_base_class.py'))

PYSYNPHOT_DATA_FILES = [(DATA_FILES_DIR,DATA_FILES), (DATA_FILES_DIR, WAVECAT_FILES),('pysynphot',testfiles)]
