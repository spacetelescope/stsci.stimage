from distutils.core import Extension
import sys, os.path, glob, string, commands

#Define the datafiles that are part of this package
PYSYNPHOT_DATA_FILES = glob.glob(os.path.join('pysynphot', 'data', 'generic', '*'))
WAVECAT_FILES = glob.glob(os.path.join('pysynphot', 'data', 'wavecat', '*'))
testfiles = glob.glob(os.path.join('pysynphot','test','etctest_base_class.py'))

#Create one list containing all those files.
PYSYNPHOT_DATA_FILES.extend(WAVECAT_FILES)
PYSYNPHOT_DATA_FILES.extend(testfiles)
