#!/usr/bin/env python

import sys, os
import string

def test25():
    packages = [
        'imagestats',
        'multidrizzle',
        'numpy',
        'numdisplay',
        'puftcorr',
        'pydrizzle',
        'pyfits',
        'rnlincor',
        'fileutil',
        'fitsdiff',
        'gfit',
        'imageiter',
        'irafglob',
        'iterfile',
        'linefit',
        'makewcs',
        'nimageiter',
        'nmpfit',
        'numcombine',
        'parseinput',
        'readgeis',
        'versioninfo',
        'xyinterp',
        'wcsutil',
        'saaclean',
        'mktrace',
        'sshift',
        'stisnoise',
        'wx2d']
    
    required_versions = {
        'imagestats':       '1.1.1',
        'multidrizzle':     '3.0.0',
        'numpy':            '1.0.3',
        'numdisplay':       '1.2',
        'puftcorr':         '0.16',
        'pydrizzle':        '6.0.0',
        'pyfits':           '1.1',
        'rnlincor':         '0.7',
        'fileutil':         '1.3.1',
        'fitsdiff':         '1.4',
        'gfit':             '1.0',
        'imageiter':        '0.2',
        'irafglob':         '1.0',
        'iterfile':         '0.2',
        'linefit':          '1.0',
        'makewcs':          '0.8.0',
        'nimageiter':       '0.6',
        'nmpfit':           '0.2',
        'numcombine':       '0.4.0',
        'parseinput':       '0.1.5',
        'readgeis':         '2.0',
        'versioninfo':      '0.2.0',
        'xyinterp':         '0.1',
        'wcsutil':          '1.1.0',
        'saaclean':         '1.0',
        'mktrace':          '1.1',
        'sshift':           '1.4',
        'stisnoise':        '5.4',
        'wx2d':             '1.1',
        }
    


    pyraf_message = ""
    
    install_messages = []
    installed_packages = {}
    if string.split(sys.version)[0] < '2.3':
        install_messages.append("\nPython version 2.3 is required to run multidrizzle.\n")

    try:
        import pyraf
        if pyraf.__version__ < "1.4" :
            pyraf_message = "\nThe latest public release of PyRAF is v 1.4.\n Pyraf v. %s was found.\n" % pyraf.__version__
    except ImportError:
        print "\nPyRAF is not installed or not on your PYTHONPATH.\nPlease correct this if you intend to use it, before you attempt to run multidrizzle.\n"


    try:
        import Pmw
    except ImportError:
        print "\nPmw is required and was not detected. It's either not installed or not on PYTHONPATH.\n"

    try:
        import urwid
    except ImportError:
        print "\nPackage urwid was not found. It is not required but if available will enable text based epar in pyraf.\n"

    try:
        import IPython
    except ImportError:
        print "\nPackage ipython was not found. It is not required but if available can be used with pyraf (pyraf --ipython).\n"
    for p in packages:
        try:
            package = __import__(p)
            installed_packages[p] = string.split(package.__version__)[0]
        except ImportError:
            installed_packages[p] = 0
            install_messages.append("Package %s is required, but is not installed.\n" % p)
                                                             
    for p in packages:
        if required_versions[p] !=  installed_packages[p] and installed_packages[p] != 0:
            message = "\n%s v %s was expected, v %s was found\n" % (p, required_versions[p], installed_packages[p])            
            install_messages.append(message)
        
    if len(install_messages) != 0:
        for m in install_messages:
            print m
	print pyraf_message
    else:
        print pyraf_message
        print "\nAll packages were successfully installed.\n"
    
    
if __name__ == '__main__':
    test25()
