#!/usr/bin/env python

import sys, os
import string

def test24():
    packages = [
        'imagestats',
        'multidrizzle',
        'numarray',
        'numdisplay',
        'puftcorr',
        'pydrizzle',
        'pyfits',
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
        'wcsutil',
        'saaclean',
        'mktrace',
        'sshift',
        'stisnoise',
        'wx2d']
    
    required_versions = {
        'imagestats':       '1.0.1',
        'multidrizzle':     '2.7.2',
        'numarray':         '1.5.2',
        'numdisplay':       '1.1',
        'puftcorr':         '0.1',
        'pydrizzle':        '5.7.0',
        'pyfits':           '1.0.1',
        'fileutil':         '1.2.0',
        'fitsdiff':         '1.3',
        'gfit':             '0.1',
        'imageiter':        '0.1',
        'irafglob':         '1.0',
        'iterfile':         '0.1',
        'linefit':          '0.1',
        'makewcs':          '0.7.0',
        'nimageiter':       '0.5',
        'nmpfit':           '0.1',
        'numcombine':       '0.3.0',
        'parseinput':       '0.1.5',
        'readgeis':         '1.8',
        'versioninfo':      '0.1.1',
        'wcsutil':          '1.0.0',
        'saaclean':         '0.9',
        'mktrace':          '1.0',
        'sshift':           '1.4',
        'stisnoise':        '5.3',
        'wx2d':             '1.0',
        }
    


    pyraf_message = ""
    
    install_messages = []
    installed_packages = {}
    if string.split(sys.version)[0] < '2.3':
        install_messages.append("\nPython version 2.3 is required to run multidrizzle.\n")
    try:
        import Numeric
        try:
            import pyraf
            if pyraf.__version__ < "1.3" :
                pyraf_message = "\nThe latest public release of PyRAF is v 1.3.\n Pyraf v. %s was found.\n" % pyraf.__version__
        except ImportError:
            print "\nPyRAF is not installed or not on your PYTHONPATH.\nPlease correct this if you intend to use it, before you attempt to run multidrizzle.\n"

    except ImportError:
        print "\nNumeric was not detected. Please install Numeric, if you intend to use PyRAF.\n"


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
    test24()
