#!/usr/bin/env python

import sys, os
import string

def test22():
    packages = [
        'numdisplay',
        'fitsdiff',
        'pydrizzle',
        'imageiter',
        'imagestats',
        'irafglob',
        'makewcs',
        'multidrizzle',
        'nimageiter',
        'numarray',
        'numcombine',
        'pyfits',
        'readgeis',
	'parseinput']

    required_versions = {
        'numdisplay':   '0.2.0',
        'fitsdiff' :    '1.3',
        'pydrizzle':    '5.5.6',
        'imageiter':    '0.1',
        'imagestats':   '1.0.0',
        'irafglob':     '1.0',
        'makewcs':      '0.7.0',
        'multidrizzle': '2.6.7',
        'nimageiter':   '0.5',
        'numarray':     '1.3.3',
        'numcombine':   '0.3.0',
        'pyfits':       '1.0',
        'readgeis' :    '1.8',
	'parseinput':	'0.1.4'
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
            if pyraf.__version__ < "1.2" :
                pyraf_message = "\nThe latest public release of PyRAF is v 1.1.1.\nAlthough it's not required, you may want to upgrade\n"
        except ImportError:
            print "\nPyRAF is not installed or not on your PYTHONPATH.\nPlease correct this if you intend to use it, before you attempt to run multidrizzle.\n"

    except ImportError:
        print "\nNumeric was not detected. Please install Numeric, if you intend to use PyRAF.\n"


    try:
        import Pmw
    except ImportError:
        print "\nPmw is required and was not detected. It's either not installed or not on PYTHONPATH.\n"

    
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
    else:
        print pyraf_message
        print "\nAll packages were successfully installed.\n"
    
    
if __name__ == '__main__':
    test22()
