#!/usr/bin/env python

import sys, os
import string


required_versions = {
        'calcos':               '2.1',
        'convolve':             '2.0',
        'imagestats':           '1.2',
        'image':                '2.0',
        'imagestats':           '1.2',
        'multidrizzle':         '3.1.0',
        'numpy':                '1.0.4',
        'numdisplay':           '1.5',
        'pydrizzle':            '6.1',
        'pyfits':               '1.4',
        'nictools.rnlincor':    '0.8',
        'nictools.puftcorr':    '0.17',
        'pytools.fileutil':     '1.3.1',
        'pytools.fitsdiff':     '1.4',
        'pytools.gfit':         '1.0',
        'pytools.imageiter':    '0.2',
        'pytools.irafglob':     '1.0',
        'pytools.iterfile':     '0.2',
        'pytools.linefit':      '1.0',
        'pytools.makewcs':      '0.8.1',
        'pytools.nimageiter':   '0.6',
        'pytools.nmpfit':       '0.2',
        'pytools.numcombine':   '0.4.0',
        'pytools.parseinput':   '0.1.5',
        'pytools.readgeis':     '2.0',
        'pytools.versioninfo':  '0.2.0',
        'pytools.xyinterp':     '0.1',
        'pytools.wcsutil':      '1.1.0',
        'nictools.saaclean':    '1.2',
        'stistools.mktrace':    '1.1',
        'stistools.sshift':     '1.4',
        'stistools.stisnoise':  '5.4',
        'stistools.wx2d':       '1.1',
        }

def testpk():
    print ""
    print "Checking installed versions"
    print ""

    packages=[ ]
    for x in required_versions :
        packages.append(x)

    packages.sort()


    pyraf_message = ""
    
    install_messages = []
    installed_packages = {}
    if string.split(sys.version)[0] < '2.3':
        install_messages.append("Python version 2.3 is required to run multidrizzle.\n")

    try:
        import pyraf
        if pyraf.__version__ < "1.4" :
            pyraf_message = "The latest public release of PyRAF is v 1.4.\n Pyraf v. %s was found.\n" % pyraf.__version__
    except ImportError:
        print "PyRAF is not installed or not on your PYTHONPATH.\nPlease correct this if you intend to use it, before you attempt to run multidrizzle.\n"


    try:
        import Pmw
    except ImportError:
        print "Pmw is required and was not detected. It's either not installed or not on PYTHONPATH.\n"

    try:
        import urwid
    except ImportError:
        print "Package urwid was not found. It is not required but if available will enable text based epar in pyraf.\n"

    try:
        import IPython
    except ImportError:
        print "Package ipython was not found. It is not required but if available can be used with pyraf (pyraf --ipython).\n"
    for p in packages:
        try:
                print p
                exec "import " + p
                try :
                    loc = eval( p + ".__path__" )
                    loc = loc[0]
                except AttributeError :
                    try :
                        loc = eval( p + ".__file__" )
                    except AttributeError :
                        loc = "???"
                try :
                        ver = eval( p + ".__version__" )
                        installed_packages[p] = [ ver.split(' ')[0], loc ]
                except :
                        installed_packages[p] = [ "???", loc ]
        except ImportError:
            installed_packages[p] = [ "not found", "???" ]

    for p in packages:
        if required_versions[p] !=  installed_packages[p][0] :
            message = "%-20s %-15s %-15s %s" % (p, required_versions[p], installed_packages[p][0], 
                installed_packages[p][1])
            install_messages.append(message)

    if len(install_messages) != 0:
        print "%-20s %-15s %-15s %s"%("package","expected","found","location")
        for m in install_messages:
            print m
        print pyraf_message
    else:
        print pyraf_message
        print "All packages were successfully installed.\n"
    
    
if __name__ == '__main__':
        testpk()
