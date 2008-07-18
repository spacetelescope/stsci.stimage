#!/usr/bin/env python

# $Id: $

import sys, os
import string


#
# required versions shows the exact version number of some
# package that we expect to find after installing stsci_python
#
# Do not list pysynphot - it cannot be imported without a
# valid CDBS tree, so we can't check the version number

required_versions = {
        'calcos':               '2.1',
        'convolve':             '2.0',
        'image':                '2.0',
        'imagestats':           '1.2',
        'multidrizzle':         '3.1.0',
        'ndimage':              '2.0',
        'nictools.puftcorr':    '0.17',
        'nictools.rnlincor':    '0.8',
        'nictools.saaclean':    '1.2',
        'numdisplay':           '1.5',
        'numpy':                '1.1.0',
        'pydrizzle':            '6.1',
        'pyfits':               '1.4',
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
        'pytools.wcsutil':      '1.1.0',
        'pytools.xyinterp':     '0.1',
        'stistools.mktrace':    '1.1',
        'stistools.sshift':     '1.4',
        'stistools.stisnoise':  '5.4',
        'stistools.wx2d':       '1.1',
        }

# optional_versions shows the exact version number that must
# be present if the module is found, but we do not complain
# if the module is missing

optional_versions = {
        'pyraf' :               '1.7',
}

report_list = [
        # do not list "Pmw" - it does not have __version__
        # do not list "PIL" - it does not have __version__
        "pyraf",
        "Sybase",
        "IPython",
        "matplotlib",
        "nose",
        "setuptools",
        "urwid",
         ]

def pkg_info(p) :
        """
        """
        try:
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
                        return [ ver.split(' ')[0], loc ]
                except :
                        return [ "???", loc ]
        except ImportError:
            return [ "not found", "???" ]
        # not reached


def testpk( ):
    print ""
    print "Checking installed versions"
    print ""

    pyraf_message = ""
    

    if string.split(sys.version)[0] < '2.3':
        print "Python version 2.3 is required to run multidrizzle."

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


    messages = []

    required_packages= list(set(required_versions))

    for p in required_packages:
        result = pkg_info(p)
        if required_versions[p] !=  result[0] :
            message = "%-20s %-15s %-15s %s" % (p, required_versions[p], result[0], result[1])
            messages.append(message)

    optional_packages = list(set(optional_versions))

    for p in optional_packages:
        result = pkg_info(p)
        if optional_versions[p] !=  result[0] and result[0] != 'not found' :
            message = "%-20s %-15s %-15s %s" % (p, optional_versions[p], result[0], result[1])
            messages.append(message)

    if len(messages) != 0:
        print "%-20s %-15s %-15s %s"%("package","expected","found","location")
        messages.sort()
        for m in messages:
            print m
        print pyraf_message
    else:
        print pyraf_message
        print "All packages were successfully installed.\n"
    

def report_pk() :
    # make a unique list of all the modules we want a report on
    a = list ( set(required_versions) | set(optional_versions) | set(report_list) )
    a.sort()
    # print the package info
    for x in a :
        i = pkg_info(x)
        print "%-20s %-15s %s"%(x,i[0],i[1])
    
if __name__ == '__main__':
        if len(sys.argv) > 1 :
            if sys.argv[1] == '-r' :
                report_pk()
            if sys.argv[1] == '-t' :
                testpk()
            if sys.argv[1] == '-d' :
                testpk(debug=1)
        else :
            testpk()

