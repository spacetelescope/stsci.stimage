#!/usr/bin/env python
from __future__ import division # confidence high

# $Id: $

import sys
import os
import string

print 'This script is disabled for now...'
sys.exit(1)

# suppress all the text that comes out when you import pysynphot
# ( not good enough... I guess we still won't check pysynphot )
os.environ['PYSYN_CDBS'] = '.' 

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
        'multidrizzle':         '3.2.1',
        'ndimage':              '2.0',
        'nictools.puftcorr':    '0.17',
        'nictools.rnlincor':    '0.8',
        'nictools.saaclean':    '1.2',
        'numdisplay':           '1.5',
        'numpy':                '1.1.0',
        'pydrizzle':            '6.2.2',
        'pyfits':               '1.4',
        'stsci.tools' :             '3.0',
        'stsci.tools.fileutil':     '1.3.1',
        'stsci.tools.fitsdiff':     '1.4',
        'stsci.tools.gfit':         '1.0',
        'stsci.tools.imageiter':    '0.2',
        'stsci.tools.irafglob':     '1.0',
        'stsci.tools.iterfile':     '0.2',
        'stsci.tools.linefit':      '1.0',
        'stsci.tools.makewcs':      '0.8.5',
        'stsci.tools.nimageiter':   '0.6',
        'stsci.tools.nmpfit':       '0.2',
        'stsci.tools.numcombine':   '0.4.0',
        'stsci.tools.parseinput':   '0.1.5',
        'stsci.tools.readgeis':     '2.1',
        'stsci.tools.versioninfo':  '0.2.0',
        'stsci.tools.wcsutil':      '1.1.0',
        'stsci.tools.xyinterp':     '0.1',
        'stistools.mktrace':    '1.1',
        'stistools.sshift':     '1.4',
        'stistools.stisnoise':  '5.4',
        'stistools.wx2d':       '1.1',
        'wfpc2tools.wfpc2cte':  '1.2.4',
        'wfpc2tools.wfpc2destreak': '2.17',
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
        except ImportError, e:
            return [ "not found", str(e) ]
        # not reached


def testpk( verbose ):

    if verbose :
        print "sys.path :"
        for x in sys.path :
            print "       ",x

    print ""
    print "Checking installed versions"
    print ""

    if string.split(sys.version)[0] < '2.3':
        print "Python version 2.3 is required to run multidrizzle."

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
            message = "%-25s %-20s %-20s %s" % (p, required_versions[p], result[0], result[1])
            messages.append(message)

    optional_packages = list(set(optional_versions))

    for p in optional_packages:
        result = pkg_info(p)
        if optional_versions[p] !=  result[0] and result[0] != 'not found' :
            message = "%-25s %-20s %-20s %s" % (p, optional_versions[p], result[0], result[1])
            messages.append(message)

    if len(messages) != 0:
        print "%-25s %-20s %-20s %s"%("package","expected","found","location")
        messages.sort()
        for m in messages:
            print m
        if not verbose :
            print "If you will be sending email to help@stsci.edu, please re-run with 'python testpk.py -v'"
    else:
        print "All packages were successfully installed.\n"
    


def report_pk(opus_mode) :
    # cannot create a proper report from the current directory where the
    # stsci_python distribution is; go somewhere else
    try :os.chdir("/")
    except : pass

    colfmt = "%-25s %-15s %-15s %s"

    if not opus_mode :
        print colfmt%("package","version","expected","location")
    else :
        print colfmt%("package","version","","location")
    print ""


    # make a unique list of all the modules we want a report on
    a = list ( set(required_versions) | set(optional_versions) | set(report_list) )
    a.sort()
    # print the package info
    for p in a :
        i = pkg_info(p)
        if not opus_mode :
            expect="-"
            if p in optional_versions :
                if i[0] != optional_versions[p] :
                    expect = optional_versions[p]
            if p in required_versions :
                if i[0] != required_versions[p] :
                    expect = required_versions[p]
            print colfmt%(p,i[0],expect,i[1])
        else :
            print colfmt%(p,i[0],'',i[1])

interactive_help = """
python testpk.py
python testpk.py -t
    test that installed versions are as expected

python testpk -r
    report versions of everything

python testpk -o
    report versions in format for opus

python testpk.py -h
    this help

"""
    
if __name__ == '__main__':
        if len(sys.argv) > 1 :
            if sys.argv[1] == '-h' :
                print interactive_help
                sys.exit(0)
            if sys.argv[1] == '-r' :
                report_pk(0)
            if sys.argv[1] == '-o' :
                report_pk(1)
            if sys.argv[1] == '-t' :
                testpk(0)
            if sys.argv[1] == '-v' :
                testpk(1)
        else :
            testpk(0)
