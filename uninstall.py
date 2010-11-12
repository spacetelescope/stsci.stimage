#!/usr/bin/env python
from __future__ import division # confidence high


import os.path
import sys


# force and never are undocumented features that are not intended for users.
force = False
if "-f" in sys.argv :
    force = True
never = False
if "-n" in sys.argv :
    never = True


make_old = False

# make_old is an undocumented features that is not intended for users.
if "-old" in sys.argv :
    make_old = True

# a function to ask yes/no
def print_and_ask(n) :
    sys.stdout.write("\n")
    sys.stdout.write(n)

    # force is an undocumented feature that is not intended
    # for users.
    if force :
        sys.stdout.write("\n")
        return True
    if never :
        return False

    sys.stdout.write("\n    (y/n)?")
    sys.stdout.flush()
    n = sys.stdin.readline()
    if len(n) > 0 :
        n = n[0]
    return n == 'y' or n == 'Y'


##
## the main program
##

#
# tell the user about this program
#

print """

Python does not have a facility to uninstall packages.  This program
will attempt to locate and (optionally) remove Python packages or
modules that look like part of STSCI_PYTHON 2.3, 2.4, 2.5, or 2.6.

It will search sys.path (initialized by Python from internal values
and your environment variable PYTHONPATH).  If it finds a thing
that looks like part of STSCI_PYTHON, the program offers to delete
it.  If you type "y" and press enter, it will try to delete it.  If
you type "n" or just press enter, it will skip that item.

It recognizes parts of STSCI_PYTHON by file or directory names.  If
you have other files that have the same names, this program cannot
recognize that.  But you can -- if this program offers to delete
something that should not be deleted, answer "n".

If there is an error deleting a package, the first error will be
reported and the rest of that package will not be deleted.

"""

if never or not print_and_ask("continue") :
    sys.exit(0)

#
# look for scisoft; issue a warning if it is found
#

have_scisoft = 0

for x in sys.path :
    if x.find("scisoft") >= 0 :
        have_scisoft = 1

if os.path.isdir("/Applications/scisoft") or os.path.isdir("/scisoft") :
    have_scisoft = 1

if have_scisoft :
    print """
It looks like you may have Scisoft on this machine.  We often receive
reports of difficulty when trying to upgrade the STSDAS or STSCI_PYTHON
software that is in the Scisoft distribution.  It may be helpful
to contact the distributors of Scisoft if you have problems.

"""
    if never or not print_and_ask("continue") :
        sys.exit(0)
    
#
# list of packages
#

all_package = [

# package from the list in setup.py

    "acstools",
    "betadrizzle",
    "calcos",
    "convolve",
    "image",
    "imagemanip",
    "imagestats",
    "multidrizzle",
    "ndimage",
    "nictools",
    "numdisplay",
    "opuscoords",
    "pydrizzle",
    "pytools",
    "reftools",
    "sample_package",
    "stimage",
    "stistools",
    "stsci_sphinxext",
    "stwcs",
    "wfpc2tools",

# others

    "pyfits",
    "pyraf",
    "pysynphot",
    "pytools",
    "stscidocs",
    "pywcs",

# old from stsci_python 2.5
    "puftcorr",
    "rnlincor",
    "saaclean",

]

#
# list of scripts
#

all_script = [
    "pyraf",
    "calcos",
    "convertwaiveredfits",
    "sample_package",
    "fitsdiff",
    "stscidocs",
]

#
# only older versions had modules (i.e. single .py files)
#

all_module = [ 

# old from stsci_python 2.5
    "evaldisp",
    "fileutil",
    "fitsdiff",
    "gettable",
    "gfit",
    "imageiter",
    "irafglob",
    "iterfile",
    "linefit",
    "makewcs",
    "mktrace",
    "nimageiter",
    "nmpfit",
    "numcombine",
    "numerixenv",
    "parseinput",
    "r_util",
    "radialvel",
    "readgeis",
    "sshift",
    "stisnoise",
    "testutil",
    "versioninfo",
    "wavelen",
    "wcsutil",
    "wx2d",
    "xyinterp",

# old from stsci_python 2.4
    "pyfits",

]


# function to delete an entire directory tree - basically, "rm -r"
# copied from the python library reference section on the "os" module

def deltree(top) :
    for root, dirs, files in os.walk(top, topdown=False):
        # print dirs, files
        for name in files:
            path = os.path.join(root, name)
            try :
                os.remove(path)
            except Exception, e:
                print path, e
                print "skipping the rest..."
                return
        for name in dirs:
            path = os.path.join(root, name)
            try :
                os.rmdir(path)
            except Exception, e:
                print path, e
                print "skipping the rest..."
                return
    try :
        os.rmdir(top)
    except Exception, e:
        print path, e
        return

# function to detect if a module file is present
def module_file(dir,module) :
    np = os.path.join(dir,module)
    for x in [ ".py", ".pyc", ".pyo" ] :
        if os.path.isfile(np+x) :
            return True
    return False

# function to delete a module file - error only if no
# files that might be the module are deleted
def rm_module_file(dir,module) :
    np = os.path.join(dir,module)
    errcount = 0
    list = [ ".py", ".pyc", ".pyo" ]
    for x in list :
        path = np+x
        try :
            os.remove(np)
        except Exception, e:
            errcount = errcount + 1
            errpath = np
    if errcount == len(list) :
        print errpath, e

# found_any will be set if we found anything that we consider deleting
found_any = 0

# get the current directory so we can recognize it if we see it in sys.path
#
# The specification of os.getcwd() does not say whether it
# returns a normalized path or not, so we normalize it.
cwd = os.getcwd()
cwd = os.path.abspath(cwd)

#
# search the whole sys.path for anything that might be ours
#

for x in sys.path :
    x = os.path.abspath(x)
    if x == cwd :
        # skip current directory - it is probably the new stuff
        continue
    if os.path.isdir(x) :
        # look for our packages or modules in the named directory
        for p in all_package :
            np = os.path.join(x,p)
            if os.path.isdir(np) :
                found_any = 1
                if print_and_ask("delete package "+np) :
                    if make_old :
                        print "rename ",np, np+".old"
                        os.rename(np, np+".old")
                    else :
                        deltree(np)

        for p in all_module :
            if module_file(x,p) :
                found_any = 1
                if print_and_ask("delete module "+x+"/"+p) :
                    if make_old :
                        # saves the py file only
                        np = x + "/" + p
                        print "rename ",np, np+".old"
                        os.rename(np+".py",np+".py.old")
                    # removes py, pyc, pyo
                    rm_module_file(x,p)

    elif os.path.isfile(x) :
        pass
        # this is probably either a mistake or an egg file.  either way,
        # we don't know what to do with it.  (But it doesn't matter --
        # stsci_python does not create eggs.)
    else :
        pass
        # not a file _or_ a directory?  whatever...

#
# search the system path for scripts.  This is only going to
# work on unix-like systems, but the only non-unix system we
# support at all is Windows, and people are probably using the
# windows-installer there.
#

p = os.getenv("PATH")
if p :
    p = p.split(':')
    for x in p :
        x = os.path.abspath(x)
        if x == cwd :
            continue
        if not os.path.isdir(x) :
            continue
        for p in all_script :
            p = os.path.join(x,p)
            if os.access(p,os.X_OK) :
                found_any = 1
                if print_and_ask("delete script "+p) :
                    try :
                        if make_old :
                            print "rename ",p, p+".old"
                            os.rename(p, p+".old")
                        else :
                            os.remove(p)
                    except Exception, e:
                        print p, e


#
# print something informative if we did not find anything
#
if not found_any :
    print """

Did not find anything to uninstall.

"""

#
# I do not know that we can be sure that we actually got everything.  Try
# to import things; if anything actually imports, it is still out there
# somewhere.
#

still_found = 0

for x in all_package :
    try :
        n = __import__(x)
        still_found=1
        print "package",x,"- not completely uninstalled"
    except ImportError :
        pass
    except :
        # it raised an exception while importing, so it must be there
        still_found=1
        print "package",x,"- maybe not completely uninstalled"

for x in all_module :
    try :
        n = __import__(x)
        still_found=1
        print "module",x,"- not completely uninstalled"
    except ImportError :
        pass
    except :
        # it raised an exception while importing, so it must be there
        still_found=1
        print "module",x,"- maybe not completely uninstalled"

if ( not found_any ) and still_found :
    print """
This program could not find anything to uninstall, but is still
able to import packages/modules.  Something strange is happening.
Maybe you have STSCI_PYTHON packages or modules stored in a python
egg or a zip file?

"""
