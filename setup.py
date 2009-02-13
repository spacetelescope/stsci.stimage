#!/usr/bin/env python

import sys
import distutils
import os.path

# list of packages to be installed - we have to populate this 
# to know what to look for.
#
# Start with a list of the packages that we install everywhere.
# Note that you only need to list top-level packages.  (e.g.
# you don't need to list pydrizzle.traits102 because you have
# already listed pydrizzle.)

all_packages_input = [
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
    "sample_package",
    "stistools",
    "wfpc2tools",
]

for x in [ "pyfits", "pysynphot" ] :
    if os.path.isdir(x) :
        all_packages_input.append(x) 
    else :
        print "WARNING:",x," not present"

if ( sys.platform != 'win32' ) :
    if os.path.isdir("pyraf") :
        all_packages_input.append("pyraf")
    else :
        print "WARNING: pyraf not present"


## If you are just adding a new package, you don't need to edit anything
## after this line.


####
#
# Fix distutils to work the way we want it to.
#
# we can't just import this because it isn't installed.  python
# won't find the module.  So, we just read the file and exec it.
#
# When we exec this file, it modifies distutils.  It also defines
# some functions we will want to use later, so we save them.

e_dict = { }
f = open("pytools/lib/stsci_distutils_hack.py","r")
exec f in e_dict
f.close()

set_svn_version = e_dict['__set_svn_version__']
set_setup_date = e_dict['__set_setup_date__']


# pretty printer for debugging
import pprint
pp = pprint.PrettyPrinter(indent=8)
pp = pp.pprint

####
#
# We collect information from defsetup.py in each of the packages,
# then modify it to account for the directory we are in (i.e. one
# level up from the actual package).  We combine the information from
# all of the different packages into a single call to setup().
#
# Because we have the single call to setup(), we can use 
# bdist_wininst to make a single Windows distribution.
#


# python files in each package
all_package_dir = { }

# native C code modules
all_ext_modules = [ ]

# scripts to go in a bin/ directory somewhere
all_scripts = [ ]

# data files to be installed
all_data_files = [ ]

# the list of all packages found.  all_packages_input only lists
# things at the top level, but if one of them has nested packages,
# all_packages will be the complete list.
all_packages = [ ]


# For each package we want to use, fetch the config data.
# We have to fix various things because each package description
# thinks it is only talking about itself, but now we are in the
# parent directory.

for lpkg in all_packages_input :

    all_packages.append(lpkg)

    # forget anything about previous packages that we processed
    pkg = None
    setupargs = None

    # "import" the defsetup file.  We can't use import because
    # none of these are in modules, but we can read/exec.
    #
    # We want to protect our current environment from contamination
    # by the exec'ed code, so we give it a private symbol table

    e_dict = { }

    fname = lpkg+"/defsetup.py"
    f = open(fname,"r")
    exec f in e_dict
    f.close()

    # Pick out the two interesting variables from the exec'ed code

    if "pkg" in e_dict :
        pkg = e_dict["pkg"]
    if "setupargs" in e_dict :
        setupargs = e_dict["setupargs"]

    if isinstance(pkg,str) :
        pkg = [ pkg ]

    # if the package doesn't report the same name that we asked
    # for, there is a major problem.

    if lpkg != pkg[0] :
        raise Exception("yow! package name doesn't match")

    # pick out the "lib" directory, where the pure python comes from

    if not 'package_dir' in setupargs :
        # not specified, use the default
        all_package_dir[pkg[0]] = "%s/%s" % ( pkg[0], 'lib' )
        
    else :
        # there is a list of package dirs to handle
        package_dir = setupargs['package_dir']
        for x in package_dir :
            if not x in all_packages :
                all_packages.append(x)
            all_package_dir[x] = "%s/%s" % (pkg[0], package_dir[x])

    # insert our subversion information and setup date into the
    # packages lib/ directory.

    set_svn_version(pkg[0])
    set_setup_date(pkg[0])

    # If there are scripts, we have to correct the file names where
    # the installer can find them.  Each is under the package directory.

    if 'scripts' in setupargs :
        for x in setupargs['scripts'] :
            all_scripts.append("%s/%s" % ( pkg[0], x ))

    # If there are external modules, we need to correct the file names
    # of source files.

    if 'ext_modules' in setupargs :
        for x in setupargs['ext_modules'] :
            l = [ ]
            for y in x.sources :
                l.append("%s/%s"%(pkg[0],y))
            x.sources = l
            all_ext_modules.append(x)

    # If there are data files, we need to correct the file names.

    if 'data_files' in setupargs :
        for x in setupargs['data_files'] :
            ( instdir, files ) = x
            t = [ ]
            for y in files :
                t.append( pkg[0] + "/" + y )
            all_data_files.append( ( instdir, t ) )


####
#
# now we have read in the in the information from every package,
# and we have also created $pkg/lib/svn_version.py
#
# If the user is asking _only_ to create the version information,
# we can stop now.  (This can happen when we are creating a
# release, or when working with a copy checked out directly
# from subversion.  The "version" command is not useful to
# an end-user.)

if "version" in sys.argv :
    sys.exit(0)

####
# leave a tombstone file with list of scripts in it
# used by opus tools
f=open(".script_tombstone","w")
for x in all_scripts :
    x = x.split("/")
    f.write(x[0]+" "+x[-1]+"\n")
f.close()

####
#
# We have accumulated all the information - now all we need is
# to run the setup.
#

distutils.core.setup(

    # This name is used in various file names to identify this item.
    name="stsci_python",

    # This version is expected to be compared to other version numbers.
    # It will also appear in some file names.
    version="2.7dev",

    # Apparently, description is not used anywhere.
    description="",

    packages = all_packages,
    package_dir = all_package_dir,
    ext_modules = all_ext_modules,
    scripts = all_scripts,
    data_files = all_data_files,
)
