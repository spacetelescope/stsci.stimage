#!/usr/bin/env python

# BUG : how do we put the svnversion tags in everything ?

import os.path

# list of packages to be installed - we have to populate this 
# to know what to look for
all_packages_input = [ 
    "pytools",
    "pydrizzle",
    "convolve",
    "image",
    "imagemanip",
    "imagestats",
    "multidrizzle",
    "ndimage",
    "nictools",
    "numdisplay",
    "sample_package",
    "stistools",
    "wfpc2tools",
]

for x in [ "pysynphot", "pyraf", "pyfits" ] :
    if os.path.isdir(x) :
        all_packages_input.append(x) 
    else :
        print ""
        print "Optional package",x,"not present"
        print ""

## If you are just adding a new package, you don't need to edit anything
## after this line.

####
#
# Fix distutils to work the way we want it to.
#
# we can't just import this because it isn't installed python
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

import sys
import distutils

# where are the python files in that package
all_package_dir = { }

# native code modules
all_ext_modules = [ ]

# scripts to go in a bin/ directory somewhere
all_scripts = [ ]

#
all_data_files = [ ]

#
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

    # if the package doesn't say that it's package name is what
    # we thought it was, we have a problem

    if lpkg != pkg[0] :
        raise "yow! package name doesn't match"

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

    set_svn_version(pkg[0])
    set_setup_date(pkg[0])

    # If there are scripts, we have to correct the file names where
    # the installer can find them.  Each is under the pkg directory.

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

if "version" in sys.argv :
    sys.exit(0)

####

distutils.core.setup(

    # This name is used in various file names to identify this item.
    name="stsci_python",

    # This version is expected to be compared to other version numbers.
    # It will also appear in some file names.
    version="2.7dev",

    # Description is not used.
    description="",

    packages = all_packages,
    package_dir = all_package_dir,
    ext_modules = all_ext_modules,
    scripts = all_scripts,
    data_files = all_data_files,
    
)
