from __future__ import division # confidence high

pkg = "costools"

setupargs = {

    'version' :         '1.0',
    'description' :     "Python Modules for COS",
    'author' :          "Warren Hack, Nadezhda Dencheva, Phil Hodge",
    'author_email' :    "help@stsci.edu",
    'license' :         "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [( pkg+"/pars", ['lib/pars/*']), (pkg, ['lib/*.help']), (pkg, ['lib/LICENSE.txt'])],
    'scripts' :         ['lib/timefilter'],
    'platforms' :       ["Linux", "Solaris", "Mac OS X", "Win"],
}
