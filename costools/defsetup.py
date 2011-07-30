from __future__ import division # confidence high

pkg = ["costools"]

setupargs = {

    'version' :         '1.0',
    'description' :     "Python Modules for COS",
    'author' :          "Warren Hack, Nadezhda Dencheva, Phil Hodge",
    'author_email' :    "help@stsci.edu",
    'license' :         "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [( "costools/pars", ['lib/costools/pars/*']), ('costools', ['lib/costools/*.help']), ('costools', ['LICENSE.txt'])],
    'scripts' :         ['lib/costools/timefilter'],
    'package_dir' :     { 'costools' : 'lib/costools', },
    'platforms' :       ["Linux", "Solaris", "Mac OS X", "Win"],
}
