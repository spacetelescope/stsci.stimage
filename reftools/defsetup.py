from __future__ import division # confidence high

pkg = "reftools"

setupargs = {

    'version' :         '1.0',
    'description' :     "Reference File Python Tools",
    'author' :          "Warren Hack, Nadezhda Dencheva",
    'author_email' :    "help@stsci.edu",
    'license' :         "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [( pkg+"/pars", ['lib/pars/*']), (pkg, ['lib/*.help']), (pkg, ['lib/LICENSE.txt'])],
    'scripts' :         ['lib/tdspysyn'],
    'platforms' :       ["Linux", "Solaris", "Mac OS X", "Win"],
    }
