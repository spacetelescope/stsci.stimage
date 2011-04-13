from __future__ import division # confidence high

pkg = "reftools2"

setupargs = {

    'version' :         '1.1',
    'description' :     "Reference File Python Tools",
    'author' :          "Warren Hack, Nadezhda Dencheva, Vicki Laidler, Matt Davis",
    'author_email' :    "help@stsci.edu",
    'license' :         "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [( pkg+"/pars", ['lib/pars/*']), (pkg, ['lib/*.help']), (pkg, ['lib/LICENSE.txt'])],
    'scripts' :         ['lib/tdspysyn'],
    'platforms' :       ["Linux", "Solaris", "Mac OS X", "Win"],
    }
