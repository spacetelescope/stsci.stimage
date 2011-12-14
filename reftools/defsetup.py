from __future__ import division # confidence high

import distutils.core
import distutils.sysconfig

pkg = "reftools"

pythoninc = distutils.sysconfig.get_python_inc()

ext = [distutils.core.Extension('reftools._computephotpars',
        ['src/compute_value.c',
         'src/py_compute_value.c'],
        include_dirs = [pythoninc])]

setupargs = {

    'version' :         '1.6.0',
    'description' :     "Reference File Python Tools",
    'author' :          "Warren Hack, Nadezhda Dencheva, Vicki Laidler, Matt Davis",
    'author_email' :    "help@stsci.edu",
    'license' :         "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [( "reftools/pars", ['lib/reftools/pars/*']), ('reftools', ['lib/reftools/*.help']), ('reftools', ['LICENSE.txt'])],
    'scripts' :         ['lib/reftools/tdspysyn'],
    'platforms' :       ["Linux", "Solaris", "Mac OS X", "Win"],
    'package_dir' :     { 'reftools' : 'lib/reftools' },
    'ext_modules':      ext,
    }
