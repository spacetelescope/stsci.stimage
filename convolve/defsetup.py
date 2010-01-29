from __future__ import division # confidence high

import distutils.extension
import numpy

def extmod(name) :
    return distutils.extension.Extension(
        pkg[0]+"."+name,
        [ "src/"+name+"module.c" ],
        include_dirs = [ numpy.get_include(), numpy.get_numarray_include() ],
        define_macros = [ ('NUMPY', '1') ]
    )


pkg = ["convolve", "convolve.test"]

setupargs = {

    'version' :         '2.0',

    'description' :     'image array convolution functions',

    'author' :          'Todd Miller',

    'author_email' :    'help@stsci.edu',

    'package_dir' :     { 'convolve':'lib', 'convolve.test':'test'},

    'ext_modules' :     [ extmod("_correlate"), extmod("_lineshape") ],

}
