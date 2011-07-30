from __future__ import division # confidence high

import distutils.extension
import numpy

def extmod(name) :
    return distutils.extension.Extension(
        "stsci.image."+name,
        [ "src/"+name+"module.c" ],
        include_dirs = [ numpy.get_include(), numpy.get_numarray_include() ],
        define_macros = [ ('NUMPY', '1') ]
    )

pkg = ["stsci.image", "stsci.image.test"]

setupargs = {

    'version' :         '2.0',

    'description' :     'image array manipulation functions',

    'author' :          'Todd Miller',

    'author_email' :    'help@stsci.edu',

    'package_dir' :     { 'stsci.image':'lib/stsci/image', 'stsci.image.test':'lib/stsci/image/test'},

    'ext_modules' :     [ extmod("_combine") ],

}

