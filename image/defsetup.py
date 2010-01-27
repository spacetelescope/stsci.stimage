from __future__ import division # confidence high

import distutils.extension
import numpy

def extmod(name) :
    return distutils.extension.Extension(
        pkg+"."+name,
        [ "src/"+name+"module.c" ],
        include_dirs = [ numpy.get_include(), numpy.get_numarray_include() ],
        define_macros = [ ('NUMPY', '1') ]
        # library_dirs = [ "/usr/local/lib" ]
        # libraries = [ "X11" ]
    )


pkg = "image"

setupargs = {

    'version' :         '2.0',

    'description' :     'image array manipulation functions',

    'author' :          'Todd Miller',

    'author_email' :    'help@stsci.edu',

    'ext_modules' :     [ extmod("_combine") ],

    'data_files' :              [ ( pkg +'/tests', ['tests/*']), ]


}

