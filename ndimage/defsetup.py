from __future__ import division # confidence high

import distutils.extension
import numpy

pkg = [ "ndimage", "ndimage.test" ]

setupargs = {

    'package_dir' :     { 'ndimage':'lib', 'ndimage.test':'test'},

    'ext_modules' :     [ 
                        distutils.extension.Extension( 
                            "ndimage._nd_image",
                            [
                                "src/nd_image.c","src/ni_filters.c", "src/ni_fourier.c",
                                "src/ni_interpolation.c", "src/ni_measure.c", 
                                "src/ni_morphology.c","src/ni_support.c"
                            ],
                            include_dirs = [ "src", numpy.get_include() ],
                            )
                        ],

}

