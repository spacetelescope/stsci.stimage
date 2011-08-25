from __future__ import division # confidence high

import distutils.extension
import numpy

pkg = [ "stsci.ndimage", "stsci.ndimage.tests" ]

setupargs = {

    # how to install your data files:
    #   [
    #       ( directory_name_files_go_to, [ file_name_in_source_tree, another_data_file, etc ] )
    #   ]
    'data_files' :      [
                        # data files in the installed package directory
                        ( 'stsci/ndimage/tests',  [ 'lib/stsci/ndimage/tests/*.png' ] ),
                        ],

    'package_dir' :     { 'stsci.ndimage':'lib/stsci/ndimage', 'stsci.ndimage.tests':'lib/stsci/ndimage/tests'},

    'ext_modules' :     [ 
                        distutils.extension.Extension( 
                            "stsci.ndimage._nd_image",
                            [
                                "src/nd_image.c","src/ni_filters.c", "src/ni_fourier.c",
                                "src/ni_interpolation.c", "src/ni_measure.c", 
                                "src/ni_morphology.c","src/ni_support.c"
                            ],
                            include_dirs = [ "src", numpy.get_include() ],
                            )
                        ],

}

