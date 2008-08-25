import distutils.extension
import numpy


pkg = "ndimage"

setupargs = {

    'ext_modules' :     [ 
                        distutils.extension.Extension( 
                            pkg+"._nd_image",
                            [
                                "src/nd_image.c","src/ni_filters.c", "src/ni_fourier.c",
                                "src/ni_interpolation.c", "src/ni_measure.c", 
                                "src/ni_morphology.c","src/ni_support.c"
                            ],
                            include_dirs = [ "src", numpy.get_include() ],
                            )
                        ],

    'date_files' :      [
                            ( pkg+"/tests", [ "tests/*" ] )
                        ],
}


#     config.add_data_dir('tests')

