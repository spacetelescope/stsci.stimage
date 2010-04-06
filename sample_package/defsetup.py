
from __future__ import division # confidence high

import distutils.extension

# list of all the packages to be installed
pkg = [ 'sample_package', 'sample_package.tests' ]

setupargs = {

    'version' :         '1.0',

    'description' :     'A sample of how the stsci_python package install system works',

    'long_description' : "This is not a real package.  It just shows an example of how to get all kinds of files installed.",

    'author' :          'Mark Sienkiewicz',

    'author_email' :    'help@stsci.edu',

    'url' :             'http://www.stsci.edu/resources/software_hardware/pyraf/stsci_python',

    'scripts' :         [ 'lib/sample_package' ],

    'license' :         'BSD',

    'platforms' :       ["Linux","Solaris","Mac OS X", "Win"],

    # what directory each python package comes from:
    'package_dir' :     { 
                        # This causes the main package to be installed, but only the .py files
                        'sample_package' : 'lib', 

                        # this causes the sub-package to be installed, but only the .py files
                        'sample_package.tests' : 'tests' 
                        },

    # how to install your data files:
    #   [
    #       ( directory_name_files_go_to, [ file_name_in_source_tree, another_data_file, etc ] )
    #   ]
    'data_files' :      [ 
                        # data files in the installed package directory
                        ( 'sample_package', [ "a.txt" ] ), 

                        # data files in a subdirectory of the installed package
                        ( 'sample_package/data', [ "data/*.dat" ] ), 

                        # data files for tests, in the "tests" subdirectory of the package
                        ( 'sample_package/tests', [ 'tests/*.dat' ] )  ],

    # extension modules written in C:
    #
    'ext_modules' :     [ 
                        distutils.extension.Extension( "sample_package.sscanf", [ "src/sscanf.c" ] ),
                        ]

}

