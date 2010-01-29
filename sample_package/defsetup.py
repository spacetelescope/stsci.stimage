
from __future__ import division # confidence high

import distutils.extension

# list of all the packages to be installed
pkg = [ 'sample_package', 'sample_package.test' ]

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
    'package_dir' :     { 'sample_package' : 'lib', 'sample_package.test' : 'test' },

    # how to install your data files:
    #   [
    #       ( directory_name_files_go_to, [ file_name_in_source_tree, another_data_file, etc ] )
    #   ]
    'data_files' :      [ ( pkg[0], [ "a.txt" ] ), ( pkg[0]+"/data", [ "data/*.dat" ] ) ],

    # extension modules:
    #
    'ext_modules' :     [ 
                        distutils.extension.Extension( pkg[0]+".sscanf", [ "src/sscanf.c" ] ),
                        ]

}

