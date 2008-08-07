import distutils.extension
pkg = "sample_package"

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

    # how to install your data files:
    #   [
    #       ( directory_name_files_go_to, [ file_name_in_source_tree, another_data_file, etc ] )
    #   ]
    'data_files' :      [ ( pkg, [ "a.txt" ] ), ( pkg+"/data", [ "data/*.dat" ] ) ],

    # extension modules:
    #
    'ext_modules' :     [ 
                        distutils.extension.Extension( pkg+".sscanf", [ "src/sscanf.c" ] ),
                        ]

}

