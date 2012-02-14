from __future__ import division # confidence high

pkg = "stsci.numdisplay"

setupargs = { 
    'version' :         "1.5",
    'description' :     "Package for displaying numpy arrays in DS9",
    'author' :          "Warren Hack",
    'maintainer_email': "help@stsci.edu",
    'url' :             "",
    'license' :         "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'platforms' :       ["any"],
    'package_dir' :     { 'stsci.numdisplay' : 'lib/stsci/numdisplay' },
    'data_files' :      [ ( 'stsci/numdisplay', ['lib/stsci/numdisplay/imtoolrc', 'LICENSE.txt','lib/stsci/numdisplay/ichar.dat'] )],
    }



