#!/usr/bin/env python

import os, sys, string, os.path
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install import install


ver = sys.version_info
python_exec = 'python' + str(ver[0])+ '.' + str(ver[1])

def getDataDir(args):
    for a in args:
        if string.find(a, '--home=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = os.path.join(dir, 'lib/python/numdisplay')
        elif string.find(a, '--prefix=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = os.path.join(dir, 'lib', python_exec, 'site-packages/numdisplay')
        elif string.find(a, '--install-data=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = dir
        else:
            data_dir = os.path.join(sys.prefix, 'lib', python_exec, 'site-packages/numdisplay')
    return data_dir
                



if __name__ == '__main__' :
    args = sys.argv
    data_dir = getDataDir(args)
    setup(
        name="Numdisplay",
        version="0.1",
        description="Package for displaying numarray arrays in DS9",
        author="Warren Hack",
        maintainer_email="help@stsci.edu",
        url="",
        license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
     platforms = ["any"],
        packages = ['numdisplay'],
	package_dir={'numdisplay':''},
	data_files = [(data_dir, ['imtoolrc'])]
        )

