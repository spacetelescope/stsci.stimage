#!/usr/bin/env python

import sys, string, os.path
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install import install
import shutil

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

def copy_doc(data_dir, args):
    if 'install' in args:
        doc_dir = os.path.join(data_dir,'doc')
        if os.path.exists(doc_dir):
            try:
                shutil.rmtree(doc_dir)
            except:
                print "Error removing doc directory\n"
        shutil.copytree('doc', doc_dir)



def dolocal():
    """Adds a command line option --local=<install-dir> which is an abbreviation for
    'put all of numdisplay in <install-dir>/numdisplay'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir =  os.path.abspath(a.split("=")[1])
            sys.argv.extend([
                "--install-lib="+dir,
                "--install-data="+os.path.join(dir,"numdisplay")
                ])
            sys.argv.remove(a)


if __name__ == '__main__' :
    args = sys.argv
    print "numdisplay", args
    dolocal()
    data_dir = getDataDir(args)

    setup(
        name="Numdisplay",
        version="1.0",
        description="Package for displaying numarray arrays in DS9",
        author="Warren Hack",
        maintainer_email="help@stsci.edu",
        url="",
        license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
     platforms = ["any"],
        packages = ['numdisplay'],
        package_dir={'numdisplay':''},
        data_files = [(data_dir, ['imtoolrc']), (data_dir, ['LICENSE.txt'])]
        )

    copy_doc(data_dir, args)
