#!/usr/bin/env python

import os, os.path
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install import install
import shutil

py_bin = parse_makefile(get_makefile_filename())['BINDIR']
ver = sys.version_info
python_exec = 'python' + str(ver[0])+'.'+str(ver[1])
args = sys.argv[2:]
f2cparams = []
params = []
moduleparams = []
for a in args:
    if a.startswith('--with-f2c='):
        f2cparams.append('--with-f2c='+os.path.abspath((string.split(a, '=')[1])))
        sys.argv.remove(a)
    elif a.startswith('--local='):
	f2cparams.append('--local='+os.path.abspath(string.split(a, '=')[1]))
	params.append('--local='+os.path.abspath(string.split(a, '=')[1]))
	sys.argv.append('--install-lib=%s' % os.path.abspath((string.split(a,"=")[1])))
	sys.argv.append('--install-data=%s' % os.path.abspath((string.split(a,"=")[1])))

	sys.argv.remove(a)
    else:
	f2cparams.append(a)
	params.append(a)

f2cpar = string.join(f2cparams)
par = string.join(params)
f2cpackages = ['pydrizzle']
packages = ['pyraf', 'numdisplay', 'imagestats', 'multidrizzle']
topdir = os.getcwd()

def getDataDir(args):
    for a in args:
        if string.find(a, '--home=') == 0:
            dir = os.path.abspath(string.split(a, '=')[1])
            data_dir = os.path.join(dir, 'lib/python')
        elif string.find(a, '--prefix=') == 0:
            dir = os.path.abspath(string.split(a, '=')[1])
            data_dir = os.path.join(dir, 'lib', python_exec, 'site-packages')
        elif a.startswith('--install-data='):
            dir = os.path.abspath(string.split(a, '=')[1])
            data_dir = dir
        else:
            data_dir = os.path.join(sys.prefix, 'lib', python_exec, 'site-packages')
    return data_dir

def dosetup():
    r = setup(
        name="STScI Python Software",
        version="2.0",
        description="",
        author="Science Software Branch, STScI",
        maintainer_email="help@stsci.edu",
        url="http://www.stsci.edu/resources/software_hardware/index_html?category=Data_Analysis",
        py_modules = ['pyfits', 'readgeis', 'fitsdiff', 'imageiter', 'irafglob',  'makewcs', 'nimageiter', 'numcombine', 'versioninfo']
        )
    for p in packages:
        os.chdir(topdir)
        os.chdir(p)
        os.system(python_exec + " setup.py install " + par)
    for p in f2cpackages:
        os.chdir(topdir)
        os.chdir(p)
        os.system(python_exec + " setup.py install " + f2cpar)

    return r


def copy_doc(data_dir, args):
    if 'install' in args:
        doc_dir = os.path.join(data_dir,'doc')
        if os.path.exists(doc_dir):
	    try:
                shutil.rmtree(doc_dir)
            except:
                print "Error removing doc directory\n"
	shutil.copytree('doc', doc_dir)


def main():
    args = sys.argv
    if '--help' in args:
	for p in (packages + f2cpackages):
	    os.chdir(topdir)
            os.chdir(p)
            os.system(python_exec + " setup.py --help ")
    else:
        data_dir = getDataDir(args)

        dosetup()
        os.chdir(topdir)
        copy_doc(data_dir, args)



if __name__ == '__main__' :
    main()


	




