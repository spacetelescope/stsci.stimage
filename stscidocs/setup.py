#!/usr/bin/env python

import sys, os, os.path, glob, shutil
from distutils.core import setup
from distutils.sysconfig import *
from distutils.command.install_data import install_data
data_dir = []

"""
This is not the most straightforward way to copy doc files
but is the only possible that I see now. 'smart_install_data' is
used to figure out the installation directory (this works with
win_bdist as well) but it copies only files. So to keep the
directory structure for the api documentation created by epydoc,
shutil.copytree is used for the api documentation.
Maybe there's a better way ... ND
"""

args = sys.argv[:]
for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
        sys.argv.extend([
                "--install-lib="+dir,
                ])
        #remove --local from both sys.argv and args
        args.remove(a)
        sys.argv.remove(a)

class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        data_dir.append(getattr(install_cmd, 'install_lib'))
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

cwd = os.getcwd()
pdf_dir = os.path.join(cwd, 'pdf')
ps_dir = os.path.join(cwd, 'ps')
html_dir = os.path.join(cwd, 'html')

html_files = []
pdf_files = []
ps_files = []

DATA_FILES = glob.glob( os.path.join(pdf_dir, '*'))
DATA_FILES.extend(glob.glob( os.path.join(ps_dir, '*')))

todel = []
for l in DATA_FILES:
    if 'CVS' in l:
        todel.append(l)

if todel != []:
    for l in todel:
        DATA_FILES.remove(l)

for l in os.listdir(html_dir):
    if l.endswith('api'):
        html_files.append(l)
    else:
        pass

for l in os.listdir(pdf_dir):       
    if l.endswith('.pdf'):
        pdf_files.append(l)
    else:
        pass

for l in os.listdir(ps_dir):        
    if l.endswith('ps'):
        ps_files.append(l)
    else:
        pass
        
def copy_doc(args):
    if 'install' in args:
        for p in html_files:
            doc_dir = os.path.join(data_dir[0], 'stscidocs', p)
            if os.path.exists(doc_dir):
                try:
                    shutil.rmtree(doc_dir)
                except:
                    print "Error removing old doc directory %s from installation directory\n" % p
            shutil.copytree(os.path.join(html_dir, p), doc_dir)


def write_index(data_dir):
    header = """
    <HTML> 
    <HEAD>
    <TITLE> STSCI PYTHON SOFTWARE DOCUMENTATION </TITLE>
    </HEAD>
    <BODY>
    <h3>STSCI PYTHON SOFTWARE DOCUMENTATION </h3>
    """


    footer = """
    </BODY>
    </HTML>
    """

    path = os.path.join(cwd, 'index.html')
    f = open(path, 'w')
    f.writelines(header)
    #PDF FILES
    f.write("<h3>PDF FILES</h3> ")
    f.write("<table>\n")
    for p in pdf_files:
        f.write("<tr><td> \n")
        ppath =  '<a href=\"file://' + os.path.join(data_dir, p)+'\"' + "> %s </a>" % p
        f.write(ppath)
        f.write('\n</td></tr><p>')
    f.write("</table>\n\n")

    #POSTSCRIPT FILES
    f.write("<h3>POSTSCIPT FILES</h3> ")
    f.write("<table>\n")
    for p in ps_files:
        f.write("<tr><td> \n")
        ppath =  '<a href=\"file://' + os.path.join(data_dir, p)+'\"' + "> %s </a>" % p
        f.write(ppath)
        f.write('\n</td></tr><p>')
    f.write("</table>\n\n")

    #HTML DOCS
    f.write("<h3>api documentation</h3> ")
    f.write("<table>\n")
    for p in html_files:
        f.write("<tr><td> \n")
        #f.write(p)
        #f.write(" \n </td><td> \n")
        ppath =  '<a href=\"file://' + os.path.join(data_dir, p, 'index.html')+'\"' + "> %s </a>" % p
        f.write(ppath)
        f.write('\n</td></tr><p>')
    f.write("</table>\n\n")

    f.writelines(footer)
    f.close()
    

if __name__ == '__main__' :
    args = sys.argv[:]

    setup(
        name="stscidocs",
        version="0.1",
        description="Package for displaying stsci_python documentation",
        author="Astronomy Tools And Applications Branch, STScI",
        maintainer_email="help@stsci.edu",
        url="http://www.stsci.edu/resources/software_hardware/index_html?category=Data_Analysis",
        license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
        platforms = ['any'],
        packages = ['stscidocs'],
	package_dir={'stscidocs':'lib'},
        cmdclass = {'install_data':smart_install_data},
	data_files = [('stscidocs', DATA_FILES)]
        )
    ddir = os.path.join(data_dir[0], 'stscidocs')
    copy_doc(args)
    print 'data_dir', data_dir
    write_index(ddir)
    shutil.copyfile('./index.html', os.path.join(ddir, 'index.html'))

