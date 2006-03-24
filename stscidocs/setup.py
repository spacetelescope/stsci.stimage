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

"""
class smart_install_data(install_data):
    def run(self):
        install_cmd = self.get_finalized_command('install')
        data_dir.append(getattr(install_cmd, 'install_lib'))
        self.install_dir = getattr(install_cmd, 'install_lib')
"""

class smart_install_data(install_data):
    def run(self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        data_dir.append(getattr(install_cmd, 'install_lib'))
        self.install_dir = getattr(install_cmd, 'install_lib')

cwd = os.getcwd()

pkgpaths = {}


for l in os.listdir(cwd):
    print l
    if l.endswith('_pkg'):
        pkgpaths[l] = l.rstrip('_pkg')
    else:
        pass

def copy_doc(args):
    if 'install' in args:
        for p in pkgpaths.keys():
            doc_dir = os.path.join(data_dir[0], 'stscidocs', p)
            if os.path.exists(doc_dir):
                try:
                    shutil.rmtree(doc_dir)
                except OSError:
                    print "Error removing old doc directory %s from installation directory\n" % p
            shutil.copytree(p, doc_dir)
        doc_dir = os.path.join(data_dir[0], 'stscidocs', 'user_docs')
        if os.path.exists(doc_dir):
            try:
                shutil.rmtree(doc_dir)
            except OSError:
                print "Error removing old doc directory %s from installation directory\n" % p
        shutil.copytree('user_docs', doc_dir)


def write_index(data_dir):
    header = """
    <HTML> 
    <HEAD>
    <TITLE> STSCI_PYTHON SOFTWARE DOCUMENTATION </TITLE>
    </HEAD>
    <BODY>
    <center><h3>STSCI_PYTHON SOFTWARE DOCUMENTATION </h3></center>
    """


    footer = """
    </BODY>
    </HTML>
    """

    path = os.path.join(cwd, 'index.html')
    f = open(path, 'w')
    f.writelines(header)

    f.write('<table align="center" width="100%" cellpadding="0" cellspacing="2">\n')

    f.write('<tr> <td valign="top"> <table align="center">\n')

    f.write('<tr><th class="tableheader"><h3> API Documentation </h3></th></tr>\n')

    f.write("<tr><td><ul>")
    for p in pkgpaths.keys():
        htmlpath =  '<a href="file://' + os.path.join(data_dir, p, pkgpaths[p]+'_api', 'index.html')+'" > %s </a> &nbsp;' % 'html'
        pdfpath =  '<a href="file://' + os.path.join(data_dir, p, pkgpaths[p]+'.pdf')+'" > %s </a> &nbsp;' % 'pdf'
        pspath =  '<a href="file://' + os.path.join(data_dir, p, pkgpaths[p]+'.ps')+'" > %s </a> &nbsp;' % 'ps'
        title = pkgpaths[p]+' api <br>'

        f.write("<li> \n")
        f.write(title)
        f.write(htmlpath)
        f.write(pdfpath)
        f.write(pspath)
        f.write('\n</li><p>')
    f.write('</ul></td></tr>')
    f.write("</table>\n\n")

    f.write('</td>\n')
    f.write('<td valign="top"> <table align="center">\n')
    f.write('<tr> <th class="tableheader"><h3>Documentation For Users</h3></th></tr>\n')
    f.write('<tr><td>')
    f.write('<ul>')
    """
    for l in os.listdir('user_docs'):
        path = '<a href="file://' + os.path.join(data_dir, 'user_docs', l)+'" > %s </a>' % l
        f.write('<li>')
        f.write(path)
        f.write('</li>\n')
    """
    f.write('<li> <a href="file://' + os.path.join(data_dir, 'user_docs', 'pyraf_tutorial.pdf')+'" > %s </a> </li><p>' % "PyRAF Tutorial")
    f.write('<li><a href="file://' + os.path.join(data_dir, 'user_docs', 'pyraf_guide.pdf')+'" > %s </a></li><p>' % "PyRAF Programmer's Guide" )
    f.write('<li><a href="file://' + os.path.join(data_dir, 'user_docs', 'pydatatut.pdf')+'" > %s </a> </li><p>' % "Interactive Data Analysis with Python Tutorial" )
    f.write('<li> <a href="file://' + os.path.join(data_dir, 'user_docs', 'pyfits_users_manual.pdf')+'" > %s </a> </li><p>' % "PyFITS Users Manual")
    f.write('<li> <a href="file://' + os.path.join(data_dir, 'user_docs', 'numarray-1.5-1.pdf')+'" > %s </a></li><p>' % "Numarray Manual")
    f.write('<li> <a href="file://' + os.path.join(data_dir, 'user_docs', 'SynphotManual.pdf')+'" > %s </a></li><p>' % "Synphot Users Guide" )
    f.write('<li> <a href="file://' + os.path.join(data_dir, 'user_docs', 'MultiDrizzle_Ch1.pdf')+'" > %s </a> </li><p>' % "Multidrizzle Users Guide")
    f.write("<ul>")
    f.write('</td></tr>\n')
    f.write('</table>')
    f.write('</td></tr>')
    f.write('</table>')
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
	data_files = [('stscidocs', '')]
        )
    #gidd=get_install_data_dir(install_data)
    ddir = os.path.join(data_dir[0], 'stscidocs')
    copy_doc(args)
    print 'data_dir', data_dir
    write_index(ddir)
    shutil.copyfile('./index.html', os.path.join(ddir, 'index.html'))


