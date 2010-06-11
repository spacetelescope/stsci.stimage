from __future__ import division # confidence high
import os
import shutil

import distutils.extension
pkg = "stsci_sphinxext"

# Sphinx > 0.6.x changed how LaTeX classes are loaded, so we need
# to install the same file with two different file names
shutil.copyfile('lib/latex/tsr.cls', 'lib/latex/sphinxtsr.cls')

setupargs = {
    'version' :         '0.1',
    'description' :
        'A set of tools and templates to customize Sphinx for use in STScI projects',

    'long_description' :
'''In short, you use this package by adding 'from stsci_docs.conf
import *' to the top of your conf.py in your Sphinx documentation
source tree.  In long, see the README file.''',

    'author' :          'Michael Droettboom',

    'author_email' :    'mdroe@stsci.edu',

    'url' :             '',

    'scripts' :         [],

    'license' :         'BSD',

    'platforms' :       ["Linux", "Solaris", "Mac OS X", "Win"],

    # how to install your data files:
    #   [
    #       ( directory_name_files_go_to, [ file_name_in_source_tree, another_data_file, etc ] )
    #   ]
    'data_files' :      [
        (os.path.join(pkg, 'stsci_sphinx_theme'), [
                'lib/stsci_sphinx_theme/theme.conf',
                ]
         ),
        (os.path.join(pkg, 'stsci_sphinx_theme', 'static'), [
                'lib/stsci_sphinx_theme/static/stsci_sphinx.css_t',
                'lib/stsci_sphinx_theme/static/stsci_logo.png',
                'lib/stsci_sphinx_theme/static/stsci_background.png'
                ]
         ),
        (os.path.join(pkg, 'latex'), [
                'lib/latex/stsci_logo.pdf',
                'lib/latex/tsr.cls',
                'lib/latex/sphinxtsr.cls'
                ]
         )
        ],
}

