"""
This module is designed to be included from Sphinx conf.py
configuration files to set reasonable defaults for STScI projects.

It should be imported from the top of your `conf.py` as::

  from stsci_sphinxext.conf import *

The "extensions" line in your `conf.py` should also be modified so it
doesn't override the extensions defined here.  For example, change::

  extensions = ['sphinx.ext.autodoc']

to::

  extensions += ['sphinx.ext.autodoc']

Also, comment out the 'html_theme' line.
"""

import glob
import os
import sys

# Store the directory that this file is in so we can get at our data
__dir__ = os.path.abspath(os.path.dirname(__file__))

# In order to get Sphinx to import extensions in this directory
sys.path.insert(0, __dir__)

primary_domain = 'python'

# A list of standard extensions
extensions = [
    'narrow_field_lists',      # Create field lists that don't waste
                               # so much horizontal space
    'abstract',                # Support an abstract directive
    'sphinx.ext.autodoc',      # Extract documentation from docstrings
    'sphinx.ext.intersphinx',  # Link to other Sphinx documentation trees
    'sphinx.ext.pngmath',      # Render math as images
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosummary',  # (Required by numpydoc)
    'numpydoc',                # Support the Numpy docstring format
    'matplotlib.sphinxext.plot_directive', # matplotlib plots
    'matplotlib.sphinxext.only_directives'
    ]

# Class documentation should contain *both* the class docstring and
# the __init__ docstring
autoclass_content = "both"

autodoc_member_order = "groupwise"

# Set the default role to the "smart" one
default_role = 'obj'

# Support linking directly to the Python standard library and Numpy
intersphinx_mapping = {
    'http://docs.python.org/': None,
    'http://docs.scipy.org/doc/numpy': None,
    'http://matplotlib.sourceforge.net': None,
    }

# Don't show summaries of the members in each class along with the
# class' docstring
numpydoc_show_class_members = False

# ----------------------------------------------------------------------
# HTML OPTIONS
# ----------------------------------------------------------------------

# Override the default HTML theme
html_theme_path = [__dir__]
html_theme = 'stsci_sphinx_theme'
html_logo = os.path.join(
    __dir__, 'stsci_sphinx_theme', 'static', 'stsci_logo.png')

# ----------------------------------------------------------------------
# LATEX OPTIONS
# ----------------------------------------------------------------------

# Additional stuff for the LaTeX preamble.
latex_preamble = r'''
\usepackage{amsmath}
\DeclareUnicodeCharacter{00A0}{\nobreakspace}

% In the parameters section, place a newline after the Parameters
% header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}

% Make Examples/etc section headers smaller and more compact
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}

% Fix footer/header
\@ifundefined{chaptermark}{%
\newcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
}{%
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
}

\@ifundefined{chaptermark}{%
\newcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
}{%
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
}

\@ifundefined{TSR}{}{%
\renewcommand{\py@HeaderFamily}{\rmfamily\bfseries}
\definecolor{TitleColor}{rgb}{0,0,0}
\renewcommand{\appendix}{\par
  \setcounter{section}{0}%
  \inapptrue%
  \renewcommand\thesection{\@Alph\c@section}}
}

\makeatother

% Make the pages always arabic, and don't do any pages without page
% numbers
\pagenumbering{arabic}
\pagestyle{plain}
'''

latex_logo = os.path.join(__dir__, 'latex', 'stsci_logo.pdf')

latex_additional_files = glob.glob(os.path.join(__dir__, 'latex', '*'))

# ----------------------------------------------------------------------
# GRAPHVIZ OPTIONS
# ----------------------------------------------------------------------

# To get graphviz working in our environment
graphviz_web_image_format = 'png'
graphviz_latex_image_mode = 'ps2:pdf'
