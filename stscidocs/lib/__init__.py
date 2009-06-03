
from __future__ import division # confidence high

import webbrowser, os.path, sys, os
__doc__ = """
stscidocs contains documentation for stsci_python packages and modules.
It uses the default browser to display an index.html file.

Usage: import stscidocs
       stscidocs.viewdocs() - view st sci_python docs in a web browser
       stscidocs.index_path - shows the location of index.html
       stscidocs.help() - prints stscidocs.__doc__
       
"""

def help():
    print __doc__

#html_dirs = []
#pdf_files = []
#ps_files = []


name = __name__
cdd =  sys.modules[name].__path__[0]

path = os.path.join(cdd, 'index.html')

index_path = 'file://' + path

def viewdocs():
    try:
        webbrowser.open(index_path, new=1)
    except:
        print "There was a problem displaying the documentation.\n"
        print "To view the web pages, please point your browser at this url:\n"
        print index_path

viewdocs()
