#!/usr/bin/env python
import sys

# suppress all the text that comes out when you import pysynphot
import warnings
warnings.filterwarnings("ignore")

def pkg_info(p) :
        """
        """
        try:
                exec "import " + p
                try :
                    loc = eval( p + ".__path__" )
                    loc = loc[0]
                except AttributeError :
                    try :
                        loc = eval( p + ".__file__" )
                    except AttributeError :
                        loc = "???"
                try :
                        ver = eval( p + ".__version__" )
                        return [ ver.split(' ')[0], loc ]
                except :
                        return [ "???", loc ]
        except ImportError, e:
            return [ "not found", str(e) ]
        # not reached

def print_opus_versions() :
    colfmt = "%-25s %-15s %-15s %s"
    print colfmt%("package","version","","location")

    for x in sys.stdin :
        x=x.split()
        p = x[0]
        i = pkg_info(p)
        print colfmt%(p,i[0],'',i[1])

print_opus_versions()
