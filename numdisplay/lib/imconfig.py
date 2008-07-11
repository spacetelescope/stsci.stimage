"""
Version 1.0alpha - 9-Oct-2003 (WJH)

loadImtoolrc (imtoolrc=None):
    Locates, then reads in IMTOOLRC configuration file from
    system or user-specified location, and returns the 
    dictionary for reference. 

    The table gets loaded into a dictionary of the form:
        {configno:{'nframes':n,'width':nx,'height':ny},...}
    It can then be accessed using the syntax:
        fbtab[configno][attribute]
    For example:
        fbtab = loadImtoolrc()
        print fbtab[34]['width']
        1056 1024

"""    
import os,string,sys

_default_imtoolrc_env = ["imtoolrc","IMTOOLRC"]
_default_system_imtoolrc = "/usr/local/lib/imtoolrc"
_default_local_imtoolrc = "imtoolrc"

def loadImtoolrc(imtoolrc=None):
    """
        Locates, then reads in IMTOOLRC configuration file from
        system or user-specified location, and returns the 
        dictionary for reference. 
        
    """    
    # Find the IMTOOLRC file
    _home = os.getenv("HOME")
    
    # Look for path to directory where this module is installed
    # This will be last-resort location for IMTOOLRC that was
    # distributed with this module.
    _module_path = os.path.split(__file__)[0]

    _name_list = []
    _name_list.append(os.getenv(_default_imtoolrc_env[0]))
    _name_list.append(os.getenv(_default_imtoolrc_env[1]))
    _name_list.append(_default_system_imtoolrc)
    
    # Add entry for 'imtoolrc' file that comes with module
    _name_list.append(_module_path+os.sep+'imtoolrc')

    if _home:
        _name_list.append(_home+os.sep+".imtoolrc")
    _name_list.append(sys.prefix+os.sep+_default_local_imtoolrc)
    
    # Search all possible IMTOOLRC names in list
    # and open the first one found...
    for name in _name_list:
        try:
            if name:
                _fdin = open(name,'r')
                break
        except IOError, error:
            pass
    
    #Parse the file, line by line and populate the dictionary
    _lines = _fdin.readlines()
    _fdin.close()

    # Build a dictionary for the entire IMTOOL table
    # It will be indexed by configno.
    fbdict = {}
    
    for line in _lines:
        # Strip out any blanks/tabs
        line = string.strip(line)
        # Ignore empty lines
        if len(line) > 1:
            _lsp = line.split()
            # Also, ignore comment lines starting with '#'
            if _lsp[0] != '#':
                configno = int(_lsp[0])
                _dict = {'nframes':int(_lsp[1]),'width':int(_lsp[2]),'height':int(_lsp[3]),'name':_lsp[5]}
                fbdict[configno] = _dict
    return fbdict                    

def help():
    print __doc__
