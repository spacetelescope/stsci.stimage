"""module iraffunctions.py -- IRAF emulation tasks and functions

This is not usually used directly -- the relevant public classes and
functions get included in iraf.py.  The implementations are kept here
to avoid possible problems with name conflicts in iraf.py (which is the
home for all the IRAF task and package names.)  Private routines have
names beginning with '_' and do not get imported by the iraf module.

The exception is that iraffunctions can be used directly for modules
that must be compiled and executed early, before the pyraf module
initialization is complete.

$Id$

R. White, 2000 January 20
"""

# define INDEF, yes, no, EOF, Verbose, userIrafHome

import sys, os, string, re, types, time

try:
    import iraf
except:
    iraf = None

# hide these modules so we can use 'from iraffunctions import *'
_sys = sys
_os = os
_string = string
_re = re
_types = types
_time = time

del sys, os, string, re, types, time

# -----------------------------------------------------
# private dictionaries:
#
# _varDict: dictionary of all IRAF cl variables (defined with set name=value)
# _tasks: all IRAF tasks (defined with task name=value)
# _mmtasks: minimum-match dictionary for tasks
# _pkgs: min-match dictionary for all packages (defined with
#                       task name.pkg=value)
# _loaded: loaded packages
# -----------------------------------------------------

# Will want to enhance this to allow a "bye" function that unloads packages.
# That might be done using a stack of definitions for each task.

_varDict = {}


# module variables that don't get saved (they get
# initialized when this module is imported)

unsavedVars = [
                        'EOF',
                        '_NullFile',
                        '_NullPath',
                        '__builtins__',
                        '__doc__',
                        '__file__',
                        '__name__',
                        '__re_var_match',
                        '__re_var_paren',
                        '_badFormats',
                        '_clearString',
                        '_exitCommands',
                        '_unsavedVarsDict',
                        '_radixDigits',
                        '_re_taskname',
                        '_sttyArgs',
                        'no',
                        'yes',
                        'userWorkingHome',
                        ]
_unsavedVarsDict = {}
for v in unsavedVars: _unsavedVarsDict[v] = 1
del unsavedVars, v


# -----------------------------------------------------
# Miscellaneous access routines:
# getVarList: Get list of names of all defined IRAF variables
# -----------------------------------------------------

def getVarDict():
    """Returns dictionary all IRAF variables"""
    return _varDict

def getVarList():
    """Returns list of names of all IRAF variables"""
    return _varDict.keys()

# -----------------------------------------------------
# listVars:
# list contents of the dictionaries
# -----------------------------------------------------

def listVars(prefix="", equals="\t= ", **kw):
    """List IRAF variables"""
    keylist = getVarList()
    if len(keylist) == 0:
        print 'No IRAF variables defined'
    else:
        keylist.sort()
        for word in keylist:
            print "%s%s%s%s" % (prefix, word, equals, envget(word))


def untranslateName(s):

    """Undo Python conversion of CL parameter or variable name"""

    s = s.replace('DOT', '.')
    s = s.replace('DOLLAR', '$')
    # delete 'PY' at start of name components
    if s[:2] == 'PY': s = s[2:]
    s = s.replace('.PY', '.')
    return s

def envget(var,default=None):
    """Get value of IRAF or OS environment variable"""
    try:
        if iraf:
            return iraf.envget(var)
        else:
            raise KeyError
    except KeyError:
        try:
            return _varDict[var]
        except KeyError:
            try:
                return _os.environ[var]
            except KeyError:
                if default is not None:
                    return default
                elif var == 'TERM':
                    # Return a default value for TERM
                    # TERM gets caught as it is found in the default
                    # login.cl file setup by IRAF.
                    print "Using default TERM value for session."
                    return 'xterm'
                else:
                    raise KeyError("Undefined environment variable `%s'" % var)

def osfn(filename):
    """Convert IRAF virtual path name to OS pathname"""

    # Try to emulate the CL version closely:
    #
    # - expands IRAF virtual file names
    # - strips blanks around path components
    # - if no slashes or relative paths, return relative pathname
    # - otherwise return absolute pathname

    ename = Expand(filename)
    dlist = ename.split(_os.sep)
    dlist = map(_string.strip, dlist)
    if len(dlist)==1 and dlist[0] not in [_os.curdir,_os.pardir]:
        return dlist[0]

    # I use string.join instead of os.path.join here because
    # os.path.join("","") returns "" instead of "/"

    epath = _os.sep.join(dlist)
    fname = _os.path.abspath(epath)
    # append '/' if relative directory was at end or filename ends with '/'
    if fname[-1] != _os.sep and dlist[-1] in ['', _os.curdir,_os.pardir]:
        fname = fname + _os.sep
    return fname

def defvar(varname):
    """Returns true if CL variable is defined"""
    if iraf:
        _irafdef = iraf.envget(varname)
    else:
        _irafdef = 0
    return _varDict.has_key(varname) or _os.environ.has_key(varname) or _irafdef

# -----------------------------------------------------
# IRAF utility procedures
# -----------------------------------------------------

# these have extra keywords (redirection, _save) because they can
# be called as tasks

def set(*args, **kw):
    """Set IRAF environment variables"""
    if len(args) == 0:
        if len(kw) != 0:
            # normal case is only keyword,value pairs
            msg = []
            for keyword, value in kw.items():
                keyword = untranslateName(keyword)
                svalue = str(value)
                _varDict[keyword] = svalue
        else:
            # set with no arguments lists all variables (using same format
            # as IRAF)
            listVars(prefix="    ", equals="=")
    else:
        # The only other case allowed is the peculiar syntax
        # 'set @filename', which only gets used in the zzsetenv.def file,
        # where it reads extern.pkg.  That file also gets read (in full cl
        # mode) by clpackage.cl.  I get errors if I read this during
        # zzsetenv.def, so just ignore it here...
        #
        # Flag any other syntax as an error.
        if len(args) != 1 or len(kw) != 0 or \
                        type(args[0]) != _types.StringType or args[0][:1] != '@':
            raise SyntaxError("set requires name=value pairs")

# currently do not distinguish set from reset
# this will change when keep/bye/unloading are implemented

reset = set

def show(*args, **kw):
    """Print value of IRAF or OS environment variables"""
    if len(kw):
        raise TypeError('unexpected keyword argument: ' + `kw.keys()`)

    if args:
        for arg in args:
            print envget(arg)
    else:
        # print them all
        listVars(prefix="    ", equals="=")

def unset(*args, **kw):
    """Unset IRAF environment variables

    This is not a standard IRAF task, but it is obviously useful.
    It makes the resulting variables undefined.  It silently ignores
    variables that are not defined.  It does not change the os environment
    variables.
    """
    if len(kw) != 0:
        raise SyntaxError("unset requires a list of variable names")
    for arg in args:
        if _varDict.has_key(arg):
            del _varDict[arg]

def time(**kw):
    """Print current time and date"""
    print _time.ctime(_time.time())

# -----------------------------------------------------
# Expand: Expand a string with embedded IRAF variables
# (IRAF virtual filename)
# -----------------------------------------------------

# Input string is in format 'name$rest' or 'name$str(name2)' where
# name and name2 are defined in the _varDict dictionary.  The
# name2 string may have embedded dollar signs, which are ignored.
# There may be multiple embedded parenthesized variable names.
#
# Returns string with IRAF variable name expanded to full host name.
# Input may also be a comma-separated list of strings to Expand,
# in which case an expanded comma-separated list is returned.

# search for leading string without embedded '$'
__re_var_match = _re.compile(r'(?P<varname>[^$]*)\$')

# search for string embedded in parentheses
__re_var_paren = _re.compile(r'\((?P<varname>[^()]*)\)')

def Expand(instring, noerror=0):
    """Expand a string with embedded IRAF variables (IRAF virtual filename)

    Allows comma-separated lists.  Also uses os.path.expanduser to
    replace '~' symbols.
    Set noerror flag to silently replace undefined variables with just
    the variable name or null (so Expand('abc$def') = 'abcdef' and
    Expand('(abc)def') = 'def').  This is the IRAF behavior, though it
    is confusing and hides errors.
    """
    # call _expand1 for each entry in comma-separated list
    wordlist = instring.split(",")
    outlist = []
    for word in wordlist:
        outlist.append(_os.path.expanduser(_expand1(word, noerror=noerror)))
    return ",".join(outlist)

def _expand1(instring, noerror):
    """Expand a string with embedded IRAF variables (IRAF virtual filename)"""
    # first expand names in parentheses
    # note this works on nested names too, expanding from the
    # inside out (just like IRAF)
    mm = __re_var_paren.search(instring)
    while mm is not None:
        # remove embedded dollar signs from name
        varname = mm.group('varname').replace('$','')
        if defvar(varname):
            varname = envget(varname)
        elif noerror:
            varname = ""
        else:
            raise ValueError,"Undefined variable `%s' in string `%s'" % (varname, instring)

        instring = instring[:mm.start()] + varname + instring[mm.end():]
        mm = __re_var_paren.search(instring)
    # now expand variable name at start of string
    mm = __re_var_match.match(instring)
    if mm is None:
        return instring
    varname = mm.group('varname')
    if defvar(varname):
        # recursively expand string after substitution
        return _expand1(envget(varname) + instring[mm.end():], noerror)
    elif noerror:
        return _expand1(varname + instring[mm.end():], noerror)
    else:
        raise ValueError,"Undefined variable `%s' in string `%s'" % (varname, instring)


def access(filename):
    """Returns true if file exists"""
    return _os.path.exists(Expand(filename))
