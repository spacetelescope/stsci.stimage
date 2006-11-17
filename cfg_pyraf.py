from distutils.core import Extension
import sys, os.path, glob, string, commands

PYRAF_DATA_FILES = ['pyraf/data/blankcursor.xbm', 'pyraf/data/epar.optionDB', 'pyraf/data/pyraflogo_rgb_web.gif', 'data/ipythonrc-pyraf','pyraf/lib/LICENSE.txt']

PYRAF_CLCACHE = glob.glob(os.path.join('pyraf', 'data', 'clcache', '*'))

PYRAF_SCRIPTS = ['lib/pyraf']
x_libraries = 'X11'


PYRAF_LIB_DIRS = []
PYRAF_INC_DIRS = []


def find_x(xdir=""):
    if xdir != "":
        PYRAF_LIB_DIRS.append(os.path.join(xdir,'lib'))
        PYRAF_INC_DIRS.append(os.path.join(xdir,'include'))
    elif sys.platform == 'darwin' or sys.platform.startswith('linux'):
        PYRAF_LIB_DIRS.append('/usr/X11R6/lib64')
        PYRAF_LIB_DIRS.append('/usr/X11R6/lib')
        PYRAF_INC_DIRS.append('/usr/X11R6/include')
    else:
        try:
            import Tkinter
        except:
            raise ImportError("Tkinter is not installed")
        tk=Tkinter.Tk()
        tk.withdraw()
        tcl_lib = os.path.join((tk.getvar('tcl_library')), '../')
        tcl_inc = os.path.join((tk.getvar('tcl_library')), '../../include')
        tk_lib = os.path.join((tk.getvar('tk_library')), '../')
        tkv = str(Tkinter.TkVersion)[:3]
        if Tkinter.TkVersion < 8.3:
            print "Tcl/Tk v8.3 or later required\n"
            sys.exit(1)
        else:
            suffix = '.so'
            tklib='libtk'+tkv+suffix
            command = "ldd %s" % (os.path.join(tk_lib, tklib))
            lib_list = string.split(commands.getoutput(command))
            for lib in lib_list:
                if string.find(lib, 'libX11') == 0:
                    ind = lib_list.index(lib)
                    PYRAF_LIB_DIRS.append(os.path.dirname(lib_list[ind + 2]))
                    #break
                    PYRAF_INC_DIRS.append(os.path.join(os.path.dirname(lib_list[ind + 2]), '../include'))

if sys.platform != 'win32':
    x_dir = ""
    find_x(x_dir)
else:
    pass


PYRAF_EXTENSIONS = [Extension('pyraf.sscanfmodule', ['pyraf/src/sscanfmodule.c'],
                           include_dirs=PYRAF_INC_DIRS),
                  Extension('pyraf.xutilmodule', ['pyraf/src/xutil.c'],
                           include_dirs=PYRAF_INC_DIRS,
                           library_dirs=PYRAF_LIB_DIRS,
                           libraries = [x_libraries])]
