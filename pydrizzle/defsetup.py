from distutils.core import Extension
import sys, os.path
import distutils.sysconfig

try:
    import numpy
    import numpy.numarray as nn
except ImportError:
    "Numpy was not found. It may not be installed or it may not be on your PYTHONPATH. Pydrizzle requires numpy v 1.0.2 or later.\n"

if numpy.__version__ < "1.0.2":
    raise SystemExit, "Numpy 1.0.2 or later required to build pydrizzle."

numpyinc = numpy.get_include()
numpynumarrayinc = nn.get_numarray_include_dirs()
pythoninc = distutils.sysconfig.get_python_inc()

if sys.platform != 'win32':
    EXTRA_LINK_ARGS = []
else:
    EXTRA_LINK_ARGS = ['/NODEFAULTLIB:MSVCRT']


########
# Figure out how to build libf2c

# find out where the 

if os.path.isdir("libf2c") :
    # we are in the pydrizzle directory, so everything is fine
    insert = ""
elif os.path.isdir("pydrizzle/libf2c") :
    # if traits102 is not there, I assume we are in the stsci_python
    # directory, so the pydrizzle code is in the pydrizzle/ directory.
    insert = "/pydrizzle"
else :
    print "pydrizzle libf2c directory not found - cannot proceed"
    sys.exit(0)

f2c_inc = os.getcwd() + insert + "/libf2c"

#
# Do we want the 32 or 64 bit version of libf2c?  Figure it out, then
# start our list of files with the architecture specific list
#
if type(0x7fffffff + 1) == type(1) :
    # overflowing a 32 bit signed int still resulted in an int,
    # so we must be on a 64 bit machine
    f2c_macros = [ ('Skip_f2c_Undefs',1), ('Allow_TYQUAD',1),  ('INTEGER_STAR_8',1) ]
    f2c_files = [ 'ftell64_.c', 'pow_qq.c', 'qbitbits.c', 'qbitshft.c' ]
else :
    # otherwise, we must be on a 32 bit machine.
    f2c_macros = [ ('Skip_f2c_Undefs',1) ]
    f2c_files = [ 'ftell_.c' ]

# Because of f2c.h, we get about 2000 warnings during the build.  This hack
# disables the definitions that are causing those warnings.  Apparently,
# we don't use any of them because everything still compiles.
f2c_macros.append( ('NO_UNKNOWN_PROCEDURE', 1) )

if sys.platform == 'win32' :
    print ""
    print "WINDOWS"
    print ""
    f2c_macros += [ ("USE_CLOCK",1),
        ("MSDOS",1),
        ("NO_ONEXIT",1) ]
 


#
# Here are all the files in the f2c library that are common to both 32 and
# 64 bit implementations.  Note: This is NOT just the result of "ls libf2c"!
#
f2c_files += [ 'abort_.c', 'backspac.c', 'c_abs.c', 'c_cos.c',
    'c_div.c', 'c_exp.c', 'c_log.c', 'c_sin.c', 'c_sqrt.c', 'cabs.c',
    'close.c', 'd_abs.c', 'd_acos.c', 'd_asin.c', 'd_atan.c', 'd_atn2.c',
    'd_cnjg.c', 'd_cos.c', 'd_cosh.c', 'd_dim.c', 'd_exp.c', 'd_imag.c',
    'd_int.c', 'd_lg10.c', 'd_log.c', 'd_mod.c', 'd_nint.c', 'd_prod.c',
    'd_sign.c', 'd_sin.c', 'd_sinh.c', 'd_sqrt.c', 'd_tan.c', 'd_tanh.c',
    # 'derf_.c', 'derfc_.c', 
    'dfe.c', 'dolio.c', 'dtime_.c', 'due.c',
    'ef1asc_.c', 'ef1cmc_.c', 'endfile.c', 
    # 'erf_.c', 'erfc_.c', 
    'err.c',
    'etime_.c', 'exit_.c', 'f77_aloc.c', 'f77vers.c', 'fmt.c', 'fmtlib.c',
    'getarg_.c', 'getenv_.c', 'h_abs.c', 'h_dim.c', 'h_dnnt.c', 'h_indx.c',
    'h_len.c', 'h_mod.c', 'h_nint.c', 'h_sign.c', 'hl_ge.c', 'hl_gt.c',
    'hl_le.c', 'hl_lt.c', 'i77vers.c', 'i_abs.c', 'i_dim.c', 'i_dnnt.c',
    'i_indx.c', 'i_len.c', 'i_mod.c', 'i_nint.c', 'i_sign.c', 'iargc_.c',
    'iio.c', 'ilnw.c', 'inquire.c', 'l_ge.c', 'l_gt.c', 'l_le.c',
    'l_lt.c', 'lbitbits.c', 'lbitshft.c', 'lread.c', 'lwrite.c', 'main.c',
    'open.c', 'pow_ci.c', 'pow_dd.c', 'pow_di.c', 'pow_hh.c', 'pow_ii.c',
    'pow_ri.c', 'pow_zi.c', 'pow_zz.c', 'r_abs.c', 'r_acos.c', 'r_asin.c',
    'r_atan.c', 'r_atn2.c', 'r_cnjg.c', 'r_cos.c', 'r_cosh.c', 'r_dim.c',
    'r_exp.c', 'r_imag.c', 'r_int.c', 'r_lg10.c', 'r_log.c', 'r_mod.c',
    'r_nint.c', 'r_sign.c', 'r_sin.c', 'r_sinh.c', 'r_sqrt.c', 'r_tan.c',
    'r_tanh.c', 'rdfmt.c', 'rewind.c', 'rsfe.c', 'rsli.c', 'rsne.c',
    's_cat.c', 's_cmp.c', 's_copy.c', 's_paus.c', 's_rnge.c', 's_stop.c',
    'sfe.c', 'sig_die.c', 'signal_.c', 'sue.c', 'system_.c', 'typesize.c',
    'uio.c', 'uninit.c', 'util.c', 'wref.c', 'wrtfmt.c', 'wsfe.c',
    'wsle.c', 'wsne.c', 'xwsne.c', 'z_abs.c', 'z_cos.c', 'z_div.c',
    'z_exp.c', 'z_log.c', 'z_sin.c', 'z_sqrt.c' ]

f2c_files = [ ( "libf2c/%s" % x ) for x in f2c_files ]


def getExtensions():
    ext = [Extension("pydrizzle.arrdriz",['src/arrdrizmodule.c',
                                'src/tdriz.c','src/tblot.c','src/twdriz.c',
                                'src/drutil.c','src/doblot.c','src/drcall.c',
                                'src/inter2d.c','src/bieval.c' ] + f2c_files ,
                   define_macros=[('NUMPY', '1')] + f2c_macros ,
                   include_dirs= [ pythoninc, numpyinc, f2c_inc ] + numpynumarrayinc ,
                   # extra_link_args=EXTRA_LINK_ARGS
                )]

    return ext


pkg = ['pydrizzle','pydrizzle.traits102', 'pydrizzle.distortion']

setupargs = { 
              'version' :       "6.0.0",
              'description' :   "Geometrically correct and combine images using Drizzle",
              'author' :        "Warren Hack",
              'author_email' :  "help@stsci.edu",
              'license' :       "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              'platforms' :     ["Linux","Solaris", "MacOS X", "Windows"],
              'package_dir' :   { 'pydrizzle':'lib/pydrizzle','pydrizzle.traits102':'lib/pydrizzle/traits102', 'pydrizzle.distortion':'lib/pydrizzle/distortion'},
              'data_files' :    [('pydrizzle',['LICENSE.txt'])],
              'ext_modules' :   getExtensions()
}
