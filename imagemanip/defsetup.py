import distutils.core
import distutils
import sys, os.path

try:
    import numpy
except:
    raise ImportError("NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH")


if sys.platform != 'win32':
    imagemanip_libraries = ['m']
else:
    imagemanip_libraries = ['']


pkg = "imagemanip"

setupargs = {
      'version' :		"1.0",

      'description' :	"General Image Manipulation Tools",

      'author' :		"Christopher Hanley",

      'author_email' :	"help@stsci.edu",

      'license' :		"http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",

      'platforms' :		["Linux","Solaris","Mac OS X", "Windows"],

      'data_files' :	[('imagemanip',['lib/LICENSE.txt'])],

      'ext_modules' :   [
                        distutils.core.Extension('imagemanip.bilinearinterp',['src/bilinearinterp.c'],
                             include_dirs = [ distutils.sysconfig.get_python_inc(), numpy.get_include() ],
                             libraries = imagemanip_libraries)
                        ]
}
