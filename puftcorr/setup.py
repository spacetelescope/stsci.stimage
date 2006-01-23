from distutils.core import setup
import sys, os.path
from distutils import sysconfig

#ver = sys.version_info
#python_exec = 'python' + str(ver[0]) + '.' + str(ver[1])
pythonlib = sysconfig.get_python_lib()
ver = sysconfig.get_python_version()
pythonver = 'python' + ver
data_dir = os.path.join(pythonlib, 'saaclean')

args = sys.argv[:]

for a in args:
    if a.startswith('--local='):
        dir = os.path.abspath(a.split("=")[1])
        sys.argv.append('--install-lib=%s' % dir)
        data_dir = os.path.join(dir, 'puftcorr')
        #remove --local from both sys.argv and args
        args.remove(a)
        sys.argv.remove(a)
    elif a.startswith('--home='):
        data_dir = os.path.join(os.path.abspath(a.split('=')[1]), 'lib', 'python', 'puftcorr')
        args.remove(a)
    elif a.startswith('--prefix='):
        data_dir = os.path.join(os.path.abspath(a.split('=')[1]), 'lib', pythonver, 'site-packages', 'puftcorr')
        args.remove(a)
    elif a.startswith('--install-data='):
        data_dir = os.path.abspath(a.split('=')[1])
        args.remove(a)
    elif a.startswith('bdist_wininst'):
        install.INSTALL_SCHEMES['nt']['data'] = install.INSTALL_SCHEMES['nt']['purelib']
        args.remove(a)



def dosetup(data_dir):
    r = setup(name = "puftcorr",
    	      version = "0.1",
              description = "Estimates and removes 'Mr. Staypuft' signal from a NICMOS exposure.",
              author = "Howard Bushouse",
              author_email = "help@stsci.edu",
	      license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris","Mac OS X"],
              packages=['puftcorr'],
              package_dir={'puftcorr':'lib'},
	      data_files = [(data_dir,['lib/LICENSE.txt'])])
		



if __name__ == "__main__":
    setup(name = "puftcorr",
    	      version = "0.1",
              description = "Estimates and removes 'Mr. Staypuft' signal from a NICMOS exposure.",
              author = "Howard Bushouse",
              author_email = "help@stsci.edu",
	      license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris","Mac OS X"],
              packages=['puftcorr'],
              package_dir={'puftcorr':'lib'},
	      data_files = [(data_dir,['lib/LICENSE.txt'])])
	

