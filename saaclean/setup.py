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
        data_dir = os.path.join(dir, 'saaclean')
        #remove --local from both sys.argv and args
        args.remove(a)
        sys.argv.remove(a)
    elif a.startswith('--home='):
        data_dir = os.path.join(os.path.abspath(a.split('=')[1]), 'lib', 'python', 'saaclean')
        args.remove(a)
    elif a.startswith('--prefix='):
        data_dir = os.path.join(os.path.abspath(a.split('=')[1]), 'lib', pythonver, 'site-packages', 'saaclean')
        args.remove(a)
    elif a.startswith('--install-data='):
        data_dir = os.path.abspath(a.split('=')[1])
        args.remove(a)
    elif a.startswith('bdist_wininst'):
        install.INSTALL_SCHEMES['nt']['data'] = install.INSTALL_SCHEMES['nt']['purelib']
        args.remove(a)

if __name__ == "__main__":
    setup(name = "saaclean",
          version = "0.5",
          description = "Estimates and removes persistent CR signal due to a prior SAA passage.",
          author = "Vicki Laidler",
          author_email = "help@stsci.edu",
          platforms = ["Linux","Solaris","Mac OS X", "Windows"],
          packages=['saaclean'],
          package_dir={'saaclean':'lib'},
          data_files = [(data_dir,['lib/SP_LICENSE'])])


