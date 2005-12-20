from distutils.core import setup
import sys, os.path, string, shutil


ver = sys.version_info
python_exec = 'python' + str(ver[0]) + '.' + str(ver[1])


def dolocal():
    """Adds a command line option --local=<install-dir> which is an abbreviation for
    'put all of puftcorr in <install-dir>/puftcorr'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir =  os.path.abspath(a.split("=")[1])
            sys.argv.extend([
                "--install-lib="+dir,
                "--install-data="+os.path.join(dir,"puftcorr")
                ])
            sys.argv.remove(a)

def getDataDir(args):
    for a in args:
        if string.find(a, '--home=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = os.path.join(dir, 'lib/python/puftcorr')
        elif string.find(a, '--prefix=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = os.path.join(dir, 'lib', python_exec, 'site-packages/puftcorr')
        elif a.startswith('--install-data='):
            dir = string.split(a, '=')[1]
            data_dir = dir
        else:
            data_dir = os.path.join(sys.prefix, 'lib', python_exec, 'site-packages/puftcorr')
    return data_dir


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
		

    return r

def main():
    args = sys.argv
    dolocal()
    data_dir = getDataDir(args)
    dosetup(data_dir)


if __name__ == "__main__":
    main()

