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
                "--install-lib="+dir
                ])
            sys.argv.remove(a)


def dosetup():
    r = setup(name = "puftcorr",
              version = "0.1",
              description = "Estimates and removes estimating and removing 'Mr. Staypuft' signal from a NICMOS exposure.",
              author = "",
              author_email = "help@stsci.edu",
              platforms = ["Linux","Solaris","Mac OS X"],
              packages=['puftcorr'],
              package_dir={'puftcorr':'lib'})

    return r

def main():
    args = sys.argv
    dolocal()
    dosetup()


if __name__ == "__main__":
    main()

