from distutils.core import setup
import sys, os.path, string, shutil

if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'final',0):
    raise SystemExit, "Python 2.3 or later required to build multidrizzle."

ver = sys.version_info
python_exec = 'python' + str(ver[0]) + '.' + str(ver[1])


def dolocal():
    """Adds a command line option --local=<install-dir> which is an abbreviation for
    'put all of multidrizzle in <install-dir>/multidrizzle'."""
    if "--help" in sys.argv:
        print >>sys.stderr
        print >>sys.stderr, " options:"
        print >>sys.stderr, "--local=<install-dir>    same as --install-lib=<install-dir>"
    for a in sys.argv:
        if a.startswith("--local="):
            dir =  os.path.abspath(a.split("=")[1])
            sys.argv.extend([
                "--install-lib="+dir,
                "--install-data="+os.path.join(dir,"multidrizzle")
                ])
            sys.argv.remove(a)

def getDataDir(args):
    for a in args:
        if string.find(a, '--home=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = os.path.join(dir, 'lib/python/multidrizzle')
        elif string.find(a, '--prefix=') == 0:
            dir = string.split(a, '=')[1]
            data_dir = os.path.join(dir, 'lib', python_exec, 'site-packages/multidrizzle')
        elif a.startswith('--install-data='):
            dir = string.split(a, '=')[1]
            data_dir = dir
        else:
            data_dir = os.path.join(sys.prefix, 'lib', python_exec, 'site-packages/multidrizzle')
    return data_dir

def copy_doc(data_dir, args):
    if 'install' in args:
        doc_dir = os.path.join(data_dir,'doc')
        if os.path.exists(doc_dir):
	    try:
                shutil.rmtree(doc_dir)
            except:
                print "Error removing doc directory\n"
        #os.mkdir(doc_dir)
	shutil.copytree('doc', doc_dir)

def dosetup(data_dir):
    r = setup(name = "multidrizzle",
              version = "2.3.6",
              description = "Automated process for HST image combination and cosmic-ray rejection",
              author = "Warren Hack, Christopher Hanley, Ivo Busko, Robert Jedrzejewski, and Anton Koekemoer",
              author_email = "help@stsci.edu",
              license = "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
              platforms = ["Linux","Solaris","Mac OS X"],
              packages=['multidrizzle'],
              package_dir={'multidrizzle':'lib'},
              data_files = [(data_dir,['lib/LICENSE.txt'])]
              )

    return r

def main():
    args = sys.argv
    dolocal()
    data_dir = getDataDir(args)
    dosetup(data_dir)
    copy_doc(data_dir, args)


if __name__ == "__main__":
    main()

