#!/usr/bin/env python

"""Python does not have a facility to uninstall packages.  This program
will attempt to locate and (optionally) remove Python packages or
modules that look like part of STSCI_PYTHON 2.3, 2.4, 2.5, or 2.6.

It will search sys.path (initialized by Python from internal values
and your environment variable PYTHONPATH).  If it finds a thing
that looks like part of STSCI_PYTHON, the program offers to delete
it.  If you type "y" and press enter, it will try to delete it.  If
you type "n" or just press enter, it will skip that item.

It recognizes parts of STSCI_PYTHON by file or directory names.  If
you have other files that have the same names, this program cannot
recognize that.  But you can -- if this program offers to delete
something that should not be deleted, answer "n".

If there is an error deleting a package, the first error will be
reported and the rest of that package will not be deleted.
"""


import glob
import optparse
import os
import shutil
import sys

from ConfigParser import ConfigParser


# List of old packages to remove; some of these have been renamed, others no
# longer exist
OLD_PACKAGES = set([
    'acstools',
    'astrodrizzle',
    'calcos',
    'convolve',
    'convolve',
    'coords',
    'costools',
    'image',
    'image',
    'imagemanip',
    'imagemanip',
    'imagestats',
    'imagestats',
    'multidrizzle',
    'ndimage',
    'ndimage',
    'nictools',
    'numdisplay',
    'numdisplay',
    'opuscoords',
    'puftcorr',
    'pydrizzle',
    'pyfits',
    'pyraf',
    'pysynphot',
    'pytools',
    'pywcs',
    'reftools',
    'rnlincor',
    'saaclean',
    'sample_package',
    'stimage',
    'stimage',
    'stistools',
    'stsci',
    'stsci_sphinxext',
    'stscidocs',
    'stwcs',
    'wfc3tools',
    'wfpc2tools',
    ])


OLD_SCRIPTS = set(['sample_package'])

# Only older versions had modules (i.e. single .py files)
OLD_MODULES = set(['evaldisp', 'fileutil', 'fitsdiff', 'gettable', 'gfit',
                   'imageiter', 'irafglob', 'iterfile', 'linefit', 'makewcs',
                   'mktrace', 'nimageiter', 'nmpfit', 'numcombine',
                   'numerixenv', 'parseinput', 'r_util', 'radialvel',
                   'readgeis', 'sshift', 'stisnoise', 'testutil',
                   'versioninfo', 'wavelen', 'wcsutil', 'wx2d', 'xyinterp',
                   'pyfits'])


class Uninstaller(object):
    def __init__(self, force=False, never=False, old=False):
        self.force = force
        self.never = never
        self.old = old

        # Include stsci_python itself by default
        self.all_distributions = set(['stsci_python'])
        self.all_packages = OLD_PACKAGES
        self.all_modules = OLD_MODULES
        self.all_scripts = OLD_SCRIPTS

        self._build_uninstall_sets()

    @classmethod
    def main(cls):
        parser = optparse.OptionParser(__doc__)
        # -y is now spelled -f
        parser.add_option('-f', '--force', action='store_true',
                          help=optparse.SUPPRESS_HELP)
        parser.add_option('-n', '--never', action='store_true',
                          help=optparse.SUPPRESS_HELP)
        parser.add_option('-o', '--old', action='store_true',
                          help=optparse.SUPPRESS_HELP)
        options, args = parser.parse_args()

        uninstaller = cls(force=options.force, never=options.never,
                          old=options.old)
        uninstaller.run()

    def run(self):
        print __doc__
        if self.never or not self._ask('continue'):
            sys.exit(0)

        # look for scisoft; issue a warning if it is found
        has_scisoft = False
        for item in sys.path:
            if 'scisoft' in item:
                has_scisoft = True
                break
        if os.path.isdir('/Applications/scisoft') or os.path.isdir('/scisoft'):
            has_scisoft = True

        if has_scisoft:
            print ('It looks like you may have Scisoft on this machine.  We '
                   'often receive reports of difficulty when trying to '
                   'upgrade the STSDAS or STSCI_PYTHON software that is in '
                   'the Scisoft distribution.  It may be helpful to contact '
                   'the distributors of Scisoft if you have problems.')

            if self.never or not self._ask('continue'):
                sys.exit(0)

        # found_any will be set if we found anything that we consider deleting
        found_any = False

        cwd = os.path.abspath(os.getcwd())
        # search the whole sys.path for anything that might be ours
        for item in sys.path:
            item = os.path.abspath(item)
            if item == cwd:
                # skip current directory - it is probably the new stuff
                continue

            if os.path.isdir(item):
                if item.endswith('.egg'):
                    eggname = os.path.basename(item)
                    # An unzipped egg directory--most of our distributions are
                    # installed this way now
                    # Check that this is one of our dists
                    distname = eggname.split('-', 1)[0]
                    if distname not in self.all_distributions:
                        continue

                    found_any = True
                    if self._ask('delete distribution %s' % item):
                        if self.old:
                            print 'renaming %s to %s.old' % (item, item)
                            old = item + '.old'
                            if os.path.exists(old):
                                shutil.rmtree(old)
                            os.rename(item, old)
                        else:
                            shutil.rmtree(item)

                        # Now we need to remove this dist from the
                        # easy-install.pth The parent directory, presumably the
                        # site-packages, that the egg is installed in
                        pardir = os.path.dirname(item)
                        easy_install_pth = os.path.join(pardir,
                                                        'easy-install.pth')
                        if not os.path.exists(easy_install_pth):
                            continue
                        contents = open(easy_install_pth).readlines()
                        out = open(easy_install_pth, 'w')
                        for line in contents:
                            line = line.strip()
                            path = os.path.abspath(
                                    os.path.join(pardir,
                                                 os.path.normpath(line)))
                            if path == item:
                                continue
                            out.write(line + '\n')
                        out.close()
                        continue

                # look for our packages or modules in the named directory
                for package in self.all_packages:
                    package_dir = os.path.join(item, package)
                    if os.path.isdir(package_dir):
                        found_any = True
                        if self._ask('delete package %s' % package_dir):
                            if self.old:
                                print ('renaming %s to %s.old' %
                                       (package_dir, package_dir))
                                old = package_dir + '.old'
                                if os.path.exists(old):
                                    shutil.rmtree(old)
                                os.rename(package_dir, old)
                            else:
                                shutil.rmtree(package_dir)

                for module in self.all_modules:
                    module_files = glob.glob(module + '.py*')
                    errors = []
                    for module_file in module_files:
                        module_file = os.path.join(item, module_file)
                        if os.path.isfile(module_file):
                            found_any = True
                            if self._ask('delete module %s' % module_file):
                                if self.old and module_file.endswith('.py'):
                                    # saves the py file only
                                    print ('renaming %s to %s.old' %
                                           (module_file, module_file))
                                    old = module_file + '.old'
                                    if os.path.exsts(old):
                                        os.remove(old)
                                    os.rename(module_file, old)
                                else:
                                    try:
                                        os.remove(module_file)
                                    except Exception, e:
                                        errors.append((module_file, unicode(e)))

                    # Only display an error message if none of the files
                    # associated with the module were deleted
                    if errors and len(errors) == len(module_files):
                        print ('The following module files could not be '
                               'deleted:')
                        for filename, msg in errors:
                            print '\t' + filename
                            print '\t\t' + msg
            elif os.path.isfile(item):
                pass
                # this is probably either a mistake or an egg file.  either way,
                # we don't know what to do with it.  (But it doesn't matter --
                # stsci_python does not create egg files)
            else:
                pass
                # not a file _or_ a directory?  whatever...

        # search the system path for scripts.  This is only going to
        # work on unix-like systems, but the only non-unix system we
        # support at all is Windows, and people are probably using the
        # windows-installer there.
        # TODO: However, on Windows scripts get installed to
        # <PythonX.Y>/Scripts by default, so this could probably be checked...
        path = os.getenv('PATH')
        if path:
            path = path.split(os.pathsep)
        for item in path:
            item = os.path.abspath(item)
            if item == cwd :
                continue
            if not os.path.isdir(item):
                continue

            for script in self.all_scripts:
                script_file = os.path.join(item, script)
                if os.access(script_file ,os.X_OK):
                    found_any = True
                    if self._ask('delete script %s' % script_file):
                        try:
                            if self.old:
                                print ('renaming %s to %s.old' %
                                       (script_file, script_file))
                                os.rename(script_file, script_file + '.old')
                            else:
                                os.remove(script_file)
                        except Exception, e:
                            print script_file, unicode(e)

        if not found_any:
            print '\n\nDid not find anything to uninstall.\n\n'

        # I do not know that we can be sure that we actually got everything.
        # Try to import things; if anything actually imports, it is still out
        # there somewhere.
        still_found = False

        for package in self.all_packages:
            try:
                __import__(package)
                still_found = True
                print 'package %s - not completely uninstalled' % package
            except ImportError:
                pass
            except:
                # it raised an exception while importing, so it must be there
                still_found = True
                print ('package %s - maybe not completely uninstalled ' %
                       package)

        for module in self.all_modules:
            try:
                __import__(module)
                still_found = True
                print 'module %s - not completely uninstalled' % module
            except ImportError:
                pass
            except:
                # it raised an exception while importing, so it must be there
                still_found = True
                print 'module %s - maybe not completely uninstalled' % module

        if not found_any and still_found:
            print ('This program could not find anything to uninstall, but is '
                   'still able to import stsci_python packages/modules.  '
                   'Something strange is happening.  Maybe you have '
                   'STSCI_PYTHON packages or modules stored in a python egg '
                   'or a zip file?')

    def _build_uninstall_sets(self):
        """Scan the setup.cfgs for the names of packages and scripts to
        uninstall.
        """

        for path in os.listdir('.'):
            setup_cfg = os.path.join(path, 'setup.cfg')
            if path == 'setup.cfg':
                # The setup.cfg for stsci_python itself
                setup_cfg = path

            if not os.path.isfile(setup_cfg):
                continue

            cfg = ConfigParser()
            cfg.read(setup_cfg)

            if not (cfg.has_section('metadata') and
                    cfg.has_option('metadata', 'name')):
                # What is this?
                continue

            # We're not worried about version number--just delete any matching
            # dists
            self.all_distributions.add(cfg.get('metadata', 'name'))

            # Get any scripts installed by this dist
            if cfg.has_option('files', 'scripts'):
                for script in cfg.get('files', 'scripts').split('\n'):
                    script = script.strip()
                    if script:
                        self.all_scripts.add(os.path.basename(script))

            # Though none of these packages should be installed outside the
            # dist, may as well...
            if cfg.has_option('files', 'packages'):
                for package in cfg.get('files', 'packages').split('\n'):
                    # Just take the top-level package...
                    package = package.strip()
                    if package:
                        self.all_packages.add(package.split('.', 1)[0])

    def _ask(self, question):
        sys.stdout.write('\n')
        sys.stdout.write(question)
        # Force is an undocumented feature unintended for users
        if self.force:
            sys.stdout.write('\n')
            return True
        if self.never:
            return False

        sys.stdout.write('\n    (y/n)? ')
        sys.stdout.flush()
        n = sys.stdin.readline()
        if len(n) > 0:
            n = n[0]
        return n.lower() == 'y'


if __name__ == '__main__':
    Uninstaller.main()
