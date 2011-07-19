#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup

import os
import sys

from ConfigParser import ConfigParser

from pkg_resources import (parse_requirements, working_set, safe_name,
                           safe_version)
from distutils import log
from distutils.command.build import build as _build
from distutils.command.clean import clean as _clean
from setuptools.command.develop import develop as _develop
from setuptools.command.egg_info import egg_info as _egg_info
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.install import install as _install

try:
    from nose.commands import nosetests as _nosetests
except ImportError:
    _nosetests = None


SUBDIST_DIRS = None


SUBDISTS = None


# TODO: Move most of this stuff into stsci.distutils; have it imported from
# there (have to add stsci.distutils to sys.path first, but that can be
# hard-coded and if it doesn't work nothing else will anyways)


def get_subdist_dirs():
    global SUBDIST_DIRS
    if SUBDIST_DIRS is None:
        SUBDIST_DIRS = [p for p in os.listdir('.')
                        if os.path.isdir(p) and
                        os.path.exists(os.path.join(p, 'setup.cfg'))]
    return SUBDIST_DIRS


def get_subdists():
    global SUBDISTS
    if SUBDISTS is None:
        SUBDISTS = {}
        for subdist_dir in get_subdist_dirs():
            setup_cfg = ConfigParser()
            setup_cfg.read(os.path.join(subdist_dir, 'setup.cfg'))
            if not setup_cfg.has_section('metadata'):
                continue
            elif not setup_cfg.has_option('metadata', 'name'):
                continue

            # safe_name ensures that the name will appear the same as the
            # pkg_resources requirement parser's normalization
            name = safe_name(setup_cfg.get('metadata', 'name'))

            if setup_cfg.has_option('metadata', 'version'):
                version = safe_version(setup_cfg.get('metadata', 'version'))
                subdist = (name, version)
            else:
                subdist = (name, None)

            SUBDISTS[subdist] = subdist_dir
    return SUBDISTS


# Try to bootstrap stsci.distutils, and in particular the easier_install
# command.  Unfortunately we have to patch setuptools.command.easy_install
# because of the way setuptools uses it to fetch setup_requires
try:
    from stsci.distutils.command.easier_install import easier_install
except ImportError:
    for subdist, location in get_subdists().iteritems():
        if subdist[0] != 'stsci.distutils':
            continue
        libdir = os.path.join(location, 'lib')
        try:
            sys.path.insert(0, libdir)
            from stsci.distutils.command.easier_install import easier_install
        except ImportError:
            from setuptools.command.easy_install import easy_install
            easier_install = easy_install
        finally:
            sys.path.pop(0)
import setuptools
setuptools.command.easy_install.easy_install = easier_install


# TODO: Whenever we switch to pure distutils2 this will have to be modified
def run_subdists_command(command, execsetup=None):
    requirements = parse_requirements(command.distribution.install_requires)
    for requirement in requirements:
        for subdist, subdist_dir in get_subdists().iteritems():
            subdist_name, subdist_version = subdist
            if subdist_name != requirement.project_name:
                continue
            if subdist_version not in requirement:
                # This checks that the minimum required version is met (or the
                # exact version, if the requirement is exact)
                continue
            # Okay, we have a matching subdistribution
            old_cwd = os.getcwd()
            os.chdir(subdist_dir)
            # Run the sub-distribution's setup.py with the same arguments that
            # were given the main dist's setup.py.
            try:
                log.info("running %s command in %s"
                         % (command.get_command_name(),
                            os.path.join(os.path.curdir, subdist_dir)))
                if '' not in sys.path:
                    sys.path.insert(0, '')
                if execsetup is None:
                    execfile(os.path.abspath('setup.py'))
                else:
                    execsetup()
            finally:
                os.chdir(old_cwd)
            break
        else:
            log.info('%s not found in sub-package distributions; skipping '
                     '%s...' % (requirement, command.get_command_name()))


# TODO: It might be nice to be able to wrap several command classes in a loop;
# in almost all cases except install this is doable.
class build(_build):
    def run(self):
        run_subdists_command(self)
        _build.run(self)


class clean(_clean):
    def run(self):
        run_subdists_command(self)
        _clean.run(self)


class develop(_develop):
    def run(self):
        # Here too it works best to call setup.py develop in a separate process
        def execsetup():
            try:
                os.system(' '.join(sys.argv))
            except SystemExit:
                pass
        run_subdists_command(self, execsetup=execsetup)
        # Don't run develop for the stsci_python package itself; there's
        # nothing really to develop *on*.  And it gets confused here when
        # processing dependencies, because none of the develop mode
        # distributions have been added to the default working set
        # TODO: Find a way to fix the working set problem.
        #_develop.run(self)


class egg_info(_egg_info):
    """This subclass of the egg_info command is required to cooperate with the
    sdist subclass for adding svn infos to the manifest.

    Unfortunately in setuptools/distribute the egg_info and sdist commands are
    deeply intertwined, so we need to make enhancements to both to support this
    functionality.
    """

    def run(self):
        _egg_info.run(self)
        sdist_cmd = self.get_finalized_command('sdist')
        if hasattr(sdist_cmd, '_svninfos'):
            self.filelist.extend(sdist_cmd._svninfos)


class sdist(_sdist):
    """This custom sdist collects a list of all the packages in each subproject
    and generates an svninfo module for each of those packages as necessary.
    This way the source can be distributed with full SVN info for each
    subproject.
    """

    def _get_packages(self):
        all_packages = []
        for subdist_dir in get_subdists().values():
            setup_cfg = ConfigParser()
            setup_cfg.read(os.path.join(subdist_dir, 'setup.cfg'))
            if not setup_cfg.has_option('files', 'packages'):
                continue

            packages = setup_cfg.get('files', 'packages')
            packages = [elem for elem in
                        (line.strip() for line in packages.split('\n'))
                        if elem]
            if not packages:
                continue

            if setup_cfg.has_option('files', 'packages_root'):
                packages_root = setup_cfg.get('files', 'packages_root')
            else:
                packages_root = '.'
            packages_root = os.path.join(subdist_dir, packages_root)

            all_packages.append((packages_root, packages))
        return all_packages

    def initialize_options(self):
        _sdist.initialize_options(self)
        self._svninfos = []

    # TODO: Eventually make this more intelligent, so that it builds manifests
    # for each subproject and then combines those manifests. This would allow
    # it to take into account each subproject's MANIFEST.in, if one exists.
    def run(self):
        try:
            from stsci.distutils.svnutils import (write_svn_info_for_package,
                                                  clean_svn_info_for_package)
        except ImportError:
            # Old version of stsci.distutils or something--can't write svninfo
            # modules, so just fall back on normal behavior
            return _sdist.run(self)

        all_packages = self._get_packages()
        try:
            for package_root, packages in all_packages:
                for package in packages:
                    try:
                        write_svn_info_for_package(package_root, package)
                        package_dir = os.path.join(package_root,
                                                   *(package.split('.')))
                        svninfo = os.path.join(package_dir, 'svninfo.py')
                        if os.path.exists(svninfo):
                            self._svninfos.append(svninfo)
                    except:
                        # This isn't that critical, so if it fails for any
                        # reason we just won't have the svn info for this
                        # package.
                        # TODO: Maybe at least issue a warning?
                        pass

            _sdist.run(self)
        finally:
            for package_root, packages in all_packages:
                for package in packages:
                    try:
                        clean_svn_info_for_package(package_root, package)
                    except:
                        # TODO: Again, maybe issue a warning?
                        pass


class install(_install):
    def run(self):
        install_lib = self.distribution.get_command_obj('install_lib')
        install_scripts = self.distribution.get_command_obj('install_scripts')
        install_cmd = self.distribution.get_command_obj('install')
        for cmd in (install_lib, install_scripts, install_cmd):
            cmd.ensure_finalized()

        # These are some options that will probably end up being passed to
        # easy_install in execsetup; ensure that the paths are absolute so we
        # don't get lost
        opts = {'prefix': install_cmd.prefix,
                'install-dir': install_lib.install_dir,
                'script-dir': install_scripts.install_dir,
                'record': install_cmd.record}
        for optname, value in opts.items():
            if value is not None:
                opts[optname] = os.path.abspath(value)
        opts['optimize'] = install_lib.optimize


        def execsetup(opts=opts):
            try:
                argv = sys.argv[:]
                if ('--root' not in sys.argv and
                    '--old-and-unmanageable' not in sys.argv and
                    '--single-version-externally-managed' not in sys.argv):
                    # Use easy_install to install instead; that way we can have
                    # more control over things like disabling dependency
                    # checking
                    # 'build' is inserted before 'easy_install' since the
                    # easy_install command by itself seems to squelch much of
                    # the build output
                    argv = [argv[0], 'build', 'easy_install', '--no-deps']
                    # Now, set the install-dir option from the install_lib
                    # command which, according to comments in the distribute
                    # source code "takes into account --prefix and --home and
                    # all that other crud"; set some other options as well
                    for optname, value in opts.iteritems():
                        if value is not None:
                            argv.append('--%s=%s' % (optname, value))

                    argv.append('.')

                os.system(' '.join(argv))
            except SystemExit:
                pass

        run_subdists_command(self, execsetup=execsetup)
        if self.old_and_unmanageable or self.single_version_externally_managed:
            _install.run(self)
        else:
            self.do_egg_install()


if _nosetests:
    class nosetests(_nosetests):
        def run(self):
            def execsetup():
                try:
                    # It's necessary to call os.system to run each project's
                    # tests in its own process; otherwise the tests interfere
                    # with each other too much, even if --with-isolation is
                    # used
                    os.system(' '.join(sys.argv))
                except SystemExit:
                    pass
            run_subdists_command(self, execsetup=execsetup)
            _nosetests.run(self)


CUSTOM_COMMANDS = {'build': build, 'clean': clean, 'develop': develop,
                   'egg_info': egg_info, 'install': install, 'sdist': sdist}
if _nosetests:
    CUSTOM_COMMANDS['nosetests'] = nosetests


setup(
    setup_requires=['d2to1>=0.2.3'],
    d2to1=True,
    use_2to3=True,
    cmdclass=CUSTOM_COMMANDS
)
