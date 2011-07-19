from __future__ import with_statement

import glob
import sys


try:
    from packaging.util import split_multiline
except ImportError:
    try:
        from distutils2.util import split_multiline
    except ImportError:
        from d2to1.util import split_multiline


try:
    reload
except NameError:
    from imp import reload


from stsci.distutils.svnutils import (write_svn_info_for_package,
                                      clean_svn_info_for_package)


def is_display_option():
    """A hack to test if one of the arguments passed to setup.py is a display
    argument that should just display a value and exit.  If so, don't bother
    running this hook (this capability really ought to be included with
    distutils2).
    """

    from setuptools.dist import Distribution

    # If there were no arguments in argv (aside from the script name) then this
    # is an implied display opt
    if len(sys.argv) < 2:
        return True

    display_opts = ['--command-packages', '--help', '-h']

    for opt in Distribution.display_options:
        display_opts.append('--' + opt[0])

    for arg in sys.argv:
        if arg in display_opts:
            return True

    return False


# TODO: With luck this can go away soon--packaging now supports adding the cwd
# to sys.path for running setup_hooks.  But it also needs to support adding
# packages_root.  Also, it currently does not support adding cwd/packages_root
# to sys.path for pre/post-command hooks, so that needs to be fixed.
def use_packages_root(config):
    """
    Adds the path specified by the 'packages_root' option, or the current path
    if 'packages_root' is not specified, to sys.path.  This is particularly
    useful, for example, to run setup_hooks or add custom commands that are in
    your package's source tree.
    """

    if 'files' in config and 'packages_root' in config['files']:
        root = config['files']['packages_root']
    else:
        root = ''

    if root not in sys.path:
        if root and sys.path[0] == '':
            sys.path.insert(1, root)
        else:
            sys.path.insert(0, root)

    # Reload the stsci namespace package in case any new paths can be added to
    # it from the new sys.path entry
    if 'stsci' in sys.modules:
        reload(sys.modules['stsci'])


def glob_data_files(config):
    """
    Allows wildcard patterns to be used in the data_files option.
    """

    if 'files' in config and 'data_files' in config['files']:
        data_files = config['files']['data_files']
    else:
        return

    # The unfortunate thing about setup_hooks is that it doesn't split lines or
    # do any processing on the config values before running the hooks, so the
    # hook has to duplicate any effort in processing values that it works on
    # TODO: Suggest a fix to this...?
    data_files = split_multiline(data_files)
    for idx, val in enumerate(data_files):
        dest, filenames = (item.strip() for item in val.split('=', 1))
        filenames = sum((glob.glob(item.strip())
                        for item in filenames.split()), [])
        data_files[idx] = '%s = %s' % (dest, ' '.join(filenames))

    config['files']['data_files'] = '\n'.join(data_files)


def svn_info_pre_hook(command_obj):
    """This command hook creates an svninfo.py file in each package that
    requires SVN info.  This is by determining if the package's __init__ tries
    to set either __svn_version__ or __full_svn_info__.  That is, it contains
    an import of or from the svninfo module.  svninfo.py will not be created
    in packages that don't use it.  It should really only be used in the main
    package of the project.
    """

    package_dir = command_obj.distribution.package_dir.get('', '.')
    packages = command_obj.distribution.packages

    for package in packages:
        write_svn_info_for_package(package_dir, package)


def svn_info_post_hook(command_obj):
    """Cleans up a previously generated svninfo.py in order to avoid
    clutter.

    Only removes the file if we're in an SVN working copy and the file is not
    already under version control.
    """

    package_dir = command_obj.distribution.package_dir.get('', '.')
    packages = command_obj.distribution.packages

    for package in packages:
        clean_svn_info_for_package(package_dir, package)


def numpy_extension_hook(command_obj):
    """A distutils2 pre-command hook for the build_ext command needed for
    building extension modules that use NumPy.

    To use this hook, add 'numpy' to the list of include_dirs in setup.cfg
    section for an extension module.  This hook will replace 'numpy' with the
    necessary numpy header paths in the include_dirs option for that extension.

    Note: Although this function uses numpy, stsci.distutils does not depend on
    numpy.  It is up to the distribution that uses this hook to require numpy
    as a dependency.
    """

    try:
        import numpy
    except ImportError:
        # It's virtually impossible to automatically install numpy through
        # setuptools; I've tried.  It's not pretty.
        # Besides, we don't want users complaining that our software doesn't
        # work just because numpy didn't build on their system.
        sys.stderr.write('\n\nNumpy is required to build this package.\n'
                         'Please install Numpy on your system first.\n\n')
        sys.exit(1)

    includes = [numpy.get_numarray_include(), numpy.get_include()]
    for extension in command_obj.extensions:
        if 'numpy' not in extension.include_dirs:
            continue
        idx = extension.include_dirs.index('numpy')
        for inc in includes:
            extension.include_dirs.insert(idx, inc)
        extension.include_dirs.remove('numpy')
