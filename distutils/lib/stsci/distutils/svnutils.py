"""Functions for getting and saving SVN info for distribution."""


from __future__ import with_statement

import datetime
import os
import subprocess

from stsci.distutils.astutils import ImportVisitor, walk


def get_svn_rev(path='.'):
    """Uses `svnversion` to get just the latest revision at the given path."""

    try:
        pipe = subprocess.Popen(['svnversion', path], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except OSError:
        return None

    if pipe.wait() != 0:
        return None

    return pipe.stdout.read().decode('ascii').strip()


def get_svn_info(path='.'):
    """Uses `svn info` to get the full information about the working copy at
    the given path.
    """

    path = os.path.abspath(path)

    try:
        pipe = subprocess.Popen(['svn', 'info', path], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        # stderr is redirected in order to squelch it.  Later Python versions
        # have subprocess.DEVNULL for this purpose, but it's not available in
        # 2.5
    except OSError:
        return 'unknown'

    if pipe.wait() != 0:
        return 'unknown'

    lines = []
    for line in pipe.stdout.readlines():
        line = line.decode('ascii').strip()
        if not line:
            continue
        if line.startswith('Path:'):
            line = 'Path: %s' % os.path.basename(path)
        lines.append(line)

    if not lines:
        return 'unknown'

    return '\n'.join(lines)


def write_svn_info(path='.', filename='svninfo.py'):
    rev = get_svn_rev(path)

    # if we are unable to determine the revision, we default to leaving the
    # existing revision file unchanged.  Otherwise, we fill it in with whatever
    # we have

    if rev is None:
        if os.path.exists(filename):
            return
        rev = 'Unable to determine SVN revision'
    else:
        if rev in ('exported', 'unknown') and os.path.exists(filename):
            return

    info = get_svn_info(path)

    with open(filename, 'w') as f:
        f.write('__svn_version__ = %s\n' % repr(rev))
        f.write('__full_svn_info__ = """\n%s\n"""\n\n' % info)


def set_setup_date(filename='svninfo.py'):
    """Update the svninfo.py with the last time a setup command was run."""

    if not os.path.exists(filename):
        return

    d = datetime.datetime.now()

    lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    with open(filename, 'w') as f:
        for line in lines:
            if not line.endswith('# setupdatetime\n'):
                f.write(line)
        f.write('import datetime # setupdatetime\n')
        f.write('__setup_datetime__ = %s # setupdatetime\n' % repr(d))


def write_svn_info_for_package(package_root, package):
    """Conditionally creates an svninfo.py module in package if there are are
    any imports from svninfo in that package's __init__.py.  See also
    stsci.distutils.hooks.svn_info_pre_hook where this is used.
    """

    pdir = os.path.join(package_root, *(package.split('.')))
    init = os.path.join(pdir, '__init__.py')
    if not os.path.exists(init):
        # Not a valid package
        # TODO: Maybe issue a warning here?
        return

    try:
        visitor = ImportVisitor()
        walk(init, visitor)
    except SyntaxError:
        # TODO: Maybe issue a warning?
        pass

    found = False
    # Check the import statements parsed from the file for an import of or
    # from the svninfo module in this package
    for imp in visitor.imports:
        if imp[0] in ('svninfo', '.'.join((package, 'svninfo'))):
            found = True
            break
    for imp in visitor.importfroms:
        mod = imp[0]
        name = imp[1]
        if (mod in ('svninfo', '.'.join((package, 'svninfo'))) or
            (mod == package and name == 'svninfo')):
            found = True
            break

    if not found:
        return

    # Write the svninfo.py file, or just touch it if it already exists
    svninfo = os.path.join(pdir, 'svninfo.py')
    write_svn_info(filename=svninfo)
    set_setup_date(svninfo)


def clean_svn_info_for_package(package_root, package):
    """Removes the generated svninfo.py module from a package, but only if
    we're in an SVN working copy.
    """

    pdir = os.path.join(package_root, *(package.split('.')))
    svninfo = os.path.join(pdir, 'svninfo.py')
    if not os.path.exists(svninfo):
        return

    try:
        pipe = subprocess.Popen(['svn', 'status', svninfo],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except OSError:
        return

    if pipe.wait() != 0:
        return

    # TODO: Maybe don't assume ASCII here.  Find out the best way to handle
    # this.
    if not pipe.stdout.read().decode('ascii').startswith('?'):
        return

    os.remove(svninfo)

