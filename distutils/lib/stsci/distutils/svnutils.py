"""Functions for getting and saving SVN info for distribution."""


from __future__ import with_statement

import datetime
import os
import subprocess


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

    lines = filter(None, [l.strip() for l in pipe.stdout.readlines()])

    if not lines:
        return 'unknown'

    return '\n'.join(l.decode('ascii') for l in lines)


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

