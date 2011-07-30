#!/usr/bin/env python

# $Id: $

# TODO: Make this work in Python3 at some point (though only once all our
# packages are Python 3 compatible anyways)

from __future__ import with_statement

try:
    import pkg_resources
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    import pkg_resources

import optparse
import os
import re
import sys
import warnings

from ConfigParser import ConfigParser

# Other non-stsci packages to include in version report
ALSO_REPORT = [
        'Sybase',
        'IPython',
        'matplotlib',
        'nose',
        'distribute',
        'setuptools',
        'urwid'
]


VERSION_SPEC_RE = re.compile(r'\s*(.+?)\s*(?:\((.*)\))?\s*$')


# This variable will be filled by calls to get_requirements() to save off the
# names of the directories in which stsci_python subprojects are found
# (getting a little pasta-like in here, but we can fix it later)
SUBPROJECT_DIRS = {}


class RequirementError(Exception):
    """Indicates that some requirement was not met."""


class NullDevice(object):
    """For quieting stdout."""

    def write(self, s):
        pass


def parse_requirement(req):
    return list(pkg_resources.parse_requirements(req))[0]


def parse_requires_dist_item(requirement):
    """Parses a requirement listing from the requires-dist option in setup.cfg
    and returns a Requirement object.
    """

    m = VERSION_SPEC_RE.match(requirement)
    if not m:
        return
    # m.groups(1) + m.groups(2) is the setuptools-compatible requirement
    # predicate, eg. foo==0.1
    req_name = pkg_resources.safe_name(m.group(1))
    pred = m.group(2) or ''
    if pred and pred[0] not in ('=', '<', '>', '!'):
        # If just a version number is specified, the default comparison
        # operator is ==
        pred = '==' + pred
    req = parse_requirement(req_name + pred)
    # More convenient than using req.specs directly:
    if len(req.specs) == 1 and req.specs[0][0] == '==':
        req.version = req.specs[0][1]
    else:
        # The original version predicate supplied in the setup.cfg
        req.version = pred
    req.predicate = pred
    if not req.version:
        req.version = 'any'

    return req


def split_list(s):
    return [item for item in
            (line.strip() for line in s.split('\n')) if item]


def get_requirements():
    """Returns a dictionary keyed on distribution names that contains both
    all the requirements for stsci_python, and all of their requirements.
    """

    # First get the full requirements for stsci_python
    requirements = {}
    # For the sake of convenience we don't specify the version of stsci_python
    _, _, requirements[('stsci_python', '')] = get_requires_dist('setup.cfg')
    SUBPROJECT_DIRS['stsci_python'] = '.'

    # Great, now traverse the subdirectories for those requirements--don't
    # worry for now if anything is missing--we'll worry about that later
    for path in os.listdir('.'):
        if not os.path.isdir(path):
            continue
        setup_cfg = os.path.join(path, 'setup.cfg')
        if not os.path.exists(setup_cfg):
            continue

        name, version, reqs = get_requires_dist(setup_cfg)
        if not name:
            continue

        required_project = False
        for req in requirements[('stsci_python', '')].values():
            if req.key == name:
                required_project = True
                break

        if not required_project:
            # This isn't a subproject we actually care about
            continue

        requirements[(name, version)] = reqs
        SUBPROJECT_DIRS[name] = path

    return requirements


def get_requires_dist(filename):
    """Reads the requirements from a setup.cfg file and returns a tuple of the
    distribution name, its version, and a dict of its Requirement specs.
    """

    if not os.path.exists(filename):
        raise EnvironmentError('setup.cfg missing--your stsci_python source '
                               'distribution is somehow incomplete')
    setup_cfg = ConfigParser()
    setup_cfg.read(filename)

    if not setup_cfg.has_option('metadata', 'name'):
        return '', '', {}

    name = pkg_resources.safe_name(setup_cfg.get('metadata', 'name'))

    if setup_cfg.has_option('metadata', 'version'):
        version = pkg_resources.safe_version(
                setup_cfg.get('metadata', 'version'))
    else:
        version = ''

    if not setup_cfg.has_option('metadata', 'requires-dist'):
        # This really shouldn't happen...
        return name, version, {}

    requires = split_list(setup_cfg.get('metadata', 'requires-dist'))

    requirements = {}
    for requirement in requires:
        req = parse_requires_dist_item(requirement)
        if req:
            requirements[req.key] = req
        else:
            warnings.warn('Invalid requirement specifier in %s: %s' %
                          (os.path.abspath(filename), requirement))
    return name, version, requirements


def get_dist_info(name):
    """Returns the location and version of the given distribution.  If the
    distribution is not found via pkg_resources, we assume it's the name of a
    Python module and try to import it to see if we can find anything.
    """

    try:
        dist = pkg_resources.get_distribution(name)
        version, location = dist.version, dist.location
    except Exception, e:
        # Some modules are noisy when you import them...
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = NullDevice()
        sys.stderr = NullDevice()
        try:
            if '.' in name:
                # If this is a namespace package just importing the full name
                # won't work as expected
                parts = name.split('.')
                pkg = __import__(parts.pop(0))
                if len(pkg.__path__) > 1:
                    # This is a namespace package, presumably
                    mod = pkg
                    while parts:
                        mod = getattr(mod, parts.pop(0))
            else:
                mod = __import__(name)
            if hasattr(mod, '__path__'):
                # This doesn't tell the whole story for a namespace package,
                # but I don't think we're likely to deal with any here
                location = mod.__path__[0]
            elif hasattr(mod, '__file__'):
                location = mod.__file__
            else:
                # This shouldn't happen?
                location = '???'
            if hasattr(mod, '__version__'):
                version = mod.__version__
            else:
                version = '???'
        except ImportError, AttributeError:
            version, location = 'not found', 'not found'
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return version, location


def test_pk(verbose=False):
    if verbose:
        print 'sys.path :'
        for entry in sys.path :
            print '       ', entry

    print
    print 'Checking installed versions...'
    print

    if sys.version_info < (2, 5, 0):
        raise RequirementError('Python version between 2.5 and 2.7 is '
                               'required to use the packages in stsci_python')

    try:
        import urwid
    except ImportError:
        print ('Package urwid was not found. It is not required but if '
               'available will enable text based epar in pyraf.\n')

    try:
        import IPython
    except ImportError:
        print ('Package ipython was not found. It is not required but if '
               'available can be used with pyraf (pyraf --ipython).\n')


    messages = []

    required_packages = get_requirements()[('stsci_python', '')]

    for reqname, req in sorted(required_packages.items()):
        version, location = get_dist_info(req.key)
        if version not in req:
            message = ('%-25s %-20s %-20s %s' %
                       (req.key, req.version, version, location))
            messages.append(message)

    if messages:
        print ('%-25s %-20s %-20s %s' %
               ('package', 'expected', 'found', 'location'))
        messages.sort()
        for m in messages:
            print m
        if not verbose:
            print ('If you will be sending email to help@stsci.edu, please '
                   're-run with `python testpk.py -v`')
    else:
        print 'All packages were successfully installed.\n'


# This seems largely redundant with the report_pk command, but we'll leave it
# in here just because
def report_pk(opus=False, verbose=False):
    colfmt = '%-25s %-15s %-15s %s'

    if opus:
        print colfmt % ('package', 'version', '', 'location')
    else :
        print colfmt % ('package', 'version', 'expected', 'location')
    print

    required_packages = get_requirements()[('stsci_python', '')]
    additional_packages = [parse_requirement(p) for p in ALSO_REPORT]

    # make a unique list of all the modules we want a report on
    all_packages = list(set(required_packages.values()) |
                        set(additional_packages))
    all_packages.sort(key=lambda r: r.key)
    # print the package info
    for pkg in all_packages:
        version, location = get_dist_info(pkg.key)
        if opus:
            print colfmt % (pkg.key, version, '', location)
        else:
            expected = '-'
            if pkg.key in required_packages:
                if version not in pkg:
                    expected = pkg.version
            print colfmt % (pkg.key, version, expected, location)


def fix_requirement(project, dist_name, version_predicate):
    """Open the setup.cfg for `project`, and update the required version of
    `dist_name` if it exists in the list of requirements for that project.
    """

    # Ensure that the SUBPROJECT_DIRS global is populated
    global SUBPROJECT_DIRS
    if not SUBPROJECT_DIRS:
        get_requirements()

    filename = os.path.join(SUBPROJECT_DIRS[project], 'setup.cfg')

    # We don't want to use ConfigParser to rewrite the values since it doesn't
    # round-trip properly (when is the stdlib going to get one of the many
    # ConfigParser extensions that do?!)  Instead we process the file manually
    # once we know which lines we want to fix

    output = []

    with open(filename, 'r') as f:
        section = ''
        fixed = False
        for line in f:
            if fixed:
                # We already made the fix so just output any remaining lines
                output.append(line)
                continue
            m = ConfigParser.SECTCRE.match(line)
            if m:
                section = m.group('header')
                output.append(line)
                continue
            if section != 'metadata':
                output.append(line)
                continue
            if not line[0].isspace() and '=' in line:
                opt, val = (x.strip() for x in line.split('=', 1))
                if opt != 'requires-dist':
                    output.append(line)
                    continue
                output.append('requires-dist =\n')
                if val:
                    # In case there was a value on the first line of the option
                    req = parse_requires_dist_item(val)
                    if req and req.key == dist_name:
                        output.append('    %s (%s)\n' % (dist_name,
                                                         version_predicate))
                    else:
                        output.append('    %s\n' % val)
                # Now go into an inner loop to deal with the requires-dist
                # option itself
                for line in f:
                    if line[0] in ('#', ';'):
                        # A comment
                        output.append(line)
                        continue
                    if not line[0].isspace():
                        # The end of the option
                        fixed = True
                        output.append(line)
                        break
                    line = line.strip()
                    req = parse_requires_dist_item(line)
                    if req and req.key == dist_name:
                        output.append('    %s (%s)\n' % (dist_name,
                                                         version_predicate))
                    elif line:
                        output.append('    %s\n' % line)
                    else:
                        output.append('\n')
            else:
                output.append(line)

    with open(filename, 'w') as f:
        f.write(''.join(output))


def list_requirements(fix=False):
    """Print a list of all the requirements in stsci_python plus their
    requirements.
    """

    all_requirements = get_requirements()
    stsci_reqs = all_requirements[('stsci_python', '')]
    for reqname, req in sorted(stsci_reqs.items()):
        for sub_name, sub_version in all_requirements:
            if sub_name != req.key:
                continue

            if sub_version in req:
                print '%s (%s)' % (sub_name, sub_version)
            elif fix:
                fix_requirement('stsci_python', sub_name, '==' + sub_version)
                print '%s (%s) [FIXED]' % (sub_name, sub_version)
            else:
                print ('%s (%s) [EXPECTED %s]' %
                       (sub_name, sub_version, req.version))

            subproj_requirements = all_requirements[(sub_name, sub_version)]
            for sub_req_name, sub_req in sorted(subproj_requirements.items()):
                if sub_req.key not in stsci_reqs:
                    print '        %s (%s)' % (sub_req.key, sub_req.version)
                else:
                    stsci_req = stsci_reqs[sub_req.key]
                    if stsci_req == sub_req:
                        print ('        %s (%s)' %
                               (sub_req.key, sub_req.version))
                    elif fix:
                        fix_requirement(sub_name, stsci_req.key,
                                        stsci_req.predicate)
                        print ('        %s (%s) [FIXED]' %
                               (sub_req.key, stsci_req.version))
                    else:
                        print ('        %s (%s) [EXPECTED %s]' %
                               (sub_req.key, sub_req.version,
                                stsci_req.version))
            break
        else:
            print '%s (%s)' % (req.key, req.version)
            print '        MISSING!'
        print


def main(args=None):
    parser = optparse.OptionParser(
            usage='usage: %prog [options]\n\nrunning with no arguments is the '
                  'same as running `%prog -t`')
    parser.add_option('-v', '--verbose', action='store_true',
                      help='enable verbose reporting')
    parser.add_option('-t', '--test', action='store_true', default=True,
                      help='test that install versions are as expected')
    parser.add_option('-r', '--report', action='store_true',
                      help='report versions of everything')

    group = optparse.OptionGroup(parser, 'Developer Options')
    group.add_option('-o', '--opus', action='store_true',
                     help='report versions in format for opus')
    group.add_option('-l', '--list', action='store_true',
                    help="print a nested list of stsci_python's requirements")
    group.add_option('-f', '--fix', action='store_true',
                     help='fix setup.cfgs to contain the correct version '
                          'number for each requirement')
    group.add_option('-d', '--debug', action='store_true',
                     help='print full tracebacks on exceptions')

    parser.add_option_group(group)

    options, args = parser.parse_args(args)

    try:
        if options.opus:
            report_pk(opus=True, verbose=options.verbose)
        elif options.report:
            report_pk(verbose=options.verbose)
        elif options.list or options.fix:
            list_requirements(fix=options.fix)
        elif options.test:
            test_pk(verbose=options.verbose)
    except Exception, e:
        if options.debug:
            raise
        else:
            sys.stderr.write('%s: %s\n' % (e.__class__.__name__, e))
            return 1

if __name__ == '__main__':
    sys.exit(main())
