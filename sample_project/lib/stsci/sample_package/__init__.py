# from __future__ import absolute_import

from __future__ import division # confidence high

__version__ = "1.0"

try:
    # from .svn_version import __svn_version__, __full_svn_info__
    from svn_version import __svn_version__, __full_svn_info__
except:
    __svn_version__ = 'Unable to determine SVN revision'
    __full_svn_info__ = __svn_version__

# If you import stsci.tools.tests _inside_ the test() function, then we don't
# need to worry about whether it exists or not.  We also don't increase
# the load time of this package.

def test(*args,**kwds):
    import stsci.tools.tester
    stsci.tools.tester.test(modname=__name__, *args, **kwds)

