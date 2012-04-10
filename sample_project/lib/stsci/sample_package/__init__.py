# from __future__ import absolute_import

from __future__ import division # confidence high

__version__ = ''
__svn_version__ = 'Unable to determine SVN revision'
__full_svn_info__ = ''
__setup_datetime__ = None

try:
    __version__ = __import__('pkg_resources').\
                        get_distribution('sample_project').version
except:
    pass

try:
    from stsci.sample_package.svninfo import (__svn_version__,
                                              __full_svn_info__,
                                              __setup_datetime__)
except ImportError:
    pass

# If you import stsci.tools.tests _inside_ the test() function, then we don't
# need to worry about whether it exists or not.  We also don't increase
# the load time of this package.

def test(*args,**kwds):
    import stsci.tools.tester
    stsci.tools.tester.test(modname=__name__, mode='pytest', *args, **kwds)

