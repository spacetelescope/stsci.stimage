# from __future__ import absolute_import
from __future__ import division # confidence high


if False :
    __version__ = ''
    __svn_version__ = 'Unable to determine SVN revision'
    __full_svn_info__ = ''
    __setup_datetime__ = None

    try:
        __version__ = __import__('pkg_resources').\
                        get_distribution('reftools').version
    except:
        pass
else :
    __version__ = '1.5.4'

try:
    from reftools.svninfo import (__svn_version__, __full_svn_info__,
                                  __setup_datetime__)
except ImportError:
    pass
