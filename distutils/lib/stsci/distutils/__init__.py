try:
    __version__ = \
        __import__('pkg_resources').get_distribution('stsci.distutils').version
except:
    __version__ = ''
