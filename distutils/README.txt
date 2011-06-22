Introduction
==============
This package contains utilities used to package some of STScI's Python
projects; specifically those projects that comprise stsci_python_ and
Astrolib_.

It currently consists mostly of some setup_hook scripts meant for use with
`distutils2/packaging`_ and/or d2to1_, and a customized easy_install command
meant for use with distribute_.

This package is not meant for general consumption, though it might be worth
looking at for examples of how to do certain things with your own packages, but
YMMV.

Features
==========

Hook Scripts
--------------
Currently the main features of this package are a couple of setup_hook scripts.
In distutils2, a setup_hook is a script that runs at the beginning of any
pysetup command, and can modify the package configuration read from setup.cfg.

stsci.distutils.hooks.use_packages_root
'''''''''''''''''''''''''''''''''''''''''
If using the `packages_root` option under the `[files]` section of setup.cfg,
this hook will add that path to `sys.path` so that modules in your package can
be imported and used in setup.  This can be used even if `packages_root` is not
specified--in this case it adds `''` to `sys.path`.

stsci.distutils.hooks.glob_data_files
'''''''''''''''''''''''''''''''''''''''
Allows filename wildcards as understood by `glob.glob()` to be used in the
data_files option.  This hook must be used in order to have this functionality
since it does not normally exist in distutils.

stsci.distutils.hooks.numpy_extension_hook
''''''''''''''''''''''''''''''''''''''''''''
This is a pre-command hook for the build_ext command.  To use it, add a
[build_ext] section to your setup.cfg, and add to it:

    pre-hook.numpy-extension-hook = stsci.distutils.hooks.numpy_extension_hook

This hook must be used to build extension modules that use Numpy.   The primary
side-effect of this hook is to add the correct numpy include directories to
`include_dirs`.  To use it, add 'numpy' to the 'include-dirs' option of each
extension module that requires numpy to build.  The value 'numpy' will be
replaced with the actual path to the numpy includes.

stsci.distutils.hooks.is_display_option
'''''''''''''''''''''''''''''''''''''''''
This is not actually a hook, but is a useful utility function that can be used
in writing other hooks.  Basically, it returns `True` if setup.py was run with
a "display option" such as --version or --help.  This can be used to prevent
your hook from running in such cases.


Commands
----------
Currently one custom command is included:
`stsci.distutils.command.easier_install`.  This is meant as a replacement for
the distribute/setuptools easy_install command.  It works exactly the same way,
but includes a new feature: Local source directories can be searched for
package dependencies.

The directories to search can be specified by adding them to the `find-links`
option in the `[easy_intall]` section of setup.cfg.  Though `find-links` can
already be used to point to egg files or source tarfiles on the local
filesystem, this adds the ability to point to existing source checkouts to
search for dependencies.

Currently this only supports source distributions that have their package
metadata in setup.cfg (distutils2 style), but this could be extended to support
other more common distribution styles.  For example, checking for EGG-INFO, or
even calling `setup.py egg_info` and using that to determine whether or not a
source checkout matches some requirement.


.. _stsci_python: http://www.stsci.edu/resources/software_hardware/pyraf/stsci_python
.. _Astrolib: http://www.scipy.org/AstroLib/
.. _distutils2/packaging: http://distutils2.notmyidea.org/
.. _d2to1: http://pypi.python.org/pypi/d2to1
.. _distribute: http://pypi.python.org/pypi/distribute
