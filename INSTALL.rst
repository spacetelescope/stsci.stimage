..
.. In restructured text, you declare a section heading by writing underlines
.. on the next line.  The underlines must be _at_ _least_ as long as the
.. section heading.
..
.. Which underlines you use for each heading are defined by the order they
.. appear in your document.  We are using:
.. H1 ===
.. H2 ---
.. H3 ~~~
..

.. role:: red
.. role:: green
.. role:: blue
.. role:: orange

================================================================================
STSCI_PYTHON Version 2.12
================================================================================

:abstract: 

    STSCI_PYTHON is a collection of Python packages (with C
    extensions) that has been developed to provide a general
    astronomical data analysis infrastructure. They can be used
    standalone from within Python or as Python tasks that are
    accessible from within STSDAS whens running under PyRAF.

    STSCI_PYTHON is developed by the Science Software Branch
    at the Space Telescope Science Institute.

.. contents::

Installing Binaries 
--------------------------------------------------------------------------------

Macintosh 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uninstalling an old installation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


If you have installed STSCI_PYTHON from a Macintosh package before, there
is no need to uninstall it.  When you install the new package, it
will remove the old software and replace it.


Installing
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The Macintosh binaries are distributed as a Macintosh package. This package includes all of

    - IRAF 2.14
    - STSDAS/TABLES
    - Python and supporting code
    - STSCI_PYTHON

in a single package. This is the same package that we use internally at the Institute.

You must be an administrator to install this package.

Download the appropriate DMG file from
http://www.stsci.edu/resources/software_hardware/pyraf/stsci_python/current/download .
Be sure to choose the correct package for OSX Leopard or OSX Snow
Leopard.  

:red:`This package contains binaries only for 64-bit capable Intel
processors.`  That includes a processor type of "Xeon" or "Core 2 Duo", but not
"Core Duo" or any variant of PowerPC.

Double-click on the .DMG file, then double-click on the .pkg file
in the window that appears. This runs the standard Macintosh
installer. Click "Continue" to move through the various screens of
the install and enter your password when prompted.

Most of the package is installed in /usr/stsci. IRAF assumes that
there is only one IRAF installation on any computer and it places
symlinks at various places about the system. There may be some
conflict if you have multiple copies of IRAF installed, but 2.12,
2.13, and 2.14 appear to be similar enough that it is not a great
problem unless you are compiling IRAF itself from source code.

Warning: This package and future releases will attempt to remove
any previously installed copies of itself. That means that changes
to the software in /usr/stsci will be lost if you run the installer
again. We strongly suggest that you do not make changes to the
software in /usr/stsci. Notably, add new IRAF tasks by editing your
login.cl file and install python packages with --home or --prefix.

User Configuration
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Because of certain IRAF dependencies, you must use csh as your
shell.  The default on OSX is bash.  To change it, type:

 ::

            chsh -s /bin/tcsh

Each user who wants to use the software from this package must execute these two commands:

 ::

        source /usr/stsci/envconfig.mac/cshrc
        iraf

We suggest that you place these commands at the end of your .cshrc
file. There are many environment variables involved, and you may
have problems if you override their values.

Note: This package is essentially the same thing that we use
internally at STScI. If you examine the script file, you will find
aliases for "irafx" and "irafdev". These are test configurations
that we use internally; they are not distributed with this package.


Windows 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Windows binaries are distributed as a Windows installer file.
It is available from
http://www.stsci.edu/resources/software_hardware/pyraf/stsci_python/current/download

Download the .exe file and double-click on it to install it.

This installer works with the native Windows Python.  You can download a Python
installer from http://python.org/download/

Running PyRAF 
--------------------------------------------------------------------------------

To run PyRAF, enter the command

    pyraf

A PyRAF tutorial is available at http://stsdas.stsci.edu/pyraf/doc.old/pyraf_tutorial/

Note: Pyraf is is not tested on MS Windows, but the latest version
of pyraf does provide partial support for Windows.  See
http://www.stsci.edu/resources/software_hardware/pyraf/current/download for
details.


Installing From Source 
--------------------------------------------------------------------------------

Required Supporting Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to install the supporting packages, we suggest that you follow the order in which they are listed below:


  =============  ================================================  ================================================================================    
  Package        Tested with                                       Web Site                                                                            
  IRAF           2.14                                              http://iraf.noao.edu/                                                               
  STSDAS/TABLES  3.14                                              http://www.stsci.edu/resources/software_hardware/stsdas/download                    
  Python         2.5.4, 2.7.1                                      http://python.org/download/                                                         
  Tcl/TK         8.5.7                                             http://www.tcl.tk/ http://www.tcl.tk/software/tcltk/download.html                   
  Pmw            1.3.2                                             http://pmw.sourceforge.net/                                                         
  urwid          0.9.9.1 (optional)                                http://excess.org/urwid/                                                            
  ipython        0.10.1 (optional)                                 http://ipython.scipy.org/ http://ipython.scipy.org/moin/Download                    
  NumPy          1.6.1                                             http://numpy.org/                                                                   
  GNU readline   6.1                                               http://tiswww.case.edu/php/chet/readline/rltop.html ftp://ftp.cwru.edu/pub/bash/    
  =============  ================================================  ================================================================================    


Some platforms have most of these packages already installed in their system directories. To test whether your Python installation has all modules needed, start Python and try to import them:

 ::

    % python
    >>> import readline
    >>> import Tkinter
    >>> Tkinter._test()
    >>> import Pmw


If you don't get an ImportError, this means that Tcl, Tk, Readline, Python and Pmw are already installed on the system.

The X11 windowing system and a C compiler are needed as well.

Follow the directions on http://iraf.noao.edu/ to get IRAF working on your system.

Some NumPy documentation is available at http://www.scipy.org/Documentation


Remove Any Old STSCI_PYTHON 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have an earlier version of STSCI_PYTHON installed, you should
either remove it or ensure that it is not present on your PYTHONPATH.


Python does not have a facility to uninstall packages, but you can
use the uninstall.py script to assist you.  It attempts to locate
things that look like they might be part of a previous stsci_python
installation.  For each thing that uninstall.py locates, it will
describe it and then offer to delete it for you.  If you want to
delete it, type "y" and press enter.  If you want to skip it, just
press enter.

 ::

    python uninstall.py


Personal Install 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a personal install, the files are installed in a directory that
you choose, instead of in the system directories. For example, if
you install in /home/user/stuff, the package files will be under
/home/user/stuff/lib/python and the pyraf program will be in
/home/user/stuff/bin.

You will need this set of commands to set up the environment.  Other
users can also run from your copy of the software by setting the
environment variables to point to the same directories.

Enter these commands now, and also place them in your .login or .cshrc file:

 ::

    set d=$HOME/stuff
    setenv PYTHONPATH $d/lib/python
    set path = ( $d/bin $path )


Extract the tar file:

 ::

    % gunzip stsci_python_2.12.tar.gz
    % tar -xvf stsci_python_2.12.tar
        use gtar (gnu tar) on Solaris
    % cd stsci_python_2.12

If you need numpy, install it:

 ::

    % cd numpy-1.6.1
    % unsetenv F77
    % unsetenv F2C
    % python setup.py install --home=$d
    % cd ..

Install the stsci_python package:

 ::

    % python setup.py install --home=$d


System-wide Install 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are root, you can install stsci_python in the system python
directories, where it will be available to all users. The commands
are almost the same as for a personal install, but you do not need
to specify the directory to install.

Extract the tar file:

 ::

    % gunzip stsci_python_2.12.tar.gz
    % tar -xvf stsci_python_2.12.tar
        use gtar (gnu tar) on Solaris
    % cd stsci_python_2.12

If you need numpy, install it:

 ::

    % cd numpy-1.6.1
    % unsetenv F77
    % unsetenv F2C
    % python setup.py install
    % cd ..

Install the stsci_python package:

 ::

    % python setup.py install


Testing the Installation 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The script testpk.py can be used to check the integrity and version
compatibility of the installation.

Make sure that PYTHONPATH is correct.

 ::

    % python testpk.py



Assistance 
--------------------------------------------------------------------------------

If you have any difficulties with the installation of any of the
packages in stsci_python, please do not hesitate to contact us for
assistance. Also, if you have questions or suggestions about
stsci_python in general or this document contact us at help@stsci.edu.
We hope that people can contribute tips to the platform specific
part of this document.


Appendix 1:  Other software versions
--------------------------------------------------------------------------------

Python 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have not converted our software for Python 3.  Python 3 is very
similar to Python 2, but it takes some care to convert existing
software, even with automated tools such as 2to3.  It is likely
that only expert python programmers would be able to perform the
conversion, so we do not recommend it for our users.


IRAF 2.15 (32 or 64 bit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have reports of Pyraf working with IRAF 2.15.1a (do not use earlier 
versions of IRAF 2.15), but we are still using IRAF 2.14 internally. 


Appendix 2:  Platform Specific Notes 
--------------------------------------------------------------------------------

Macintosh 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pyraf now works with native Macintosh graphics.  You no longer need a special
X-windows capable Python.

If you install from source, compilers can be installed with the Developer's tools.


Linux 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some linux distributions have separate developer packages for Python
and Numpy, If you are installing from source, you may need to install
those developer packages, especially if you get an error message
about a file name that ends with ".h"

The packaging systems for the various flavors of Linux can be
used to install the supporting packages. For example, on Redhat,
rpms can be used for supporting packages. However, Tkinter rpms
should be matched with the version of python and the operating
system. On a Linux system, the rpms for these packages are on the
installation CD. Both libraries and the header files are needed for
the installation.

Problems runnning graphics tasks in Pyraf were reported on some
operating systems (for example Ubuntu and Suse). The error message
is


 ::

    TclError: expected floating point number but got "1.0"

Although we do not understand the reasons for this we know it is
caused by a default non-english locale on the system. One possible
solution is to start pyraf by running:

 ::

    env LC_ALL=C pyraf



Appendix 3: Installing the Supporting Packages 
--------------------------------------------------------------------------------

Some of the packages require IRAF to be present on the system. IRAF
installation is not discussed in this document; it is maintained
by the IRAF group at NOAO. For installation instructions or problems,
see http://www.iraf.net.

Note: A full installation of the supporting packages is needed,
including libraries and header files. On some operating systems the
header files may be in a separate package. For example on Redhat
they are in the corresponding "devel" rpm package.

If installation from source is necessary, on most systems the
following will work:

To unpack a source file:

 ::

    % gunzip package.tar.gz
    % tar -xvf package.tar

To configure and build a package:

 ::

    % cd package
    % ./configure --prefix=/example
    % make
    % make install

This will create directories bin, lib, include under /example. The
option "--prefix=" in the above "./configure" command may be omitted
for installations in /usr/local.

If you install any of these packages in a personal directory, you
will most likely need to set your path:

 ::

    % set path ( /example/bin $path )

and change LD_LIBRARY_PATH with one of these commands:

 ::

    % setenv LD_LIBRARY_PATH /example/lib:$LD_LIBRARY_PATH

 ::

    % setenv PATH /example/bin:$PATH


Tcl/Tk 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is very likely that Tcl and Tk are already installed on your
system.  Type "wish" to start TK; if it creates a new window, TK is 
installed.  Nearly all Linux distributions provide Tcl/TK packages 
that you can install if it is not already on your machine. 

If you have to build these packages from source, build them as
shared libraries. On most systems the following set of commands
will work for Tcl and Tk:

 ::

    % cd tcl8.3.5/unix
    % ./configure --enable-shared --prefix=/installation-directory
    % make
    % make install

Readline 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Readline is probably already installed on your system. Make sure
the header files are installed as well. A possible location to look
for them is /usr/include/readline. In case you need to install
Readline in your personal directories, the following commands will
install it on most systems:

 ::

    % cd readline
    % ./configure --prefix=/installation-directory
    % make
    % make install

Python 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source Installation: Python is available from the python web site
at http://www.python.org/ . If Tcl/Tk and Readline libraries are
on LD_LIBRARY_PATH or in a system directory, the next three commands
are usually sufficient to install Python from source:

    % cd python
    % ./configure --prefix=/installation-directory
    % make
    % make install

To test whether your Python installation has all required modules
enabled, try to import the modules as described in Section 1.0.

To build Tkinter as part of Python, you may need to edit the file
Modules/Setup in the Python source distribution, to let Python know
where Tcl/Tk and X11 libraries are. Below is an example of this
section of the Setup file on Solaris. Note, that some lines are
uncommented and the paths on your system may be different.

 ::
  
  # The _tkinter module.
  #
  # The command for _tkinter is long and site specific. Please
  # uncomment and/or edit those parts as indicated. If you don't have a
  # specific extension (e.g. Tix or BLT), leave the corresponding line
  # commented out. (Leave the trailing backslashes in! If you
  # experience strange errors, you may want to join all uncommented
  # lines and remove the backslashes -- the backslash interpretation is
  # done by the shell's "read" command and it may not be implemented on
  # every system.
  # *** Always uncomment this (leave the leading underscore in!):
  _tkinter _tkinter.c tkappinit.c -DWITH_APPINIT \
  # *** Uncomment and edit to reflect where your Tcl/Tk libraries are:
  -L/usr/local/lib \
  # *** Uncomment and edit to reflect where your Tcl/Tk headers are:
  -I/usr/local/include \
  # *** Uncomment and edit to reflect where your X11 header files are:
  # -I/usr/X11R6/include \
  # *** Or uncomment this for Solaris:
  -I/usr/openwin/include \
  # *** Uncomment and edit for Tix extension only:
  # -DWITH_TIX -ltix8.1.8.2 \
  # *** Uncomment and edit for BLT extension only:
  # -DWITH_BLT -I/usr/local/blt/blt8.0-unoff/include -lBLT8.0 \
  # *** Uncomment and edit for PIL (TkImaging) extension only:
  # (See http://www.pythonware.com/products/pil/ for more info)
  # -DWITH_PIL -I../Extensions/Imaging/libImaging tkImaging.c \
  # *** Uncomment and edit for TOGL extension only:
  # -DWITH_TOGL togl.c \
  # *** Uncomment and edit to reflect your Tcl/Tk versions:
  -ltk8.3 -ltcl8.3 \
  # *** Uncomment and edit to reflect where your X11 libraries are:
  # -L/usr/X11R6/lib \
  # *** Or uncomment this for Solaris:
  -L/usr/openwin/lib \
  # *** Uncomment these for TOGL extension only:
  # -lGL -lGLU -lXext -lXmu \
  # *** Uncomment for AIX \
  # -lld \
  # *** Always uncomment this; X11 libraries to link with:
  -lX11
  

libf2c 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of stsci_python 2.7, it is no longer necessary to provide libf2c.

Urwid 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Urwid ( http://excess.org/urwid/ ) can be installed optionally. It
is needed for support of tpar (a text based epar) in PyRAF. It can
be installed by :

% python setup.py install


Ipython 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ipython ( http://ipython.scipy.org/moin/ ) can be installed optionally
as well. If available PyRAF can run in the Ipython interpreter
(pyraf --ipython). To install Ipython, execute the command:

% python setup.py install


