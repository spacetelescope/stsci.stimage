"""module tkutils.py -- Tk utility functions

tkread          Read n bytes from file while running Tk mainloop
tkreadline     Read a line from file while running Tk mainloop

$Id$

W.J. Hack, 2002 Mar 08
"""

import os, sys, types, select
import Tkinter

def tkread(file, n=0):

    """Read n bytes from file (or socket) while running Tk mainloop.

    If n=0 then this runs the mainloop until some input is ready on
    the file.  (See tkreadline for an application of this.)  The
    file must have a fileno method.
    """

    return _TkRead().read(file, n)

def tkreadline(file=None):

    """Read a line from file while running Tk mainloop.

    If the file is not line-buffered then the Tk mainloop will stop
    running after one character is typed.  The function will still work
    but Tk widgets will stop updating.  This should work OK for stdin and
    other line-buffered filehandles.  If file is omitted, reads from
    sys.stdin.

    The file must have a readline method.  If it does not have a fileno
    method (which can happen e.g. for the status line input on the
    graphics window) then the readline method is simply called directly.
    """

    if file is None:
        file = sys.stdin
    if not hasattr(file, "readline"):
        raise TypeError("file must be a filehandle with a readline method")
    if hasattr(file, 'fileno'):
        fd = file.fileno()
        tkread(fd, 0)
        # if EOF was encountered on a tty, avoid reading again because
        # it actually requests more data
        if not select.select([fd],[],[],0)[0]:
            return ''
    return file.readline()

class _TkRead:

    """Run Tk mainloop while waiting for a pending read operation"""

    def read(self, file, nbytes):
        """Read nbytes characters from file while running Tk mainloop"""
        if isinstance(file, types.IntType):
            fd = file
        elif hasattr(file, "fileno"):
            fd = file.fileno()
        else:
            raise TypeError("file must be an integer or a filehandle/socket")
        self.widget = Tkinter._default_root
        if not self.widget:
            # no Tk widgets yet, so no need for mainloop
            s = []
            while nbytes>0:
                snew = os.read(fd, nbytes)
                if snew:
                    s.append(snew)
                    nbytes -= len(snew)
                else:
                    # EOF -- just return what we have so far
                    break
            return "".join(s)
        else:
            self.nbytes = nbytes
            self.value = []
            Tkinter.tkinter.createfilehandler(fd,
                                    Tkinter.tkinter.READABLE | Tkinter.tkinter.EXCEPTION,
                                    self._read)
            try:
                self.widget.mainloop()
            finally:
                Tkinter.tkinter.deletefilehandler(fd)
            return "".join(self.value)

    def _read(self, fd, mask):
        """Read waiting data and terminate Tk mainloop if done"""
        try:
            # if EOF was encountered on a tty, avoid reading again because
            # it actually requests more data
            if select.select([fd],[],[],0)[0]:
                snew = os.read(fd, self.nbytes)
                self.value.append(snew)
                self.nbytes -= len(snew)
            else:
                snew = ''
            if (self.nbytes <= 0 or len(snew) == 0) and self.widget:
                # stop the mainloop
                self.widget.quit()
        except OSError, error:
            raise IOError("Error reading from %s" % (fd,))
