#! /usr/bin/env python
"""timefilter - filter a corrtag table based on the TIMELINE extension"""

from __future__ import division         # confidence high

__author__ = "Phil Hodge, STScI, April 2011."
__usage__ = """
1. To run this task from within Python::

    >>> from hstcos import timefilter
    >>> timefilter.Timelinefilter ("xyz_corrtag.fits", "temp_corrtag.fits",
                                   "sun_alt > 0.")

.. note:: make sure the `hstcos` package is on your Python path

2. To run this task using the TEAL GUI to set the parameters under PyRAF::

    >>> import hstcos
    >>> epar timefilter   # or "teal timefilter"

3. To run this task from the operating system command line::

    # just print info:
    % timefilter.py xyz_corrtag.fits

    # flag events with sun_alt > 0 as bad, writing output to a new file
    # temp_corrtag.fits:
    % timefilter.py xyz_corrtag.fits temp_corrtag.fits 'sun_alt > 0'

    # clear the bad time interval flag (2048) from the DQ column
    % timefilter.py temp_corrtag.fits '' reset
    % timefilter.py temp_corrtag.fits '' clear

.. note:: make sure the file "timefilter.py" is on your executable path
"""

import copy
import getopt
import math
import os
import sys
import numpy as np
import pyfits
from stsci.tools import parseinput, teal
from calcos import ccos
from calcos import calcosparam, cosutil
import saamodel

__taskname__ = "timefilter"
__version__ = "0.3"
__vdate__ = "2011 Apr 27"

DEGtoRAD = math.pi / 180.
TWOPI = 2. * math.pi

# This cutoff is based on current models in saamodel.py; this is used when
# finding the middle of the SAA region (middle_SAA).
SAA_LONGITUDE_CUTOFF = 200.

# conjunctions for the filter
AND = "and"
OR = "or"
XOR = "xor"

# Note:  longitude presents a problem because of the discontinuity at 360,
# so a filter such as "longitude > l1 and longitude < l2" would not work
# for a region extending from longitude 300 to 20 degrees, for example.

def main():
    """Filter a corrtag file using its timeline extension."""

    try:
        (options, pargs) = getopt.getopt (sys.argv[1:], "hv", ["help"])
    except Exception, error:
        print str (error)
        prtOptions()
        return

    help = False
    verbose = False
    for i in range (len (options)):
        if options[i][0] == "-h":
            help = True
        elif options[i][0] == "--help":
            help = True
        elif options[i][0] == "-v":
            verbose = True

    if help:
        print getHelpAsString()
        # print __doc__
        # print __usage__
        # print "\t", __version__ + " (" + __vdate__ + ")"
        return

    # timefilter.py input                       # to just print info
    # timefilter.py input output filter
    nargs = len (pargs)
    if nargs < 1 or nargs > 3:
        prtOptions()
        return

    input = pargs[0]
    output = None
    filter = None
    if nargs < 2 or pargs[1] == "" or pargs[1] == " " or \
       pargs[1].lower() == "none":
        output = None
    else:
        output = pargs[1]

    if nargs == 3:
        filter = pargs[2]

    tlf = TimelineFilter (input, output, filter, verbose=verbose)

def prtOptions():
    """Print a list of command-line options and arguments."""

    print "The command-line arguments are:"
    print "  -h (print help)"
    print "  --help (print help)"
    print "  -v (print messages)"
    print "  input:  input corrtag file name"
    print "  output:  output corrtag file name, or '' or none"
    print "      if output was not specified, the input file will be modified"
    print "      in-place (unless filter was also not specified, equivalent to"
    print "      filter='info')"
    print "  filter:  column name, relation, cutoff value"
    print "      e.g. 'sun_alt > -0.5 or ly_alpha > 2'"
    print "      or 'info' or 'reset' ('clear' is synonymous with 'reset')"

def expandFilename (filename):
    """Expand environment variables to get the real file name.

    Parameters
    ----------
    filename: str
        A file name.

    Returns
    -------
    str
        The real file name.
    """

    if not filename or filename.strip() == "":
        return None

    fname = filename
    done = False
    count = 0
    MAX_COUNT = 100
    while not done:
        temp = os.path.expandvars (fname)       # $stuff/file
        count += 1
        if temp == fname:
            done = True
        fname = temp
        if count >= MAX_COUNT:
            break
    if not done:
        raise RuntimeError, "%d iterations exceeded while expanding " \
                "variables in file name %s" % (MAX_COUNT, filename)
    fname = os.path.abspath (fname)             # ../file
    fname = os.path.expanduser (fname)          # ~/file
    real_file_name = os.path.normpath (fname)   # remove redundant strings

    return real_file_name

def findMedian (x):
    """Compute the median of x.

    Parameters
    ----------
    x: array_like
        An array that can be sorted.

    Returns
    -------
    median_x: float
        If there are an odd number of elements in x, median_x is the
        middle element; otherwise, median_x is the average of the two
        middle elements.
    """

    nelem = len (x)
    middle = nelem // 2
    if middle * 2 < nelem:
        odd = True
    else:
        odd = False
    index = x.argsort()
    if odd:
        median_x = x[index[middle]]
    else:
        median_x = (x[index[middle]-1] + x[index[middle]]) / 2.

    return median_x

def toRect (longitude, latitude):
    """Convert longitude and latitude to rectangular coordinates.

    Parameters
    ----------
    longitude: float
        longitude in degrees.

    latitude: float
        latitude in degrees.

    Returns
    --------
    array_like
        Unit vector in rectangular coordinates.
    """

    rect = np.array ([1.0, 0.0, 0.0], dtype=np.float64)
    longitude *= DEGtoRAD
    latitude *= DEGtoRAD

    rect[0] = math.cos (latitude) * math.cos (longitude)
    rect[1] = math.cos (latitude) * math.sin (longitude)
    rect[2] = math.sin (latitude)

    return rect

def testWithinSAA (hst, vertices, middle_SAA):
    """Test whether HST is within the polygon for an SAA contour.

    Parameters
    ----------
    hst: array_like
        Unit vector pointing from the center of the Earth toward the
        location of HST at a particular time.

    vertices: array_like, shape (nvertices,3)
        vertices[i] is a unit vector from the center of the Earth toward
        vertex number i of a polygon that defines one of the SAA contour.

    middle_SAA: array_like
        Unit vector from the center of the Earth toward a point near the
        middle of the SAA region.  This is for making a quick check that
        `hst` is close enough to the SAA contour to be worth making a
        detailed check.

    Returns
    -------
    boolean
        True if hst is within the SAA contour defined by vertices.
    """

    # This test is primarily to exclude points that are diametrically
    # opposite to the SAA contour (because this would not be caught by
    # the code below!), but it should also save unnecessary arithmetic
    # most of the time.
    if np.dot (hst, middle_SAA) < 0.:
        return False

    nvertices = len (vertices)

    sin_lat_hst = hst[2]
    cos_lat_hst = math.sqrt (1. - sin_lat_hst**2)
    cos_long_hst = hst[0] / cos_lat_hst
    sin_long_hst = hst[1] / cos_lat_hst

    # vertices rotated to put hst in the x-z plane
    v_rot = vertices.copy()
    v_rot[:,0] =  vertices[:,0] * cos_long_hst + vertices[:,1] * sin_long_hst
    v_rot[:,1] = -vertices[:,0] * sin_long_hst + vertices[:,1] * cos_long_hst

    # v_rot rotated to put hst on the x axis
    v_rotrot = v_rot.copy()
    v_rotrot[:,0] =  v_rot[:,0] * cos_lat_hst + v_rot[:,2] * sin_lat_hst
    v_rotrot[:,2] = -v_rot[:,0] * sin_lat_hst + v_rot[:,2] * cos_lat_hst

    azimuth = np.arctan2 (v_rotrot[:,2], v_rotrot[:,1])
    azimuth = np.where (azimuth < 0., azimuth + TWOPI, azimuth)

    delta_az = azimuth[1:] - azimuth[0:-1]
    delta_az = np.where (delta_az < -np.pi, delta_az + TWOPI, delta_az)
    delta_az = np.where (delta_az > np.pi, delta_az - TWOPI, delta_az)

    sum_delta_az = delta_az.sum()

    return not (sum_delta_az < 0.1 and sum_delta_az > -0.1)

class TimelineFilter (object):
    """Filter a TIME-TAG table by setting a flag in the DQ column.

    There are no user-callable methods.  Instantiating the class does
    all the work.

    Attributes
    ----------
    input: str
        Name of input corrtag file.

    output: str
        Name of output file (may be None).

    filter_str: str
        The filter, either as specified by the user, or possibly with
        further modification.

    filter: str
        Info about column name, relation, and cutoff, or "info" or "clear".

    verbose: bool
        True if messages should be printed.

    fd: file
        File handle for input file.

    events_list: list of two integers
        [extver, hdunum] for each EVENTS extension.

    gti_list: list of two integers
        [extver, hdunum] for each GTI extension.

    tl_list: list of two integers
        [extver, hdunum] for each TIMELINE extension.

    events_hdunum: int
        HDU number for EVENTS extension.

    gti_hdunum: int
        HDU number for last GTI extension.

    tl_hdunum: int
        HDU number for TIMELINE extension.

    events_time: array_like
        Array of times from the TIME column in the EVENTS table, but
        copied to a local array in native format.  This will initially
        be set to None, then assigned later if we need the times.

    dq: array_like
        The DQ column from the EVENTS table.

    first_gti_hdunum: int
        HDU number for the first GTI extension.

    first_gti: list of two-element lists
        The contents of the first GTI table.  Each element of `first_gti`
        is a list giving the start and stop times (in seconds since
        EXPSTART) for a "good time interval," before any change to the
        GTI table resulting from filtering by this module.
    """

    def __init__ (self, input, output=None, filter=None, verbose=False):
        """Set DQ flag to mark bad time intervals.

        Parameters
        ----------
        input: str
            Name of input corrtag file.

        output: str or None
            Optional name of output file.  If an output file was specified,
            the input file will be copied to output and possibly modified.

        filter: str or None
            This string specifies which time intervals should be flagged
            as bad, based on columns in the TIMELINE extension.
            filter = "info" means that information about the input file
            should be printed.
            filter = "clear" or "reset" means that flag 2048 (bad time
            interval) should be cleared from the DQ column.

        verbose: bool
            If True, information will be printed.
        """

        self.verbose = verbose

        self.input = expandFilename (input)
        if self.verbose:
            print "Input file", self.input
        self.output = output
        if self.output is not None:
            self.output = expandFilename (output)

        self.filter_str = filter        # may be replaced later (currently not)
        self.interpretFilter (filter)

        if self.filter == ["info"]:
            iomode = "readonly"
        elif self.output is None:
            iomode = "update"
        else:
            iomode = "readonly"

        # If output was specified, check that it doesn't already exist.
        if self.output and os.access (self.output, os.F_OK):
            raise RuntimeError, "output file %s already exists" % self.output

        self.fd = pyfits.open (self.input, mode=iomode)
        if iomode == "update" and self.verbose:
            print "Input file opened read/write"
        self.findExtensions()
        self.findHduNum()

        # events_time will be gotten later if we need it.
        self.events_time = None
        try:
            self.dq = self.fd[self.events_hdunum].data.field ("dq")
        except (KeyError, AttributeError):
            self.fd.close()
            raise RuntimeError, \
                "The input %s does not appear to be a corrtag file." % input

        # Assign self.first_gti from the first GTI table in the input file.
        self.getFirstGTI()

        if self.filter == ["info"]:
            self.printInfo()
        elif self.filter == ["clear"]:
            self.clearDqFlag()
        else:
            self.setDqFlag()

        if self.output is not None:
            self.writeNewOutputFile()

        self.fd.close()

    def interpretFilter (self, filter):
        """Split `filter` into its parts.

        Parameters
        ----------
        filter: str
            Specification of how to filter, e.g. column name in TIMELINE
            table, cutoff value, and whether values to be flagged as bad
            are greater than the cutoff, less than, etc.  `filter` may
            alternatively be "info" or "clear" or "reset".
        """

        if filter is None:
            self.filter = ["info"]
            return

        filter_lower = filter.lower()
        information = "information"
        len_filter = max (4, len (filter))
        len_filter = min (len_filter, len (information))
        if filter_lower[:len_filter] == information[:len_filter]:
            self.filter = ["info"]
            return

        if filter_lower == "clear" or filter_lower == "reset":
            self.filter = ["clear"]
            return

        words = filter.split()
        nwords = len (words)
        if nwords == 0:
            self.filter = []
            return

        error_msg = "don't understand filter '%s'" % filter

        i = 0
        done = False
        self.filter = []
        while not done:
            if i >= nwords:
                done = True
                break

            colname = words[i]
            colname_l = colname.lower()
            conj = ""
            if colname_l == "and":
                conj = AND
                self.filter.append (conj)
                delta_i = 1
            elif colname_l == "or":
                conj = OR
                self.filter.append (conj)
                delta_i = 1
            elif colname_l == "xor":
                conj = XOR
                self.filter.append (conj)
                delta_i = 1
            elif colname_l == "saa":
                relation = None
                if i+1 >= nwords:
                    raise RuntimeError, error_msg
                cutoff = int (words[i+1])       # SAA model number
                delta_i = 2
            else:
                if i+2 >= nwords:
                    raise RuntimeError, error_msg
                relation = words[i+1]           # ">", "<", etc.
                cutoff = float (words[i+2])
                delta_i = 3

            if not conj:
                if relation == ">":
                    relation_fcn = np.greater
                elif relation == ">=":
                    relation_fcn = np.greater_equal
                elif relation == "<":
                    relation_fcn = np.less
                elif relation == "<=":
                    relation_fcn = np.less_equal
                elif relation == "==" or relation == "=":
                    relation_fcn = np.equal
                elif relation == "!=":
                    relation_fcn = np.not_equal
                elif colname_l == "saa":        # "column name" given as "saa"
                    relation_fcn = self.saaFilter
                else:
                    raise RuntimeError, error_msg
                self.filter.append ((colname, relation_fcn, cutoff))

            i += delta_i

    def printInfo (self):
        """Print information about the input file.

        The information printed includes:
        The names of the input and output files.
        The good-time intervals table (the one with largest EXTVER).
        The following values at the beginning, middle, and end of the
        range of times in the TIMELINE TIME column:
            sun altitude, target altitude, longitude, latitude, shift1.
        The minimum, maximum, median, of shift1, ly_alpha, darkrate.
        """

        print "input =", self.input
        if self.output is not None:
            print "output =", self.output
        if self.gti_hdunum is None:
            print "no GTI extension"
        else:
            gti = self.fd[self.gti_hdunum].data
            if gti is None:
                print "GTI:  no good time intervals"
            else:
                print "GTI:  start     stop"
                for i in range (len (gti)):
                    print "%11.3f %8.3f" % (gti[i][0], gti[i][1])

        events_hdu = self.fd[self.events_hdunum]
        dq = events_hdu.data.field ("dq")
        dq_2048 = np.where (np.bitwise_and (dq, calcosparam.DQ_BAD_TIME) > 0,
                            1., 0.)
        n_bad = dq_2048.sum()
        n_total = float (len (dq))
        print "%.1f %% of DQ column flagged with %d" % \
               (100. * n_bad / n_total, calcosparam.DQ_BAD_TIME)
        del (dq, dq_2048)
        print ""

        if self.tl_hdunum is None:
            print "no TIMELINE extension"
            return

        tl_hdunum = self.tl_hdunum
        # TIME column in the TIMELINE extension
        time_col = self.fd[tl_hdunum].data.field ("time")
        nelem = len (time_col)
        print "%d rows in TIMELINE" % nelem
        if nelem <= 0:
            return

        sun_alt_col    = self.fd[tl_hdunum].data.field ("sun_alt")
        target_alt_col = self.fd[tl_hdunum].data.field ("target_alt")
        longitude_col  = self.fd[tl_hdunum].data.field ("longitude")
        latitude_col   = self.fd[tl_hdunum].data.field ("latitude")
        shift1_col     = self.fd[tl_hdunum].data.field ("shift1")
        ly_alpha_col   = self.fd[tl_hdunum].data.field ("ly_alpha")
        darkrate_col   = self.fd[tl_hdunum].data.field ("darkrate")

        middle = nelem // 2
        median_shift1 = findMedian (shift1_col)
        median_ly_alpha = findMedian (ly_alpha_col)
        median_darkrate = findMedian (darkrate_col)

        print "column   beginning    middle       end"
        print "time      %8.2f  %8.2f  %8.2f" % \
                (time_col[0], time_col[middle], time_col.max())
        print "sun_alt    %7.2f   %7.2f   %7.2f" % \
                (sun_alt_col[0], sun_alt_col[middle], sun_alt_col[-1])
        print "target_alt %7.2f   %7.2f   %7.2f" % \
                (target_alt_col[0], target_alt_col[middle], target_alt_col[-1])
        print "longitude  %7.2f   %7.2f   %7.2f" % \
                (longitude_col[0], longitude_col[middle], longitude_col[-1])
        print "latitude   %7.2f   %7.2f   %7.2f" % \
                (latitude_col[0], latitude_col[middle], latitude_col[-1])
        print "shift1     %7.2f   %7.2f   %7.2f" % \
                (shift1_col[0], shift1_col[middle], shift1_col[-1])

        print ""
        print "column           min          max       median"
        print "shift1       %7.2f      %7.2f      %7.2f" % \
                (shift1_col.min(), shift1_col.max(), median_shift1)
        print "Ly_alpha %11.5g  %11.5g  %11.5g" % \
                (ly_alpha_col.min(), ly_alpha_col.max(), median_ly_alpha)
        print "darkrate %11.5g  %11.5g  %11.5g" % \
                (darkrate_col.min(), darkrate_col.max(), median_darkrate)

    def clearDqFlag (self):
        """Clear (reset) the bad-time-interval flag in the DQ column.

        The bit corresponding to the bad-time-interval flag value 2048
        will be set to 0 for every row of the DQ column in the EVENTS
        table.

        If there is more than one GTI table, the next to last one will
        be copied to overwrite the last one (based on keyword EXTVER).
        This is not foolproof; the last one may record intervals rejected
        due to FUV bursts, and this information could be lost.  A safer
        way to clear the bad-time-interval flag would be to go back to a
        previous version of the file.
        """

        if self.events_hdunum is None:
            raise RuntimeError, "No EVENTS extension in file %s" % self.input

        events_hdu = self.fd[self.events_hdunum]
        dq = events_hdu.data.field ("dq")
        not_badt = -1 ^ calcosparam.DQ_BAD_TIME
        dq[:] = np.bitwise_and (dq, not_badt)
        history = "Flag %d cleared in DQ column." % calcosparam.DQ_BAD_TIME
        self.fd[0].header.add_history (history)
        if self.verbose:
            print "Flag %d cleared" % calcosparam.DQ_BAD_TIME

        ngti = len (self.gti_list)
        if ngti > 1:
            last_gti_info = self.gti_list[-1]
            prev_gti_info = self.gti_list[-2]
            last_gti_extver = last_gti_info[0]          # [extver, hdunum]
            last_gti_hdunum = last_gti_info[1]
            prev_gti_hdunum = prev_gti_info[1]
            # overwrite last GTI extension with previous one
            self.fd[last_gti_hdunum] = self.fd[prev_gti_hdunum].copy()
            self.fd[last_gti_hdunum].header.update ("extver", last_gti_extver)
            if self.verbose:
                print "GTI extension %d overwritten by GTI extension %d" % \
                        (last_gti_hdunum, prev_gti_hdunum)

        # Recompute the exposure time.  While DQ_BAD_TIME has been cleared,
        # DQ_BURST might still be set.
        new_gti = self.recomputeExptime()

    def setDqFlag (self):
        """Set the bad-time-interval flag in the DQ column.

        The bit corresponding to the bad-time-interval flag value (2048)
        will be set to 1 in the DQ column in the EVENTS table for each
        event that is within a time interval that is "bad," according to
        the filter specified by the user.

        The time resolution (0.032 s) in the EVENTS table is finer than
        that of the TIMELINE table (1 s).  The bad time intervals are
        determined by using the TIMELINE table; the flagging can therefore
        be off by some fraction of a second.

        The GTI table will be updated.  If the input file has two or more
        GTI table extensions, the last one (highest EXTVER) will be
        overwritten with the new good time intervals; otherwise, a new
        GTI extension will be appended to the file.
        """

        if self.events_hdunum is None:
            raise RuntimeError, "No EVENTS extension in file %s" % self.input

        if self.tl_hdunum is None:
            raise RuntimeError, "No TIMELINE extension in file %s" % self.input

        # this is the complete header/data unit
        events_hdu = self.fd[self.events_hdunum]

        # these are columns in the EVENTS extension
        if self.events_time is None:
            self.events_time = cosutil.getColCopy (data=events_hdu.data,
                                                   column="time")
        nevents = len (self.events_time)

        # This is the column in the TIMELINE extension.
        time_col = cosutil.getColCopy (data=self.fd[self.tl_hdunum].data,
                                       column="time")

        # the first loop is for a single condition or for combining conditions
        # with "and"
        flags = []
        flag_a = None
        conj = None
        saved = False
        for item in self.filter:
            if isinstance (item, tuple):
                # this corresponds to a condition, such as "sun_alt < 0"
                (colname, relation_fcn, cutoff) = item

                if conj is None or conj == AND:
                    # filter_col is in the TIMELINE extension.
                    if colname.lower() == "saa":
                        filter_col = ""     # just to have a definite value
                    else:
                        filter_col = \
                        self.fd[self.tl_hdunum].data.field (colname)
                    nelem = len (time_col)
                    if flag_a is None:
                        flag_a = relation_fcn (filter_col, cutoff)
                    else:
                        # combine conditions with "and"
                        flag_b = relation_fcn (filter_col, cutoff)
                        flag_a = np.logical_and (flag_a, flag_b)
                    saved = False
            else:
                # the current item is a conjunction:  "and", "or", "xor"
                if item == AND:
                    conj = item
                elif item == OR or item == XOR:
                    flags.append (flag_a.copy())
                    flags.append (item)
                    flag_a = None
                    saved = True
                    conj = None
                else:
                    raise RuntimeError, \
                        "don't understand filter %s" % self.filter

        if not saved:
            flags.append (flag_a.copy())

        # this loop combines conditions with "or" or "xor"
        conj = None
        for item in flags:
            if isinstance (item, np.ndarray):
                if conj is None:
                    flag = item.copy()
                elif conj == OR:
                    flag = np.logical_or (flag, item)
                elif conj == XOR:
                    flag = np.logical_xor (flag, item)
                else:
                    raise RuntimeError, "don't recognize conjunction %s" % conj
                conj = None
            else:
                conj = item

        # `flag` is an array of boolean flags, one element for each row of
        # the TIMELINE extension.  Now we need to find and flag the rows
        # in the EVENTS extension corresponding to ranges in `flag` that
        # are True (bad).
        in_bad_interval = False             # initial values
        bad_time_intervals = []
        # indices in TIMELINE extension for a bad interval are ki:kj
        for k in range (nelem):             # index in timeline column
            if flag[k]:
                if not in_bad_interval:
                    # start a bad time interval
                    in_bad_interval = True
                    ki = k
            elif in_bad_interval:
                # end a bad time interval, and set flags in DQ
                in_bad_interval = False
                kj = k
                t0 = time_col[ki]
                t1 = time_col[kj]
                (i, j) = ccos.range (self.events_time, t0, t1)
                self.dq[i:j] |= calcosparam.DQ_BAD_TIME
                bad_time_intervals.append ((t0, t1))
        if in_bad_interval:
            # a bad time interval extends to the end of the exposure
            t0 = time_col[ki]
            t1 = self.events_time[nevents-1]
            (i, j) = ccos.range (self.events_time, t0, t1)       # ignore j
            self.dq[i:nevents] |= calcosparam.DQ_BAD_TIME
            bad_time_intervals.append ((t0, t1))

        history = "%s flagged as bad." % self.filter_str
        self.fd[0].header.add_history (history)
        if self.verbose:
            print "%s flagged as bad" % self.filter_str

        # Compute the exposure time, update the EXPTIME keyword, and get
        # an updated list of good time intervals, then write the updated
        # GTI to the file.
        gti = self.recomputeExptime()
        self.saveNewGTI (gti)

    def saaFilter (self, filter_col, model):
        """Flag within the specified SAA contour as bad.

        Parameters
        ----------
        filter_col: arbitrary
            This argument is not used.  It's included so that the calling
            sequence will be the same as functions np.greater, etc.

        model: int
            The SAA model number.  Currently these range from 2 to 32
            inclusive.  (Models 0 and 1 are radio frequence interference
            contours.)

        Returns
        -------
        flag: array_like
            This is a boolean array, one element for each row of the
            TIMELINE table.  True means that HST was within the SAA
            contour (specified by `model`) at the time corresponding to
            the TIMELINE row.
        """

        longitude_col = self.fd[self.tl_hdunum].data.field ("longitude")
        latitude_col  = self.fd[self.tl_hdunum].data.field ("latitude")

        nelem = len (longitude_col)

        flag = np.zeros (nelem, dtype=np.bool8)

        model_vertices = saamodel.saaModel (model)
        model_vertices.append (model_vertices[0])       # make a closed loop
        nvertices = len (model_vertices)
        # will be unit vectors from center of Earth pointing toward vertices
        vertices = np.zeros ((nvertices, 3), dtype=np.float64)
        minmax_long = [720., -360.]     # will be minimum, maximum longitudes
        minmax_lat = [90., -90.]        # will be minimum, maximum latitudes
        for i in range (nvertices):
            (latitude, longitude) = model_vertices[i]
            vertices[i] = toRect (longitude, latitude)  # change the order
            if model > 1 and longitude < SAA_LONGITUDE_CUTOFF:
                longitude += 360.
            minmax_long[0] = min (longitude, minmax_long[0])
            minmax_long[1] = max (longitude, minmax_long[1])
            minmax_lat[0] = min (latitude, minmax_lat[0])
            minmax_lat[1] = max (latitude, minmax_lat[1])
        middle_long = (minmax_long[0] + minmax_long[1]) / 2.
        middle_lat = (minmax_lat[0] + minmax_lat[1]) / 2.
        middle_SAA = toRect (middle_long, middle_lat)

        # for each row in TIMELINE table
        for k in range (nelem):
            hst = toRect (longitude_col[k], latitude_col[k])
            flag[k] = testWithinSAA (hst, vertices, middle_SAA)

        return flag

    def shift1Info (self, shift1, cutoff):
        """Compute information about the SHIFT1 column in TIMELINE.

        Parameters
        ----------
        shift1: array_like
            Array of SHIFT1[abc] values during the exposure.

        cutoff: float
            The cutoff value specified by the user; in this case this
            will be interpreted as the factor by which the standard
            deviation of SHIFT1 is to be multiplied in order to get
            the actual cutoff value.

        Returns
        -------
        tuple of two floats
            The first element is the median of the shift1 values, taken
            after clipping outliers.  The second element is the cutoff
            value specified by the user multiplied by the standard
            deviation of the sigma-clipped shift1 values.
        """

        # filter = "shift1 > X" is interpreted to mean that events should
        # be flagged as bad if shift1 - median (shift1) > X * sigma.
        # Clip outliers before computing the value of sigma to use.

        SIGMA_CLIP = 5.
        TINY_SIGMA = 0.001
        median_shift1 = findMedian (shift1)
        sigma_0 = shift1.std()
        if sigma_0 < TINY_SIGMA:        # this is a test for sigma = 0
            return (median_shift1, cutoff * TINY_SIGMA)

        select = np.ones (len (shift1), dtype=np.bool8)         # all True
        diff = shift1 - median_shift1
        select = np.where (np.abs (diff) > SIGMA_CLIP * sigma_0, False, select)

        median_shift1 = findMedian (shift1[select])
        sigma_1 = shift1[select].std()
        diff = shift1[select] - median_shift1
        select = np.where (np.abs (diff) > SIGMA_CLIP * sigma_1, False, select)

        median_shift1 = findMedian (shift1[select])
        sigma_2 = shift1[select].std()

        return (median_shift1, cutoff * sigma_2)

    def recomputeExptime (self):
        """Compute the exposure time and update the EXPTIME keyword.

        Returns
        -------
        gti: list of two-element lists
            Each element of `gti` is a two-element list, the start and stop
            times (in seconds since EXPSTART) for a "good time interval."
        """

        if self.events_time is None:
            events_hdu = self.fd[self.events_hdunum]
            self.events_time = cosutil.getColCopy (data=events_hdu.data,
                                                   column="time")
        nevents = len (self.events_time)
        zero = np.zeros (1, dtype=np.int8)
        one = np.ones (1, dtype=np.int8)
        # flag1 and flag2 are boolean arrays.  An element will be True if
        # the corresponding element of the DQ column (self.dq) is flagged as
        # being within a bad time interval (flag1) or as a burst (flag2).
        flag1 = np.greater (np.bitwise_and (self.dq,
                                            calcosparam.DQ_BAD_TIME),
                            0)
        flag2 = np.greater (np.bitwise_and (self.dq, calcosparam.DQ_BURST), 0)
        flag = np.logical_or (flag1, flag2)
        # iflag is an array of 8-bit signed integer flags, 1 where self.dq
        # includes either the burst flag or the bad-time flag, 0 elsewhere.
        iflag = np.where (flag, one, zero)
        del (flag, flag1, flag2)

        # dflag is non-zero (+1 or -1) at elements where iflag changes
        # from 0 to 1 or from 1 to 0.
        dflag = iflag[1:] - iflag[0:-1]
        # non_zero will be something like:  (array([ 2,  7, 11, 13]),)
        # For each value i in non_zero[0], dq[i+1] differs from dq[i].
        non_zero = np.where (dflag != 0)
        dflag_nz = dflag[non_zero]
        nz = non_zero[0]                # extract the array of indices
        n_indices = len (nz)

        gti_indices = []                # list of good time intervals
        # it_begin and it_end are the indices in events_time of the
        # beginning and end respectively of a good time interval.
        it_begin = None
        it_end = None
        if iflag[0] == 0:
            it_begin = 0
        for i in range (n_indices):
            if dflag[nz[i]] > 0:        # end of a good time interval
                it_end = nz[i]
                gti_indices.append ([it_begin, it_end])
            elif dflag[nz[i]] < 0:      # end of a bad time interval
                it_begin = nz[i] + 1
                it_end = None
            else:
                print "internal error:  dflag = %d" % dflag[nz[i]]
        if it_end is None and it_begin is not None:
            gti_indices.append ([it_begin, nevents-1])

        # Add up the good time intervals, and create a GTI list.
        gti = []
        for (it_begin, it_end) in gti_indices:
            gti.append ([self.events_time[it_begin],
                         self.events_time[it_end]])

        # The original GTI table (self.first_gti) may exclude some region or
        # regions (e.g. if data are ignored when the buffer is full), and
        # these would not show up in the DQ column if there were no events
        # during those time intervals.  To account for this, use the original
        # GTI table as a mask for the gti that we just found.
        gti = self.mergeGTI (self.first_gti, gti)
        # Round off the start and stop times to three decimals.
        gti = self.roundGTI (gti, precision=3)

        exptime = 0.
        for (t0, t1) in gti:
            exptime += (t1 - t0)

        # Update the EXPTIME keyword.
        old_exptime = self.fd[self.events_hdunum].header.get ("exptime", 0.)
        self.fd[self.events_hdunum].header.update ("exptime", exptime)
        if self.verbose and abs (exptime - old_exptime) > 0.032:
            print "EXPTIME changed from %.8g to %.8g" % (old_exptime, exptime)

        return gti

    def mergeGTI (self, first_gti, second_gti, precision=None):
        """Merge two good time intervals tables.

        Parameters
        ----------
        first_gti: list of two-element lists
            A list of [start, stop] good time intervals.  This is the list
            from the first GTI table.

        second_gti: list of two-element lists
            A second list of [start, stop] good time intervals.  This is
            the list based on the DQ column.

        Returns:
            A new gti list, consisting of intervals that overlap both
            `first_gti` and `second_gti`.
        """

        gti = []
        for (start_1, stop_1) in first_gti:
            for (start_2, stop_2) in second_gti:
                if stop_2 <= start_1:
                    continue
                if start_2 >= stop_1:
                    continue
                if start_2 >= start_1 and stop_2 <= stop_1:
                    gti.append ([start_2, stop_2])
                elif start_2 <= start_1 and stop_2 <= stop_1:
                    # we already know that stop_2 > start_1
                    gti.append ([start_1, stop_2])
                elif start_2 >= start_1 and stop_2 > stop_1:
                    # we already know that start_2 < stop_1
                    gti.append ([start_2, stop_1])
                elif start_2 <= start_1 and stop_2 >= stop_1:
                    gti.append ([start_1, stop_1])
                else:
                    print "oops ... confused!"

        return gti

    def roundGTI (self, input_gti, precision=3):
        """Round the start and stop times to `precision` decimals.

        Parameters
        ----------
        input_gti: list of two-element lists
            A list of [start, stop] good time intervals.

        precision: int
            The number of decimal places for rounding.

        Returns:
            A new gti list with times rounded off.
        """

        gti = []
        for (t0, t1) in input_gti:
            t0 = round (t0, precision)
            t1 = round (t1, precision)
            gti.append ([t0, t1])

        return gti

    def saveNewGTI (self, gti):
        """Append new GTI information as a BINTABLE extension.

        Create and save a GTI extension.  If there is no GTI extension,
        or if there is only one, the new GTI will be appended as a new
        extension.  If there are already two or more GTI extensions, the
        last one (highest EXTVER) will be replaced.

        Parameters
        ----------
        gti: list of two-element lists
            A list of [start, stop] good time intervals.
        """

        len_gti = len (gti)
        col = []
        col.append (pyfits.Column (name="START", format="1D", unit="s"))
        col.append (pyfits.Column (name="STOP", format="1D", unit="s"))
        cd = pyfits.ColDefs (col)
        hdu = pyfits.new_table (cd, nrows=len_gti)
        hdu.header.update ("extname", "GTI")
        outdata = hdu.data
        startcol = outdata.field ("START")
        stopcol = outdata.field ("STOP")
        for i in range (len_gti):
            startcol[i] = gti[i][0]
            stopcol[i] = gti[i][1]

        if self.gti_hdunum is None:
            extver = 1
            inplace = False             # create a new GTI extension
        else:
            last_gti = self.fd[self.gti_hdunum]
            extver = last_gti.header.get ("extver", 1)
            # if there are already two GTI extensions, overwrite the last one
            inplace = (extver > 1)
            if not inplace:
                extver += 1

        hdu.header.update ("extver", extver)
        if inplace:
            self.fd[self.gti_hdunum] = hdu
            if self.verbose:
                print "GTI extension updated in-place"
        else:
            self.fd.append (hdu)
            self.fd[0].header.update ("nextend", len(self.fd)-1)
            if self.verbose:
                print "New GTI extension appended"

    def findExtensions (self):
        """Find EVENTS, GTI and TIMELINE extensions.

        This checks each extension in the input file to find all
        extensions with keyword EXTNAME equal to (case insensitive)
        "EVENTS" (there should be exactly one), "GTI" (we expect one or
        two), or "TIMELINE" (we expect one, but a raw file or an old
        corrtag file might have none).

        These three attributes will be assigned by this method:
            self.events_list        EVENTS tables
            self.gti_list           GTI tables
            self.tl_list            TIMELINE tables
        Each element is a two-element list (extver and hdunum) that
        identifies one extension in the input file.  `extver` is the
        value of keyword EXTVER, the extension version number.  `hdunum`
        is the header/data unit number of the extension (primary header
        is 0).
        """

        self.events_list = []
        self.gti_list = []
        self.tl_list = []
        for hdunum in range (1, len(self.fd)):
            extname = self.fd[hdunum].header.get ("extname", "MISSING")
            extname = extname.upper()
            if extname == "EVENTS" or extname == "GTI" or \
               extname == "TIMELINE":
                extver = self.fd[hdunum].header.get ("extver", 1)
                if extname == "EVENTS":
                    self.events_list.append ([extver, hdunum])
                if extname == "GTI":
                    self.gti_list.append ([extver, hdunum])
                if extname == "TIMELINE":
                    self.tl_list.append ([extver, hdunum])
        self.gti_list.sort()            # sort by EXTVER

    def findHduNum (self):
        """Select a header/data unit from each list.

        A RuntimeError exception will be raised if there is more than one
        EVENTS table or more than one TIMELINE table.

        These three attributes will be assigned by this method:
            self.events_hdunum      HDU number of EVENTS table
            self.first_gti_hdunum   HDU number of first GTI table
            self.gti_hdunum         HDU number of last GTI table
            self.tl_hdunum          HDU number of TIMELINE table
        These are the header/data unit numbers of the EVENTS table,
        the last (highest EXTVER) GTI table, and the TIMELINE table
        respectively.  The value will be None if there are no elements
        in the corresponding self.events_list, self.gti_list, or
        self.tl_list.
        """

        if len (self.events_list) > 1:
            raise RuntimeError, "%d EVENTS tables in %s" % \
                                (len (self.events_list), self.input)

        if self.events_list:
            # extract hdunum from [extver, hdunum] for the EVENTS HDU
            self.events_hdunum = self.events_list[0][1]
        else:
            self.events_hdunum = None

        if self.gti_list:
            # extract hdunum from [extver, hdunum] for the first and last GTI
            self.first_gti_hdunum = self.gti_list[0][1]
            self.gti_hdunum = self.gti_list[-1][1]
        else:
            self.first_gti_hdunum = None
            self.gti_hdunum = None

        if len (self.tl_list) > 1:
            raise RuntimeError, "%d TIMELINE tables in %s" % \
                                (len (self.tl_list), self.input)
        if self.tl_list:
            # extract hdunum from [extver, hdunum] for the TIMELINE HDU
            self.tl_hdunum = self.tl_list[0][1]
        else:
            self.tl_hdunum = None

    def getFirstGTI (self):
        """Get the contents of the first GTI table.

        Attribute first_gti will be assigned by this method.
        """

        data = self.fd[self.first_gti_hdunum].data
        startcol = data.field ("START")
        stopcol = data.field ("STOP")
        self.first_gti = []
        for i in range (len (data)):
            self.first_gti.append ([startcol[i], stopcol[i]])

    def writeNewOutputFile (self):
        """Write the current HDUList to a new output file."""

        self.fd[0].header.update ("filename",
                                  os.path.basename (self.output))
        if self.verbose:
            print "Writing to", self.output
        self.fd.writeto (self.output)

#
#### Interfaces used by TEAL
#
def run(configobj=None):
    """TEAL interface for running this code."""
    ### version 2011 Apr 7

    tlf = TimelineFilter (configobj["input"], configobj["output"],
                          filter=configobj["filter"],
                          verbose=configobj["verbose"])

def getHelpAsString(fulldoc=True):
    """Return help info from <module>.help in the script directory"""

    if fulldoc:
        basedoc = __doc__
    else:
        basedoc = ""
    helpString = basedoc + "\n"
    helpString += "Version " + __version__ + "\n"

    helpString += teal.getHelpFileAsString(__taskname__, __file__)

    helpString += __usage__

    return helpString

# Set up doc string without the module level docstring included for
# use with Sphinx, since Sphinx will already include module level docstring
# xxx TimelineFilter.__doc__ = getHelpAsString(fulldoc=False)

def help():
    print getHelpAsString()

if __name__ == "__main__":

    main()
