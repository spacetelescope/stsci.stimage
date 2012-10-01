#! /usr/bin/env python
#
# Author : Dave Grumm (based on an IDL routine by Adam Riess)
# Program: nic_rem_persist.py
# Purpose: routine to remove persistence from NICMOS data
# History:
#  11/19/07 - first version
#  11/26/07 - incorporates dq arrays to make combined mask
#  08/05/08 - script now takes a single *ped.fits file as input.
#           - script now calculates the ring median image.
#           - in pipeline use, execution of this task is controlled from within runcalsaa.py
#             by the calibration switch keyword BEPCORR
#           - the following keywords have been added, with the source of their default values:
#             -- the names of the persistence model and persistence mask are read from the
#                keywords PMODFILE and PMSKFILE in the header of the input *ped.fits file
#             -- the minimum accepted value for the calculated value of the persistence is given by
#                the keyword BEPVALLO in the file PMODFILE; this value is written to the input file
#             -- the minimum accepted value for the fraction of pixels used is given by the
#                keyword BEPUSELO in the file PMODFILE; this value is written to the input file
#             -- the calculated value of the persistence is written to the keyword BEPSCALE
#             -- the calculated value of the fraction of pixels used is written to the keyword BEPFRAC
#           - routines added to check validity of input parameters
#  08/13/08 - for pipeline use, added attribute for output to trailer file
#           - moved setting of parameters into constructor and separate functions in persutil
#  08/25/08 - for consistency with other NICMOS tasks in the pipeline, changing BEPCORR
#             to BEPCORR plus BEPDONE. BEPCORR will not be updated; BEPDONE will be updated to
#             PERFORMED, SKIPPED, or OMITTED
#  12/15/08 - Will use default values for used_lo and and persist_lo if BEPEUSELO and BEPVALLO are not in the PMODFILE.
#  01/15/09 - Changing interface so that ped file and cal file are input, and correction based on calculation of
#             ped file (CALCFILE) is applied to cal file (TARGFILE). This corresponds to TRAC ticket #317.
#
from __future__ import division # confidence high
import pyfits
import numpy as N
import sys, time
import opusutil
import persutil
import string
import stsci.ndimage as ndimage    # for median_filter

__version__ = "1.6 (2009 Mar 26)"


_success  = 0
_none     = 1
_error    = 2
_abort    = 3



ERROR_RETURN = -2
CLIP = 4  # number of std to use in sigma clipping

class NicRemPersist:
    """ Remove bright earth persistence persistence from CAL file of NICMOS data (on which pedsub has been run) using a medianed persistence model,
        based on correction calculated from PED file.

    Notes
    ------
    The syntax for using this class under Python or PyRAF::

      --> nrp = nic_rem_persist.NicRemPersist('n9r7b2bjq_ped.fits', persist_model = 'persistring.fits', persist_mask = 'persist_mask.fits',
                  used_lo = .3, persist_lo = 0.8)
      --> nrp.persist()

    The linux command line syntax for calling this module::

       hal> ./nic_rem_persist.py 'n9r7b2bjq_ped.fits','n9r7b2bjq_cal.fits' -d 'same_as_persistring.fits' -m 'same_as_persist_mask.fits' -u 0.4 -p 1.3 -v

    The full description of the parameters required by this class is as follows.
    """

    def __init__( self, calcfile, targfile, verbosity=1 ,persist_lo=None, used_lo=None, persist_model=None, persist_mask=None, run_stdout=None):
        """constructor

        Parameters
        ----------
        calcfile : string
            name of ped file
        targfile : string
            name of cal file
        persist_lo : float
            minimum allowed value of the persistence
        used_lo : float
            minimum allowed value of the fraction of pixels used
        persist_model : string
            filename containing persistence frame (ring median of)
        persist_mask : string
            filename containing pixel mask
        run_stdout : file handle
            open trailer file (pipeline use only)

        """
        if (run_stdout != None): # set standard output to trailer file
            self.orig_stdout = sys.stdout # save original stdout
            sys.stdout = run_stdout  # set stdout to open trailer file

        persutil.setVerbosity(verbosity)

        if (verbosity > 0 ):
            current_time = time.asctime()
            print '=== BEP', __version__,' starting at ', current_time

        # get parameter values if not specified on command line
        if (persist_lo == None ):
            persist_lo = persutil.getPersist_lo( calcfile)

        if (used_lo == None ):
            used_lo = persutil.getUsed_lo( calcfile)

        if (persist_model == None ):
            persist_model = persutil.getPersist_model( calcfile)

        if (persist_mask == None ):
            persist_mask = persutil.getPersist_mask( calcfile)

        # do some parameter type checking, if not already performed at the command-line
        if ( __name__ != "__main__"):
              [ persist_lo, used_lo, persist_model, persist_mask ] = check_py_pars(self, calcfile, targfile, persist_lo, used_lo, persist_model, persist_mask)

        self.calcfile = calcfile
        self.targfile = targfile
        self.persist_lo = persist_lo
        self.used_lo = used_lo
        self.persist_model = persist_model
        self.persist_mask = persist_mask
        self.verbosity = verbosity
        self.run_stdout = run_stdout
        self._applied = None

    def persist( self ):
        """ remove persistence due to the full bright Earth.

        """

        calcfile = self.calcfile
        targfile = self.targfile
        persist_lo = self.persist_lo
        used_lo = self.used_lo
        persist_model = self.persist_model
        persist_mask = self.persist_mask
        verbosity = self.verbosity

       # get PED file header to read PMODFILE and PMSKFILE
        try:
           fh_calcfile = pyfits.open( calcfile, mode='update')
        except:
           opusutil.PrintMsg("F","ERROR "+ str('Unable to open PED file') + str(calcfile))
           self._applied = _error

           sys.exit( ERROR_RETURN)

        try:  # read the image for the persistence model
           fh_per_median = pyfits.open( self.persist_model )
        except:
           opusutil.PrintMsg("F"," ERROR "+ str('Unable to open medianed persistence model file') + str(persist_model))
           self._applied = _error

           sys.exit( ERROR_RETURN)
        ps = fh_per_median[0].data

        try:
           fh_targfile = pyfits.open( targfile, mode='update')
        except:
           opusutil.PrintMsg("F","ERROR "+ str('Unable to open CAL file') + str(targfile))
           self._applied = _error

           sys.exit( ERROR_RETURN)

        try:  # read the image for the mask
           fh_per_mask = pyfits.open( self.persist_mask )
        except:
           opusutil.PrintMsg("F"," ERROR "+ str('Unable to open the mask file') + str(persist_mask))
           self._applied = _error

           sys.exit( ERROR_RETURN)
        per_mask = fh_per_mask[0].data  # persistence mask data (0 is good, non-0 is bad)

        calcimage = fh_calcfile[1].data    #  load input PED image data

        targimage = fh_targfile[1].data    #  load input CAL image data

        med_mask = calcimage *0  # for size; needed for median call

        dq_data = fh_calcfile[3].data  # read the dq array data (0 is good, non-0 is bad)

        dq_plus_per_mask = dq_data + per_mask # combined mask

        cir_mask = make_footprint( 3, 6 ) # make annular mask for ring median with inner,outer radii = 3,6

        imrm = calcimage * 0.0
        ndimage.median_filter(calcimage, footprint = cir_mask, output = imrm)  #  generate ring median image

        src = calcimage - imrm  # this image is zero except for sources

        md, std = iterstatc( CLIP, src ) # returns clipped mean and standard deviation

        level = median(calcimage, med_mask)

    #   Sigma clip sources, and reject bright regions >1.5*median and un-masked.
    #   Only use pixels that are:
    #      within 'CLIP' standard deviations, and
    #      less than 150% of the median, and
    #      not set in combined mask
        gd = ((src < md+CLIP*std) & (src > md-CLIP*std) & (calcimage < 1.5*level) & (dq_plus_per_mask == 0))

        num_gd = N.add.reduce( N.add.reduce( gd ))
        used =  float(num_gd)/float( src.shape[0]* src.shape[1])
        self.used = used

    #     set up the matrices for a linear system of equations
        y = calcimage[gd]   #this is the image
        ng = len(y)
        x = N.zeros( [ng,2], dtype=N.float64 )
        x[:,0] = 1.0     # this part of the model is a flat sky
        x[:,1] = ps[gd]  # this part of the model is an image, i.e., the  persistence image

    #     inverts matrix to solve for the parameters
        delta = N.dot(N.dot(N.linalg.inv( N.dot( N.transpose(x),x)),N.transpose(x)),y)

        sky = delta[0]        # parameter 1: sky
        self.sky = sky

        persist = delta[1]    # parameter 2: level of persistence
        self.persist = persist

        correct = targimage - delta[1]*ps  # remove the least-squares fraction of the persistence image from the cal file image

        if (verbosity > 1):
              print ' sky =', sky, ' persist = ' , persist,' used =',used

    #     write correction if fraction of pixels used and calculated persistence exceed thresholds
        if ( persist < self.persist_lo ):
              self._applied = _none

              if (verbosity > 0):
                  print ' The calculated persistence for this file is ', persist,', which is below the minimum allowed value of the persistence (', persist_lo,').  Therefore no corrected image will be written.'

        elif ( used < self.used_lo ):
              self._applied = _none

              if (verbosity > 0):
                  print ' The fraction of ped file pixels used for the calculation is ', used,', which is below the minimum allowed value of the fraction (', used_lo,').  Therefore no corrected image will be written.'

              fh_targfile[0].header.update( key = 'BEPDONE', value = 'SKIPPED')
        else:   # update target file (CAL) with required keywords
              self._applied = _success

              fh_targfile[1].data = correct
              fh_targfile[0].header.update( key = 'BEPSCALE', value = persist)
              fh_targfile[0].header.update( key = 'BEPFRAC', value = used)
              fh_targfile[0].header.update( key = 'BEPVALLO', value = self.persist_lo)
              fh_targfile[0].header.update( key = 'BEPUSELO', value = self.used_lo)
              fh_targfile[0].header.update( key = 'BEPDONE', value = 'PERFORMED')

        if fh_calcfile:
           fh_calcfile.close()

        if fh_targfile:
           fh_targfile.close()

        if (verbosity > 0 ):
           self.print_pars()
           current_time = time.asctime()
           print '=== BEP finished at ', current_time

        if (self.run_stdout  != None):  #reset stdout if needed
           sys.stdout = self.orig_stdout

        return self._applied

    # end of def persist

    def print_pars(self ):
        """ Print input parameters and calculated values.
        """
        print 'The input parameters used are :'
        print '  The PED input file:  ' , self.calcfile
        print '  The CAL input file:  ' , self.targfile
        print '  Lower limit on persistence (persist_lo):  ' , self.persist_lo
        print '  Lower limit on fraction of pixels used (used_lo):  ' , self.used_lo
        print '  The (medianed) persistence model from the file (persist_model):  ' , self.persist_model
        print '  The mask from the file (persist_mask): ' ,self.persist_mask
        print 'The calculated values are :'
        print '  The persistence :  ' , self.persist
        print '  The sky :  ' , self.sky
        print '  The fraction of pixels used :  ' , self.used


def check_py_pars(self, calcfile, targfile, persist_lo, used_lo, persist_model, persist_mask):
       """ When run under python, check validity of input parameters. For unspecified \*_lo parameters, the
        user will be given the option of typing in a value or accepting the default value. For an unspecified
        model(mask) file, will try to get file name from PMODFILE(PMSKFILE) from input file header, otherwise
        will get default value from persutil.

        Parameters
        ------------
        calcfile : string
            name of PED file
        targfile : string
            name of CAL file
        persist_lo : float
            minimum allowed value of the persistence
        used_lo : float
            minimum allowed value of the fraction of pixels used
        persist_model : string
            filename containing persistence frame (ring median of)
        persist_mask : string
            filename containing pixel mask

        Returns
        --------
        persist_lo, used_lo : float
        persist_model, persist_mask : string

       """

       try:
            fh_calcfile = pyfits.open(calcfile)
            #in_ped_hdr = fh_calcfile[0].header
            fh_calcfile.close()
       except:
            opusutil.PrintMsg("F","ERROR - unable to open the PED file  "+ str(calcfile))
            self._applied = _error

            sys.exit( ERROR_RETURN)

       try:
            fh_targfile = pyfits.open(targfile)
            #in_cal_hdr = fh_targfile[0].header
            fh_targfile.close()
       except:
            opusutil.PrintMsg("F","ERROR - unable to open the CAL file  "+ str(targfile))
            self._applied = _error

            sys.exit( ERROR_RETURN)

       if (persist_lo == None):
            persist_lo = persutil.persist_lo
            print ' You have not been specified a value for persist_lo; the default is:', persutil.persist_lo
            print ' If you want to use the default, hit <enter>, otherwise type in the desired value'
            inp = raw_input('? ')
            if inp == '':
               print ' The default value of ', persist_lo,' will be used'
            else:
               try:
                   persist_lo = string.atof(inp)
               except:
                   print ' The value entered (',inp,') is invalid so the default will be used'

       if (used_lo == None):
            used_lo = persutil.used_lo
            print ' You have not been specified a value for used_lo; the default is:', persutil.used_lo
            print ' If you want to use the default, hit <enter>, otherwise type in the desired value'
            inp = raw_input('? ')
            if inp == '':
               print ' The default value of ', used_lo,' will be used'
            else:
               try:
                   used_lo = string.atof(inp)
               except:
                   print ' The value entered (',inp,') is invalid so the default will be used'

       if ( persist_model == None): # try getting PMODFILE from input file
           try:
               persist_model = in_hdr['PMODFILE']
           except:  # get default value
               persist_model = persutil.persist_model

       if ( persist_mask == None):# try getting PMSKFILE from input file
           try:
               persist_mask = in_hdr['PMSKFILE']
           except:  # get default value
               persist_mask = persutil.persist_mask

       return  persist_lo, used_lo, persist_model, persist_mask


def check_cl_pars(calcfile, targfile, persist_lo, used_lo):
   """ When run from linux command line, verify that each parameter is valid.

        Parameters
        ----------
        calcfile : string
            name of ped file
        targfile : string
            name of cal file
        persist_lo : float
            minimum allowed value of the persistence
        used_lo : float
            minimum allowed value of the fraction of pixels used

        Returns
        --------
        persist_lo, used_lo : float, float

   """

   try:
        fh_c0 = pyfits.open(calcfile)
        fh_c0.close()
   except:
        opusutil.PrintMsg("F","ERROR - unable to open the PED file  "+ str(calcfile))
        #self._applied = _error

        sys.exit( ERROR_RETURN)

   try:
        fh_c0 = pyfits.open(targfile)
        fh_c0.close()
   except:
        opusutil.PrintMsg("F","ERROR - unable to open the cal file  "+ str(targfile))
        #self._applied = _error

        sys.exit( ERROR_RETURN)

   try:
       if (type( persist_lo ) == str):
          persist_lo = string.atof(persist_lo)
   except:
       print ' The persist_lo value entered (',persist_lo,') is invalid. Try again'
       #self._applied = _error

       sys.exit( ERROR_RETURN)

   try:
       if (type( used_lo ) == str):
          used_lo = string.atof(used_lo)
   except:
       print ' The used_lo value entered (',used_lo,') is invalid. Try again'
       #self._applied = _error

       sys.exit( ERROR_RETURN)

   return  persist_lo, used_lo


def make_footprint(rin, rout):
    """ Make an annular mask footprint for use in ndimage.median_filter to create a ring median image.
        Sets all pixels between rin and rout to 1.

    Parameters
    ----------
    rin : int
        inner radius
    rout : int
        outer radius

    Returns
    --------
    mask : ndarray

    """
    asize = 2*rout+1
    mask = N.zeros( [asize,asize], dtype=N.int )

    x_ctr = int(asize/2.);  y_ctr = x_ctr

    for ii in range(asize):
       for jj in range(asize):
           d2 = (ii-x_ctr)**2 + (jj-y_ctr)**2
           if (d2 >= rin*rin and d2 <= rout*rout):
               mask[ ii, jj ] = 1

    return mask


def iterstatc( clip, d ):
    """  version of nicmos iterstat: calculate sigma-clipped mean
         and standart deviation

    Parameters
    ----------
    clip : float
        number of std to use in sigma clipping
    d : float
        array of values to sigma clip

    Returns
    --------
    clipped mean, clipped std :  float, float
    """

    img = d
    md = img.mean()
    std = img.std()
    for ii in range(6):
      gd = N.where((img < md+CLIP*std) & (img > md-CLIP*std))
      md = img[gd].mean()
      std = img[gd].std()

    return md, std


# Return the median of the array y, ignoring masked elements.
#
def median( y, mask):
    """Return the median of the array y, ignoring masked elements.

    Parameters
    ----------
    y : float
        array of values [2d]
    mask : int
        array of ones or zeros (0 indicates a good value) [2d]

    Returns
    -------
    median_y : float
        median of y, ignoring masked elements

    """

   # make the arrays 1d :
    yshape =  y.shape[0]* y.shape[1]
    y_1d = N.resize(y, yshape)
    labels = N.where( mask == 0, 1, 0)      # 1 indicates a good value
    labels_1d = N.resize(labels, yshape)

    y_ok = N.compress( labels_1d, y_1d)

    leny = len( y_ok)
    if leny < 1:
        return None
    elif leny == 1:
        return y_ok[0]
    elif leny == 2:
        return (y_ok[0] + y_ok[1]) / 2.

    index = y_ok.argsort()
    half_leny = leny // 2
    if half_leny * 2 < leny:
        return y_ok[index[half_leny]]
    else:
        return (y_ok[index[half_leny]] + y_ok[index[half_leny+1]]) / 2.

# end of median()


if __name__=="__main__":
        """Get input file and other arguments, and call nic_rem_persist.
            The command-line options are::

              -q (quiet)
              -v (very verbose)

            Parameters
            ----------
            cmdline : list of strings
                command-line arguments

        """

        if ( sys.argv[1] ): calcfile = sys.argv[1]
        if ( sys.argv[2] ): targfile = sys.argv[2]
        [ options, args, parser ] = persutil.getOptions()

        try:
            [ persist_lo, used_lo] = check_cl_pars(calcfile, targfile, persist_lo, used_lo)
            nrp = NicRemPersist( calcfile, targfile, persist_lo=options.persist_lo, used_lo=options.used_lo,
                                persist_model=options.persist_model, persist_mask=options.persist_mask,
                                verbosity=options.verbosity )
            nrp_stat = nrp.persist()

            del nrp
        except Exception, errmess:
            opusutil.PrintMsg("F","ERROR "+ str(errmess))
            sys.exit( ERROR_RETURN)
