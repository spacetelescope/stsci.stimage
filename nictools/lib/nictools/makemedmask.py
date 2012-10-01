#! /usr/bin/env python
#
# Author: Dave Grumm (based on work by Tomas Dahlen and Eddie Bergeron)
# Program: makemedmask.py
# Purpose: routine to create median mask for 'Finesky'
# History: 03/04/08 - first version

import numpy as N
import pyfits, sys, string, time
from stsci.convolve import boxcar
from optparse import OptionParser
import fsutil, opusutil

__version__ = "0.1 (2008 Mar 4)"

ASIZE = 256 # length of cal array
ERROR_RETURN = 2

class Makemedmask:
    """ Create and output a median mask from cal files and blt files

    Notes
    ------
    Syntax for using this class::

        m_mask = makemedmask.Makemedmask( medfile='medout2.fits', callist='/hal/data2/dev/nicmos_ped/inlist1.lst',
                thresh = 0.7,verbosity = 1)
        m_mask.makemask()

    Full signatures for methods are as follows:
    """

    def __init__( self, thresh=None, medfile=None, callist=None, verbosity=0 ):
        """constructor

        Parameters
        ----------
        thresh : float
            threshold used in making mask
        medfile : string
            name of output masked median image
        callist : string
            name of text file containing cal file names
        verbosity : {0,1,2}
            verbosity level (0 for quiet, 1 verbose, 2 very verbose)

        """

        if (thresh == None):  thresh = fsutil.thresh
        if (medfile == None):  medfile = fsutil.medfile
        if (callist == None):  callist = fsutil.callist
        if (type(thresh) == str) : # ensure that thresh is a float
           thresh = string.atof(thresh)

        self.thresh = thresh
        self.medfile = medfile
        self.callist = callist
        self.verbosity = verbosity
        self.mm_version = __version__
        self.mm_run = time.asctime()

        if (self.verbosity >=1):
            print ' Makemedmask run on ',self.mm_run, ', using code version: ', self.mm_version


    def makemask( self ):
      """ Make and output mask
      """
      thresh = self.thresh
      medfile = self.medfile
      callist = self.callist
      verbosity = self.verbosity

      blot1 = N.zeros((ASIZE,ASIZE), dtype=N.float64)

      # open callist and read file names
      cfile = open(callist, 'r')
      calfiles = []
      num_files = 0
      while 1: # first count number of files in list and generate file list
          line = cfile.readline()
          if not line: break
          num_files += 1
          calfiles.append( line )
      cfile.close()

      bltfiles = [] # list of blot files

      if (verbosity >=1 ):  print 'There are' ,num_files,'cal files. They are : '
      for ii in range(num_files):
         calfiles[ii].lstrip().rstrip() # strip leading and trailing whitespace
         calfile_prefix = calfiles[ii].split('_')[0]
         if (verbosity >=1 ): print '  calfiles[',ii,'] = ',calfiles[ii]

      #  associate blt files with cal files
         bltfile =  calfile_prefix+str("_cal_sci1_blt.fits")
         bltfile.lstrip().rstrip() # strip leading and trailing whitespace
         bltfiles.append( bltfile )
         bltfiles[ii] = bltfile

      im_cube = N.zeros((ASIZE, ASIZE, num_files), dtype=N.float64)
      blot_cube = N.zeros((ASIZE, ASIZE, num_files), dtype=N.float64)

      for kk in range(num_files):
         fh_cal = pyfits.open(calfiles[ kk ])
         fh_blot = pyfits.open(bltfiles[ kk ])
         im_cube[:,:,kk] = fh_cal[1].data
         blot_cube[:,:,kk] = fh_blot[0].data

   # make mask from blotted images
      mask_cube = N.zeros((ASIZE, ASIZE, num_files), dtype=N.float64)

      for ii in range(num_files):
         mm = N.zeros((ASIZE, ASIZE), dtype=N.float64)
         dif_0 = blot_cube[:,:,ii]
         dif = N.reshape( dif_0,((ASIZE,ASIZE)))
         ur =  dif > thresh
         mm[ ur ] = 1

   # expand the mask
         mm = boxcar( mm,(3,3))  # smooth over 3x3 ; this will differ from IDL's "smooth" which ...
         #  ... leaves boundary values unchanged, which is not an option in convolve's boxcar

         ur =  mm <> 0.0
         mm = N.zeros((ASIZE, ASIZE), dtype=N.float64)
         mm[ ur ] = 1
         mask_cube[:,:,ii] = mm

   ## make the masked median image
      if (verbosity >=1 ):  print ' Making the masked median image ... '

      maskall= N.zeros((ASIZE, ASIZE), dtype=N.float64)

      for jj in range(ASIZE):
        for kk in range(ASIZE):
           uu = mask_cube[ kk,jj,:] <> 1
           im_sub =  im_cube[kk,jj,uu]
           im_sub_size = im_sub.size
           im_1d = N.reshape( im_sub, im_sub.size)
           if ( im_sub_size  > 0 ):  maskall[ kk,jj ]= N.median(im_1d)

   # get primary header of 1st cal file to copy to output
      fh_cal0 = pyfits.open(calfiles[ 0 ])
      pr_hdr = fh_cal0[0].header

      write_to_file(maskall, medfile, pr_hdr, verbosity)

      if (verbosity >=1 ):  print 'DONE'

    def print_pars(self):
        """ Print parameters.
        """
        print 'The input parameters are :'
        print '  thresh:  ' , self.thresh
        print '  medfile:  ' , self.medfile
        print '  callist:  ' , self.callist



def write_to_file(data, filename, hdr, verbosity):
    """ Write data to specified filename with specified header

    Parameters
    -----------
    data : ndarray
        numpy array
    filename : string
        name of output file
    hdr : pyfits Header object
        header for output file
    verbosity : {0,1,2}
        verbosity level (0 for quiet, 1 verbose, 2 very verbose)

    """
    fimg = pyfits.HDUList()
    fimghdu = pyfits.PrimaryHDU( header = hdr)
    fimghdu.data = data
    fimg.append(fimghdu)
    fimg.writeto(filename)
    if (verbosity >=1 ): print '...wrote masked median image to: ',filename


if __name__=="__main__":

     """Get input file and other arguments, and call CalTempFromBias.
        The command-line options are::

            -q (quiet)
            -v (very verbose)

        Parameters
        -----------
        cmdline : list of strings
            command-line arguments
     """

     usage = "usage:  %prog [options]"
     parser = OptionParser( usage)

    # add options and set defaults for parameters
     parser.set_defaults( verbosity = fsutil.QUIET)
     parser.add_option( "-q", "--quiet", action = "store_const",
                        const = fsutil.QUIET, dest = "verbosity",default=None,
                        help = "quiet, print nothing")
     parser.add_option( "-v", "--verbose", action="store_const",
                        const = fsutil.VERY_VERBOSE, dest="verbosity",default=None,
                        help="very verbose, print lots of information")
     parser.add_option( "-t", "--thresh", dest = "thresh",default = fsutil.thresh,
                        help = "threshold for making masked median image.")
     parser.add_option( "-m", "--medfile", dest = "medfile",default = fsutil.medfile,
                        help = "name of output masked median file.")
     parser.add_option( "-c", "--callist", dest = "callist",default = fsutil.callist,
                        help = "name of file containing list of cal files.")

     (options, args) = parser.parse_args()

     fsutil.setVerbosity( options.verbosity)
     verbosity = options.verbosity

     fsutil.setThresh(options.thresh )
     if options.thresh!=None: thresh = options.thresh

     fsutil.setMedfile(options.medfile )
     if options.medfile!=None: medfile = options.medfile

     fsutil.setCallist(options.callist )
     if options.callist!=None: callist = options.callist

     try:
       m_mask = Makemedmask( thresh=thresh, medfile=medfile, callist=callist, verbosity=verbosity )

       if (verbosity >=1 ):  m_mask.print_pars()

       m_mask.makemask()

       del m_mask

     except Exception, errmess:
       opusutil.PrintMsg("F","FATAL ERROR "+ str(errmess))
       sys.exit( ERROR_RETURN)
