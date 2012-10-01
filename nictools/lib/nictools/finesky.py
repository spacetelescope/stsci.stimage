#! /usr/bin/env python
#
# Author: Dave Grumm (based on work by Tomas Dahlen and Eddie Bergeron)
# Program: finesky.py
# Purpose: routine to create median mask
# History: 03/07/08 - first version

from __future__ import division  # confidence high
import numpy as N
import pyfits, sys, string, time
from stsci.convolve import boxcar
from optparse import OptionParser
import fsutil, opusutil, shutil

__version__ = "0.1 (2008 Mar 12)"

ASIZE = 256 # length of cal array
ERROR_RETURN = 2
HUGE_VAL = 9.9E99      # used in creating mask
MAX_NUM_MISSING = 1000 # warning displayed if number of pixels having no valid values exceeds this

class Makemedmask:
    """ Create and output a median mask from cal files and blt files

    Notes
    ------
    Syntax for using this class::

       m_mask = finesky.Makemedmask( medfile='medout2.fits', callist='/hal/data2/dev/nicmos_ped/inlist1.lst',
                thresh = 0.7,verbosity = 1)
       m_mask.makemask()

    Full set of parameters for class methods are as follows.
    """

    def __init__( self, thresh=None, medfile=None, callist=None, verbosity=0 ):
        """constructor

        Parameters
        -----------
        thresh : real
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
            print ' Finesky run on ',self.mm_run, ', using code version: ', self.mm_version


    def makemask( self ):
      """ Make and output mask
      """
      thresh = self.thresh
      medfile = self.medfile
      callist = self.callist
      verbosity = self.verbosity

      num_missing = 0  # number of pixels having no valid value among all images
      num_no_neigh  = 0 # number of pixels having no valid value among all images and having no valid neighboring values
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
      none_valid= N.zeros((ASIZE, ASIZE), dtype=N.int8)   # flag for pixels with no valid values
      no_neigh= N.zeros((ASIZE, ASIZE), dtype=N.int8)   # flag for pixels with no valid neighbors

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

   # make the masked median image
      if (verbosity >=1 ):  print ' Making the masked median image ... '

      maskall= N.zeros((ASIZE, ASIZE), dtype=N.float64) - 1 # -1 for later validity check
      for jj in range(ASIZE):
        for kk in range(ASIZE):
           uu = mask_cube[ kk,jj,:] <> 1
           im_sub =  im_cube[kk,jj,uu]
           im_sub_size = im_sub.size
           im_1d = N.reshape( im_sub, im_sub.size)
           if ( im_sub_size > 0 ):
               maskall[ kk,jj ]= N.median(im_1d)
           else:
               num_missing += 1
               none_valid[ kk, jj ] = 1

      new_maskall = maskall.copy()

      neigh_radius = 1
      for jj in range(ASIZE): # start loop to handle pixels with no valid values
        for kk in range(ASIZE):
           if ( none_valid[ kk, jj ] == 1 ): # use mean of surrounding valid median values
              ymin = max( kk - neigh_radius,0 ); ymax = min( kk + neigh_radius,ASIZE )
              xmin = max( jj - neigh_radius,0 ); xmax = min( jj + neigh_radius,ASIZE )

              med_neigh = maskall[ ymin:ymax+1, xmin:xmax+1]

              # select only neighbors that are valid and not the center pixel (because non_valid flagged)
              vv = (med_neigh > -HUGE_VAL)
              vv = True and ( med_neigh < HUGE_VAL )
              vv = True and ( med_neigh <> -1 )

              med_neigh_size = med_neigh[ vv ].size

              if med_neigh_size == 0 :
                 num_no_neigh += 1
                 no_neigh[ kk, jj ] = 1

              if med_neigh_size > 0 :
                 new_maskall[ kk, jj ] = N.mean(med_neigh[ vv ])

   # loop over pixels to handle pixels having no valid values, by using averages of valid neighboring pixels
      neigh_radii = [ 2,3,100 ]
      for iter in range(3):
          num_no_neigh = 0 # reset for this iteration
          this_radius =   neigh_radii[ iter ]

          for jj in range(ASIZE): # start loop to handle pixels with no valid values
            for kk in range(ASIZE):
               if ( no_neigh[ kk, jj ] == 1 ): # use mean of surrounding valid median values
                  ymin = max( kk-this_radius,0 ); ymax = min( kk+this_radius,ASIZE )
                  xmin = max( jj-this_radius,0 ); xmax = min( jj+this_radius,ASIZE )

                  med_neigh = maskall[ ymin:ymax+1, xmin:xmax+1]

                  # select only neighbors that are valid and not the center pixel
                  vv = (med_neigh > -HUGE_VAL)
                  vv = True and ( med_neigh < HUGE_VAL )
                  vv = True and ( med_neigh <> -1 )

                  med_neigh_size = med_neigh[ vv ].size

                  if med_neigh_size == 0 :
                     num_no_neigh += 1

                  if med_neigh_size > 0 :
                     new_maskall[ kk, jj ] = N.mean(med_neigh[ vv ])
                     no_neigh[ kk, jj ] =  0 # reset for next iteration

   # overwrite maskall invalids with new_maskall neighbor means
      pp = (none_valid == 1)
      maskall[ pp ] = new_maskall[ pp ]

   # get primary header of 1st cal file to copy to output
      fh_cal0 = pyfits.open(calfiles[ 0 ])
      pr_hdr = fh_cal0[0].header

      write_to_file(maskall, medfile, pr_hdr, verbosity)

      if num_missing > MAX_NUM_MISSING:
         print '  '
         print '********************************************************************************'
         print 'WARNING : The number of pixels having no valid values among all images is ', num_missing,','
         print ' which exceeds the specified threshold of ', MAX_NUM_MISSING,' pixels.'
         print '********************************************************************************'
         print '  '

      if (verbosity >=1 ):
          print 'The number of pixels having no valid values among all images is ', num_missing
          print 'Because they have no valid values, these pixels could not have their medians calculated,'
          print 'so instead of a median value an attempt was made to calculate an average of valid neighboring '
          print 'values. The number of these pixels that were replaced by an average of their neighbors is ', num_missing-num_no_neigh


   # subtract counts in med image (maskall) from SCI ext in each cal file, output to cal2 file
      if (verbosity >=1 ):
          print 'Subtracting sky median image from SCI ext for the cal files: '
      for ii in range(num_files):
          fh_cal = pyfits.open(calfiles[ ii ])
          cal_data =  fh_cal[1].data
          new_cal_data = cal_data - maskall
          calfile_prefix = calfiles[ii].split('.')[0]
          new_cal_filename = str(calfile_prefix)+str("2.fits")
          src_file = calfiles[ii].split('.')[0] # need to strip trailing tabs and \n for filecopy
          src_file = str(src_file)+str(".fits")  # need to strip trailing tabs and \n for filecopy
          shutil.copyfile(src_file, new_cal_filename)
          fh_new_cal = pyfits.open(new_cal_filename, mode='update' )
          fh_new_cal[1].data -= maskall
          fh_new_cal.close()
          if (verbosity >=1 ):
             print '  Writing new cal file: ',new_cal_filename
      if (verbosity >=1 ):
          print 'Done subtracting sky median image from SCI ext for the cal files. '
          print 'DONE'

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
    ----------
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
        ----------
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
