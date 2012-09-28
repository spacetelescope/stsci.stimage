#! /usr/bin/env python
"""
WFPC2Destreak - Module for performing destreak correction on WFPC2 images

:Outline:

1. In the 'interior' image region (starting to the right of the pyramid region), eliminate the CRs, and
      calculate the mean (im_mean) and sigma (im_sigma):

     - Over the entire c0 image, cosmic rays are identified and masked in the c0 data
     - For the entire image, the global mean and the (clipped) sigma is calculated for all unmasked pixels
     - For each row, the mean is calculated for all unmasked pixels
     - For each row, the difference between the mean and the global mean is subtracted from the c0 data

2. The modified c0 data is written to the file <dataset>_bjc_<chip>.fits ( 'bjc' stands for 'bias jump corrected')

:Command Line Options:

Linux command line short options and defaults (set in wfpc2util.py)::

     -g: group (default = 4)
     -b: bias_thresh (default = 100000.)
     -r: row_thresh (default = 0.1)
     -v: verbosity (default = verbose)
     -m: input_mask (default = None)
     -i: niter (default = 5)

:Examples:

A. For a dataset with multiple groups, to process group 4 using a bias threshold=280 and row threshold=0.2::

    hal> ./wfpc2destreak.py "u96r0603m_c0h.fits"  -g 4 -b 280. -r 0.2  -v

  This can also be specified using the 'long options'::

    hal> ./wfpc2destreak.py "u96r0603m_c0h.fits" --group=4 --bias_thresh=280. --row_thresh=0.2

B. To allow the routine to run with all of the defaults::

    hal> ./wfpc2destreak.py "u96r0603m_c0h.fits"

C. For a dataset with a single group, using defaults for the thresholds::

    hal> ./wfpc2destreak.py "u96r0603m_c0h.fits"  -g 0

D. Same as example F, but specifing an input mask to use::

    hal> ./wfpc2destreak.py "u96r0603m_c0h.fits"  -g 0  -m "mask_u8zq0104.fits"

E. Same as example F, but specifing 3 iterations for the CR rejection ::

    hal> ./wfpc2destreak.py "u96r0603m_c0h.fits"  -g 0  -i 3

F. Run the routine for group 3 of a geis image  ::

    hal> ./wfpc2destreak.py "ub080106m.c0h" -g 3


Example 'A' under pyraf::

 --> wfp = wfpc2destreak.Wfpc2destreak( "u96r0603m_c0h.fits", group=4, bias_thresh=280, row_thresh=0.2)
 --> wfp.destreak()

Example 'A' under stsdas ( after loading hst_calib and wfpc )::

    --> from wfpc2tools import wfpc2destreak
    --> wfp = wfpc2destreak.Wfpc2destreak( "u96r0603m_c0h.fits", group=4, bias_thresh=280, row_thresh=0.2)
    --> wfp.destreak()

or ::

    --> import wfpc2tools
    --> wfp = wfpc2tools.wfpc2destreak.Wfpc2destreak( "u96r0603m_c0h.fits")
    --> wfp.destreak()

:Functions:

"""
#
# Authors: Dave Grumm
# Program: wfpc2destreak.py
# Purpose: routine to remove streaks due to bias from specified chip of wfpc2 data
# History: 07/10/07 - first version
#          08/10/07 - use overscan in the specified chip in the input d0f file, and apply it to the
#                     corresponding chip in the c0f file,
#                   - optionally do the 'overscan correction' only by setting overscan_only=True (default),
#                   - output file name is now dervied from the input prefix and relevant chip
#          09/04/07 - now copies input header to output file
#                   - hard-coded chip-specific column cutoff values are used when calculating bias
#                     corrections for each row.
#          10/10/07 - uses calculated values of the pyramid region's clipped mean and standard deviation
#                     and values of the image's clipped mean and standard deviation to determine which
#                     algorithm to use: Pass1 only, Pass1 and Pass2, or no correction
#                   - the user parameter 'overscan_only' is no longer used:
#                        -- "overscan_only=True" (Pass 1 only) => internal variable alg_type=PASS1
#                        -- "overscan_only=False" (Pass 1 & 2) => internal variable alg_type=PASS12
#                        --  alg_type=None for no correction to be applied
#                   - the keyword 'CORR_ALG' is written to the header specifying which algorithm was applied
#                   - copies header from appropriate chip of c0f file
#          10/17/07 - change code to copy header from appropriate group of c0h file
#          11/06/07 - change code to create output header as union of the group header and end of primary
#                     header from c0h file; modified code to handle single group cases
#          11/08/07 - change routine so it can be used as standalone program
#          11/09/07 - made script usable also as a unix standlaone program; output keywords for
#                     pyr_mean, pyr_sigma, im_mean, im_sigma
#          11/15/07 - added check for number of good values in image; will abort without correction if there are none
#          01/08/08 - to eliminate cases in which NSIGMA < 0, switched CR rejection method for image region
#                     to use cr_reject(), as pyramid region does; forced line slope fit to be 0.0 in both regions
#          01/30/08 - added user parameter 'bias_thresh': if this value is exceeded by the value of BIASEVEN from
#                     the c0f file, no correction will be applied.  The default is to apply a correction if no
#                     threshold is specified.
#                   - added user parameter 'row_thresh': if this value exceeds the absolute value of the
#                     calculated correction for a given row, no correction will be performed for that row.
#                   - added user parameter 'force_alg_type' to force a particular algorithm type. Allowed
#                     values are PASS1, PASS2, PASS12, and OMIT. Leaving this parameter blank is the default,
#                     and results in the program determining the algorithm type and applying the correction.
#          02/08/08 - Because the group-specific value for BIASEVEN is not easily accessible, changed the code to
#                     instead compare 'bias_thresh' to the calculated value of the image mean (im_mean).
#                   - the keyword 'BIASTHRE' is written to the header specifying the value of bias_thresh.
#                   - the keyword 'ROWTHRES' is written to the header specifying the value of row_thresh.
#                   - the keyword 'FORCETYP' is written to the header designating if the user has forced a specific
#                     correction type.
#          02/14/08 - Changed the logic of the routine so that if an algorithm type is specified, that correction
#                     will be made unless IMMEAN > BIASTHRESH, in which case no correction will be made and CORR_ALG
#                     will be set to 'skipped'.  As before, only rows will be corrected for which the absolute
#                     value of the correction is above the row threshold that the user set.
#          03/14/08 - CR mask written as root+'_bjc_'+'Pmask'+(group)+'.fits' for the d0f pyramid region
#                   - CR mask written as root+'_bjc_'+'Imask'+(group)+'.fits' for the d0f image region
#                   - statistics are now written to the corrected file for the d0f pyramid region, original
#                     c0f image region, corrected c0f image region, original d0f image region, and masked d0f image region.
#                     Keywords are written for the mean, standard deviation, minimum, maximum, and number of pixels
#                     for each of these 5 regions. The naming convention for these 30 keywords is:
#                   -- P/C/D for either pyramid region, c0f image region, or d0f image region;
#                   -- ORG/MSK/COR for either original (uncorrected) data, masked data, or corrected data;
#                   -- MEAN/STD/MIN/MAX/PIX for either mean, sigma, minimum, maximum, of number of pixels in the region.
#                      For example, PMSKMAX is the value of the maximum of the masked pixels in the pyramid region; and
#                      CCORSTD is the sigma of the corrected (c0f) image region.
#                   - These values are also optionally written to the screen.
#                   - The following keywords have been replaced (old -> new):
#                        PYRMEAN -> PMSKMEAN
#                        PYRSIGMA -> PMSKSTD
#                        IMMEAN -> DMSKMEAN
#                        IMSIGMA -> DMSKSTD
#          03/20/08 - the pyraf command line parser is now used for all arguments. See the "Usage" section
#                     below for examples in which parameters are overwritten.
#                   - when using the pyraf/python command line, for each parameter that the user does not specify,
#                     he/she is prompted for accepting the default value or overriding it.
#          03/21/08 - added some parameter type checking for linux comand line usage
#          04/03/08 - the calculation of the row-specific correction using PASS2 will ignore pixels that deviate from
#                     the mode of the row by more than 4 sigma. The mode will be calculated by binning all
#                     pixels in row in bins of width 1, and determining which bin has the highest frequency. Sigma
#                     will be calculated by iterative sigma-clipping all pixels over the entire image.
#          05/02/08 - instead of calculating corrections based on the d0f data and applying them to the c0f data, the
#                     routine now reads a c0h fits file (whose image data are reals), calculates corrections
#                     based on this image data, and applies the corrections to this data, updating the file
#                     in-place. As a result, the keywords that are output have been modified.
#          06/05/08 - reverting back to writing the corrected output to the file <dataset>_bjc_<chip>.fits,
#                     as in "u9ux010gm_c0h_bjc_4.fits"
#          06/13/08 - exposed n_mad as a settable parameter; output as keyword CRNMAD
#          06/19/08 - added option for user to supply an input mask instead of using the mask generated by the
#                     cosmic ray rejection routine.
#                   - change PASS2 so that for each row, the mean of all unmasked pixels is used; the mode will no longer be used
#          06/24/08 - added option for user to supply the number of iterations for the CR rejection
#          07/10/08 - changed output image size to match that of the input image for PASS2
#          07/17/08 - because calculations involving the pyramid region are no longer relevant, PASS2
#                     is the only correction type now supported, as specifying a correction type is
#                     no longer allowed.  All pyramid region-related calculations and keywords
#                     have been removed. The bias threshold and n_mad parameters have been removed
#          07/22/08 - fixed bug in cr rejection, in which the slope of the fit plane was forced to be 0 (left over
#                     from earlier tests of NSIGMA)
#          07/23/08 - put bias_thresh back in as a settable parameter; set default row_thresh = 0.1
#          09/25/08 - added check for __name__
#          03/12/10 - added support for geis, waiver fits, and multi-extension fits input

from __future__ import division # confidence medium
import pyfits
import numpy as N
from stsci.convolve import boxcar
import stsci.tools
from optparse import OptionParser
import stsci.ndimage as ndimage
from stsci.tools import fileutil
import wfpc2util, opusutil, sys, string

__version__ = "2.2 (2010 March 12)"

NUM_SIG = 2.5  # number of sigma to use in sigma clipping
TOT_ITER = 4   # maximum number of iterations for sigma clipping
ERROR_RETURN = 2

class Wfpc2destreak:
    """ Calculate magnitude of and remove streaks from specified group of wfpc2 data.

    """

    def __init__( self, input_file, input_mask=None, group=None, verbosity=0, bias_thresh=None, row_thresh=None, \
                  niter=None):
        """

        Parameters
        ----------
        input_file : str
            name of the c0h file to be processed
        input_mask : str
            name of the input mask
        group : int
            number of group to process
        verbosity : str
            verbosity level (0 for quiet, 1 verbose, 2 very verbose)
        bias_thresh : float
            bias threshold (no correction will be performed if this is exceeded by im_mean)
        row_thresh : float
            row threshold (no correction will be performed if this exceeds the calculated row correction)
        niter : int
            number of iterations for CR rejection

        Examples
        --------
        >>> wfpc2_d = wfpc2destreak.Wfpc2destreak( filename, input_mask=input_mask, group=group, verbosity=verbosity,
               bias_thresh=bias_thresh, row_thresh=row_thresh,  niter=niter)
        >>> wfpc2destreak.Wfpc2destreak.destreak(wfpc2_d)

        """

        # do some parameter type checking, and for python interface, set defaults and check unspecified pars
        if (( __name__ == 'wfpc2destreak') | ( __name__ == 'wdestreak') | ( __name__ == 'wfpc2tools.wfpc2destreak')):
           [group, bias_thresh, row_thresh, niter] = check_py_pars(input_file, group, bias_thresh, row_thresh, \
                                                                                    input_mask, niter)
        else:
           [group, bias_thresh, row_thresh, niter] = check_cl_pars(input_file, group, bias_thresh, row_thresh, \
                                                                                    input_mask, niter)

        self.input_file = input_file
        self.group = int(group)
        self.verbosity = verbosity
        self.bias_thresh = float(bias_thresh)
        self.row_thresh = float(row_thresh)
        self.input_mask = input_mask
        self.niter = niter

    def destreak( self ):
        """ Method to perform destreak correction.
        """

        input_file = self.input_file
        group = self.group
        verbosity = self.verbosity
        bias_thresh = self.bias_thresh
        row_thresh = self.row_thresh
        input_mask = self.input_mask
        niter = self.niter

        file_prefix = input_file.split('.')[0]
        fh_c0 = fileutil.openImage( input_file )
        data_cube = fh_c0[group].data
        c0_data = data_cube
        c0_hdr = fh_c0[0].header
        xsize = c0_data.shape[0]; ysize = c0_data.shape[1]

        clip_row_mean = N.zeros(ysize); good_row_mean = N.zeros(ysize)

       # if input_mask is specified, read data and verify that it has the correct shape
        if input_mask != None:
           fh_mask = pyfits.open(input_mask)
           mask_data = fh_mask[0].data
           if (c0_data.shape != mask_data.shape):
                opusutil.PrintMsg("F","ERROR - input mask has incorrect shape.")
                sys.exit( ERROR_RETURN)

        SubarrayLevel = c0_data[:,:]

    # perform CR rejection on desired subarray of image section
        image_data = c0_data[ :, :]  # NOTE - this is now the entire image section

        if ( input_mask != None ):
           fh_mask = pyfits.open(input_mask)
           masked_image = fh_mask[0].data
           if (verbosity >=1 ): print 'Using masked image from input mask file:', input_mask
        else:
           im_mask = cr_reject(image_data, niter)
           im_mask_2d = N.resize( im_mask, [image_data.shape[0], image_data.shape[1]])
           masked_image = image_data * (1-im_mask_2d)  # gives 0 where there are Cosmic Rays in image section
           self.input_mask = "None"  # for writing to header

        if (verbosity >=1 ):  print 'Masked image has min, mean, max = ', masked_image.min(),',', masked_image.mean(),',', masked_image.max()


    # calculate stats for all pixels in c0 image region
        self.dorgmean = image_data.mean()
        self.dorgstd = image_data.std()
        self.dorgmin = image_data.min()
        self.dorgmax = image_data.max()
        self.dorgpix = image_data.shape[0]*image_data.shape[1]

    # calc stats for unmasked pixels in c0 image region
        IR_nz = N.where( masked_image <> 0)
        self.dmskmean = masked_image[IR_nz].mean()
        self.dmskstd = masked_image[IR_nz].std()
        self.dmskmin = masked_image[IR_nz].min()
        self.dmskmax = masked_image[IR_nz].max()
        self.dmskpix = masked_image[IR_nz].shape[0]

    # calculate image mean from all unmasked pixels in image data
        good_image_pix = N.where( masked_image > 0.0)
        good_image_data = masked_image[ good_image_pix ]

        if (len(good_image_data) < 1 ):  # no good values in image, so will abort this run, making no correction
             opusutil.PrintMsg("F","ERROR - unable to calculate statistics on image, so will perform no correction")
             sys.exit( ERROR_RETURN)

        im_mean = good_image_data.mean()
        im_sigma =good_image_data.std()

        if (bias_thresh < im_mean): # so apply no correction
             alg_type = "Skipped"
             alg_cmt = "No correction applied"
             if (verbosity >1 ): print ' The specified correction will be skipped because bias_thresh < im_mean'
             sys.exit( ERROR_RETURN )

    # write mask file for c0 image region
        mask_file = file_prefix + str('_bjc_Imask_')+str(group)+str('.fits')

        write_mask( masked_image, mask_file )

        if (verbosity >=1 ): print 'Wrote mask for c0 image region to: ',mask_file

    # calculate statistics for original c0 image data
        orig_c0_data = c0_data[:,:].copy().astype(N.float32)

        self.dorgmin = orig_c0_data.min()
        self.dorgmax = orig_c0_data.max()
        self.dorgmean = orig_c0_data.mean()
        self.dorgstd = orig_c0_data.std()
        self.dorgpix = orig_c0_data.shape[0]*orig_c0_data.shape[1]

        to_sub_1_max = -wfpc2util.HUGE_VAL ; to_sub_1_min = wfpc2util.HUGE_VAL
        to_sub_2_max = -wfpc2util.HUGE_VAL ; to_sub_2_min = wfpc2util.HUGE_VAL

        low_row_p2 = 0 #  number of rows below row_threshold in PASS2
        high_row_p2 = 0 #  number of rows above row_threshold in PASS2

        row_data_b_mean = N.zeros(ysize)
        to_subtract_2 = N.zeros((ysize),dtype = N.float32)

        for i_row in range (0, ysize-1 ):
           row_data_b = masked_image[ i_row, :]

           good_pix = N.where( row_data_b > 0.0 )
           good_val =  row_data_b[ good_pix ]

           bad_pix = N.where( row_data_b <= 0.0 )
           bad_val =  row_data_b[ bad_pix ]

           if (verbosity >1 ):  # print stats of good and rejected pixels for this row
              print '' ; print ' row = ' , i_row
              if   good_val.size > 0 :
                 print ' good pixels: number, min, mean, max, std = ' ,  good_val.size,good_val.min(),good_val.mean(),good_val.max(),good_val.std()

           if (  len(good_val) > 0 ):
               clip_row_mean[ i_row] = good_val.mean()
           else:
               clip_row_mean[ i_row] = 0.0

        glob_im_clip_mean = clip_row_mean[1:ysize-2].mean() # calculate global clipped mean

        for i_row in range (1, ysize-2 ):      # for each row, subtract difference between clipped mean and global clipped mean

           to_subtract_2[ i_row ] = clip_row_mean[ i_row ]- glob_im_clip_mean

           if (abs(to_subtract_2[ i_row ]) < row_thresh ): # make no correction
             to_subtract_2[ i_row ] = 0.0
             low_row_p2 +=1
           else:
             high_row_p2 +=1
             to_sub_2_max = max(to_sub_2_max, to_subtract_2[ i_row ])
             to_sub_2_min = min(to_sub_2_min, to_subtract_2[ i_row ])

           c0_data[i_row,:] -=  to_subtract_2[ i_row ]


        corr_c0_data = c0_data[: ,:].astype(N.float32)

        # calculate and display statistics for corrected c0 image data
        self.dcormin = corr_c0_data.min()
        self.dcormax = corr_c0_data.max()
        self.dcormean = corr_c0_data.mean()
        self.dcorstd = corr_c0_data.std()
        self.dcorpix = corr_c0_data.shape[0]*corr_c0_data.shape[1]

        if (verbosity >=1 ):
            print 'The following means and sigmas pertain to all (uncorrected and corrected) rows:'
            print '  the total number of uncorrected, corrected rows: ', low_row_p2,',', high_row_p2
            if (high_row_p2 > 0 ):
               print '  the fraction of rows corrected: ', (high_row_p2+0.0)/( high_row_p2+ low_row_p2+0.0)
            print '  min, max of corrections are: ', to_sub_2_min,',',to_sub_2_max
            print '  mean, std of corrections are: ',to_subtract_2.mean(),',',to_subtract_2.std()

            print 'The following statistics keywords are written to the corrected data output:'
            print '  - For the unmasked pixels in the image region of the input data, the keywords and values for the '
            print '    mean, std, min, max, and number of pixels are :'
            print '    DMSKMEAN,    DMSKSTD,    DMSKMIN,    DMSKMAX,    DMSKPIX  '
            print '   ', self.dmskmean,'   ', self.dmskstd,'   ', self.dmskmin,'   ', self.dmskmax,'   ', self.dmskpix

            print '  - For all pixels in the image region of the original input data, the keywords and values for the '
            print '    mean, std, min, max, and number of pixels are :'
            print '    DORGMEAN,    DORGSTD,    DORGMIN,    DORGMAX,    DORGPIX  '
            print '   ', self.dorgmean,'   ', self.dorgstd,'   ', self.dorgmin,'   ', self.dorgmax,'   ', self.dorgpix

            print '  - For all pixels in the image region of the corrected input data, the keywords and values for the '
            print '    mean, std, min, max, and number of pixels are :'
            print '    DCORMEAN,    DCORSTD,    DCORMIN,    DCORMAX,    DCORPIX  '
            print '   ', self.dcormean,'   ', self.dcorstd,'   ', self.dcormin,'   ', self.dcormax,'   ', self.dcorpix

        outfile = file_prefix + str('_bjc_')+str(group)+str('.fits')
        update_header(self, c0_hdr) # add statistics keywords to header c0_hdr
        write_to_file(corr_c0_data, outfile, c0_hdr, verbosity, im_mean, im_sigma)

        # close open file handles
        if (fh_c0):
           fh_c0.close()

        if (verbosity >=1 ): print 'DONE '

    def print_pars(self):
        """ Print parameters method.
        """
        print 'The values of the input parameters are :'
        print '  input c0 file:  ' , self.input_file
        print '  input mask file:  ' , self.input_mask
        print '  group:  ' , self.group
        print '  bias_thresh:  ' , self.bias_thresh
        print '  row thresh:  ' , self.row_thresh
        print '  number of CR rejection iterations:  ' , self.niter


def check_py_pars(input_file, group, bias_thresh, row_thresh, input_mask, niter):
       """ When run under python, verify that each unspecified parameter should take the default value, and
            give the user the opportunity to change it.

        Parameters
        ----------
        input_file : str
            name of input file
        group : int
            number of group to process
        bias_thresh : float
            bias threshold (no correction will be performed if this is exceeded by im_mean)
        row_thresh : float
            row threshold (no correction will be performed if this exceeds the calculated row correction)
        input_mask : str
            name of input mask
        niter : int
            number of CR rejection iterations

        Returns
        -------
        group : int
        bias_thresh : float
        row_thresh : float
       """

       if (group == None):
            group = wfpc2util.group
            print ' You have not been specified a value for group; the default is:',  wfpc2util.group
            print ' If you want to use the default, hit <enter>, otherwise type in the desired value'
            inp = raw_input('? ')
            if inp == '':
               print ' The default value of ', group,' will be used'
            else:
               try:
                   group = string.atoi(inp)
               except:
                   print ' The value entered (',inp,') is invalid so the default will be used'

       if (bias_thresh == None):
            bias_thresh = wfpc2util.bias_thresh
            print ' You have not specified a value for bias_thresh; the default is:',  wfpc2util.bias_thresh
            print ' If you want to use the default, hit <enter>, otherwise type in the desired value'
            inp = raw_input('? ')
            if inp == '':
               print ' The default value of ', bias_thresh,' will be used'
            else:
               try:
                   bias_thresh = string.atof(inp)
               except:
                   print ' The value entered (',inp,') is invalid so the default will be used'

       if (row_thresh == None):
            row_thresh = wfpc2util.row_thresh
            print ' You have not specified a value for row_thresh; the default is:',  wfpc2util.row_thresh
            print ' If you want to use the default, hit <enter>, otherwise type in the desired value'
            inp = raw_input('? ')
            if inp == '':
               print ' The default value of ', row_thresh,' will be used'
            else:
               try:
                   row_thresh = string.atof(inp)
               except:
                   print ' The value entered (',inp,') is invalid so the default will be used'


       if (input_mask <> None):
            try:
               fh_mask = pyfits.open(input_mask)
            except:
               opusutil.PrintMsg("F","ERROR - unable to open the input mask "+ str(input_mask))
               sys.exit( ERROR_RETURN)

       if (niter == None):
            niter = wfpc2util.niter
            print ' You have not specified a value for niter; the default is:',  wfpc2util.niter
            print ' If you want to use the default, hit <enter>, otherwise type in the desired value'
            inp = raw_input('? ')
            if inp == '':
               print ' The default value of ', niter,' will be used'
            else:
               try:
                   niter = string.atoi(inp)
               except:
                   print ' The value entered (',inp,') is invalid so the default will be used'


       return group, bias_thresh, row_thresh, niter



def check_cl_pars(input_file, group,  bias_thresh, row_thresh, input_mask, niter):
       """ When run from linux command line, verify that each parameter is valid.

        Parameters
        ----------
        input_file : str
            name of input file
        group : int
            number of group to process
        bias_thresh : float
            bias threshold (no correction will be performed if this is exceeded by im_mean)
        row_thresh : float
            row threshold (no correction will be performed if this exceeds the calculated row correction)
        input_mask : str
            name of input mask file
        niter : int
            number of CR rejection iterations

        Returns
        -------
        group : int
        row_thresh : float
        niter : int

       """

       try:
           if (type( group ) == str):
              group = string.atoi(group)
       except:
           print ' The group value entered (',group,') is invalid. Try again'
           sys.exit( ERROR_RETURN)

       try:
           bias_thresh = string.atof(bias_thresh)
       except:
           print ' The bias threshold value entered (',bias_thresh,') is invalid.'
           sys.exit( ERROR_RETURN)

       try:
           row_thresh = string.atof(row_thresh)
       except:
           print ' The row threshold value entered (',row_thresh,') is invalid. Try again.'
           sys.exit( ERROR_RETURN)

       if (input_mask <> None):
            try:
               fh_mask = pyfits.open(input_mask)
            except:
               opusutil.PrintMsg("F","ERROR - unable to open the input mask "+ str(input_mask))
               sys.exit( ERROR_RETURN)

       try:
           niter = string.atoi(niter)
       except:
           pass

       if ((niter < 0 ) or (niter > 10 )):
           print ' The number of CR rejection iterations entered (',niter,') is invalid. Try again.'
           sys.exit( ERROR_RETURN)


       return group, bias_thresh, row_thresh, niter


def cr_reject( SubArray, niter ):

        """Identify and replace cosmic rays in the given subarray.

        Parameters
        ----------
        SubArray : ndarray
            subarray of the data
        niter : int
            the number of iterations used when rejecting cosmic rays

        """

        n_rms=[4., 4., 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
        n_neighbor=[2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
        n_mad = 15.

        sub_shape = SubArray.shape
        ny = sub_shape[0]; nx = sub_shape[1]

        linenum = N.arange( ny, dtype=N.float64)         # image line numbers
        # Independent variable (image line numbers, repeated) for fitting.
        indep_var = linenum.repeat( nx)  # 2-D

        # Note that sub0 is 1-D
        sub0 = SubArray.ravel().astype( N.float64)

        # no_value flags points where no value has been assigned to the
        # subarray level.
        no_value = N.where( sub0 <= 0., 1, 0)
        n = len( sub0)

        # First find extreme outliers, so they will not be included when
        # we do a least-squares fit to the data.

        line0 = float( ny//2 - 1) / 2.          # middle line, lower half
        line1 = float( ny//2 + ny - 1) / 2.     # middle line, upper half
        y0 = median( sub0[0:n//2], no_value[0:n//2])
        y1 = median( sub0[n//2:n], no_value[n//2:n])

        # fit is a straight line passing through (line0,y0) & (line1,y1).
        slope = (y1 - y0) / (line1 - line0)
        fit = slope * linenum + (y0 - slope * line0)

        residual = sub0 - fit.repeat( nx)


        # MAD is the median of the absolute values of deviations; for a
        # normal distribution the MAD is about 2/3 of the standard deviation.
        mad = median( N.absolute( residual), no_value)

        # First find very bright cosmic rays (large, positive outliers).
        cr = N.where( residual > (n_mad * mad), 1, 0)

        # Replace points identified as cosmic rays by the value of the
        # fit at that point; this is not really necessary because we'll
        # ignore currently identified cosmic rays when doing the fit.
        sub_pts = N.where( cr, fit.repeat( nx), sub0)

        mask = N.logical_or( no_value, cr)

        (slope, intercept) = fitline( indep_var, sub_pts, mask)
        # Note that fit has length ny, so will be shorter than sub_pts for
        # by the factor nx.

        fit = slope * linenum + intercept

        for iter in range( niter):
            residual = sub_pts - fit.repeat( nx)
            labels = N.where( mask == 0, 1, 0)
            rms = ndimage.standard_deviation( residual, labels=labels)
            rms = max (rms, 1.)

            new_cr = N.where( residual > (n_rms[iter] * rms), 1, 0)

            if ndimage.sum( new_cr) > 0:

                new_cr = check_neighbors( new_cr,
                             residual, n_neighbor[iter] * rms, sub_shape)
            cr = N.logical_or( cr, new_cr)

            mask = N.logical_or( no_value, cr)

            (slope, intercept) = fitline( indep_var, sub_pts, mask)

            fit = slope * linenum + intercept
            sub_pts = N.where( cr, fit.repeat( nx), sub_pts)

        return mask

# end of cr_reject()


def check_neighbors( new_cr, residual, cutoff, sub_shape):
        """Check for cosmic rays in neighboring pixels.

        Parameters
        ----------
        new_cr : ndarray
            1-D array of (int) ones or zeros (1 indicates a cosmic ray)
        residual : ndarray
            1-D array of residuals (float64), subarray - fit
        cutoff : float
            criterion for flagging an outlier as a cosmic ray
        sub_shape : tuple
            numbers of lines and columns in subarray

        Returns
        -------
        new_cr : int
            pixel position associated with identified cosmic ray,
                possibly with additional cosmic rays flagged
        """

        (ny, nx) = sub_shape
        new_cr_2D = N.reshape( new_cr, sub_shape)
        residual_2D = N.reshape( residual, sub_shape)

        (yindex, xindex) = N.where( new_cr_2D > 0)

        for iii in range( len( yindex)):
            i = xindex[iii]
            j = yindex[iii]
            # check neighbors to the left
            for ii in range (i-1, -1, -1):
                if new_cr_2D[j,ii] > 0:         # already flagged?
                    break
                if residual_2D[j,ii] > cutoff:
                    new_cr_2D[j,ii] = 1
                else:
                    break
            # check neighbors to the right
            for ii in range (i+1, nx):
                if new_cr_2D[j,ii] > 0:
                    break
                if residual_2D[j,ii] > cutoff:
                    new_cr_2D[j,ii] = 1
                else:
                    break

        return new_cr_2D.ravel()

# end of check_neighbors()


# Return the median of the array y, ignoring masked elements.
#
def median(  y, mask):
        """Return the median of the array y, ignoring masked elements.

        Parameters
        -----------
        y : ndarray
            array of values
        mask : ndarray
            array of (int32) ones or zeros (0 indicates a good value)

        Returns
        --------
        median : float
            median of y, ignoring masked elements

        """
        labels = N.where( mask == 0, 1, 0)      # 1 indicates a good value
        y_ok = N.compress( labels, y)

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



#-------------------------------------------------------------------------------
# Fit a straight line to y vs x, where mask is 0.
#
def fitline( x, y, mask):
        """Fit a straight line to y vs x, where mask is 0.

        Parameters
        -----------
        x : ndarray
            float64 array of independent-variable values
        y :  ndarray
            float64 array of dependent-variable values
        mask : ndarray
            int32 array of ones or zeros (0 indicates a good value)

        Returns
        --------
        coeffs : tuple
            coefficients of fit: tuple of the slope and intercept

        """

        labels = N.where( mask == 0, 1, 0)      # 1 indicates a good value
        mean_x = ndimage.mean( x, labels=labels)
        mean_y = ndimage.mean( y, labels=labels)
        dx = x - mean_x
        temp1 = dx * (y - mean_y)
        temp2 = dx * dx
        num = ndimage.sum( temp1, labels=labels)
        denom = ndimage.sum( temp2, labels=labels)
        if denom == 0.:
            raise ValueError, "Error fitting a line to subarray."
        slope = num / denom

        intercept = mean_y - slope * mean_x

        return (slope, intercept)

# end of fitline()


def update_header(self, hdr):
    """ update header from input c0 file with specified header, and updated data

    Parameters
    ----------
    self : object
        Wfpc2destreak object containing results to be recorded to header
    hdr : object
        PyFITS header object

    """
    hdr.update(key='BIASTHRE', value=self.bias_thresh, comment="bias threshold" )
    hdr.update(key='ROWTHRES', value=self.row_thresh, comment="row threshold" )

    hdr.update(key='DCORPIX', value=self.dcorpix, comment="number of pixels in corrected c0 image region" )
    hdr.update(key='DCORMEAN', value=self.dcormean, comment="mean in corrected c0 image region" )
    hdr.update(key='DCORSTD', value=self.dcorstd, comment="sigma in corrected c0 image region" )
    hdr.update(key='DCORMIN', value=self.dcormin, comment="minimum in corrected c0 image region" )
    hdr.update(key='DCORMAX', value=self.dcormax, comment="maximum in corrected c0 image region" )

    hdr.update(key='DORGPIX', value=self.dorgpix, comment="number of pixels in original c0 image region" )
    hdr.update(key='DORGMEAN', value=self.dorgmean, comment="mean in original c0 image region" )
    hdr.update(key='DORGSTD', value=self.dorgstd, comment="sigma in original c0 image region" )
    hdr.update(key='DORGMIN', value=self.dorgmin, comment="minimum in original c0 image region" )
    hdr.update(key='DORGMAX', value=self.dorgmax, comment="maximum in original c0 image region" )

    hdr.update(key='DMSKPIX', value=self.dmskpix, comment="number of unmasked pixels in c0 image region" )
    hdr.update(key='DMSKMEAN', value=self.dmskmean, comment="mean of unmasked c0 image region" )
    hdr.update(key='DMSKSTD', value=self.dmskstd, comment="sigma of unmasked c0 image region" )
    hdr.update(key='DMSKMIN', value=self.dmskmin, comment="minimum of unmasked c0 image region" )
    hdr.update(key='DMSKMAX', value=self.dmskmax, comment="maximum of unmasked c0 image region" )

    hdr.update(key='USERMASK', value=self.input_mask, comment="name of input mask file" )
    hdr.update(key='CRITER', value=self.niter, comment="number of CR rejection iterations" )


def write_mask(data, filename):
    """ write specified mask

    Parameters
    ----------
    data : ndarray
        mask array
    filename : string
        mask file name

    """
    fimg = pyfits.HDUList()
    fimghdu = pyfits.PrimaryHDU()
    fimghdu.data = data
    fimg.append(fimghdu)
    fimg.writeto(filename)


def write_to_file(data, filename, hdr, verbosity, im_mean, im_sigma):
    """
    Write mean and sigma to file.

    Parameters
    -----------
    data : ndarray
        array of floats
    filename : string
        mask file name
    hdr: object
        Pyfits header object
    verbosity: int
        verbosity level (0 for quiet, 1 verbose, 2 very verbose)
    im_mean : float
        clipped mean of image region
    im_sigma : float
        clipped sigma of image region

    """

    fimg = pyfits.HDUList()
    hdr.update(key='IMMEAN', value=im_mean, comment="clipped mean of image region" )
    hdr.update(key='IMSIGMA', value=im_sigma, comment="clipped sigma of image region" )
    fimghdu = pyfits.PrimaryHDU( header = hdr)
    fimghdu.data = data
    fimg.append(fimghdu)
    fimg.writeto(filename)

    if (verbosity >=1 ): print 'Wrote updated data to: ',filename


if __name__ =="__main__":
    """Get input file and other arguments, and call Wfpc2destreak

    Notes
    -----
    The command-line options are:
        -q (quiet)
        -v (very verbose)

    Parameters
    -----------
    cmdline : list
        command-line arguments as list of strings

    """

    usage = "usage:  %prog [options] inputfile"
    parser = OptionParser( usage)

    if ( len(sys.argv) > 1 ):
        filename = sys.argv[1]
    else:
        opusutil.PrintMsg("F","ERROR - the input file must be specified.")
        sys.exit( ERROR_RETURN)

    parser.set_defaults( verbosity = wfpc2util.VERBOSE)

    parser.add_option( "-q", "--quiet", action = "store_const",
            const = wfpc2util.QUIET, dest = "verbosity",
            help = "quiet, print nothing")
    parser.add_option( "-v", "--verbose", action="store_const",
            const = wfpc2util.VERY_VERBOSE, dest="verbosity",
            help="very verbose, print lots of information")
    parser.add_option( "-g", "--group", dest = "group",default = wfpc2util.group,
            help = "number of group to process.")
    parser.add_option( "-b", "--bias_thresh", dest = "bias_thresh",default = wfpc2util.bias_thresh,
            help = "bias threshold.")
    parser.add_option( "-r", "--row_thresh", dest = "row_thresh",default = wfpc2util.row_thresh,
            help = "row threshold.")
    parser.add_option( "-m", "--input_mask", dest = "input_mask",default = wfpc2util.input_mask,
            help = "input mask file name.")
    parser.add_option( "-i", "--niter", dest = "niter",default = wfpc2util.niter,
            help = "number of CR rejection iterations.")

    (options, args) = parser.parse_args()

    wfpc2util.setVerbosity( options.verbosity)
    verbosity = options.verbosity

    wfpc2util.setGroup(options.group )
    if options.group!=None: group = options.group

    wfpc2util.setBias_thresh(options.bias_thresh )
    if options.bias_thresh!=None: bias_thresh = options.bias_thresh

    wfpc2util.setRow_thresh(options.row_thresh )
    if options.row_thresh!=None: row_thresh = options.row_thresh

    wfpc2util.setInput_mask(options.input_mask )
    input_mask = options.input_mask

    wfpc2util.setNiter(options.niter )
    if options.niter!=None: niter = options.niter

    try:
       wfpc2_d = Wfpc2destreak( filename, input_mask=input_mask, group=group, verbosity=verbosity, bias_thresh=bias_thresh,\
                                row_thresh=row_thresh, niter=niter)

       if (verbosity >=1 ):
            print 'The version of this routine is: ',__version__
            wfpc2_d.print_pars()
       Wfpc2destreak.destreak(wfpc2_d)

       del wfpc2_d

    except Exception, errmess:
       opusutil.PrintMsg("F","FATAL ERROR "+ str(errmess))
       sys.exit( ERROR_RETURN)
