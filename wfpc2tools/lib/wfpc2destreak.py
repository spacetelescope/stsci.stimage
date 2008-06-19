#! /usr/bin/env python 
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
#
# Outline:
#
# 1. In the pyramid region of the c0 data, eliminate the CRs, then calculate the mean (pyr_mean) and sigma (pyr_sigma)
# 2. In the 'interior' image region (starting to the right of the pyramid region), eliminate the CRs, and
#      calculate the mean (im_mean) and sigma (im_sigma)
# 3. Calculate how high above the pyr values the image values are by:
#        im_mean = pyr_mean + NSIGMA*pyr_sigma
# 4. If the algorithm type is not forced by the user, the routine determines which correction algorithm
#    to use and applies it:
#   A.  If NSIGMA < = 1.0 apply no correction
#   B.  If NSIGMA >1.0 and NSIGMA < =11.0 and im_sigma <= 2.0 apply Pass1 & 2
#   C.  If NSIGMA >1.0 and NSIGMA < =11.0 and im_sigma > 2.0 apply no correction
#   D.  ELSE apply Pass1 only
#
# ... where Pass1 is done as follows:
#     - In the leading columns (2->group-specific value ), cosmic rays are identified and
#         masked in the c0 data
#     - For each row within these columns, the mean is calculated of the unmasked
#         pixels; for all rows the 'global pyramid mean' is calculated
#     - For each row, the smoothed mean is calculated over 3 rows
#     - For each row in the image, the difference between the smoothed mean and the 'global pyramid mean'
#         is subtracted from the c0 data
#
# ... and Pass2 is done as follows:
#     - Over the entire c0 image, cosmic rays are identified and masked in the c0 data
#     - For the entire image, the global mean and the (clipped) sigma is calculated for all unmasked pixels
#     - For each row, the mean is calculated for all unmasked pixels 
#     - For each row, the difference between the mean and the global mean is subtracted from the c0 data
#
# 5. The modified c0 data is written to the file <dataset>_bjc_<chip>.fits
#      
# Linux command line short options and defaults (set in wfpc2util.py):
#     -g: group (default = 4)
#     -b: bias_thresh (default = 99999.)
#     -r: row_thresh (default = 0.)
#     -v: verbosity (default = verbose)
#     -f: force_alg_type (default = None)
#     -n: n_mad (default = 15.)    
#     -m: input_mask (default = None)    
#
# Usage examples:
#    A. For a dataset with multiple groups, to process group 4 using a bias threshold=280 and row threshold=0.1,
#       letting the routine decide which correction type to apply:
#         hal> ./wfpc2destreak.py "u8zq0104m_c0h.fits"  -g 4 -b 280. -r 0.1  -v
#    B. To force a specific correction type (say PASS2):
#         hal> ./wfpc2destreak.py "u8zq0104m_c0h.fits"  -g 4 -b 280. -r 0.1 -f PASS2 
#       This can also be specified using the 'long options':
#         hal> ./wfpc2destreak.py "u8zq0104m_c0h.fits" --group=4 --bias_thresh=280. --row_thresh=0.1
#                --force_alg_type=PASS2 
#    C. To force the routine to run without applying a correction:
#         hal> ./wfpc2destreak.py "u8zq0104m_c0h.fits"  -g 4 -b 280. -r 0.1 -f OMIT -v
#    D. To allow the routine to run with all of the defaults:
#         hal> ./wfpc2destreak.py "u8zq0104m_c0h.fits"
#    E. For a dataset with a single group, using defaults for the thresholds, forcing PASS1:
#         hal> ./wfpc2destreak.py "u8zq0104m_c0h.fits"  -g 0 -f PASS1
#    F. For a dataset with a single group, using defaults for the thresholds, setting CR rejection parameter to 20:
#         hal> ./wfpc2destreak.py "u8zq0104m_c0h.fits"  -g 0 -n 20. 
#    F. Same as example F, but specifing an input mask to use:
#         hal> ./wfpc2destreak.py "u8zq0104m_c0h.fits"  -g 0 -n 20. -m "mask_u8zq0104.fits"
#
# Example 'A' under pyraf:
# --> wfp = wfpc2destreak.Wfpc2destreak( "u8zq0104m_c0h.fits", group=4, bias_thresh=280, row_thresh=0.1,
#                                          force_alg_type="PASS2")
# --> wfp.destreak()
#
###########################################################################

import pyfits
import numpy as N
from convolve import boxcar 
import pytools
from pytools import numerixenv      
from optparse import OptionParser
import ndimage
import wfpc2util, opusutil, sys, string

__version__ = "2.13 (2008 June 19)"

NUM_SIG = 2.5  # number of sigma to use in sigma clipping
TOT_ITER = 4   # maximum number of iterations for sigma clipping
ERROR_RETURN = 2

class Wfpc2destreak:
    """ Calculate magnitude of and remove streaks from specified group of wfpc2 data.

    example: 
       wfpc2_d = wfpc2destreak.Wfpc2destreak( filename, input_mask=input_mask, group=group, verbosity=verbosity,
               bias_thresh=bias_thresh, row_thresh=row_thresh, force_alg_type=force_alg_type, n_mad=n_mad) 
       wfpc2destreak.Wfpc2destreak.destreak(wfpc2_d) 
    """

    def __init__( self, input_file, input_mask=None, group=None, verbosity=0, bias_thresh=None, row_thresh=None, \
                  force_alg_type=None, n_mad=None):   
        """constructor

        @param input_file: name of the c0h file to be processed
        @type input_file: string
        @param input_mask: name of the input mask 
        @type input_mask: string
        @param group: number of group to process
        @type group: int
        @param verbosity: verbosity level (0 for quiet, 1 verbose, 2 very verbose)
        @type verbosity: string
        @param bias_thresh: bias threshold (no correction will be performed if this is exceeded by im_mean)
        @type bias_thresh: float
        @param row_thresh: row threshold (no correction will be performed if this exceeds the calculated row correction)
        @type row_thresh: float
        @param force_alg_type: algorithm type to force
        @type force_alg_type: string
        @param n_mad: CR rejection parameter    
        @type n_mad: float  
        """

        # do some parameter type checking
        if ( __name__ == 'wfpc2destreak'):  # for python interface, set defaults and check unspecified pars
           [group, bias_thresh, row_thresh, force_alg_type, n_mad] = check_py_pars(input_file, group, bias_thresh, row_thresh, \
                                                                                   force_alg_type, n_mad, input_mask)  
        else:
           [group, bias_thresh, row_thresh, force_alg_type, n_mad] = check_cl_pars(input_file, group, bias_thresh, row_thresh, \
                                                                                   force_alg_type, n_mad, input_mask)  

        self.input_file = input_file
        self.group = int(group) 
        self.verbosity = verbosity
        self.bias_thresh = float(bias_thresh)  
        self.row_thresh = float(row_thresh)
        self.force_alg_type = force_alg_type
        self.n_mad = float(n_mad)  
        self.input_mask = input_mask
        
    def destreak( self ):

        input_file = self.input_file
        group = self.group
        verbosity = self.verbosity
        bias_thresh = self.bias_thresh 
        row_thresh = self.row_thresh
        force_alg_type = self.force_alg_type
        n_mad = self.n_mad   
        input_mask = self.input_mask

        file_prefix = input_file.split('.')[0] 

    # read data from input c0h fits file:
        fh_c0 = pyfits.open(input_file) 
        c0_hdr = fh_c0[0].header 

        data_cube = fh_c0[0].data

        if ( group == 0 ): # for single group image case
            c0_data = data_cube
        else:
            c0_data = data_cube[group-1] # 0-based 
            
        xsize = fh_c0[0].header.get( "NAXIS1"); ysize = fh_c0[0].header.get( "NAXIS2")

        clip_row_mean = N.zeros(ysize); good_row_mean = N.zeros(ysize)

       # if input_mask is specified, read data and verify that it has the correct shape
        if input_mask != None:
           fh_mask = pyfits.open(input_mask) 
           mask_data = fh_mask[0].data
           if (c0_data.shape != mask_data.shape):
                opusutil.PrintMsg("F","ERROR - input mask has incorrect shape.")
                sys.exit( ERROR_RETURN)

    # read c0 leading columns (between col_min and col_max) and reject Cosmic Rays
        col_min = 2

        if ( group == 0 ): # for single group case only 
           col_max = 25; ix_min = 50; iy_min = 60  
        if ( group == 1 ): # set group-dependent value for col_max
           col_max = 25; ix_min = 50; iy_min = 60 
        if ( group == 2 ):
           col_max = 25; ix_min = 40; iy_min = 50 
        if ( group == 3 ):
           col_max = 15; ix_min = 40; iy_min = 50 
        if ( group == 4 ):
           col_max = 20 ; ix_min = 40; iy_min = 50

        SubarrayLevel = c0_data[:,col_min:col_max]

        mask = cr_reject(SubarrayLevel, n_mad)    # in pyramid region           
        mask_2d = N.resize( mask, [SubarrayLevel.shape[0], SubarrayLevel.shape[1]])    
        masked_SubarrayLevel = SubarrayLevel * (1-mask_2d)  # gives 0 where there are Cosmic rays in the pyramid region

    # calculate stats for all pixels in c0 pyramid region
        self.porgpix = SubarrayLevel.shape[0]* SubarrayLevel.shape[1]
        self.porgmean =  SubarrayLevel.mean(); self.porgstd = SubarrayLevel.std()
        self.porgmin = SubarrayLevel.min(); self.porgmax = SubarrayLevel.max()

    # write mask file for c0 pyramid region
        mask_file = file_prefix + str('_bjc_Pmask_')+str(group)+str('.fits')   
    
        write_mask( masked_SubarrayLevel, mask_file, verbosity )
        if (verbosity >=1 ): print 'Wrote mask for c0 pyramid region to: ',mask_file
        
    # calculate global mean from all unmasked pyramid region pixels in c0 file
        c0_data_pix = N.where( masked_SubarrayLevel > 0.0)
        all_good_data = masked_SubarrayLevel[ c0_data_pix ]
        
        glob_pyr_mean = all_good_data.mean() 
        pyr_mean = all_good_data.mean()  # clipped pyramid mean of c0 data
        pyr_sigma = all_good_data.std()  # clipped pyramid sigma of c0 data

    # calculate stats for unmasked pixels in c0 pyramid region
        BL_nz = N.where(masked_SubarrayLevel <> 0)
        self.pmskpix = masked_SubarrayLevel[BL_nz].shape[0]
        self.pmskmean = all_good_data.mean()
        self.pmskstd =  all_good_data.std()
        self.pmskmin =  all_good_data.min() 
        self.pmskmax = all_good_data.max()  
        
    # loop over all rows and calculate mean of unmasked pixels for each row for the pyramid region
        for i_row in range( 0, ysize-1 ):
           row_data_a = masked_SubarrayLevel[i_row,:].astype( N.float64) 

           if (row_data_a.std() <> 0.0) or (row_data_a.mean() <> 0.0 ):# for cases in which all values are 0.0
               row_data_pix = N.where( row_data_a > 0.0)
               good_row_data = row_data_a[ row_data_pix ]
               good_row_mean[ i_row ] = good_row_data.mean()
           else:
               good_row_mean[ i_row ] = 0.0 

        smoothed_row_mean = boxcar( good_row_mean,(3,))  # smooth row mean over 3 rows

    # perform CR rejection on desired subarray of image section 
        image_data = c0_data[ :, :]  # NOTE - this is now the entire image section

        if ( input_mask != None ):
           fh_mask = pyfits.open(input_mask) 
           masked_image = fh_mask[0].data
           if (verbosity >=1 ): print 'Using masked image from input mask file:', input_mask
        else:    
           im_mask = cr_reject(image_data, n_mad)  
           im_mask_2d = N.resize( im_mask, [image_data.shape[0], image_data.shape[1]])    
           masked_image = image_data * (1-im_mask_2d)  # gives 0 where there are Cosmic Rays in image section
           self.input_mask = "None"  # for writing to header

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

    # write mask file for c0 image region
        mask_file = file_prefix + str('_bjc_Imask_')+str(group)+str('.fits')

        if (verbosity >=1 ): print 'Wrote mask for c0 image region to: ',mask_file
        write_mask( masked_image, mask_file, verbosity )

    # calculate image mean from all unmasked pixels in image data
        good_image_pix = N.where( masked_image > 0.0)
        good_image_data = masked_image[ good_image_pix ]
    
        if (len(good_image_data) < 1 ):  # no good values in image, so will abort this run, making no correction
             opusutil.PrintMsg("F","ERROR - unable to calculate statistics on image, so will perform no correction")
             sys.exit( ERROR_RETURN)

        im_mean = good_image_data.mean()
        im_sigma =good_image_data.std()

    # Calculate the number of pyramid sigma above the clipped pyramid mean for the clipped image mean 
        nsigma = (im_mean - pyr_mean)/pyr_sigma

        if (verbosity >1 ):print 'Determining which algorithm to use ...'
   
        if (force_alg_type == None):#  apply the correction selected by the routine
           forced_type = 'No'
           if (bias_thresh >= im_mean):#  determine the type of correction to apply
              if ( nsigma < 1.0 ): # Case A            
                 if (verbosity >=1 ): print '  nsigma = ' , nsigma, ' < 1.0, so will apply no correction'
                 alg_type = "None"
                 alg_cmt = "No correction applied"
              elif (nsigma >= 1.0 and nsigma <= 11.0 and im_sigma <= 2.0): # Case B
                 if (verbosity >1 ):
                    print '  nsigma = ', nsigma , '( between 1.0 and 11.), and im_sigma = ', im_sigma ,' <=2.0 so will apply both PASS 1 & 2'
                    print '  Will use columns 2 through ', col_max,' in pyramid region to calculate Pass 1 corrections.'
                 alg_type = "PASS12"
                 alg_cmt = "Pass 1 and 2 corrections applied"
              elif (nsigma >= 1.0 and nsigma <= 11.0 and im_sigma > 2.0): # Case C
                 if (verbosity >1 ): print '  nsigma = ', nsigma , '( between 1.0 and 11.), and im_sigma = ', im_sigma , ' >2.0 so will apply no correction'
                 alg_type = "None"
                 alg_cmt = "No correction applied"
              else: # Case D
                 if (verbosity >1 ):             
                    print '  nsigma = ', nsigma ,' and im_sigma = ', im_sigma
                    print '  Neither   nsigma < 1.0 or  (nsigma >= 1.0 and nsigma <= 11.0 and im_sigma <= 2.0) so '
                    print '  will apply Pass 1 only'
                    print 'Will use columns 2 through ', col_max,' in pyramid region to calculate Pass 1 corrections.'
                 alg_type = "PASS1"
                 alg_cmt = "Pass 1 correction applied"
           else : # bias_thresh < im_mean) so apply no correction 
              alg_type = "Skipped"
              alg_cmt = "No correction applied"
              if (verbosity >1 ): print ' The specified correction will be skipped because bias_thresh < im_mean'
        else: # (force_alg_type <> None): correction type has been specified by user
           forced_type = 'Yes'
           if (bias_thresh >= im_mean):#  determine type of correction specified
              alg_type = force_alg_type
              if ( force_alg_type == 'PASS1'):
                 alg_cmt = "Pass 1 correction applied"
              elif ( force_alg_type == 'PASS2'):
                 alg_cmt = "Pass 2 correction applied"
              elif ( force_alg_type == 'PASS12'):
                 alg_cmt = "Pass 1 and 2 corrections applied"
              else:
                 alg_type = "None"
                 alg_cmt = "No correction applied"
              if (verbosity >1 ):
                 print ' The user has forced algorithm type: ' ,alg_type
           else:  # let the program decide which algorithm type
              alg_type = "Skipped"
              alg_cmt = "No correction applied"
              if (verbosity >1 ): print ' The specified correction will be skipped because bias_thresh < im_mean'
            
        # calculate statistics for original c0 image data
        orig_c0_data = c0_data[0: xsize-1,col_max+1:].copy().astype(N.float32) 
        orig_c0_data = c0_data[0: xsize-1,col_max+1:].copy().astype(N.float32) 

        self.dorgmin = orig_c0_data.min()
        self.dorgmax = orig_c0_data.max()
        self.dorgmean = orig_c0_data.mean()
        self.dorgstd = orig_c0_data.std()
        self.dorgpix = orig_c0_data.shape[0]*orig_c0_data.shape[1]
                
        low_row_p1 = 0 #  number of rows below row_threshold in PASS1 
        high_row_p1 = 0 #  number of rows above row_threshold in PASS1
        low_row_p2 = 0 #  number of rows below row_threshold in PASS2 
        high_row_p2 = 0 #  number of rows above row_threshold in PASS2

        to_sub_1_max = -wfpc2util.HUGE_VAL ; to_sub_1_min = wfpc2util.HUGE_VAL
        to_sub_2_max = -wfpc2util.HUGE_VAL ; to_sub_2_min = wfpc2util.HUGE_VAL        
  
        if (( alg_type == "PASS1")or ( alg_type == "PASS12")) : # do Pass 1  
          # PASS 1: for first correction, subtract difference between smoothed row mean from leading
          #         columns and global mean in the data from the data
           to_subtract_1 = N.zeros((ysize),dtype = N.float32)     
           for i_row in range( 0, xsize-1 ):
              to_subtract_1[ i_row ] = smoothed_row_mean[ i_row ]- glob_pyr_mean

              if (abs(to_subtract_1[ i_row ]) < row_thresh ): # make no correction 
                 to_subtract_1[ i_row ] = 0.0 
                 low_row_p1 +=1  
              else:
                 high_row_p1 +=1 
                 to_sub_1_max = max(to_sub_1_max, to_subtract_1[ i_row ])  
                 to_sub_1_min = min(to_sub_1_min, to_subtract_1[ i_row ])  

              c0_data[i_row,col_max+1:] -= to_subtract_1[ i_row ]    #  modify in place
      

        if (( alg_type == "PASS2")or ( alg_type == "PASS12")) : # do Pass 2 
          # PASS 2: for second correction, calculate row mean of unmasked pixels in the row, and 
          #         subtract difference between this row mean and the global mean   
          
           row_data_b_mean = N.zeros(ysize)
           to_subtract_2 = N.zeros((ysize),dtype = N.float32)
                  
          # calculation of the row-specific correction using PASS2 will ignore pixels that deviate from
          #   the mode of the row by more than 4 sigma. The mode will be calculated by binning all pixels in row
          #   in bins of width 1, and determining which bins has the highest frequency. 
           min_val = int(c0_data[0, col_max+1:].min()) # lower value of histogram   
           max_val = int(c0_data[0, col_max+1:].max()) # upper value of histogram 
           
           xbin = N.arange(min_val, max_val )  

           for i_row in range (0, ysize-1 ): 
              row_data_b = masked_image[ i_row, col_max+1:]
              good_pix = N.where( row_data_b > 0.0 )
              good_val =  row_data_b[ good_pix ]
              clip_row_mean[ i_row] = good_val.mean()

              bad_pix = N.where( row_data_b <= 0.0 )
              bad_val =  row_data_b[ bad_pix ]

              if (verbosity >1 ):  # print stats of good and rejected pixels for this row
                 print '' ; print ' row = ' , i_row 
                 if   good_val.size > 0 :
                    print ' good pixels: number, min, mean, max, std = ' ,  good_val.size,good_val.min(),good_val.mean(),good_val.max(),good_val.std()

              clip_row_mean[ i_row] = good_val.mean()

           # for each row, subtract difference between clipped mean and global clipped mean 
           glob_im_clip_mean = clip_row_mean[1:ysize-2].mean() # calculate global clipped mean
           for i_row in range (1, ysize-2 ):

              to_subtract_2[ i_row ] = clip_row_mean[ i_row ]- glob_im_clip_mean
            #  print '   to_subtract_2[ i_row ] = ' ,   to_subtract_2[ i_row ]

              if (abs(to_subtract_2[ i_row ]) < row_thresh ): # make no correction
                 to_subtract_2[ i_row ] = 0.0  
                 low_row_p2 +=1  
              else:
                 high_row_p2 +=1 
                 to_sub_2_max = max(to_sub_2_max, to_subtract_2[ i_row ])  
                 to_sub_2_min = min(to_sub_2_min, to_subtract_2[ i_row ])  

              c0_data[i_row,col_max+1:] -=  to_subtract_2[ i_row ] 

        # calculate and display statistics for corrected c0 image data
        corr_c0_data = c0_data[0: xsize-1,col_max+1:].astype(N.float32)   

        self.dcormin = corr_c0_data.min()
        self.dcormax = corr_c0_data.max()
        self.dcormean = corr_c0_data.mean()
        self.dcorstd = corr_c0_data.std()
        self.dcorpix = corr_c0_data.shape[0]*corr_c0_data.shape[1]

        if (verbosity >=1 ):
            print 'The following means and sigmas pertain to all (uncorrected and corrected) rows:'
            if ( alg_type == "PASS1" or  alg_type == "PASS12"):
               print '  For PASS1: the total number of uncorrected, corrected rows: ', low_row_p1,',', high_row_p1  
               if (high_row_p1 > 0 ):
                  print '  For PASS1: fraction of rows corrected: ', (high_row_p1+0.0)/( high_row_p1+ low_row_p1+0.0)
               print '  For PASS1: min, max of corrections are:', to_sub_1_min,',',to_sub_1_max  
               print '  For PASS1: mean, std of corrections are: ',to_subtract_1.mean(),',',to_subtract_1.std()  
            if ( alg_type == "PASS2" or  alg_type == "PASS12"):
               print '  For PASS2: the total number of uncorrected, corrected rows: ', low_row_p2,',', high_row_p2  
               if (high_row_p2 > 0 ):
                  print '  For PASS2: fraction of rows corrected: ', (high_row_p2+0.0)/( high_row_p2+ low_row_p2+0.0)
               print '  For PASS2: min, max of corrections are: ', to_sub_2_min,',',to_sub_2_max  
               print '  For PASS2: mean, std of corrections are: ',to_subtract_2.mean(),',',to_subtract_2.std()  

            print 'The number of sigma that the clipped image mean exceeds the clipped pyramid region mean is NSIGMA=' , nsigma  

            print 'The following statistics keywords are written to the corrected data output:'
            print '  - For all pixels in the pyramid region of the c0 data, the keywords and values for the '
            print '    mean, std, min, max, and number of pixels are :'
            print '    PORGMEAN,    PORGSTD,    PORGMIN,    PORGMAX,    PORGPIX  '
            print '   ', self.porgmean,'   ', self.porgstd,'   ', self.porgmin,'   ', self.porgmax,'   ', self.porgpix

            print '  - For the unmasked pixels in the pyramid region of the input data, the keywords and values for the '
            print '    mean, std, min, max, and number of pixels are :'
            print '    PMSKMEAN,    PMSKSTD,    PMSKMIN,    PMSKMAX,    PMSKPIX  '
            print '   ', self.pmskmean,'   ', self.pmskstd,'   ', self.pmskmin,'   ', self.pmskmax,'   ', self.pmskpix 

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

            print 'The correction type applied is CORR_ALG = ', alg_type

        outfile = file_prefix + str('_bjc_')+str(group)+str('.fits')

        update_header(self, c0_hdr, forced_type, alg_type, alg_cmt, verbosity) # add statistics keywords to header c0_hdr

        write_to_file(corr_c0_data, outfile, c0_hdr, alg_type, alg_cmt, verbosity, pyr_mean, pyr_sigma, im_mean, im_sigma) 
        if (verbosity >=1 ):
            print 'The output file written to is ',outfile

            
        # close open file handles
        if (fh_c0):
           fh_c0.close()
           
        if (verbosity >=1 ): print 'DONE '

    def print_pars(self):
        """ Print parameters.
        """
        print 'The values of the input parameters are :'
        print '  input c0 file:  ' , self.input_file
        print '  input mask file:  ' , self.input_mask
        print '  group:  ' , self.group
        print '  bias_thresh:  ' , self.bias_thresh 
        print '  row thresh:  ' , self.row_thresh 
        print '  force algorithm type:  ' , self.force_alg_type
        print '  CR rejection factor:  ' , self.n_mad


def check_py_pars(input_file, group, bias_thresh, row_thresh, force_alg_type, n_mad, input_mask):  
       """ When run under python, verify that each unspecified parameter should take the default value, and
           give the user the opportunity to change it.

       @param input_file: name of input file
       @type input_file: string
       @param group: number of group to process
       @type group: int
       @param bias_thresh: bias threshold (no correction will be performed if this is exceeded by im_mean)
       @type bias_thresh: float
       @param row_thresh: row threshold (no correction will be performed if this exceeds the calculated row correction)
       @type row_thresh: float
       @param force_alg_type: algorithm type to force
       @type force_alg_type: string
       @param n_mad: CR rejection factor   
       @type n_mad: float
       @param input_mask: name of input mask
       @type input_mask: string
       @return: group, bias_thresh, row_thresh, force_alg_type, n_mad
       @rtype:  int, float, float, string, float
       """
       
       try:
            fh_c0 = pyfits.open(input_file)
       except:
            opusutil.PrintMsg("F","ERROR - unable to open the input file  "+ str(input_file))
            sys.exit( ERROR_RETURN)
       
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
               
       if (force_alg_type == None):
            force_alg_type = wfpc2util.force_alg_type
            print ' You have not specified a value for force_alg_type; the default is:',  wfpc2util.force_alg_type
            print ' If you want to use the default, hit <enter>, otherwise type in the desired value'
            inp = raw_input('? ')
            if inp == '':
               print ' The default value of ', force_alg_type,' will be used'
            else:
               inp = inp.upper() 
               if ((inp <> "PASS1") and (inp <> "PASS2") and (inp <> "PASS12") and (inp <> "OMIT"))  :
                  print ' The value entered (',inp,') is invalid so the default will be used'
               else:
                  force_alg_type = inp

       if (n_mad == None):
            n_mad = wfpc2util.n_mad
            print ' You have not specified a value for n_mad; the default is:',  wfpc2util.n_mad
            print ' If you want to use the default, hit <enter>, otherwise type in the desired value'
            inp = raw_input('? ')
            if inp == '':
               print ' The default value of ', n_mad,' will be used'
            else:
               try:
                   n_mad = string.atof(inp)
               except:
                   print ' The value entered (',inp,') is invalid so the default will be used'


       if (input_mask <> None):
            try:
               fh_mask = pyfits.open(input_mask)
            except:
               opusutil.PrintMsg("F","ERROR - unable to open the input mask "+ str(input_mask))
               sys.exit( ERROR_RETURN)


       return group, bias_thresh, row_thresh, force_alg_type, n_mad

  

def check_cl_pars(input_file, group, bias_thresh, row_thresh, force_alg_type, n_mad, input_mask):  
       """ When run from linux coammand line, verify that each parameter is valid.

       @param input_file: name of input file
       @type input_file: string
       @param group: number of group to process
       @type group: int
       @param bias_thresh: bias threshold (no correction will be performed if this is exceeded by im_mean)
       @type bias_thresh: float
       @param row_thresh: row threshold (no correction will be performed if this exceeds the calculated row correction)
       @type row_thresh: float
       @param force_alg_type: algorithm type to force
       @type force_alg_type: string
       @param n_mad: CR rejection factor
       @type n_mad: float       
       @param input_mask: name of input mask file
       @type input_mask: string

       @return: group, bias_thresh, row_thresh, force_alg_type, n_mad
       @rtype:  int, float, float, string. float
       """
       
       try:
            fh_c0 = pyfits.open(input_file)
       except:
            opusutil.PrintMsg("F","ERROR - unable to open the input file  "+ str(input_file))
            sys.exit( ERROR_RETURN)

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

       if (force_alg_type <> None ):
          force_alg_type = force_alg_type.upper()  

          if ((force_alg_type <> "PASS1") and (force_alg_type <> "PASS2") and (force_alg_type <> "PASS12") and (force_alg_type <> "OMIT"))  :
               print ' The value entered (',force_alg_type,') is invalid. Try again.'
               sys.exit( ERROR_RETURN)

       try:  
           n_mad = string.atof(n_mad)
       except:
           print ' The CR rejection parameter value entered (',n_mad,') is invalid. Try again.'
           sys.exit( ERROR_RETURN)

       if (input_mask <> None):
            try:
               fh_mask = pyfits.open(input_mask)
            except:
               opusutil.PrintMsg("F","ERROR - unable to open the input mask "+ str(input_mask))
               sys.exit( ERROR_RETURN)

       return group, bias_thresh, row_thresh, force_alg_type, n_mad


def get_hist(a, bins):
      nn =  N.searchsorted( N.sort(a), bins)
      nn = N.concatenate([nn, [len(a)]])
      return  nn[1:]-nn[:-1]


def cr_reject( SubArray, n_mad ):
           
        """Identify and replace cosmic rays in the given subarray.

        SubArray is the input to this function; that is, the
            a subarray of the data is expected to be in this attribute.
            This array will be 2-D .
        The results are assigned to attributes:
        SubArrayCleaned is the array with cosmic rays replaced by
            a fitted line or plane.  This will be the same shape as _SubArray.
        SubCR is a tuple of indices of cosmic rays detected.
        SubArrayFit is a 1-D array containing the fit evaluated
            at each image line number.  For 1-D data this will be the
            same length as the input _SubArray, but for 2-D data this
            will be the length of shape[0] of _SubArray.
        SubSlope is the slope of the fit (SubArrayFit).
        SubIntercept is the intercept of SubArrayFit.
        SubVariance is the average of the squares of the deviations
            of the subarray level from the fit.
        SubVarSlope is the variance of the slope, normalized by
            SubVariance.
        SubVarIntercept is the variance of the the intercept,
            normalized by SubVariance.
        SubCoVar is the covariance variance of the slope and the
            intercept, normalized by SubVariance.
       """

        n_rms=[4., 4., 3.5]; n_neighbor=[2.5, 2.5, 2.5] 

        # Number of iterations in the loop to reject outliers.
        niter = len( n_rms)

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
        slope = 0.0 # forcing, as non-zero value is not expected, and for consistency between regions

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

        saveInfo( indep_var, sub_pts, fit, (ny, nx), slope, intercept, mask, cr, SubArray)
        return mask
     
# end of cr_reject()


def check_neighbors( new_cr, residual, cutoff, sub_shape):
        """Check for cosmic rays in neighboring pixels.

        @param new_cr:  1-D array of ones or zeros (1 indicates a cosmic ray)
        @type new_cr:  Int32
        @param residual:  1-D array of residuals, subarray - fit
        @type residual:  Float64
        @param cutoff:  criterion for flagging an outlier as a cosmic ray
        @type cutoff:  float
        @param sub_shape:  numbers of lines and columns in subarray
        @type sub_shape:  tuple
        @return:  new_cr, possibly with additional cosmic rays flagged
        @rtype:  Int32
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

        @param y:  array of values
        @type y:  Float64
        @param mask:  array of ones or zeros (0 indicates a good value)
        @type mask:  Int32
        @return:  median of y, ignoring masked elements
        @rtype:  float
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

        @param x:  array of independent-variable values
        @type x:  Float64
        @param y:  array of dependent-variable values
        @type y:  Float64
        @param mask:  array of ones or zeros (0 indicates a good value)
        @type mask:  Int32
        @return:  coefficients of fit, the slope and intercept
        @rtype:  tuple of two floats
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
        slope = 0.0  # forcing, as non-zero value is not expected, and for consistency between regions 

        intercept = mean_y - slope * mean_x

        return (slope, intercept)

# end of fitline()


#-------------------------------------------------------------------------------
# Compute variances and save the information in attributes
#
def saveInfo( indep_var, subarray, fit, shape,
                  slope, intercept, mask, cr, SubarrayLevel):
        """Compute variances; save info in attributes.

        This function computes:  (a) the variance, the mean of the squares
        of the deviations of the subarray level from the fit to the subarray level;
        (b) the covariance matrix (see computeVarCovar(), which is called
        to do this calculation).  Points flagged as bad by the mask (1 is
        bad) are not included in these calculations. There is a
        separate argument cr that flags only cosmic rays; this is used to
        assign a list of coordinates of cosmic rays to _SubCR.  The
        computed values will be saved in attributes; some of the arguments
        will be saved directly to attributes.

        @param indep_var:  independent variable that was used for fitting;
            this contains the image line numbers, but with repeated values
            if the subarray level data is a 2-D array 
        @type indep_var:  array of Float64

        @param subarray:  array of subarray level values (flattened, if 2-D);
            these are the (cosmic-ray cleaned) dependent variable values
            to which the line (or plane) was fit
        @type subarray:  array of Float64

        @param fit:  the 1-D array containing the fit to the subarray level;
            for 1-D data this will be the same size as indep_var or
            subarray, but for 2-D data fit will be smaller, because fit has
            just one element per image line number
        @type fit:  array of Float64

        @param shape:  numbers of lines and columns in the subarray level
            array; the number of columns will be 1 if not 2-D
        @type shape:  tuple of two elements

        @param slope:  slope of the fitted line (or plane), counts per pixel
        @type slope:  float

        @param intercept:  intercept (line number) of the fitted line
            (or plane)
        @type intercept:  float

        @param mask:  array of ones or zeros (1 is bad, either a cosmic
            ray or no data); this is a 1-D array of the same size as
            indep_var and subarray, i.e. for 2-D data it was flattened
        @type mask:  array of Int

        @param cr:  array of ones or zeros, 1 indicates a cosmic ray; this
            is a 1-D array of the same size as mask, i.e. flattened if 2-D;
            cr is None means no cosmic-ray rejection was done
        @type cr:  array of Int, or None
        """

        if len (shape) < 2:
            nx = 1
        else:
            nx = shape[1]

        SubarrayLevelCleaned = N.reshape( subarray, shape)
        SubarrayLevelFit = fit

        if cr is None:
            SubCR = None
        else:
            cr_2d = N.reshape( cr, shape)
            locations = N.where( cr_2d > 0)
            del cr_2d
            sub0 = SubarrayLevel.ravel().astype( N.float64)

            norm = 1.
                
            values = []
            for k in N.arange( len( locations[0])):
                x = locations[1][k]
                y = locations[0][k]
                i = x + y * shape[1]
                values.append( (sub0[i] - fit[y]) * norm)
            del sub0
            # a tuple of three lists, containing row, column, value
            SubCR = (locations[0], locations[1], values)

        residual = subarray - fit.repeat( nx)
        labels = N.where( mask == 0, 1, 0)
        variance = ndimage.variance( residual, labels=labels)

        (var_slope, var_intercept, covar) = \
                computeVarCovar( indep_var, labels)

# Delete these later if not needed; if they are needed, pass back or make attributes 
        SubSlope = slope
        SubIntercept = intercept
        SubVariance = variance
        SubVarSlope = var_slope
        SubVarIntercept = var_intercept
        SubCoVar = covar

# end of  saveInfo()


def computeVarCovar( x, labels):
        """Compute the variances and covariance of the fit.

        These values are normalized, i.e. multiply by the variance
        of the deviations of the subarray level from the fit to get the
        actual variances of the slope and intercept and their covariance.

        Note that argument for the mask is inverted from other functions
        in this file, i.e. labels instead of mask.

        @param x:  array of independent-variable values
        @type x:  Float64
        @param labels:  array of ones or zeros (1 indicates a good value)
        @type labels:  Int32
        @return:  elements from the normalized covariance matrix, i.e. the
            variance of the slope, the variance of the intercept, and the
            covariance of the slope and intercept
        @rtype:  tuple of floats
        """

        n = labels.astype(N.float64).sum()
        mean_x = ndimage.mean( x, labels=labels)
        dx = x - mean_x
        temp = dx**2
        # sum of (x[i] - mean_x)**2
        sum_dx2 = ndimage.sum( temp, labels=labels)

        var_slope = 1. / sum_dx2
        var_intercept = 1. / n + mean_x**2 / sum_dx2
        covar = -mean_x / sum_dx2

        return (var_slope, var_intercept, covar)

# end of computeVarCovar()


def update_header(self, hdr, forced_type, alg_type, alg_cmt, verbosity):   
    """ update header from input c0 file with specified header, and updated data

    @param hdr: hdr
    @type hdr: Pyfits header object
    @param forced_type: specifies if an algorithm type is forced (yes/no)
    @type forced_type: string
    @param alg_type: specifies which algorithm type was used
    @type alg_type: string
    @param alg_cmt: description of algorithm type used
    @type alg_cmt: string
    @param verbosity: verbosity level (0 for quiet, 1 verbose, 2 very verbose)
    @type verbosity: string

    """

    hdr.update(key='CORR_ALG', value=alg_type, comment=alg_cmt ) #  denoting which algorithm was used
    hdr.update(key='BIASTHRE', value=self.bias_thresh, comment="bias threshold" ) 
    hdr.update(key='ROWTHRES', value=self.row_thresh, comment="row threshold" ) 
    hdr.update(key='FORCETYP', value=forced_type, comment="correction type applied was forced (yes/no)" ) 

    hdr.update(key='PORGPIX', value=self.porgpix, comment="total number of pixels in pyramid region" ) 
    hdr.update(key='PORGMEAN', value=self.porgmean, comment="mean of all pixels in pyramid region" ) 
    hdr.update(key='PORGSTD', value=self.porgstd, comment="sigma of all pixels in pyramid region" ) 
    hdr.update(key='PORGMIN', value=self.porgmin, comment="min of all pixels in pyramid region" ) 
    hdr.update(key='PORGMAX', value=self.porgmax, comment="max of all pixels in pyramid region" )

    hdr.update(key='PMSKPIX', value=self.pmskpix, comment="number of unmasked pixels in pyramid region" ) 
    hdr.update(key='PMSKMEAN', value=self.pmskmean, comment="mean of unmasked pixels in pyramid region" ) 
    hdr.update(key='PMSKSTD', value=self.pmskstd, comment="sigma of unmasked pixels in pyramid region" ) 
    hdr.update(key='PMSKMIN', value=self.pmskmin, comment="min of unmasked pixels in pyramid region" ) 
    hdr.update(key='PMSKMAX', value=self.pmskmax, comment="max of unmasked pixels in pyramid region" ) 

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
    hdr.update(key='CRNMAD', value=self.n_mad, comment="CR rejection factor" )

# write  mask
def write_mask(data, filename, verbosity):

    fimg = pyfits.HDUList()
    fimghdu = pyfits.PrimaryHDU()
    fimghdu.data = data
    fimg.append(fimghdu)
    fimg.writeto(filename)


def write_to_file(data, filename, hdr, alg_type, alg_cmt, verbosity, pyr_mean, pyr_sigma, im_mean, im_sigma):

    fimg = pyfits.HDUList()
    hdr.update(key='CORR_ALG', value=alg_type, comment=alg_cmt ) #  denoting which algorithm was used
    hdr.update(key='PYRMEAN', value=pyr_mean, comment="clipped mean of pyramid region" )
    hdr.update(key='PYRSIGMA', value=pyr_sigma, comment="clipped sigma of pyramid region" )
    hdr.update(key='IMMEAN', value=im_mean, comment="clipped mean of image region" )
    hdr.update(key='IMSIGMA', value=im_sigma, comment="clipped sigma of image region" )
    fimghdu = pyfits.PrimaryHDU( header = hdr)
    fimghdu.data = data
    fimg.append(fimghdu)
    fimg.writeto(filename)
    if (verbosity >=1 ): print 'Wrote updated data to: ',filename


if __name__ =="__main__":
    """Get input file and other arguments, and call Wfpc2destreak

    The command-line options are:
        -q (quiet)
        -v (very verbose)

    @param cmdline: command-line arguments
    @type cmdline: list of strings
    """

    usage = "usage:  %prog [options] inputfile"
    parser = OptionParser( usage)

    if ( sys.argv[1] ): filename = sys.argv[1]   

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
    parser.add_option( "-f", "--force_alg_type", dest = "force_alg_type",default = wfpc2util.force_alg_type,
            help = "algorithm type to force.")         
    parser.add_option( "-n", "--n_mad", dest = "n_mad",default = wfpc2util.n_mad, 
            help = "CR rejection factor.")         
    parser.add_option( "-m", "--input_mask", dest = "input_mask",default = wfpc2util.input_mask, 
            help = "input mask file name.")         

    (options, args) = parser.parse_args()
  
    wfpc2util.setVerbosity( options.verbosity)  
    verbosity = options.verbosity
 
    wfpc2util.setGroup(options.group )
    if options.group!=None: group = options.group

    wfpc2util.setBias_thresh(options.bias_thresh )
    if options.bias_thresh!=None: bias_thresh = options.bias_thresh

    wfpc2util.setRow_thresh(options.row_thresh )
    if options.row_thresh!=None: row_thresh = options.row_thresh

    wfpc2util.setForce_alg_type(options.force_alg_type )
    force_alg_type  = options.force_alg_type  

    wfpc2util.setN_mad(options.n_mad )  
    if options.n_mad!=None: n_mad = options.n_mad

    wfpc2util.setInput_mask(options.input_mask )  
    input_mask = options.input_mask
     
    try:
       wfpc2_d = Wfpc2destreak( filename, input_mask=input_mask, group=group, verbosity=verbosity, bias_thresh=bias_thresh, \
                                row_thresh=row_thresh, force_alg_type=force_alg_type, n_mad=n_mad)
   
       if (verbosity >=1 ):
            print 'The version of this routine is: ',__version__
            wfpc2_d.print_pars()
       Wfpc2destreak.destreak(wfpc2_d)

       del wfpc2_d

    except Exception, errmess:
       opusutil.PrintMsg("F","FATAL ERROR "+ str(errmess))
       sys.exit( ERROR_RETURN)

