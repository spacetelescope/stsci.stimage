#! /usr/bin/env python 
#
# Authors: Dave Grumm
# Program: wfpc2destreak.py
# Purpose: routine to remove streaks due to bias from specified chip of wfpc2 data
# History: 07/10/07 - first version
#          08/10/07 - use overscan in the specified chipd in the d0f file, and apply it to the
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
#
# Outline:
#
# 1. In the pyramid region, eliminate the CRs, then calculate the mean (pyr_mean) and sigma (pyr_sigma)
# 2. In the 'interior' image region (starting to the right of the pyramid region), use iterative sigma
#    clipping to calculate the mean (im_mean) and sigma (im_sigma)
# 3. Calculate how high above the pyr values the image values are by:
#        im_mean = pyr_mean + NSIGMA*pyr_sigma
# 4. Determine which correction algorithm to use and apply it:
#   A.  If NSIGMA < = 1.0 apply no correction
#   B.  If NSIGMA >1.0 and NSIGMA < =11.0 and im_sigma <= 2.0 apply Pass1 & 2
#   C.  If NSIGMA >1.0 and NSIGMA < =11.0 and im_sigma > 2.0 apply no correction
#   D.  ELSE apply Pass1 only
#
# ... where Pass1 is done as follows:
#     - In the leading columns (2->group-specific value ), cosmic rays are identified and
#         masked in the d0f data
#     - For each row within these columns, the mean is calculated of the unmasked
#         pixels; for all rows the global mean is calculated
#     - For each row, the smoothed mean is calculated over 3 rows
#     - For each row in the image, the difference between the smoothed mean and the global
#         is subtracted from the c0f data
#
# ... and Pass2 is done as follows:
#     - Over the entire d0f image, iterative sigma-clipping is performed to mask
#         cosmic rays and bright sources         
#     - For each row, the mean is calculated of the unmasked pixels; for all
#         rows the global mean is calculated
#     - For each row, the difference between the mean and the global
#         is subtracted from the c0f data
#
# 5. The modified c0f data is written to a new file; e.g. "u8zq0104m_bjc_4.fits" for group 4
#     
# Usage :
#     For a dataset with multiple groups, to process group 4: hal> ./wfpc2destreak.py -q "u8zq0104m_d0f.fits" 4
#     For a dataset with a single group: hal> ./wfpc2destreak.py -v "u9wp0401m_bjc_0.fits" 0
#
###########################################################################

import pyfits
import numpy as N
from convolve import boxcar 
import pytools
from pytools import numerixenv, readgeis
from optparse import OptionParser
import ndimage
import cwdutil, opusutil, sys

__version__ = "1.0 (2007 Nov 09)"

NUM_SIG = 2.5  # number of sigma to use in sigma clipping
TOT_ITER = 4   # maximum number of iterations for sigma clipping
ERROR_RETURN = 2

class Wfpc2destreak:
    """ Calculate magnitude of and remove streaks from specified group of wfpc2 data.

    example: 
       wfpc2_d = Wfpc2destreak( filename, group, verbosity)
       Wfpc2destreak.destreak(wfpc2_d)
       
    """

    def __init__( self, input_file, group, verbosity):
        """constructor

        @param input_file: name of the file to be processed
        @type input_file: string
        @param group: number of group to process
        @type group: int
        @param verbosity: verbosity level (0 for quiet, 1 verbose, 2 very verbose)
        @type verbosity: string
        """

        self.input_file = input_file
        self.group = group
        self.verbosity = verbosity

    def destreak( self ):
        
        input_file = self.input_file
        group = int(self.group)
        verbosity = self.verbosity

        # read data from d0f file:
        fh_d0f = pyfits.open(input_file)
        data_cube = fh_d0f[0].data

        if ( group == 0 ): # for single group image case
            d0f_data = data_cube
        else:
            d0f_data = data_cube[group-1] # 0-based

        xsize = fh_d0f[0].header.get( "NAXIS1"); ysize = fh_d0f[0].header.get( "NAXIS2")

        # read data from c0f file:   
        c0f_prefix = input_file.split('_')[0]    
        c0f_file = c0f_prefix + str("_c0f.fits")
        fh_c0f = pyfits.open(c0f_file) 

       # read header from appropriate group in c0h file and make cardlist  
        c0h_file = c0f_prefix + str(".c0h")
        c0_group_hdr = (readgeis.readgeis( c0h_file ))[group-1].header     
      
        group_hdr_c =  c0_group_hdr.ascardlist()
        
        if (verbosity >=1 ): print 'Getting header information from c0h_file =', c0h_file 

       # read primary header in c0h file and make cardlist 
        c0_pr_hdr = (readgeis.readgeis( c0h_file ))[0].header
        p_hdr_c =  c0_pr_hdr.ascardlist()

       # combine group cardlist and primary cardlist starting with WFPC2 keywords, and make header from it 
        new_hdr_c = group_hdr_c +  p_hdr_c      # waiting for Matt's input on how to revise this
        new_h = pyfits.Header (cards = new_hdr_c ) 

        if (verbosity >=1 ):print 'Will update data in c0f_file: ' ,c0f_file

        c0f_data_cube = fh_c0f[0].data 

        if ( group == 0 ): # for single group image case
           c0f_data = c0f_data_cube
        else:
           c0f_data = c0f_data_cube[group-1] # 0-based

        clip_row_mean = N.zeros(ysize); good_row_mean = N.zeros(ysize)

    #   read d0f leading columns (between col_min and col_max) and reject Cosmic Rays
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

        BlackLevel = d0f_data[:,col_min:col_max]

        mask = black_cr_reject(BlackLevel)    
        mask_2d = N.resize( mask, [BlackLevel.shape[0], BlackLevel.shape[1]])    
        masked_BlackLevel = BlackLevel * (1-mask_2d)  # gives 0 where there are Cosmic rsys

    # calculate global mean from all unmasked pixels in d0f file
        d0f_data_pix = N.where( masked_BlackLevel > 0.0)
        all_good_data = masked_BlackLevel[ d0f_data_pix ]
        glob_mean = all_good_data.mean()
        pyr_mean = all_good_data.mean()  # clipped pyramid mean of d0f data 
        pyr_sigma = all_good_data.std()  # clipped pyramid sigma of d0f data

    # loop over all rows and calculate mean of unmasked pixels for each row
        for i_row in range( 0, ysize-1 ):
           row_data_a = masked_BlackLevel[i_row,:].astype( N.float64) 

           if (row_data_a.std() <> 0.0) or (row_data_a.mean() <> 0.0   ):# for cases in which all values are 0.0
             row_data_pix = N.where( row_data_a > 0.0)
             good_row_data = row_data_a[ row_data_pix ]
             good_row_mean[ i_row ] = good_row_data.mean()
           else:
             good_row_mean[ i_row ] = 0.0 

        smoothed_row_mean = boxcar( good_row_mean,(3,))  # smooth row mean over 3 rows

        # do iterative sigma clipping on all 'interior' d0f pixels
        int_d0f_data = d0f_data[ ix_min:xsize-1, iy_min:ysize-1 ] 
        for ii_iter in range(4):
              if ( ii_iter == 0):
                 int_d0f_mean1 = int_d0f_data.mean()
                 int_d0f_std1 = int_d0f_data.std()
              else:
                 int_d0f_mean1 = good_val.mean()
                 int_d0f_std1 = good_val.std()

              good_pix = N.where((( int_d0f_data < (int_d0f_mean1 + (NUM_SIG*int_d0f_std1))) &
                                  (( int_d0f_data > (int_d0f_mean1 - (NUM_SIG*int_d0f_std1))))))
              good_val =  int_d0f_data[good_pix]

              good_val = good_val.astype(N.float32)
              if ( good_val.size == 0 ):
                 break

        if (len(good_val) < 1 ):  # no good values in image, so will abort this run, making no correction
              opusutil.PrintMsg("F","ERROR - unable to calculate statistics on image, so will perform no correction")
              sys.exit( ERROR_RETURN)
        
        im_mean = good_val.mean()
        im_sigma =good_val.std()

        if (verbosity >1 ):
           print 'For the pyramid region: clipped mean = ' ,  pyr_mean, ', clipped_sigma = ' ,  pyr_sigma
           print 'For image pixels: clipped mean =', im_mean,' , clipped image sigma= ', im_sigma

    # Calculate the number of pyramid sigma above the clipped pyramid mean for the clipped image mean 
        nsigma = (im_mean - pyr_mean)/pyr_sigma

        if (verbosity >1 ):print 'Determining which algorithm to use  ..'
        if ( nsigma < 1.0 ): # Case A
           if (verbosity >=1 ): print '  nsigma = ' , nsigma, ' < 1.0, so will apply no correction'
           alg_type = "None"
           alg_cmt = "No correction will be applied"
        elif (nsigma >= 1.0 and nsigma <= 11.0 and im_sigma <= 2.0): # Case B
           if (verbosity >1 ):
               print '  nsigma = ', nsigma , '( between 1.0 and 11.), and im_sigma = ', im_sigma ,' <=2.0 so will apply both PASS 1 & 2'
               print '  Will use columns 2 through ', col_max,' in pyramid region to calculate Pass 1 corrections.'
           alg_type = "PASS12"
           alg_cmt = "Pass 1 and 2 corrections applied"
        elif (nsigma >= 1.0 and nsigma <= 11.0 and im_sigma > 2.0): # Case C
           if (verbosity >1 ): print '  nsigma = ', nsigma , '( between 1.0 and 11.), and im_sigma = ', im_sigma , ' >2.0 so will apply no correction'
           alg_type = "None"
           alg_cmt = "No correction will be applied"
        else: # Case D
           if (verbosity >1 ):             
               print '  nsigma = ', nsigma ,' and im_sigma = ', im_sigma
               print '  Neither   nsigma < 1.0 or  (nsigma >= 1.0 and nsigma <= 11.0 and im_sigma <= 2.0) so '
               print '  will apply Pass 1 only'
               print 'Will use columns 2 through ', col_max,' in pyramid region to calculate Pass 1 corrections.'
           alg_type = "PASS1"
           alg_cmt = "Pass 1 correction applied"

        if (verbosity >1 ):
           print 'Will use the correction algorithm: ', alg_type 
            
        new_c0f_data = c0f_data.copy().astype(N.float32)  # will be updated array for c0f data

        if ( alg_type == "PASS1") : # do Pass 1 only; Case D
          # PASS 1: for first correction, subtract difference between smoothed row mean from leading
          #         columns and global mean in the d0f data from the c0f data, updating data in c0f file   
           to_subtract_1 = N.zeros((ysize),dtype = N.float32)     
           for i_row in range( 0, xsize-1 ):
              to_subtract_1[ i_row ] = smoothed_row_mean[ i_row ]- glob_mean
              new_c0f_data[i_row,col_max+1:] = c0f_data[i_row,col_max+1:] - to_subtract_1[ i_row ]  
        elif ( alg_type == "PASS12"): # do both Pass 1 & 2; Case B
          # PASS 1: for first correction, subtract difference between smoothed row mean from leading
          #         columns and global mean in the d0f data from the c0f data, updating data in c0f file   
           to_subtract_1 = N.zeros((ysize),dtype = N.float32)     
           for i_row in range( 0, xsize-1 ):
              to_subtract_1[ i_row ] = smoothed_row_mean[ i_row ]- glob_mean
              new_c0f_data[i_row,col_max+1:] = c0f_data[i_row,col_max+1:] - to_subtract_1[ i_row ]  
          # PASS 2: for second correction, calculate row mean over entire c0f image using iterative sigma clipping and
          #         subtract difference between this row mean and the global mean    
           row_data_b_mean = N.zeros(ysize)
           to_subtract_2 = N.zeros((ysize),dtype = N.float32)

           for i_row in range (0, ysize-1 ):
              row_data_b = new_c0f_data[i_row, col_max+1:]   
              mean1 = row_data_b.mean()
              row_data_b_mean[ i_row ] = row_data_b.mean()

            #  for current row, do sigma clipping until either 4 iterations are perfomed or clipping removes all pixels
              for ii_iter in range(4):
                 if ( ii_iter == 0):
                      mean1 = row_data_b.mean()
                      std1 = row_data_b.std()
                 else:
                      mean1 = good_val.mean()
                      std1 = good_val.std()
                 good_pix = N.where(((row_data_b < (mean1 + (NUM_SIG*std1))) & ((row_data_b > (mean1 - (NUM_SIG*std1))))))
                 good_val = row_data_b[good_pix]
                 good_val = good_val.astype(N.float32)
                 if ( good_val.size == 0 ):
                      break

                 clip_row_mean[ i_row] = good_val.mean() 

            # for each row, subtract difference between clipped mean and global clipped mean 
           glob_clip_mean = clip_row_mean[1:ysize-2].mean() # calculate global clipped mean
           for i_row in range (1, ysize-2 ):
              to_subtract_2[ i_row ] = clip_row_mean[ i_row ]- glob_clip_mean
              new_c0f_data[i_row,col_max+1:] -=  to_subtract_2[ i_row ]
        else: # will apply no correction; Cases A and C
           new_c0f_data = c0f_data.copy().astype(N.float32) 

        file_prefix = input_file.split('.')[0]
        outfile = c0f_prefix + str('_bjc_')+str(group)+str('.fits')

        write_to_file(new_c0f_data, outfile, new_h, alg_type, alg_cmt, verbosity, pyr_mean, pyr_sigma, im_mean, im_sigma)

        # close open file handles
        if (fh_d0f):
           fh_d0f.close()
        if (fh_c0f):
           fh_c0f.close()
        if (verbosity >=1 ): print 'DONE '

    def print_pars(self):
        """ Print parameters.
        """
        print 'The input parameters are :'
        print '  input d0f file:  ' , self.input_file
        print '  group:  ' , self.group


# end of class Wfpc2destreak

def black_cr_reject(  BlackLevel):
           
        """Identify and replace cosmic rays in the black level.

        @param n_mad:  for initial rejection, reject outliers of more than
            n_mad * (median of absolute values of deviations)
        @type n_mad:  float
        @param n_rms:  list of n-sigma factors, i.e. a residual greater
            than n_rms[i] for the ith iteration should be flagged as a
            cosmic ray; the length of the list is the number of iterations.
            Specify n_rms=[] to turn off cosmic-ray rejection.
        @type n_rms:  list
        @param n_neighbor:  like n_rms, but for neighboring pixels in 2-D
        @type n_neighbor:  list

        BlackLevel is the input to this function; that is, the
            raw black level data are expected to be in this attribute.
            This array will be 1-D or 2-D .
        The results are assigned to attributes:
        BlackLevelCleaned is the array with cosmic rays replaced by
            a fitted line or plane.  This will be the same shape as _BlackLevel.
        BlackCR is a tuple of indices of cosmic rays detected.
        BlackLevelFit is a 1-D array containing the fit evaluated
            at each image line number.  For 1-D data this will be the
            same length as the input _BlackLevel, but for 2-D data this
            will be the length of shape[0] of _BlackLevel.
        BlackSlope is the slope of the fit (BlackLevelFit).
        BlackIntercept is the intercept of BlackLevelFit.
        BlackVariance is the average of the squares of the deviations
            of the black level from the fit.
        BlackVarSlope is the variance of the slope, normalized by
            BlackVariance.
        BlackVarIntercept is the variance of the the intercept,
            normalized by BlackVariance.
        BlackCoVar is the covariance variance of the slope and the
            intercept, normalized by BlackVariance.
       """

        n_mad=15.; n_rms=[4., 4., 3.5]; n_neighbor=[2.5, 2.5, 2.5]

        # Number of iterations in the loop to reject outliers.
        niter = len( n_rms)

        black_shape = BlackLevel.shape
        ny = black_shape[0]; nx = black_shape[1]

        linenum = N.arange( ny, dtype=N.float64)         # image line numbers
        # Independent variable (image line numbers, repeated) for fitting.
        indep_var = linenum.repeat( nx)  # 2-D 

        # Note that black0 is 1-D
        black0 = BlackLevel.ravel().astype( N.float64)
        # no_value flags points where no value has been assigned to the
        # black level.
        no_value = N.where( black0 <= 0., 1, 0)
        n = len( black0)

        # First find extreme outliers, so they will not be included when
        # we do a least-squares fit to the data.

        line0 = float( ny//2 - 1) / 2.          # middle line, lower half
        line1 = float( ny//2 + ny - 1) / 2.     # middle line, upper half
        y0 = median( black0[0:n//2], no_value[0:n//2])
        y1 = median( black0[n//2:n], no_value[n//2:n])

        # fit is a straight line passing through (line0,y0) & (line1,y1).
        slope = (y1 - y0) / (line1 - line0)
        fit = slope * linenum + (y0 - slope * line0)
        residual = black0 - fit.repeat( nx)

        # MAD is the median of the absolute values of deviations; for a
        # normal distribution the MAD is about 2/3 of the standard deviation.
        mad = median( N.absolute( residual), no_value)

        # First find very bright cosmic rays (large, positive outliers).
        cr = N.where( residual > (n_mad * mad), 1, 0)
        # Replace points identified as cosmic rays by the value of the
        # fit at that point; this is not really necessary because we'll
        # ignore currently identified cosmic rays when doing the fit.
        black = N.where( cr, fit.repeat( nx), black0)

        mask = N.logical_or( no_value, cr)

        (slope, intercept) = fitline( indep_var, black, mask)
        # Note that fit has length ny, so will be shorter than black for
        # by the factor nx.

        fit = slope * linenum + intercept

        for iter in range( niter):
            residual = black - fit.repeat( nx)
            labels = N.where( mask == 0, 1, 0)
            rms = ndimage.standard_deviation( residual, labels=labels)
            rms = max (rms, 1.)
            new_cr = N.where( residual > (n_rms[iter] * rms), 1, 0)

            if ndimage.sum( new_cr) > 0:

                new_cr = check_neighbors( new_cr,
                             residual, n_neighbor[iter] * rms, black_shape)
            cr = N.logical_or( cr, new_cr)

            mask = N.logical_or( no_value, cr)

            (slope, intercept) = fitline( indep_var, black, mask)

            fit = slope * linenum + intercept
            black = N.where( cr, fit.repeat( nx), black)

        saveInfo( indep_var, black, fit, (ny, nx), slope, intercept, mask, cr, BlackLevel)
        return mask
     
# end of black_cr_reject()


def check_neighbors( new_cr, residual, cutoff, black_shape):
	"""Check for cosmic rays in neighboring pixels.

	@param new_cr:  1-D array of ones or zeros (1 indicates a cosmic ray)
	@type new_cr:  Int32
	@param residual:  1-D array of residuals, black - fit
	@type residual:  Float64
	@param cutoff:  criterion for flagging an outlier as a cosmic ray
	@type cutoff:  float
	@param black_shape:  numbers of lines and columns in black
	@type black_shape:  tuple
	@return:  new_cr, possibly with additional cosmic rays flagged
	@rtype:  Int32
	"""

	(ny, nx) = black_shape
	new_cr_2D = N.reshape( new_cr, black_shape)
	residual_2D = N.reshape( residual, black_shape)

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
            raise ValueError, "Error fitting a line to black level array."
        slope = num / denom
        intercept = mean_y - slope * mean_x

        return (slope, intercept)

# end of fitline()


#-------------------------------------------------------------------------------
# Compute variances and save the information in attributes
#
def saveInfo( indep_var, black, fit, shape,
                  slope, intercept, mask, cr, BlackLevel):
        """Compute variances; save info in attributes.

        This function computes:  (a) the variance, the mean of the squares
        of the deviations of the black level from the fit to the black level;
        (b) the covariance matrix (see computeVarCovar(), which is called
        to do this calculation).  Points flagged as bad by the mask (1 is
        bad) are not included in these calculations. There is a
        separate argument cr that flags only cosmic rays; this is used to
        assign a list of coordinates of cosmic rays to _BlackCR.  The
        computed values will be saved in attributes; some of the arguments
        will be saved directly to attributes.

        @param indep_var:  independent variable that was used for fitting;
            this contains the image line numbers, but with repeated values
            if the black level data is a 2-D array 
        @type indep_var:  array of Float64

        @param black:  array of black level values (flattened, if 2-D);
            these are the (cosmic-ray cleaned) dependent variable values
            to which the line (or plane) was fit
        @type black:  array of Float64

        @param fit:  the 1-D array containing the fit to the black level;
            for 1-D data this will be the same size as indep_var or
            black, but for 2-D data fit will be smaller, because fit has
            just one element per image line number
        @type fit:  array of Float64

        @param shape:  numbers of lines and columns in the black level
            array; the number of columns will be 1 if not 2-D
        @type shape:  tuple of two elements

        @param slope:  slope of the fitted line (or plane), counts per pixel
        @type slope:  float

        @param intercept:  intercept (line number) of the fitted line
            (or plane)
        @type intercept:  float

        @param mask:  array of ones or zeros (1 is bad, either a cosmic
            ray or no data); this is a 1-D array of the same size as
            indep_var and black, i.e. for 2-D data it was flattened
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

        BlackLevelCleaned = N.reshape( black, shape)
        BlackLevelFit = fit

        if cr is None:
            BlackCR = None
        else:
            cr_2d = N.reshape( cr, shape)
            locations = N.where( cr_2d > 0)
            del cr_2d
            black0 = BlackLevel.ravel().astype( N.float64)

            norm = 1.
                
            values = []
            for k in N.arange( len( locations[0])):
                x = locations[1][k]
                y = locations[0][k]
                i = x + y * shape[1]
                values.append( (black0[i] - fit[y]) * norm)
            del black0
            # a tuple of three lists, containing row, column, value
            BlackCR = (locations[0], locations[1], values)

        residual = black - fit.repeat( nx)
        labels = N.where( mask == 0, 1, 0)
        variance = ndimage.variance( residual, labels=labels)

        (var_slope, var_intercept, covar) = \
                computeVarCovar( indep_var, labels)

# Delete these later if not needed; if they are needed, pass back or make attributes 
        BlackSlope = slope
        BlackIntercept = intercept
        BlackVariance = variance
        BlackVarSlope = var_slope
        BlackVarIntercept = var_intercept
        BlackCoVar = covar

# end of  saveInfo()


def computeVarCovar( x, labels):
        """Compute the variances and covariance of the fit.

        These values are normalized, i.e. multiply by the variance
        of the deviations of the black level from the fit to get the
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

# write to specified filename with specified header
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


def main( cmdline):
    """Get input file and other arguments, and call Wfpc2destreak

    The command-line options are:
        -q (quiet)
        -v (very verbose)

    @param cmdline: command-line arguments
    @type cmdline: list of strings
    """

    usage = "usage:  %prog [options] inputfile"
    parser = OptionParser( usage)

    parser.set_defaults( verbosity = cwdutil.VERBOSE)

    parser.add_option( "-q", "--quiet", action = "store_const",
            const = cwdutil.QUIET, dest = "verbosity",
            help = "quiet, print nothing")
    parser.add_option( "-v", "--verbose", action="store_const",
            const = cwdutil.VERY_VERBOSE, dest="verbosity",
            help="very verbose, print lots of information")

    (options, args) = parser.parse_args()
    cwdutil.setVerbosity( options.verbosity)  
    verbosity = options.verbosity

    if ( args[0] ):
       filename = args[0]
    if ( args[1] ):
       group = args[1]

    try:            
       wfpc2_d = Wfpc2destreak( filename, group, verbosity)
       if (verbosity >=1 ):
            wfpc2_d.print_pars()
       Wfpc2destreak.destreak(wfpc2_d)

       del wfpc2_d

    except Exception, errmess:
       opusutil.PrintMsg("F","FATAL ERROR "+ str(errmess))
       sys.exit( ERROR_RETURN)

if __name__ == "__main__":

    # Process
    main( sys.argv[1:])


