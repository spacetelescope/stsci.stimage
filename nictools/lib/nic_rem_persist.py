#! /usr/bin/env python
#
# Author : Dave Grumm (based on IDL routine by Adam Riess)
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
#

import pyfits
import numpy as N
import sys, time
import opusutil
import persutil
import string 
import ndimage    # for median_filter

__version__ = "1.3 (2008 Aug 26)"


ERROR_RETURN = -2
CLIP = 4  # number of std to use in sigma clipping  

class NicRemPersist:
    """ Remove bright earth persistence persistence from NICMOS data (on which pedsub has been run) using a medianed persistence model. 

    pyraf example:
    --> nrp = nic_rem_persist.NicRemPersist('n9r7b2bjq_ped.fits', persist_model = 'persistring.fits', persist_mask = 'persist_mask.fits',
                  used_lo = .3, persist_lo = 0.8)
    --> nrp.persist()

    linux command line example: 
       hal> ./nic_rem_persist.py 'n9r7b2bjq_ped.fits' -d 'same_as_persistring.fits' -m 'same_as_persist_mask.fits' -u 0.4 -p 1.3 -v

    """

    def __init__( self, input_file, verbosity=1 ,persist_lo=None, used_lo=None, persist_model=None, persist_mask=None, run_stdout=None): 
        """constructor

        @param input_file: name of ineput file
        @type input_file:  string
        @param persist_lo: minimum allowed value of the persistence  
        @type persist_lo:  Float32
        @param used_lo: minimum allowed value of the fraction of pixels used  
        @type used_lo:  Float32
        @param persist_model: filename containing persistence frame (ring median of)
        @type persist_model:  string
        @param persist_mask: filename containing pixel mask 
        @type persist_mask:  string
        @param run_stdout: open trailer file (pipeline use only) 
        @type run_stdout: file handle 

        """

        if (run_stdout != None): # set standard output to trailer file
            self.orig_stdout = sys.stdout # save original stdout
            sys.stdout = run_stdout  # set stdout to open trailer file

        if (verbosity > 0 ):
            current_time = time.asctime()
            print '=== BEP', __version__,' starting at ', current_time  

        [ options, args, parser ] = persutil.getOptions()

        persutil.setVerbosity( options.verbosity)  
        verbosity = options.verbosity 

        # get parameter values if not specified on command line
        if (persist_lo == None ):
            persist_lo = persutil.getPersist_lo( input_file, options) 

        if (used_lo == None ):
            used_lo = persutil.getUsed_lo( input_file, options)

        if (persist_model == None ):
            persist_model = persutil.getPersist_model( input_file, options)

        if (persist_mask == None ):
            persist_mask = persutil.getPersist_mask( input_file, options)

        # do some parameter type checking        
        if ( __name__ == 'nic_rem_persist'):  
              [ persist_lo, used_lo, persist_model, persist_mask ] = check_py_pars(input_file, persist_lo, used_lo, persist_model, persist_mask)
        else:
              [ persist_lo, used_lo] = check_cl_pars(input_file, persist_lo, used_lo)

        [ persist_lo, used_lo] = check_cl_pars(input_file, persist_lo, used_lo)
        
        self.input_file = input_file
        self.persist_lo = persist_lo
        self.used_lo = used_lo
        self.persist_model = persist_model
        self.persist_mask = persist_mask
        self.verbosity = verbosity
        self.run_stdout = run_stdout   

 
    def persist( self ):
        """ remove persistence due to the full bright Earth.    

        """

        input_file = self.input_file
        persist_lo = self.persist_lo 
        used_lo = self.used_lo 
        persist_model = self.persist_model 
        persist_mask = self.persist_mask 
        verbosity = self.verbosity

       # get header to read PMODFILE and PMSKFILE
        try:
           fh_infile = pyfits.open( input_file, mode='update')
        except:
           opusutil.PrintMsg("F","ERROR "+ str('Unable to open input file') + str(input_file))
           sys.exit( ERROR_RETURN)

        try:  # read the image for the persistence model
           fh_per_median = pyfits.open( self.persist_model )
        except:    
           opusutil.PrintMsg("F"," ERROR "+ str('Unable to open medianed persistence model file') + str(persist_model))
           sys.exit( ERROR_RETURN)
        ps = fh_per_median[0].data 


        try:  # read the image for the mask
           fh_per_mask = pyfits.open( self.persist_mask )
        except:    
           opusutil.PrintMsg("F"," ERROR "+ str('Unable to open the mask file') + str(persist_mask))
           sys.exit( ERROR_RETURN)           
        per_mask = fh_per_mask[0].data  # persistence mask data (0 is good, non-0 is bad)

        im = fh_infile[1].data    #  load input image data

        med_mask = im *0  # for size; needed for median call

        dq_data = fh_infile[3].data  # read the dq array data (0 is good, non-0 is bad)
        
        dq_plus_per_mask = dq_data + per_mask # combined mask

        cir_mask = make_footprint( 3, 6 ) # make annular mask for ring median with inner,outer radii = 3,6

        imrm = im * 0.0
        ndimage.median_filter(im, footprint = cir_mask, output = imrm)  #  generate ring median image

        src = im - imrm  # this image is zero except for sources
          
        md, std = iterstatc( CLIP, src ) # returns clipped mean and standard deviation

        level = median(im, med_mask)

    #   Sigma clip sources, and reject bright regions >1.5*median and un-masked.
    #   Only use pixels that are:
    #      within 'CLIP' standard deviations, and
    #      less than 150% of the median, and
    #      not set in combined mask
        gd = ((src < md+CLIP*std) & (src > md-CLIP*std) & (im < 1.5*level) & (dq_plus_per_mask == 0)) 

        num_gd = N.add.reduce( N.add.reduce( gd )) 
        used =  float(num_gd)/float( src.shape[0]* src.shape[1]) 
        self.used = used

    #     set up the matrices for a linear system of equations
        y = im[gd]   #this is the image
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

        correct = im - delta[1]*ps  # remove the least-squares fraction of the persistence image
        if (verbosity > 1):
              print ' sky =', sky, ' persist = ' , persist,' used =',used


    #     write correction if fraction of pixels used and calculated persistence exceed thresholds
        if ( persist < self.persist_lo ):
              if (verbosity > 0):
                  print ' The calculated persistence for this file is ', persist,', which is below the minimum allowed value of the persistence (', persist_lo,').  Therefore no corrected image will be written.'

        elif ( used < self.used_lo ):

              if (verbosity > 0):
                  print ' The fraction of pixels for this file is ', used,', which is below the minimum allowed value of the fraction (', used_lo,').  Therefore no corrected image will be written.'
              fh_infile[0].header.update( key = 'BEPDONE', value = 'SKIPPED') 
        else:
              fh_infile[1].data = correct
              fh_infile[0].header.update( key = 'BEPSCALE', value = persist) 
              fh_infile[0].header.update( key = 'BEPFRAC', value = used) 
              fh_infile[0].header.update( key = 'BEPVALLO', value = self.persist_lo) 
              fh_infile[0].header.update( key = 'BEPUSELO', value = self.used_lo)
              fh_infile[0].header.update( key = 'BEPDONE', value = 'PERFORMED')  

        if fh_infile:
           fh_infile.close()

        if (verbosity > 0 ):  
           self.print_pars()
           current_time = time.asctime()
           print '=== BEP finished at ', current_time  

        if (self.run_stdout  != None):  #reset stdout if needed
           sys.stdout = self.orig_stdout 
          
    # end of def persist

    def print_pars(self ):
        """ Print input parameters and calculated values.
        """
        print 'The input parameters used are :'
        print '  The ped input file:  ' , self.input_file
        print '  Lower limit on persistence (persist_lo):  ' , self.persist_lo
        print '  Lower limit on fraction of pixels used (used_lo):  ' , self.used_lo
        print '  The (medianed) persistence model from the file (persist_model):  ' , self.persist_model
        print '  The mask from the file (persist_mask): ' ,self.persist_mask
        print 'The calculated values are :'
        print '  The persistence :  ' , self.persist
        print '  The sky :  ' , self.sky
        print '  The fraction of pixels used :  ' , self.used


def check_py_pars( input_file, persist_lo, used_lo, persist_model, persist_mask):  
       """ When run under python, check validity of input parameters. For unspecified *_lo parameters, the
           user will be given the option of typing in a value or accepting the default value. For an unspecified
           model(mask) file, will try to get file name from PMODFILE(PMSKFILE) from input file header, otherwise
           will get default value from persutil.           

       @param input_file: name of input file
       @type input_file: string
       @param persist_lo: minimum allowed value of the persistence 
       @type persist_lo: Float32
       @param used_lo: minimum allowed value of the fraction of pixels used 
       @type used_lo: Float32
       @param persist_model: filename containing persistence frame (ring median of)
       @type persist_model:  string
       @param persist_mask: filename containing pixel mask 
       @type persist_mask:  string

       @return: persist_lo, used_lo, persist_model, persist_mask
       @rtype: float, float, string, string
       """

       try:
            fh_infile = pyfits.open(input_file)
            in_hdr = fh_infile[0].header
       except:
            opusutil.PrintMsg("F","ERROR - unable to open the input file  "+ str(input_file))
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


def check_cl_pars(input_file, persist_lo, used_lo):
   """ When run from linux command line, verify that each parameter is valid.
       @param input_file: name of input file
       @type input_file: string
       @param persist_lo: minimum allowed value of the persistence 
       @type persist_lo: Float32
       @param used_lo: minimum allowed value of the fraction of pixels used 
       @type used_lo: Float32

       @return: persist_lo, used_lo
       @rtype: float, float
   """

   try:
        fh_c0 = pyfits.open(input_file)
   except:
        opusutil.PrintMsg("F","ERROR - unable to open the input file  "+ str(input_file))
        sys.exit( ERROR_RETURN)
   try:
       if (type( persist_lo ) == str):
          persist_lo = string.atof(persist_lo)
   except:
       print ' The persist_lo value entered (',persist_lo,') is invalid. Try again'
       sys.exit( ERROR_RETURN)

   try:
       if (type( used_lo ) == str):
          used_lo = string.atof(used_lo)
   except:
       print ' The used_lo value entered (',used_lo,') is invalid. Try again'
       sys.exit( ERROR_RETURN)

   return  persist_lo, used_lo


def make_footprint(rin, rout): 
    """ Make an annular mask footprint for use in ndimage.median_filter to create a ring median image.
        Sets all pixels between rin and rout to 1.

    @param rin: inner radius
    @type rin: integer
    @param rout: outer radius
    @type rout: integer

    @return: mask
    @rtype: ndarray
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
    
    @param clip: number of std to use in sigma clipping  
    @type clip:  Float32
    @param d: array of values to sigma clip
    @type clip:  Float32
    
    @return:  clipped mean, clipped std
    @rtype:  float, float    
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

    @param y:  array of values [2d]
    @type y:  Float32
    @param mask:  array of ones or zeros (0 indicates a good value) [2d]
    @type mask:  Int32

    @return:  median of y, ignoring masked elements
    @rtype:  float
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
            The command-line options are:
            -q (quiet)
            -v (very verbose)

            @param cmdline: command-line arguments
            @type cmdline: list of strings
         """

         if ( sys.argv[1] ): input_file = sys.argv[1] 
       
         try:
            nrp = NicRemPersist( input_file, persist_lo=None, used_lo=None, persist_model=None, persist_mask=None, verbosity=None )                      
            nrp.persist()

            del nrp
         except Exception, errmess:
            opusutil.PrintMsg("F","ERROR "+ str(errmess))
            sys.exit( ERROR_RETURN)
