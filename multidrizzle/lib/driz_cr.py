# DRIZ_CR  -- mask blemishes in dithered data by comparison of an image
#             with a model image and the derivative of the model image.
#
# IRAF CL HISTORY
# ---------------
# V1.0     -- released June 1998
# V1.01    -- error in imcalc quotation fixed -- 30 June 1998
# V1.02    -- updated to handle cps   --  26 Oct 1998
# V1.03    -- added ability to weight using flatfield  -- A. Fruchter
# V1.04    -- explicitly specify the ".pl" extension in the last imcalc,
#               and imtype for _bl and _bl_deriv files.  -- JC Hsu, 21 Jan 2000
# V1.05    -- added ability to change suffixes for input and output files  --  26 Jun 2000
#
# PYTHON HISTORY
# --------------
# 22 Oct 2003 -- Version 0.0.0 -- Created a Python version of the DRIZ_CR CL script. -- C.J.Hanley
# 18 Nov 2003 -- Version 0.1.0 -- Removed the bitvalue parameter as a user selectable parameter.
#                                   Since we want to work on a boolean dqmask, bad pixels are being
#                                   set to 0.  The 0 value is hard coded in the putmask step.
# 30 Jan 2004 -- Version 0.1.1 -- A corr file will now be created in the corrName is set to value. -- CJH
# 23 Feb 2004 -- Version 0.1.2 -- If the background keyword is not found, the background level for the image
#                                   is assumed to be 0.  A warning message is printed.  If the exposure time
#                                   keyword is not found, the countrate is assumed to be 1.  This is only an
#                                   issue if the units are specified as "CPS" for the image in question.  A
#                                   warning message is issued when this occurs. -- CJH
# 22 Mar 2004 -- Version 0.1.3 -- Have completely refactored the driz_cr module.  I have added methods for updating
#                                 the DQ arrays with the CR information using a specified bit value.  Also, have
#                                 added a method for outputing the CR mask as a fits files.
# 12 May 2004 -- Version 0.1.4 -- Fixed logic error in the update of dq arrays methods.  Instead of adding 4096 to
#                                 the dqarray we now do a bitwise and operation.
# 20 May 2004 -- Version 0.1.5 -- Fixed bug in the method for creating a cr mask fits file.
# 30 Sep 2004 -- Version 0.1.6 -- Modified the cor and cr mask file creation methods to install a copy of the
#                                   input's primary + extension header
# 05 Apr 2005 -- Version 0.2.0 -- Modified the cor and cr mask file creation methods to remove extension specific
#                                   keywords from the header it creates. -- CJH
# 02 Jun 2005 -- Version 1.0.0 -- Added parameters driz_cr_grow and driz_cr_ctegrow for CTE masking of cosmic 
#                                   rays. -- DMG
# 16 Jan 2006 -- Version 1.0.1 -- Enforced the format of the crmask file to be UInt8, instead of upcasting to  
#                                   system defined integer (eg., Int64 for 64-bit systems). -- WJH

# Import external packages
import numarray as N
import numarray.convolve as NC
import pyfits
import os
from numarray import * 

# Version
__version__ = '1.0.1'

class DrizCR:
    """mask blemishes in dithered data by comparison of an image
    with a model image and the derivative of the model image."""

    def __init__(self,
        image,                      # Image for cosmic ray cleaning, --read only--
        header,                     # Input Image header  --read only--
        blotImg,                    # Input blotted image (read only)
        blotDerivImg,               # Input blotted image derivative (read only)
        dqMask,                     # Mask to which the generated CR mask is to be combined (read/write mode required)
        gain     = 7,               # Detector gain, e-/ADU
        grow     = 1,               # Radius around CR pixel to mask [default=1 for 3x3 for non-NICMOS]   
        ctegrow  = 0,               # Length of CTE correction to be applied
        ctedir   = 1,               # ctr correction direction   
        amp      = 'A',             # amplifier (used for HRC)
        rn       = 5,               # Read noise in electrons
        SNR      = "4.0 3.0",       # Signal-to-noise ratio
        scale    = "0.5 0.4",       # scaling factor applied to the derivative
        units    = "counts",        # counts or cps
        backg    = 0,               # Background value
        expkey   = "exptime"        # exposure time keyword
        ):

        # Initialize input parameters
        self.__gain = gain
        self.__grow = grow   
        self.__ctegrow = ctegrow
        self.__ctedir = ctedir  
        self.__amp = amp   
        self.__rn = rn
        self.__inputImage = image
        self.__header = header
        self.__blotImg = blotImg
        self.__blotDerivImg = blotDerivImg

        __SNRList = SNR.split()
        self.__snr1  = float(__SNRList[0])
        self.__snr2 = float(__SNRList[1])

        __scaleList = scale.split()
        self.__mult1 = float(__scaleList[0])
        self.__mult2 = float(__scaleList[1])

        self.__units = units
        self.__back = backg
        self.__expkey = expkey.upper()

        # Masks we wish to retain
        self.dqMask = dqMask
        self.crMask = None

        # Define output parameters
        __crMask = N.zeros(self.__inputImage.shape,N.Bool)

        # Determine a scaling factor depending on the units of the input image, "counts" or "cps"
        if (self.__units == "counts"):
            self.__expmult = 1
        elif (self.__units == "cps"):
            try:
                self.__expmult = self.__header[self.__expkey]
            except:
                print "WARNING: Exposure time keyword ", self.__expkey, " was not found.  Count rate set to 1."
        else:
            raise ValueError, "UNITS KEYWORD NOT RECOGONIZED"

        # Part 1 of computation

        # IRAF Syntax
#       cos_var1 = "if (abs(im1-im2) .gt. "//mult1//" * im3 + ("//snr1
#       cos_var2 = "* sqrt("//__gain//"*abs(im2*"//expmult//" + "//back//"*"//expmult//")+"//__rn//"*"//__rn//")/"//__gain//")/"//expmult//") then 0 else 1"
#       tmp4 = mktemp("drz")
#       print(cos_var1, cos_var2, > tmp4)
#       imcalin = inimg//","+bl+","+bl_deriv
#       imcalc (imcalin, tmp1, "@"//tmp4, verb-)
#       convolve(tmp1, tmp2, "", "1 1 1", "1 1 1", bilin=yes, radsym=no)

        # Create a temp array mask
        __t1 = N.absolute(self.__inputImage - self.__blotImg)
        __ta = N.sqrt(self.__gain * N.absolute(self.__blotImg * self.__expmult + self.__back * self.__expmult) + self.__rn * self.__rn)
        __tb = ( self.__mult1 * self.__blotDerivImg + self.__snr1 * __ta / self.__gain )
        del __ta
        __t2 = __tb / self.__expmult
        del __tb
        __tmp1 = N.logical_not(N.greater(__t1, __t2))
        del __t1
        del __t2

        # Create a convolution kernel that is 3 x 3 of 1's
        __kernel = N.ones((3,3),'Bool')
        # Create an output tmp file the same size as the input temp mask array
        __tmp2 = N.zeros(__tmp1.shape,N.Int16)
        # Convolve the mask with the kernel
        NC.convolve2d(__tmp1,__kernel,output=__tmp2,fft=0,mode='nearest',cval=0)
        del __kernel
        del __tmp1

        # Part 2 of computation

        # IRAF Syntax
#       cos_var1 = "if ((abs(im1-im2) .gt. "//mult2//" * im3 + ("//snr2
#       cos_var2 = "* sqrt("//__gain//"*abs(im2*"//expmult//" + "//back//"*"//expmult//")+"//__rn//"*"//__rn//")/"//__gain//")/"//expmult//") .and. (im4 .lt. 9)) then 0 else 1"
#       tmp5 = mktemp("drz")
#       print(cos_var1, cos_var2, > tmp5)
#       imcalout = img0+cr_suffix+".pl"
#       imcalin = inimg+","+bl+","+bl_deriv+","+tmp2
#       imcalc (imcalin, imcalout, "@"//tmp5, verb-)

        # Create the CR Mask
        __xt1 = N.absolute(self.__inputImage - self.__blotImg)
        __xta = N.sqrt(self.__gain * N.absolute(self.__blotImg * self.__expmult + self.__back * self.__expmult) + self.__rn * self.__rn)
        __xtb = ( self.__mult2 * self.__blotDerivImg + self.__snr2 * __xta / self.__gain )
        del __xta
        __xt2 = __xtb / self.__expmult
        del __xtb
        # It is necessary to use a bitwise 'and' to create the mask with numarray objects.
        __crMask = N.logical_not(N.greater(__xt1, __xt2) & N.less(__tmp2,9) )
        del __xt1
        del __xt2
        del __tmp2

        # Part 3 of computation - flag additional cte 'radial' and 'tail' pixels surrounding CR pixels as CRs

        # In both the 'radial' and 'length' kernels below, 0->good and 1->bad, so that upon
        # convolving the kernels with __crMask, the convolution output will have low->bad and high->good 
        # from which 2 new arrays are created having 0->bad and 1->good. These 2 new arrays are then 'anded'
        # to create a new __crMask.

        # recast __crMask to int for manipulations below; will recast to Bool at end
        __crMask_orig_bool= __crMask.copy() 
        __crMask= __crMask_orig_bool.astype( Int8 )
        
        # make radial convolution kernel and convolve it with original __crMask 
        cr_grow_kernel = N.ones((grow, grow))     # kernel for radial masking of CR pixel
        cr_grow_kernel_conv = __crMask.copy()   # for output of convolution
        NC.convolve2d( __crMask, cr_grow_kernel, output = cr_grow_kernel_conv)
        
        # make tail convolution kernel and convolve it with original __crMask
        cr_ctegrow_kernel = N.zeros((2*ctegrow+1,2*ctegrow+1))  # kernel for tail masking of CR pixel
        cr_ctegrow_kernel_conv = __crMask.copy()  # for output convolution 

        # which pixels are masked by tail kernel depends on sign of ctedir (i.e.,readout direction):
        if ( ctedir == 1 ):  # HRC: amp C or D ; WFC: chip = sci,1 ; WFPC2
           cr_ctegrow_kernel[ 0:ctegrow, ctegrow ]=1    #  'positive' direction
        if ( ctedir == -1 ): # HRC: amp A or B ; WFC: chip = sci,2
           cr_ctegrow_kernel[ ctegrow+1:2*ctegrow+1, ctegrow ]=1    #'negative' direction
        if ( ctedir == 0 ):  # NICMOS: no cte tail correction
           pass
       
        # do the convolution
        NC.convolve2d( __crMask, cr_ctegrow_kernel, output = cr_ctegrow_kernel_conv)    

        # select high pixels from both convolution outputs; then 'and' them to create new __crMask
        where_cr_grow_kernel_conv    = where( cr_grow_kernel_conv < grow*grow,0,1 )        # radial
        where_cr_ctegrow_kernel_conv = where( cr_ctegrow_kernel_conv < ctegrow, 0, 1 )     # length
        __crMask = N.logical_and( where_cr_ctegrow_kernel_conv, where_cr_grow_kernel_conv) # combine masks

        __crMask = __crMask.astype( Bool) # cast back to Bool

        del __crMask_orig_bool
        del cr_grow_kernel 
        del cr_grow_kernel_conv 
        del cr_ctegrow_kernel 
        del cr_ctegrow_kernel_conv
        del where_cr_grow_kernel_conv  
        del where_cr_ctegrow_kernel_conv 

        # set up the 'self' objects
        self.crMask = __crMask

        # update the dq mask with the cr mask information
        self.__updatedqmask()

    # driz_cr class methods

    def __updatedqmask(self):

        """ Update the dq file generated mask with the cr mask information """

        # Apply CR mask to the DQ array in place
        N.bitwise_and(self.dqMask,self.crMask,self.dqMask)

    def createcorrfile(self,
        corrName = None, # Name of output file corr image
        header = None # Optionally provided header for output image
        ):

        """ Create a clean image by replacing any pixel flagged as "bad" with the corresponding values from the blotted image."""

#       imcalc(s1,img0//cor_suffix,"if (im2 .eq. 0) then im3 else im1", verb-)
        try:
            # CREATE THE CORR IMAGE
            __corrFile = N.zeros(self.__inputImage.shape,self.__inputImage.type())
            __corrFile = N.where(N.equal(self.dqMask,0),self.__blotImg,self.__inputImage)
            
            # Remove the existing cor file if it exists
            try:
                os.remove(corrName)
                print "Removing file:",corrName
            except:
                pass

            # Create the output file
            fitsobj = pyfits.HDUList()
            if (header != None):
                del(header['NAXIS1'])
                del(header['NAXIS2'])
                if header.has_key('XTENSION'):
                    del(header['XTENSION'])
                if header.has_key('EXTNAME'):
                    del(header['EXTNAME'])
                if header.has_key('EXTVER'):
                    del(header['EXTVER'])

                if header.has_key('NEXTEND'):
                    header['NEXTEND'] = 0
                
                hdu = pyfits.PrimaryHDU(data=__corrFile,header=header)
                del hdu.header['PCOUNT']
                del hdu.header['GCOUNT']

            else:
                hdu = pyfits.PrimaryHDU(data=__corrFile)
            fitsobj.append(hdu)
            fitsobj.writeto(corrName)
            
        finally:
            # CLOSE THE IMAGE FILES
            fitsobj.close()
            del fitsobj,__corrFile

    def updatedqarray(self,
        dqarray,            # The data quality array to be updated.
        cr_bits_value       # Bit value set asside to represent a cosmic ray hit.
        ):

        """ Update the dqarray with the cosmic ray detection information using the provided bit value """

        __bitarray = N.logical_not(self.crMask) * cr_bits_value
        N.bitwise_or(dqarray,__bitarray,dqarray)

    def createcrmaskfile(self,
        crName = None, # Name of outputfile cr mask image
        header = None # Optionally provided header for output image
        ):

        """ Create a fits file containing the generated cosmic ray mask. """
        try:
            _cr_file = N.zeros(self.__inputImage.shape,N.UInt8)
            _cr_file = N.where(self.crMask,1,0).astype(N.UInt8)
            
            # Remove the existing cor file if it exists
            try:
                os.remove(crName)
                print "Removing file:",corrName
            except:
                pass

            # Create the output file
            fitsobj = pyfits.HDUList()
            if (header != None):
                del(header['NAXIS1'])
                del(header['NAXIS2'])
                if header.has_key('XTENSION'):
                    del(header['XTENSION'])
                if header.has_key('EXTNAME'):
                    del(header['EXTNAME'])
                if header.has_key('EXTVER'):
                    del(header['EXTVER'])

                if header.has_key('NEXTEND'):
                    header['NEXTEND'] = 0

                hdu = pyfits.PrimaryHDU(data=_cr_file,header=header)
                del hdu.header['PCOUNT']
                del hdu.header['GCOUNT']
            else:
                hdu = pyfits.PrimaryHDU(data=_cr_file)
            fitsobj.append(hdu)
            fitsobj.writeto(crName)
            
        finally:
            # CLOSE THE IMAGE FILES
            fitsobj.close()
            del fitsobj,_cr_file
