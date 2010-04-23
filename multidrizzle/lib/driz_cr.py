# DRIZ_CR  -- mask blemishes in dithered data by comparison of an image
#             with a model image and the derivative of the model image.
#
#
from __future__ import division # confidence high

# Import external packages
import numpy as np
import convolve as NC
import pyfits
import os

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
        __crMask = np.zeros(self.__inputImage.shape,dtype=np.uint8)

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
        __t1 = np.absolute(self.__inputImage - self.__blotImg)
        __ta = np.sqrt(self.__gain * np.absolute(self.__blotImg * self.__expmult + self.__back * self.__expmult) + self.__rn * self.__rn)
        __tb = ( self.__mult1 * self.__blotDerivImg + self.__snr1 * __ta / self.__gain )
        del __ta
        __t2 = __tb / self.__expmult
        del __tb
        __tmp1 = np.logical_not(np.greater(__t1, __t2))
        del __t1
        del __t2

        # Create a convolution kernel that is 3 x 3 of 1's
        __kernel = np.ones((3,3),dtype=np.uint8)
        # Create an output tmp file the same size as the input temp mask array
        __tmp2 = np.zeros(__tmp1.shape,dtype=np.int16)
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
        __xt1 = np.absolute(self.__inputImage - self.__blotImg)
        __xta = np.sqrt(self.__gain * np.absolute(self.__blotImg * self.__expmult + self.__back * self.__expmult) + self.__rn * self.__rn)
        __xtb = ( self.__mult2 * self.__blotDerivImg + self.__snr2 * __xta / self.__gain )
        del __xta
        __xt2 = __xtb / self.__expmult
        del __xtb
        # It is necessary to use a bitwise 'and' to create the mask with numarray objects.
        __crMask = np.logical_not(np.greater(__xt1, __xt2) & np.less(__tmp2,9) )
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
        __crMask= __crMask_orig_bool.astype( np.int8 )
        
        # make radial convolution kernel and convolve it with original __crMask 
        cr_grow_kernel = np.ones((grow, grow))     # kernel for radial masking of CR pixel
        cr_grow_kernel_conv = __crMask.copy()   # for output of convolution
        NC.convolve2d( __crMask, cr_grow_kernel, output = cr_grow_kernel_conv)
        
        # make tail convolution kernel and convolve it with original __crMask
        cr_ctegrow_kernel = np.zeros((2*ctegrow+1,2*ctegrow+1))  # kernel for tail masking of CR pixel
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
        where_cr_grow_kernel_conv    = np.where( cr_grow_kernel_conv < grow*grow,0,1 )        # radial
        where_cr_ctegrow_kernel_conv = np.where( cr_ctegrow_kernel_conv < ctegrow, 0, 1 )     # length
        __crMask = np.logical_and( where_cr_ctegrow_kernel_conv, where_cr_grow_kernel_conv) # combine masks

        __crMask = __crMask.astype(np.uint8) # cast back to Bool

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
        np.bitwise_and(self.dqMask,self.crMask,self.dqMask)

    def createcorrfile(self,
        corrName = None, # Name of output file corr image
        header = None # Optionally provided header for output image
        ):

        """ Create a clean image by replacing any pixel flagged as "bad" with the corresponding values from the blotted image."""

#       imcalc(s1,img0//cor_suffix,"if (im2 .eq. 0) then im3 else im1", verb-)
        try:
            # CREATE THE CORR IMAGE
            __corrFile = np.zeros(self.__inputImage.shape,dtype=self.__inputImage.dtype)
            __corrFile = np.where(np.equal(self.dqMask,0),self.__blotImg,self.__inputImage)
            
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

        __bitarray = np.logical_not(self.crMask).astype(np.int16) * cr_bits_value
        np.bitwise_or(dqarray,__bitarray,dqarray)

    def createcrmaskfile(self,
        crName = None, # Name of outputfile cr mask image
        header = None # Optionally provided header for output image
        ):

        """ Create a fits file containing the generated cosmic ray mask. """
        try:
            _cr_file = np.zeros(self.__inputImage.shape,np.uint8)
            _cr_file = np.where(self.crMask,1,0).astype(np.uint8)
            
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
