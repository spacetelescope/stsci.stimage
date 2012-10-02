"""
puftcorr: Module for estimating and removing "Mr. Staypuft" signal from
          a NICMOS exposure.

:Usage:
    Normally used via the STSDAS task puftcorr in the nicmos package.
    To use as pure python, just invoke the clean method::

        >>> puftcorr.clean('inputfile.fits','outputfile.fits')

:For more information:
    Additional user information, including parameter definitions and more
    examples, can be found in the help file for the STSDAS puftcorr task,
    located in nicmos$doc/puftcorr.hlp.

    The algorithm and IDL prototype were developed by L.Bergeron, but
    never made publicly available.

:Dependencies:

    - numpy v1.0.2dev3534 or higher
    - pyfits v1.1b4 or higher
    - convolve v2.0 or higher
    - ndimage  v2.0 or higher

"""
#.....................................................
__version__="0.17"
__vdate__="2007-11-30"

#.....................................................
from stsci.tools import numerixenv  #Temporary NUMERIX environment check

#.....................................................
import os, sys, shutil
import exceptions
import math
import numpy as N
import stsci.ndimage as ndimage
import stsci.convolve as conv
import pyfits

#History:
# Modified for numpy usage
# Initial python implementation: Dec 2005, Bushouse
# Based on IDL implementation by Bergeron

#Notes for future improvement:
# - possibly make filename its own class so it can have a method for nref
#   instead of using the osfn helper function
#........................................................................
#Class definitions
#.........................................................................

class params:
    def __init__(self, camera):

        if camera == 1:
           self.hh   = 0.8
           self.bmax = 87.0
           self.tx   = 2.0
           self.ampscale = 20.0*1.30024e-5
           self.a1   = 0.0
           self.a2   = 0.003
           self.la   = 0
        elif camera == 2:
           self.hh   = 0.8
           self.bmax = 87.0
           self.tx   = 2.0
           self.ampscale = 20.0*1.30024e-5
           self.a1   = -0.00014
           self.a2   = 0.07
           self.la   = 128
        elif camera == 3:
           self.hh   = 0.8
           self.bmax = 87.0
           self.tx   = 2.0
           self.ampscale = 20.0*1.30024e-5
           self.a1   = -0.00014
           self.a2   = math.fabs(128.0*self.tx*self.a1)
           self.la   = 128
        else:
           print " unknown camera number %d" %(camera)


class InputFile:
    """ Stores a collection of keywords and the header for an exposure. """

    def __init__(self,imgfile):

        # get the input file name and open it
        self.filename = osfn(imgfile)
        f = pyfits.open(self.filename)
        self.f = f
        h = f[0].header
        self.h = h

        # retrieve the necessary header keyword values
        self.camera = h['camera']
        self.nsamp  = h['nsamp']
        self.darkname = osfn(h['darkfile'])

        # open the dark reference file
        self.dark = pyfits.open(self.darkname)

        # load the zeroth-read image for use later
        self.zread = f['sci',self.nsamp].data.astype(N.dtype('float32'))


class Readout:

    def __init__(self,input,sampnum):

        self.imset  = sampnum
        self.data   = input.f['sci',self.imset].data.astype(N.dtype('float32'))
        self.header = input.f['sci',self.imset].header
        self.npix   = self.header['naxis1']

        # subtract zeroth read
        #self.data = self.data - input.zread
        self.data -= input.zread

        # subtract dark
        if self.imset < input.nsamp:
           self.data -= input.dark['sci',self.imset+26-input.nsamp].data

#..........................................................................
# Exception definitions
class NoPuftError(exceptions.Exception):
    pass

#..........................................................................
#Helper functions:
#-............................................................................
def osfn(filename):
    """Return a filename with iraf syntax and os environment names substituted out"""
    if filename is None:
        return filename

    #Baby assumptions: suppose that the env variables will be in front.

    if filename.startswith('$'):  #we need to translate a logical
        symbol,rest=filename.split('/',1)
    elif '$' in filename: #we need to fix the iraf syntax
        symbol,rest=filename.split('$',1)
    else:
        return filename
    newfilename=os.environ[symbol]+'/'+rest
    return newfilename

#..............................................................................
# General functions
#..........................................................................

def get_totsig (im,la):

    ln1 = im.npix / 2   # length of 1 quad
    ln2 = im.npix       # length of 2 quads

    # split the full image into individual quadrants
    q1 = im.data[0:ln1,0:ln1]
    q2 = im.data[ln1:ln2,0:ln1]
    q3 = im.data[ln1:ln2,ln1:ln2]
    q4 = im.data[0:ln1,ln1:ln2]

    # transpose axes
    q1 = N.transpose(q1)
    q2 = N.transpose(q2)
    q3 = N.transpose(q3)
    q4 = N.transpose(q4)

    # unravel the 2-d quads into 1-d vectors
    q1 = N.ravel(q1)
    q2 = N.ravel(q2)
    q3 = N.ravel(q3)
    q4 = N.ravel(q4)

    # compute the total of the signals in all 4 quads,
    # and then reverse the ordering
    totsig = q1+q2+q3+q4
    totsig = totsig[::-1]

    # compute the "look ahead" signal vector
    quad_len = ln1 * ln1
    totsigla = N.zeros((quad_len),dtype=N.dtype('float32'))
    totsigla[0:quad_len-la]  = totsig[la:]
    totsigla[quad_len-la-1:] = totsig[quad_len-la-1:]

    return totsig,totsigla


def get_corr (im, pars):

    # transform 2-d image to 1-d total signal vectors
    totsig, totsigla = get_totsig(im,pars.la)

    ln1 = im.npix / 2
    ln2 = im.npix
    quad_len = ln1*ln1

    # compute staypuft signal for each pixel
    mask = totsig > 40.0
    p0 = N.fabs(pars.ampscale * mask * totsig)
    p1 = N.fabs(pars.ampscale * mask * totsigla)

    ekern = N.exp(-N.arange(ln1*pars.tx)/pars.hh)
    qkern = pars.a1*N.arange(ln1*pars.tx) + pars.a2

    e = conv.convolve (p0, ekern, mode=conv.FULL)
    q = conv.convolve (p1, qkern, mode=conv.FULL)
    b = e[0:quad_len] + q[0:quad_len]

    # transform the correction vector back into a 2-d image quad
    b = b[::-1]
    b = N.reshape (b, (ln1,ln1))
    b = N.transpose(b)

    # replicate the correction into all 4 full image quads
    im.data[0:ln1,0:ln1] = b
    im.data[0:ln1,ln1:ln2] = b
    im.data[ln1:ln2,0:ln1] = b
    im.data[ln1:ln2,ln1:ln2] = b

    return im

#....................................................................
# The "main" program
#....................................................................
def clean (usr_imgfile, usr_outfile):
    numerixenv.check() #Temporary NUMERIX environment check
    print "puftcorr version %s" %__version__
    print "Input file:  %s" %usr_imgfile
    print "Output file: %s" %usr_outfile

    imgfile = osfn(usr_imgfile)
    outfile = osfn(usr_outfile)

    # check for existence of output file
    if os.access(outfile,os.F_OK):
       s = "\nERROR: Output file %s already exists\n" %(outfile)
       sys.stdout.write(s)
       return

    # create the output file as a copy of the input raw file
    shutil.copyfile (imgfile, outfile)

    # retrieve the input file
    img = InputFile(imgfile)

    # set correction parameter values
    pars = params(img.camera)

    # loop over readouts in the input file
    for i in range(img.nsamp):
        imset = i+1

        if imset == 1:
           s = " processing imset 1"
        elif imset == img.nsamp:
           s = " %d\n" % imset
        else:
           s = " %d" % imset
        sys.stdout.write(s)
        sys.stdout.flush()

        # get individual readout
        im = Readout(img,imset)

        # rotate, if necessary
        if img.camera == 1:
            im.data = ndimage.rotate(im.data,-90,mode='nearest')
        elif img.camera == 3:
            im.data = ndimage.rotate(im.data,180,mode='nearest')

        # get correction image
        im = get_corr(im, pars)

        # rotate back, if necessary
        if img.camera == 1:
            im.data = ndimage.rotate(im.data,+90,mode='nearest')
        elif img.camera == 3:
            im.data = ndimage.rotate(im.data,180,mode='nearest')

        # subtract correction image from original raw image
        im.data = img.f['sci',imset].data - im.data

        # make sure corrected pixel values don't go off the
        # ends of the Int16 data range before writing to output
        im.data = N.clip(im.data,-32768.0,32767.0)

        # write corrected image to output file
        pyfits.update(outfile,im.data.astype(N.dtype('int16')),
                      'sci',imset,
                      header=im.header)


    # close the input files
    img.f.close()
    img.dark.close()

    return
