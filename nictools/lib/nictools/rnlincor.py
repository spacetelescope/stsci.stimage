#! /usr/bin/env python
"""
rnlincor: Module to correct for the countrate-dependent nonlinearity in
          a NICMOS image.

:Usage:   Normally used via the STSDAS task rnlincor in the nicmos package.
          To use as pure python, just invoke the run method::

            >>> rnlincor.run('inputfile.fits','outputfile.fits')

          It may also be run from the shell::

            % rnlincor.py infile.fits [outfile.fits] [--nozpcorr]


:For more information:
          Additional user information, including parameter definitions and more
          examples, can be found in the help file for the STSDAS rnlincor task,
          located in nicmos$doc/rnlincor.hlp.

          This task is based on prototype code developed by R. de Jong. The
          algorithm is described in more detail in ISR NICMOS 2006-003 by
          de Jong.

:Dependencies:
    - numpy v1.0.2dev3534 or higher
    - pyfits v1.1b4 or higher

"""
from __future__ import division

__version__="0.8"
__vdate__="2007-12-14"

from stsci.tools import numerixenv #Temporary NUMERIX environment check

import numpy as N
import pyfits
from stsci.tools.fileutil import osfn
from stsci.tools import irafglob
from stsci.tools.xyinterp import xyinterp
import sys, os, glob, time


#Generic helper functions:
#  getcurve: return two column vectors from a FITS table
#  getrow: return a row from a FITS table selected by camera and filter
#
#Generic helper class:
#  FitsRowObject: returned by getrow

#Task-specific helper functions:
#  check_infile: do all the checks and warnings before proceeding
#  update_data: do all the data updates
#  update_header: do all the header updates

#Task wrappers:
#  parrun(parfile): support running from a keyword/value style parfile
#  run(*args): handles all possible filename/list variations

#The main program:
#  rnlincor


#......................................................................
#Generic helper functions
#......................................................................


def getcurve(fname,col1="wavelength",col2="correction",pad=0):
    """ Gets a 2-column table from the first extension of a FITS file.
    Defaults to "wavelength" and "correction" for column names, but others
    can be specified.
    If pad keyword is nonzero, the wavelength table will be extended by
    the pad amount in each direction."""

    f=pyfits.open(fname)
    wave=f[1].data.field(col1)
    corr=f[1].data.field(col2)
    f.close()

    if pad != 0.0:
        wave[0]-=pad
        wave[-1]+=pad

    return wave,corr



def getrow(photmode,corrname):
    """ Pick out the correction parameters from the proper row of
    the table located in corrname, based on photmode. Return
    an object that contains the row fields as attributes. """
    f=pyfits.open(corrname)
    t=f[1].data
    cols=f[1].columns

    modes=t.field('photmode')
    idx = ((modes == photmode))

    try:
        row=t[idx]
        row=FitsRowObject(t[idx])
        return row
    except KeyError:
        print "No match found for %s %s in %s"%(camera,filter,corrname)
        raise KeyError



class FitsRowObject(object):
    """ Class to facilitate working with single table rows. """
    def __init__(self,fitsrecord):
        for name in fitsrecord.names:
            val=fitsrecord.field(name)[0]
            self.__setattr__(name.lower(),val)

    def __repr__(self):
        return str(self.__dict__)


def expandname(pattern):
    """ Select the latest file that matches the pattern in the directory
    specified in the pattern """
    fpattern=osfn(pattern)
    flist=glob.glob(fpattern)
    flist.sort()
    try:
        return flist[-1]
    except IndexError:
        raise IOError,"%s file not found"%pattern


#......................................................................
#Task-specific helper functions
#......................................................................

def check_infile(infile):
    """Open the input file and check all the things that can go wrong.
    If we pass all the tests, return the handle to the open file to
    pass to the main routine."""

    #Open the file
    f=pyfits.open(infile)

    #Make sure it's not already been done:
    if f[0].header.get('rnlcindone',' ') == 'PERFORMED':
        print "Non-linearity correction already performed on %s"%infile
        f.close()
        return()

    #Make sure it's a legal image type:
    imgtype=f[0].header.get('imagetyp','').lower()
    if imgtype not in ['ext','object','target']:

        print """
        WARNING, this is not a science image: this is a %s image.
        This task is intended to run only on science images. Continuing anyway.\n"""%imgtype.upper()

    else: #exclude grism images
        filter=f[0].header.get('filter','').lower()
        if filter.startswith('g'):
            f.close()
            raise ValueError, """Correction cannot be performed on grism images: %s image detected.
            Task aborting."""%filter

    #Construct verbose error messages for required keywords
    modemsg="""Required keyword PHOTMODE not found in %s primary header.
    This keyword is used to select the correct row from the ZPRATTAB
    and RNLPHTTB tables. It specifies the camera and spectral element
    used in the observation and should be of the form
                         NICMOS,1,F160w,DN"""

    plammsg="""Required keyword PHOTPLAM not found in %s primary header.
    This keyword is used to perform the correct interpolation for the function
    specified in the RNLCORTB table. It specifies the pivot wavelength
    for the bandpass specified in PHOTMODE. The correct value can be
    obtained by consulting the PHOTTAB, which is located in
                        nref$*_pht.fits"""


    zptmsg="""Required keyword ZPRATTAB not found in %s primary header.
    This keyword is used to apply the zeropoint correction to the
    nonlinear correction calculation. These tables have names of the
    form
                         nref$*_zpr.fits """


    rncormsg="""Required keyword RNLCORTB not found in %s primary header.
    This keyword is used to specify the nonlinearity correction table
    to be used for this calculation. These tables have names of the
    form
                         nref$*_nlc.fits """


    reqmsg = {'photmode':modemsg,'zprattab':zptmsg,
              'rnlcortb':rncormsg,'photplam':plammsg}

    for kwd in reqmsg:
        try:
            val=f[0].header[kwd]
        except KeyError,e:
            print reqmsg[kwd]%(infile)
            raise e



    #Check multidrizzle status & print warning if necessary
    if f[0].header.has_key('ndrizim'):
        print """WARNING: Detected multidrizzle image, but unable to
        verify image units. Image may be in electrons/s, not DN/s. If so,
        image should be divided by ADCGAIN value before correction.
        Proceeding without unit verification. """


    #Determine which extension has the image.
    if len(f) > 1:
        ext=1
    else:
        ext=0

    return f, ext

def update_data(f,imgext,img,mul):
    #Correct the data
    f[imgext].data=img
    if len(f) > 1:
        f[imgext+1].data*=mul

def update_header(f,alpha,zpcorr,zpratio=None):
    """Update all the header keywords"""

    f[0].header.update('RNLCDONE',
                         'PERFORMED',
                         'corrected count-rate dependent non-linearity')
    f[0].header.update('RNLCALPH',
                         alpha,
                         'power-law of non-linearity correction')

    if not zpcorr:
        f[0].header.update('RNLCZPRT',
                           zpratio,
                           'Ratio to correct data to match PHOTFNU')
        f[0].header.add_history('%s rnlincorr: no zeropt correction applied'%time.ctime(),after='RNLCZPRT')

#......................................................................
#The main routine
#......................................................................
def rnlincor(infile,outfile,**opt):
    """ The main routine """
    numerixenv.check() #Temporary NUMERIX environment check
    print "rnlincor version: ",__version__

    #Translate an option
    zpcorr = not opt['nozpcorr']

    #Get the image data
    try:
        f,imgext=check_infile(infile)
    except Exception, e:
        print str(e)
        return

    img=f[imgext].data

    #Correct it for sky subtraction if necessary
    try:
        skyval=f[imgext].header['skyval']
        print "Sky subtraction detected: compensating"
        img=img+skyval
    except KeyError:
        skyval=None

    #Get the relevant filenames.
    zpratfile=expandname(f[0].header['zprattab'])
    nlfile=expandname(f[0].header['rnlcortb'])

    #Get the correct key into the tables
    photmode=f[0].header['photmode']
    pivlam  =f[0].header['photplam']
    #Pick out the right row from the photometric correction table
    try:
        zprat=getrow(photmode,zpratfile)
    except IndexError, e:
        print """Task Aborting: No zero-point correction is available in the
        %s file
        for this image's observing mode %s.
        You may wish to rerun the task with the 'nozpcorr' option set."""%(zpratfile,photmode.upper())
        f.close()
        return

    #Read in the nonlinearity correction
    #Pad the wavelength table by 5 Angstroms on each end to protect
    #against rounding errors and lagging reference file updates.
    wave,corr = getcurve(nlfile,pad=5.0)

    #Interpolate to get the correction at the pivot wavelength
    nonlcor = xyinterp(wave,corr,pivlam)
    print "Using non-linearity correction %6.4f mag/dex"%nonlcor

    #Correct from mag/dex to alpha in power law
    alpha = (nonlcor/2.5) + 1  #Add 1 to avoid divide-by-zero in next step?
    inv_alpha = (1.0/alpha) - 1

    #Compute the correction
    mul = N.where(N.not_equal(img,0),abs(img),1)**inv_alpha

    #Apply zero point correction if requested
    if zpcorr:
        print "Applying zeropoint correction"
        mul /= zprat.zpratio
    else:
        print "NOT applying zeropoint correction"


    #Apply the correction
    img*=mul

    #If the sky subtraction was added in, take it back out.
    #Use the mean value of the correction so that the skyval taken
    #back out is a constant, like the one added in earlier.
    if skyval is not None:
        img -= skyval*mul.mean()

    #Update the HDUlist and write out results
    update_data(f,imgext,img,mul)
    update_header(f,alpha,zpcorr,zpratio=zprat.zpratio)
    f.writeto(outfile,clobber=True)

#.....................................................................
#Support running with a (non-IRAF) parameter file
#.....................................................................
def parrun(parfile):
    d={}
    f=open(parfile)
    for line in f:
        key,value=line.split()
        d[key.lower()]=value
    f.close()
    rnlincor(d['infile'],d['outfile'])
#.....................................................................
# Fancy option handling in case we didn't come in from __main__
#.....................................................................
def set_default_options(inopt):
    opt=inopt.copy()
    opt['nozpcorr']=inopt.get('nozpcorr',False)
    return opt
#.....................................................................
#Wrapper for filename handling
#.....................................................................
def run(*args,**inopt):
    opt=set_default_options(inopt)
    #We always provide input filename
    infile=args[0]
    if not os.path.isfile(infile):
        raise IOError, '%s not found; task aborting.'%infile

    #We sometimes provide output filename
    if (len(args) == 2 and len(args[1])!=0):
        outfile=args[1]
    else:
        #But if not, we have to construct it
        if '_' in os.path.basename(infile):
            root,junk=infile.split('_',1)
            outfile=root+'_nlc.fits'
        else:
            outfile=infile.replace('.fits','_nlc.fits')

    #Now that we have both names, call the task.
    rnlincor(infile,outfile,**opt)


#......................................................................
#Support running from the shell
#......................................................................
if __name__ == '__main__':
    import optparse
    #Define UI
    p=optparse.OptionParser()
    p.add_option("--nozpcorr",action="store_true",default=False,
                 help="Do not include zeropoint correction when calculating correction")
    p.set_usage("usage: rnlincor.py infile [outfile] [--nozpcorr]")
    #Get results
    opt, args=p.parse_args()
    optd={}
    for o in vars(opt):
        optd[o]=getattr(opt,o)

    #and run the task
    run(*args, **optd)
