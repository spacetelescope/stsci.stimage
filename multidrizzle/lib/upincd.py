"""
UPINCD.PY - Updated the WCS in an image header so that
            it matches the geometric distortion defined in an IDC table
            which is referenced in the image header and is adjusted
            for the velocity aberration correction.

The improved CD is calculated by combining the true output CD from the
known scale and orientation with the geometric distortion. This is
then corrected for the VAFACTOR velocity aberration factor. The CRVALs
are also adjusted appropriately.

The WCS is first copied to OCD1_1 etc before being updated.

First try, Richard Hook, ST-ECF/STScI, August 2002.
Version 0.0.1 (WJH) - Obtain IDCTAB using PyDrizzle function.
Version 0.1 (WJH) - Added support for processing image lists.
                    Revised to base CD matrix on ORIENTAT, instead of PA_V3
                    Supports subarrays by shifting coefficients as needed.
Version 0.2 (WJH) - Implemented orientation computation based on PA_V3 using
                    Troll function from Colin to compute new ORIENTAT value.
Version 0.3 (WJH) - Supported filter dependent distortion models in IDCTAB
                    fixed bugs in applying Troll function to WCS.
Version 0.4 (WJH) - Updated to support use of 'defaultModel' for generic
                    cases: XREF/YREF defaults to image center and idctab
                    name defaults to None.

Version 0.5 (WJH) - Added support for WFPC2 OFFTAB updates, which updates
                    the CRVALs.  However, for WFPC2 data, the creation of
                    the backup values does not currently work.
UPINWCS V0.1 (RNH) - UPINWCS created from UPINCD. September 2003.

Version 0.6 (RNH,WJH)
                   - Converted UPINWCS to UPINCD, and made compatible with
                   WFPC2, as well as ACS. Corrects for VAFACTOR, and
                   recomputes CRVALs based on V2REF/V3REF.
Version 0.6.1 (WJH) - Updated to work with MDRIZZLE.
Version 0.6.2 (CJH) - Removed iraf boolean values and calls to hedit.

"""

#import iraf
from math import *
import string
import pydrizzle
import time

from pydrizzle import wcsutil,fileutil,drutil,buildasn
from pydrizzle.drutil import combin

import numarray as N

yes = True
no = False

# Define parity matrices for supported detectors.
# These provide conversion from XY to V2/V3 coordinate systems.
# Ideally, this information could be included in IDCTAB...
PARITY = {'WFC':[[1.0,0.0],[0.0,-1.0]],'HRC':[[-1.0,0.0],[0.0,1.0]],
          'SBC':[[-1.0,0.0],[0.0,1.0]],'default':[[1.0,0.0],[0.0,1.0]],
          'WFPC2':[[-1.0,0.],[0.,1.0]]}

NUM_PER_EXTN = {'ACS':3,'WFPC2':1}

__version__ = '0.6.2 (19 May 2004)'

# Top level module (named "run" now)
def run(image,quiet=yes,restore=no):

    print "+ UPINCD Version %s" % __version__

    if image.find('[') > -1:
        # We were told to only work with a specific extension
        if restore == no:
            _update(image,quiet=quiet)
        else:
            if not quiet:
                print 'Restoring original WCS values for',image
            restoreWCS(image)
    else:
        # Work with all extensions of all images in list
        _files = buildasn._findFiles(image)
        for img in _files:
            _phdu = img[0]+'[0]'
            _numext = int(drutil.findNumExt(_phdu))
            _instrument = fileutil.getKeyword(_phdu,keyword='INSTRUME')
            if NUM_PER_EXTN.has_key(_instrument):
                _num_per_extn = NUM_PER_EXTN[_instrument]
            else:
                raise "Instrument %s not supported yet. Exiting..."%_instrument

            _nimsets = _numext / _num_per_extn
            for i in xrange(_nimsets):
                if not img[0].find('.fits'):
                    _image = img[0]+'[sci,'+repr(i+1)+']'
                else:
                    _image = img[0]+'['+repr(i+1)+']'
                if not restore:
                    _update(_image,quiet=quiet,instrument=_instrument)
                else:
                    if not quiet:
                        print 'Restoring original WCS values for',_image
                    restoreWCS(_image)

def _update(image,quiet=None,instrument=None):


    # Check whether we are already updated
    if (fileutil.getKeyword(image,'WCSCORR') == 'DONE'):
        print 'Image header has already been updated (WCSCORR=DONE).'
        return

    # First get the name of the IDC table
###    idctab = drutil.getIDCFile(image,keyword='idctab')
    idctab = fileutil.getKeyword(image,'idctab')
    
    if idctab == '': idctab = None
    # Read in any specified OFFTAB, if present
    offtab = fileutil.getKeyword(image,'OFFTAB')
    dateobs = fileutil.getKeyword(image,'DATE-OBS')
    
    print "-Updating image ",image
    if not quiet:
        print "-Reading IDCTAB file ",idctab

    # This section gets several keyword values from the headers

    # Get telescope orientation from image header
    pvt = float(fileutil.getKeyword(image,'PA_V3'))
    
    # Read in ra,dec, both reference point and target (for computing
    # orientation at aperture and for scaling by VAFACTOR)
    alpha = float(fileutil.getKeyword(image,'CRVAL1'))
    dec = float(fileutil.getKeyword(image,'CRVAL2'))
    
    # Get the target position (this is where velocity aberration is
    # suppressed)
    ra_targ = float(fileutil.getKeyword(image,'RA_TARG'))
    dec_targ = float(fileutil.getKeyword(image,'DEC_TARG'))
    
    # Get the velocity aberration keyword
    _vafac = fileutil.getKeyword(image,'VAFACTOR')
    
    if _vafac: VA_fac = float(_vafac)
    else: VA_fac = 1.0

    if not quiet:
        print '-VA_factor: ',VA_fac

    detector = fileutil.getKeyword(image,'DETECTOR')

    if instrument == 'WFPC2':
        filter1 = fileutil.getKeyword(image,'FILTNAM1')
        filter2 = fileutil.getKeyword(image,'FILTNAM2')
    else:
        filter1 = fileutil.getKeyword(image,'FILTER1')
        filter2 = fileutil.getKeyword(image,'FILTER2')

    if filter1 == None: filter1 = 'CLEAR'
    else: filter1 = filter1.strip()
    if filter2 == None: filter2 = 'CLEAR'
    else: filter2 = filter2.strip()
    if filter1.find('CLEAR') == 0: filter1 = 'CLEAR'
    if filter2.find('CLEAR') == 0: filter2 = 'CLEAR'

    # Set up parity matrix for chip
    if instrument == 'WFPC2':
        parity = PARITY[instrument]
    elif PARITY.has_key(detector):
        parity = PARITY[detector]
    else:
        raise 'Detector ',detector,' Not supported at this time. Exiting...'

    # Get the chip number
    _s = fileutil.getKeyword(image,'CCDCHIP')
    _d = detector
    if _s == None and _d == None:
        chip = 1
    else:
        if _s:
            chip = int(_s)
        elif _d.isdigit():
            chip = int(_d)
        else:
            chip = 1

    if not quiet:
        print "-PA_V3: ",pvt," CHIP #",chip

    # Extract the appropriate information from the IDCTAB
    fx,fy,refpix,order=fileutil.readIDCtab(idctab,chip=chip,
            direction='forward',filter1=filter1,filter2=filter2,
            offtab=offtab, date=dateobs)

    # Also get the entry for reference filter (ACS/WFC only)
    if instrument == 'WFPC2':
        R_filter1=filter1
        R_filter2=filter2
        R_dateobs = '1991-01-01'
    elif instrument == 'ACS':
        if detector != 'SBC':
            R_filter1='F475W'
            R_filter2='CLEAR2L'
            R_dateobs = dateobs
        else:
            R_filter1 = 'F150LP'
            R_filter2 = 'N/A'
            R_dateobs = dateobs
    else:
        # Default to using same reference as input
        R_filter1 = filter1
        R_filter2 = filter2
        R_dateobs = dateobs

    R_fx,R_fy,R_refpix,R_order=fileutil.readIDCtab(idctab,chip=chip,
            direction='forward',filter1=R_filter1,filter2=R_filter2,
            offtab=offtab, date=R_dateobs)

    if not quiet:
        print "For ",filter1," v2/v3_ref: ",refpix['V2REF'],refpix['V3REF']
        print "For ",R_filter1," v2/v3_ref: ",R_refpix['V2REF'],R_refpix['V3REF']


    # Convert the PA_V3 orientation to the orientation at the aperture
    pv = _troll(pvt,dec,refpix['V2REF'],refpix['V3REF'])
    if refpix['THETA']:
        pv += refpix['THETA']

    # Get the current WCS from the image header
    oldcrval1=fileutil.getKeyword(image,'CRVAL1')
    oldcrval2=fileutil.getKeyword(image,'CRVAL2')
    oldcrpix1=fileutil.getKeyword(image,'CRPIX1')
    oldcrpix2=fileutil.getKeyword(image,'CRPIX2')
    oldcd11=fileutil.getKeyword(image,'CD1_1')
    oldcd12=fileutil.getKeyword(image,'CD1_2')
    oldcd21=fileutil.getKeyword(image,'CD2_1')
    oldcd22=fileutil.getKeyword(image,'CD2_2')
    oldorient=fileutil.getKeyword(image,'ORIENTAT')

    if not quiet:
        print "  Scale (arcsec/pix): ",refpix['PSCALE']

    # Update the CRVALs for the filter dependent shift
    dV2 = refpix['V2REF'] - R_refpix['V2REF']
    dV3 = refpix['V3REF'] - R_refpix['V3REF']

    # Get radian values for trig functions
    pvRad = pv*pi/180.0
    decRad = dec*pi/180.0

    # These are in arcseconds
    dEast = dV2 * cos(pvRad) + dV3 * sin(pvRad)
    dNorth = -dV2 * sin(pvRad) + dV3 * cos(pvRad)

    # Apply corrections and convert to degrees
    alpha = alpha + dEast/3600. / cos(decRad)
    dec = dec + dNorth/3600.

    if not quiet:
        print ' Filter shift: CRVAL1 position from ',oldcrval1,' to ',alpha
        print ' Alpha shift (arcsecs)= ',(alpha-float(oldcrval1))*3600.*cos(decRad)
        print ' Filter Shift: CRVAL2 position from ',oldcrval2,' to ',dec
        print ' Dec shift (arcsecs)= ',(dec-float(oldcrval2))*3600.

    # Extract the Geo matrix from the fx,fy values
    # This is in units of arcsecs
    Geo11=fx[1,1] * parity[0][0]
    Geo12=fx[1,0] * parity[0][0]
    Geo21=fy[1,1] * parity[1][1]
    Geo22=fy[1,0] * parity[1][1]

    # Create the "truth" output CD
    # The pv3 is in degrees and has to be converted to radians
    # The 3600 factor takes us back to degrees at the next step
    Tru11 = cos(pvRad)/3600.
    Tru12 = sin(pvRad)/3600.
    Tru21 = -Tru12
    Tru22 = Tru11

    if not quiet:
        print ' Geo11,12,21,22: ',Geo11,Geo12,Geo21,Geo22
        print ' Tru11,12,21,22: ',Tru11,Tru12,Tru21,Tru22

    # Now, we may need to shift the distortion coefficients if
    # the observation is a subarray taken away from the chip reference
    # position...
    #
    # Start by determining whether there is any offset that can
    # be determined from the header.
    _s = fileutil.getKeyword(image,'LTV1')
    if _s != None:
        ltv1 = float(_s)
    else:
        ltv1=0.0

    _s = fileutil.getKeyword(image,'LTV2')
    if _s != None:
        ltv2 = float(_s)
    else:
        ltv2=0.0

    # Now, compute full chip coordinates of reference pixel
    _s = fileutil.getKeyword(image,'CRPIX1')
    crpix1 = float(_s) - ltv1
    _s = fileutil.getKeyword(image,'CRPIX2')
    crpix2 = float(_s) - ltv2

    naxis1 = float(fileutil.getKeyword(image,'NAXIS1'))
    naxis2 = float(fileutil.getKeyword(image,'NAXIS2'))

    # Finally, compute the offset of observation ref. point from
    # distortion model reference point.
    # Add default value in case of non-existent distortion model
    # reference point should default to image center. WJH 25-Aug-03
    if refpix['XREF'] == None: refpix['XREF'] = naxis1/2.
    if refpix['YREF'] == None: refpix['YREF'] = naxis2/2.

    xs = refpix['XREF'] - crpix1
    ys = refpix['YREF'] - crpix2

    # Shift the distortion coefficients accordingly...
    fx,fy = shift_coeffs(fx,fy,xs,ys,order)

    # Multiply the geometric distortion (linear part) matrix by the
    # truth - also convert to a string to constrain the format
    #
    # Note that the VA_fac is included here in UPINWCS
    In_tru11 = string.upper("%13.9e" % (VA_fac*(Tru11*Geo11 + Tru12*Geo21)))
    In_tru12 = string.upper("%13.9e" % (VA_fac*(Tru11*Geo12 + Tru12*Geo22)))
    In_tru21 = string.upper("%13.9e" % (VA_fac*(Tru21*Geo11 + Tru22*Geo21)))
    In_tru22 = string.upper("%13.9e" % (VA_fac*(Tru21*Geo12 + Tru22*Geo22)))

    # The CRVALs are also moved apart
    New_CRVAL1 = string.upper("%13.9e" % (ra_targ + VA_fac*(alpha - ra_targ)))
    New_CRVAL2 = string.upper("%13.9e" % (dec_targ + VA_fac*(dec - dec_targ)))
    In_orient = string.upper("%13.9g" % (atan2(float(In_tru12),float(In_tru22))*180./pi))

    if not quiet:
        print ' Computed new ORIENTAT of ',In_orient
        print ' Shifting CRVAL1 position from ',alpha,' to ',(ra_targ + VA_fac*(alpha - ra_targ))
        print ' Shifting CRVAL2 position from ',dec,' to ',(dec_targ + VA_fac*(dec - dec_targ))

    # Copy the old CD values to the header with new names
    yes = True
    no = False

    # Open image as PyFITS object and add/update keywords as necessary
    _fname,_extn = fileutil.parseFilename(image)
    _fimg = fileutil.openImage(image,memmap=0,mode='update')
    _hdu = fileutil.getExtn(_fimg,_extn)

    # Check for copies of the original WCS
    #_s = fileutil.getKeyword(image,'OCD1_1')
    #if _s != None:
    if _hdu.header.has_key('OCD1_1'):
        print "Note - older copies of OCD1_1 etc exist and are being updated"

    # Write a full header
    if not quiet:
        print "-Copying the old WCS to OCD1_1 etc..."

    _hdu.header.update('OCRVAL1',float(oldcrval1),comment='Original CRVAL1')
    _hdu.header.update('OCRVAL2',float(oldcrval2),comment='Original CRVAL2')
    _hdu.header.update('OCRPIX1',float(oldcrpix1),comment='Original CRPIX1')
    _hdu.header.update('OCRPIX2',float(oldcrpix2),comment='Original CRPIX2')

    _hdu.header.update('OCD1_1',float(oldcd11),comment='Original CD1_1')
    _hdu.header.update('OCD1_2',float(oldcd12),comment='Original CD1_2')
    _hdu.header.update('OCD2_1',float(oldcd21),comment='Original CD2_1')
    _hdu.header.update('OCD2_2',float(oldcd22),comment='Original CD2_2')

    _hdu.header.update('O_ORIENT',float(oldorient),comment='Original ORIENTAT')

    if not quiet:
        print "-Updating the image header WCS with the new values..."

    # Write the new header keywords, CRPIX doesn't change
    _hdu.header.update('CRVAL1',float(New_CRVAL1))
    _hdu.header.update('CRVAL2',float(New_CRVAL2))
    _hdu.header.update('CD1_1',float(In_tru11))
    _hdu.header.update('CD1_2',float(In_tru12))
    _hdu.header.update('CD2_1',float(In_tru21))
    _hdu.header.update('CD2_2',float(In_tru22))
    _hdu.header.update('ORIENTAT',float(In_orient))

    # Add some information about the UPINWCS run
    _hdu.header.update('UPWVER','Header updated by UPINWCS V'+__version__)
    _hdu.header.update('UPWTIM','Header updated at '+time.asctime())

    # Set flag
    _hdu.header.update('WCSCORR','DONE',comment='Flag for use of UPINCD')
    _fimg.close()


def shift_coeffs(cx,cy,xs,ys,norder):
    """
    Shift reference position of coefficients to new center
    where (xs,ys) = old-reference-position - subarray/image center.
    This will support creating coeffs files for drizzle which will
    be applied relative to the center of the image, rather than relative
    to the reference position of the chip.

    Derived directly from PyDrizzle V3.3d.
    """

    _cxs = N.zeros(shape=cx.shape,type=cx.type())
    _cys = N.zeros(shape=cy.shape,type=cy.type())
    _k = norder + 1

    # loop over each input coefficient
    for m in xrange(_k):
        for n in xrange(_k):
            if m >= n:
                # For this coefficient, shift by xs/ys.
                _ilist = N.array(range(_k - m)) + m
                # sum from m to k
                for i in _ilist:
                    _jlist = N.array(range( i - (m-n) - n + 1)) + n
                    # sum from n to i-(m-n)
                    for j in _jlist:
                        _cxs[m,n] = _cxs[m,n] + cx[i,j]*combin(j,n)*combin((i-j),(m-n))*pow(xs,(j-n))*pow(ys,((i-j)-(m-n)))
                        _cys[m,n] = _cys[m,n] + cy[i,j]*combin(j,n)*combin((i-j),(m-n))*pow(xs,(j-n))*pow(ys,((i-j)-(m-n)))
    _cxs[0,0] = _cxs[0,0] - xs
    _cys[0,0] = _cys[0,0] - ys
    #_cxs[0,0] = 0.
    #_cys[0,0] = 0.

    return _cxs,_cys

def _troll(roll, dec, v2, v3):
    """ Computes the roll angle at the target position based on:
            the roll angle at the V1 axis(roll),
            the dec of the target(dec), and
            the V2/V3 position of the aperture (v2,v3) in arcseconds.

        Based on the algorithm provided by Colin Cox that is used in
        Generic Conversion at STScI.
    """
    # Convert all angles to radians
    _roll = wcsutil.DEGTORAD(roll)
    _dec = wcsutil.DEGTORAD(dec)
    _v2 = wcsutil.DEGTORAD(v2 / 3600.)
    _v3 = wcsutil.DEGTORAD(v3 / 3600.)

    # compute components
    sin_rho = sqrt((pow(sin(_v2),2)+pow(sin(_v3),2)) - (pow(sin(_v2),2)*pow(sin(_v3),2)))
    rho = asin(sin_rho)
    beta = asin(sin(_v3)/sin_rho)
    if _v2 < 0: beta = pi - beta
    gamma = asin(sin(_v2)/sin_rho)
    if _v3 < 0: gamma = pi - gamma
    A = pi/2. + _roll - beta
    B = atan2( sin(A)*cos(_dec), (sin(_dec)*sin_rho - cos(_dec)*cos(rho)*cos(A)))

    # compute final value
    troll = wcsutil.RADTODEG(pi - (gamma+B))

    return troll

def restoreWCS(image):
    """ Restores the original WCS values.
    """
    _WCS_KEYWORDS = ['CRVAL1','CRVAL2','CRPIX1','CRPIX2','CD1_1','CD1_2','CD2_1','CD2_2']
    _ocd = {}
    
    for keyword in _WCS_KEYWORDS:
        _ocd[keyword] = fileutil.getKeyword(image,'O'+keyword)
        
    for val in _ocd.values():
        if val == None:
            print 'No original WCS values found. Exiting...'
            return

    # Open image as PyFITS object and add/update keywords as necessary
    _fname,_extn = fileutil.parseFilename(image)
    _fimg = fileutil.openImage(image,memmap=0,mode='update')
    _handle = fileutil.getExtn(_fimg,_extn)
    
    for keyword in _WCS_KEYWORDS:
#        iraf.hedit(image,keyword,_ocd[keyword],add=no,verify=no,show=no)
        _handle.header[keyword] = _ocd[keyword]

    # Set the flag to allow the task to be re-run
#    iraf.hedit(image,'WCSCORR','PERFORM',add=no,verify=no,show=no)
    _handle.header['WCSCORR'] = 'PERFORM'

    _fimg.close()


def help():
    _help_str = """ upincd - a task for updating an image header WCS to make
          it consistent with the distortion model and velocity aberration.

    This task will read in a distortion model from the IDCTAB and generate
    a new WCS matrix based on the value of ORIENTAT.  It will support subarrays
    by shifting the distortion coefficients to image reference position before
    applying them to create the new WCS, including velocity aberration.
    Original WCS values will be moved to an O* keywords (OCD1_1,...).
    Currently, this task will only support ACS and WFPC2 observations.

    Syntax:
        run(image,quiet=False)
    where
        image - either a single image with extension specified,
                or a substring common to all desired image names,
                or a wildcarded filename
                or '@file' where file is a file containing a list of images
        quiet - turns off ALL reporting messages: 'True' or 'False'(default)
    Usage:
        --> import upinwcs
        --> upinwcs.run('raw') # This will update all _raw files in directory
        --> upinwcs.run('j8gl03igq_raw.fits[sci,1]')
    """
    print _help_str
