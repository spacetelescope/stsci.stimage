"""
Utility functions for PyDrizzle that rely on PyRAF's interface to IRAF
tasks.
"""
#
# Revision History:
#   Nov 2001: Added function (getLTVOffsets) for reading subarray offsets
#               from extension header's LTV keywords.  WJH
#   Mar 2002: Restructured to only contain IRAF based functions.
#   20-Sept-2002:  Replaced all calls to 'keypar' with calls to 'hselect'.
#               This eliminates any parameter writing, making it safer for
#               pipeline/multi-process use.
#   24-Sept-2003:  Replaced all calls to hselect with calls to
#                   fileutil.getKeyword() to remove dependencies on IRAF.
#                   Also, replaced 'yes' and 'no' with Python Bool vars.
#

import string,os,types
#import iraf

import numarray as N
from numarray import linear_algebra
#import LinearAlgebra2 # required for matrix inversion in fitlin

import fileutil

# Convenience definitions
DEGTORAD = fileutil.DEGTORAD

no = False
yes = True

# Constants
IDCTAB  = 1
DRIZZLE = 2
TRAUGER = 3

try:
    DEFAULT_IDCDIR = fileutil.osfn('stsdas$pkg/analysis/dither/drizzle/coeffs/')
except:
    DEFAULT_IDCDIR = os.getcwd()


def factorial(n):
    """ Compute a factorial for integer n. """
    m = 1
    for i in range(int(n)):
        m = m * (i+1)
    return m

def combin(j,n):
    """ Return the combinatorial factor for j in n."""
    return (factorial(j) / (factorial(n) * factorial( (j-n) ) ) )


#################
#
#
#               Utility Functions based on IRAF
#
#
#################
def findNumExt(filename):
    # Open the file given by 'rootname' and return the
    # number of extensions written out for the observation.
    # It will be up to the Exposure classes to determine how
    # many IMSETS they have to work with.
    #
    # Only look in the primary extension or first group.
    #_s = iraf.hselect(filename,'NEXTEND',expr='yes',Stdout=1)
    _s = fileutil.getKeyword(filename,keyword='NEXTEND')
    if not _s:
        _s = fileutil.getKeyword(filename,keyword='GCOUNT')
    # This may need to be changed to support simple image without
    # extensions (such as simple FITS images).
    if _s == '':
        raise ValueError,"There are NO extensions to be read in this image!"

    return _s

# Utility function to extract subarray offsets
# from LTV keywords.  Could be expanded to use
# different keywords for some cases.
# Added: 7-Nov-2001 WJH
def getLTVOffsets(rootname,header=None):

    _ltv1 = None
    _ltv2 = None
    if header:
        if header.has_key('LTV1'):
            _ltv1 = header['LTV1']
        if header.has_key('LTV2'):
            _ltv2 = header['LTV2']
    else:
        _ltv1 = fileutil.getKeyword(rootname,'LTV1')
        _ltv2 = fileutil.getKeyword(rootname,'LTV2')

    if _ltv1 == None: _ltv1 = 0.
    if _ltv2 == None: _ltv2 = 0.

    return _ltv1,_ltv2

def getChipId(header):

    if header.has_key('CCDCHIP'):
        chip = int(header['CCDCHIP'])
    elif header.has_key('DETECTOR') and str(header['DETECTOR']).isdigit():
        chip = int(header['DETECTOR'])
    elif header.has_key('CAMERA') and str(header['CAMERA']).isdigit():
        chip = int(header['CAMERA'])
    else:
        chip = 1

    return chip


def getIDCFile(image,keyword=None,directory=None):
    # Open the primary header of the file and read the name of
    # the IDCTAB.
    # Parameters:
    #   header -  primary and extension header object to read IDCTAB info
    #
    #   keyword(optional) - header keyword with name of IDCTAB
    #            --OR-- 'HEADER' if need to build IDCTAB name from scratch
    #           (default value: 'IDCTAB')
    #   directory(optional) - directory with default drizzle coeffs tables
    #                   (default value: 'drizzle$coeffs')
    #
    # This function needs to be generalized to support
    # keyword='HEADER', as would be the case for WFPC2 data.
    #


    if isinstance(image, types.StringType):
        # We were provided an image name, so read in the header...
        header = fileutil.getHeader(image)
    else:
        # otherwise, we were provided an image header we can work with directly
        header = image

    if keyword.lower() == 'header':
        idcfile,idctype = __getIDCTAB(header)
        if (idcfile == None):
            idcfile,idctype = __buildIDCTAB(header,directory)

    elif string.lower(keyword) == 'idctab':
        # keyword specifies header keyword with IDCTAB name
        idcfile,idctype = __getIDCTAB(header)

    elif keyword == '':
        idcfile = None
        idctype = None
    else:
        # Need to build IDCTAB filename from scratch
        idcfile,idctype = __buildIDCTAB(header,directory,kw = keyword)

    # Account for possible absence of IDCTAB name in header
    if idcfile == 'N/A':
        idcfile = None

    if idcfile != None and idcfile != '':
        # Now we need to recursively expand any IRAF symbols to full paths...
        idcfile = fileutil.osfn(idcfile)

    if idcfile == None:
        print 'WARNING: No valid distortion coefficients available!'
        print 'Using default unshifted, unscaled, unrotated model.'

    return idcfile,idctype


def __buildIDCTAB(header, directory, kw = None):
    # Need to build IDCTAB filename from scratch
    instrument = header['INSTRUME']
    if instrument != 'NICMOS':
        detector = header['DETECTOR']
    else:
        detector = str(header['CAMERA'])

    # Default non-IDCTAB distortion model
    if (kw == None):
        keyword = 'cubic'
    else :
        keyword = kw

    if not directory:
        default_dir = DEFAULT_IDCDIR
    else:
        default_dir = directory

    if instrument == 'WFPC2':
        if detector == 1:
            detname = 'pc'
        else:
            detname = 'wf'
        idcfile = default_dir+detname+str(detector)+'-'+string.lower(keyword)

    elif instrument == 'STIS':
        idcfile = default_dir+'stis-'+string.lower(detector)

    elif instrument == 'NICMOS':
        if detector != None:
            idcfile = default_dir+'nic-'+detector
        else:
            idcfile = None
    else:
        idcfile = None

    idctype = getIDCFileType(fileutil.osfn(idcfile))

    return idcfile,idctype

def __getIDCTAB(header):
    # keyword specifies header keyword with IDCTAB name
    try:
        idcfile = header['idctab']
    except:
        print 'Warning: No IDCTAB specified in header!'
        idcfile = None

    return idcfile,'idctab'

def getIDCFileType(idcfile):
    """ Open ASCII IDCFILE to determine the type: cubic,trauger,... """
    if idcfile == None:
        return None

    ifile = open(idcfile,'r')
    # Search for the first line of the coefficients
    _line = fileutil.rAsciiLine(ifile)

    # Search for first non-commented line...
    while _line[0] == '#':
        _line = fileutil.rAsciiLine(ifile)

    _type = string.rstrip(_line.lower())

    if _type in ['cubic','quartic','quintic'] or _type.find('poly') > -1:
        _type = 'cubic'
    elif _type == 'trauger':
        _type = 'trauger'
    else:
        _type = None

    ifile.close()
    del ifile

    return _type

#
# Function to read in coefficients from ASCII Cubic Drizzle
# Coefficients table.
#

def readCubicTable(idcfile):
    # Assumption: this will only be used for cubic file...
    order = 3
    # Also, this function does NOT perform any scaling on
    # the coefficients, it simply passes along what is found
    # in the file as is...

    # Return a default geometry model if no coefficients filename
    # is given.  This model will not distort the data in any way.
    if idcfile == None:
        return fileutil.defaultModel()

    ifile = open(idcfile,'r')
    # Search for the first line of the coefficients
    _line = fileutil.rAsciiLine(ifile)

    _found = no
    while _found == no:
        if _line[:7] in  ['cubic','quartic','quintic'] or _line[:4] == 'poly':
            found = yes
            break
        _line = fileutil.rAsciiLine(ifile)

    # Read in each row of coefficients, without line breaks or newlines
    # split them into their values, and create a list for A coefficients
    # and another list for the B coefficients
    _line = fileutil.rAsciiLine(ifile)
    a_coeffs = string.split(_line)

    x0 = float(a_coeffs[0])
    _line = fileutil.rAsciiLine(ifile)
    a_coeffs[len(a_coeffs):] = string.split(_line)
    # Scale coefficients for use within PyDrizzle
    for i in range(len(a_coeffs)):
        a_coeffs[i] = float(a_coeffs[i])

    _line = fileutil.rAsciiLine(ifile)
    b_coeffs = string.split(_line)
    y0 = float(b_coeffs[0])
    _line = fileutil.rAsciiLine(ifile)
    b_coeffs[len(b_coeffs):] = string.split(_line)
    # Scale coefficients for use within PyDrizzle
    for i in range(len(b_coeffs)):
        b_coeffs[i] = float(b_coeffs[i])

    ifile.close()
    del ifile
    # Now, convert the coefficients into a Numeric array
    # with the right coefficients in the right place.
    # Populate output values now...
    fx = N.zeros(shape=(order+1,order+1),type=N.Float64)
    fy = N.zeros(shape=(order+1,order+1),type=N.Float64)
    # Assign the coefficients to their array positions
    fx[0,0] = 0.
    fx[1] = N.array([a_coeffs[2],a_coeffs[1],0.,0.],type=N.Float64)
    fx[2] = N.array([a_coeffs[5],a_coeffs[4],a_coeffs[3],0.],type=N.Float64)
    fx[3] = N.array([a_coeffs[9],a_coeffs[8],a_coeffs[7],a_coeffs[6]],type=N.Float64)
    fy[0,0] = 0.
    fy[1] = N.array([b_coeffs[2],b_coeffs[1],0.,0.],type=N.Float64)
    fy[2] = N.array([b_coeffs[5],b_coeffs[4],b_coeffs[3],0.],type=N.Float64)
    fy[3] = N.array([b_coeffs[9],b_coeffs[8],b_coeffs[7],b_coeffs[6]],type=N.Float64)

    # Used in Pattern.computeOffsets()
    refpix = {}
    refpix['XREF'] = None
    refpix['YREF'] = None
    refpix['V2REF'] = x0
    refpix['V3REF'] = y0
    refpix['XDELTA'] = 0.
    refpix['YDELTA'] = 0.
    refpix['PSCALE'] = None
    refpix['DEFAULT_SCALE'] = no
    refpix['centered'] = yes

    return fx,fy,refpix,order

# Function to compute the index of refraction for MgF2 at
# the specified wavelength for use with Trauger coefficients
def _MgF2(lam):
    _sig = pow((1.0e7/lam),2)
    return N.sqrt(1.0 + 2.590355e10/(5.312993e10-_sig) +
        4.4543708e9/(11.17083e9-_sig) + 4.0838897e5/(1.766361e5-_sig))

#
# Function to read Trauger ASCII file and return cubic coefficients
#
def readTraugerTable(idcfile,wavelength):

    # Return a default geometry model if no coefficients filename
    # is given.  This model will not distort the data in any way.
    if idcfile == None:
        return fileutil.defaultModel()

    # Trauger coefficients only result in a cubic file...
    order = 3
    numco = 10
    a_coeffs = [0] * numco
    b_coeffs = [0] * numco
    indx = _MgF2(wavelength)

    ifile = open(idcfile,'r')
    # Search for the first line of the coefficients
    _line = fileutil.rAsciiLine(ifile)
    while string.lower(_line[:7]) != 'trauger':
        _line = fileutil.rAsciiLine(ifile)
    # Read in each row of coefficients,split them into their values,
    # and convert them into cubic coefficients based on
    # index of refraction value for the given wavelength
    # Build X coefficients from first 10 rows of Trauger coefficients
    j = 0
    while j < 20:
        _line = fileutil.rAsciiLine(ifile)
        if _line == '': continue
        _lc = string.split(_line)
        if j < 10:
            a_coeffs[j] = float(_lc[0])+float(_lc[1])*(indx-1.5)+float(_lc[2])*(indx-1.5)**2
        else:
            b_coeffs[j-10] = float(_lc[0])+float(_lc[1])*(indx-1.5)+float(_lc[2])*(indx-1.5)**2
        j = j + 1

    ifile.close()
    del ifile

    # Now, convert the coefficients into a Numeric array
    # with the right coefficients in the right place.
    # Populate output values now...
    fx = N.zeros(shape=(order+1,order+1),type=N.Float64)
    fy = N.zeros(shape=(order+1,order+1),type=N.Float64)
    # Assign the coefficients to their array positions
    fx[0,0] = 0.
    fx[1] = N.array([a_coeffs[2],a_coeffs[1],0.,0.],type=N.Float64)
    fx[2] = N.array([a_coeffs[5],a_coeffs[4],a_coeffs[3],0.],type=N.Float64)
    fx[3] = N.array([a_coeffs[9],a_coeffs[8],a_coeffs[7],a_coeffs[6]],type=N.Float64)
    fy[0,0] = 0.
    fy[1] = N.array([b_coeffs[2],b_coeffs[1],0.,0.],type=N.Float64)
    fy[2] = N.array([b_coeffs[5],b_coeffs[4],b_coeffs[3],0.],type=N.Float64)
    fy[3] = N.array([b_coeffs[9],b_coeffs[8],b_coeffs[7],b_coeffs[6]],type=N.Float64)

    # Used in Pattern.computeOffsets()
    refpix = {}
    refpix['XREF'] = None
    refpix['YREF'] = None
    refpix['V2REF'] = None
    refpix['V3REF'] = None
    refpix['XDELTA'] = 0.
    refpix['YDELTA'] = 0.
    refpix['PSCALE'] = None
    refpix['DEFAULT_SCALE'] = no
    refpix['centered'] = yes

    return fx,fy,refpix,order

def rotateCubic(fxy,theta):
    # This function transforms cubic coefficients so that
    # they calculate pixel positions oriented by theta (the same
    # orientation as the PC).
    # Parameters: fxy - cubic-coefficients Numeric array
    #               theta - angle to rotate coefficients
    # Returns new array with same order as 'fxy'
    #
    # Set up some simplifications
    newf = fxy * 0.
    cost = N.cos(DEGTORAD(theta))
    sint = N.sin(DEGTORAD(theta))
    cos2t = pow(cost,2)
    sin2t = pow(sint,2)
    cos3t = pow(cost,3)
    sin3t = pow(sint,3)

    # Now compute the new coefficients
    newf[1][1] = fxy[1][1] * cost - fxy[1][0] * sint
    newf[1][0] = fxy[1][1] * sint + fxy[1][0] * cost

    newf[2][2] = fxy[2][2] * cos2t - fxy[2][1] * cost * sint + fxy[2][0] * sin2t
    newf[2][1] = fxy[2][2] * 2 * cost * sint + fxy[2][1] * (cos2t - sin2t) + fxy[2][0] * 2 * sint * cost
    newf[2][0] = fxy[2][2] * sin2t + fxy[2][1] * cost * sint + fxy[2][0] * cos2t

    newf[3][3] = fxy[3][3] * cos3t - fxy[3][2] * sint * cos2t + fxy[3][1] * sin2t * cost - fxy[3][0] * sin3t
    newf[3][2] = fxy[3][3] * 3. * cos2t * sint + fxy[3][2]* (cos3t - 2. * sin2t * cost) +fxy[3][1] * (sin3t + 2 * sint * cos2t) - fxy[3][0] * sin2t * cost
    newf[3][1] = fxy[3][3] * 3. * cost * sin2t + fxy[3][2] *(2.*cos2t*sint - sin3t) + fxy[3][1] * (2 * sin2t * cost + cos3t) + fxy[3][0] * sint * cos2t
    newf[3][0] = fxy[3][3] * sin3t + fxy[3][2] * sin2t * cost + fxy[3][1] * sint * cos2t + fxy[3][0] * cos3t

    return newf


def buildRotMatrix(theta):
    _theta = DEGTORAD(theta)
    _mrot = N.zeros(shape=(2,2),type=N.Float64)
    _mrot[0] = (N.cos(_theta),N.sin(_theta))
    _mrot[1] = (-N.sin(_theta),N.cos(_theta))

    return _mrot

def rotatePos(pos, theta,offset=None,scale=None):

    if scale == None:
        scale = 1.

    if offset == None:
        offset = N.array([0.,0.],type=N.Float64)
    mrot = buildRotMatrix(theta)
    xr = ((pos[0] * mrot[0][0]) + (pos[1]*mrot[0][1]) )/ scale + offset[0]
    yr = ((pos[0] * mrot[1][0]) + (pos[1]*mrot[1][1]) )/ scale + offset[1]

    return xr,yr

# Function for determining the positions of the image
# corners after applying the geometry model.
# Returns a dictionary with the limits and size of the
# image.
def getRange(members,ref_wcs,verbose=None):
    xma,yma = [],[]
    xmi,ymi = [],[]

    # Compute corrected positions of each chip's common point
    crpix = (ref_wcs.crpix1,ref_wcs.crpix2)
    ref_rot = ref_wcs.orient
    _rot = ref_wcs.orient - members[0].geometry.wcslin.orient

    for member in members:
        # Need to make sure this is populated with proper defaults
        # for ALL exposures!
        _model = member.geometry.model
        _wcs = member.geometry.wcs
        _wcslin = member.geometry.wcslin
        _theta = _wcslin.orient - ref_rot

        # Need to scale corner positions to final output scale
        _scale =_wcslin.pscale/ ref_wcs.pscale

        # Compute the corrected,scaled corner positions for each chip
        #xyedge = member.calcNewEdges(pscale=ref_wcs.pscale)
        xypos = member.geometry.calcNewCorners() * _scale

        if _theta != 0.0:
            #rotate coordinates to match output orientation
            # Now, rotate new coord
            _mrot = buildRotMatrix(_theta)
            xypos = N.dot(xypos,_mrot)

        _oxmax = N.maximum.reduce(xypos[:,0])
        _oymax = N.maximum.reduce(xypos[:,1])
        _oxmin = N.minimum.reduce(xypos[:,0])
        _oymin = N.minimum.reduce(xypos[:,1])

        # Update the corners attribute of the member with the
        # positions of the computed, distortion-corrected corners
        #member.corners['corrected'] = N.array([(_oxmin,_oymin),(_oxmin,_oymax),(_oxmax,_oymin),(_oxmax,_oymax)],type=N.Float64)
        member.corners['corrected'] = xypos

        xma.append(_oxmax)
        yma.append(_oymax)
        xmi.append(_oxmin)
        ymi.append(_oymin)

    # Determine the full size of the metachip
    xmax = N.maximum.reduce(xma)
    ymax = N.maximum.reduce(yma)
    ymin = N.minimum.reduce(ymi)
    xmin = N.minimum.reduce(xmi)

    # Compute offset from center that distortion correction shifts the image.
    # This accounts for the fact that the output is no longer symmetric around
    # the reference position...
    # Scale by ratio of plate-scales so that DELTAs are always in input frame
    #
    _ratio = ref_wcs.pscale / _wcslin.pscale

    nref = ( (xmin + xmax)*_ratio, (ymin + ymax)*_ratio )
    if _rot != 0.:
        _mrot = buildRotMatrix(_rot)
        nref = N.matrixmultiply(nref,_mrot)

    # Now, compute overall size based on range of pixels and offset from center.
    #xsize = int(xmax - xmin + nref[0])
    #ysize = int(ymax - ymin + nref[1])
    # Add '2' to each dimension to allow for fractional pixels at the
    # edge of the image. Also, 'drizzle' centers the output, so
    # adding 2 only expands image by 1 row on each edge.
    # An additional two is added to accomodate floating point errors in drizzle.
    xsize = int(xmax - xmin) + 2
    ysize = int(ymax - ymin) + 2

    meta_range = {}
    meta_range = {'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'nref':nref}
    meta_range['xsize'] = xsize
    meta_range['ysize'] = ysize

    if verbose:
        print 'Meta_WCS:'
        print '    NREF         :',nref
        print '    X range      :',xmin,xmax
        print '    Y range      :',ymin,ymax
        print '    Computed Size: ',xsize,ysize

    return meta_range

def computeRange(corners):
    """ Determine the range spanned by an array of pixel positions. """
    _xrange = (N.minimum.reduce(corners[:,0]),N.maximum.reduce(corners[:,0]))
    _yrange = (N.minimum.reduce(corners[:,1]),N.maximum.reduce(corners[:,1]))
    return _xrange,_yrange

def convertWCS(inwcs,drizwcs):
    """ Copy WCSObject WCS into Drizzle compatible array."""
    drizwcs[0] = inwcs.crpix1
    drizwcs[1] = inwcs.crval1
    drizwcs[2] = inwcs.crpix2
    drizwcs[3] = inwcs.crval2
    drizwcs[4] = inwcs.cd11
    drizwcs[5] = inwcs.cd21
    drizwcs[6] = inwcs.cd12
    drizwcs[7] = inwcs.cd22

    return drizwcs

def updateWCS(drizwcs,inwcs):
    """ Copy output WCS array from Drizzle into WCSObject."""
    inwcs.crpix1   = drizwcs[0]
    inwcs.crval1   = drizwcs[1]
    inwcs.crpix2   = drizwcs[2]
    inwcs.crval2   = drizwcs[3]
    inwcs.cd11     = drizwcs[4]
    inwcs.cd21     = drizwcs[5]
    inwcs.cd12     = drizwcs[6]
    inwcs.cd22     = drizwcs[7]
    inwcs.pscale = N.sqrt(N.power(inwcs.cd11,2)+N.power(inwcs.cd21,2)) * 3600.
    inwcs.orient = N.arctan2(inwcs.cd12,inwcs.cd22) * 180./N.pi

def wcsfit(img_geom, ref):
    """
    Perform a linear fit between 2 WCS for shift, rotation and scale.
    Based on 'WCSLIN' from 'drutil.f'(Drizzle V2.9) and modified to
    allow for differences in reference positions assumed by PyDrizzle's
    distortion model and the coeffs used by 'drizzle'.

    Parameters:
        img      - Exposure instance for input image
        ref_wcs  - Undistorted WCSObject instance for output frame
    """
    # Define objects that we need to use for the fit...
    img_wcs = img_geom.wcs
    in_refpix = img_geom.model.refpix

    # Only work on a copy to avoid unexpected behavior in the
    # call routine...
    ref_wcs = ref.copy()

    # Convert the RA/Dec positions back to X/Y in output product image
    _cpix_xyref = N.zeros((4,2),type=N.Float64)

    # Start by setting up an array of points +/-0.5 pixels around CRVAL1,2
    # However, we must shift these positions by 1.0pix to match what
    # drizzle will use as its reference position for 'align=center'.
    _cpix = (img_wcs.crpix1,img_wcs.crpix2)
    _cpix_arr = N.array([_cpix,(_cpix[0],_cpix[1]+1.),
                       (_cpix[0]+1.,_cpix[1]+1.),(_cpix[0]+1.,_cpix[1])], type=N.Float64)

    # Convert these positions to RA/Dec
    _cpix_rd = img_wcs.xy2rd(_cpix_arr)

    for pix in xrange(len(_cpix_rd[0])):
        _cpix_xyref[pix,0],_cpix_xyref[pix,1] = ref_wcs.rd2xy((_cpix_rd[0][pix],_cpix_rd[1][pix]))

    # Now, apply distortion model to input image XY positions
    _cpix_xyc = N.zeros((4,2),type=N.Float64)
    _cpix_xyc[:,0],_cpix_xyc[:,1] = img_geom.apply(_cpix_arr)

    if in_refpix:
        _cpix_xyc += (in_refpix['XDELTA'], in_refpix['YDELTA'])

    # Perform a fit between:
    #       - undistorted, input positions: _cpix_xyc
    #       - X/Y positions in reference frame: _cpix_xyref
    abxt,cdyt = fitlin(_cpix_xyc,_cpix_xyref)

    # We need to subtract 1 from CRPIX value here to account for
    # offset introduced in 'drizzle' with 'align=center'
    abxt[2] -= ref_wcs.crpix1 - 1.0
    cdyt[2] -= ref_wcs.crpix2 - 1.0

    return abxt,cdyt


def fitlin(imgarr,refarr):
    """ Compute the least-squares fit between two arrays.
        A Python translation of 'FITLIN' from 'drutil.f' (Drizzle V2.9).
    """
    # Initialize variables
    _mat = N.zeros((3,3),type=N.Float64)
    _xorg = imgarr[0][0]
    _yorg = imgarr[0][1]
    _xoorg = refarr[0][0]
    _yoorg = refarr[0][1]
    _sigxox = 0.
    _sigxoy = 0.
    _sigxo = 0.
    _sigyox = 0.
    _sigyoy = 0.
    _sigyo = 0.

    _npos = len(imgarr)
    # Populate matrices
    for i in xrange(_npos):
        _mat[0][0] += N.power((imgarr[i][0] - _xorg),2)
        _mat[0][1] += (imgarr[i][0] - _xorg) * (imgarr[i][1] - _yorg)
        _mat[0][2] += (imgarr[i][0] - _xorg)
        _mat[1][1] += N.power((imgarr[i][1] - _yorg),2)
        _mat[1][2] += imgarr[i][1] - _yorg

        _sigxox += (refarr[i][0] - _xoorg)*(imgarr[i][0] - _xorg)
        _sigxoy += (refarr[i][0] - _xoorg)*(imgarr[i][1] - _yorg)
        _sigxo += refarr[i][0] - _xoorg
        _sigyox += (refarr[i][1] - _yoorg)*(imgarr[i][0] -_xorg)
        _sigyoy += (refarr[i][1] - _yoorg)*(imgarr[i][1] - _yorg)
        _sigyo += refarr[i][1] - _yoorg

    _mat[2][2] = _npos
    _mat[1][0] = _mat[0][1]
    _mat[2][0] = _mat[0][2]
    _mat[2][1] = _mat[1][2]

    # Now invert this matrix
    _mat = linear_algebra.inverse(_mat)

    _a  = _sigxox*_mat[0][0]+_sigxoy*_mat[0][1]+_sigxo*_mat[0][2]
    _b  = -1*(_sigxox*_mat[1][0]+_sigxoy*_mat[1][1]+_sigxo*_mat[1][2])
    #_x0 = _sigxox*_mat[2][0]+_sigxoy*_mat[2][1]+_sigxo*_mat[2][2]

    _c  = _sigyox*_mat[1][0]+_sigyoy*_mat[1][1]+_sigyo*_mat[1][2]
    _d  = _sigyox*_mat[0][0]+_sigyoy*_mat[0][1]+_sigyo*_mat[0][2]
    #_y0 = _sigyox*_mat[2][0]+_sigyoy*_mat[2][1]+_sigyo*_mat[2][2]

    _xt = _xoorg - _a*_xorg+_b*_yorg
    _yt = _yoorg - _d*_xorg-_c*_yorg

    return [_a,_b,_xt],[_c,_d,_yt]

def getRotatedSize(corners,angle):
    """ Determine the size of a rotated (meta)image."""
    # If there is no rotation, simply return original values
    if angle == 0.:
        _corners = corners
    else:
        # Find center
        #_xr,_yr = computeRange(corners)
        #_cen = ( ((_xr[1] - _xr[0])/2.)+_xr[0],((_yr[1]-_yr[0])/2.)+_yr[0])
        _rotm = buildRotMatrix(angle)
        # Rotate about the center
        #_corners = N.dot(corners - _cen,_rotm)
        _corners = N.dot(corners,_rotm)

    return computeRange(_corners)
