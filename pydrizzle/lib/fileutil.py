""" fileutil.py -- General file functions designed for use with PyDrizzle

These functions only rely on booleans 'yes' and 'no', PyFITS and readgeis.

This file contains both PyDrizzle-specific and general functions.
General functions included are:
    DEGTORAD(deg), RADTODEG(rad)
    DIVMOD(num,val)
    convertDate(date)
        Converts the DATE-OBS date string into an integer of the
        number of seconds since 1970.0 using calendar.timegm().

    buildRootname(filename, extn=None, extlist=None)
    buildNewRootname(filename, ext=None)
    getKeyword(filename, keyword, default=None, handle=None)
    getHeader(filename,handle=None)
         Return a copy of the PRIMARY header, along with any group/extension
         header, for this filename specification.
    updateKeyword(filename, key, value)
    openImage(filename,mode='readonly',memmap=0,fitsname=None)
         Opens file and returns PyFITS object.
         It will work on both FITS and GEIS formatted images.

    findFile(input)
    checkFileExists(filename,directory=None)
    removeFile(inlist):
        Utility function for deleting a list of files or a single file.
    rAsciiLine(ifile)
        Returns the next non-blank line in an ASCII file.

    readAsnTable(input,output=None,prodonly=yes)
        Reads an association (ASN) table and interprets inputs and output.
        The 'prodonly' parameter specifies whether to use products as inputs
            or not; where 'prodonly=no' specifies to only use EXP as inputs.
"""
import pyfits, readgeis
import string,os,types,shutil,copy
import calendar
import numarray as N
import time as _time

# Environment variable handling - based on iraffunctions.py
from irafvars import *

yes = True
no = False

# List of supported default file types
# It will look for these file types by default
# when trying to recognize input rootnames.
EXTLIST =  ['_crj.fits','_flt.fits','_sfl.fits','_cal.fits','_raw.fits','.c0h','.hhh','.fits']

def help():
    print __doc__

#################
#
#
#               Generic Functions
#
#
#################
def DEGTORAD(deg):
    return (deg * N.pi / 180.)

def RADTODEG(rad):
    return (rad * 180. / N.pi)

def DIVMOD(num,val):
    if isinstance(num,N.NumArray):
    # Treat number as numarray object
        _num = N.remainder(num,val)
    else:
        _num = divmod(num,val)[1]
    return _num

def getLTime():
    """ Returns a formatted string with the current local time."""
    _ltime = _time.localtime(_time.time())
    tlm_str = _time.strftime('%H:%M:%S (%d/%m/%Y)',_ltime)
    return tlm_str

def getDate():
    """ Returns a formatted string with the current date."""
    _ltime = _time.localtime(_time.time())
    date_str = _time.strftime('%Y-%m-%dT%H:%M:%S',_ltime)

    return date_str

def convertDate(date):
    """ Converts the DATE-OBS date string into an integer of the
        number of seconds since 1970.0 using calendar.timegm().

        INPUT: DATE-OBS in format of 'YYYY-MM-DD'.
        OUTPUT: Date (integer) in seconds.
    """

    _dates = date.split('-')
    _val = 0
    _date_tuple = (int(_dates[0]), int(_dates[1]), int(_dates[2]), 0, 0, 0, 0, 0, 0)

    return calendar.timegm(_date_tuple)

#################
#
#
#               Generic File/Header Functions
#
#
#################
def getFilterNames(header,filternames=None):
    """
    Returns a comma-separated string of filter names extracted from the
    input header (PyFITS header object).  This function has been hard-coded
    to support the following instruments:
        ACS, WFPC2, STIS
    This function relies on the 'INSTRUME' keyword to define what instrument
    has been used to generate the observation/header.

    The 'filternames' parameter allows the user to provide a list of keyword
    names for their instrument, in the case their instrument is not supported.

    """
    # Define the keyword names for each instrument
    _keydict = {'ACS':['FILTER1','FILTER2'],'WFPC2':['FILTNAM1','FILTNAM2'],
                'STIS':['OPT_ELEM','FILTER'], 'NICMOS':['FILTER1','FILTER2']}

    # Find out what instrument the input header came from, based on the
    # 'INSTRUME' keyword
    if header.has_key('INSTRUME'):
        instrument = header['INSTRUME']
    else:
        raise ValueError,'Header does not contain INSTRUME keyword.'

    # Check to make sure this instrument is supported in _keydict
    if _keydict.has_key(instrument):
        _filtlist = _keydict[instrument]
    else:
        _filtlist = filternames

    # At this point, we know what keywords correspond to the filter names
    # in the header.  Now, get the values associated with those keywords.
    # Build a list of all filter name values, with the exception of the
    # blank keywords. Values containing 'CLEAR' or 'N/A' are valid.
    _filter_values = []
    for _key in _filtlist:
        _val = header[_key]
        if _val.strip() != '':
            _filter_values.append(header[_key])

    # Return the comma-separated list
    return string.join(_filter_values,',')


def buildNewRootname(filename,extn=None,extlist=None):
    """ Build rootname for a new file.
        Use 'extn' for new filename if given, does NOT
        append a suffix/extension at all.

        Does NOT check to see if it exists already.
        Will ALWAYS return a new filename.
    """
    # Search known suffixes to replace ('_crj.fits',...)
    _extlist = copy.deepcopy(EXTLIST)
    # Also, add a default where '_dth.fits' replaces
    # whatever extension was there ('.fits','.c1h',...)
    _extlist.append('.')
    # Also append any user-specified extensions...
    if extlist:
        _extlist += extlist

    for suffix in _extlist:
        _indx = filename.find(suffix)
        if _indx > 0: break

    if _indx < 0:
         # default to entire rootname
        _indx = len(filename)

    if extn == None: extn = ''

    return filename[:_indx]+extn

def buildRootname(filename,ext=None):
    """
    Built a new rootname for an existing file and given extension.
    Any user supplied extensions to use for searching for file
    need to be provided as a list of extensions.

    Usage:
        rootname = buildRootname(filename,ext=['_dth.fits'])

    """
    # Get complete list of filenames from current directory
    flist = os.listdir(os.curdir)
    #First, assume given filename is complete and verify
    # it exists...
    rootname = None

    for name in flist:
        if name == filename:
            rootname = filename
            break
        elif name == filename+'.fits':
            rootname = filename+'.fits'
            break
    # If we have an incomplete filename, try building a default
    # name and seeing if it exists...
    #
    # Set up default list of suffix/extensions to add to rootname
    _extlist = []
    for extn in EXTLIST:
        _extlist.append(extn)

    if rootname == None:
        # Add any user-specified extension to list of extensions...
        if ext != None:
            for i in ext:
                _extlist.insert(0,i)
        # loop over all extensions looking for a filename that matches...
        for extn in _extlist:
            rname = string.lower(filename) + extn
            for name in flist:
                if rname == name:
                    rootname = name
                    break
            if rootname != None:
                break

    # If we still haven't found the file, see if we have the
    # info to build one...
    if rootname == None and ext != None:
        # Check to see if we have a full filename to start with...
        _indx = string.find(filename,'.')
        if _indx > 0:
            rootname = filename[:_indx]+ext[0]
        else:
            rootname = filename + ext[0]

    # It will be up to the calling routine to verify
    # that a valid rootname, rather than 'None', was returned.
    return rootname

def getKeyword(filename,keyword,default=None,handle=None):
    """
    General, write-safe method for returning a keyword value
    from the header of a IRAF recognized image.
    It returns the value as a string.
    """
    # Insure that there is at least 1 extension specified...
    if string.find(filename,'[') < 0:
        filename += '[0]'

    _fname,_extn = parseFilename(filename)

    if not handle:
        # Open image whether it is FITS or GEIS
        _fimg = openImage(_fname)
    else:
        # Use what the user provides, after insuring
        # that it is a proper PyFITS object.
        if isinstance(handle, pyfits.HDUList):
            _fimg = handle
        else:
            raise ValueError,'Handle must be PyFITS object!'

    # Address the correct header
    _hdr = getExtn(_fimg,_extn).header

    try:
        value =  _hdr[keyword]
    except KeyError:
        _nextn = findKeywordExtn(_fimg, keyword)
        try:
            value = _fimg[_nextn].header[keyword]
        except KeyError:
            value = ''

    if not handle:
        _fimg.close()
        del _fimg

    if value == '':
        if default == None:
            value = None
        else:
            value = default
    # NOTE:  Need to clean up the keyword.. Occasionally the keyword value
    # goes right up to the "/" FITS delimiter, and iraf.keypar is incapable
    # of realizing this, so it incorporates "/" along with the keyword value.
    # For example, after running "pydrizzle" on the image "j8e601bkq_flt.fits",
    # the CD keywords look like this:
    #
    #   CD1_1   = 9.221627430999639E-06/ partial of first axis coordinate w.r.t. x
    #   CD1_2   = -1.0346992614799E-05 / partial of first axis coordinate w.r.t. y
    #
    # so for CD1_1, iraf.keypar returns:
    #       "9.221627430999639E-06/"
    #
    # So, the following piece of code CHECKS for this and FIXES the string,
    # very simply by removing the last character if it is a "/".
    # This fix courtesy of Anton Koekemoer, 2002.
    elif type(value) is types.StringType:
        if value[-1:] == '/':
            value = value[:-1]

    return value

def getHeader(filename,handle=None):
    """ Return a copy of the PRIMARY header, along with any group/extension
        header, for this filename specification.
    """
    _fname,_extn = parseFilename(filename)

    # Allow the user to provide an already opened PyFITS object
    # to derive the header from...
    #
    if not handle:
        # Open image whether it is FITS or GEIS
        _fimg = openImage(_fname,mode='readonly')
    else:
        # Use what the user provides, after insuring
        # that it is a proper PyFITS object.
        if isinstance(handle,pyfits.HDUList):
            _fimg = handle
        else:
            raise ValueError,'Handle must be PyFITS object!'

    _hdr = _fimg['PRIMARY'].header.copy()

    if _extn > 0:
        # Append correct extension/chip/group header to PRIMARY...
        for _card in getExtn(_fimg,_extn).header.ascardlist():
            _hdr.ascard.append(_card)

    if not handle:
        # Close file handle now...
        _fimg.close()
        del _fimg

    return _hdr


def updateKeyword(filename, key, value,show=yes):
    """ Add/update keyword to header with given value. """

    _fname,_extn = parseFilename(filename)

    # Open image whether it is FITS or GEIS
    _fimg = openImage(_fname,mode='update')

    # Address the correct header
    _hdr = getExtn(_fimg,_extn).header

    # Assign a new value or add new keyword here.
    try:
        _hdr[key] = value
    except KeyError:
        if show:
            print 'Adding new keyword ',key,'=',value
        _hdr.update(key, value)

    # Close image
    _fimg.close()
    del _fimg

def buildFITSName(geisname):
    """ Build a new FITS filename for a GEIS input image. """
    # User wants to make a FITS copy and update it...
    _indx = geisname.rfind('.')
    _fitsname = geisname[:_indx]+'_'+geisname[_indx+1:-1]+'f.fits'

    return _fitsname

def openImage(filename,mode='readonly',memmap=0,fitsname=None):
    """ Opens file and returns PyFITS object.
        It will work on both FITS and GEIS formatted images.

    """
    # Insure that the filename is always fully expanded
    # This will not affect filenames without paths or
    # filenames specified with extensions.
    filename = osfn(filename)

    # Extract the rootname and extension specification
    # from input image name
    _fname,_iextn = parseFilename(filename)

    # Parse out the filename (without extension) for the
    # provided fitsname, if it was provided...
    _fitsroot = None
    if fitsname:
        _fitsroot,_fextn = parseFilename(fitsname)

    if _fname.find('.fits') > 0:
        try:
            # Open the FITS file
            return pyfits.open(_fname,mode=mode,memmap=memmap)
        except:
            raise IOError("Could not open file:",_fname)

    elif findFile(_fitsroot):
        # Input was specified as a GEIS image, but a FITS copy
        # already exists, so open it instead...
        try:
            return pyfits.open(_fitsroot,mode=mode,memmap=memmap)
        except:
            raise IOError("Could not open FITS copy:",_fitsroot)

    else:
        # Input was specified as a GEIS image, but no FITS copy
        # exists.  Read it in with 'readgeis' and make a copy
        # then open the FITS copy...
        try:
            # Open as a GEIS image for reading only
            fimg =  readgeis.readgeis(_fname)
        except:
            raise IOError("Could not open GEIS input:",_fname)

        # Check to see if user wanted to update GEIS header.
        if mode != 'readonly':
            if fitsname != None:
                # User wants to make a FITS copy and update it
                # using the filename they have provided

                # Write out GEIS image as multi-extension FITS.
                try:
                    fimg.writeto(_fitsroot)
                except:
                    print ' --ERROR CHECK-- '
                    print '    Check whether "%s" already exists and remove it.'%_fitsroot
                    print ' -------- '
                    fimg.close()
                    del fimg
                    raise ValueError,'Problem creating FITS copy of GEIS image: %s'%_fitsroot

                # Now close input GEIS image, and open writable
                # handle to output FITS image instead...
                fimg.close()
                del fimg
                fimg = pyfits.open(_fitsroot,mode=mode,memmap=memmap)

            else:
                print ' --ERROR CHECK-- '
                print '    Convert GEIS image to multi-extension FITS or'
                print '    set "fitsname" to a new FITS filename.'
                print ' -------- '
                raise TypeError,'Updating GEIS image headers not supported!'

        # Return handle for use by user
        return fimg


def parseFilename(filename):
    """
        Parse out filename from any specified extensions.
        Returns rootname and string version of extension name.

    """
    # Parse out any extension specified in filename
    _indx = filename.find('[')
    if _indx > 0:
        # Read extension name provided
        _fname = filename[:_indx]
        _extn = filename[_indx+1:-1]
    else:
        _fname = filename
        _extn = None
    return _fname,_extn

def parseExtn(extn):
    """ Convert full extension specification into a EXTVER ID.
        Input 'extn' assumed to be of the form: 'sci,1' or '1'.
        This will always returns an integer, and default to 1.
    """
    # If extn is None, return a default of 1
    if not extn:
        return 1

    if repr(extn).find(',') > 1:
        _extn = int(extn.split(',')[1])
        # Two values given for extension:
        #    for example, 'sci,1' or 'dq,1'
    elif repr(extn).find('/') > 1:
        # We are working with GEIS group syntax
        _indx = str(extn[:extn.find('/')])
        _extn = int(_indx)
    elif type(extn) == types.StringType:
        # Only one extension value specified...
        if extn.isdigit():
            # We only have EXTNAME specified...
            _extn = int(extn)
        else:
            # Only extension name given,
            # so default of 1 is returned.
            _extn = 1
    else:
        # Only integer extension number given, so return it.
        _extn = int(extn)

    return _extn

def getExtn(fimg,extn=None):
    """ Returns the PyFITS extension corresponding to
        extension specified in filename.
        Defaults to returning the first extension with
        data or the primary extension, if none have data.
    """
    # If no extension is provided, search for first extension
    # in FITS file with data associated with it.
    if not extn:
        # Set up default to point to PRIMARY extension.
        _extn = fimg[0]
        # then look for first extension with data.
        for _e in fimg:
            if _e.data != None:
                _extn = _e
                break
    else:
        # An extension was provided, so parse it out...
        if repr(extn).find(',') > 1:
            _extns = extn.split(',')
            # Two values given for extension:
            #    for example, 'sci,1' or 'dq,1'
            _extn = fimg[_extns[0],int(_extns[1])]
        elif repr(extn).find('/') > 1:
            # We are working with GEIS group syntax
            _indx = str(extn[:extn.find('/')])
            _extn = fimg[int(_indx)]
        elif type(extn) == types.StringType:
            # Only one extension value specified...
            if extn.isdigit():
                # We only have an extension number specified as a string...
                _nextn = int(extn)
            else:
                # We only have EXTNAME specified...
                _nextn = extn
            _extn = fimg[_nextn]
        else:
            # Only integer extension number given, or default of 0 is used.
            _extn = fimg[int(extn)]

    return _extn
#
#Revision History:
#    Nov 2001: findFile upgraded to accept full filenames with paths,
#               instead of working only on files from current directory. WJH
#
# Base function for
#   with optional path.
def findFile(input):

    """ Search a directory for full filename with optional path. """
    # If no input name is provided, default to returning 'no'(FALSE)
    if not input:
        return no

    # We use 'osfn' here to insure that any IRAF variables are
    # expanded out before splitting out the path...
    _fdir,_fname = os.path.split(osfn(input))

    if _fdir == '':
        _fdir = os.curdir

    flist = os.listdir(_fdir)

    _root,_extn = parseFilename(_fname)

    found = no
    for name in flist:
        #if not string.find(name,_fname):
        if name == _root:
            # Check to see if given extension, if any, exists
            if _extn == None:
                found = yes
                continue
            else:
                _split = _extn.split(',')
                _extnum = None
                _extver = None
                if  _split[0].isdigit():
                    _extname = None
                    _extnum = int(_split[0])
                else:
                    _extname = _split[0]
                    if len(_split) > 1:
                        _extver = int(_split[1])
                    else:
                        _extver = 1
                f = openImage(_root)
                f.close()
                if _extnum:
                    if _extnum < len(f):
                        found = yes
                        del f
                        continue
                    else:
                        del f
                else:
                    _fext = findExtname(f,_extname,extver=_extver)
                    if _fext != None:
                        found = yes
                        del f
                        continue
    return found


def checkFileExists(filename,directory=None):
    """ Checks to see if file specified exists in current
        or specified directory. Default is current directory.
        Returns 1 if it exists, 0 if not found.
    """
    if directory == None or directory == '': directory = '.'
    _ldir = os.listdir(directory)

    _exist = 0
    # for each file in directory...
    for file in _ldir:
        # compare filename with file
        if string.find(file,filename) > -1:
            _exist = 1
            break

    return _exist


def copyFile(input,output,replace=None):

    """ Copy a file whole from input to output. """

    _found = findFile(output)
    if not _found or (_found and replace ):
        shutil.copy2(input, output)
        #finput = open(input,'r')
        #fout = open(output,'w')
        #fout.writelines(finput.readlines())
        #fout.close()
        #finput.close()
        #del finput
        #del fout

def _remove(file):
    # Check to see if file exists.  If not, return immediately.
    if not findFile(file):
        return

    if file.find('.fits') > 0:
        try:
            os.remove(file)
        except StandardError:
            pass
    elif file.find('.imh') > 0:
        # Delete both .imh and .pix files
        os.remove(file)
        os.remove(file[:-3]+'pix')
    else:
        # If we have a GEIS image that has separate header
        # and pixel files which need to be removed.
        # Assumption: filenames end in '.??h' and '.??d'
        #
        os.remove(file)
        os.remove(file[:-1]+'d')

def removeFile(inlist):
    """ Utility function for deleting a list of files or a single file. """
    if type(inlist) != types.StringType:
    # We do have a list, so delete all filenames in list.
        # Treat like a list of full filenames
        _ldir = os.listdir('.')
        for f in inlist:
        # Now, check to see if there are wildcards which need to be expanded
            if f.find('*') >= 0 or f.find('?') >= 0:
                # We have a wild card specification
                regpatt = f.replace('?','.?')
                regpatt = regpatt.replace('*','.*')
                _reg = sre.compile(regpatt)
                for file in _ldir:
                    if _reg.match(file):
                        _remove(file)
            else:
                # This is just a single filename
                _remove(f)
    else:
        # It must be a string then, so treat as a single filename
        _remove(inlist)


def findKeywordExtn(ft,keyword,value=None):

    """
    This function will return the index of the extension in
    a multi-extension FITS file which contains the desired keyword
    with the given value.
    """

    i = 0
    extnum = -1
    # Search through all the extensions in the FITS object
    for chip in ft:
        hdr = chip.header
        # Check to make sure the extension has the given keyword
        if hdr.has_key(keyword):
            if value != None:
                # If it does, then does the value match the desired value
                # MUST use 'string.strip' to match against any input string!
                if string.strip(hdr[keyword]) == value:
                    extnum = i
                    break
            else:
                extnum = i
                break
        i = i + 1
    # Return the index of the extension which contained the
    # desired EXTNAME value.
    return extnum


def findExtname(fimg,extname,extver=None):
    """ This function returns the list number of the extension
            corresponding to EXTNAME given.
    """
    i = 0
    extnum = None
    for chip in fimg:
        hdr = chip.header
        if hdr.has_key('EXTNAME'):
            if string.strip(hdr['EXTNAME']) == string.upper(extname):
                if extver == None or hdr['EXTVER'] == extver:
                    extnum = i
                    break
        i = i + 1
    return extnum

def rAsciiLine(ifile):

    """ Returns the next non-blank line in an ASCII file. """

    _line = string.strip(ifile.readline())
    while len(_line) == 0:
        _line = string.strip(ifile.readline())
    return _line

#################
#
#
#               PyDrizzle Functions
#
#
#################

def defaultModel():
    """ This function returns a default, non-distorting model
        that can be used with the data.
    """
    order = 3

    fx = N.zeros(shape=(order+1,order+1),type=N.Float64)
    fy = N.zeros(shape=(order+1,order+1),type=N.Float64)

    fx[1,1] = 1.
    fy[1,0] = 1.

    # Used in Pattern.computeOffsets()
    refpix = {}
    refpix['XREF'] = None
    refpix['YREF'] = None
    refpix['V2REF'] = 0.
    refpix['V3REF'] = 1.
    refpix['XDELTA'] = 0.
    refpix['YDELTA'] = 0.
    refpix['PSCALE'] = None
    refpix['DEFAULT_SCALE'] = no
    refpix['centered'] = yes
    return fx,fy,refpix,order

# This function read the IDC table and generates the two matrices with
# the geometric correction coefficients.
#
#       INPUT: FITS object of open IDC table
#       OUTPUT: coefficient matrices for Fx and Fy
#
#### If 'tabname' == None: This should return a default, undistorted
####                        solution.
#
def readIDCtab (tabname, chip=1, date=None, direction='forward',
                filter1=None,filter2=None, offtab=None):
    """
        Read IDCTAB, and optional OFFTAB if sepcified, and generate
        the two matrices with the geometric correction coefficients.

        If tabname == None, then return a default, undistorted solution.
        If offtab is specified, dateobs also needs to be given.

    """
    # Return a default geometry model if no IDCTAB filename
    # is given.  This model will not distort the data in any way.
    if tabname == None:
        return defaultModel()

    # Implement default values for filters here to avoid the default
    # being overwritten by values of None passed by user.
    if filter1 == None or filter1.find('CLEAR') == 0:
        filter1 = 'CLEAR'
    if filter2 == None or filter2.find('CLEAR') == 0:
        filter2 = 'CLEAR'

    # Insure that tabname is full filename with fully expanded
    # IRAF variables; i.e. 'jref$mc41442gj_idc.fits' should get
    # expanded to '/data/cdbs7/jref/mc41442gj_idc.fits' before
    # being used here.
    # Open up IDC table now...
    try:
        ftab = openImage(tabname)
    except:
        raise IOError,"IDC table '%s' not valid as specified!" % tabname

    #First thing we need, is to read in the coefficients from the IDC
    # table and populate the Fx and Fy matrices.

    detector = ftab['PRIMARY'].header['DETECTOR']
    # Set default filters for SBC
    if detector == 'SBC':
        if filter1 == 'CLEAR':
            filter1 = 'F115LP'
            filter2 = 'N/A'
        if filter2 == 'CLEAR':
            filter2 = 'N/A'

    # Read FITS header to determine order of fit, i.e. k
    order = ftab['PRIMARY'].header['NORDER']

    fx = N.zeros(shape=(order+1,order+1),type=N.Float64)
    fy = N.zeros(shape=(order+1,order+1),type=N.Float64)

    #Determine row from which to get the coefficients.
    # How many rows do we have in the table...
    fshape = ftab[1].data.getshape()
    colnames = ftab[1].data._names
    row = -1

    # Loop over all the rows looking for the one which corresponds
    # to the value of CCDCHIP we are working on...
    for i in xrange(fshape[0]):

        try:
            # Match FILTER combo to appropriate row,
            #if there is a filter column in the IDCTAB...
            if 'FILTER1' in colnames and 'FILTER2' in colnames:

                filt1 = ftab[1].data.field('FILTER1')[i]
                if filt1.find('CLEAR') > -1: filt1 = filt1[:5]

                filt2 = ftab[1].data.field('FILTER2')[i]
                if filt2.find('CLEAR') > -1: filt2 = filt2[:5]
            else:
                if 'OPT_ELEM' in colnames:
                    filt1 = ftab[1].data.field('OPT_ELEM')
                    if filt1.find('CLEAR') > -1: filt1 = filt1[:5]
                else:
                    filt1 = filter1

                if 'FILTER' in colnames:
                    _filt = ftab[1].data.field('FILTER')[i]
                    if _filt.find('CLEAR') > -1: _filt = _filt[:5]
                    if 'OPT_ELEM' in colnames:
                        filt2 = _filt
                    else:
                        filt1 = _filt
                        filt2 = 'CLEAR'
                else:
                    filt2 = filter2
        except:
            # Otherwise assume all rows apply and compare to input filters...
            filt1 = filter1
            filt2 = filter2

        if 'DETCHIP' in colnames:
            detchip = ftab[1].data.field('DETCHIP')[i]
        else:
            detchip = 1

        if 'DIRECTION' in colnames:
            direct = string.strip(string.lower(ftab[1].data.field('DIRECTION')[i]))
        else:
            direct = 'forward'

        if filt1 == filter1.strip() and filt2 == filter2.strip():
            if direct == direction.strip():
                if int(detchip) == int(chip) or int(detchip) == -999:
                    row = i
                    break
    if row < 0:
        print 'Row corresponding to DETCHIP of ',detchip,' was not found!'
        raise LookupError
    else:
        print '- IDCTAB: Distortion model from row',str(row+1),'for chip',detchip,':',filter1.strip(),'and',filter2.strip()

    # Read in V2REF and V3REF: this can either come from current table,
    # or from an OFFTAB if time-dependent (i.e., for WFPC2)
    theta = None
    if 'V2REF' in colnames:
        v2ref = ftab[1].data.field('V2REF')[row]
        v3ref = ftab[1].data.field('V3REF')[row]
    else:
        # Read V2REF/V3REF from offset table (OFFTAB)
        if offtab:
            v2ref,v3ref,theta = readOfftab(offtab, date, chip=detchip)
        else:
            v2ref = 0.0
            v3ref = 0.0

    if 'THETA' in colnames:
        theta = ftab[1].data.field('THETA')[row]
    else:
        theta = 0.0

    refpix = {}
    refpix['XREF'] = ftab[1].data.field('XREF')[row]
    refpix['YREF'] = ftab[1].data.field('YREF')[row]
    refpix['XSIZE'] = ftab[1].data.field('XSIZE')[row]
    refpix['YSIZE'] = ftab[1].data.field('YSIZE')[row]
    refpix['PSCALE'] = round(ftab[1].data.field('SCALE')[row],8)
    refpix['V2REF'] = v2ref
    refpix['V3REF'] = v3ref
    refpix['THETA'] = theta
    refpix['XDELTA'] = 0.0
    refpix['YDELTA'] = 0.0
    refpix['DEFAULT_SCALE'] = yes
    refpix['centered'] = no

    # Now that we know which row to look at, read coefficients into the
    #   numeric arrays we have set up...
    # Setup which column name convention the IDCTAB follows
    # either: A,B or CX,CY
    if 'CX10' in ftab[1].data._names:
        cxstr = 'CX'
        cystr = 'CY'
    else:
        cxstr = 'A'
        cystr = 'B'

    for i in xrange(order+1):
        if i > 0:
            for j in xrange(i+1):
                xcname = cxstr+str(i)+str(j)
                ycname = cystr+str(i)+str(j)
                fx[i,j] = ftab[1].data.field(xcname)[row]
                fy[i,j] = ftab[1].data.field(ycname)[row]

    ftab.close()
    del ftab

    # If CX11 is 1.0 and not equal to the PSCALE, then the
    # coeffs need to be scaled
    if fx[1,1] == 1.0 and abs(fx[1,1]) != refpix['PSCALE']:
        fx *= refpix['PSCALE']
        fy *= refpix['PSCALE']

    # Return arrays and polynomial order read in from table.
    # NOTE: XREF and YREF are stored in Fx,Fy arrays respectively.
    return fx,fy,refpix,order

def readOfftab(offtab, date, chip=None):
    """ Read V2REF,V3REF from a specified offset table (OFFTAB). """
    # Return a default geometry model if no IDCTAB filename
    # is given.  This model will not distort the data in any way.
    if offtab == None:
        return 0.,0.

    # Provide a default value for chip
    if chip:
        detchip = chip
    else:
        detchip = 1

    # Open up IDC table now...
    try:
        ftab = openImage(offtab)
    except:
        raise IOError,"Offset table '%s' not valid as specified!" % offtab

    #Determine row from which to get the coefficients.
    # How many rows do we have in the table...
    fshape = ftab[1].data.getshape()
    colnames = ftab[1].data._names
    row = -1

    row_start = None
    row_end = None

    v2end = None
    v3end = None
    date_end = None
    theta_end = None

    num_date = convertDate(date)
    # Loop over all the rows looking for the one which corresponds
    # to the value of CCDCHIP we are working on...
    for ri in xrange(fshape[0]):
        i = fshape[0] - ri - 1
        if 'DETCHIP' in colnames:
            detchip = ftab[1].data.field('DETCHIP')[i]
        else:
            detchip = 1

        obsdate = convertDate(ftab[1].data.field('OBSDATE')[i])

        # If the row is appropriate for the chip...
            # Interpolate between dates
        if int(detchip) == int(chip) or int(detchip) == -999:
            if num_date <= obsdate:
                date_end = obsdate
                v2end = ftab[1].data.field('V2REF')[i]
                v3end = ftab[1].data.field('V3REF')[i]
                theta_end = ftab[1].data.field('THETA')[i]
                row_end = i
                continue

            if row_end == None and (num_date > obsdate):
                date_end = obsdate
                v2end = ftab[1].data.field('V2REF')[i]
                v3end = ftab[1].data.field('V3REF')[i]
                theta_end = ftab[1].data.field('THETA')[i]
                row_end = i
                continue

            if num_date > obsdate:
                date_start = obsdate
                v2start = ftab[1].data.field('V2REF')[i]
                v3start = ftab[1].data.field('V3REF')[i]
                theta_start = ftab[1].data.field('THETA')[i]
                row_start = i
                break

    ftab.close()
    del ftab

    if row_start == None and row_end == None:
        print 'Row corresponding to DETCHIP of ',detchip,' was not found!'
        raise LookupError
    elif row_start == None:
        print '- OFFTAB: Offset defined by row',str(row_end+1)
    else:
        print '- OFFTAB: Offset interpolated from rows',str(row_start+1),'and',str(row_end+1)

    # Now, do the interpolation for v2ref, v3ref, and theta
    if row_start == None or row_end == row_start:
        # We are processing an observation taken after the last calibration
        date_start = date_end
        v2start = v2end
        v3start = v3end
        _fraction = 0.
        theta_start = theta_end
    else:
        _fraction = float((num_date - date_start)) / float((date_end - date_start))

    v2ref = _fraction * (v2end - v2start) + v2start
    v3ref = _fraction * (v3end - v3start) + v3start
    theta = _fraction * (theta_end - theta_start) + theta_start

    return v2ref,v3ref,theta

def readWCSCoeffs(header):
    """
    Read distortion coeffs from WCS header keywords and
    populate distortion coeffs arrays.
    """
    # Read in order for polynomials
    _xorder = header['a_order']
    _yorder = header['b_order']
    order = max(max(_xorder,_yorder),3)

    fx = N.zeros(shape=(order+1,order+1),type=N.Float64)
    fy = N.zeros(shape=(order+1,order+1),type=N.Float64)

    # Read in CD matrix
    _cd11 = header['cd1_1']
    _cd12 = header['cd1_2']
    _cd21 = header['cd2_1']
    _cd22 = header['cd2_2']
    _cdmat = N.array([[_cd11,_cd12],[_cd21,_cd22]])
    _theta = N.arctan2(-_cd12,_cd22)
    _rotmat = N.array([[N.cos(_theta),N.sin(_theta)],
                      [-N.sin(_theta),N.cos(_theta)]])
    _rCD = N.dot(_rotmat,_cdmat)
    _skew = N.arcsin(-_rCD[1][0] / _rCD[0][0])
    _scale = _rCD[0][0] * N.cos(_skew) * 3600.
    _scale2 = _rCD[1][1] * 3600.

    # Set up refpix
    refpix = {}
    refpix['XREF'] = header['crpix1']
    refpix['YREF'] = header['crpix2']
    refpix['XSIZE'] = header['naxis1']
    refpix['YSIZE'] = header['naxis2']
    refpix['PSCALE'] = _scale
    refpix['V2REF'] = 0.
    refpix['V3REF'] = 0.
    refpix['THETA'] = RADTODEG(_theta)
    refpix['XDELTA'] = 0.0
    refpix['YDELTA'] = 0.0
    refpix['DEFAULT_SCALE'] = yes
    refpix['centered'] = no


    # Set up template for coeffs keyword names
    cxstr = 'A_'
    cystr = 'B_'
    # Read coeffs into their own matrix
    for i in xrange(_xorder+1):
        for j in xrange(i+1):
            xcname = cxstr+str(j)+'_'+str(i-j)
            if header.has_key(xcname):
                fx[i,j] = header[xcname]

    # Extract Y coeffs separately as a different order may
    # have been used to fit it.
    for i in xrange(_yorder+1):
        for j in xrange(i+1):
            ycname = cystr+str(j)+'_'+str(i-j)
            if header.has_key(ycname):
                fy[i,j] = header[ycname]

    # Now set the linear terms
    fx[0][0] = 1.0
    fy[0][0] = 1.0

    return fx,fy,refpix,order

def readAsnTable(fname,output=None,prodonly=yes):
    """
     This function reads the filenames/rootnames and shifts from a FITS
     ASN table.

     Column names expected are:
       MEMNAME     - rootname of each member
       MEMTYPE     - type of member in association(PROD-* or EXP-*)
       XOFFSET     - shift in X for this observation: pixels (default) or arcseconds
       YOFFSET     - shift in Y for this observation: pixels (default) or arcseconds
       DELTAX      - delta shift in RA (arcseconds) for this observation
       DELTAY      - delta shift in Dec (arcseconds) for this observation
       ROTATION    - additional rotation to be applied
       SCALE       - scale image by this value

     If absolute shifts are provided, then it will look for header
     keywords:
        REFIMAGE   - image which output pixel shifts were calculated from
        SHFRAME    - frame for absolute shifts: input or output

     This will return a nested dictionary corresponding to this
     association table.
     Observation dictionary: {'xsh':0.,'ysh':0.,'rot':0.,'scale':1.}
     Product dictionary: {'output':type, 'memname1':dict, 'memname2':dict,...}
               where dict: Observation dictionary
     The dictionary using EXP* will be:
       p = {'output':'dthname',
            'order':[name1,name2,...]
            'members':{
                    'name2':{'xsh':0.,...}, 'name1':{'xsh':0.,...},...
                       }
                    }
     You get a list of input names using 'p.keys()'.

     Parameters:
               output:     output - desired name of output image (None,'EXP', or user-filename)
               prodonly:   yes - only select MEMTYPE=PROD* as input observations
                                       no - use MEMTYPE=EXP* as input observations

     Output: If none is specified by user, the first 'PROD-DTH' filename found in the
                    ASN table will be used.  If there is no 'PROD-DTH' entry, the first 'PROD'
                    entry will be used instead.  Finally, if 'output' = 'EXP', the value 'EXP'
                    will be passed along and interpreted as a switch to use the input filename
                    as the output resulting in every input creating a separate output.
    """
    # Initialize this dictionary for output
    asndict = {'output':None,'order':[],'members':{'abshift':no,'dshift':no}}

    # Insure that ASN filename is full filename with fully expanded
    # IRAF variables; i.e. 'mydir$jx001a010_asn.fits' should get
    # expanded to '/my/data/dir/jx001a010_asn.fits' before
    # being used here.
    # This operation is a no-op if no such variables are present in fname.
    fname = osfn(fname)

    # Open the table...
    try:
        ftab = pyfits.open(fname)
    except:
        raise IOError,"Association table '%s' not valid as specified!" % fname

    tablen = ftab[1].data.getshape()[0]
    colnames = ftab[1].columns.names
    colunits = ftab[1].columns.units

    _shift_units = None

    # Now, put it together with rootname and type...
    for row in xrange(tablen):
        # Read in required columns for each row
        if 'MEMNAME' in colnames and 'MEMTYPE' in colnames:
            # We need to make sure no EOS characters remain part of
            # the strings we read out...
            mname = string.split(ftab[1].data.field('MEMNAME')[row],'\0',1)[0]
            mtype = string.split(ftab[1].data.field('MEMTYPE')[row],'\0',1)[0]
            memname = string.strip(mname)
            memtype = string.strip(mtype)
            memrow = row
        else:
            print 'Association table incomplete: required column(s) MEMNAME/MEMTYPE NOT found!'
            raise LookupError

        # Do we care about this entry?
        # Entries that should be used to build DTH product are:
        #  PROD-RPT, PROD-CRJ, EXP-DTH, or EXP-TARG
        if memtype.find('PROD') < 0 and memtype.find('EXP-DTH') < 0 and memtype.find('EXP-TARG') < 0:
            if prodonly:
                # We are looking at an EXP* entry we don't want...
                continue

        memdict = {}
        # Keep track of which order they were read in by their row number
        memdict['row'] = memrow
        memdict['xoff'] = 0.
        memdict['yoff'] = 0.
        memdict['rot'] = 0.
        memdict['abshift'] = no
        memdict['dshift'] = no
        memdict['shift_units'] = _shift_units

        # Read in optional data from columns
        # X offset
        if 'XOFFSET' in colnames:
            memdict['xshift'] = -ftab[1].data.field('XOFFSET')[row]
            if memdict['xshift'] != 0.: memdict['abshift'] = yes

            # Read in units of shifts provided in ASN table
            for i in xrange(len(colnames)):
                if colnames[i] == 'XOFFSET':
                    _abshift_units = colunits[i]
            # Default to shifts in pixels, not arcseconds
            if _abshift_units == '': _abshift_units = 'pixels'

            # Read in frame of shift provided in ASN table
            try:
                _shift_frame = ftab[0].header['shframe']
            except KeyError:
                # Default to input pixel frame for shifts
                _shift_frame = 'input'
            try:
                _refimage = ftab[0].header['refimage']
            except KeyError:
                _refimage = None
            # Record frame and units in dictionary
            memdict['shift_frame'] = _shift_frame
            memdict['refimage'] = _refimage
        else:
            memdict['xshift'] = 0.

        # Y offset
        if 'YOFFSET' in colnames:
            memdict['yshift'] = -ftab[1].data.field('YOFFSET')[row]
            if memdict['yshift'] != 0.: memdict['abshift'] = yes
        else:
            memdict['yshift'] = 0.

        # Rotation angle
        if 'ROTATION' in colnames:
            memdict['delta_rot'] = ftab[1].data.field('ROTATION')[row]
        else:
            memdict['delta_rot'] = 0.

        # Scale: output pixel size
        if 'SCALE' in colnames:
            memdict['delta_scale'] = ftab[1].data.field('SCALE')[row]
        else:
            memdict['delta_scale'] = 1.

        # X offset
        if 'XDELTA' in colnames:
            memdict['delta_x'] = ftab[1].data.field('XDELTA')[row]
            if memdict['delta_x'] != 0.: memdict['dshift'] = yes

            # Read in units of shifts provided in ASN table
            for i in xrange(len(colnames)):
                if colnames[i] == 'XDELTA':
                    _dshift_units = colunits[i]
            # Default to shifts in pixels, not arcseconds
            if _dshift_units == '': _dshift_units = 'pixels'

        else:
            memdict['delta_x'] = 0.

        # Y offset
        if 'YDELTA' in colnames:
            memdict['delta_y'] = ftab[1].data.field('YDELTA')[row]
            if memdict['delta_y'] != 0.: memdict['dshift'] = yes
        else:
            memdict['delta_y'] = 0.



        # Record that absolute shifts were provided and should be applied
        if memdict['abshift']:
            asndict['members']['abshift'] = yes
            memdict['shift_units'] = _abshift_units
            _shift_units = _abshift_units


        # Set switch specifying that delta shifts are given, ONLY
        # if no absolute shifts are given...
        if memdict['dshift'] and not memdict['abshift']:
            asndict['members']['dshift'] = yes
            memdict['shift_units'] = _dshift_units
            _shift_units = _dshift_units

        # Build the shifts dictionary now...
        if string.find(memtype,'PROD') < 0 and not prodonly:
            # We want to use this EXP* entry.
            asndict['members'][memname] = memdict
            asndict['order'].append(memname)
        elif memtype == 'PROD-DTH' or memtype == 'PROD-TARG':
            # We have found a dither product specified in ASN
            # Not to be used for input, but
            # has one already been specified as the final output?
            if output == None:
                # Use default output name
                asndict['output'] = memname
            else:
                # Use user-specified output name here
                asndict['output'] = output

        else:
            # We are working with a PROD* entry...
            if prodonly:
                asndict['members'][memname] = memdict
                asndict['order'].append(memname)
            elif asndict['output'] == None:
                asndict['output'] = memname

        # Set up a default output filename
        # This will be overwritten by a different output
        # name if a PROD-DTH entry is found in the ASN table
        # and 'output' was not specified by the user.
        # Useful for CR-SPLIT/REPEAT-OBS ASN tables.
        if asndict['output'] == None:
            if output == None and prodonly:
                asndict['output'] = memname
            else:
                asndict['output'] = output

    # Establish default units in case none are specified...
    if _shift_units == None: _shift_units = 'pixels'

    # Go back and make sure that all rows have consistent values of
    # 'dshift' and 'abshift'
    _members = asndict['members']
    for member in _members:
        if member != 'abshift' and member != 'dshift':
            _members[member]['abshift'] = _members['abshift']
            _members[member]['dshift'] = _members['dshift']
            _members[member]['shift_units'] = _shift_units

    # Finished reading all relevant rows from table
    ftab.close()
    del ftab

    return asndict
