""" buildasn.py - Module to generate an ACS-style association
table from all matching files from the current working directory.

This module will also recognize shiftfiles should they be specified
and add the offsets to the ASN table.

W.J. Hack 9 April 2001
    *** INITIAL VERSION

W.J Hack 24 April 2001
    Version 0.1b - Modified to support FITS version 0.4.2

W.J Hack, 1 May 2001
    Version 0.2  - Computes delta from given shifts file and header info

WJH, 20 Nov 2001:
    Version 0.3 - Added ROTATION column to default ASN table created.

WJH, 4 Dec 2001:
    Added help() function and printing of Version number.

WJH, 21 Jan 2002:
    Fixed bugs in writing out ASN table with default columns.
    Also added check to insure that files are found to build ASN table.

WJH, 8 Mar 2002:
    Added MEMPRSNT column and converted to use PyFITS v0.6.2 and numarray.

WJH, 12 June 2002 (0.5a):
    Changed 'pyfits.version' to 'pyfits.__version__' to keep up with PyFITS.

WJH, 20 Aug 2002 (0.6):
    Revised interface to accept wild-card specification and @files as well
        as filename partial (such as 'crj' or '440w').

WJH, 3 Dec 2002 (0.6a):
    Simplified interface to 'readShiftFile' for use with 'updateasn'; it now
    returns a dictionary without requiring it be pre-defined.

WJH, 3 Dec 2002 (0.6b):
    Blank lines in shiftfiles will now work.

WJH, 11 Apr 2003 (0.7):
    Support new shiftfile convention, as well as optional SCALE column.

WJH, 3 Jun 2003 (0.7a):
    Added logic to ignore blank lines in shift file.

WJH, 25 Aug 2003 (0.8)
    Updated to support numarray 0.6 using the new format letter codes.

CJH, 13 Jul 2004 (0.9)
    Updated to support input of simple python list as linename input.

CJH, 02 Aug 2004 (0.9a)
    Added support to find files for having 2 columns of filenames in an @inputlist
    file.  This is used by Multidrizzle to implement the use of IVM files.
WJH/CJH, 28 Sept 2004 (0.9.2)
    Added support to findFiles for comma separated string input.  Also corrected
    a bug in the processing of @files in which rstrip() was not being applied to
    the filenames leaving a trailing newline character.
WJH, 15 Nov 2004 (0.9.3)
    Moved 'readShiftFile' to 'fileutil.py' so it would not need to import
        'buildasn'. Instead, we use 'from fileutil import readShiftFile' here
WJH, 20 Dec 2004 (0.9.4)
    Added 'writeAsnDict' to output the ASN table dictionary read in using
    'readAsnTable' to a file.
WJH, 11 May 2005 (0.9.5)
    Modified to append reference WCS extension if given in shiftfile.
WJH, 28 June 2005 (0.9.5a)
    New 'order' keyword in dictionary from reading the shift file now used
        to index files.

"""

import os, re, types, copy
import time
import pyfits
#import numarray,chararray
# New numarray 0.6 import mode
import numpy as N

from fileutil import readShiftFile,buildRootname
import wcsutil

# List of supported default file types
# It will look for these file types by default
# when trying to recognize input rootnames.
#EXTLIST =  ['_crj.fits','_flt.fits','_sfl.fits','_raw.fits','_drz.fits']
EXTLIST =  ['_crj.fits','_flt.fits','_sfl.fits','_cal.fits','_raw.fits','.c0h','.hhh','.fits']

# Define the format of the Table data dictionary used internally
TABLE_DATA = {'units':None,'frame':None,'xsh':None,'ysh':None,
            'dx':None,'dy':None,'rot':None,'scl':None,
            'name':None,'mtype':None,'mprsnt':None}

__version__ = '0.9.5a (28-June-2005)'

_prihdr = pyfits.Header([pyfits.Card('SIMPLE', pyfits.TRUE,'Fits standard'),
                pyfits.Card('BITPIX  ',                    16 ,' Bits per pixel'),
                pyfits.Card('NAXIS   ',                     0 ,' Number of axes'),
                pyfits.Card('ORIGIN  ',  'NOAO-IRAF FITS Image Kernel July 1999' ,'FITS file originator'),
                pyfits.Card('IRAF-TLM',  '18:26:13 (27/03/2000)' ,' Time of last modification'),
                pyfits.Card('EXTEND  ',pyfits.TRUE ,' File may contain standard extensions'),
                pyfits.Card('NEXTEND ',                     1 ,' Number of standard extensions'),
                pyfits.Card('DATE    ',  '2001-02-14T20:07:57',' date this file was written (yyyy-mm-dd)'),
                pyfits.Card('FILENAME',  'hr_box_asn.fits'            ,' name of file'),
                pyfits.Card('FILETYPE',  'ASN_TABLE'          ,' type of data found in data file'),
                pyfits.Card('TELESCOP',  'HST'                ,' telescope used to acquire data'),
                pyfits.Card('INSTRUME',  'ACS   '             ,' identifier for instrument used to acquire data'),
                pyfits.Card('EQUINOX ',                2000.0 ,' equinox of celestial coord. system'),
                pyfits.Card('ROOTNAME',  'hr_box  '              ,' rootname of the observation set'),
                pyfits.Card('PRIMESI ',  'ACS   '             ,' instrument designated as prime'),
                pyfits.Card('TARGNAME',  'SIM-DITHER'                     ,'proposer\'s target name'),
                pyfits.Card('RA_TARG ',                    0. ,' right ascension of the target (deg) (J2000)'),
                pyfits.Card('DEC_TARG',                    0. ,' declination of the target (deg) (J2000)'),
                pyfits.Card('DETECTOR',  'HRC     '           ,' detector in use: WFC, HRC, or SBC'),
                pyfits.Card('ASN_ID  ',  'hr_box  '           ,' unique identifier assigned to association'),
                pyfits.Card('ASN_TAB ',  'hr_box_asn.fits'         ,' name of the association table')])

def buildAsnTable(asnroot,suffix=None,shiftfile=None,verbose='no'):
    """ Create an association table from all specified files in
    the current directory.  By default, it will include all 'crj' files.
    The user may also specify their own list of files in different ways:
        as a file with a list of filenames using '@filename'
        using wild card specification of filenames, e.g. '*02t?q_flt* or '*f440w*'
        using part of a filename; such as 'f440w' or 'flt' or 'sfl'
        using a simple python list.

    It would also add the columns XOFFSET, YOFFSET and ROTATION
    and populate them with values from a shiftfile if one was specified
    and can be read.

    This function returns the name of the DITHER PRODUCT that was
    created, then inserted into the ASN table. It will be
    the rootname of asnroot with '_drz' appended, following the
    general ACS pipeline filename conventions.

    """
    # Set default specification or simply use one given by user
    if suffix == None:
        files = 'crj'
    else:
        files = suffix

    # Find all files in current directory with given suffix
    flist = _findFiles(files)

    if verbose == 'yes':
        # print out version number for reference
        print 'BuildAsn Version '+__version__
        print 'Building table from the following files: '
        for fl in flist: print fl[0]

    if len(flist) == 0:
        print 'NOTE: Please check suffix used to build list of files.'
        _errmsg = 'Could NOT find any files to build into an association!'
        raise Exception,_errmsg

    # Read in values from shiftfile, if present
    if shiftfile != None:
        # Build dictionary of shift values
        # Set to 'None' upon problems with file
        shiftdict = readShiftFile(shiftfile)

        # Find file with minimum shift, assume it is reference file
        ref_file = _findRefFile(shiftdict,flist)
        # Compute deltas by subtracting commanded offsets from headers
        # from total offsets in shiftdict
        #_computeDeltaShifts(shiftdict,ref_file,flist)
        shframe = shiftdict['frame']
        refimage = shiftdict['refimage']
    else:
        shiftdict = {'units':'pixels','frame':'input','order':{}}
        shframe = None
        refimage = None

    _output = asnroot+'_asn.fits'

    # Open output ASN table file
    fasn = pyfits.HDUList()

    # Build ASN file primary header
    _buildAsnPrimary(fasn,_output,flist[0][0],frame=shframe,refimage=refimage)

    # Make table records
    outTable = _buildTableRows(flist,shiftdict, asnroot)

    # Create an extension HDU which contains a table
    exthdu = _makeTableHDU(outTable)

    fasn.append(exthdu)

    # Close ASN table file
    fasn.writeto(_output)
    fasn.close()
    del fasn

    # Now, add reference WCS extension, if given in shiftfile
    if refimage != None:
        whdu = wcsutil.WCSObject(refimage)
        whdu.createReferenceWCS(_output,overwrite=False)
        del whdu

    # Generate output product name
    dthprod = _buildNewRootname(asnroot,extn='_drz',suffix='_asn')
    dthprod = dthprod+'.fits'

    if verbose == 'yes':
        print 'Product from this ASN table will be named: ',dthprod

    return dthprod

def _findFiles(inlist):
    """ Builds list of all files which contain suffix.

        If a simple python list is supplied it will already
        be the list of files needed to build the ASN table.
    """
    _ldir = os.listdir('.')
    # Determine which form of input was provided:
    #   suffix (crj, raw,...)
    #   wild card list of files (*raw*, *02d?_raw*, ...)
    #   file with list of names (@files.list,...)
    if (isinstance(inlist,list)):
        # We have a simple Python 'list' object
        regpatt = None
        suffix = inlist

    elif inlist.find(',') >=0:
        # We have a comma separated list
        regpatt = None
        inlist = inlist.split(',')
        suffix = inlist

    elif inlist[0] == '@':
        # We have a file list as input
        regpatt = None
        suffix = None

    elif inlist.find('*') >= 0 or inlist.find('?') >= 0:
        # We have a wild card specification
        regpatt = inlist.replace('.','\.')
        regpatt = regpatt.replace('?','.?')
        regpatt = regpatt.replace('*','.*')
        suffix = None

    else:
        # default to assumption of single suffix
        regpatt = '.*'+inlist+'.*$'
        suffix = inlist

    # build file list
    flist = []

    if regpatt:
        # compile regular expression
        _reg = re.compile(regpatt)

        for file in _ldir:
            if _reg.match(file) and file.find('_OrIg') < 0:
            # Determine suffix here...
                suffix = _findSuffix(file)
                # Append tuple with (filename,suffix) to list
                flist.append((file,suffix))
    elif (suffix and not regpatt):
        for file in inlist:
            suffix = _findSuffix(file)
            flist.append((file,suffix))
    else:
        f = open(inlist[1:],'r')
        while 1:
            file = f.readline()
            if not file:
                break

            # Parse the line on spaces
            fline = file.split(' ')

            # Determine suffix here...
            if suffix == None:
                suffix = _findSuffix(fline[0])

            ivmfile = None
            if (len(fline) > 1):
                ivmfile = fline[1].rstrip()

            flist.append((fline[0].rstrip(),suffix,ivmfile))

    del regpatt
    flist.sort()
    return flist

def _findSuffix(fname):

    suffix = None
    for s in EXTLIST:
        _indx = fname.find(s)
        if _indx > 0:
            suffix = s[:-5]
            break

    return suffix


def _updateKeyword(key,inhdr,outhdr,default='UNKNOWN'):
    """ Safely updates keyword key in outhdr from value in inhdr.
        Uses value given by 'default' if keyword is not found in
        input header.
    """
    try:
        _keyw = inhdr[key]
    except KeyError:
        _keyw = default
    outhdr[key] = _keyw


def _findRefFile(sdict,flist):
    """ Function which identifies the image from an association
        which serves as the reference, zero-offset position.
        It simply returns the filename for the image with the
        smallest total shift.
    """
    min_shift = 99999.
    reffile = None

    for img in sdict['order']:
        offx = sdict[img][0]
        offy = sdict[img][1]
        shift = N.sqrt(pow(offx,2) + pow(offy,2))

        if shift < min_shift:
            min_shift = shift
            reffile = img

        for fl in flist:
            if fl[0].find(reffile) > -1:
                reffile = fl[0]

    return reffile


def _computeDeltaShifts(sdict,ref_file,flist):
    """ Function which subtracts commanded offsets from headers
        from total offsets found in sdict.
        Commanded offsets will always be relative to ref_file.
        <<OBSOLETE>>
    """
    # Determine pointing for reference image
    # Open reference image
    fref = pyfits.open(ref_file)
    ref_pos = 0.
    # Get pointing position from SCI extension header
    ref_hdr = fref['SCI',1].header
    ref_pos = (ref_hdr['CRVAL1'],ref_hdr['CRVAL2'])
    ref_cd = (ref_hdr['CD1_1'],ref_hdr['CD1_2'])
    # Close and delete reference image object
    fref.close()
    del fref
    pscale = pow((pow(float(ref_cd[0]),2)+pow(float(ref_cd[1]),2)),0.5) * 3600.

    for img in sdict['order']:
        for fl in flist:
            if fl[0].find(img) > -1:
                imgname = fl[0]

                # Open science image
                fimg = pyfits.open(imgname)
                scihdr = fimg['SCI',1].header
                # Extract commanded position from image header
                img_pos = (scihdr['CRVAL1'],scihdr['CRVAL2'])
                # Done with image: close and delete FITS object
                fimg.close()
                del fimg

                # Compute commanded shift here: image minus reference
                # This has to be in units of pixels
                pos = (N.array(img_pos) - N.array(ref_pos))/pscale
                pos = pos * 3600. * N.cos(ref_pos[1]) * N.array([-1.0,1.0])

                # Compute delta offset: total minus commanded
                delta_pos = N.array((sdict[img][0],sdict[img][1])) - pos

                # Replace total shift with delta in shift dictionary
                sdict[img].append(tuple(delta_pos))

# Build table rows from input data
def _buildTableRows(filelist, shiftdict, dthprod):
    """
        Create dictionary with each entry corresponding to a list of
        values for each column.

    """

    xsh,ysh,dx,dy,rot,scl,name,mtype,mprsnt = [],[],[],[],[],[],[],[],[]
    ot = ord('T')
    of = ord('F')

    for fl in filelist:
        file = fl[0]
        found = 0
        # Do match between filename given in filelist
        # and name provided in shiftfile
        # (which could only be a rootname, or other partial)
        #
        for sfile in shiftdict['order']:
            if file.find(sfile) > -1:
                file = sfile
                found = 1
        if found:
            # Count the number of valid entries for this file...
            _numvars = 0
            for val in shiftdict[file]:
                if val != None: _numvars += 1

            dxshift = 0.
            dyshift = 0.
            xshift = 0.
            yshift = 0.
            if shiftdict['form'] != 'delta':
                xshift = float(shiftdict[file][0])
                yshift = float(shiftdict[file][1])
            else:
                dxshift = float(shiftdict[file][0])
                dyshift = float(shiftdict[file][1])

            if _numvars >= 3:
                rotation = float(shiftdict[file][2])
                scale = 0.0
                if _numvars >= 4:
                    scale = float(shiftdict[file][3])
            else:
                rotation = 0.0
                scale = 0.0
        else:
            print 'No shifts provided for ',file,'.  Assuming (0.,0.)'
            xshift = 0.0
            yshift = 0.0
            dxshift = 0.
            dyshift = 0.
            rotation = 0.0
            scale = 0.0

        xsh.append(xshift)
        ysh.append(yshift)
        dx.append(dxshift)
        dy.append(dyshift)
        rot.append(rotation)
        scl.append(scale)

        # Build rootname for file
        name.append(_buildNewRootname(file,suffix=fl[1]))
        mprsnt.append(ot)
        mtype.append("EXP-DTH")

    # Now add product row, based on output
    name.append(dthprod)
    xsh.append(0.0)
    ysh.append(0.0)
    dx.append(0.)
    dy.append(0.)
    rot.append(0.0)
    scl.append(0.0)

    mtype.append("PROD-DTH")
    mprsnt.append(of)
    table = {'units':shiftdict['units'],'frame':shiftdict['frame'],'xsh':xsh,'ysh':ysh,'dx':dx,'dy':dy,'rot':rot,'scl':scl,'name':name,'mtype':mtype,'mprsnt':mprsnt}

    return table

def _buildNewRootname(filename,extn=None,suffix=None):
    """ Build rootname for a new file.
        Use 'extn' for new filename if given, otherwise
        does NOT append a suffix at all.
        Search for suffix if given.

        Does NOT check to see if it exists already.
        Will ALWAYS return a new filename.
    """
    # Search known suffixes to replace ('_crj.fits',...)
    extlist = copy.deepcopy(EXTLIST)
    # Also, add a default where '_dth.fits' replaces
    # whatever extension was there ('.fits','.c1h',...)
    extlist.append('.')
    if suffix != None:
        extlist.insert(0,suffix)

    for s in extlist:
        _indx = filename.find(s)
        if _indx > 0: break

    if _indx < 0:
         # default to entire rootname
        _indx = len(filename)

    if extn != None:
        output = filename[:_indx]+extn
    else:
        output = filename[:_indx]

    return output

# Make a Table extension HDU
def _makeTableHDU(data):
    """ Create new_table object for ASN table, including definitions
        for optional Offset/Rotation columns.
    """

    # Compute maximum length of MEMNAME for table column definition
    _maxlen = 0
    for _fname in data['name']:
        if len(_fname) > _maxlen: _maxlen = len(_fname)
    # Enforce a mimimum size of 24
    if _maxlen < 24: _maxlen = 24
    namelen_str = str(_maxlen+2)+'A'

    # Column definitions use the FITS Table TFORM value for the format
    col1 = pyfits.Column(name='MEMNAME',format=namelen_str,array=N.char.array(data['name']))
    col2 = pyfits.Column(name='MEMTYPE',format='14A',array=N.char.array(data['mtype']))
    col3 = pyfits.Column(name='MEMPRSNT',format='L',array=N.array(data['mprsnt']).astype(N.uint8))
    # Build columns for optional Offset/Rotation columns
    #col4,col5,col6 = _makeOffsetColumns(numarray.array(data['xsh']), numarray.array(data['ysh']), numarray.array(data['rot']))
    xsh = pyfits.Column(name='XOFFSET',format='E',unit=data['units'],array=N.array(data['xsh']))
    ysh = pyfits.Column(name='YOFFSET',format='E',unit=data['units'],array=N.array(data['ysh']))
    rot = pyfits.Column(name='ROTATION',format='E',unit='degrees',array=N.array(data['rot']))
    dx = pyfits.Column(name='XDELTA',format='E',unit=data['units'],array=N.array(data['dx']))
    dy = pyfits.Column(name='YDELTA',format='E',unit=data['units'],array=N.array(data['dy']))
    scl = pyfits.Column(name='SCALE',format='E',unit='',array=N.array(data['scl']))

    hdu = pyfits.new_table([col1,col2,col3,xsh,ysh,dx,dy,rot,scl],nrows=len(data['name']))

    return hdu


def _buildAsnPrimary(fasn,output,img1,frame=None,refimage=None):
    """ Creates complete Primary header for ASN table based on template.
    'fasn' is the file handle of the newly opened ASN table file.
    Uses keyword values from 'img1' to update ASN header keywords.
    """
    origin_str = 'PyFITS Version '+pyfits.__version__

    # Format time values for keywords IRAF-TLM, and DATE
    _ltime = time.localtime(time.time())
    tlm_str = time.strftime('%H:%M:%S (%d/%m/%Y)',_ltime)
    date_str = time.strftime('%Y-%m-%dT%H:%M:%S',_ltime)


    # Build PRIMARY HDU
    _hdu = pyfits.PrimaryHDU(header=_prihdr)
    fasn.append(_hdu)
    newhdr = fasn['PRIMARY'].header
    # Verify that input image is a FITS file...
    try:
        # Open img1 to obtain keyword values for updating template
        fimg1 = pyfits.open(img1)
        prihdr = fimg1['PRIMARY'].header
        _updateKeyword('INSTRUME',prihdr,newhdr)
        _updateKeyword('PRIMESI',prihdr,newhdr)
        _updateKeyword('TARGNAME',prihdr,newhdr)
        _updateKeyword('DETECTOR',prihdr,newhdr)
        _updateKeyword('RA_TARG',prihdr,newhdr)
        _updateKeyword('DEC_TARG',prihdr,newhdr)
        #_updateKeyword('MEMSUFFIX',suffix,newhdr)
        # All done with input image. Close it now.
        fimg1.close()
        del fimg1
    except:
        pass

    # Update Primary header with values from input image
    newhdr['IRAF-TLM']=tlm_str
    newhdr['DATE'] = date_str
    newhdr['ORIGIN'] = origin_str
    _indx = output.find('.')
    if _indx < 1:
        _indx = len(output)
    newhdr['ROOTNAME'] = output[:_indx]
    newhdr['FILENAME'] = output
    newhdr['ASN_ID'] = output[:_indx]
    newhdr['ASN_TAB'] = output

    # Now, if given, add keywords to allow shifts to be interpreted correctly
    if frame == None: _frame = ''
    else: _frame = frame
    newhdr.update('SHFRAME',_frame,comment="Frame which shifts are measured")

    if refimage == None: _refimg = ''
    else: _refimg = refimage
    newhdr.update('REFIMAGE',_refimg,comment="Image shifts were measured from")

def makeTableDict(asndict):
    """
        Read in the dictionary created by 'readAsnTable' for an existing
        ASN table and convert it to the internal data dictionary used by
        'makeTableHDU'.
    """
    # Initialize empty data dictionary
    tbldict = copy.deepcopy(TABLE_DATA)

    # Create arrays for each column
    xsh,ysh,dx,dy,rot,scl,name,mtype,mprsnt = [],[],[],[],[],[],[],[],[]
    ot = ord('T')
    of = ord('F')

    mem0name = asndict['order'][0]
    # Populate arrays with data from ASN dictionary for input exposures
    for fname in asndict['order']:
        # extract member info
        memdict = asndict['members'][fname]
        name.append(fname)
        xsh.append(memdict['xshift'])
        ysh.append(memdict['yshift'])
        dx.append(memdict['delta_x'])
        dy.append(memdict['delta_y'])
        rot.append(memdict['delta_rot'])
        scl.append(memdict['delta_scale'])
        mtype.append("EXP-DTH")
        mprsnt.append(ot)

    # Now, populate arrays for product...
    name.append(asndict['output'])
    xsh.append(0.0)
    ysh.append(0.0)
    dx.append(0.)
    dy.append(0.)
    rot.append(0.0)
    scl.append(0.0)
    mtype.append("PROD-DTH")
    mprsnt.append(of)

    # Update table dictionary with arrays
    tbldict['units']  = asndict['members'][mem0name]['shift_units']
    tbldict['frame']  = asndict['members'][mem0name]['shift_frame']
    tbldict['xsh']    = xsh
    tbldict['ysh']    = ysh
    tbldict['dx']     = dx
    tbldict['dy']     = dy
    tbldict['rot']    = rot
    tbldict['scl']    = scl
    tbldict['name']   = name
    tbldict['mtype']  = mtype
    tbldict['mprsnt'] = mprsnt

    return tbldict

def writeAsnDict(asndict,output=None):
    """
    writeAsnDict:
    =============
    Write out a new ASN table using a dictionary in memory.
    The input dictionary should be read in using 'readAsnTable',
    and can be modified as needed before writing out the new table.

    SYNTAX:
        buildasn.writeAsnDict(asndict,output=None)

    PARAMETERS:
            asndict:    dictionary from 'readAsnTable
            output:     rootname or filename for output ASN file
        if output == None (default), product name from asndict will be
        used to define the output filename for the table.
    """

    # Convert the ASN dictionary from 'readAsnTable'
    # into a format usable by 'buildasn' functions.
    tbldict = makeTableDict(asndict)

    # Extract info from table necessary for writing it out
    if output == None:
        outfile = asndict['output']+'_asn.fits'
    else:
        if output.find('_asn.fits') < 0:
            outfile = output+'_asn.fits'
        else:
            outfile = output

    # Delete the file if it exists.
    if os.path.exists(outfile):
        warningmsg =  "\n#########################################\n"
        warningmsg += "#                                       #\n"
        warningmsg += "# WARNING:                              #\n"
        warningmsg += "#  The exisiting assocation table,      #\n"
        warningmsg += "           " + str(outfile) + '\n'
        warningmsg += "#  is being replaced by buildasn.       #\n"
        warningmsg += "#                                       #\n"
        warningmsg += "#########################################\n\n"
        print warningmsg
        os.remove(outfile)

    mem0name = asndict['order'][0]
    refimg = asndict['members'][mem0name]['refimage']
    shframe = tbldict['frame']

    # Input image to be used as template
    fname = buildRootname(mem0name)

    # Open output ASN table file
    fasn = pyfits.HDUList()

    # Build ASN file primary header
    _buildAsnPrimary(fasn,outfile,fname,frame=shframe,refimage=refimg)

    # Create an extension HDU which contains a table
    exthdu = _makeTableHDU(tbldict)

    fasn.append(exthdu)

    # Close ASN table file
    fasn.writeto(outfile)
    fasn.close()
    del fasn

    return outfile

def help():
    _str = """
    buildAsnTable:
    ==============
    Create an association table from all specified files in
    the current directory. It returns the filename of the final product
    for the table, created by appending '_drz.fits' to the given rootname.

    The columns XOFFSET, YOFFSET and ROTATION are added to this table
    and populated with values from a shiftfile, if one was specified
    and can be read. It should be in the format used by 'dither.shiftfind'
    or 'dither.avshift' from the STSDAS DITHER package.  These values should
    be offsets in units of arcseconds in RA and Dec.

    SYNTAX:
        dthprod = buildasn.buildAsnTable(rootname,suffix=None,shiftfile=None)
      where
        rootname  - user-specified string for creating the output table name
        suffix      - look for user-specified files as inputs for table
                    either as a filelist ('@filename'),
                    wild-card list ('*8cd*flt*'), or partial ('crj' or 'f440w')
        shiftfile - name of file containing shifts to be used by PyDrizzle
        dthprod   - full filename of final product for association table
        verbose   - print additional info on files used to build ASN
    EXAMPLE:
      Create ASN table 'mymosaic_asn.fits' from all f440w FLT images.
        >>> import buildasn
        >>> dthprod = buildasn.buildAsnTable('mymosaic',suffix='*f440w*flt*')
    """

    print 'Help for buildasn version '+__version__
    print _str
    print writeAsnDict.__doc__
