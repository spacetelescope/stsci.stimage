""" updateasn.py - Module to update an ACS-style association
    table.

W.J. Hack 25 Jul 2002
    Initial Version
Version 0.5 (11-May-2005) (WJH) - Updated to package reference WCS as extension
                of ASN table when a shiftfile has been provided.

"""
import os,string
import pyfits, numarray

import pydrizzle
from pydrizzle import buildasn,fileutil,wcsutil

IDCKEYS = {'WFPC2':'cubic','ACS':'idctab','STIS':'cubic','NICMOS':'cubic','detector':'idctab'}
yes = True
no = False

__version__ = '0.6 (18-July-2005)'

def updateAsnTable(tabname, rootname, xsh=None, ysh=None, form="absolute",
    rot=None, scale=None, frame="input", units="pixels", output=None, mode="replace"):
    """ Updates an existing ASN table with shifts in arcseconds of RA and Dec.
        Parameters:
            tabname     - name of ASN table to be updated
            rootname    - name of image whose pointing needs to be updated
            xsh         - X shift from nominal to be applied
            ysh         - Y shift from nominal to be applied
            rot         - additional rotation to be applied
            scale       - scale factor to be applied
            form        - specifies how shifts are computed:
                            absolute (default) or delta
            frame       - specifies whether xsh/ysh are given in terms of
                            input (default) or output frame
            units       - specifies units of xsh/ysh:
                            pixels (default) or arcseconds
            output      - filename (with or without path) of output image
                          from which output pixels shifts were computed.
            mode        - replace or add shifts to existing values
                            'replace' (default) or 'sum'
        The task will:
            - open the ASN file (tabname)
            - search for the row matching the rootname provided by
                the user (img),
            - convert any pixel shifts to arcseconds
            - update the XOFFSET, YOFFSET, ROTATION columns
                (or create them if they don't already exist)

        This task assumes that the shifts given as 'input' pixels are
        already distortion-corrected, but not scaled/rotated to output frame.

        The conversion of shifts given in terms of output pixel coordinates
        requires the use of the WCS from the output frame in order to correctly
        take into account the output frame's pixel scale and orientation.

        Both input and output image specifications assume '.fits' and that
        the WCS information appropriate for this are in either [0] or [sci,1].
        If necessary a different extension can be specified in the filename.
        Also, the output filename can include a path which will be used.

    EXAMPLE:
            1. Update ASN table 'j8cw03020_asn.fits' for 'j8cw03021_crj.fits'
                which has pixel shifts of (10.3,4.72) in (undistorted) input
                frame.
            >>> updateasn.updateAsnTable('j8cw03020_asn.fits','j8cw03021',
                xsh=10.3,ysh=4.72,rot=0.14)
    NOTE:
        This task still does not correctly work with 'NaN' entries in
        the table.  For those values, it will not change the values at all.

    """
    if frame == 'output' and output == None:
        print 'Please specify output image needed to compute deltas...'
        return

    print 'updateAsnTable Version ',__version__

    _tmpname = 'buildasn_'+tabname
    # Start by opening the ASN table
    _asntab = pyfits.open(tabname,'update')

    if form == 'absolute' or form == 'relative':
        # absolute/relative shifts
        _colx = 'XOFFSET'
        _coly = 'YOFFSET'

    else:
        # delta shifts
        _colx = 'XDELTA'
        _coly = 'YDELTA'
    #
    # Work out what needs to be done to update ASN table
    # Do the XOFFSET/YOFFSET/ROTATION columns already exist or not?
    #
    _update = no
    _add_rot = yes
    _add_scale = yes
    _colnum = 0
    for name in _asntab[1].columns.names:
        if name == _colx:
            # Make the assumption that XOFFSET/YOFFSET and ROTATION
            # are all present if one is present...
            _update = yes
            break
        _colnum += 1

    if _update:
        _tab_units = _asntab[1].columns.units[_colnum]
        if _tab_units == '' or _tab_units == None: _tab_units = 'pixels'
    else:
        _tab_units = units

    for name in _asntab[1].columns.names:
        if name == 'ROTATION':
            _add_rot = no
            break
    for name in _asntab[1].columns.names:
        if name == 'SCALE':
            _add_scale = no
            break

    # Get the shift frame and refimage if present
    # If keywords don't exist, assume default of 'input' shifts
    # with no need for refimage.
    try:
        _tab_frame = _asntab[0].header['shframe']
    except KeyError:
        _tab_frame = 'input'

    try:
        _tab_refimg = _asntab[0].header['refimage']
    except KeyError:
        _tab_refimg = None

    # Find row which corresponds to input image
    _rownum = 0
    for row in _asntab[1].data:
        if string.find(rootname,row.field('MEMNAME')) > -1:
            _row = row
            break
        _rownum += 1
    print 'Updating table row ',_rownum,' for image: ',rootname

    #
    # Determine where to get the distortion coefficients
    # create complete filename from given rootname
    img = fileutil.buildRootname(rootname)
    if img == None:
        print 'No valid input image for filename ',rootname
        raise ValueError

    _instrument = fileutil.getKeyword(img,'instrume')
    if not IDCKEYS.has_key(_instrument): _instrument = 'detector'
    _key = IDCKEYS[_instrument]
    if xsh == None: xsh = 0.
    if ysh == None: ysh = 0.

    # If we have any trouble computing offsets, abort and
    # do NOT update table.
    try:
        # Start by determining values to be used to update the table.
        #
        # Need to compute: _delta_ra, _delta_dec (undistorted arcsec)
        #
        if units == _tab_units or (xsh == 0. and ysh == 0.):
            _delta_xoff = xsh
            _delta_yoff = ysh
            if rot == None:
                rot = 0.
            if scale == None:
                scale = 1.0

        elif units.find('pixels') > -1 and _tab_units.find('arcsec') > -1:
            #
            # We need to convert the pixel values to arcseconds
            #
            _wcs = _getExposure(img,output,frame,_key)

            # If there is any additional rotation, account for it first
            if rot != None:
                if scale != None:
                    pscale = scale * _wcs.pscale
                else:
                    pscale = None
                _wcs.updateWCS(orient=_wcs.orient+rot,pixel_scale=pscale)

            # determine delta CRVALs from CRPIX+(xsh,ysh)
            _rd_delta = _wcs.xy2rd((_wcs.crpix1+xsh,_wcs.crpix2+ysh))

            # Now, convert from degrees to arcseconds
            #### Do we need to multiply by cos(dec) for delta(RA)???
            _delta_xoff = (_rd_delta[0] - _wcs.crval1) * 3600.
            _delta_yoff = (_rd_delta[1] - _wcs.crval2) * 3600.

        elif units.find('arcsec') > -1 and _tab_units.find('pixels') > -1:
            _wcs = _getExposure(img,_tab_refimg,_tab_frame,_key)
            # If there is any additional rotation, account for it first
            if rot != None:
                if scale != None:
                    pscale = scale * _wcs.pscale
                else:
                    pscale = None
                _wcs.updateWCS(orient=_wcs.orient+rot,pixel_scale=pscale)

            _xy_delta = _wcs.rd2xy((_wcs.crval1+xsh,_wcs.crval2+ysh))
            _delta_xoff = (_xy_delta[0] - _wcs.crpix1)
            _delta_yoff = (_xy_delta[1] - _wcs.crpix2)

    except:
        print 'ERROR: Error in updating table ',tabname
        print 'Closing existing table without updating it...'
        _asntab.close()
        return

    if _update:
        # Offset columns exist, just update the values in the table
        if mode == 'sum':
            _shift_xoff = _asntab[1].data.field(_colx)[_rownum]
            _shift_yoff = _asntab[1].data.field(_coly)[_rownum]
            _shift_rot = _asntab[1].data.field('ROTATION')[_rownum]
            _shift_scale = _asntab[1].data.field('SCALE')[_rownum]

            # Account for INDEF values in table
            # Logic: If set to INDEF, adding 1.0 will not change its value
            if _shift_xoff+1.0 == _shift_xoff: _shift_xoff = 0.
            if _shift_yoff+1.0 == _shift_yoff: _shift_yoff = 0.
            if _shift_rot+1.0 == _shift_rot : _shift_rot = 0.
            if _shift_scale+1.0 == _shift_scale: _shift_scale = 1.0
        else:
            _shift_xoff = 0.
            _shift_yoff = 0.
            _shift_rot = 0.
            _shift_scale = 1.

        _asntab[1].data.field(_colx)[_rownum] = _shift_xoff + _delta_xoff
        _asntab[1].data.field(_coly)[_rownum] = _shift_yoff + _delta_yoff
        if rot != None:
            _asntab[1].data.field('ROTATION')[_rownum] = _shift_rot + rot
        if scale != None:
            _asntab[1].data.field('SCALE')[_rownum] = _shift_scale * scale
    else:
        # We need to add the extra columns to the table
        # Build arrays for each additional column
        _numrows = len(_asntab[1].data)
        _xsh = numarray.zeros(_numrows,type=numarray.Float32)
        _ysh = numarray.zeros(_numrows,type=numarray.Float32)

        # Now update entry associated with this particular image
        _xsh[_rownum] = _delta_xoff
        _ysh[_rownum] = _delta_yoff

        # Build the column objects for the table
        #_xc,_yc,_rc = buildasn._makeOffsetColumns(_xsh,_ysh,_rot)
        _xc = pyfits.Column(name=_colx,format='E',unit=_tab_units,array=_xsh)
        _yc = pyfits.Column(name=_coly,format='E',unit=_tab_units,array=_ysh)
        _newcols = _asntab[1].columns + _xc
        _newcols += _yc
        #_asndefs = _asntab[1].get_coldefs()
        #_asndefs.add_col(_xc)
        #_asndefs.add_col(_yc)

        # If rotation column does NOT exist,
        if _add_rot:
            _rot = numarray.zeros(_numrows,type=numarray.Float32)
            if rot != None:
                _rot[_rownum] = rot
            _rc = pyfits.Column(name='ROTATION',format='E',unit='degrees',array=_rot)
            #  add newly created column
            #_asndefs.add_col(_rc)
            _newcols += _rc
        else:
            # else, update column directly.
            _asntab[1].data.field('ROTATION')[_rownum] = rot

        # If scaling column does NOT exist,
        if _add_scale:
            _scale = numarray.zeros(_numrows,type=numarray.Float32)
            if scale != None:
                _scale[_rownum] = scale
            _sc = pyfits.Column(name='SCALE',format='E',unit='',array=_scale)
            _newcols += _sc
        else:
            # else, update column directly.
            _asntab[1].data.field('SCALE')[_rownum] = scale

        _asnhdu = pyfits.new_table(_newcols)
        _asnhdu.writeto(_tmpname)
        # remove old data
        #del _asntab[1]
        # add table extension with old data and new columns
        #_asntab.append(_asnhdu)

    # Close and clean-up
    _asntab.close()
    del _asntab
    if os.path.exists(_tmpname):
        os.remove(tabname)
        os.rename(_tmpname,tabname)

def _getExposure(img,output,frame,idckey):

    # setup Exposure object with distortion model and WCS info
    if frame == 'input':
        if string.find(img,'[') < 0:
            img += '[sci]'
            _val = fileutil.getKeyword(img,'CD1_1')
            if _val == None:
                img[:-7] += '[0]'

        if fileutil.getKeyword(img,'CD1_1') == None:
            print "Input %s not valid!"%img
            raise Exception

        _exp = pydrizzle.Exposure(img,idckey)
        _wcs = _exp.geometry.wcslin

    else:
        # Use output frame for converting pixel shifts
        # to units of arcseconds...
        # Make sure we have a recognized file:
        if not fileutil.findFile(output):
            if fileutil.findFile(output+'.fits'):
                output += '.fits'
            else:
                print "Can NOT find output file %s!"%output
                raise Exception

        if string.find(output,'[') < 0:
            output += '[0]'
            _val = fileutil.getKeyword(output,'CD1_1')
            if _val == None:
                output[:-3] += '[sci,1]'

        if fileutil.getKeyword(output,'CD1_1') == None:
            print "Output %s not valid!"%output
            raise Exception

        _exp = pydrizzle.Exposure(output,idckey)
        _wcs = _exp.geometry.wcs

    return _wcs

def updateShifts(tabname,shiftfile, mode='replace',clean=no):
    """ Update an ASN table with shifts provided in shiftfile.
        The 'units', 'frame', and 'reference' will all be derived
        from the shiftfile itself, with defaults set by 'readShiftFile'.

        If 'clean' == yes, then remove the shiftfile and any
        reference WCS file after they have been used
        to update the ASN table.
    """

    # Read in the shifts from the shiftfile
    sdict = buildasn.readShiftFile(shiftfile)

    # Apply user-specified values
    # Values from shiftfile ALWAYS take precedence
    if sdict.has_key('units'):
        _units = sdict['units']
    else:
        _units = None

    if sdict.has_key('frame'):
        _frame = sdict['frame']
    else:
        _frame = None

    _reference = sdict['refimage']

    if _frame == 'output' and _reference == None:
        # No reference frame specified, yet shifts given in
        # output units, so raise an Exception
        print "No reference image provided for shifts given in output units!"
        raise ValueError

    # Extract list of rootnames from ASN table
    ftab = pyfits.open(tabname)
    fnames = ftab[1].data.field('memname')
    ftab.close()
    del ftab

    # For each rootname in ASN table, apply shifts from shiftfile
    for mname in fnames:
        _lname = mname.lower()
        for key in sdict.keys():
            # Is this key the one that goes with 'mname'?
            if key.find(_lname) > -1:
                _s = sdict[key]

                updateAsnTable(tabname,mname,xsh=_s[0],ysh=_s[1],rot=_s[2],
                    scale=_s[3],form=sdict['form'],frame=_frame,units=_units,
                    output=_reference,mode=mode)

    # If we have a new reference image, update header keyword to point
    # to appended WCS object
    if _reference != None:
        # Append reference WCS to ASN table
        whdu = wcsutil.WCSObject(_reference)
        whdu.createReferenceWCS(tabname,overwrite=no)
        # Now update ASN header to point to new reference WCS
        ftab = pyfits.open(tabname,mode='update')
        ftab['primary'].header.update('refimage', tabname+"[wcs]")
        ftab.close()
        del ftab
        del whdu

    # If user specifies, remove shiftfile and reference WCS
    # after it has been appended to ASN table.
    if clean == yes:
        os.remove(shiftfile)
        if _reference != None:
            _refname,_refextn = fileutil.parseFilename(_reference)
            os.remove(_refname)

def help():
    print updateAsnTable.__doc__
