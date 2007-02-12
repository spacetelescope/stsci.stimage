"""
These functions rely on booleans for 'yes' and 'no', PyFITS and readgeis.
"""
import pyfits, readgeis
import string,os,types
import numpy as N
import fileutil

# Required keywords to build an extension with WCS info
DTH_KEYWORDS=['CD1_1','CD1_2', 'CD2_1', 'CD2_2', 'CRPIX1',
'CRPIX2','CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2']
RESERVED_KEYS = ['NAXIS','BITPIX','DATE','IRAF-TLM']

yes = True
no = False


def buildDthProduct (pardict, output, extlist=None, outlist=None, wcs=None):
    """
    Parameters:
        pardict - a dictionary containing (at least):
            'data','outdata','outweight','outcontext'
            where 'data' serves as the name of the backup template image,
                and the others are filenames of the drizzle products.
        output - filename of the final combined output image
        extlist - list of EXTNAME's to be searched in the template image
        outlist - list of EXTNAME's to be used for naming the output extensions

    This function will package the two or three output files from
    'drizzle' or 'tdrizzle' into a single output multi-extension
    FITS file.
    It uses a pre-existing multi-extension FITS file as a template
    for the output product.
    The list 'extlist' contains the names of the extensions from the
    template image which will be used to build the output file.

    A number of keywords are also updated based on values from the
    drizzle products, including the CD matrix, CRPIX[1,2], CRVAL[1,2].
    In addition, the template images will be of different size than
    the drizzle products, yet this is handled automatically by PyFITS.

    NOTE:
    The output file will ALWAYS contain 'SCI','WHT', and 'CTX'
    extensions.
    """
    # Set up default extlist list
    if extlist == None:
        extlist = ('SCI','ERR','DQ')
    if outlist == None:
        outlist = ('SCI','WHT','CTX')

    # Get default headers from multi-extension FITS file
    # If input data is not in MEF FITS format, it will return 'None'
    # and those headers will have to be generated from drizzle output
    # file FITS headers.
    # NOTE: These are HEADER objects, not HDUs
    prihdr,scihdr,errhdr,dqhdr = getTemplates(pardict['data'],extlist)

    if prihdr == None:
        # Use readgeis to get header for use as Primary header.
        _indx = pardict['data'].find('[')
        if _indx < 0:
            _data = pardict['data']
        else:
            _data = pardict['data'][:_indx]

        fpri = readgeis.readgeis(_data)
        prihdr = fpri[0].header.copy()
        fpri.close()
        del fpri


    # Setup primary header as an HDU ready for appending to output FITS file
    prihdu = pyfits.PrimaryHDU(header=prihdr,data=None)

    # Start by updating PRIMARY header keywords...
    prihdu.header.update('EXTEND',pyfits.TRUE)
    prihdu.header.update('NEXTEND',3)
    prihdu.header.update('FILENAME', output)

    # Update the ROOTNAME with the new value as well
    _indx = output.find('_drz')
    if _indx < 0:
        prihdu.header.update('ROOTNAME', output)
    else:
        prihdu.header.update('ROOTNAME', output[:_indx])


    # Open the dither output SCI image
    fsci = readgeis.readgeis(pardict['goutdata'])
    inhdr = fsci[1].header.copy()

    # Get the total exposure time for the image
    # If not calculated by PyDrizzle and passed through
    # the pardict, then pulled from the template image.
    if pardict.has_key('texptime'):
        _exptime = pardict['texptime']
        _expstart = pardict['expstart']
        _expend = pardict['expend']
    else:
        _exptime = fsci[0].header['EXPTIME']
        _expstart = fsci[0].header['EXPSTART']
        _expend = fsci[0].header['EXPEND']

    prihdu.header.update('EXPTIME', _exptime)
    #prihdu.header.update('TEXPTIME',_exptime)
    prihdu.header.update('EXPSTART', _expstart)
    prihdu.header.update('EXPEND', _expend)

    #Update ASN_MTYPE to reflect the fact that this is a product
    # Currently hard-wired to always output 'PROD-DTH' as MTYPE
    prihdu.header.update('ASN_MTYP', 'PROD-DTH')

    # Update DITHCORR calibration keyword if present
    # Remove when we can modify FITS headers in place...
    if prihdu.header.has_key('DRIZCORR') > 0:
        prihdu.header['DRIZCORR'] = 'COMPLETE'
    if prihdu.header.has_key('DITHCORR') > 0:
        prihdu.header['DITHCORR'] = 'COMPLETE'

    # Append drizzle output keywords to PRIMARY header
    appendDrizCards(prihdu.header, fsci[0].header)
    #
    # Now, build full SCI extension HDU
    if scihdr == None:
        # Use drizzle header directly for new output header
        scihdr = inhdr
        # Append gsci to gpri
        for _card in fsci[0].header.ascard:
            if _card.key not in RESERVED_KEYS and prihdu.header.has_key(_card.key) == 0:
                scihdr.ascard.append(_card)

    # Now, build the output file
    fo = pyfits.open(output,'append')
    # Add primary header to output file...
    fo.append(prihdu)

    del scihdr['OBJECT']
    if scihdr.has_key('CCDCHIP'): scihdr.update('CCDCHIP','-999')

    if wcs != None:
        # Update ORIENTAT based on PyDrizzle product's value
        # since 'drizzle' itself doesn't update that keyword.
        scihdr.update('ORIENTAT',wcs.orient)

    hdu = pyfits.ImageHDU(data=fsci[1].data,header=scihdr,name=outlist[0])
    updateDTHKeywords(hdu.header,fsci[1].header,output)
    fo.append(hdu)
    fo.flush()

    _shape = fsci[1].data.shape

    del fsci[1].data
    fsci.close()
    del fsci
    del hdu

    # Write out the WEIGHT image in the ERR array extension
    fweight = readgeis.readgeis(pardict['goutweight'])
    _drizcards = None
    if errhdr == None:
        errhdr = fweight[1].header.copy()
        # Append gsci to gpri
        for _card in fweight[0].header.ascard:
            if _card.key not in RESERVED_KEYS and prihdu.header.has_key(_card.key) == 0:
                errhdr.ascard.append(_card)
    errhdr.update('CCDCHIP','-999')

    hdu = pyfits.ImageHDU(data=fweight[1].data,header=errhdr,name=outlist[1])
    updateDTHKeywords(hdu.header,fweight[1].header,output)

    fo.append(hdu)
    fo.flush()

    del fweight[1].data
    fweight.close()
    del fweight
    del hdu

    # Write out the Context image (if any was created)
    _ctx = yes
    if fileutil.findFile(pardict['goutcontext']):
        fctx = readgeis.readgeis(pardict['goutcontext'])
        if dqhdr == None:
            dqhdr = fctx[1].header.copy()
            # Append gsci to gpri
            for _card in fctx[0].header.ascard:
                if _card.key not in RESERVED_KEYS and prihdu.header.has_key(_card.key) == 0:
                    dqhdr.ascard.append(_card)

        hdu = pyfits.ImageHDU(data=fctx[1].data,header=dqhdr,name=outlist[2])
        updateDTHKeywords(hdu.header,fctx[1].header,output)

    else:
        _ctx = no

        # Use the SCI HDU for the shape and build a default array
        imarr = N.ones(shape=_shape,dtype=N.int16)
        imarr = N.reshape(imarr,_shape)
        if dqhdr == None:
            dqhdr = scihdr
            try:
                dqhdr.update('EXTVER',scihdr['EXTVER'])
            except:
                # If this keyword doesn't exist, simply pass...
                pass
        hdu = pyfits.ImageHDU(data=imarr,header=dqhdr,name=outlist[2])
        print 'Dither Product: writing out empty context extension.'

    fo.append(hdu)
    # Close the output and template file
    print 'Finished creating FINAL dither product ',output

    fo.close()
    del fo
    if _ctx:
        del fctx[0].data
        fctx.close()
        del fctx

def appendDrizCards(extnhdr, drizhdr):
    """ Copy over the Drizzle keywords from the drizzle output image
        to the output FITS extension header.
    """

    _drizcards = None
    # Copy out unique 'drizzle' information into new header
    _dc = 0
    for _dcard in drizhdr.ascard:
        if _dcard.key == 'NDRIZIM':
            break
        _dc += 1
    _drizcards = drizhdr.ascard[_dc:]
    for _dzcard in _drizcards:
        extnhdr.ascard.append(_dzcard)


def getTemplates(fname,extlist):
    # Obtain default headers for output file
    # If the output file already exists, use it
    # If not, use an input file for this information.
    #
    # NOTE: Returns 'pyfits.Header' objects, not HDU objects!
    #
    if fname == None:
        print 'No data files for creating FITS output.'
        raise Exception

    if fname.find('.fits') > 0:
        # Open an calibrated ACS image as a template
        _indx = fname.find('[')
        if _indx > 0:
            template = fname[:_indx]
        else:
            template = fname
        ftemplate = pyfits.open(template,mode='readonly',memmap=0)

        # Setup which keyword we will use to select each
        # extension...
        _extkey = 'EXTNAME'

        #
        # Now, extract the headers necessary for output (as copies)
        # 1. Find the SCI extension in the template image
        # 2. Make a COPY of the extension header for use in new output file
        prihdr = pyfits.Header(cards=ftemplate['PRIMARY'].header.ascard.copy())
        extnum = fileutil.findKeywordExtn(ftemplate,_extkey,extlist[0])
        scihdr = pyfits.Header(cards=ftemplate[extnum].header.ascard.copy())

        extnum = fileutil.findKeywordExtn(ftemplate,_extkey,extlist[1])
        errhdr = pyfits.Header(cards=ftemplate[extnum].header.ascard.copy())
        extnum = fileutil.findKeywordExtn(ftemplate,_extkey,extlist[2])
        dqhdr = pyfits.Header(cards=ftemplate[extnum].header.ascard.copy())

        ftemplate.close()
        del ftemplate

    else:
        # Create default headers from scratch
        prihdr = None
        scihdr = None
        errhdr = None
        dqhdr = None

    # Now, safeguard against having BSCALE and BZERO
    try:
        del scihdr['bscale']
        del scihdr['bzero']
        del errhdr['bscale']
        del errhdr['bzero']
        del dqhdr['bscale']
        del dqhdr['bzero']
    except:
        # If these don't work, they didn't exist to start with...
        pass


    # At this point, check errhdr and dqhdr to make sure they
    # have all the requisite keywords (as listed in updateDTHKeywords).
    # Simply copy them from scihdr if they don't exist...
    if errhdr != None and dqhdr != None:
        for keyword in DTH_KEYWORDS:
            if not errhdr.has_key(keyword):
                errhdr.update(keyword,scihdr[keyword])
            if not dqhdr.has_key(keyword):
                dqhdr.update(keyword,scihdr[keyword])

    return prihdr,scihdr,errhdr,dqhdr

def updateDTHKeywords(hdr,extnhdr,filename):
    """ Update header keywords in output to reflect values from
     the dither product.
    """
    for keyword in DTH_KEYWORDS:
        hdr.update(keyword,extnhdr[keyword])
    hdr.update('EXTVER', 1)


def geisToFITS(geisname, fitsname):
    """ Converts GEIS file to Simple FITS file using readgeis."""
    _geiskeys = ['XTENSION','PCOUNT','DATAMAX','DATAMIN','GCOUNT']
    _fitskeys = ['GROUPS','INHERIT','EXTNAME']

    # If there is no file to convert, return...
    if geisname == None or geisname == '' or not fileutil.findFile(geisname):
        return

    # Open input and output files
    geisimg = readgeis.readgeis(geisname)
    fimg = pyfits.open(fitsname,'append')

    # Create new Primary HDU with data from extension
    fpri = pyfits.PrimaryHDU(header=geisimg[0].header,data=geisimg[1].data)

    #
    # We need to work around a problem with HEDIT to avoid having it
    # remove keywords and shrink the header to leave an empty block
    # between the end card and the start of the image data. WJH 21-May-03
    #
    #
    # Append all cards from extension header to Primary header
    for _card in geisimg[1].header.ascard:
        # Skip any GEIS-specific keywords
        if _card.key in _geiskeys:
            continue
        # Add any unique keywords, and update existing ones
        if fpri.header.has_key(_card.key):
            fpri.header[_card.key] = _card.value
        else:
            fpri.header.ascard.append(_card)
    # Now remove extraneous FITS extension keywords
    for _key in _fitskeys:
        del fpri.header[_key]

    # Add Primary HDU to PyFITS object
    fimg.append(fpri)
    # Write output and close both input and output files
    fimg.close()
    del geisimg[1].data
    geisimg.close()
    # Cleanup...
    del fimg, geisimg
