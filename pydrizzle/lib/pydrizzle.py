from drutil import DEFAULT_IDCDIR
from pytools import fileutil, wcsutil, asnutil

import string, os, types, sys
import shutil
import arrdriz
import outputimage
from pattern import *
from observations import *
#import buildasn
#import p_m_input

import pyfits
import numpy as N

yes = True  # 1
no = False  # 0
# default_pars is passed as a default parameter dictionary in  selectInstrument.
# Looks like this is not needed any more. 'default_rot' was originally 'rot'.
_default_pars = {'psize':None,'default_rot':None,'idckey':None}

INSTRUMENT = ["ACS","WFPC2","STIS","NICMOS","WFC3"]

__version__ = "6.3.3 (23-Jan-2010)"


class _PyDrizzle:
    """
Program to process and/or dither-combine image(s) using (t)drizzle.
To create an object named 'test' that corresponds to a drizzle product:
    --> test = pydrizzle.PyDrizzle(input)
where input is the FULL filename of an ACS observation or ASN table.
This computes all the parameters necessary for running drizzle on all
the input images.  Once this object is created, you can run drizzle using:
    --> test.run()

The 'clean()' method can be used to remove files which would interfere with
running Drizzle again using the 'run()' method after a product has already
been created.

Optional parameters:
    output      User-specified name for output products
    field       User-specified parameters for output image
                includes: psize, orient, ra, dec, shape
    units       Units for final product: 'counts' or 'cps'(DEFAULT)
    section     Extension/group to be drizzled: list or single FITS extension(s)
                 or group(s) syntax ('1' or 'sci,1') or None (DEFAULT: Use all chips).
    kernel      Specify which kernel to use in TDRIZZLE
                'square'(default),'point','gaussian','turbo','tophat'
    pixfrac     drizzle pixfrac value (Default: 1.0)
    idckey      User-specified keyword for determining IDCTAB filename
                'IDCTAB'(ACS default),'TRAUGER'(WFPC2),'CUBIC'(WFPC2)
    idcdir      User-specified directory for finding coeffs files:
                'drizzle$coeffs' (default)
    bits_single Specify DQ values to be considered good when
                    drizzling with 'single=yes'. (Default: 0)
    bits_final  Specify DQ values to be considered good when
                    drizzling with 'single=no'. (Default: 0)
    updatewcs   Flag whether to run makewcs on the input (Default: True)
Bits parameters will be interpreted as:
    None - No DQ information to be used, no mask created
    Int  - sum of DQ values to be considered good

Optional Parameters for '.run()':
    build       create multi-extension output: yes (Default) or no
    save        keeps the individual inputs from drizzle: yes or no (Default)
    single      drizzle to separate output for each input: yes or no (Default)
    blot        run blot on drizzled products: yes or no (Default)
    clean       remove coeffs and static mask files: yes or no (Default)

Optional Parameters for '.clean()':
    coeffs      Removes coeffs and static mask files: yes or no (Default)
    final       Removes final product: yes or no (Default)

Usage of optional parameters:
    --> test = pydrizzle.PyDrizzle('test_asn.fits',units='counts')
To keep the individual 'drizzle' output products:
    --> test.run(save=yes)

Output frame parameters can be modified 'on-the-fly' using 'resetPars'.
Given an already drizzled image 'refimg_drz.fits' as a reference,
reset drizzle parameters using:
    --> wcsref = wcsutil.WCSObject('refimg_drz.fits[sci,1]')
    --> f = pydrizzle.SkyField(wcs=wcsref)
Use either:
    --> test.resetPars(wcsref)
Or:
    --> test.resetPars(f)
Return to default parameters using no parameters at all:
    --> test.resetPars()
More help on SkyField objects and their parameters can be obtained using:
    --> f.help()
    """
    def __init__(self, input, output=None, field=None, units=None, section=None,
        kernel=None,pixfrac=None,bits_final=0,bits_single=0,
        wt_scl='exptime', fillval=0.,idckey='', in_units='counts',
        idcdir=DEFAULT_IDCDIR,memmap=0,dqsuffix=None):

        if idcdir == None: idcdir = DEFAULT_IDCDIR
        

        print 'Starting PyDrizzle Version ',__version__,' at ', _ptime()


        self.input = input
        self.output = output
        asndict = input

        # Insure that the section parameter always becomes a list
        if isinstance(section,list) == False and section != None:
            section = [section]
            
        # Set the default value for 'build'
        self.build = yes

        # Extract user-specified parameters, if any have been set...
        # 'field' will be a SkyField object...
        if field != None:
            psize = field.psize
            orient = field.orient
            orient_rot = orient
        else:
            psize = None
            orient = None
            orient_rot = 0.
        
        #self.pars['orient_rot'] was originaly self.pars['rot']. It is set based on 
        # orientat but it looks like it is not used. In SkyField.mergeWCS() rot is 
        #defined again based on orientat. 
        
        # These can also be set by the user.
        # Minimum set needed: psize, rot, and idckey
        self.pars = {'psize':psize,'units':units,'kernel':kernel,'orient_rot':orient_rot,
            'pixfrac':pixfrac,'idckey':idckey,'wt_scl':wt_scl,
            'fillval':fillval,'section':section, 'idcdir':idcdir+os.sep,
            'memmap':memmap,'dqsuffix':dqsuffix, 'in_units':in_units,
            'bits':[bits_final,bits_single], 'mt_wcs': None}
        
        # Watch out for any errors.
        # If they arise, all open files need to be closed...
        self.observation = None
        self.parlist = None
        
        if len(asndict['order']) > 1:
            self.observation = DitherProduct(asndict,pars=self.pars)
        else:
            inroot = asndict['order'][0]
            pardict = asndict['members'][inroot]
            infile = fileutil.buildRootname(inroot)
            if infile == None:
                raise IOError,'No product found for association table.'
            # Append user specified parameters to shifts dictionary
            pardict.update(self.pars)
            #self.observation = selectInstrument(infile,output,pars=pardict)
            self.observation = selectInstrument(infile,self.output,pars=pardict)
        """
        if self.output == None:
            self.output = fileutil.buildNewRootname(asndict['output'],extn='_drz.fits')
            print 'Setting up output name: ',output
        else:
        self.output = output
        """
        # This call puts together the parameters for the input image
        # with those for the output to create a final parameter list
        # for running 'drizzle'.
        # It relies on the buildPars() methods for each exposure to
        # generate a complete set of parameters for all inputs
        #
        self.translateShifts(asndict)
        self.parlist = self.observation.buildPars(ref=field)

        # Something went wrong, so we need to make sure all file
        # handles get closed...
        #self.close()
        #print 'PyDrizzle could not initialize due to errors.'
        self.observation.closeHandle()

        # Let the user know parameters have been successfully calculated
        print 'Drizzle parameters have been calculated. Ready to .run()...'
        print 'Finished calculating parameters at ',_ptime()
    
    def translateShifts(self, asndict):
        """
        Translate the shifts specified in the ASNDICT (as read in from the 
        shiftfile) into offsets in the sky, so they can be translated back
        into the WCS of the PyDrizzle output product.
        
        NOTE:  Only works with 'delta' shifts now, and 
                    requires that a 'refimage' be specified.
        """
                
        # for each set of shifts, translate them into delta(ra,dec) based on refwcs
        for img in asndict['order']:

            xsh = asndict['members'][img]['xshift']
            ysh = asndict['members'][img]['yshift']

            if xsh == 0.0 and ysh == 0.0:
                delta_ra = 0.0
                delta_dec = 0.0
            else:
                #check the units for the shifts...
                if asndict['members'][img]['shift_units'] == 'pixels':
                    # Initialize the reference WCS for use in translation
                    # NOTE: This assumes that a 'refimage' has been specified for
                    #       every set of shifts.
                    refwcs = wcsutil.WCSObject(asndict['members'][img]['refimage'])
                    cp1 = refwcs.crpix1
                    cp2 = refwcs.crpix2

                    nra,ndec = refwcs.xy2rd((cp1+xsh,cp2+ysh))
                    
                    delta_ra = refwcs.crval1-nra
                    delta_dec = refwcs.crval2-ndec
                else:
                    # Shifts already in units of RA/Dec (decimal degrees)
                    # No conversion necessary
                    delta_ra = xsh
                    delta_dec = ysh
            
            asndict['members'][img]['delta_ra'] = delta_ra
            asndict['members'][img]['delta_dec'] = delta_dec
            
    def clean(self,coeffs=no,final=no):
        """ Removes intermediate products from disk. """
        for img in self.parlist:
            # If build == no, then we do not want to delete the only
            # products created by PyDrizzle; namely,
            #     outdata, outcontext, outweight
            if self.build == yes:
                fileutil.removeFile([img['outdata'],img['outcontext'],img['outweight']])

            fileutil.removeFile([img['outsingle'],img['outsweight']])
            #fileutil.removeFile([img['outsingle'],img['outsweight'],img['outscontext']])
            fileutil.removeFile(img['outblot'])
            if coeffs:
                os.remove(img['coeffs'])
                if img['driz_mask'] != None:
                    fileutil.removeFile(img['driz_mask'])
                if img['single_driz_mask'] != None:
                    fileutil.removeFile(img['single_driz_mask'])
        if final:
            fileutil.removeFile(self.output)


    # Run 'drizzle' here...
    #
    def run(self,save=no,build=yes,blot=no,single=no,clean=no,interp='linear',sinscl=1.0, debug=no):
        """Perform drizzle operation on input to create output.
         This method would rely on the buildPars() method for
         the output product to produce correct parameters
         based on the inputs. The output for buildPars() is a list
         of dictionaries, one for each input, that matches the
         primary parameters for an IRAF drizzle task.

         This method would then loop over all the entries in the
         list and run 'drizzle' for each entry. """

        # Store the value of build set by the user for use, if desired,
        # in the 'clean()' method.
        self.build = build
        self.debug = debug

        print 'PyDrizzle: drizzle task started at ',_ptime()
        _memmap = self.pars['memmap']

        # Check for existance of output file.
        if single == no and build == yes and fileutil.findFile(self.output):
            print 'Removing previous output product...'
            os.remove(self.output)
        #
        # Setup the versions info dictionary for output to PRIMARY header
        # The keys will be used as the name reported in the header, as-is
        #
        _versions = {'PyDrizzle':__version__,'PyFITS':pyfits.__version__,'Numpy':N.__version__}

        # Set parameters for each input and run drizzle on it here.

        if blot:
            #
            # Run blot on data...
            #

            _hdrlist = []

            for plist in self.parlist:

                _insci = N.zeros((plist['outny'],plist['outnx']),dtype=N.float32)
                _outsci = N.zeros((plist['blotny'],plist['blotnx']),dtype=N.float32)
                _hdrlist.append(plist)
                # Open input image as PyFITS object
                if plist['outsingle'] != plist['outdata']:
                    _data = plist['outsingle']
                else:
                    _data = plist['outdata']

                # PyFITS can be used here as it will always operate on
                # output from PyDrizzle (which will always be a FITS file)
                # Open the input science file
                _fname,_sciextn = fileutil.parseFilename(_data)
                _inimg = fileutil.openImage(_fname)

                # Return the PyFITS HDU corresponding to the named extension
                _scihdu = fileutil.getExtn(_inimg,_sciextn)
                _insci = _scihdu.data.copy()

                # Read in the distortion correction arrays, if specified
                _pxg,_pyg = plist['exposure'].getDGEOArrays()

                # Now pass numpy objects to callable version of Blot...
                #runBlot(plist)
                build=no
                misval = 0.0
                kscale = 1.0

                xmin = 1
                xmax = plist['outnx']
                ymin = 1
                ymax = plist['outny']
                #
                # Convert shifts to input units
                #
                xsh = plist['xsh'] * plist['scale']
                ysh = plist['ysh'] * plist['scale']
                # ARRDRIZ.TBLOT needs to be updated to support 'poly5' interpolation,
                # and exptime scaling of output image.
                #
                """
                #
                # This call to 'arrdriz.tdriz' uses the F2PY syntax
                #
                arrdriz.tblot(_insci, _outsci,xmin,xmax,ymin,ymax,
                            plist['xsh'],plist['ysh'],
                            plist['rot'],plist['scale'], kscale, _pxg, _pyg,
                            'center',interp, plist['coeffs'], plist['exptime'],
                            misval, sinscl, 1)
                #
                # End of F2PY syntax
                #
                """
                #
                # This call to 'arrdriz.tdriz' uses the F2C syntax
                #
                if (_insci.dtype > N.float32):
                    #WARNING: Input array recast as a float32 array
                    _insci = _insci.astype(N.float32)
                t = arrdriz.tblot(_insci, _outsci,xmin,xmax,ymin,ymax,
                            xsh,ysh, plist['rot'],plist['scale'], kscale,
                            0.0,0.0, 1.0,1.0, 0.0, 'output',
                            _pxg, _pyg,
                            'center',interp, plist['coeffs'], plist['exptime'],
                            misval, sinscl, 1,0.0,0.0)

                #
                # End of F2C syntax
                #

                # Write output Numpy objects to a PyFITS file
                # Blotting only occurs from a drizzled SCI extension
                # to a blotted SCI extension...
                _header = fileutil.getHeader(plist['data'])
                _wcs = wcsutil.WCSObject(plist['data'],header=_header)

                _outimg = outputimage.OutputImage(_hdrlist, build=no, wcs=_wcs, blot=yes)
                _outimg.outweight = None
                _outimg.outcontext = None
                _outimg.writeFITS(plist['data'],_outsci,None,versions=_versions)

                #_buildOutputFits(_outsci,None,plist['outblot'])
                _insci *= 0.
                _outsci *= 0.
                _inimg.close()
                del _inimg
                _hdrlist = []

                del _pxg,_pyg

                del _insci,_outsci
            del _outimg

        else:
            #
            # Perform drizzling...
            #
            # Only work on a copy of the product WCS, so that when
            # this gets updated for the output image, it does not
            # modify the original WCS computed by PyDrizzle
            _wcs = self.observation.product.geometry.wcs.copy()

            _numctx = {'all':len(self.parlist)}
#            if single:
            # Determine how many chips make up each single image
            for plist in self.parlist:
                plsingle = plist['outsingle']
                if _numctx.has_key(plsingle): _numctx[plsingle] += 1
                else: _numctx[plsingle] = 1
            #
            # A image buffer needs to be setup for converting the input
            # arrays (sci and wht) from FITS format to native format
            # with respect to byteorder and byteswapping.
            # This buffer should be reused for each input.
            #
            plist = self.parlist[0]
            _outsci = N.zeros((plist['outny'],plist['outnx']),dtype=N.float32)
            _outwht = N.zeros((plist['outny'],plist['outnx']),dtype=N.float32)
            _inwcs = N.zeros([8],dtype=N.float64)

            # Compute how many planes will be needed for the context image.
            _nplanes = int((_numctx['all']-1) / 32) + 1
            # For single drizzling or when context is turned off,
            # minimize to 1 plane only...
            if single or self.parlist[0]['outcontext'] == '':
                _nplanes = 1

            # Always initialize context images to a 3-D array
            # and only pass the appropriate plane to drizzle as needed
            _outctx = N.zeros((_nplanes,plist['outny'],plist['outnx']),dtype=N.int32)

            # Keep track of how many chips have been processed
            # For single case, this will determine when to close
            # one product and open the next.
            _numchips = 0
            _nimg = 0
            _hdrlist = []

            for plist in self.parlist:
                # Read in the distortion correction arrays, if specifij8cw08n4q_raw.fitsed
                _pxg,_pyg = plist['exposure'].getDGEOArrays()

                _hdrlist.append(plist)
                # Open the SCI image
                _expname = plist['data']
                _handle = fileutil.openImage(_expname,mode='readonly',memmap=0)
                _fname,_extn = fileutil.parseFilename(_expname)
                _sciext = fileutil.getExtn(_handle,extn=_extn)

                # Make a local copy of SCI array and WCS info
                #_insci = _sciext.data.copy()
                _inwcs = drutil.convertWCS(wcsutil.WCSObject(_fname,header=_sciext.header),_inwcs)

                # Determine output value of BUNITS
                # and make sure it is not specified as 'ergs/cm...'
                # we want to use the actual value from the input image header directly
                # when possible in order to account for any unit conversions that may
                # be applied to the input image between initialization of PyDrizzle
                # and the calling of this run() method.
                if _sciext.header.has_key('bunit') and _sciext.header['bunit'] not in ['','N/A']:
                    _bunit = _sciext.header['bunit']
                else:
                    # default based on instrument-specific logic
                    _bunit = plist['bunit']
                
                _bindx = _bunit.find('/')
                if plist['units'] == 'cps':
                    # If BUNIT value does not specify count rate already...
                    if _bindx < 1:
                        # ... append '/SEC' to value
                        _bunit += '/S'
                    else:
                        # reset _bunit here to None so it does not 
                        #    overwrite what is already in header
                        _bunit = None
                else:
                    if _bindx > 0:
                        # remove '/S'
                        _bunit = _bunit[:_bindx]
                    else:
                        # reset _bunit here to None so it does not 
                        #    overwrite what is already in header
                        _bunit = None                 

                # Compute what plane of the context image this input would
                # correspond to:
                _planeid = int(_numchips /32)

                # Select which mask needs to be read in for drizzling
                if single:
                    _mask = plist['single_driz_mask']
                else:
                    _mask = plist['driz_mask']
                    
                # Check to see whether there is a mask_array at all to use...
                if isinstance(_mask,types.StringType):
                    if _mask != None and _mask != '':
                        _wht_handle = fileutil.openImage(_mask,mode='readonly',memmap=0)
                        _inwht = _wht_handle[0].data.astype(N.float32)
                        _wht_handle.close()
                        del _wht_handle
                    else:
                        print 'No weight or mask file specified!  Assuming all pixels are good.'
                        _inwht = N.ones((plist['blotny'],plist['blotnx']),dtype=N.float32)
                elif _mask != None:
                    _inwht = _mask.astype(N.float32)
                else:
                    print 'No weight or mask file specified!  Assuming all pixels are good.'
                    _inwht = N.ones((plist['blotny'],plist['blotnx']),dtype=N.float32)

                if plist['wt_scl'] != None:
                    if isinstance(plist['wt_scl'],types.StringType):
                        if  plist['wt_scl'].isdigit() == False :
                            # String passed in as value, check for 'exptime' or 'expsq'
                            _wtscl_float = None
                            try:
                                _wtscl_float = float(plist['wt_scl'])
                            except ValueError:
                                _wtscl_float = None
                            if _wtscl_float != None:
                                _wtscl = _wtscl_float
                            elif plist['wt_scl'] == 'expsq':
                                _wtscl = plist['exptime']*plist['exptime']
                            else:
                                # Default to the case of 'exptime', if
                                #   not explicitly specified as 'expsq'
                                _wtscl = plist['exptime']
                        else:
                            # int value passed in as a string, convert to float
                            _wtscl = float(plist['wt_scl'])
                    else:
                        # We have a non-string value passed in...
                        _wtscl = float(plist['wt_scl'])
                else:
                    # Default case: wt_scl = exptime
                    _wtscl = plist['exptime']

                #print 'WT_SCL: ',plist['wt_scl'],' _wtscl: ',_wtscl
                # Set additional parameters needed by 'drizzle'
                _in_units = plist['in_units']
                if _in_units == 'cps':
                    _expin = 1.0
                else:
                    _expin = plist['exptime']
                _shift_fr = 'output'
                _shift_un = 'output'
                _uniqid = _numchips + 1
                ystart = 0
                nmiss = 0
                nskip = 0
                _vers = plist['driz_version']

                _con = yes
                _imgctx = _numctx['all']
                if single:
                    _imgctx = _numctx[plist['outsingle']]
                #if single or (plist['outcontext'] == '' and single == yes):
                if _nplanes == 1:
                    _con = no
                    # We need to reset what gets passed to TDRIZ
                    # when only 1 context image plane gets generated
                    # to prevent overflow problems with trying to access
                    # planes that weren't created for large numbers of inputs.
                    _planeid = 0
                    _uniqid = ((_uniqid-1) % 32) + 1

                """
                #
                # This call to 'arrdriz.tdriz' uses the F2PY syntax
                #
                #_dny = plist['blotny']
                # Call 'drizzle' to perform image combination
                tdriz,nmiss,nskip,_vers = arrdriz.tdriz(
                            _sciext.data,_inwht, _outsci, _outwht,
                            _outctx[_planeid], _con, _uniqid, ystart, 1, 1,
                            plist['xsh'],plist['ysh'], 'output','output',
                            plist['rot'],plist['scale'], _pxg,_pyg,
                            'center', plist['pixfrac'], plist['kernel'],
                            plist['coeffs'], 'counts', _expin,_wtscl,
                            plist['fillval'], _inwcs, 1, nmiss, nskip,_vers)
                #
                # End of F2PY syntax
                #
                """
                #
                # This call to 'arrdriz.tdriz' uses the F2C syntax
                #
                _dny = plist['blotny']
                # Call 'drizzle' to perform image combination
                if (_sciext.data.dtype > N.float32):
                    #WARNING: Input array recast as a float32 array
                    _sciext.data = _sciext.data.astype(N.float32)
                    
                _vers,nmiss,nskip = arrdriz.tdriz(_sciext.data,_inwht, _outsci, _outwht,
                            _outctx[_planeid], _uniqid, ystart, 1, 1, _dny,
                            plist['xsh'],plist['ysh'], 'output','output',
                            plist['rot'],plist['scale'],
                            0.0,0.0, 1.0,1.0,0.0,'output',
                            _pxg,_pyg, 'center', plist['pixfrac'], plist['kernel'],
                            plist['coeffs'], _in_units, _expin,_wtscl,
                            plist['fillval'], _inwcs, nmiss, nskip, 1,0.0,0.0)
                """
                _vers,nmiss,nskip = arrdriz.tdriz(_sciext.data,_inwht, _outsci, _outwht,
                            _outctx[_planeid], _uniqid, ystart, 1, 1, _dny,
                            plist['xsh'],plist['ysh'], 'output','output',
                            plist['rot'],plist['scale'],
                            _pxg,_pyg, 'center', plist['pixfrac'], plist['kernel'],
                            plist['coeffs'], 'counts', _expin,_wtscl,
                            plist['fillval'], _inwcs, nmiss, nskip, 1)
                """
                #
                # End of F2C syntax
                #

                plist['driz_version'] = _vers

                if nmiss > 0:
                    print '! Warning, ',nmiss,' points were outside the output image.'
                if nskip > 0:
                    print '! Note, ',nskip,' input lines were skipped completely.'
                # Close image handle
                _handle.close()
                del _handle,_fname,_extn,_sciext
                del _inwht

                del _pxg,_pyg

                # Remember the name of the first image that goes into 
                # this particular product
                # This will insure that the header reports the proper
                # values for the start of the exposure time used to make
                # this product; in particular, TIME-OBS and DATE-OBS.
                if _numchips == 0:
                    _template = plist['data']

                if _nimg == 0 and self.debug == yes:
                    # Only update the WCS from drizzling the
                    # first image in the list, just like IRAF DRIZZLE
                    drutil.updateWCS(_inwcs,_wcs)
                    #print '[run] Updated WCS now:'
                    #print _wcs

                # Increment number of chips processed for single output
                _numchips += 1
                if _numchips == _imgctx:
                    ###########################
                    #
                    #   IMPLEMENTATION REQUIREMENT:
                    #
                    # Need to implement scaling of the output image
                    # from 'cps' to 'counts' in the case where 'units'
                    # was set to 'counts'... 21-Mar-2005
                    #
                    ###########################
                    # Start by determining what exposure time needs to be used
                    # to rescale the product.
                    if single:
                        _expscale = plist['exptime']
                    else:
                        _expscale = plist['texptime']

                    #If output units were set to 'counts', rescale the array in-place
                    if plist['units'] == 'counts':
                        N.multiply(_outsci, _expscale, _outsci)

                    #
                    # Write output arrays to FITS file(s) and reset chip counter
                    #
                    _outimg = outputimage.OutputImage(_hdrlist, build=build, wcs=_wcs, single=single)
                    _outimg.set_bunit(_bunit)
                    _outimg.set_units(plist['units'])
                    
                    _outimg.writeFITS(_template,_outsci,_outwht,ctxarr=_outctx,versions=_versions)
                    del _outimg
                    #
                    # Reset chip counter for next output image...
                    #
                    _numchips = 0
                    _nimg = 0
                    N.multiply(_outsci,0.,_outsci)
                    N.multiply(_outwht,0.,_outwht)
                    N.multiply(_outctx,0,_outctx)

                    _hdrlist = []
                else:
                    _nimg += 1

            del _outsci,_outwht,_inwcs,_outctx, _hdrlist

        # Remove temp files, if desired
        # Files to be removed are:
        #   parlist['coeffs']
        #   parlist['driz_mask']
        if not save and clean:
            for img in self.parlist:
                fileutil.removeFile(img['coeffs'])
                if img['driz_mask'] != None:
                    fileutil.removeFile(img['driz_mask'])
                if img['single_driz_mask'] != None:
                    fileutil.removeFile(img['single_driz_mask'])

        print 'PyDrizzle drizzling completed at ',_ptime()


    def resetPars(self,field=None,pixfrac=None,kernel=None,units=None):
        """
        Recompute the output parameters based on a new
            SkyField or WCSObject object.
        """
        if field and not isinstance(field, SkyField):
            if isinstance(field, wcsutil.WCSObject):
                _ref = SkyField(wcs=field)
            else:
                raise TypeError, 'No valid WCSObject or SkyField object entered...'
        else:
            _ref = field
        # Create new version of the parlist with the new values of the
        # parameters based on the reference image.
        new_parlist = self.observation.buildPars(ref=_ref)


        # Copy the parameters from the new parlist into the existing
        # parlist (self.parlist) to preserve any changes/updates made
        # to the parlist externally to PyDrizzle.
        for i in xrange(len(self.parlist)):
            for key in new_parlist[i]:
                self.parlist[i][key] = new_parlist[i][key]
        del new_parlist

        if pixfrac or kernel or units:
            #print 'resetting additional parameter(s)...'
            for p in self.parlist:
                if kernel:
                    p['kernel'] = kernel
                if pixfrac:
                    p['pixfrac'] = pixfrac
                if units:
                    if units == 'counts' or units == 'cps':
                        p['units'] = units
                    else:
                        print 'Units ',units,' not valid! Parameter not reset.'
                        print 'Please use either "cps" or "counts".'

    def help(self):
        """
                Run the module level help function to provide syntax information.
        """
        help()

    def printPars(self,pars='xsh,ysh,rot,scale,outnx,outny,data',format=no):
        """ Prints common parameters for review. """
        if format:
            _title = pars.replace(',','    ')
            print _title
            print '-'*72

        _pars = pars.split(',')
        for pl in self.parlist:
            for _p in _pars: print pl[_p],
            print ''

    def getMember(self,memname):
        """ Returns the class instance for the specified member name."""
        return self.observation.getMember(memname)




def _ptime():
    import time

    # Format time values for keywords IRAF-TLM, and DATE
    _ltime = time.localtime(time.time())
    tlm_str = time.strftime('%H:%M:%S (%d/%m/%Y)',_ltime)
    #date_str = time.strftime('%Y-%m-%dT%H:%M:%S',_ltime)
    return tlm_str

#################



class SkyField:
    """
    An class for specifying the parameters and building a WCS object
        for a user-specified drizzle product.
    The user may optionally modify the values for:
        psize       - size of image's pixels in arc-seconds
        orient      - value of ORIENTAT for the field
        shape       - tuple containing the sizes of the field's x/y axes
        ra,dec      - position of the center of the field
                      decimal (124.5678) or
                      sexagesimal string _in quotes_ ('hh:mm:ss.ss')
        crpix       - tuple for pixel position of reference point in
                        output image
    Usage:
        To specify a new field with a fixed output size of 1024x1024:
        --> field = pydrizzle.SkyField(shape=(1024,1024))

        The 'set()' method modifies one of the parameters listed above
        without affecting the remainder of the parameters.
        --> field.set(psize=0.1,orient=0.0)
        --> field.set(ra=123.45678,dec=0.1000,crpix=(521,576))

        View the WCS or user-specified values for this object:
        --> print field

    """
    def __init__(self,shape=None,psize=None,wcs=None):

        self.shape = shape
        self.ra = None
        self.dec = None
        self.orient = None
        self.psize = psize

        self.crpix = None

        # Set up proper shape tuple for WCSObject
        if shape != None and psize != None:
            wshape = (shape[0],shape[1],psize)
        else:
            wshape = None
            if wcs:
                self.crpix = (wcs.crpix1,wcs.crpix2)

        # Specify 'new=yes' with given rootname to unambiguously create
        #   a new WCS from scratch.
        if wcs == None:
            self.wcs = wcsutil.WCSObject("New",new=yes)
        else:
            self.wcs = wcs.copy()
            # Need to adjust WCS to match the 'drizzle' task
            # align=center conventions. WJH 29-Aug-2005
            #self.wcs.crpix1 -= 1.0
            #self.wcs.crpix2 -= 1.0
            self.wcs.recenter()

        self.wcs.updateWCS(size=shape,pixel_scale=psize)

        # Set this to keep track of total exposure time for
        # output frame... Not user settable.
        self.exptime = None
        # Keep track of whether this is for a Dither product or not
        self.dither = None

    def mergeWCS(self,wcs,overwrite=yes):
        """ Sets up the WCS for this object based on another WCS.
            This method will NOT update object attributes other
            than WCS, as all other attributes reflect user-settings.
        """
        #
        # Start by making a copy of the input WCS...
        #
        if self.wcs.rootname == 'New':
            self.wcs = wcs.copy()
        else:
            return
        self.wcs.recenter()

        if self.ra == None:
            _crval = None
        else:
            _crval = (self.ra,self.dec)

        if self.psize == None:
            _ratio = 1.0
            _psize = None
            # Need to resize the WCS for any changes in pscale
        else:
            _ratio = wcs.pscale / self.psize
            _psize = self.psize

        if self.orient == None:
            _orient = None
            _delta_rot = 0.
        else:
            _orient = self.orient
            _delta_rot = wcs.orient - self.orient

        _mrot = fileutil.buildRotMatrix(_delta_rot)

        if self.shape == None:
            _corners = N.array([[0.,0.],[wcs.naxis1,0.],[0.,wcs.naxis2],[wcs.naxis1,wcs.naxis2]])
            _corners -= (wcs.naxis1/2.,wcs.naxis2/2.)
            _range = drutil.getRotatedSize(_corners,_delta_rot)
            shape = ((_range[0][1] - _range[0][0])*_ratio,(_range[1][1]-_range[1][0])*_ratio)
            old_shape = (wcs.naxis1*_ratio,wcs.naxis2*_ratio)

            _cen = (shape[0]/2., shape[1]/2.)

            #if _delta_rot == 0.:
            #    _crpix = (self.wcs.crpix1,self.wcs.crpix2)
            #else:
                # Rotate original scaled crpix position to new orientation
                #_crpix = N.dot((wcs.crpix1*_ratio - _cen[0],wcs.crpix2*_ratio -_cen[1]),_mrot)+_cen
            _crpix = _cen
        else:
            shape = self.shape
            if self.crpix == None:
                _crpix = (self.shape[0]/2.,self.shape[1]/2.)
            else:
                _crpix = self.crpix

        # Set up the new WCS based on values from old one.
        self.wcs.updateWCS(pixel_scale=_psize,orient=_orient,refpos=_crpix,refval=_crval)
        self.wcs.naxis1 =  int(shape[0])
        self.wcs.naxis2 =  int(shape[1])

    def set(self,psize=None,orient=None,ra=None,dec=None,
            shape=None,crpix=None):
        """
        Modifies the attributes of the SkyField and
            updates it's WCS when appropriate.
        """
        # Converts(if necessary), then updates the RA and Dec.
        _ra,_dec = None,None
        if ra != None:
            if string.find(repr(ra),':') > 0:
                _hms = string.split(repr(ra)[1:-1],':')
                if _hms[0][0] == '-': _sign = -1
                else: _sign = 1

                for i in range(len(_hms)): _hms[i] = float(_hms[i])
                _ra = _sign * (_hms[0] + ((_hms[1] + _hms[2]/60.) / 60.)) * 15.
            else:
                _ra = float(ra)
            self.ra = _ra

        if dec != None:
            if string.find(repr(dec),':') > 0:
                _dms = string.split(repr(dec)[1:-1],':')
                if _dms[0][0] == '-': _sign = -1
                else: _sign = 1

                for i in range(len(_dms)): _dms[i] = float(_dms[i])
                _dec = _sign * (_dms[0] + ((_dms[1] + _dms[2]/60.) / 60.))
            else:
                _dec = float(dec)
            self.dec = _dec

        if self.ra != None and self.dec != None:
            _crval = (self.ra,self.dec)
        else:
            _crval = None

        # Updates the shape, and reference position,
        # only if a new value is specified.
        _crpix = None
        if crpix == None:
            if shape != None:
                self.shape = shape
                _crpix = (self.shape[0]/2.,self.shape[1]/2.)
        else:
            _crpix = crpix

        self.crpix=_crpix

        if psize != None:
            self.psize = psize

        if orient != None:
            self.orient = orient

        # Updates the WCS with all the changes, if there is enough info.
        self.wcs.updateWCS(pixel_scale=psize,orient=orient,refpos=_crpix,
                                refval=_crval,size=self.shape)

    def __str__(self):
        """ Prints the WCS information set for this object.
        """
        if self.psize != None and self.orient != None:
            block =  self.wcs.__str__()
        else:
            block = 'User parameters for SkyField object: \n'
            block = block + '       ra = '+repr(self.ra)+' \n'
            block = block + '      dec = '+repr(self.dec)+' \n'
            block = block + '    shape = '+repr(self.shape)+' \n'
            block = block + '    crpix = '+repr(self.crpix)+' \n'
            block = block + '    psize = '+repr(self.psize)+' \n'
            block = block + '   orient = '+repr(self.orient)+' \n'
            block = block + '   No WCS.\n'

        return block

    def help(self):
        """ Creates and prints usage information for this class.
        """
        print self.__doc__

class DitherProduct(Pattern):
    """
    Builds an object for a set of dithered inputs, each of which
    will be one of the Observation objects.
    """
    def __init__(self, prodlist, pars=None):
        # Build temporary output drizzle product name
        if prodlist['output'].find('.fits') < 0:
            if prodlist['output'].rfind('_drz') < 0:
                output = fileutil.buildNewRootname(prodlist['output'],extn='_drz.fits')
            else:
                output = prodlist['output']+'.fits'
        else:
            output = prodlist['output']

        # Setup a default exposure to contain the results
        Pattern.__init__(self, None, output=output, pars=pars)
        self.pars = prodlist['members']
        self.nmembers = self.nimages = len(prodlist['members']) 
        self.offsets = None

        self.addMembers(prodlist,pars,output)

        if len(self.members) == 0:
            print 'No suitable inputs from ASN table. Quitting PyDrizzle...'
            raise Exception

        self.exptime = self.getExptime()

        self.buildProduct(output)
    
    def closeHandle(self):
        """ Close image handle for each member."""
        for member in self.members:
            member.closeHandle()
            
    
        
    def buildProduct(self,output):
        # Build default Metachip based on unmodified header WCS values
        output_wcs = self.buildMetachip()

        self.size = (output_wcs.naxis1,output_wcs.naxis2)
        self.product = Exposure(output,wcs=output_wcs,new=yes)
        self.product.geometry.wcslin = self.product.geometry.wcs.copy()

        # Compute default offsets between all inputs...
        # This will be used when applying relative shifts from ASN table
        # or shiftfile.
        self.computeOffsets()
        # Preserve default DitherProduct Metachip WCS as wcslin
        self.product.exptime = self.exptime
        # Update the corners arrays for the product now...
        self.product.setCorners()
        #self.product.corners['corrected'] = self.product.corners['raw']
        _prodcorners = []
        for prod in self.members:
            _prodcorners +=  prod.product.corners['corrected'].tolist()
        self.product.corners['corrected'] = N.array(_prodcorners,dtype=N.float64)

    def computeOffsets(self):
        """
        This method will rely on final product's 'rd2xy' method
        to compute offsets between the different input chips.
        """
        ref_wcs = self.product.geometry.wcs
        ref_crv= (ref_wcs.crval1,ref_wcs.crval2)

        _tot_ref = {'pix_shift':(0.,0.),'ra_shift':(0.,0.),
                    'val':999999999.,'name':''}

        _member_num = 0
        for member in self.members:
            in_wcs = member.product.geometry.wcs

            x,y = ref_wcs.rd2xy((in_wcs.crval1,in_wcs.crval2))
            xoff = x - ref_wcs.crpix1
            yoff = y - ref_wcs.crpix2

            raoff = (in_wcs.crval1 - ref_wcs.crval1,in_wcs.crval2 - ref_wcs.crval2)

            _delta_rot = in_wcs.orient - ref_wcs.orient
            # Determine which image has the smallest offset
            _tot = N.sqrt(N.power(xoff,2)+N.power(yoff,2))

            """
            # Use first image as reference
            if _member_num == 0:
                _tot_ref['val'] = _tot
                _tot_ref['pix_shift'] = (xoff,yoff)
                _tot_ref['name'] = member.rootname
                _tot_ref['ra_shift'] = raoff
                # This will be used primarily for manual verification
                _tot_ref['wcs'] = in_wcs
            """

            _member_num += 1

            # Keep track of the results for later use/comparison
            member.offsets = {'pixels':(xoff,yoff),'arcseconds':raoff}
        """
        for member in self.members:
            member.refimage = _tot_ref
        """

    def getExptime(self):
        """
        Add up the exposure time for all the members in
        the pattern, since 'drizzle' doesn't have the necessary
        information to correctly set this itself.
        """
        _exptime = 0.
        _start = []
        _end = []
        for member in self.members:
            _memexp = member.product.exptime
            _exptime = _exptime + _memexp[0]
            _start.append(_memexp[1])
            _end.append(_memexp[2])
        _expstart = min(_start)
        _expend = max(_end)

        return (_exptime,_expstart,_expend)

    def addMembers(self,prodlist,pars,output):
        """
        For each entry in prodlist, append the appropriate type
        of Observation to the members list.

        If it's a moving target observation apply the wcs of the first 
        observation to  all other observations.
        """     

        member = prodlist['order'][0]
        filename = fileutil.buildRootname(member)
        mtflag = fileutil.getKeyword(filename, 'MTFLAG')

        if mtflag == 'T':
            mt_member = selectInstrument(filename,output, pars=pars)
            mt_wcs = {}
            for member in mt_member.members:
                mt_wcs[member.chip] = member.geometry.wcs
            
            pars['mt_wcs'] = mt_wcs
            del mt_member   
 
        for memname in prodlist['order']:
            pardict = self.pars[memname]
            pardict.update(pars)
            filename = fileutil.buildRootname(memname)
            if filename:
                self.members.append(selectInstrument(filename,output,pars=pardict))
            else:
                print 'No recognizable input! Not building parameters for ',memname

    def getMember(self,memname):
        """ Return the class instance for the member with name memname."""
        member = None
        for mem in self.members:
            member = mem.getMember(memname)
            if member != None:
                break
        return member

    def getMemberNames(self):
        """ Returns a dictionary with one key for each member.
            The value for each key is a list of all the extension/chip
            names that make up each member.
            Output: {member1:[extname1,extname2,...],...}
        """
        memlist = []
        for member in self.members:
            memlist.extend(member.getMemberNames())

        return memlist

    def buildPars(self,ref=None):

        # If an output field is specified, update product WCS to match
        if ref != None:
            _field = ref
            _field.mergeWCS(self.product.geometry.wcslin)
            #_field.exptime = self.exptime

            # Update product WCS based on input
            self.transformMetachip(_field)
            #_field.mergeWCS(self.product.geometry.wcs)
        else:
            _field = SkyField()
            _field.mergeWCS(self.product.geometry.wcslin)
            # Reset Product WCS to reflect use of default WCS
            self.product.geometry.wcs = self.product.geometry.wcslin.copy()
        _field.exptime = self.exptime
        # Specify that this product comes represents a dither product
        # not just a single observation product.
        _field.dither = yes

        # Copy the new reference WCS into the product WCS
        self.product.geometry.wcs = _field.wcs.copy()
        parlist = []
        for member in self.members:
            parlist = parlist + member.buildPars(ref=_field)

        # Set value for number of images combined by PyDrizzle
        # based on DitherPattern attribute.
        for pl in parlist:
            pl['nimages'] = self.nimages

        return parlist

    def buildMetachip(self):
        """
        This method combines the results of the member's buildMetachip()
        methods into a composite WCS.
        (DitherProduct method)
        """
        prodlist = []

        _ref = self.members[0].product.geometry

        # Get pscale from model
        pscale = _ref.model.pscale

        for member in self.members:
            # merge the corrected shapes into a corrected meta-chip here
            # Start by computing the corected positions for reference points
            prodlist.append(member.product)

        # Use first input image WCS as initial reference WCS
        meta_wcs = _ref.wcs.copy()

        ref_crval = (meta_wcs.crval1,meta_wcs.crval2)
        ref_crpix = (meta_wcs.crpix1,meta_wcs.crpix2)

        # Specify the output orientation angle
        ref_orient = meta_wcs.orient

        _delta_x = _delta_y = 0.0
        _xr,_yr,_dpx,_dpy = [],[],[],[]
        # Compute offsets for each input relative to this reference WCS
        for member in prodlist:
            _wcs = member.geometry.wcs

            # Rigorously compute the orientation changes from WCS
            # information using algorithm provided by R. Hook from WDRIZZLE.
            abxt,cdyt = drutil.wcsfit(member.geometry, meta_wcs)

            # Compute the rotation between input and reference from fit coeffs.
            _angle = RADTODEG(N.arctan2(abxt[1],cdyt[0]))
            _dpos = (abxt[2],cdyt[2])

            _delta_x += _dpos[0]
            _delta_y += _dpos[1]
            _dpx.append(_dpos[0])
            _dpy.append(_dpos[1])

            # Compute the range of pixels each input spans in the
            # output meta-frame coordinates
            # Each product in this list could have a different plate scale.
            # apply
            _scale = meta_wcs.pscale / _wcs.pscale

            # Now, account for any rotation difference between
            # input and output frames
            #_angle = meta_wcs.orient - _wcs.orient
            _xrange,_yrange = drutil.getRotatedSize(member.corners['corrected'],_angle)

            #_range = [(_xrange[1] - _xrange[0]),(_yrange[1] - _yrange[0])]
            #_range[0] *= _scale
            #_range[1] *= _scale

            _xr.append(_dpos[0] + _xrange[0]*_scale)
            _xr.append(_dpos[0] + _xrange[1]*_scale)
            _yr.append(_dpos[1] + _yrange[0]*_scale)
            _yr.append(_dpos[1] + _yrange[1]*_scale)

        # Determine the full size of the metachip
        _xmin = N.minimum.reduce(_xr)
        _ymin = N.minimum.reduce(_yr)
        _xmax = N.maximum.reduce(_xr)
        _ymax = N.maximum.reduce(_yr)

        _dxmin = N.minimum.reduce(_dpx)
        _dymin = N.minimum.reduce(_dpy)
        _dxmax = N.maximum.reduce(_dpx)
        _dymax = N.maximum.reduce(_dpy)

        _nimg = len(prodlist)
        _delta_x /= _nimg
        _delta_y /= _nimg

        # Computes the offset from center of the overall set of shifts
        # as applied to the pixels.  This insures that the visible pixels
        # are centered in the output.
        _delta_dx = ((_xmax - _delta_x) + (_xmin - _delta_x))/2.
        _delta_dy = ((_ymax - _delta_y) + (_ymin - _delta_y))/2.

        if _xmin > 0.: _xmin = 0.
        if _ymin > 0.: _ymin = 0.

        # Need to account for overall offset in images.
        # by adding in average offset induced by shifts to
        # output size: delta_dx/dy.
        # This accounts for relative offsets that are not centered
        # on the final output image.
        #xsize = int(_xmax - _xmin + 2.0 + abs(_delta_dx) )
        #ysize = int(_ymax - _ymin + 2.0 + abs(_delta_dy) )
        xsize = int(_xmax - _xmin)
        ysize = int(_ymax - _ymin)

        # Compute difference between new size and
        # original size of meta_wcs frame.
        _dxsize = xsize - meta_wcs.naxis1
        _dysize = ysize - meta_wcs.naxis2

        # Update reference WCS for new size
        meta_wcs.naxis1 =  xsize
        meta_wcs.naxis2 =  ysize
        _cen = (float(xsize)/2.,float(ysize)/2.)

        # Determine offset of centers from center of image.
        # d[x/y]size  : difference between input and output frame sizes
        # delta_[x/y] : average of all shifts applied to input images
        # delta_d[x/y]: amount off-center of all shifts
        #
        # Need to take into account overall offset in shifts, such that
        # minimum, not maximum, shift is always 0.0. This is needed to allow
        # subarrays to align properly with full frame observations without
        # introducing an overall offset to the output.  WJH 13 Sept 2004.
        #
        _nref = ( _dxsize/2. - _delta_x - _delta_dx, _dysize/2. - _delta_y - _delta_dy)

        # Have to adjust the CRPIX by how much the center needs to shift
        # to fit into the reference frame.
        meta_wcs.crpix1 = meta_wcs.crpix1 + _nref[0]
        meta_wcs.crpix2 = meta_wcs.crpix2 + _nref[1]
        meta_wcs.recenter()

        return meta_wcs

#
#
# Set up default pars dictionary for external calls

def selectInstrument(filename,output,pars=_default_pars):
    """
    Method which encapsulates the logic for determining
    which class to instantiate for each file.

    """
    # Determine the instrument...
    instrument = fileutil.getKeyword(filename+'[0]','INSTRUME')

    #try:
    # ... then create an appropriate object.
    if instrument == INSTRUMENT[0]:
        member = ACSObservation(filename,output,pars=pars)
    elif instrument == INSTRUMENT[1]:
        member = WFPCObservation(filename,output,pars=pars)
    elif instrument == INSTRUMENT[2]:
        member = STISObservation(filename,output,pars=pars)
    elif instrument == INSTRUMENT[3]:
        member = NICMOSObservation(filename,output,pars=pars)
    elif instrument == INSTRUMENT[4]:
        member = WFC3Observation(filename,output,pars=pars)
    else:
        #raise AttributeError, "Instrument '%s' not supported now."%instrument
        member = GenericObservation(filename,output,pars=pars)
    #except:
    #    raise IOError,"Image %s could not be processed."%filename

    return member

