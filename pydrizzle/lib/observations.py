from __future__ import division # confidence medium
from pattern import *
from pytools import fileutil
from distortion import mutil
import numpy as np

class ACSObservation(Pattern):
    """This class defines an observation with information specific
       to ACS WFC exposures, including knowledge of how to mosaic both
       chips."""

    # Define a class variable for the gap between the chips
    PARITY = {'WFC':[[1.0,0.0],[0.0,-1.0]],'HRC':[[-1.0,0.0],[0.0,1.0]],'SBC':[[-1.0,0.0],[0.0,1.0]]}

    def __init__(self, filename, output, pars=None):

        # Now, initialize Pattern with all member Exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'ACS'

        # build required name attributes:
        #  outname, output, outsingle
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        # Build up list of chips in observation
        self.addMembers(filename)

        # Only need to worry about IDC tables for ACS...
        self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)

class GenericObservation(Pattern):
    """
        This class defines an observation stored in a Simple FITS format;
        i.e., only a Primary header and image without extensions.
    """

    REFPIX = {'x':512.,'y':512.}
    DETECTOR_NAME = 'INSTRUME'

    def __init__(self, filename, output, pars=None):

        # Now initialize Pattern with all member exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        # Determine the instrument...
        if self.header.has_key('INSTRUME'):
            _instrument = self.header['INSTRUME']
        else:
            _instrument = self.DETECTOR_NAME
        self.instrument = _instrument

        if self.header.has_key('crpix1'):
            self.REFPIX['x'] = self.header['crpix1']
            self.REFPIX['y'] = self.header['crpix2']
        else:
            self.REFPIX['x'] = self.header['naxis1'] /2.
            self.REFPIX['y'] = self.header['naxis2'] /2.

        # build output rootnames here...
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        self.nmembers = 1

        # Now, build list of members and initialize them
        self.addMembers(filename)

        _ikey = self.members[0].geometry.ikey
        if  _ikey != 'idctab' and _ikey != 'wcs' :
            # Correct distortion coefficients to match output pixel scale
            self.computeCubicCoeffs()
        else:
            self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)


class STISObservation(Pattern):
    """This class defines an observation with information specific
       to STIS exposures.
    """

    # Default coefficients table to use for this instrument
    IDCKEY = 'cubic'

    __theta = 0.0
    __parity = fileutil.buildRotMatrix(__theta) * np.array([[-1.,1.],[-1.,1.]])
    PARITY = {'CCD':__parity,'NUV-MAMA':__parity,'FUV-MAMA':__parity}

    # The dictionaries 'REFDATA' and 'REFPIX' are required for use with
    # cubic and Trauger coefficients tables in 'computeCubicCoeffs'.
    #
    # This information provides the absolute relationship between the chips
    # Latest plate scales: 0.05071, 0.0246
    #REFDATA = {'CCD':{'psize':0.05,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':__theta},
    #         'NUV-MAMA':{'psize':0.024,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':__theta},
    #         'FUV-MAMA':{'psize':0.024,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':__theta}}
    #REFPIX = {'x':512.,'y':512.}

    def __init__(self, filename, output, pars=None):

        # Now initialize Pattern with all member exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'STIS'
        self.__theta = 0.0
        self.REFDATA = {'CCD':{'psize':0.05,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':self.__theta},
                        'NUV-MAMA':{'psize':0.024,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':self.__theta},
                        'FUV-MAMA':{'psize':0.024,'xoff':0.0,'yoff':0.0,'v2':-213.999,'v3':-224.897,'theta':self.__theta}}
        self.REFPIX = {'x':512.,'y':512.}

        # build output rootnames here...
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        # Now, build list of members and initialize them
        self.addMembers(filename)

        if self.members[0].geometry.ikey != 'idctab':
            # Correct distortion coefficients to match output pixel scale
            self.computeCubicCoeffs()
        else:
            self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)

    def getExptime(self):

        header = fileutil.getHeader(self.name+'[sci,1]')
        _exptime = float(header['EXPTIME'])
        if _exptime == 0.: _exptime = 1.0

        if header.has_key('EXPSTART'):
            _expstart = float(header['EXPSTART'])
            _expend = float(header['EXPEND'])
        else:
            _expstart = 0.
            _expend = _exptime

        return (_exptime,_expstart,_expend)




class NICMOSObservation(Pattern):
    """This class defines an observation with information specific
       to NICMOS exposures.
    """

    # Default coefficients table to use for this instrument
    IDCKEY = 'cubic'

    DETECTOR_NAME = 'camera'
    NUM_IMSET = 5

    __theta = 0.0
    __parity = fileutil.buildRotMatrix(__theta) * np.array([[-1.,1.],[-1.,1.]])
    PARITY = {'1':__parity,'2':__parity,'3':__parity}

    # The dictionaries 'REFDATA' and 'REFPIX' are required for use with
    # cubic and Trauger coefficients tables in 'computeCubicCoeffs'.
    #
    # This information provides the absolute relationship between the chips
    # Latest plate scales: 0.05071, 0.0246
    #REFDATA = {'1':{'psize':0.0432,'xoff':0.0,'yoff':0.0,'v2':-296.9228,'v3':290.1827,'theta':__theta},
    #         '2':{'psize':0.076,'xoff':0.0,'yoff':0.0,'v2':-319.9464,'v3':311.8579,'theta':__theta},
    #         '3':{'psize':0.0203758,'xoff':0.0,'yoff':0.0,'v2':-249.8170,'v3':235.2371,'theta':__theta}}
    #REFPIX = {'x':128.,'y':128.}

    def __init__(self, filename, output, pars=None):

        # Now initialize Pattern with all member exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'NICMOS'
        self.__theta = 0.0
        self.REFDATA = {'1':{'psize':0.0432,'xoff':0.0,'yoff':0.0,'v2':-296.9228,'v3':290.1827,'theta':self.__theta},
                        '2':{'psize':0.076,'xoff':0.0,'yoff':0.0,'v2':-319.9464,'v3':311.8579,'theta':self.__theta},
                        '3':{'psize':0.203758,'xoff':0.0,'yoff':0.0,'v2':-249.8170,'v3':235.2371,'theta':self.__theta}}
        self.REFPIX = {'x':128.,'y':128.}
        # build output rootnames here...
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        # Now, build list of members and initialize them
        self.addMembers()

        if self.members[0].geometry.ikey != 'idctab':
            # Correct distortion coefficients to match output pixel scale
            self.computeCubicCoeffs()
        else:
            self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)


    def addMembers(self):
        """ Build rootname for each SCI extension, and
            create the mask image from the DQ extension.
            It would then append a new Exposure object to 'members'
            list for each extension.
        """

        self.detector = detector = str(self.header[self.DETECTOR_NAME])

        if self.pars['section'] == None:
            self.pars['section'] = [None]*self.nmembers

        # Build rootname here for each SCI extension...
        for i in range(self.nmembers):
            _sciname = self.imtype.makeSciName(i+1,section=self.pars['section'][i])
            _dqname = self.imtype.makeDQName(i+1)
            _extname = self.imtype.dq_extname

            # Build mask files based on input 'bits' parameter values
            _masklist = []
            _masknames = []
            #
            # If we have a valid bits value...
            # Creat the name of the output mask file
            _maskname = buildmask.buildMaskName(_dqname,i+1)
            _masknames.append(_maskname)
            # Create the actual mask file now...
            outmask = buildmask.buildMaskImage(_dqname,self.bitvalue[0],_maskname,extname=_extname,extver=i+1)
            _masklist.append(outmask)

            #
            # Check to see if a bits value was provided for single drizzling...
            # Different bits value specified for single drizzle step
            # create new filename for single_drizzle mask file
            _maskname = _maskname.replace('final_mask','single_mask')
            _masknames.append(_maskname)

            # Create new mask file with different bit value.
            outmask = buildmask.buildMaskImage(_dqname,self.bitvalue[1],_maskname,extname=_extname,extver=i+1)
            # Add new name to list for single drizzle step
            _masklist.append(outmask)
            _masklist.append(_masknames)

            self.members.append(Exposure(_sciname, idckey=self.idckey, dqname=_dqname,
                    mask=_masklist, pa_key=self.pa_key, parity=self.PARITY[detector],
                    idcdir=self.pars['idcdir'], group_indx = i+1,
                    handle=self.image_handle,extver=i+1,exptime=self.exptime[0], ref_pscale=self.REFDATA[self.detector]['psize'],mt_wcs=self.pars['mt_wcs']))

class WFC3Observation(Pattern):
    """This class defines an observation with information specific
       to ACS WFC exposures, including knowledge of how to mosaic both
       chips."""

    #__theta = 45.00
    #__ir_parity = fileutil.buildRotMatrix(__theta) * np.array([[-1.,1.],[-1.,1.]])
    # Define a class variable for the gap between the chips
    PARITY = {'UVIS':[[-1.0,0.0],[0.0,1.0]],'IR':[[-1.0,0.0],[0.0,1.0]]}

    def __init__(self, filename, output, pars=None):

        # Now, initialize Pattern with all member Exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'WFC3'

        # build required name attributes:
        #  outname, output, outsingle
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()
        
        # Set binned factor for exposure
        #self.binned = fileutil.getKeyword(filename+'[sci,1]', 'BINAXIS1')

        # Build up list of chips in observation
        self.addMembers(filename)

        # Only need to worry about IDC tables for ACS...
        self.computeOffsets()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)


class WFPCObservation(Pattern):
    """This class defines an observation with information specific
       to WFPC2 exposures, including knowledge of how to mosaic the
       chips."""

    # Default coefficients table to use for this instrument
    IDCKEY = 'idctab'

    # This parity is the multiplication of PC1 rotation matrix with
    # a flip in X for output image.
    #__theta = 44.67
    __pmat = np.array([[-1.,0.],[0.,1.]])
    __refchip = 3
    PARITY = {'1':__pmat,'2':__pmat,'3':__pmat,'4':__pmat,'WFPC':__pmat}

    NUM_IMSET = 1

    # The dictionaries 'REFDATA' and 'REFPIX' are required for use with
    # cubic and Trauger coefficients tables in 'computeCubicCoeffs'.
    #
    # This information provides the absolute relationship between the chips

    #REFDATA ={'1':{'psize':0.04554,'xoff':354.356,'yoff':343.646,'v2':2.374,'v3':-30.268,'theta':224.8480},
    #         '2':{'psize':0.0996,'xoff':345.7481,'yoff':375.28818,'v2':-51.368,'v3':-5.698,'theta':314.3520},
    #          '3':{'psize':0.0996,'xoff':366.56876,'yoff':354.79435,'v2':0.064,'v3':48.692,'theta':44.67},
    #         '4':{'psize':0.0996,'xoff':355.85016,'yoff':351.29183,'v2':55.044,'v3':-6.098,'theta':135.2210}}
    #REFPIX = {'x':400.,'y':400.}

    def __init__(self, filename, output, pars=None):

        # Now initialize Pattern with all member exposures...
        Pattern.__init__(self, filename, output=output, pars=pars)

        self.instrument = 'WFPC2'
        self.REFDATA ={'1':{'psize':0.04554,'xoff':354.356,'yoff':343.646,'v2':2.374,'v3':-30.268,'theta':224.8480},
        '2':{'psize':0.0996,'xoff':345.7481,'yoff':375.28818,'v2':-51.368,'v3':-5.698,'theta':314.3520},
        '3':{'psize':0.0996,'xoff':366.56876,'yoff':354.79435,'v2':0.064,'v3':48.692,'theta':44.67},
        '4':{'psize':0.0996,'xoff':355.85016,'yoff':351.29183,'v2':55.044,'v3':-6.098,'theta':135.2210}}

        self.REFPIX = {'x':400.,'y':400.}
        gcount = None

        # build output rootnames here...
        self.setNames(filename,output)

        # Set EXPTIME for exposure
        self.exptime = self.getExptime()

        _mode = fileutil.getKeyword(filename, 'MODE')
        if _mode == 'AREA':
            self.binned = 2
            if self.idckey == 'cubic':
                for l in self.REFPIX.keys(): self.REFPIX[l]= self.REFPIX[l] / self.binned
                for l in self.REFDATA.keys():
                    self.REFDATA[l]['psize'] = self.REFDATA[l]['psize'] * self.binned
                    self.REFDATA[l]['xoff'] = self.REFDATA[l]['xoff'] / self.binned
                    self.REFDATA[l]['yoff'] = self.REFDATA[l]['yoff'] / self.binned

        # Now, build list of members and initialize them
        self.addMembers(filename)
        self.setBunit('COUNTS')

        chips = [int(member.chip) for member in self.members]
        try:
            chip_ind = chips.index(self.__refchip)
        except ValueError:
            chip_ind = 0
        self.refchip = chips[chip_ind]

        if self.members[0].geometry.ikey != 'idctab':
            # Correct distortion coefficients to match output pixel scale
            self.computeCubicCoeffs()
        else:
            #self.computeOffsets(refchip=self.__refchip)
            self.computeOffsets(refchip=self.refchip)
        # Determine desired orientation of product
        self.setOrient()

        # Set up the input members and create the product meta-chip
        self.buildProduct(filename, output)

        # Correct the crpix position of the metachip product
        # in order to account for 'align=center' convention.
        #self.product.geometry.wcs.crpix1 -= 1.0
        #self.product.geometry.wcs.crpix2 -= 1.0

    def addMembers(self,filename):

        # The PC chip defines the orientation of the metachip, so use
        # it for the PARITY as well.
        self.detector = 'WFPC'

        _chip1_rot = None
        # Build rootname here for each SCI extension...
        if self.pars['section'] == None:
            self.pars['section'] = [None] * self.nmembers
            group_indx = range(1,self.nmembers+1)
        else:
            group_indx = self.pars['section']

        for i in range(self.nmembers):
            _extname = self.imtype.makeSciName(i+1,section=self.pars['section'][i])

            _detnum = fileutil.getKeyword(_extname,self.DETECTOR_NAME)

            # Start by looking for the corresponding WFPC2 'c1h' files
            _dqfile, _dqextn = self._findDQFile()

            # Reset dqfile name in ImType class to point to new file
            self.imtype.dqfile = _dqfile
            self.imtype.dq_extn = _dqextn

            # Build mask file for this member chip
            _dqname = self.imtype.makeDQName(extver=group_indx[i])
            _masklist = []
            _masknames = []

            if _dqname != None:
                _maskname = buildmask.buildMaskName(fileutil.buildNewRootname(_dqname),_detnum)
            else:
                _maskname = None
            _masknames.append(_maskname)

            outmask = buildmask.buildShadowMaskImage(_dqname,_detnum,group_indx[i],_maskname, bitvalue=self.bitvalue[0], binned=self.binned)
            _masklist.append(outmask)

            _maskname = _maskname.replace('final_mask','single_mask')
            _masknames.append(_maskname)
            outmask = buildmask.buildShadowMaskImage(_dqname,_detnum,group_indx[i],_maskname, bitvalue=self.bitvalue[1], binned=self.binned)
            _masklist.append(outmask)
            _masklist.append(_masknames)


            self.members.append(Exposure(_extname, idckey=self.idckey, dqname=_dqname,
                mask=_masklist, parity=self.PARITY[str(i+1)],
                idcdir=self.pars['idcdir'], group_indx = i+1,
                rot=_chip1_rot, handle=self.image_handle, extver=_detnum,
                exptime=self.exptime[0], ref_pscale=self.REFDATA['1']['psize'], binned=self.binned))

            if self.idckey != 'idctab':
                _chip1_rot = self.members[0].geometry.def_rot

    def _findDQFile(self):
        """ Find the DQ file which corresponds to the input WFPC2 image. """
        dqfile=""
        dqextn = ""
        if self.name.find('.fits') < 0:
            # Working with a GEIS image...
            dqfile = self.name[:-2]+'1h'
            dqextn = '[sdq,1]'
        else:
            # Looking for c1f FITS DQ file...
            # In the WFPC2 pipeline the extensions are called 'c0f' and 'c1f'
            # and EXTNAME is 'sci'. Files are in MEF format.
            # Readgeis creates output with extensions 'c0h' and 'c1h'
            # and EXPNAME is 'sdq'
            # Hence the code below ...
            if 'c0h.fits' in self.name:
                dqfile = self.name.replace('0h.fits','1h.fits')
                dqextn = '[sdq,1]'
            elif 'c0f.fits' in self.name:
                dqfile = self.name.replace('0f.fits','1f.fits')
                dqextn = '[sci,1]'
            elif 'c0m.fits' in self.name:
                dqfile = self.name.replace('0m.fits','1m.fits')
                dqextn = '[sci,1]'

        return dqfile, dqextn

    def setOrient(self):

        """ Determine desired orientation of product."""
        meta_orient = None
        for exp in self.members:
            if int(exp.chip) == 1:
                meta_orient = exp.geometry.wcslin.orient

        if meta_orient == None:
            meta_orient = self.members[0].geometry.wcslin.orient

        # Set orient for all groups
        # Dither coefficients rotate all chips to chip 1 orientation
        # if not using Trauger coefficients
        for exp in self.members:
            exp.geometry.wcs.orient = meta_orient
            # We need to correct each chip's full-frame reference
            # position to account for 'align=center' convention.
            #exp.geometry.wcs.chip_xref += 1.0
            #exp.geometry.wcs.chip_yref += 1.0

