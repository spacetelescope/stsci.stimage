import string
import time
import os, shutil, sys

import numarray
import numarray.ieeespecial
from numarray.ieeespecial import *

import pydrizzle
from pydrizzle import drutil, fileutil, buildasn, updateasn, dqpars
import pyfits

import acs_input

import mdzhandler

import manager
from manager import ImageManager

__version__ = '2.0.1 (6 July 2004)'

def printout(text):
    print(' *** ' + text + '\n ***')

def timestamp():
    print('\n----------------------------')
    print time.strftime('%c %Z', time.localtime(time.time()))
    print('----------------------------\n')

def versioninfo():
    """ Print version information for packages used by Multidrizzle """

    # Set up version ID's for printing to the log file
    _mdrizzle_version  = " MultiDrizzle "+__version__
    _numarray_version  = " Numarray Version  "+numarray.__version__
    _pydrizzle_version = " PyDrizzle Version "+pydrizzle.__version__
    _pyfits_version    = " PyFITS Version    "+pyfits.__version__

    _sys_version_list = sys.version.split("\n")
    _sys_version = " # "
    for _sys_str in _sys_version_list:
        _sys_version += _sys_str+"\n # "
    _python_version   = " Python Version: \n" +_sys_version

    # Print out version information for libraries used by MultiDrizzle
    print "\n\n"
    print " Version Information for"+_mdrizzle_version
    print "-------------------------------------------- "
    try:
        import pyraf
        _pyraf_version     = " PyRAF Version     "+pyraf.__version__
        print _pyraf_version
    finally:
        print _numarray_version
        print _pyfits_version
        print _pydrizzle_version
        print _python_version
        print "\n\n"

def toBoolean(flag):
    if (flag == 1):
        return True
    return False

def cleanNaN(value):
    a = numarray.array(value)
    b = getnan(a)
    if len(b[0]) == 1:
        return None
    return value

def cleanInt(value):
    # THIS MAY BE MACHINE-DEPENDENT !!!
    #if value == -2147483647:
    # Try to use 'sys.maxint' as value (WJH)
    if value == -sys.maxint:
        return None
    return value

def cleanBlank(value):
    if value == '' or value == ' ':
        return None
    return value

def BuildDrizPars(outnx   = None,
                  outny   = None,
                  kernel  = "turbo",
                  pixfrac = 1.0,
                  scale   = None,
                  rot     = None,
                  fillval = "INDEF"):
    """ Function that builds PyDrizzle parameter sets.
        Note that only parameters that can change from the
        drizzle separate step to the final drizzle step are
        handled here. Parameters that are common to both runs
        are handled by the Multidrizzle constructor instead.
    """
    pars = {}

    pars['outnx']   = outnx
    pars['outny']   = outny
    pars['kernel']  = kernel
    pars['pixfrac'] = pixfrac
    pars['scale']   = scale
    pars['rot']     = rot
    pars['fillval'] = str(fillval)

    return pars

def BuildInstrPars(gain       = '1.0',
                   gainkw     = None,
                   rdnoise    = 0.0,
                   rdnoisekw  = None,
                   exptime    = 1.0,
                   exptimekw  = None,
                   crbit      = 64):
    """ Function that builds instrument parameter sets.
    """
    pars = {}

    pars['gain']      = gain
    pars['gainkw']    = gainkw
    pars['rdnoise']   = rdnoise
    pars['rdnoisekw'] = rdnoisekw
    pars['exptime']   = exptime
    pars['exptimekw'] = exptimekw
    pars['crbit']     = crbit

    return pars


class Multidrizzle:

    def __init__(self,
                 input      = 'flt.fits',
                 output     = None,
                 mdriztab   = False,
                 refimage   = '',
                 runfile    = 'multidrizzle.run',
                 workinplace = False,
                 context    = True,
                 clean      = True,
                 group      = None,
                 bits       = 0,
                 ra         = None,
                 dec        = None,
                 coeffs     = 'header',
                 build      = False,
                 driz_sep   = None,
                 driz_final = None,
                 shiftfile  = None,
                 staticfile = None,
                 static_sig = 3.0,
                 instrpars  = None):

        timestamp()
        print 'Running MultiDrizzle ',__version__

        # Create object that controls step execution and mark
        # initialization step.

        self.steps = ProcSteps()
        self.steps.doStep(ProcSteps.doInitialize)

        self.skypars    = {}  # These should be initialized with defaults
        self.medianpars = {}  # in order to allow the user to skip running
        self.drizcrpars = {}  # the 'set' methods for these parameters when
        self.blotpars   = {}  # using the defaults.

        self.tabswitches = {} # Dictionary with switches from MDRIZTAB

        # Finalize building PyDrizzle and instrument parameters.
        # If not defined by the user, use defaults.

        if driz_sep == None:
            driz_sep = BuildDrizPars()
        if driz_final == None:
            driz_final = BuildDrizPars()

        self.driz_sep_pars = driz_sep
        self.driz_sep_pars['group'] = group
        self.driz_sep_pars['bits'] = bits
        self.driz_sep_pars['ra'] = ra
        self.driz_sep_pars['dec'] = dec
        self.driz_sep_pars['coeffs'] = coeffs
        self.driz_sep_pars['build'] = False

        self.driz_final_pars = driz_final
        self.driz_final_pars['group'] = group
        self.driz_final_pars['bits'] = bits
        self.driz_final_pars['ra'] = ra
        self.driz_final_pars['dec'] = dec
        self.driz_final_pars['coeffs'] = coeffs
        self.driz_final_pars['build'] = build

        self.instrpars = instrpars
        if self.instrpars == None:
            self.instrpars = BuildInstrPars()

        # Remember input parameters for use throughout this class.

        self.input = input
        self.refimage = refimage
        self.context = context
        self.workinplace = workinplace
        self.clean = clean
        self.shiftfile = shiftfile
        self.staticfile = staticfile
        self.static_sig = static_sig
        self.mdriztab = mdriztab
        # Remember the name for the output script for the final drizzling
        self.runfile = runfile
        self.clean = clean
        self.coeffs = coeffs

        # Parse input to get the list of filenames to be processed.
        self.files,self.output = self._parseInput(input,output)

        # Create copies of input files for processing
        if not self.workinplace:
            self._createInputCopies(self.files)
        else:
            print "\n\n********************"
            print "WARNING:  Sky will be subtracted from sci extensions"
            print "WARNING:  Units of sci extensions will be electrons"
            print "WARNING:  Value of MDRIZSKY is in units of input data sci extensions."
            print "********************\n\n"


        # Check input files.
        self._checkInputFiles(self.files)
        self._reportFileNames()

        # MDRIZTAB must be opened here, before the association is built.
        if mdriztab:
            record = mdzhandler.getMultidrizzleParameters(self.files)
            self._handleMdriztab(record)

        # Initialize the dqpar file based upon instrument type
        _instrument = fileutil.getKeyword(self.files[0]+'[0]','INSTRUME')
        self._initdqpars(_instrument,self.driz_sep_pars['bits'])

        # Build association object
        association = self._setupAssociation()

        # Build the manager object.

        self.image_manager = ImageManager(association, self.context, self.instrpars, self.workinplace)

        # Once MDRIZTAB is ingested, a new set of instrument parameters
        # becomes available. It must supersede the set that was initially
        # passed to the manager object via its constructor.

        if mdriztab:
            self.image_manager.setInstrumentParameters(self.instrpars)

        # Check static file. Can only be done after reading MDRIZTAB.
        self._checkStaticFile(self.staticfile)

        # Done with initialization.
        self.steps.markStepDone(ProcSteps.doInitialize)

    def _printInputPars(self):
        print "\n\n**** Multidrizzle Parameter Input ****"

        print "\n** General Parameters:"
        print "input = ", self.input
        print "output  = ",self.output
        print "mdriztab = ", self.mdriztab
        print "refimage  = ", self.refimage
        print "runfile = ", self.runfile
        print "workinplace = ", self.workinplace
        print "context = ", self.context
        print "clean  = ", self.clean
        print "shiftfile =  ",self.shiftfile
        
        print "\n** Static Mask:"
        print "staticfile = ",self.staticfile
        print "static_sig = ", self.static_sig

        print "\n** Sky Subtraction:"
        self._printDictEntries(self.skypars)
                
        print "\n** Separate Drizzle Images:"
        self._printDictEntries(self.driz_sep_pars)
        
        print "\n** Create Median Image:"
        self._printDictEntries(self.medianpars)
 
        print "\n** Blot Back the Median Image:"
        self._printDictEntries(self.blotpars)

        print "\n** Remove Cosmic Rays with DERIV, DRIZ_CR:"
        self._printDictEntries(self.drizcrpars)

        print "\n** Combined Drizzled Image:"
        self._printDictEntries(self.driz_final_pars)
        
        print "\n** Instrument Parameters:"
        self._printDictEntries(self.instrpars)
        print "\n"

    def _printDictEntries(self,dict):
        for item in dict.items():
            print item[0], " = ", item[1]
            
    def _initdqpars(self,_instrument,_bits):
        print "Initializing DQpars file..."
        if (_instrument.lower() == 'acs'):
            _parfile = dqpars.ACSPars()
            _parfile.update(_bits)
        else:
            print " "
            print "****************"
            print "*"
            print "* UNABLE TO IDENTIFY INSTRUMENT DQPARAMETERS.  ALL DQ PARAMETERS TREATED AS GOOD"
            print "*"
            print "****************"

    def _handleMdriztab(self, rec):

        # Collect task parameters from the MDRIZTAB record and
        # stick them into the dictionaries used by Multidrizzle.

        # Note that parameters read from the MDRIZTAB record must
        # be cleaned up in a similar way that parameters read
        # from the user interface are.

        self.setSkyPars(
            skywidth  = cleanNaN(rec.field('skywidth')),
            skystat   = cleanBlank(rec.field('skystat')),
            skylower  = cleanNaN(rec.field('skylower')),
            skyupper  = cleanNaN(rec.field('skyupper')),
            skyclip   = cleanInt  (rec.field('skyclip')),
            skylsigma = cleanNaN(rec.field('skylsigma')),
            skyusigma = cleanNaN(rec.field('skyusigma')))

##        print self.skypars

        self.setMedianPars(
            newmasks   = toBoolean (rec.field('median_newmasks')),
            type       = cleanBlank(rec.field('combine_type')),
            nsigma     = cleanBlank(rec.field('combine_nsigma')),
            nlow       = cleanInt  (rec.field('combine_nlow')),
            nhigh      = cleanInt  (rec.field('combine_nhigh')),
            lthresh    = cleanNaN(rec.field('combine_lthresh')),
            hthresh    = cleanNaN(rec.field('combine_hthresh')),
            grow       = cleanNaN(rec.field('combine_grow')))

##        print self.medianpars

        self.setDrizCRPars(
            driz_cr_snr   = cleanBlank(rec.field('driz_cr_snr')),
            driz_cr_scale = cleanBlank(rec.field('driz_cr_scale')),
            driz_cr_corr  = toBoolean (rec.field('driz_cr_corr')))

##        print self.drizcrpars
        self.setBlotPars(
# Uncomment these if the column(s) are added to the MDRIZTAB
#            blot_interp  = cleanBlank(rec.field('blot_interp')),
#            blot_sinscl  = cleanNaN(rec.field('blot_sinscl'))
             blot_interp = 'poly5',
             blot_sinscl = 1.0
        )

        self.instrpars = BuildInstrPars(
            gain      = cleanBlank(rec.field('gain')),
            gainkw    = cleanBlank(rec.field('gnkeyword')),
            rdnoise   = cleanBlank(rec.field('readnoise')),
            rdnoisekw = cleanBlank(rec.field('rnkeyword')),
            exptime   = cleanNaN(rec.field('exptime')),
            exptimekw = cleanBlank(rec.field('expkeyword')),
            crbit     = cleanInt  (rec.field('crbitval')))

##        print self.instrpars

        _fillval = cleanNaN(rec.field('driz_sep_fillval'))
        if _fillval == None:
            _fillval = "INDEF"

        self.driz_sep_pars = BuildDrizPars(
            outnx                    = cleanInt  (rec.field('driz_sep_outnx')),
            outny                    = cleanInt  (rec.field('driz_sep_outny')),
            kernel                   = cleanBlank(rec.field('driz_sep_kernel')),
            pixfrac                  = cleanNaN(rec.field('driz_sep_pixfrac')),
            scale                    = cleanNaN(rec.field('driz_sep_scale')),
            rot                      = cleanNaN(rec.field('driz_sep_rot')),
            fillval                  = str(_fillval))
        self.driz_sep_pars['group']  = cleanBlank(rec.field('group'))
        self.driz_sep_pars['bits']   = cleanInt  (rec.field('bits'))
        self.driz_sep_pars['ra']     = cleanNaN(rec.field('ra'))
        self.driz_sep_pars['dec']    = cleanNaN(rec.field('dec'))
        self.driz_sep_pars['coeffs'] = cleanBlank(rec.field('coeffs'))
        self.driz_sep_pars['build']  = False

##        print self.driz_sep_pars

        _fillval = cleanNaN(rec.field('final_fillval'))
        if _fillval == None:
            _fillval = "INDEF"


        self.driz_final_pars = BuildDrizPars(
            outnx                      = cleanInt  (rec.field('final_outnx')),
            outny                      = cleanInt  (rec.field('final_outny')),
            kernel                     = cleanBlank(rec.field('final_kernel')),
            pixfrac                    = cleanNaN(rec.field('final_pixfrac')),
            scale                      = cleanNaN(rec.field('final_scale')),
            rot                        = cleanNaN(rec.field('final_rot')),
            fillval                    = str(_fillval))
        self.driz_final_pars['group']  = cleanBlank(rec.field('group'))
        self.driz_final_pars['bits']   = cleanInt  (rec.field('bits'))
        self.driz_final_pars['ra']     = cleanNaN(rec.field('ra'))
        self.driz_final_pars['dec']    = cleanNaN(rec.field('dec'))
        self.driz_final_pars['coeffs'] = cleanBlank(rec.field('coeffs'))
        self.driz_final_pars['build']  = toBoolean(rec.field('build'))

##        print self.driz_final_pars

        self.tabswitches['static'] = toBoolean(rec.field('static'))
        self.tabswitches['subsky'] = toBoolean(rec.field('subsky'))
        self.tabswitches['driz_separate'] = toBoolean(rec.field('driz_separate'))
        self.tabswitches['median'] = toBoolean(rec.field('median'))
        self.tabswitches['blot'] = toBoolean(rec.field('blot'))
        self.tabswitches['driz_cr'] = toBoolean(rec.field('driz_cr'))
        self.tabswitches['driz_combine'] = toBoolean(rec.field('driz_combine'))

        # Now collect remaining assorted task parameters.

        self.refimage = cleanBlank(rec.field('refimage'))
        self.context = toBoolean(rec.field('context'))
        self.shiftfile = cleanBlank(rec.field('shiftfile'))
        self.staticfile = cleanBlank(rec.field('staticfile'))
#        self.static_sig = rec.field('static_sig')
        self.runfile = cleanBlank(rec.field('runfile'))
        self.clean = toBoolean(rec.field('clean'))

    def setSkyPars(self,
                   skywidth = 50.,
                   skystat  = 'median',
                   skylower = -50.,
                   skyupper = 200.,
                   skyclip = 5,
                   skylsigma = 4,
                   skyusigma = 4,
                   skyuser = ''
                   ):

        """ Sets the sky parameters. """
        self.skypars['skywidth']  = skywidth
        self.skypars['skystat']   = skystat
        self.skypars['skylower']  = skylower
        self.skypars['skyupper']  = skyupper
        self.skypars['skyclip']   = skyclip
        self.skypars['skylsigma'] = skylsigma
        self.skypars['skyusigma'] = skyusigma
        self.skypars['skyuser']   = skyuser

    def setDrizCRPars(self,
                      driz_cr_snr = '3.0 2.5',
                      driz_cr_scale = '1.2 0.7',
                      driz_cr_corr = False):
        """ Sets the driz CR parameters. """
        self.drizcrpars['snr'] = driz_cr_snr
        self.drizcrpars['scale'] = driz_cr_scale
        self.drizcrpars['corr_file'] = driz_cr_corr


    def setBlotPars(self,
                    blot_interp = 'poly5',
                    blot_sinscl = 1.0):
        self.blotpars['interp'] = blot_interp
        self.blotpars['sinscl'] = blot_sinscl

    def setMedianPars(self,
                      newmasks   = True,
                      type       = "median",
                      nsigma     = "6 3",
                      nlow       = 0,
                      nhigh      = 1,
                      lthresh    = None,
                      hthresh    = None,
                      grow       = 1.0):
        """ Sets the median parameters. """

        (_nsigma1, _nsigma2) = self._splitNsigma(nsigma)

        self.medianpars['newmasks'] = newmasks
        self.medianpars['type']     = type
        self.medianpars['nsigma1']  = _nsigma1
        self.medianpars['nsigma2']  = _nsigma2
        self.medianpars['nlow']     = nlow
        self.medianpars['nhigh']    = nhigh
        self.medianpars['lthresh']  = lthresh
        self.medianpars['hthresh']  = hthresh
        self.medianpars['grow']     = grow

    def _createInputCopies(self,files):
        """ Creates copies of all input images."""

        for _img in files:
            # Only make copies of files that exist
            if os.path.exists(_img):
                # Create filename for copy
                _copy = manager.modifyRootname(_img)
                # Make sure we remove any previous copies first...
                if os.path.exists(_copy): os.remove(_copy)

                # Copy file into new filename
                shutil.copyfile(_img,_copy)

    def _checkInputFiles(self, files):

        """ Checks input files before they are required later. """

        # Make sure that UPINCD is run on any ACS images.
        acs_input.checkACS(files)

    def _checkStaticFile(self, static_file):

        """ Checks input files before they are required later. """

        # Checks existence of static mask.
        # Setup error string in case it can not find file...
        if (static_file != None):
            _err_str = "Cannot find static mask file: " + static_file
            try:
                # This call avoids unnecessary file open calls.
                _exist = fileutil.checkFileExists(static_file)
                if not _exist:
                    raise ValueError, _err_str
            except IOError:
                raise ValueError, _err_str

    def _parseInput(self,input,output):

        """ Interprets input from user to build list of files to process. """

        _flist = []
        _indx = input.find('_asn.fits')

        if  _indx > -1:
            # Input is an ASN table, so read it and return the input filenames
            asndict = fileutil.readAsnTable(input, None, prodonly=False)
            if not output:
                output = asndict['output']+'_drz.fits'

            for f in asndict['members'].keys():
                if f != 'abshift' and f != 'dshift':
                    _flist.append(fileutil.buildRootname(f))
        elif input.find(',') > -1:
            # We have been given a list already, so format it as necessary
            _flist = input.split(',')
        else:
            # We are working with either a suffix pattern,
            # a user-supplied at-file with a list of filenames or
            # a single filename for specifying the input file(s).
            # Parse this and build the appropriate list of filenames
            _files = buildasn._findFiles(input)
            for f in _files:
                _flist.append(f[0])

        # Setup default output name if none was provided either by user or in ASN table
        if (output == None) or (len(output) == 0):
            output = 'final_drz.fits'

        return _flist,output

    def _reportFileNames(self):
        print "Input files: "
        for filename in self.files:
            print "             " + filename
            print "Output file: " + self.output

    def _setupAssociation(self):

        """ Builds the PyDrizzle association object. """

        timestamp()
        printout('Setting up associated inputs..')

        # Keep track of whether an ASN table needs to be built at all
        # For single input exposures, no ASN table needs to be created
        _buildasn = True
        if self.input.find('_asn.fits') > -1:
            driz_asn_file = self.input
        elif fileutil.findFile(self.input):
            driz_asn_file = self.input
            _buildasn = False
        else:
            driz_asn_file = self._buildAsnName(self.output)

        if _buildasn:
            # Is this a completely new run, or is it a continuation from
            # an earlier run in this directory?
            newrun = self._checkAndUpdateAsn(driz_asn_file, self.shiftfile)

            self._generateAsnTable(driz_asn_file, self.output,
                                   self.input, self.shiftfile)

        # Run PyDrizzle; make sure there are no intermediate products
        # laying around...
        assoc = pydrizzle.PyDrizzle(driz_asn_file,
                                    section=self.driz_sep_pars['group'],
                                    prodonly=False)

        # Use PyDrizzle to clean up any previously produced products...
        if self.clean:
            assoc.clean()

        self._setOutputFrame(assoc)
        print 'Initial parameters: '
        assoc.printPars(format=1)

        return assoc

    def _setOutputFrame(self, association):

        """ Set up user-specified output frame using a SkyField object."""

        _sky_field = None

        if self.refimage != '' and self.refimage != None:
            # Use the following if the refimage isn't actually going to be
            # drizzled, we just want to set up the pydrizzle object
            #
            _refimg = pydrizzle.wcsutil.WCSObject(self.refimage)
            refimg_wcs = _refimg.copy()

            # If the user also specified a rotation to be applied,
            # apply that as well...
            if self.drizpars['rot']:
                _orient = self.drizpars['rot']
            else:
                _orient = refimg_wcs.orientat

           # Now, build output WCS using the SkyField class
            # and default product's WCS as the initial starting point.
            #
            _sky_field = pydrizzle.SkyField(wcs=refimg_wcs)
            # Update with user specified scale and rotation
            _sky_field.set(psize=self.driz_sep_pars['scale'],orient=_orient)

        elif self.driz_sep_pars['rot']   != None  or \
             self.driz_sep_pars['scale'] != None or \
             self.driz_sep_pars['ra']    != None:

            _sky_field = pydrizzle.SkyField()

            if self.driz_sep_pars['rot'] == None:
                _orient = association.observation.product.geometry.wcslin.orient
            else:
                _orient = self.driz_sep_pars['rot']

            print 'Default orientation for output: ',_orient,'degrees'

            _sky_field.set(psize=self.driz_sep_pars['scale'],
                           orient=_orient,
                           ra=self.driz_sep_pars['ra'],
                           dec=self.driz_sep_pars['dec'])

        # Now that we have built the output frame, let the user know
        # what was built...
        if _sky_field != None:
            print ('\n Image parameters computed from reference image WCS: \n')
            print _sky_field.wcs

            # Apply user-specified output to ASN using the resetPars method.
            # If field==None, it will simply reset to default case.
            #
            association.resetPars(field=_sky_field)

    def _generateAsnTable(self, name, output, input, shfile):

        """ Generates the association table. """

        if not fileutil.findFile(name):
            print 'building ASN table for: ',output
            print 'based on suffix of:     ',input
            print 'with a shiftfile of:    ',shfile

            # Must strip out _asn.fits from name since buildAsnTable adds
            # it back again...
            buildasn.buildAsnTable(name[:string.find(name,'_asn.fits')],
                                   suffix=input,
                                   shiftfile=shfile)

        print 'Assoc. table = ', name
        print('')
##        tprint(name,prparam=no)
        print('')

    def _checkAndUpdateAsn(self, name, shfile):
        """ Updates association table with user-supplied shifts, if given. """
        new_run = True
        if fileutil.checkFileExists(name):
            new_run = False
            if (shfile != None) and (len(shfile) > 0):
                updateasn.updateShifts(name,shfile,mode='replace')
        return new_run

    def _buildAsnName(self, name):
        """ Builds name for association table. """
        for (search_for,suffix) in [('_asn.fits',''),('_asn','.fits'),('','_asn.fits')]:
            if name.find(search_for) > -1:
                return name + suffix

    def _preMedian(self, static_file, skysub):
        """ Perform the steps that take place before median computation:
            build static mask, subtract sky, drizzle into separate images.
        """
        # Build static mask

        if self.steps.doStep(ProcSteps.doBuildStatic):
            self.image_manager.createStatic(static_file,static_sig=self.static_sig)
            self.steps.markStepDone(ProcSteps.doBuildStatic)

        # Process sky.
        #
        # Note that we must use the boolean flag returned by the step
        # controller, not the local skysub variable. This is because
        # the MDRIZTAB reading routine may redefine the flag; thus
        # rendering the local variable inaccurate.
        if self.steps.getFlag(ProcSteps.doSky):
            self.steps.printTimestamp(ProcSteps.doSky)
        self.steps.resetStep(ProcSteps.doSky)
        self.image_manager.doSky(self.skypars,self.steps.getFlag(ProcSteps.doSky))
        self.steps.markStepDone(ProcSteps.doSky)

        # Drizzle separate

        if self.steps.doStep(ProcSteps.doDrizSeparate):
            self.image_manager.doDrizSeparate (self.driz_sep_pars)
            self.steps.markStepDone(ProcSteps.doDrizSeparate)

    def _splitNsigma(self,s):

        # Split up the "combine_nsigma" string. If a second value is
        # specified, then this will be used later down in the "minmed"
        # section where a second-iteration rejection is done. Typically
        # the second value should be around 3 sigma, while the first
        # can be much higher.

        _sig = string.split(s, " ")
        _nsigma1 = float(_sig[0])
        _nsigma2 = float(_sig[0])
        if len(_sig) > 1:
            _nsigma2 = float(_sig[1])
        return (_nsigma1, _nsigma2)

    def _postMedian(self, blotpars, drizcrpars, skypars):
        """ Perform the steps that take place after median computation:
            blot, and drizcr.
        """

        # Blot

        if self.steps.doStep(ProcSteps.doBlot):
            self.image_manager.doBlot(blotpars)
            self.steps.markStepDone(ProcSteps.doBlot)

        # Driz CR

        if self.steps.doStep(ProcSteps.doDrizCR):
            self.image_manager.doDrizCR(drizcrpars, skypars)
            self.steps.markStepDone(ProcSteps.doDrizCR)

    def run(self,
            static          = True,
            skysub          = True,
            driz_separate   = True,
            median          = True,
            blot            = True,
            driz_cr         = True,
            driz_combine    = True,
            timing          = True):

        # Print the input parameters now that MDRIZTAB has had a chance to modify the default values
        self._printInputPars()


        # Update object that controls step execution. Use either user
        # interface switches, or MDRIZTAB switches.

        if len(self.tabswitches.keys()) > 0:
            self.steps.addSteps(self.tabswitches['static'],
                                self.tabswitches['subsky'],
                                self.tabswitches['driz_separate'],
                                self.tabswitches['median'],
                                self.tabswitches['blot'],
                                self.tabswitches['driz_cr'],
                                self.tabswitches['driz_combine'])
        else:
            self.steps.addSteps(static, skysub, driz_separate, median,
                                blot, driz_cr, driz_combine)

        # Insure that if an error occurs,
        # all file handles are closed before exiting...
        try:
            self._preMedian(self.staticfile,skysub)

            if self.steps.doStep(ProcSteps.doMedian):
                self.image_manager.createMedian(self.medianpars)
                self.steps.markStepDone(ProcSteps.doMedian)

            self._postMedian(self.blotpars, self.drizcrpars, self.skypars)

            if self.steps.doStep(ProcSteps.doFinalDriz):
                self.image_manager.doFinalDriz(self.driz_final_pars, self.runfile)
                self.steps.markStepDone(ProcSteps.doFinalDriz)

        finally:
            # Close open file handles opened by PyDrizzle
            # Now that we are done with the processing,
            # delete any input copies we created.
            if not self.workinplace:
                self.image_manager.removeInputCopies()

            # If clean has been set, remove intermediate products now.
            if self.clean:
                # Start by deleting the runfile
                if os.path.exists(self.runfile):
                    os.remove(self.runfile)

                # Now have image_manager remove all image products
                self.image_manager.removeMDrizProducts()

        if timing:
            self.steps.reportTimes()
        versioninfo()



class ProcSteps:
    """ The ProcSteps class encapsulates the logic for deciding
        which processing steps get performed by Multidrizzle based
        on user or mdriztab input. It also keeps track of elapsed
        time used by each step.
    """

    # Step names are kept as class variables so they can be
    # used to access individual step info.

    doInitialize    = '1  Initialize:         '
    doBuildStatic   = '2  Build static mask:  '
    doSky           = '3  Subtract sky:       '
    doDrizSeparate  = '4  Drizzle separate:   '
    doMedian        = '5  Median/sum/ave.:    '
    doBlot          = '6  Blot:               '
    doDrizCR        = '7  DrizCR:             '
    doFinalDriz     = '8  Final drizzle:      '

    __report_header = '   Step                Elapsed time'

    def __init__(self):

        # Step objects are kept in a dictionary keyed by step names.
        # Dictionary starts with initialization step added in.

        self.__steps = {}

        self.__steps[ProcSteps.doInitialize] = \
            ProcStep(True, 'Initializing...')

    def addSteps(self, static, skysub, driz_separate, median, blot,
                 driz_cr, driz_combine):

        self.__steps[ProcSteps.doBuildStatic] = \
            ProcStep(static,'Building static bad-pixel mask...')

        self.__steps[ProcSteps.doSky] = \
            ProcStep(skysub,'Subtracting sky...')

        self.__steps[ProcSteps.doDrizSeparate] = \
            ProcStep(driz_separate,'Drizzling separate...')

        self.__steps[ProcSteps.doMedian] = \
            ProcStep(median,'Computing combined image...')

        self.__steps[ProcSteps.doBlot] = \
            ProcStep(blot,'Blotting back the combined image...')

        self.__steps[ProcSteps.doDrizCR] = \
                ProcStep(driz_cr,'Doing driz_cr...')

        self.__steps[ProcSteps.doFinalDriz] = \
            ProcStep(driz_combine,'Doing final drizzle...')

    def getFlag(self, step_name):
        """ Gets the boolean flag associated with this step. """

        return self.__getUserSelection(step_name)

    def doStep(self, step_name):
        """ Checks if a step should be performed. """

        if self.getFlag(step_name) and \
           self.__isCompleted(step_name) == False:

            self.__steps[step_name].recordStartTime()

            self.printTimestamp(step_name)

            return True

        else:
            return False

    def printTimestamp(self, step_name):
        """ Prints a time stamp message. """

        self.__steps[step_name].printTimestamp()

    def reportTimes(self):
        """ Generates report with elapsed times used by each step. """
        keys = self.__steps.keys()
        keys.sort()

        print '\n' + ProcSteps.__report_header + '\n'

        _total = 0
        for key in keys:
            _time = self.__steps[key].getElapsedTime()
            _total += _time
            print key + str(_time) + ' sec.'

        print '   Total               ' + str(_total) + 'sec.'

    # Methods delegate the query to the step object asociated
    # with the provided step name.

    def markStepDone(self, step_name):
        """ Records that step has been completed. """
        self.__steps[step_name].markStepDone()

    def resetStep(self, step_name):
        """ Resets the status of the processing step to not completed."""
        self.__steps[step_name].resetStep()

    def __getUserSelection(self, step_name):
        """ Returns status of user setting. """
        return self.__steps[step_name].getUserSelection()

    def __isCompleted(self, step_name):
        """ Returns status of step processing: complete or not. """
        return self.__steps[step_name].isCompleted()


class ProcStep:
    """ This class encapsulates the information
        associated with the processing status of
        a single execution step.
    """

    def __init__(self, user_switch, message):
        self.__message = message
        self.__switch = user_switch
        self.__elapsed_time = 0

        self.resetStep()

    def recordStartTime(self):
        self.__start_time = time.time()

    def markStepDone(self):
        self.__elapsed_time = time.time() - self.__start_time
        self.__completed = True

    def isCompleted(self):
        return self.__completed

    def resetStep(self):
        self.recordStartTime()
        self.__completed = False

    def getUserSelection(self):
        return self.__switch

    def getElapsedTime(self):
        return self.__elapsed_time

    def printTimestamp(self):
        timestamp()
        printout(self.__message)
        print('')
