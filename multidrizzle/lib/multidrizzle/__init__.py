"""
MultiDrizzle combines astronomical images while removing
distortion and cosmic-rays. The Python syntax for using
MultiDrizzle relies on initializing a MultiDrizzle object,
building up the parameters necessary for combining the images,
then performing the actual processing necessary to generate
the final product.  The process starts with the input of the
image names to initialize the MultiDrizzle object:

>>> import multidrizzle
>>> md = multidrizzle.MultiDrizzle(input,'other parameters')
>>> md.build()
>>> md.run()

MultiDrizzle defines default values for all inputs, and only
those values which need to be over-ridden should be entered.

Further help can be obtained interactively using:
>>> md.help()

:author: Warren Hack, Christopher Hanley, Ivo Busko, and David Grumm

"""
from __future__ import division # confidence high


__docformat__ = 'restructuredtext'



# Begin Import Modules ---------------------------------------------------
import os, shutil, sys, string
import numpy as np
from pydrizzle import pydrizzle, process_input
from pydrizzle import drutil
import pyfits
from stsci.tools import fileutil
import mdzhandler
import manager
from manager import ImageManager
import mdrizpars
from procstep import ProcSteps, timestamp
# This module is used to replicate IRAF style inputs.
from stsci.tools import parseinput
from stsci.tools.parseinput import parseinput

# End Import Modules -----------------------------------------------------


# Begin Version Information -------------------------------------------
if False :
    __version__ = ''
    __svn_version__ = 'Unable to determine SVN revision'
    __full_svn_info__ = ''
    __setup_datetime__ = None

    try:
        __version__ = __import__('pkg_resources').\
                            get_distribution('multidrizzle').version
    except:
        pass
else :
    __version__ = '3.3.9'


__vdate__ = '02-Apr-2012'
# End Version Information ---------------------------------------------
# Revision based version info
try:
    from multidrizzle.svninfo import (__svn_version__, __full_svn_info__,
                                      __setup_datetime__)
except ImportError:
    pass


class Multidrizzle:
    """
The MultiDrizzle object manages the processing of the images.  All
input parameters, including the input, have default values, so only
those parameters which need to be changed should be specified. The
processing requires the following steps to be performed in order:
  - input all parameters and convert all input images to
    multi-extension FITS (MEF) files

    >>> md = multidrizzle.MultiDrizzle (input, output=None,
                                        editpars=no,**input_dict)

    where editpars turns on/off the GUI parameter editor
          input_dict contains all parameters which have non-default values

  - (optionally) edit all input parameter values with Traits-based GUI

    >>> md.editpars()

  - build parameters necessary for combining the images

    >>> md.build()

  - process the images through the steps which were turned on

    >>> md.run( static = None,          skysub = None,
                driz_separate = None,   median = None,
                blot = None,            driz_cr = None,
                driz_combine = None,    timing = None):

    where each parameter controls whether a processing step gets performed.

A full list of the parameters can be obtained from the MultiDrizzle
help file.


    """
    init_keys = ['mdriztab','runfile','workinplace',
                'context','clean','shiftfile','staticfile',
                'static_sig','coeffs']

    driz_keys = ['refimage','group','ra','dec','build']

    instr_keys = ['gain','gnkeyword','rdnoise','rnkeyword',
                    'exptime', 'expkeyword','crbit']
    sky_keys = ['skywidth','skystat', 'skylower','skyupper',
                    'skyclip','skylsigma','skyusigma','skyuser']

    median_keys = ['median_newmasks','combine_type','combine_nsigma',
                    'combine_nlow', 'combine_nhigh','combine_lthresh',
                    'combine_hthresh','combine_grow','combine_maskpt','nsigma1','nsigma2' ]

    drizcr_keys = ['driz_cr_snr','driz_cr_scale', 'driz_cr_corr',
                   'driz_cr_grow','driz_cr_ctegrow']

    blot_keys = ['blot_interp','blot_sinscl']

    def __init__(self,
                 input      = '*flt.fits',
                 output     = None,
                 editpars   = False,
                 shiftfile  = None,
                 updatewcs  = True,
                 **input_dict):

        timestamp()
        print 'Running MultiDrizzle ',__version__

        # Print version information for all external python modules used
        self.versions = versioninfo()

        self.shiftfile = shiftfile
        self.updatewcs = updatewcs
        if 'proc_unit' in input_dict:
            self.proc_unit = input_dict['proc_unit']
        else:
            self.proc_unit = "native"


        # We need to parse the input to get the list of filenames
        # that are to be processed by Multidrizzle.
        #
        # VARIABLE DEFINITIONS
        # self.output: the 'user' specified output string used to name
        #             the output file created by Multidrizzle.
        #
        # self.files: A python list containing the input file names to be
        #             processed by Multidrizzle.
        #
        # self.ivmlist: A python list containing the input filenames representing
        #               input IVM files
        #
        # self.numInputs: Integer value representing the number of individual science
        #                 data files that will need to drizzled by Multidrizzle
        #
        # self.numASNfiles: The number of association files that Multidrizzle has
        #                   received as input
        #
        # self.parseSTISflag: Boolean value used to indicate if a STIS association
        #                     file was given as input and subsequently split into
        #                     multiple files.
        #
        # self.parseWFPC2flag: Boolean value used to indicate if WFPC2 data was given
        #                      as GEIS format files and convereted to multi extension FITS.
        #
        # self.translatedNames: Dictionary mapping input filenames to translated filenames
        #
        # self.translatedNameOrder: List copying order of original inputs
        #
        # self.excludedFileList: List containing names of input files excluded from
        #                        Multidrizzle processing
        #
        #
        #self.output,self.files,self.ivmlist,self.numInputs,\
        #self.numASNfiles,self.parseSTISflag,self.parseWFPC2flag, \
        #self.translatedNames, self.translatedNameOrder, self.excludedFileList \
        #    = self._parseInput(input,output)

        # Initialize attribute for zero exposure time asn table.  Will only
        # be created if the input association file containings members whose
        # EXPTIME value is zero
        #self.zeroExptimeAsnTable = None

        # We need to make certain that we have not thrown out all of the data
        # because of the zero exposure time problem.
        #
        # numInputs <= 0: We have no input.
        self.errorstate = False

        # Remember the original user 'input' value
        self.input = input
        asndict, ivmfiles, output = process_input.process_input(input, output=output, updatewcs=self.updatewcs, shiftfile=shiftfile)
        self.asndict = asndict
        self.ivmlist = ivmfiles
        self.output = output
        if not self.asndict:
            self.errorstate = True
            return
        # Check status of file processing.  If all files have been
        # Report the names of the input files that have just been parsed.
        self.files = [fileutil.buildRootname(f) for f in self.asndict['order']]

        self.printInputFiles()

        # Check input files.  This is the private method used to call the
        # MAKEWCS application.  MAKEWCS is used to recompute and update the
        # WCS for input images, which updatewcs makes optional

        #self._checkInputFiles(self.files, updatewcs)

        # Initialize the master parameter dictionary.
        # This needs to be done after any input file conversion
        # since it needs to be able to open the file to read the
        # MDRIZTAB keyword, if this parameter is set to TRUE.
        self.pars = mdrizpars.MDrizPars(self.input, self.output,
                            dict=input_dict,files=self.files)
        
        # Initialize attributes needed for each processing step
        # These get populated by the 'build' method.
        self.steps = None
        self.skypars = {}
        self.medianpars = {}
        self.drizcrpars = {}
        self.blotpars = {}
        self.driz_sep_pars = {}
        self.driz_final_pars = {}
        self.instrpars = {}

        self.image_manager = None

        # Convenience for user: if they specify 'editpars' on
        # command line as parameter, then automatically run GUI
        # editor for them.
        self.traits_edited = False
        if editpars:
            self.editpars()

    def editpars(self):
        """ Run Python GUI parameter editor to set parameter values. """
        self.pars.edit_traits()
        self.traits_edited = True


    def build(self):
        """ Parses parameter list into dictionaries for use by
            each processing step, builds the PyDrizzle input
            association table, builds the PyDrizzle object and
            uses that to build the InputImage instances needed
            for MultiDrizzle processing.
        """

        if self.errorstate == True:
            # If we are in this conditional, the Multidrizzle constructor
            # exited with a return without actually completing it's normal
            # processing.  This exit from Multidrizzle is to allow for
            # the stopping of execution withour raising an acception.  This
            # keeps the HST pipeline from crashing because of a raised
            # reception.  This state likely occured because all of the input
            # images the user provided to Multidrizzle were excluded from
            # processing because of problems with the data (like a 0 EXPTIME
            # value).
            #
            # Just return and end exection
            return

        #Update master_pars dictionary, and switches dictionary
        # based on last set values.
        if self.traits_edited:
            self.pars.updateMasterPars()

        # Use the values in the master parlist to set the values
        # for the attributes necessary for initializing this class
        for kw in self.init_keys:
            self.__dict__[kw] = self.pars.master_pars[kw]
        # runfile could have been converted to an integer by the above,
        # so we have to make sure it's a string
        self.runfile = str(self.runfile)


        # Create object that controls step execution and mark
        # initialization step.
        self.steps = self.pars.steps
        self.steps.doStep(ProcSteps.doInitialize)

        self.skypars    = self.pars.getParList(self.sky_keys)
        self.medianpars = self.pars.getParList(self.median_keys,prefix='combine_')
        self.drizcrpars = self.pars.getParList(self.drizcr_keys)
        self.blotpars   = self.pars.getParList(self.blot_keys, prefix='blot_')

        # Finalize building PyDrizzle and instrument parameters.
        # If not defined by the user, use defaults.
        self.driz_sep_pars = self.pars.getDrizPars(prefix='driz_sep',keylist=self.driz_keys)
        self.driz_final_pars = self.pars.getDrizPars(prefix='driz_final',keylist=self.driz_keys)
        self.instrpars = self.pars.getParList(self.instr_keys)

        # Finish massaging median pars parameters to account for
        # special processing of input values
        self.setMedianPars()


        # SINGELTON TEST: Verify that if only one file is provided for
        # processing that the median, blot, and driz_cr steps are
        # then turned off.  If they are not turned off, the program
        # will fail because you cannot create a median image with
        # only 1 input. ;-)
        if (len(self.files) == 1):
            if self.pars.switches['median'] == True:
                errorstr =  "####################################\n"
                errorstr += "#                                  #\n"
                errorstr += "# WARNING:                         #\n"
                errorstr += "#  Step 4: CREATE MEDIAN IMAGE has #\n"
                errorstr += "#  been turned off because only    #\n"
                errorstr += "#  one file was provided as input. #\n"
                errorstr += "#                                  #\n"
                errorstr += "####################################\n\n"
                print errorstr
                self.pars.switches['median'] = False
            if self.pars.switches['blot'] == True:
                errorstr =  "############################################\n"
                errorstr += "#                                          #\n"
                errorstr += "# WARNING:                                 #\n"
                errorstr += "#  Step 5: BLOT BACK THE  MEDIAN IMAGE has #\n"
                errorstr += "#  been turned off because only one file   #\n"
                errorstr += "#  was provided as input.                  #\n"
                errorstr += "#                                          #\n"
                errorstr += "############################################\n\n"
                print errorstr
                self.pars.switches['blot'] = False
            if self.pars.switches['driz_cr'] == True:
                errorstr =  "#######################################################\n"
                errorstr += "#                                                     #\n"
                errorstr += "# WARNING:                                            #\n"
                errorstr += "#  Step 6: REMOVE COSMIC RAYS WITH DERIV, DRIZ_CR has #\n"
                errorstr += "#  been turned off because only one file was provided #\n"
                errorstr += "#  as input.                                          #\n"
                errorstr += "#                                                     #\n"
                errorstr += "#######################################################\n\n"
                print errorstr
                self.pars.switches['driz_cr'] = False



        # Extract the 'driz_final_wht_type' parameter
        wht_type = self.pars.master_pars['driz_final_wht_type']
        # Verify that ERR extension exists if final_wht_type = ERR
        if ( wht_type != None and wht_type.upper() == 'ERR'):
            for file in self.files:
                if not fileutil.findFile(file+"[err,1]"):
                    raise ValueError,"! final_wht_type = ERR, no ERR array found for %s"%(file)
        # Check static file. Can only be done after reading MDRIZTAB.
        if (self.staticfile != None):
            self._checkStaticFile(self.staticfile)


        # Create copies of input files for processing
        if not self.workinplace:
            self._createInputCopies(self.files)
        else:
            print "\n\n********************"
            print "WARNING:  Sky will be subtracted from sci extensions"
            print "WARNING:  Units of sci extensions will be electrons"
            print "WARNING:  Value of MDRIZSKY is in units of input data sci extensions."
            print "********************\n\n"

        # Extract bits value and final units from master dictionary for use in setupAssociation
        self.driz_sep_bits = self.pars.master_pars['driz_sep_bits']
        self.final_bits = self.pars.master_pars['driz_final_bits']
        self.final_units = self.pars.master_pars['driz_final_units']

        # Build association object
        #association = self._setupAssociation()
        # Run PyDrizzle; make sure there are no intermediate products
        # laying around...
        #self.asndict.update(shiftfile=self.shiftfile)
        #asnname = fileutil.buildNewRootname(self.asndict['output'], extn='_asn.fits')
        #print 'asnname', asnname        
        assoc = pydrizzle._PyDrizzle(self.asndict, output=self.output,
                                    idckey=self.coeffs,
                                    section=self.driz_sep_pars['group'],
                                    bits_single=self.driz_sep_bits,
                                    bits_final=self.final_bits,
                                    )

        # Use PyDrizzle to clean up any previously produced products...
        if self.clean:
            assoc.clean()

        print 'Initial parameters: '
        assoc.printPars(format=1)

        # Add any specifed IVM filenames to the association object
        for plist in assoc.parlist:
            fname,extname = fileutil.parseFilename(plist['data'])
            ivmname = None
            if len(self.ivmlist) > 0:
                for index in xrange(len(self.ivmlist)):
                    if self.files[index] == fname:
                        ivmname = self.ivmlist[index]
            plist['ivmname'] = ivmname

        # Build the manager object.
        #self.image_manager = ImageManager(association, self.context, self.instrpars, self.workinplace, \
        #self.staticfile, self.updatewcs)
        self.image_manager = ImageManager(assoc, self.context, self.instrpars, self.workinplace, self.staticfile, self.proc_unit)

        # Done with initialization.
        self.steps.markStepDone(ProcSteps.doInitialize)


    def printInputFiles(self):
        """

        METHOD  : printInputFiles
        PURPOSE : Print out the names of the file that Multidrizzle has identified
                  for processing based upon the given input string.
        INPUT   : String representing the user provided input string
        OUTPUT  : none

        """

        print "Input string provided to Multidrizzle: ", str(self.input)
        print "The following files have been identified by the given input string:"
        for filename in self.files:
            print "          ",str(filename)
        print "Output rootname: ", str(self.output)


    def setMedianPars(self):
        """ Sets the special median parameters which need to
            be parsed out of the original input parameters. """

        (_nsigma1, _nsigma2) = _splitNsigma(self.medianpars['nsigma'])
        self.medianpars['nsigma1']  = _nsigma1
        self.medianpars['nsigma2']  = _nsigma2
        self.medianpars['newmasks'] = self.medianpars['median_newmasks']


    def _createInputCopies(self,files):
        """
        Creates copies of all input images.

        If a previous execution of multidrizzle has failed and _OrIg
        files already exist, before removing the _OrIg files, we will
        copy the 'sci' extensions out of those files _OrIg files and
        use them to overwrite what is currently in the existing
        input files.  This protects us against crashes in the HST
        pipeline where Multidrizzle is restarted after the sky
        has already been subtracted from the input files.
        """

        for _img in files:
            # Only make copies of files that exist
            if os.path.exists(_img):
                # Create filename for copy
                _copy = manager.modifyRootname(_img)
                # Make sure we remove any previous copies first,
                # after we copy 'sci' extension into the
                # possibly corrupted input file.  This
                # ensures that Multidrizzle restarts will
                # always have pristine input to use.
                if os.path.exists(_copy):
                    fimage = fileutil.openImage(_img,mode='update')
                    fcopy = fileutil.openImage(_copy)
                    index = 0
                    for extn in fcopy:
                        if extn.name.upper() == 'SCI':
                            fimage[index].data = fcopy[index].data
                        index += 1
                    fimage.close()
                    fcopy.close()
                    os.remove(_copy)

                # Copy file into new filename
                shutil.copyfile(_img,_copy)

    #def _checkInputFiles(self, files, updatewcs):

        """ Checks input files before they are required later. """

        """ Checks that MAKEWCS is run on any ACS image in 'files' list. """
        """
        for p in files:

            if fileutil.getKeyword(p,'idctab') != None:
                if fileutil.getKeyword(p,'PA_V3') != None:
                    # Update the CD matrix using the new IDCTAB
                    # Not perfect, but it removes majority of errors...
                    if updatewcs == True:
                        makewcs.run(input=p)   # optionally update wcs

                elif (os.path.exists(p[0:p.find('_')]+'_spt.fits')):
                    msgstr =  "\n########################################\n"
                    msgstr += "#                                      #\n"
                    msgstr += "# WARNING:                             #\n"
                    msgstr += "#   The keyword PA_V3 was not found in #\n"
                    msgstr += "#   the primary header of file:        #\n"
                    msgstr += "          "+str(p)+"\n"
                    msgstr += "#   Attempting to extract value from   #\n"
                    msgstr += "#   corresponding SPT file and append  #\n"
                    msgstr += "#   it to the primary header file of   #\n"
                    msgstr += "#   the input image.                   #\n"
                    msgstr += "#                                      #\n"
                    msgstr += "########################################\n"
                    print msgstr

                    try:
                        # Build the name of the SPT we are trying to find
                        sptfilename = p[0:p.find('_')]+'_spt.fits'
                        # Extract the PA_V3 value from the primary header
                        # of the SPT file
                        pa_v3 = fileutil.getKeyword(sptfilename,'PA_V3')
                        # If the PA_V3 value is None, raise an exception
                        if (pa_v3 == None):
                            raise ValueError
                        # Open the file we are tyring to process in update
                        # mode.
                        img = pyfits.open(p,mode='update')
                        # Add the PA_V3 value to the input file
                        img[0].header.update("PA_V3",float(pa_v3))
                        # Close the input file.
                        img.close()
                        # Optionally run makewcs
                        if updatewcs == True: makewcs.run(input=p)
                    except:
                        # We have failed to add the PA_V3 value to the
                        # input file.  Raise an exception and give up.
                        self.__pav3errmsg(p)
                        raise ValueError, "Multidrizzle exiting..."
                else:
                    self.__pav3errmsg(p)
                    raise ValueError, "Multidrizzle exiting..."
        """
    """
    def __pav3errmsg(self,filename):
        msgstr =  "\n*******************************************\n"
        msgstr += "*                                         *\n"
        msgstr += "* Primary header keyword PA_V3 not found! *\n"
        msgstr += "* World Coordinate keywords cannot be     *\n"
        msgstr += "* recomputed without a valid PA_V3 value. *\n"
        msgstr += "* Please insure that PA_V3 is populated   *\n"
        msgstr += "* in the primary header of                *\n"
        msgstr += "      %s \n"%(filename)
        msgstr += "* This keyword is generated by versions   *\n"
        msgstr += "* of OPUS 15.7 or later. If the data were *\n"
        msgstr += "* obtained from an earlier version of     *\n"
        msgstr += "* OPUS, please re-retrieve the data from  *\n"
        msgstr += "* the archive after OPUS 15.7 has been    *\n"
        msgstr += "* installed, or manually add the keyword  *\n"
        msgstr += "* with the proper value to the header (it *\n"
        msgstr += "* may be found in the SPT file header)    *\n"
        msgstr += "*                                         *\n"
        msgstr += "*******************************************\n"

        print msgstr

    """
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

            # Verify that the number of sci extensions in the input is equal
            # to the number of MASK extensions in the static file.

            try:
                # Count the number of sci extensions
                sciimg = pyfits.open(self.files[0])
                scicount = 0
                for extension in sciimg:
                    if 'extname' in extension.header:
                        if (extension.header['extname'].upper() == 'SCI'):
                            scicount += 1
                # Count the number of mask extensions
                maskimg = pyfits.open(static_file)
                maskcount = 0
                for extension in maskimg:
                    if 'extname' in extension.header:
                        if (extension.header['extname'].upper() == 'MASK'):
                            maskcount += 1

                if (scicount != maskcount):
                    raise ValueError
            except ValueError:
                # If the sci and mask extension counts don't mask raise a value error
                errorstr =  "\n############################################\n"
                errorstr += "#                                          #\n"
                errorstr += "#  ERROR:                                  #\n"
                errorstr += "#      The user supplied static mask file  #\n"
                errorstr += "          " + str(static_file) + "\n"
                errorstr += "#      does not contain enough MASK        #\n"
                errorstr += "#      extensions.  There needs to be 1    #\n"
                errorstr += "#      MASK extension for every SCI        #\n"
                errorstr += "#      extension.                          #\n"
                errorstr += "#                                          #\n"
                errorstr += "############################################\n"
                raise ValueError, errorstr
            except:
                errorstr =  "\n############################################\n"
                errorstr += "#                                          #\n"
                errorstr += "#  Error:                                  #\n"
                errorstr += "#      Unable to verify the user supplied  #\n"
                errorstr += "#      static mask file is in the correct  #\n"
                errorstr += "#      format.  Test could not complete.   #\n"
                errorstr += "#                                          #\n"
                errorstr += "############################################\n"
                raise IOError, errorstr



    def _preMedian(self, skysub):
        """ Perform the steps that take place before median computation:
            build static mask, subtract sky, drizzle into separate images.
        """
        # Build static mask

        if self.steps.doStep(ProcSteps.doBuildStatic):
            self.image_manager.createStatic(static_sig=self.static_sig)
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
            static          = None,
            skysub          = None,
            driz_separate   = None,
            median          = None,
            blot            = None,
            driz_cr         = None,
            driz_combine    = None,
            timing          = None):

        if self.errorstate == True:
            # If we are in this conditional, the Multidrizzle constructor
            # exited with a return without actually completing it's normal
            # processing.  This exit from Multidrizzle is to allow for
            # the stopping of execution without raising an acception.  This
            # keeps the HST pipeline from crashing because of a raised
            # exception.  This state likely occured because all of the input
            # images the user provided to Multidrizzle were excluded from
            # processing because of problems with the data (like a 0 EXPTIME
            # value).
            #
            # Just return and end exection
            return


        # Update object that controls step execution. Use either user
        # interface switches, or MDRIZTAB switches.
        self.pars.setProcSteps(static=static,
                                skysub=skysub,
                                driz_separate = driz_separate,
                                median = median,
                                blot = blot,
                                driz_cr = driz_cr,
                                driz_combine = driz_combine,
                                timing = timing)

        # Start by applying input parameters to redefine
        # the output frame as necessary
        self.image_manager._setOutputFrame(self.driz_sep_pars)

        self._preMedian(skysub)
        if self.steps.doStep(ProcSteps.doMedian):
            self.image_manager.createMedian(self.medianpars)
            self.steps.markStepDone(ProcSteps.doMedian)

        self._postMedian(self.blotpars, self.drizcrpars, self.skypars)

        if self.steps.doStep(ProcSteps.doFinalDriz):
            self.driz_final_pars['runfile'] = self.runfile
            self.image_manager.doFinalDriz(self.driz_final_pars)
            self.runfile = self.driz_final_pars['runfile']
            self.image_manager.updateMdrizVerHistory(self.driz_final_pars['build'],self.versions)
            self.steps.markStepDone(ProcSteps.doFinalDriz)

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

        if self.pars.switches['timing']:
            self.steps.reportTimes()

    def help(self):
        """

        Purpose
        =======
        Help function on Multidrizzle Class that
        prints __doc__ string.

        """

        print 'MultiDrizzle Version ',__version__
        print self.__doc__


def _splitNsigma(s):
    """

    Purpose
    =======
    This is a help function that is used to split up the "combine_nsigma"
    string. If a second value is specified, then this will be used later
    in "minmed" where a second-iteration rejection is done. Typically
    the second value should be around 3 sigma, while the first
    can be much higher.

    """
    _sig = s.split(" ")
    _nsigma1 = float(_sig[0])
    _nsigma2 = float(_sig[0])
    if len(_sig) > 1:
        _nsigma2 = float(_sig[1])
    return (_nsigma1, _nsigma2)


def help():
    """

    Purpose
    =======
    Function prints help information for MultiDrizzle.

    """


    help_str =  "MultiDrizzle combines astronomical images while removing       \n"
    help_str += "distortion and cosmic-rays. The Python syntax for using        \n"
    help_str += "MultiDrizzle relies on initializing a MultiDrizzle object,     \n"
    help_str += "building up the parameters necessary for combining the images, \n"
    help_str += "then performing the actual processing necessary to generate    \n"
    help_str += "the final product.  The process starts with the input of the   \n"
    help_str += "image names to initialize the MultiDrizzle object:             \n"
    help_str += "                                                               \n"
    help_str += ">>> import multidrizzle                                        \n"
    help_str += ">>> md = multidrizzle.MultiDrizzle(input,args,...)             \n"
    help_str += ">>> md.build()                                                 \n"
    help_str += ">>> md.run()                                                   \n"
    help_str += "                                                               \n"
    help_str += "MultiDrizzle defines default values for all inputs, and only   \n"
    help_str += "those values which need to be over-ridden should be entered.   \n"
    help_str += "                                                               \n"
    print 'MultiDrizzle Version ',__version__
    print help_str


def versioninfo():
    """

    Purpose
    =======
    Function prints version information for packages used by Multidrizzle

    """

    # Initialize version dictionary
    version_dict = {}

    # Set up version ID's for printing to the log file
    mdrizzle_key = " MultiDrizzle "
    array_key = " NUMPY Version  "
    pydrizzle_key = " PyDrizzle Version "
    pyfits_key =  " PyFITS Version    "
    python_key = " Python Version: "

    version_dict[mdrizzle_key] = __version__
    version_dict[array_key]= np.__version__
    version_dict[pydrizzle_key]= pydrizzle.__version__
    version_dict[pyfits_key] = pyfits.__version__

    _sys_version_list = sys.version.split("\n")
    _sys_version = " # "
    for _sys_str in _sys_version_list:
        _sys_version += _sys_str+"\n # "
    version_dict[python_key] = '\n'+_sys_version

    # Print out version information for libraries used by MultiDrizzle
    print "\n"
    print " Version Information for MultiDrizzle "+version_dict[mdrizzle_key]
    print "-------------------------------------------- "
    print array_key + version_dict[array_key]
    print pyfits_key + version_dict[pyfits_key]
    print pydrizzle_key + version_dict[pydrizzle_key]
    print python_key + version_dict[python_key]
    print "\n"

    return version_dict
