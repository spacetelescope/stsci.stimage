#import string
import os, shutil, sys

#import numarray
#import numarray.ieeespecial
#from numarray.ieeespecial import *

import pydrizzle
from pydrizzle import drutil, fileutil, buildasn, updateasn
import pyfits
import readgeis

import mdzhandler
import manager
from manager import ImageManager

import mdrizpars
from procstep import ProcSteps, timestamp

import geissupport
from geissupport import *

import makewcs

# This module is used to replicate IRAF style inputs.
import parseinput
from parseinput import parseinput
from parseinput import countinputs

# This module is used to parse IVM file input
import parseIVM
from parseIVM import parseIVM

# This module is used to parse STIS association files
import stis_assoc_support
from stis_assoc_support import parseSTIS
from stis_assoc_support import parseSTISIVM

__version__ = '2.4.1 (17 January 2005)'

__help_str = """
MultiDrizzle combines astronomical images while removing
distortion and cosmic-rays. The Python syntax for using
MultiDrizzle relies on initializing a MultiDrizzle object,
building up the parameters necessary for combining the images,
then performing the actual processing necessary to generate
the final product.  The process starts with the input of the
image names to initialize the MultiDrizzle object:

>>> import multidrizzle
>>> md = multidrizzle.MultiDrizzle(input)
>>> md.editpars()  
>>> md.build()
>>> md.run()

The 'editpars()' method starts the Traits GUI for editing all the
input parameters, and is a step that can be skipped. MultiDrizzle
defines default values for all inputs, and only those values which
need to be over-ridden should be entered.

Further help can be obtained interactively using:
>>> md.help()

"""
def help():
    """ Prints help information for MultiDrizzle."""
    print 'MultiDrizzle Version ',__version__
    print __help_str


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
    print "\n"
    print " Version Information for"+_mdrizzle_version
    print "-------------------------------------------- "
    try:
        import pyraf
        _pyraf_version     = " PyRAF Version     "+pyraf.__version__
        print _pyraf_version
    except:
        print "PyRAF cannot be found!"
    print _numarray_version
    print _pyfits_version
    print _pydrizzle_version
    print _python_version
    print "\n"

def _splitNsigma(s):

    # Split up the "combine_nsigma" string. If a second value is
    # specified, then this will be used later down in the "minmed"
    # section where a second-iteration rejection is done. Typically
    # the second value should be around 3 sigma, while the first
    # can be much higher.

    _sig = s.split(" ")
    _nsigma1 = float(_sig[0])
    _nsigma2 = float(_sig[0])
    if len(_sig) > 1:
        _nsigma2 = float(_sig[1])
    return (_nsigma1, _nsigma2)

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

    driz_keys = ['refimage','group','bits','ra','dec','build']
    
    instr_keys = ['gain','gnkeyword','rdnoise','rnkeyword',
                    'exptime', 'expkeyword','crbit']
    sky_keys = ['skywidth','skystat', 'skylower','skyupper',
                    'skyclip','skylsigma','skyusigma','skyuser']   
                        
    median_keys = ['median_newmasks','combine_type','combine_nsigma',
                    'combine_nlow', 'combine_nhigh','combine_lthresh',
                    'combine_hthresh','combine_grow','nsigma1','nsigma2' ]

    drizcr_keys = ['driz_cr_snr','driz_cr_scale', 'driz_cr_corr']
    
    blot_keys = ['blot_interp','blot_sinscl']
    
    def __init__(self,
                 input      = '*flt.fits',
                 output     = None,
                 editpars   = False,
                 **input_dict):

        timestamp()
        print 'Running MultiDrizzle ',__version__

        # Print version information for all external python modules used
        versioninfo()        

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
        self.output,self.files,self.ivmlist,self.numInputs,\
        self.numASNfiles,self.parseSTISflag,self.parseWFPC2flag, \
        self.translatedNames, self.translatedNameOrder, self.excludedFileList \
            = self._parseInput(input,output)
            
        # We need to make certain that we have not thrown out all of the data
        # because of the zero exposure time problem.
        #                
        # numInputs <= 0: We have no input.
        self.errorstate = False
        try:
            if len(self.files) <= 0 or self.numInputs <= 0:
                errorstr =  "#############################################\n"
                errorstr += "#                                           #\n"
                errorstr += "# ERROR:                                    #\n"
                errorstr += "#  No valid input available for processing! #\n"
                errorstr += "#                                           #\n"
                errorstr += "#  The following files were excluded from   #\n"
                errorstr += "#  Multidrizzle processing because their    #\n"
                errorstr += "#  header keyword EXPTIME values were 0.0:  #\n"
                for name in self.excludedFileList:
                    errorstr += "         "+ str(name) + "\n" 
                errorstr += "#                                           #\n"
                errorstr += "#############################################\n\n"
                print errorstr
                raise ValueError
        except ValueError:
            # If all of the input has been rejected, build an empty DRZ product
            # that the pipeline can ingest.
            self._buildEmptyDRZ()
            # End Multidrizzle processing cleanly
            self.errorstate = True
            return
            

        # Remember the original user 'input' value
        self.input = input

        # Check status of file processing.  If all files have been         
        # Report the names of the input files that have just been parsed.
        self.printInputFiles()

        # Check input files.  This is the private method used to call the 
        # MAKEWCS application.  MAKEWCS is used to recompute and update the
        # WCS for input images
        self._checkInputFiles(self.files)

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
        if editpars:
            self.editpars()
            
    def editpars(self):
        """ Run Python GUI parameter editor to set parameter values. """
        self.pars.edit_traits()


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
            # keeps the HST pipeline from crashing because of a rasied
            # reception.  This state likely occured because all of the input
            # images the user provided to Multidrizzle were excluded from
            # processing because of problems with the data (like a 0 EXPTIME
            # value).
            #
            # Just return and end exection
            return 
        
        #Update master_pars dictionary, and switches dictionary
        # based on last set values.        
        self.pars.updateMasterPars()
        
        # Use the values in the master parlist to set the values
        # for the attributes necessary for initializing this class
        for kw in self.init_keys:
            self.__dict__[kw] = self.pars.master_pars[kw]
        
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
                    


        # Verify that ERR extension exists if final_wht_type = ERR
        if ( self.pars.master_pars['driz_final_wht_type'] != None and
                self.pars.master_pars['driz_final_wht_type'].upper() == 'ERR'):
            for file in self.files:
                if not fileutil.findFile(file+"[err,1]"):
                    raise ValueError,"! final_wht_type = ERR, no ERR array found for %s"%(file)

        
        # Create copies of input files for processing
        if not self.workinplace:
            self._createInputCopies(self.files)
        else:
            print "\n\n********************"
            print "WARNING:  Sky will be subtracted from sci extensions"
            print "WARNING:  Units of sci extensions will be electrons"
            print "WARNING:  Value of MDRIZSKY is in units of input data sci extensions."
            print "********************\n\n"


        # Initialize the dqpar file based upon instrument type
        #_instrument = fileutil.getKeyword(self.files[0]+'[0]','INSTRUME')
        #self._initdqpars(_instrument,self.driz_sep_pars['bits'])
        
        # Extract bits value from master dictionary for use in setupAssociation
        self.bits = self.driz_sep_pars['bits']

        # Build association object
        association = self._setupAssociation()

        # Build the manager object.

        self.image_manager = ImageManager(association, self.context, self.instrpars, self.workinplace)

        # Do unit conversion of the 'sci' data if necessary
        self.image_manager.doUnitConversions()

        # Check static file. Can only be done after reading MDRIZTAB.
        self._checkStaticFile(self.staticfile)

        # Done with initialization.
        self.steps.markStepDone(ProcSteps.doInitialize)

    def _buildEmptyDRZ(self):
        """
        
        METHOD  : _buildEmptyDRZ
        PURPOSE : Create an empty DRZ file in a valid FITS format so that the HST
                  pipeline can handle the Multidrizzle zero expossure time exception
                  where all data has been excluded from processing.
        INPUT   : None
        OUTPUT  : DRZ file on disk
         
        """

        # Open the first image of the excludedFileList to use as a template to build
        # the DRZ file.
        try :
            img = pyfits.open(self.excludedFileList[0])        
        except:
            errstr  = "#################################\n"
            errstr += "#                               #\n"
            errstr += "# ERROR: Unable to open file:   #\n"
            errstr += "      " + str(inputfile) + "\n" 
            errstr += "#                               #\n"
            errstr += "#################################\n"
            raise RuntimeError,errstr

        # Create the fitsobject
        fitsobj = pyfits.HDUList()
        # Copy the primary header
        hdu = img[0].copy()
        fitsobj.append(hdu)
        
        # Modify the 'NEXTEND' keyword of the primary header to 3 for the 
        #'sci, wht, and ctx' extensions of the newly created file.
        fitsobj[0].header['NEXTEND'] = 3

        # Create the 'SCI' extension
        hdu = pyfits.ImageHDU(header=img['sci',1].header.copy(),data=None)
        hdu.header['EXTNAME'] = 'SCI'    
        fitsobj.append(hdu)
        
        # Create the 'WHT' extension
        hdu = pyfits.ImageHDU(header=img['sci',1].header.copy(),data=None)
        hdu.header['EXTNAME'] = 'WHT'    
        fitsobj.append(hdu)
        
        # Create the 'CTX' extension
        hdu = pyfits.ImageHDU(header=img['sci',1].header.copy(),data=None)
        hdu.header['EXTNAME'] = 'CTX'    
        fitsobj.append(hdu)        
        
        # Add HISTORY comments explaining the creation of this file.
        fitsobj[0].header.add_history("** Multidrizzle has created this empty DRZ **")
        fitsobj[0].header.add_history("** product because all input images were   **")
        fitsobj[0].header.add_history("** excluded from processing because their  **")
        fitsobj[0].header.add_history("** header EXPTIME values were 0.0.  If you **")
        fitsobj[0].header.add_history("** still wish to use this data make the    **")
        fitsobj[0].header.add_history("** EXPTIME values in the header non-zero.  **")
        
        # Change the filename in the primary header to reflect the name of the output
        # filename.
        fitsobj[0].header['FILENAME'] = "'"+str(self.output)+"_drz.fits"+"'"
                
        errstr =  "#############################################\n"
        errstr += "#                                           #\n"
        errstr += "# ERROR:                                    #\n"
        errstr += "#  Multidrizzle has created this empty DRZ  #\n"
        errstr += "#  product because all input images were    #\n"
        errstr += "#  excluded from processing because their   #\n"
        errstr += "#  header EXPTIME values were 0.0.  If you  #\n"
        errstr += "#  still wish to use this data make the     #\n"
        errstr += "#  EXPTIME values in the header non-zero.   #\n"
        errstr += "#                                           #\n"
        errstr += "#############################################\n\n"
        print errstr
        
        # If the file is already on disk delete it and replace it with the
        # new file
        dirfiles = os.listdir(os.curdir)
        if (dirfiles.count(self.output+"_drz.fits") > 0):
            os.remove(self.output+"_drz.fits")
            print "       Replacing "+self.output+"_drz.fits"+"..."
            
        # Write out the empty DRZ file
        fitsobj.writeto(self.output+"_drz.fits")
        return

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
        
    
    def _parseInput(self,input,output):

        """ 
        
        METHOD : _parseInput (private method)
        PURPOSE: Interprets input from user to build list of files to process. 
        INPUT  : input - String representing user input string
                 output - String representing user output string
        OUTPUT : 
        """

        # Define the return variables        
        newoutput = output # String containing output file name.  This value could
                           # be modified here if it is not provided by the user and
                           # it is contained in the assocation table.  Alternatively,
                           # if no value is ever provided by the user or an assocation table,
                           # it will be set to a value of 'final'
        
        filelist = []  # Python list containing the name of all files that will need to
                       # to be processed by Multidrizzle.
        
        ivmlist = [] # Python list containing the name of all inverse variance map files (IVM)
                     # that will need to be processed by Multidrizzle

        numInputs = 0 # Number of input files that have been identified for processing by
                      # by Multidrizzle
                      
        numASNfiles = 0 # Number of ASN files that were given to Multidrizzle as input.

        translatedNames = {} # Tranlation dictionary giving the relationship between
                             # newly created file names and the names of files on which
                             # they are originally based.
                             
        translatedNameOrder = [] # Simple list used to track the order in which input files
                                 # were given to Multidrizzle.

        excludedFileList = [] # Simple list used to store the names of files excluded
                              # from multidrizzle processing because the EXPTIME value
                              # in their header was 0.

        # Examine the input to determine the number of input files and the number of 
        # association files (in any) provided by the user.
        numInputs,numASNfiles = countinputs(input)

        # Parse the user input into a python list object and extract the
        # output name from the assocation table if given.
        inputfiles,newoutput = parseinput(input,output)
        
        # Check that input files were actually found on disk.  If not, raise an exception
        # and print an error message.  There are a number of reasons that no data could
        # have been found on disk.
        #   1) The user is in the wrong directory
        #   2) The user is trying to give a partial file name and not using wild-cards
        #   etc...
        if (len(inputfiles) == 0):
            errorstr =  "\n######################################################\n"
            errorstr += "#                                                    #\n"
            errorstr += "# ERROR:                                             #\n"
            errorstr += "#  No valid files were found for an input string of: #\n"
            errorstr += "          " + str(input) + "\n"
            errorstr += "#                                                    #\n"
            errorstr += "#  Two possible reasons this error may have occured  #\n"
            errorstr += "#  are:                                              #\n"
            errorstr += "#    - With the release of Multidrizzle 2.4.0,       #\n"
            errorstr += "#      the processing of filename fragments without  #\n"
            errorstr += "#      the use of wild-cards is no longer allowed.   #\n"
            errorstr += "#      (use '*flt.fits' instead of 'flt.fits')       #\n"
            errorstr += "#                                                    #\n"
            errorstr += "#    - The directory you are running Multidrizzle    #\n"
            errorstr += "#      in does not contain the specified files.      #\n"     
            errorstr += "#                                                    #\n"
            errorstr += "######################################################\n\n"
            raise ValueError,errorstr
        
        # 
        # We now need to determine if IVM files have been included as input.
        # If IVM files have been provided, we need to separate the list into
        # one for the standard input images and one for the IVM files.
        filelist,ivmlist = parseIVM(inputfiles)
        
        # Now we need to determine what type of data we are dealing with.  At this time
        # there are 4  cases that we need to be concerned about:
        # 1) WFPC2 GEIS Input (c0h or hhh file extensions)
        # 2) STIS association files
        # 3) A default case
        #
        # Case 1: WFPC2 GEIS INPUT
        #   GEIS formated files must be converted to multi extension FITS format prior
        #   to Multidrizzle processing.  
        #
        # Case 2: STIS association file input
        #   STIS assocation files (i.e. *_flt.fits files with multiple 'sci' extensions) must
        #   be split into separate multi extension FITS files prior to Multidrizzle processing.
        # 
        # Case 3: Default Case
        #   All input files are already in multi extension FITS format.
        #
        # For all of the above cases, each file EXPTIME keyword will be checked for a 0
        # value.  If the data is corrupt (i.e. EXPTIME  = 0), it will be thrown out of the 
        # Multidrizzle processing list.  This error checking for STIS and WFPC2 will be
        # handled by the instrument specifc parsing fucntions (parseWFPC2 and parseSTIS).
        #
        
        # The list 'newfilelist' represents the final list of files to be processed.  GEIS
        # files will have been converted to multi extension FITS format.  STIS association
        # files will have been split in separate exposures.
        newfilelist = []
        
        # Initialize Boolean values
        parseWFPC2flag = False
        parseSTISflag = False
                
        for inputfile in filelist:                 
            if fileutil.getKeyword(inputfile,'instrume') == 'WFPC2':
                # ParseWFPC2 will return a single file name to add to the list of
                # files to process
                tmpfilename,tmpFlag = parseWFPC2(inputfile)
                
                # Determine if the 
                if (tmpfilename == None):
                    excludedFileList.append(inputfile)
                else:
                    newfilelist.append(tmpfilename)
                
                # Populate translation dictionary and ordered list
                translatedNames[tmpfilename] = inputfile
                translatedNameOrder.append(tmpfilename)
                
                # If the flag is set True once, it needs to always be True.  This is
                # just an indicator that WFPC2 file conversions have occured.
                if tmpFlag == True:
                    parseWFPC2flag == True

            elif fileutil.getKeyword(inputfile,'instrume') == 'STIS':
                # ParseSTIS will be dealing with both single exposure and multiple exposure
                # STIS files.  Because of that, parseSTIS will return a Python list that
                # will extend the current newfilelist
                tmplist,tmpExcludedList,tmpFlag = parseSTIS(inputfile)
                
                if (len(tmpExcludedList) != 0):
                    excludedFileList.extend(tmpExcludedList)
                else:                 
                    newfilelist.extend(tmplist)
                
                # Populate translation dictionary
                for fname in tmplist:
                    translatedNames[fname] = inputfile
                    translatedNameOrder.append(fname)

                # If the flag is set True once, it needs to always be True.  This is 
                # just an indicator that STIS file conversions have occured.
                if tmpFlag == True:
                    parseSTISflag == True
            else:
                # We currently only need to worry about the conversion of WFPC2 and STIS
                # files as special cases so this is the current DEFAULT case.
                if (fileutil.getKeyword(inputfile,'EXPTIME') == 0.0):
                    excludedFileList.append(inputfile)
                else:
                    newfilelist.append(inputfile)    

                # Populate translation dictionary
                translatedNames[inputfile] = inputfile
                translatedNameOrder.append(inputfile)


        # We need to determine if the IVM files need to be convereted from GEIS to
        # FITS format or STIS association to simple FITS format

        #Create a newivmlist object
        newivmlist = []

        if (len(ivmlist) != 0):
            if (len(excludedFileList) == 0):
                errstr =  "#######################################\n"
                errstr += "#                                     #\n"
                errstr += "# WARNING:                            #\n"
                errstr += "#  IVM files will not be used during  #\n"
                errstr += "#  processing due to the exclusion    #\n"
                errstr += "#  by Multidrizzle of input files     #\n"
                errstr += "#  with EXPTIME values of 0.0         #\n"
                errstr += "#                                     #\n"
                errstr += "#  To use IVM files all input data    #\n"
                errstr += "#  must by populated with non-zero    #\n"
                errstr += "#  EXPTIME header keyword values.     #\n"
                errstr += "#                                     #\n"
                errstr += "#######################################\n"
                raise ValueError, errstr
            else:
                # Examine each file in the ivmlist
                for ivmfile in ivmlist:
                    # initialize temporary objects
                    tmpfilename = None
                    tmplist = []

                    if fileutil.getKeyword(ivmfile,'instrume') == 'WFPC2':
                        # ParseWFPC2 will return a single file name to add to the list of
                        # files to process
                        tmpfilename,tmpFlag = parseWFPC2(ivmfile)
                        newivmlist.append(tmpfilename)
                    elif fileutil.getKeyword(ivmfile,'instrume') == 'STIS':
                        # ParseSTIS will be dealing with both single exposure and multiple exposure
                        # STIS files.  Because of that, parseSTIS will return a Python list that
                        # will extend the current newfilelist
                        tmplist = parseSTISIVM(ivmfile) 
                        newivmlist.extend(tmplist)
                    else:
                        newivmlist.append(ivmfile)
                    
        if  ( (len(newivmlist)>0) and (len(newivmlist) != len(newfilelist))):
            errorstr =  "#########################################\n"
            errorstr += "#                                       #\n"
            errorstr += "# ERROR: Number of IVM files does not   #\n"
            errorstr += "# equal the number of input files.      #\n"
            errorstr += "#                                       #\n"
            errorstr += "# When providing IVM files as input,    #\n"
            errorstr += "# there needs to be exactly one IVM     #\n"
            errorstr += "# file for each input file.             #\n"
            errorstr += "#                                       #\n"
            errorstr += "#########################################\n"
            raise ValueError, errorstr

        # Setup default output name if none was provided either by user or in ASN table
        if (newoutput == None) or (len(newoutput) == 0):
            newoutput = 'final'

        return newoutput, newfilelist, newivmlist, numInputs, numASNfiles, \
                parseSTISflag, parseWFPC2flag, translatedNames, \
                translatedNameOrder, excludedFileList
        
    def _printInputPars(self,switches):
        print "\n\n**** Multidrizzle Parameter Input ****"

        print "\n** General Parameters:"
        print "input = ", self.input
        print "output  = ",self.output
        print "mdriztab = ", self.mdriztab
        print "refimage  = ", self.driz_sep_pars['refimage']
        print "runfile = ", self.runfile
        print "workinplace = ", self.workinplace
        print "coeffs = ", self.coeffs
        print "context = ", self.context
        print "clean  = ", self.clean
        print "group = ", self.driz_sep_pars['group']
        print "bits = ", self.driz_sep_pars['bits']
        print "ra = ", self.driz_sep_pars['ra']
        print "dec = ", self.driz_sep_pars['dec']
        print "build = ", self.driz_sep_pars['build']
        print "shiftfile =  ",self.shiftfile
        
        print "\n** Static Mask:"
        print "static = ", switches['static']
        print "staticfile = ",self.staticfile
        print "static_sig = ", self.static_sig

        print "\n** Sky Subtraction:"
        print "skysub = ", switches['skysub']
        self._printDictEntries("",self.skypars)
                
        print "\n** Separate Drizzle Images:"
        print "driz_separate = ", switches['driz_separate']
        self._printDictEntries('driz_sep_',self.driz_sep_pars)
        
        print "\n** Create Median Image:"
        print "median = ", switches['median']
        print "combine_newmasks = ",self.medianpars['newmasks']
        print "combine_nsigma = ",self.medianpars['nsigma1']," ",self.medianpars['nsigma2']
        self._printDictEntries('combine_',self.medianpars)
 
        print "\n** Blot Back the Median Image:"
        print "blot = ", switches['blot']
        self._printDictEntries('blot_',self.blotpars)

        print "\n** Remove Cosmic Rays with DERIV, DRIZ_CR:"
        print "driz_cr = ", switches['driz_cr']
        self._printDictEntries("",self.drizcrpars)

        print "\n** Combined Drizzled Image:"
        print "driz_combine = ", switches['driz_combine']
        self._printDictEntries('final_',self.driz_final_pars)
        
        print "\n** Instrument Parameters:"
        self._printDictEntries("",self.instrpars)
        print "\n"

    def _printDictEntries(self,prefix,dict):
        # Define list of items that is to be handeled as a special printing case.
        itemlist = ['newmasks','ra','dec','build','bits','group','coeffs','nsigma1','nsigma2']
        
        sortedkeys = dict.keys()
        sortedkeys.sort()
        
        for key in sortedkeys:
            if (itemlist.count(key) == 0):
                print prefix+key, " = ", dict[key]
                        

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

    def _checkInputFiles(self, files):

        """ Checks input files before they are required later. """

        """ Checks that MAKEWCS is run on any ACS image in 'files' list. """
        for p in files:

            if fileutil.getKeyword(p,'idctab') != None:
                if fileutil.getKeyword(p,'PA_V3') != None:
                    # Update the CD matrix using the new IDCTAB
                    # Not perfect, but it removes majority of errors...
                    makewcs.run(image=p)
                else:
                    self.__pav3errmsg(p)
                    raise ValueError, "Multidrizzle exiting..."

    def __pav3errmsg(self,filename):
        str =  "*******************************************\n"
        str += "*                                         *\n"
        str += "* Primary header keyword PA_V3 not found! *\n"
        str += "* World Coordinate keywords cannot be     *\n"
        str += "* recomputed without a valid PA_V3 value. *\n"
        str += "* Please insure that PA_V3 is populated   *\n"
        str += "* in the primary header of                *\n" 
        str += "      %s \n"%(filename)
        str += "* This keyword is generated by versions   *\n"
        str += "* of OPUS 15.5 or later. If the data were *\n"
        str += "* obtained from an earlier version of     *\n"
        str += "* OPUS, please re-retrive the data from   *\n"
        str += "* the archive after OPUS 15.5 has been    *\n"
        str += "* installed, or manually add the keyword  *\n"
        str += "* with the proper value to the header (it *\n"
        str += "* may be found in the SPT file header)    *\n"
        str += "*                                         *\n"
        str += "*******************************************\n"
        
        print str
        

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

    def _setupAssociation(self):

        """ 
        
        METHOD NAME : _setupAssocation 
        PURPOSE     : Build the PyDrizzle association object.
        INPUT       : Attributes of the multidrizzle object.  Required attributes are:
                        - self.input
                        - self.files
                        - self.output
                        - self.translatedNames
                        - self.translatedNameOrder
                        - self.ivmlist
                        - self.numASNfiles
                        - self.numInputs
                        - self.parseSTISflag
                        - self.parseWFPC2flag
                        - self.shiftfile
                        - self.excludedFileList
                        - self.bits
        OUTPUT      : A new ASN file used as input to Pydrizzle.  Remember, PyDrizzle is
                      invoked by Multidrizzle and used the assocation table to create the
                      assoc.par object.
        """

        timestamp()
        print ' *** '+'Setting up associated inputs..'+'\n *** '

        # There are a number of cases that we need to deal with when creating an assocation
        # file for Pydrizzle use.
                
        # Case 1: No support for WFPC2 and STIS image combination.  Actually, there is no
        # support for the combination of different HST data types in general.  However,
        # we can only check for the STIS and WFPC2 case. 
        if ((self.parseSTISflag == True) and (self.parseWFPC2flag == True)):
            errorstr =  "####################################\n"
            errorstr += "#                                  #\n"
            errorstr += "# ERROR:                           #\n"
            errorstr += "#  Multidrizzle does not support   #\n"
            errorstr += "#  the combination of different    #\n"
            errorstr += "#  types of instrument data.  The  #\n"
            errorstr += "#  use of WFPC2 and STIS input is  #\n"
            errorstr += "#  not currently supported.        #\n"
            errorstr += "#                                  #\n"
            errorstr += "####################################\n"
            raise ValueError, errorstr
        
        # Case 2: Pass an existing association file to Pydrizzle.  No addition processing needed
        # to be done to build an assocation table.
        elif ((self.parseSTISflag != True) and (self.parseWFPC2flag != True) and \
              (self.numInputs == 1) and (self.numASNfiles == 1)):
            # We have only 1 entry in the file list and it is an association table.
            #
            # Define the assocation table variable.
            driz_asn_file = self.input

            # Update assocation table with shiftfile information if available.
            self._checkAndUpdateAsn(driz_asn_file, self.shiftfile)

        # Case 3: Only one file given as input. len(self.files) == 1.
        elif (len(self.files) == 1):
            # The name of the single file is set as Pydrizzle input variable.
            driz_asn_file = self.files[0]

        # Case 4: Parse flags set while throwing out assocation table shift information.
        # There are either zero or multiple assocation files
        # given as input.  Therfore we can ignore any shift information the assocation
        # files may contain.  We only need to be concerned about shift information 
        # provided from shiftfiles.  This requires a filename translation from the
        # input filename to a name representing a single exposure multi extension FITS
        # format file. 
        elif ( ((self.parseWFPC2flag == True) or (self.parseSTISflag == True )) \
            and ( (self.numASNfiles != 1) or ( (self.numASNfiles == 1) and (self.numInputs > 1)))):
        
            # If a shiftfile exists, populate a shiftfile dictionary
            if self.shiftfile != None:
                # read the shiftfile 
                shiftdict = fileutil.readShiftFile(self.shiftfile)
                
                # Create a new shiftfile using the translated file names
                #
                # Open a new shiftfile
                newshiftfile = 'MDZ_' + str(self.shiftfile)
                fshift = open(newshiftfile,'w')
                
                # Add units, reference, form and frame values to the newly created shiftfile
                fshift.write('# units: ' + shiftdict['units'] + '\n')
                fshift.write('# frame: ' + shiftdict['frame'] + '\n')
                fshift.write('# refimage: ' + shiftdict['refimage'] + '\n')
                fshift.write('# form: ' + shiftdict['form'] + '\n')
                
                for fname in self.translatedNames.keys():
                    shiftRecord = shiftdict[translatedNames[fname]]
                    outputline =  str(fname) + "  " 
                    # Report X shift
                    outputline += str(shiftRecord[0]) + "  " 
                    # Report Y shift
                    outputline += str(shiftRecord[1]) + "  "
                    # Check for existence of rotation value
                    if shiftRecord[2] != None:
                        outputline += str(shiftRecord[2]) + "  "
                        
                        # Check for existence of scaling value.  This will only
                        # exist if a rotaion is provided as well.
                        if shiftRecord[3] !=   None:
                            outputline += str(shiftRecord[3])
                   
                    outputline += '\n'
                    # Write out the current line
                    fshift.write(outputline)
                
                # Close the new shiftfile
                fshift.close()

            else:
                # Case for no shiftfile provided
                shiftdict = None
                newshiftfile = None
                    
            # Create a new assocation file
            driz_asn_file = self._generateAsnTable(self.output, self.files, newshiftfile)
        
        # Case 5: Parse flages set, single assocation file given as only input.  A shift file
        # may have been provided.  Name mangling needs to occur for STIS assocation files
        # and WFPC2 GEIS input.
        #
        # To deal with this case, what we will do is use the exisiting shiftfile (if providied) to
        # update the shift information in the user provided association table.  In this way,
        # we will only be working with the original filenames.  Then, we will take the newly
        # updated association table and convert the entries to use the names of the newly created
        # files that now exist in multi extension FITS format.
        elif ( ((self.parseSTISflag == True) or (self.parseWFPC2flag == True)) and \
             ((self.numASNfiles == 1 ) and (self.numInputs == 1))): 

            # We have only 1 entry in the file list and it is an association table.
            # However, this assocation table is populated 
            #
            # Define a variable for the original assocation table.
            orig_driz_asn_file = self.input

            # Update original assocation table with original shiftfile information if available.
            self._checkAndUpdateAsn(orig_driz_asn_file, self.shiftfile)

            # Open the existing association table
            origassocdict = readAsnTable(orig_driz_asn_file, None, prodonly=False)

            # Create a new association table.
            newassocdict = {}

            # Step through the new file names and build new rows for the assocation table. 
            #
            # Extract 'drz' product output name
            newassocdict['output']=self.output
            
            # Add the order information for new assocation table
            for name in self.translatedNameOrder:
                # Extract the rootname of the file for inclusion in the new association table
                rootname = name[:name.rfind('.fits')]
                newassocdict['order'].append(rootname)
            
            # Initialize Members dictionary.  Copying shift flags from original assoication table.
            newassocdict['members'] = \
                {'abshift':origassocdict['members']['abshift'],'dshift':origassocdict['members']['dshift']}
            
            
            #Extract rootname of original file
            #
            # Build dictionary matching old rootnames with old full file names
            oldrootname ={}
            
            for memname in origassocdict['members'].keys():
                if memname != 'abshift' and memname != 'dshift':
                    oldfilename = fileutil.buildRootname(memname)
                    oldrootname[oldfilename]=memname
                
            # Add new names and shift information
            for fname in newassocdict['order']:
                # Extract original memname
                origmemname = oldrootname[self.translatedNames[fname+'.fits']]
                
                # Copy assocation table row from the original association table and
                # append it to the appropriate row for the translated memname.
                newassocdict['members'][fname] =  origassocdict['members'][origmemname]

            # Write out the mangled association dictionary to new association table.
            driz_asn_file = buildasn.writeAsnDict(newassocdict,output='MDZ_'+self.output)
            
            
        # Default Case: List of files.  This case includes the situation where a user has
        # provided multiple ASN tables as inputs.  We do not support shift information
        # from multiple ASN tables.  Any shift information provided in the ASN tables
        # will be discarded.  The user can optionally provide a shfitfile relating all
        # of the images from the multiple assocation tables.
        #
        # This also applies for a mix of assocation tables and other forms of input.  For
        # example, if a user were to give an ASN table and a '@' file as input, the shift
        # information from the ASN table would be discarded.
        else:
            # Create a new assocation file
            driz_asn_file = self._generateAsnTable(self.output, self.files, self.shiftfile)
             

        # Run PyDrizzle; make sure there are no intermediate products
        # laying around...        
        assoc = pydrizzle.PyDrizzle(driz_asn_file, idckey=self.coeffs,
                                    section=self.driz_sep_pars['group'],
                                    bits=self.bits,
                                    prodonly=False)

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
                for file in self.ivmlist:
                    if file[0] == fname:
                        ivmname = file[1] 
                       
            plist['ivmname'] = ivmname

        return assoc


    def _generateAsnTable(self,output, input, shfile):

        """ Generates the association table. """

        # Create the full filename of the association table
        name = output + '_asn.fits'

        # Delete the file if it exists.
        if os.path.exists(name):
            warningmsg =  "\n#########################################\n"
            warningmsg += "#                                       #\n"
            warningmsg += "# WARNING:                              #\n"
            warningmsg += "#  The exisiting assocation table,      #\n"
            warningmsg += "           " + str(name) + '\n'
            warningmsg += "#  is being replaced by Multidrizzle.   #\n"
            warningmsg += "#                                       #\n"
            warningmsg += "#########################################\n\n"
            print warningmsg
            os.remove(name)

        print 'building ASN table for: ',name
        print 'based on suffix of:     ',input
        print 'with a shiftfile of:    ',shfile

        # Must strip out _asn.fits from name since buildAsnTable adds
        # it back again...
        buildasn.buildAsnTable(output,
                               suffix=input,
                               shiftfile=shfile)

        print 'Assoc. table = ', name
        print('')
        print('')
        
        # Return the name of the new assocation table.
        return name

    def _checkAndUpdateAsn(self, name, shfile):
        """ Updates association table with user-supplied shifts, if given. """
        if fileutil.checkFileExists(name):
            if (shfile != None) and (len(shfile) > 0):
                updateasn.updateShifts(name,shfile,mode='replace')

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
            # the stopping of execution withour raising an acception.  This
            # keeps the HST pipeline from crashing because of a rasied
            # reception.  This state likely occured because all of the input
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
        
        # Print the input parameters now that MDRIZTAB has had a chance to modify the default values
        self._printInputPars(self.pars.switches)

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

        if self.pars.switches['timing']:
            self.steps.reportTimes()

    def help(self):
        """ Prints help string for MultiDrizzle class."""
        print 'MultiDrizzle Version ',__version__
        print self.__doc__        
