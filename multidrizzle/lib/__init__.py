#import string
import os, shutil, sys

#import numarray
#import numarray.ieeespecial
#from numarray.ieeespecial import *

import pydrizzle
from pydrizzle import drutil, fileutil, buildasn, updateasn, dqpars
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

__version__ = '2.3.6 (13 October 2004)'

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
    print "\n\n"
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
    print "\n\n"

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
                 input      = 'flt.fits',
                 output     = None,
                 editpars   = False,
                 **input_dict):

        timestamp()
        print 'Running MultiDrizzle ',__version__

        # Print version information for all external python modules used
        versioninfo()        

        # Determine if the input to Multidrizzle are WFPC2 GEIS files.
        # If the input is GEIS format, it will need to be converted to
        # multiextension FITS format for processing.
        input,output = self._convertGEIS(input,output)        

        # Parse input to get the list of filenames to be processed.
        self._parseInput(input,output)
        # Check input files.
        self._checkInputFiles(self.files)
        self._reportFileNames()

        # Remember input parameters for use throughout this class.
        self.input = input

        # Initialize the master parameter dictionary.
        # This needs to be done after any input file conversion
        # since it needs to be able to open the file to read the
        # MDRIZTAB keyword, if this parameter is set to TRUE.
        self.pars = mdrizpars.MDrizPars(input, self.output, 
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

        # Verify that ERR extension exists if final_wht_type = ERR
        if ( self.pars.master_pars['driz_final_wht_type'] != None and
                self.pars.master_pars['driz_final_wht_type'].upper() == 'ERR'):
            for file in self.files:
                if not fileutil.findFile(file+"[err,1]"):
                    raise IOError,"! final_wht_type = ERR, no ERR array found for %s"%(file)

        
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
        _instrument = fileutil.getKeyword(self.files[0]+'[0]','INSTRUME')
        self._initdqpars(_instrument,self.driz_sep_pars['bits'])

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

    def _convertGEIS(self,input,output):
    
        """
        Converts GEIS input from the user into multiextension FITS input
        
        This method returns as output a string listing the name of the
        new input file(s) to use in multidrizzle replacing the GEIS
        input.
        """
        
        # Define a list to hold the input files in fits format.
        _flist = []
                            
        # Case 1
        _indx = input.find('_asn.fits')
        if  _indx > -1:
            # Input is an ASN table, so read it and return the input filenames
            asndict = fileutil.readAsnTable(input, None, prodonly=False)
          
            # Determine if the file is a GEIS file.
            # If the input is not in GEIS format do nothing and return
            # original input string  
            if findvalidgeisfile(asndict['order'][0]) == None:
                newinput = input
                newoutput = output
            # Otherwise we need to convert each file in the asn table
            # to multiextension FITS format and build a new association
            # table.
            else:
                # Create a list of geis images contained in the asn table
                geislist = []
                for name in asndict['order']:
                    geislist.append(findvalidgeisfile(name))
                    
                # Convert all of the geis images to fits format
                print "! Converting GEIS format files in ASN table to multiextension FITS"
                _flist = convertgeisinlist(geislist)
                
                # Create a new association table
                newinput = "MDZ_"+input+'_asn.fits'
                if output == None:
                    newoutput = "MDZ_"+input[:name.find('_asn.fits')]
                else:
                    newoutput = output
                    
                print "!Creating new ASN table called ",newoutput

                # if the file you wish to create already exists, delete it
                dirfiles = os.listdir(os.curdir)
                if (dirfiles.count(newoutput+'_asn.fits') > 0):
                    os.remove(newoutput+'_asn.fits')
                # Build the new association table
                buildasn.buildAsnTable(newoutput,
                                        suffix=_flist,
                                        shiftfile=None,
                                        verbose='yes')
        # Case 2
        elif input.find(',') > -1:
            # We have been given a list already, so format it as necessary
            inputlist = input.split(',')
            
            # Determine if the file is a GEIS file.
            # If the input is not in GEIS format do nothing and return
            # original input string
            if findvalidgeisfile(inputlist[0][0:inputlist[0].rfind('.')]) == None:
                newinput = input
                newoutput = output
            else:
                print "! Converting GEIS format files in comma separated list to multiextension FITS"
                _flist = convertgeisinlist(inputlist)
                count = 0
                newinput = ""
                newoutput = output
                for file in _flist:
                    if count !=0:
                        newinput = newinput+','+file
                    else:
                        newinput = newinput+file
                
                        
        # Case 3
        else:
            # We are working with either a suffix pattern,
            # a user-supplied at-file with a list of filenames or
            # a single filename for specifying the input file(s).
            # Parse this and build the appropriate list of filenames
            ilist = buildasn._findFiles(input)
            inputlist = []
            ivmcount = 0

            if (len(ilist[0]) > 2):
                for f in ilist:
                    if f[2] != None:
                        if fileutil.findFile(f[2]):
                            ivmcount += 1
                        else:
                            raise ValueError, "! Not all inverse variance maps present!"
            
            if (ivmcount == 0 or ivmcount == len(ilist)):             
                for f in ilist: 
                    inputlist.append(f[0])
            else:
                raise ValueError, "! Not all inverse variance maps present!"
            
            # Determine if the file is a GEIS file.
            # If the input is not in GEIS format do nothing and return
            # original input string  
            if findvalidgeisfile(inputlist[0][0:inputlist[0].rfind('.')]) == None:
                newinput = input
                newoutput = output
            else:
                # Set a default value for output if None is given
                if output == None:
                    output = "final"
                
                print "! Converting GEIS format files to multiextension FITS"
                _flist = convertgeisinlist(inputlist)
                if len(inputlist) == 1:
                    newinput = _flist[0]
                    newoutput = output
                else:
                    # if the file you wish to create already exists, delete it
                    dirfiles = os.listdir(os.curdir)
                    if (dirfiles.count("MDZ_"+output+'_asn.fits') > 0):
                        os.remove("MDZ_"+output+'_asn.fits')
                    # Build the new association table
                    buildasn.buildAsnTable("MDZ_"+output,
                                        suffix=_flist,
                                        shiftfile=None,
                                        verbose='yes')
                    newinput = "MDZ_"+output+'_asn.fits'
                    newoutput = "MDZ_"+output
                  
        return newinput,newoutput
        
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
            
            
    def _initdqpars(self,_instrument,_bits):
        print "Initializing DQpars file..."
        if (_instrument.lower() == 'acs'):
            _parfile = dqpars.ACSPars()
            _parfile.update(_bits)
        elif (_instrument.lower() == 'wfpc2'):
            _parfile = dqpars.WFPC2Pars()
            _parfile.update(_bits)
        elif (_instrument.lower() == 'stis'):
            _parfile = dqpars.STISPars()
            _parfile.update(_bits)
        elif (_instrument.lower() == 'nicmos'):
            _parfile = dqpars.NICMOSPars()
            _parfile.update(_bits)
        else:
            print " "
            print "****************"
            print "*"
            print "* UNABLE TO IDENTIFY INSTRUMENT DQPARAMETERS.  ALL DQ PARAMETERS TREATED AS GOOD"
            print "*"
            print "****************"


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

#            if fileutil.getKeyword(p,'instrume') == 'ACS':
#                print('\nNote: Synchronizing ACS WCS to specified distortion coefficients table\n')
#                # Update the CD matrix using the new IDCTAB
#                # Not perfect, but it removes majority of errors...
#                makewcs.run(image=p)


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

    def _parseInput(self,input,output):

        """ Interprets input from user to build list of files to process. """

        self.ivmlist = []
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
            
            ivmcount = 0
            if (len(_files[0]) > 2):
                for f in _files:
                    if f[2] != None:
                        if fileutil.findFile(f[2]):
                            ivmcount += 1
                            self.ivmlist.append((f[0],f[2]))
                        else:
                            raise ValueError, "! Not all inverse variance maps present!"            

            if (ivmcount == 0 or ivmcount == len(_files)):             
                for f in _files: 
                    _flist.append(f[0])
            else:
                raise ValueError, "! Not all inverse variance maps present!"

        # Setup default output name if none was provided either by user or in ASN table
        if (output == None) or (len(output) == 0):
            output = 'final'

        self.files = _flist
        self.output = output
                
#        return _flist,output

    def _reportFileNames(self):
        print "Input files: "
        for filename in self.files:
            print "             " + filename
            print "Output file: " + self.output

    def _setupAssociation(self):

        """ Builds the PyDrizzle association object. """

        timestamp()
        print ' *** '+'Setting up associated inputs..'+'\n *** '

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
        assoc = pydrizzle.PyDrizzle(driz_asn_file, idckey=self.coeffs,
                                    section=self.driz_sep_pars['group'],
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

    def _generateAsnTable(self, name, output, input, shfile):

        """ Generates the association table. """

        if not fileutil.findFile(name):
            print 'building ASN table for: ',output
            print 'based on suffix of:     ',input
            print 'with a shiftfile of:    ',shfile

            # Must strip out _asn.fits from name since buildAsnTable adds
            # it back again...
            buildasn.buildAsnTable(name[:name.find('_asn.fits')],
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
