import procstep as ps
import mdzhandler
import string
import sys

import numarray
import numarray.ieeespecial
from numarray.ieeespecial import *

from pydrizzle import fileutil
from pydrizzle import traits102
from pydrizzle.traits102 import *
from pydrizzle.traits102.tktrait_sheet import TraitEditorBoolean, \
                        TraitEditorText, TraitGroup

from procstep import ProcSteps

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

def findFormat(format):
    # Parses record array format string for type
    _fmt = None
    for ltr in string.letters:
        if format.find(ltr) > -1:
            _fmt = ltr
            break
    return _fmt


class MDrizPars (HasTraits):
    """ This class defines the default values for all MultiDrizzle
        parameters, and provides the mechanisms for updating them 
        from any of the available interfaces: EPAR, MDRIZTAB, or
        directly from the Python interface.  
        
        It defines a dictionary containing all the input parameters
        for MultiDrizzle.  The MultiDrizzle class inputs all but the
        three required parameters as a variable-length argument
        dictionary.  The input parameter dictionary from MultiDrizzle
        then gets used to initialize this class which then updates
        the default values it already knows about with the values
        passed in upon initialization.  It can perform parameter name
        checking in case of typos upon input, recognize the use of an 
        MDRIZTAB and pull the values from that table, then update the
        master dictionary.  This master dictionary would then serve as
        the primary attribute which would be used to set the desired 
        attributes in the MultiDrizzle class.
                
        A method supports resetting these values based on 
        new inputs, such as from MDRIZTAB.
        
        Another method supports resetting ProcStep settings, as they 
        need to be handled differently from regular MultiDrizzle attributes
        since they rely on another class.  
    """
    true_boolean = Trait('true',
                        TraitComplex(
                            TraitPrefixMap( {
                                    'true':1, 'yes':1,
                                    'false':0, 'no':0 } ),
                                TraitMap({1:True,0:False} )))   
    bit_editor = TraitEditorBoolean()


    # The following enumerated lists are only necessary 
    # to replace the use of TraitEnum for versions of Pmw 
    # earlier than 1.3, versions which have a bug.
    enum_stat  = Trait('median',TraitPrefixMap({
                        'median': 'median',
                        'mode': 'mode',
                        'mean': 'mean',}) 
                        )
    enum_kernel = Trait('square',TraitPrefixMap({
                        'square': 'square',
                        'point': 'point',
                        'gaussian': 'gaussian',
                        'turbo': 'turbo',
                        'tophap':'tophat',
                        'lanczos3': 'lanczos3'}) 
                        )
    enum_combine = Trait('median',TraitPrefixMap({
                        'median': 'median',
                        'sum': 'sum',
                        'minmed': 'minmed',}) 
                        )
    enum_interp = Trait('poly5',TraitPrefixMap({
                        'nearest': 'nearest',
                        'linear': 'linear',
                        'poly3': 'poly3',
                        'poly5': 'poly5',
                        'sinc':'sinc'}) 
                        )
    text_editor = TraitEditorText()


    """
    This definition of the MDRIZPAR traits enables proper 
    handling of integer TraitRanges and TraitEnum with the
    Pmw widgets.  Unfortunately, due to a bug in Pmw
    Versions 1.2 and less, these features will not work. 
    
    Therefore, this definition can only be activated upon updating
    to Pmw 1.3 and greater.
    
    __traits__ = {'input':Trait('flt.fits',TraitString()),
            'output':Trait('',TraitString()),
            'mdriztab':Trait(False, true_boolean, editor=bit_editor),
            'refimage':'','runfile':'multidrizzle.run',
            'workinplace':Trait(False, true_boolean, editor=bit_editor),
            'context':Trait(True, true_boolean, editor=bit_editor), 
            'clean':Trait(True, true_boolean, editor=bit_editor),
            'group':Trait('',AnyValue),
            'bits':Trait(0,TraitRange(0,65535)), 
            'ra':Trait('',AnyValue), 
            'dec':Trait('',AnyValue),
            'coeffs':Trait('header',TraitString()), 
            'build':Trait(False, true_boolean, editor=bit_editor), 
            'shiftfile':Trait('',AnyValue),
            'staticfile':Trait('',TraitString()), 
            'static_sig':Trait(3.0,TraitRange(0.0,9.0)), 
            'skywidth':Trait(0.1,TraitRange(0.0,1.0)), 
            'skystat':Trait('median',TraitEnum(['median','mode','mean'])), 
            'skylower':Trait(-50.,AnyValue),
            'skyupper':Trait(200.,AnyValue), 
            'skyclip':Trait(5,TraitRange(0,10)), 
            'skylsigma':Trait(4.0,TraitRange(0.0,9.0)),
            'skyusigma':Trait(4.0,TraitRange(0.0,9.0)), 
            'skyuser':Trait('',TraitString()),
            'driz_sep_outnx':Trait('',AnyValue), 
            'driz_sep_outny':Trait('',AnyValue),
            'driz_sep_kernel':Trait('turbo',TraitEnum(['square',
                'point','gaussian','turbo','tophat','lanczos3'])), 
            'driz_sep_pixfrac':Trait(1.0,TraitRange(0.0,2.0)),
            'driz_sep_scale':Trait('',AnyValue), 
            'driz_sep_rot':Trait('',AnyValue),
            'driz_sep_fillval':Trait('INDEF',TraitString()),
            'median_newmasks':Trait(True, true_boolean, editor=bit_editor), 
            'combine_type':Trait('median',TraitEnum(['median','sum','minmed'])), 
            'combine_nsigma':Trait('6 3',TraitString()),
            'combine_nlow':Trait(0,AnyValue), 
            'combine_nhigh':Trait(1,AnyValue), 
            'combine_lthresh':Trait('',AnyValue), 
            'combine_hthresh':Trait('',AnyValue), 
            'combine_grow':Trait(1.0,TraitRange(0.0,21.0)),
            'blot_interp':Trait('poly5',TraitEnum(['nearest','linear',
                    'poly3','poly5','sinc'])), 
            'blot_sinscl':Trait(1.0,TraitRange(0.0,21.0)),
            'driz_cr_corr':Trait(False, true_boolean, editor=bit_editor),
            'driz_cr_snr': Trait('3.0 2.5',TraitString()), 
            'driz_cr_scale':Trait('1.2 0.7',TraitString()),
            'driz_final_outnx':Trait('',AnyValue), 
            'driz_final_outny':Trait('',AnyValue),
            'driz_final_kernel':Trait('square',TraitEnum(['square',
                'point','gaussian','turbo','tophat','lanczos3'])), 
            'driz_final_pixfrac':Trait(1.0,TraitRange(0.0,2.0)),
            'driz_final_scale':Trait('',AnyValue), 
            'driz_final_rot':Trait(0.0,AnyValue),
            'driz_final_fillval':Trait('INDEF',TraitString()),
            'gain':Trait('',TraitString()), 
            'gnkeyword':Trait('',TraitString()),
            'rdnoise':Trait('',TraitString()), 
            'rnkeyword':Trait('',TraitString()), 
            'exptime':Trait('',TraitString()),
            'expkeyword':Trait('',TraitString()), 
            'crbit': Trait(64,TraitRange(0,65535)),
            'static':Trait(True, true_boolean, editor=bit_editor), 
            'skysub':Trait(True, true_boolean, editor=bit_editor), 
            'driz_separate':Trait(True, true_boolean, editor=bit_editor),
            'median':Trait(True, true_boolean, editor=bit_editor), 
            'blot':Trait(True, true_boolean, editor=bit_editor), 
            'driz_cr':Trait(True, true_boolean, editor=bit_editor), 
            'driz_combine':Trait(True, true_boolean, editor=bit_editor),
            'timing':Trait(True, true_boolean, editor=bit_editor)
            }
    """
    __traits__ = {'input':Trait('flt.fits',TraitString()),
            'output':Trait('',TraitString()),
            'mdriztab':Trait(False, true_boolean, editor=bit_editor),
            'refimage':'','runfile':'multidrizzle.run',
            'workinplace':Trait(False, true_boolean, editor=bit_editor),
            'context':Trait(True, true_boolean, editor=bit_editor), 
            'clean':Trait(True, true_boolean, editor=bit_editor),
            'group':Trait('',AnyValue),
            'bits':Trait(0,AnyValue), 
            'ra':Trait('',AnyValue), 
            'dec':Trait('',AnyValue),
            'coeffs':Trait('header',TraitString()), 
            'build':Trait(False, true_boolean, editor=bit_editor), 
            'shiftfile':Trait('',AnyValue),
            'staticfile':Trait('',TraitString()), 
            'static_sig':Trait(3.0,TraitRange(0.0,9.0)), 
            'skywidth':Trait(0.1,TraitRange(0.0,1.0)), 
            'skystat':Trait('median',enum_stat, editor=text_editor), 
            'skylower':Trait(-50.,AnyValue),
            'skyupper':Trait(200.,AnyValue), 
            'skyclip':Trait(5,AnyValue), 
            'skylsigma':Trait(4.0,TraitRange(0.0,9.0)),
            'skyusigma':Trait(4.0,TraitRange(0.0,9.0)), 
            'skyuser':Trait('',TraitString()),
            'driz_sep_outnx':Trait('',AnyValue), 
            'driz_sep_outny':Trait('',AnyValue),
            'driz_sep_kernel':Trait('turbo',enum_kernel, editor=text_editor), 
            'driz_sep_pixfrac':Trait(1.0,TraitRange(0.0,2.0)),
            'driz_sep_scale':Trait('',AnyValue), 
            'driz_sep_rot':Trait('',AnyValue),
            'driz_sep_fillval':Trait('INDEF',TraitString()),
            'median_newmasks':Trait(True, true_boolean, editor=bit_editor), 
            'combine_type':Trait('median',enum_combine, editor=text_editor), 
            'combine_nsigma':Trait('6 3',TraitString()),
            'combine_nlow':Trait(0,AnyValue), 
            'combine_nhigh':Trait(1,AnyValue), 
            'combine_lthresh':Trait('',AnyValue), 
            'combine_hthresh':Trait('',AnyValue), 
            'combine_grow':Trait(1.0,TraitRange(0.0,21.0)),
            'blot_interp':Trait('poly5',enum_interp, editor=text_editor), 
            'blot_sinscl':Trait(1.0,TraitRange(0.0,21.0)),
            'driz_cr_corr':Trait(False, true_boolean, editor=bit_editor),
            'driz_cr_snr': Trait('3.0 2.5',TraitString()), 
            'driz_cr_scale':Trait('1.2 0.7',TraitString()),
            'driz_final_outnx':Trait('',AnyValue), 
            'driz_final_outny':Trait('',AnyValue),
            'driz_final_kernel':Trait('square',enum_kernel, editor=text_editor), 
            'driz_final_pixfrac':Trait(1.0,TraitRange(0.0,2.0)),
            'driz_final_scale':Trait('',AnyValue), 
            'driz_final_rot':Trait(0.0,AnyValue),
            'driz_final_fillval':Trait('INDEF',TraitString()),
            'gain':Trait('',TraitString()), 
            'gnkeyword':Trait('',TraitString()),
            'rdnoise':Trait('',TraitString()), 
            'rnkeyword':Trait('',TraitString()), 
            'exptime':Trait('',TraitString()),
            'expkeyword':Trait('',TraitString()), 
            'crbit': Trait(64,AnyValue),
            'static':Trait(True, true_boolean, editor=bit_editor), 
            'skysub':Trait(True, true_boolean, editor=bit_editor), 
            'driz_separate':Trait(True, true_boolean, editor=bit_editor),
            'median':Trait(True, true_boolean, editor=bit_editor), 
            'blot':Trait(True, true_boolean, editor=bit_editor), 
            'driz_cr':Trait(True, true_boolean, editor=bit_editor), 
            'driz_combine':Trait(True, true_boolean, editor=bit_editor),
            'timing':Trait(True, true_boolean, editor=bit_editor)
            }

    __editable_traits__= TraitGroup(
            TraitGroup(
            TraitGroup(
                'input','output','mdriztab','refimage','runfile',
                'workinplace','context', 'clean','group', 'bits', 
                'ra', 'dec','coeffs', 'build', 'shiftfile','timing',
                label='Init'),
            TraitGroup('static',
                'staticfile', 'static_sig', 
                label='Static Mask'),
            TraitGroup('skysub',
                'skywidth', 'skystat', 'skylower',
                'skyupper', 'skyclip', 'skylsigma',
                'skyusigma', 'skyuser',
                label='Sky')
                ),
            TraitGroup(
            TraitGroup('driz_separate',
                'driz_sep_outnx', 'driz_sep_outny', 'driz_sep_kernel',
                'driz_sep_pixfrac','driz_sep_scale', 'driz_sep_rot',
                'driz_sep_fillval',
                label='Separate Drizzle'),
            TraitGroup('median',
                'median_newmasks', 'combine_type', 'combine_nsigma',
                'combine_nlow', 'combine_nhigh','combine_lthresh',
                'combine_hthresh', 'combine_grow',
                label='Median')
            ),
            TraitGroup(
            TraitGroup('blot',
                'blot_interp', 'blot_sinscl',
                label='Blot'),
            TraitGroup('driz_cr',
                'driz_cr_corr','driz_cr_snr', 'driz_cr_scale',
                label='Driz CR'),
            TraitGroup('driz_combine',
                'driz_final_outnx', 'driz_final_outny',
                'driz_final_kernel', 'driz_final_pixfrac',
                'driz_final_scale', 'driz_final_rot',
                'driz_final_fillval',
                label='Final Drizzle'),
            TraitGroup(
                'gain', 'gnkeyword','rdnoise', 'rnkeyword', 
                'exptime','expkeyword', 'crbit',
                label='Instrument')
                ),
            orientation='horizontal')            
         
    input_list = ['input','output']
    
    switches_list = ['static', 'skysub', 'driz_separate',
            'median', 'blot', 'driz_cr', 'driz_combine','timing']
            
    master_list = ['mdriztab','refimage','runfile','workinplace',
            'context', 'clean','group', 'bits', 'ra', 'dec',
            'coeffs', 'build', 'shiftfile', 
            'staticfile', 'static_sig', 
            'skywidth', 'skystat', 'skylower',
            'skyupper', 'skyclip', 'skylsigma',
            'skyusigma', 'skyuser',
            'driz_sep_outnx', 'driz_sep_outny', 'driz_sep_kernel',
            'driz_sep_pixfrac','driz_sep_scale', 'driz_sep_rot',
            'driz_sep_fillval',
            'median_newmasks', 'combine_type', 'combine_nsigma',
            'combine_nlow', 'combine_nhigh','combine_lthresh',
            'combine_hthresh', 'combine_grow',
            'blot_interp', 'blot_sinscl',
            'driz_cr_corr','driz_cr_snr', 'driz_cr_scale',
            'driz_final_outnx', 'driz_final_outny',
            'driz_final_kernel', 'driz_final_pixfrac',
            'driz_final_scale', 'driz_final_rot',
            'driz_final_fillval',
            'gain', 'gnkeyword','rdnoise', 'rnkeyword', 
            'exptime','expkeyword', 'crbit']
    #
    # List of parameter names for which blank values need to be 
    # converted to a value of None in the master_par dictionary.
    #
    clean_string_list = [ 'output', 'group', 'shiftfile', 'staticfile',
            'ra', 'dec', 'coeffs', 'combine_lthresh', 'combine_hthresh',
            'bits', 'driz_sep_scale', 'driz_sep_rot', 'driz_final_scale',
            'driz_final_rot']
               
    SHELVENAME = 'mdrizpars'
    
    def __init__(self, input, output, dict=None, files = None):
        """ The input parameter 'dict' needs to be a Python dictionary
            of attributes whose values need to be updated. 
        """        
        self.input = input
        self.output = output
        
        # Initialize switches and master_pars dictionaries
        # based on defaults set up using the traits
        self.switches = {}
        self.master_pars = {}
        self.updateMasterPars()

        # Initialize the attributes for storing the parameters
        # as a shelve file
        #
        self.shelve_dir = self._findShelveDir()
        self.shelve = self.shelve_dir +self.SHELVENAME
        # If a previous shelve file existed, then restore those
        # values to override the basic defaults.
        self.restorePars()
              
        # Now, apply any new values input through keywords 
        # upon start up. This will further override any previous
        # settings for those parameters.
        if dict != None:
            self.updatePars(dict)
                            
        # Initialize ProcSteps here as well
        self.steps = ps.ProcSteps()
        self.setProcSteps()     

        # MDRIZTAB must be opened here, now that the final form of the 
        # input files has been determined and 
        # before the association is built.
        if self.master_pars['mdriztab']:
            record = mdzhandler.getMultidrizzleParameters(files)
            self._handleMdriztab(record)

    def updateMasterPars(self):
        for _par in self.switches_list:
            self.switches[_par] = getattr(self,_par)
            
        for _par in self.master_list:
            value = getattr(self,_par)
            if value == 'None' or (_par in self.clean_string_list and value == ''):
                value = None
            self.master_pars[_par] = value
        
    def updatePars(self,dict):
        # Verify that all inputs correspond to keywords in the
        # master dictionary, otherwise, raise an exception
        self.verifyInput(dict)

        # Copy values for input keywords into master dictionary
        #
        # NOTE:
        # Updating these values later will require copying the 
        # values to two places: these dictionaries and to __dict__
        #
        for k in dict.keys():
            # Update value of trait and coerce the
            # TraitType on the value in the process
            setattr(self,k,dict[k])
            
            if k in self.switches_list:
                # If key is a processing switch, update the
                # switches dictionary
                self.switches[k] = getattr(self,k)
            elif k in self.master_list:
                # If it is not a switch, update its value
                # in the master list.
                self.master_pars[k] = getattr(self,k)
                           

    def _findShelveDir(self):
        """ Tracks down the location of the shelve files
        used for keeping the user-specified bit values. """
        return fileutil.osfn('home$')

    def setShelveDir(self,dirname):
        """ Sets the name of the directory to be used for
            saving the paramter values as a shelve file.
            
            We use 'osfn' to make sure any variables get fully
            expanded.
        """
        self.shelve_dir = fileutil.osfn(dirname)

    def savePars(self):
        """ Open the shelve file and write/reset the 'dqpar' entry. """

        _shelve = shelve.open(self.shelve)
        _shelve[self.SHELVENAME] = self.__dict__
        _shelve.close()

    def restorePars(self):
        """ Find what shelve file is to be used and read the 'dqpar' entry. """
        if fileutil.findFile(self.shelve):
            _shelve = shelve.open(self.shelve)
            self.__dict__ = _shelve[self.SHELVENAME]
            _shelve.close()
          
    def editPars(self):
        """ Edits the traits and updates values in appropriate 
            dictionaries.
        """
        self.edit_traits()            
        self.updatePars(self.__dict__)
        self.savePars()
            
 
    def verifyInput(self,dict):
        """ Verifies that all entries provided in the input dictionary
            correspond to keys in the master dictionary. 
            
            If there are mismatches, then it will report those errant
            keywords and raise an Exception. This comparison will be 
            case-insensitive, for simplicity.
        """
        if dict != None:
            _err_str = 'MultiDrizzle inputs which are not recognized:\n'
            _num_invalid = 0
            for k in dict.keys():
                if (not k.lower() in self.master_list and 
                    not k.lower() in self.switches_list ):
                    _err_str += 'Unrecognized key: '+str(k)+'\n'
                    _num_invalid += 1
            if _num_invalid > 0:
                print _err_str
                raise ValueError
            
    def setProcSteps(self, **switches):
        """ Update the master parameter list with the step settings
            given in 'switches', then update the ProcStep 
            instance 'self.steps' appropriately.
             
        """
        # Start by updating the master list of values for 
        # the switch settings, 
        # after verifying that all entries are valid.
        self.verifyInput(switches)

        for k in switches:
            # Only update the switch setting if it has been set
            # to something other than None.
            if switches[k] != None:
                self.switches[k] = switches[k]
        
        # Now, update the step settings for the class    
        self.steps.addSteps(self.switches['static'],
                            self.switches['skysub'],
                            self.switches['driz_separate'],
                            self.switches['median'],
                            self.switches['blot'],
                            self.switches['driz_cr'],
                            self.switches['driz_combine'])
                            
    def getDrizPars(self,prefix='driz_sep',keylist=None):
        """ Returns a dictionary of values used for the drizzle 
            processing steps. The prefix defines which set of 
            keywords from the master parameter list needs
            to be returned. 
            
            The member names in the output dictionary, though, will
            not include the specified prefix as required by PyDrizzle.

            Additional keywords used for this dictionary are listed
            in the module parameter 'driz_keys'.
        """
        
        _driz_dict = {}
        _prefix_len = len(prefix)+1
        for kw in self.master_pars.keys():
            if self.master_pars[kw] == '': 
                self.master_pars[kw] = None
            if kw.find(prefix) > -1:
                _kwname =  kw[_prefix_len:]
                if _kwname != 'fillval':
                    _driz_dict[_kwname] = self.master_pars[kw]
                else:
                    _driz_dict[_kwname] = str(self.master_pars[kw])
                
        # Append any values for keywords provided by user in keylist
        if keylist != None:
            for kw in keylist:
                _driz_dict[kw] = self.master_pars[kw]
        
        return _driz_dict
        
    def getParList(self,keylist,prefix=None):
        """ Returns a dictionary of values used for setting 
            the parameters listed in keylist.
            
            If a prefix is specified, then remove that prefix
            from the master parameter name when creating the 
            output dictionary.            
        """
        
        _instr_dict = {}
        for kw in keylist:
            if prefix != None and kw.find(prefix) > -1:
                _kw = kw[len(prefix):]
            else:
                _kw = kw

            if kw in self.input_list:
                _inst_dict[_kw] = self.__dict__[kw]
            elif self.master_pars.has_key(kw):
                _instr_dict[_kw] = self.master_pars[kw]
            elif self.switches.has_key(kw):
                _instr_dict[_kw] = self.switches[kw]
            else:
                _instr_dict[_kw] = None
        
        return _instr_dict
        
    def _handleMdriztab(self, rec):
        """
        Collect task parameters from the MDRIZTAB record and
        update the master parameters list with those values

        Note that parameters read from the MDRIZTAB record must
        be cleaned up in a similar way that parameters read
        from the user interface are.
        """
        # for each entry in the record... 
        for indx in xrange(len(rec.array.names)):
            # ... get the name, format, and value.
            _name = rec.array.names[indx]
            _format = rec.array.formats[indx]
            _value = rec.field(_name)
            
            # We do not care about the first two columns at this point
            # as they are only used for selecting the rows
            if _name != 'filter' and _name != 'numimages':
                # start by determining the format type of the parameter
                _fmt = findFormat(_format)
                
                # Based on format type, apply proper conversion/cleaning
                if _fmt == 'a':
                    _val = cleanBlank(_value)
                elif _format == 'i1':
                    _val = toBoolean(_value)
                elif _fmt == 'i':
                    _val = cleanInt(_value)
                elif _fmt == 'f':
                    _val = cleanNaN(_value)
                else:
                    print 'MDRIZTAB column ',_name,' has unrecognized format',_format 
                    raise ValueError
                # Set master parameters dictionary or 
                #     master switch dictionary with value
                if self.switches.has_key(_name):
                    self.switches[_name] = _val
                else:
                    self.master_pars[_name] = _val

        # Take care of special parameter values
        if self.master_pars['driz_sep_fillval'] == None:
            self.master_pars['driz_sep_fillval'] = 'INDEF'
        if self.master_pars['final_fillval'] == None:
            self.master_pars['final_fillval'] = 'INDEF'
        
