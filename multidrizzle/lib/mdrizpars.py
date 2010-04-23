from __future__ import division # confidence high

import procstep as ps
import mdzhandler
import string
import sys,types,os

import numpy as np

from pytools import fileutil
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
    a = np.array(value)
#    b = np.where(np.isnan(a))
    if np.any(np.isnan(a)): return None
    return value

def cleanInt(value):
    # THIS MAY BE MACHINE-DEPENDENT !!!
    if value == -2147483647:
    # Try to use 'sys.maxint' as value (WJH)
    #if value == -sys.maxint:
        return None
    return value

def cleanBlank(value):
    if value.strip() == '':
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
                        'tophat':'tophat',
                        'lanczos3': 'lanczos3'}) 
                        )
    enum_wht = Trait('',TraitPrefixMap({
                        'ERR': 'ERR',
                        'IVM': 'IVM',
                        'EXP': 'EXP'}) 
                        )

    enum_combine = Trait('minmed',TraitPrefixMap({
                        'median': 'median',
                        'sum': 'sum',
                        'minmed': 'minmed',
                        'minimum': 'minimum'}) 
                        )
    enum_interp = Trait('poly5',TraitPrefixMap({
                        'nearest': 'nearest',
                        'linear': 'linear',
                        'poly3': 'poly3',
                        'poly5': 'poly5',
                        'sinc':'sinc'}) 
                        )

    enum_finalunits = Trait('cps',TraitPrefixMap({ 
                        'cps': 'cps',
                        'counts': 'counts'}) 
                        )
    
    enum_procunit = Trait('native',TraitPrefixMap({ 
                        'native': 'native',
                        'electrons': 'electrons'}) 
                        )

    text_editor = TraitEditorText()

    __traits__ = {'input':Trait('flt.fits',TraitString()),
            'output':Trait('',TraitString()),
            'mdriztab':Trait(False, true_boolean, editor=bit_editor),
            'refimage':Trait('',AnyValue),
            'runfile':Trait('multidrizzle.run',TraitString()),
            'workinplace':Trait(False, true_boolean, editor=bit_editor),
            'updatewcs':Trait(True, true_boolean, editor=bit_editor),   
            'proc_unit':Trait('native',enum_procunit, editor=text_editor),
            'context':Trait(True, true_boolean, editor=bit_editor), 
            'clean':Trait(False, true_boolean, editor=bit_editor),
            'group':Trait('',AnyValue),
            'ra':Trait('',AnyValue), 
            'dec':Trait('',AnyValue),
            'coeffs':Trait('header',TraitString()), 
            'build':Trait(True, true_boolean, editor=bit_editor), 
            'shiftfile':Trait('',AnyValue),
            'staticfile':Trait('',AnyValue), 
            'static_sig':Trait(4.0,TraitRange(0.0,9.0)), 
            'skywidth':Trait(0.1,TraitRange(0.0,1.0)), 
            'skystat':Trait('median',enum_stat, editor=text_editor), 
            'skylower':Trait(None,AnyValue),
            'skyupper':Trait(None,AnyValue), 
            'skyclip':Trait(5,AnyValue), 
            'skylsigma':Trait(4.0,TraitRange(0.0,9.0)),
            'skyusigma':Trait(4.0,TraitRange(0.0,9.0)), 
            'skyuser':Trait('',TraitString()),
            'driz_sep_outnx':Trait('',AnyValue), 
            'driz_sep_outny':Trait('',AnyValue),
            'driz_sep_kernel':Trait('turbo',enum_kernel, editor=text_editor),
            'driz_sep_wt_scl':Trait('exptime',TraitString()), 
            'driz_sep_pixfrac':Trait(1.0,TraitRange(0.0,2.0)),
            'driz_sep_scale':Trait('',AnyValue), 
            'driz_sep_rot':Trait('',AnyValue),
            'driz_sep_fillval':Trait('INDEF',TraitString()),
            'driz_sep_bits':Trait(0,AnyValue), 
            'median_newmasks':Trait(True, true_boolean, editor=bit_editor), 
            'combine_type':Trait('minmed',enum_combine, editor=text_editor), 
            'combine_nsigma':Trait('4 3',TraitString()),
            'combine_nlow':Trait(0,AnyValue), 
            'combine_nhigh':Trait(1,AnyValue), 
            'combine_lthresh':Trait('',AnyValue), 
            'combine_hthresh':Trait('',AnyValue), 
            'combine_grow':Trait(1.0,TraitRange(0.0,21.0)),
            'combine_maskpt':Trait(0.7,TraitRange(0.0,1.0)),
            'blot_interp':Trait('poly5',enum_interp, editor=text_editor), 
            'blot_sinscl':Trait(1.0,TraitRange(0.0,21.0)),
            'driz_cr_corr':Trait(False, true_boolean, editor=bit_editor),
            'driz_cr_snr': Trait('3.5 3.0',TraitString()), 
            'driz_cr_scale':Trait('1.2 0.7',TraitString()),
            'driz_cr_grow':Trait(1,TraitRange(1,10)),  
            'driz_cr_ctegrow':Trait(0,TraitRange(0,30)),
            'driz_final_wht_type':Trait('EXP',enum_wht, editor=text_editor),
            'driz_final_outnx':Trait('',AnyValue), 
            'driz_final_outny':Trait('',AnyValue),
            'driz_final_kernel':Trait('square',enum_kernel, editor=text_editor),
            'driz_final_wt_scl':Trait('exptime',TraitString()),  
            'driz_final_pixfrac':Trait(1.0,TraitRange(0.0,2.0)),
            'driz_final_scale':Trait('',AnyValue), 
            'driz_final_rot':Trait(0.0,AnyValue),
            'driz_final_fillval':Trait('INDEF',TraitString()),
            'driz_final_bits':Trait(0,AnyValue),
            'driz_final_units':Trait('cps',enum_finalunits, editor=text_editor),  
            'gain':Trait('',AnyValue), 
            'gnkeyword':Trait('',AnyValue),
            'rdnoise':Trait('',AnyValue), 
            'rnkeyword':Trait('',AnyValue), 
            'exptime':Trait('',AnyValue),
            'expkeyword':Trait('',AnyValue), 
            'crbit': Trait('',AnyValue),
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
                'workinplace','context', 'clean','group', 'updatewcs',  
                'proc_unit','ra', 'dec','coeffs', 'build', 'shiftfile','staticfile',
                'timing',
                label='Init'),
            TraitGroup('static',
                'static_sig', 
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
                'driz_sep_wt_scl', 'driz_sep_pixfrac', 'driz_sep_scale', 
                'driz_sep_rot', 'driz_sep_fillval','driz_sep_bits',
                label='Separate Drizzle'),
            TraitGroup('median',
                'median_newmasks', 'combine_type', 'combine_nsigma',
                'combine_nlow', 'combine_nhigh','combine_lthresh',
                'combine_hthresh', 'combine_grow', 'combine_maskpt',
                label='Median')
            ),
            TraitGroup(
            TraitGroup('blot',
                'blot_interp', 'blot_sinscl',
                label='Blot'),
            TraitGroup('driz_cr',
                'driz_cr_corr','driz_cr_snr', 'driz_cr_scale', 'driz_cr_grow', 'driz_cr_ctegrow', 
                label='Driz CR'),
            TraitGroup('driz_combine','driz_final_wht_type',
                'driz_final_outnx', 'driz_final_outny',
                'driz_final_kernel', 'driz_final_wt_scl', 
                'driz_final_pixfrac','driz_final_scale', 
                'driz_final_rot', 'driz_final_fillval',
                'driz_final_bits', 'driz_final_units',
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
            
    master_list = ['mdriztab','refimage','runfile','workinplace','updatewcs', 
            'proc_unit','context', 'clean','group', 'bits', 'ra', 'dec',
            'coeffs', 'build', 'shiftfile', 
            'staticfile', 'static_sig', 
            'skywidth', 'skystat', 'skylower',
            'skyupper', 'skyclip', 'skylsigma',
            'skyusigma', 'skyuser',
            'driz_sep_outnx', 'driz_sep_outny', 'driz_sep_kernel', 'driz_sep_wt_scl',
            'driz_sep_pixfrac','driz_sep_scale', 'driz_sep_rot',
            'driz_sep_fillval', 'driz_sep_bits',
            'median_newmasks', 'combine_type', 'combine_nsigma',
            'combine_nlow', 'combine_nhigh','combine_lthresh',
            'combine_hthresh', 'combine_grow', 'combine_maskpt',
            'blot_interp', 'blot_sinscl',
            'driz_cr_corr','driz_cr_snr', 'driz_cr_scale','driz_final_wht_type',
            'driz_final_outnx', 'driz_final_outny', 'driz_cr_grow', 'driz_cr_ctegrow',
            'driz_final_kernel', 'driz_final_wt_scl', 'driz_final_pixfrac',
            'driz_final_scale', 'driz_final_rot',
            'driz_final_fillval', 'driz_final_bits', 'driz_final_units',
            'gain', 'gnkeyword','rdnoise', 'rnkeyword', 
            'exptime','expkeyword', 'crbit']
    #
    # List of parameter names for which blank values need to be 
    # converted to a value of None in the master_par dictionary.
    #
    clean_string_list = [ 'output', 'group', 'shiftfile', 'staticfile',
            'ra', 'dec', 'coeffs', 'combine_lthresh', 'combine_hthresh',
            'driz_sep_scale', 'driz_sep_rot', 'driz_sep_bits', 
            'driz_final_bits', 'driz_final_scale',
            'driz_final_rot']
    
    attribute_list = ['switches','input','_set_trait_value','_set_event_value',
            'steps','output','master_pars']
                       
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
            tabdict = self._handleMdriztab(record)
            self.updatePars(tabdict)

    def verifyPar(self,kw,value):
        if value == 'None' or (kw in self.clean_string_list and value == ''):
            value = None
        if isinstance(value,types.StringType):
            if value.isdigit():
                value = int(value)
        if value == 'INDEF' and kw.find('fillval') < 0: value = None

        return value
    
    def updateMasterPars(self):
        for _par in self.switches_list:
            _val = getattr(self,_par)
            self.switches[_par] = _val
            
        for _par in self.master_list:
            if _par != 'bits':
                value = self.verifyPar(_par,getattr(self,_par))
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
            #print "keys updated in master dictionary: ",k
            # Update value of trait and coerce the
            # TraitType on the value in the process
            #
            if k[-1] != '_':

                setattr(self,k,dict[k])
            
                if k in self.switches_list:
                    # If key is a processing switch, update the
                    # switches dictionary
                    self.switches[k] = getattr(self,k)
                elif k in self.master_list:
                    # If it is not a switch, update its value
                    # in the master list.
                    value =  self.verifyPar(k,getattr(self,k))
                    self.master_pars[k] = value

        if 'bits' in dict.keys():
            if 'driz_final_bits' not in dict.keys():
                self.master_pars['driz_final_bits'] = int(dict['bits'])
                self.master_pars['driz_sep_bits'] = None
 
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
                if k[-1] != '_' and k not in self.attribute_list:
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
                if kw == 'group' and self.master_pars[kw] is not None:
                    # Always set up group as a list, if specified
                    group = []
                    glist = str(self.master_pars[kw]).split(',')
                    for g in glist: group.append(int(g))
                    _driz_dict[kw] = group
                else:
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
                _instr_dict[_kw] = self.__dict__[kw]
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
        tabdict = {}
        # for each entry in the record... 
        for indx in xrange(len(rec.array.names)):
            # ... get the name, format, and value.
            _name = rec.array.names[indx]
            _format = rec.array.formats[indx]
            _value = rec.field(_name)
             
            # Translate names from MDRIZTAB columns names to 
            # input parameter names found in IRAF par file.
            #
            if _name.find('final') > -1: _name = 'driz_'+_name
            elif _name == 'subsky': _name = 'skysub'
            elif _name == 'crbitval': _name = 'crbit'
            elif _name == 'readnoise': _name = 'rdnoise'
                        
            # We do not care about the first two columns at this point
            # as they are only used for selecting the rows
            if _name != 'filter' and _name != 'numimages':
                # start by determining the format type of the parameter
                _fmt = findFormat(_format)
                
                # Based on format type, apply proper conversion/cleaning
                if (_fmt == 'a') or (_fmt == 'A'):
                    _val = cleanBlank(_value)
                    if _val == None: _val = ''
                elif (_format == 'i1') or (_format=='1L'):
                    _val = toBoolean(_value)
                elif (_format == 'i4') or (_format == '1J'):
                    _val = cleanInt(_value)
                elif (_format == 'f4') or (_format == '1E'):
                    _val = cleanNaN(_value)
                else:
                    print 'MDRIZTAB column ',_name,' has unrecognized format',_format 
                    raise ValueError
                if _name.find('fillval') > -1 and _val == None:
                    _val = 'INDEF'
                tabdict[_name] = _val
        
        return tabdict
