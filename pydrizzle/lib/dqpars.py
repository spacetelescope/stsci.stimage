"""This module manages the specification of which DQ bits are considered
    good for each instrument.
"""
import shelve  # Used as a replacement for the IRAF 'par' files
import os, types
import fileutil

from traits102 import *
from traits102.tktrait_sheet import TraitEditorBoolean, TraitGroup
# Newer version requires Python2.3 (for BooleanType)
#from traits import *
#from traits.tktrait_sheet import TraitEditorBoolean, TraitGroup

class DQmask (HasTraits):
    NUMBITS = 15

    true_boolean = Trait('true',
                        TraitComplex(
                            TraitPrefixMap( {
                                    'true':1, 'yes':1,
                                    'false':0, 'no':0 } ),
                                TraitMap({1:1,0:0} )))
    bit_editor = TraitEditorBoolean()
    __traits__ = { 'bit1' : Trait(0, true_boolean, editor=bit_editor),
                 'bit2': Trait(0, true_boolean, editor=bit_editor),
                 'bit3': Trait(0, true_boolean, editor=bit_editor),
                 'bit4': Trait(0, true_boolean, editor=bit_editor),
                 'bit5': Trait(0, true_boolean, editor=bit_editor),
                 'bit6': Trait(0, true_boolean, editor=bit_editor),
                 'bit7': Trait(0, true_boolean, editor=bit_editor),
                 'bit8': Trait(0, true_boolean, editor=bit_editor),
                 'bit9': Trait(0, true_boolean, editor=bit_editor),
                 'bit10': Trait(0, true_boolean, editor=bit_editor),
                 'bit11': Trait(0, true_boolean, editor=bit_editor),
                 'bit12': Trait(0, true_boolean, editor=bit_editor),
                 'bit13': Trait(0, true_boolean, editor=bit_editor),
                 'bit14': Trait(0, true_boolean, editor=bit_editor),
                 'bit15': Trait(0, true_boolean, editor=bit_editor),
               }

    __editable_traits__ = ['bit1','bit2','bit3','bit4',
                            'bit5','bit6','bit7','bit8',
                            'bit9','bit10','bit11','bit12',
                            'bit13','bit14','bit15'
                            ]
    def __init__(self, values=None):
        self.mask = values
        if values:
            for _bit in xrange(len(values)):
                if _bit < self.NUMBITS:
                    self.__dict__['bit'+str(_bit+1)] = values[_bit]

    def update(self):
        try:
            clone = self.__class__()
            clone.clone_traits( self.dqmask)
        except:
            clone = None

        if clone is not None:
            self.clone_traits(clone)

        self.mask = []
        for _bit in self.editable_traits():
            self.mask.append(eval('self.'+_bit))

    def setLabels(self,labels=None):
        """ Set the labels for the bit value traits. """
        if labels:
            for _bit in self.editable_traits():
                self.__traits__[_bit].label = labels[_bit]

class DQPars:
    """ Base class for setting and remembering which DQ values are
        to be interpreted as 'good'.
    """
    INSTNAME = 'default'
    BITLABELS = { 'bit1':'Bit Value 1',
                    'bit2':'Bit Value 2',
                    'bit3':'Bit Value 4',
                    'bit4':'Bit Value 8',
                    'bit5':'Bit Value 16',
                    'bit6':'Bit Value 32',
                    'bit7':'Bit Value 64',
                    'bit8':'Bit Value 128',
                    'bit9':'Bit Value 256',
                    'bit10':'Bit Value 512',
                    'bit11':'Bit Value 1024',
                    'bit12':'Bit Value 2048',
                    'bit13':'Bit Value 4096',
                    'bit14':'Bit Value 8192',
                    'bit15':'Bit Value 16384'
                }

    def __init__(self):
        self.uparm = None
        self.pars = None
        self.mask = None
        self.bits = 0 # integer-sum of mask
        self.shelve_dir = self._findShelveDir()

        # Set up shelve for the instrument
        self.shelve = self.shelve_dir +'pydrizzle_dqpars'

        self.readMaskPar()
        self.setBits()

        # Initialize DQmask object to manage bit values mask as traits
        self.dqmask = None

    def _findShelveDir(self):
        """ Tracks down the location of the shelve files
        used for keeping the user-specified bit values. """
        return fileutil.osfn('home$')

    def readMaskPar(self):
        """ Find what shelve file is to be used and read the 'dqpar' entry. """
        _shelve = shelve.open(self.shelve)
        if _shelve.has_key(self.INSTNAME):
            self.mask = _shelve[self.INSTNAME]
        else:
            self.mask = None
        _shelve.close()

    def edit(self):
        """ Use GUI editor to edit bit values for mask. """
        _dqmask = DQmask(values = self.mask)
        _dqmask.setLabels(labels=self.BITLABELS)
        _dqmask.edit_traits()

        # Transfer the changes made in the editor
        _dqmask.update()

        self.update(mask = _dqmask.mask)

    def update(self, mask=None):
        """ Open the shelve file and write/reset the 'dqpar' entry. """

        _shelve = shelve.open(self.shelve)
        _shelve[self.INSTNAME] = mask
        _shelve.close()

        # Update mask value with new one
        self.mask = mask

        # ...and update bits value at the same time
        self.setBits()

    def setBits(self):
        """ Computes the integer summed value for use by PyDrizzle of
        the DQ bits specified in the mask. """

        if isinstance(self.mask,types.IntType):
            # mask was set to an integer, so just pass it along...
            self.bits = self.mask
        elif self.mask != None:
            _indx = 0
            self.bits = 0
            for _bit in self.mask:
                b = _bit*pow(2,_indx)
                self.bits += b
                _indx += 1
        else:
            self.bits = None

class ACSPars(DQPars):
    """ DQPars sub-class for interpreting DQ arrays for ACS data. """
    INSTNAME = 'acs'
    BITLABELS = { 'bit1':'Reed-Solomon decoding error',
                'bit2':'Replaced by fill value',
                'bit3':'bad detector pixel',
                'bit4':'masked by aperture feature',
                'bit5':'hot pixel',
                'bit6':'large blemish',
                'bit7':'N/A',
                'bit8':'bias level pixel',
                'bit9':'saturated pixel',
                'bit10':'bad pixel in reference file',
                'bit11':'small blemish',
                'bit12':'A-to-D saturated pixel',
                'bit13':'N/A',
                'bit14':'affected by cosmic-ray',
                'bit15':'N/A' }

    def __init__(self):
        DQPars.__init__(self)


class WFPC2Pars(DQPars):
    """ DQPars sub-class for interpreting DQ arrays for WFPC2 data. """
    INSTNAME = 'wfpc2'
    def __init__(self):
        DQPars.__init__(self)


class STISPars(DQPars):
    """ DQPars sub-class for interpreting DQ arrays for STIS data. """
    INSTNAME = 'stis'
    def __init__(self):
        DQPars.__init__(self)


class NICMOSPars(DQPars):
    """ DQPars sub-class for interpreting DQ arrays for NICMOS data. """
    INSTNAME = 'nicmos'
    def __init__(self):
        DQPars.__init__(self)
