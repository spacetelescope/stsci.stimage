#-------------------------------------------------------------------------------
#
#  Define common, low-level capabilities needed by the 'traits' package.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  Refactored into a separate module: 07/04/2003
#
#  Symbols defined: SequenceTypes
#                   Undefined
#                   trait_editors
#                   class_of
#
#  (c) Copyright 2002, 2003 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from types import ListType, TupleType, StringType, UnicodeType, IntType, \
                            LongType, FloatType, ComplexType, BooleanType

import sys
#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

SequenceTypes = ( ListType, TupleType )

TraitNotifier = '__trait_notifier__'

#-------------------------------------------------------------------------------
#  'UndefinedObject' class:
#-------------------------------------------------------------------------------

class UndefinedObject:

    def __repr__ ( self ):
        return '<undefined>'

#-------------------------------------------------------------------------------
#  'SelfObject' class:
#-------------------------------------------------------------------------------

class SelfObject:

    def __repr__ ( self ):
        return '<self>'

#-------------------------------------------------------------------------------
#  Create singleton-like instances (so users don't have to):
#-------------------------------------------------------------------------------

Undefined = UndefinedObject() # Undefined trait name and/or value
Self      = SelfObject()      # Global object reference to current 'object'

#-------------------------------------------------------------------------------
#  Define a special 'string' coercion function:
#-------------------------------------------------------------------------------

def strx ( arg ):
    if type( arg ) in StringTypes:
        return str( arg )
    raise TypeError

#-------------------------------------------------------------------------------
#  Define a special 'unicode' string coercion function:
#-------------------------------------------------------------------------------

def unicodex ( arg ):
    if type( arg ) in StringTypes:
        return unicode( arg )
    raise TypeError

#-------------------------------------------------------------------------------
#  Define a special 'int' coercion function:
#-------------------------------------------------------------------------------

def intx ( arg ):
    try:
        return int( arg )
    except:
        try:
            return int( float( arg ) )
        except:
            return int( long( arg ) )

#-------------------------------------------------------------------------------
#  Define a special 'long' coercion function:
#-------------------------------------------------------------------------------

def longx ( arg ):
    try:
        return long( arg )
    except:
        return long( float( arg ) )

#-------------------------------------------------------------------------------
#  Define a special 'float' coercion function:
#-------------------------------------------------------------------------------

def floatx ( arg ):
    try:
        return float( arg )
    except:
        return float( long( arg ) )

#-------------------------------------------------------------------------------
#  Define a special 'complex' coercion function:
#-------------------------------------------------------------------------------

def complexx ( arg ):
    try:
        return complex( arg )
    except:
        return complex( long( arg ) )

#-------------------------------------------------------------------------------
#  Define a special 'boolean' coercion function:
#-------------------------------------------------------------------------------

def booleanx ( arg ):
    if arg:
        return True
    return False

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

NumericFuncs = { IntType:   ( intx,   'an integer' ),
                 LongType:  ( longx,  'a long integer' ),
                 FloatType: ( floatx, 'a floating point number' ) }

StringTypes  = ( StringType, UnicodeType, IntType, LongType, FloatType,
                 ComplexType )

#-------------------------------------------------------------------------------
#  Define a mapping from types to coercion functions:
#-------------------------------------------------------------------------------

CoercableFuncs = { IntType:     intx,
                   LongType:    longx,
                   FloatType:   floatx,
                   ComplexType: complexx,
                   StringType:  strx,
                   UnicodeType: unicodex,
                   BooleanType: booleanx }

#-------------------------------------------------------------------------------
#  Return the module defining the set of trait editors we should use:
#-------------------------------------------------------------------------------

trait_editors_module      = None
trait_editors_module_name = None

def trait_editors ( module_name = None ):
    global trait_editors_module, trait_editors_module_name
    if module_name is not None:
        if module_name != trait_editors_module_name:
            trait_editors_module_name = module_name
            trait_editors_module      = None
        return
    # Determine actual module name for traits package
    traits_prefix = 'enthought.traits'
    for key in sys.modules.keys():
        if key.find('.traits') > -1: traits_prefix = key

    if trait_editors_module is None:
        if trait_editors_module_name is None:
            try:
                __import__( 'wxPython' )
                trait_editors_module_name = traits_prefix[:-6]+'wxtrait_sheet'
            except ImportError:
                try:
                    __import__( 'Tkinter' )
                    trait_editors_module_name = traits_prefix[:-6]+'tktrait_sheet'
                except ImportError:
                    return None

        try:
            trait_editors_module = sys.modules[ trait_editors_module_name ]
        except:
            try:
                trait_editors_module = __import__( trait_editors_module_name )
                for item in trait_editors_module_name.split( '.' )[1:]:
                    trait_editors_module = getattr( trait_editors_module, item )
            except ImportError:
                trait_editors_module = None

    return trait_editors_module

#-------------------------------------------------------------------------------
#  Return a string containing the class name of an object with the correct
#  article (a or an) preceding it (e.g. 'an Image', 'a PlotValue'):
#-------------------------------------------------------------------------------

def class_of ( object ):
    if type( object ) is StringType:
        name = object
    else:
        name = object.__class__.__name__
    if name[:1].lower() in 'aeiou':
        return 'an ' + name
    return 'a ' + name
