#-------------------------------------------------------------------------------
#
#  Define a 'traits' mix-in class that allows other classes to easily define
#  'type-checked' and/or 'delegated' traits for their instances.
#
#  Note: A 'trait' is a synonym for 'property', but is used instead of the
#  word 'property' to differentiate it from the Python language 'property'
#  feature.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  (c) Copyright 2002 by Enthought, Inc.
#
#-------------------------------------------------------------------------------
#
#  To use the 'HasTraits' class, you must do two things:
#
#  1) Make 'HasTraits' one of the base classes for the class which is to have
#     traits.
#
#  2) Define a '__traits__' class attribute for the class which is a
#     dictionary whose keys are the desired trait names, and whose values
#     are one of the following:
#     - default_value
#     - ( default_value, other_value_2, other_value_3, ... )
#     - Trait( ... )
#     - TraitDelegate( ... )
#
#-------------------------------------------------------------------------------
#
#  The following is an example of a class which defines several traits:
#
#  class MyClass ( other_class, traits.HasTraits ):
#
#     __traits__ = {
#         'an_int':    -1,
#         'color':     [ 'red', 'blue', 'yellow' ],
#         'foo':       Trait( None, complex_test ),
#         'sine':      Trait( 0.0, traits.TraitRange( -1.0, 1.0 ) ),
#         'window':    [ None, wx.wxWindow ],
#         'anything':  Trait( 'nothing', traits.AnyValue ),
#         'combo':     [ 1, 'hello', wx.wxWindow ],
#         'size':      Trait( 'small', TraitPrefixList( [
#                         'small', 'medium', 'large' ] ) )
#     }
#
#     def combo_changed ( self, old_value, new_value ):
#         ... rest of notifier definition
#
#     ... rest of class definition...
#
#  'MyClass' defines these traits:
#    - an_int:   Must be an integer, and has a default value of -1.
#    - color:    Must be one of the strings: 'red', 'blue', 'yellow', and has
#                a default value of 'red'.
#    - foo:      Must have a valid value accepted by the 'complex_test'
#                function, and has a default value of 'None'.
#    - sine:     Must be a float in the range from -1.0 to 1.0, and has a
#                default value of 0.0.
#    - window:   Must be an instance of the wx.wxWindow class or None, and
#                has a default value of 'None'.
#    - anything: Can have any type of value. Its default value is the string
#                'nothing'
#    - combo:    Must be either the number 1, the string 'hello', or an
#                instance of the wx.wxWindow class. The default value is 1.
#                The 'combo_changed' method will be called each time the
#                trait is successfully assigned a new value.
#    - size:     Must be one of the strings 'small', 'medium', or 'large' (or
#                any non-ambiguous prefix, such as 's', 'smal', 'me', ...).
#                The default value is 'small'. If the user assigns an
#                abbreviation, the full value will be assigned to the trait.
#
#  There are many more types and variations on traits available. Refer to
#  the documentation for more details.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from types import NoneType,     IntType,     LongType,   FloatType, ComplexType,\
                  StringType,   UnicodeType, ListType,   TupleType, DictType,   \
                  FunctionType, ClassType,   MethodType, InstanceType,          \
                  TypeType

import exceptions
import sys

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Boolean values:
TRUE  = 1
FALSE = 0

SequenceTypes = ( ListType, TupleType )

ConstantTypes = ( NoneType, IntType, LongType, FloatType, ComplexType,
                  StringType, UnicodeType )

StringTypes   = ( StringType, UnicodeType, IntType, LongType, FloatType,
                  ComplexType )

NumericFuncs  = { IntType:     ( int,     'an integer' ),
                  LongType:    ( long,    'a long integer' ),
                  FloatType:   ( float,   'a floating point number' ),
                  ComplexType: ( complex, 'a complex number' ) }

PythonTypes   = ( StringType,   UnicodeType,  IntType,    LongType,
                  FloatType,    ComplexType,  ListType,   TupleType,
                  DictType,     FunctionType, MethodType, ClassType,
                  InstanceType, TypeType,     NoneType )

CallableTypes = ( FunctionType, MethodType )

EmptyDict     = {}

TraitNotifier = '__trait_notifier__'

#-------------------------------------------------------------------------------
#  Define a special string coercion function:
#-------------------------------------------------------------------------------

def strx ( arg ):
    if type( arg ) in StringTypes:
        return str( arg )
    raise TypeError

#-------------------------------------------------------------------------------
#  Return a string containing the class name of an object with the correct
#  article (a or an) preceding it (e.g. 'an Image', 'a PlotValue'):
#-------------------------------------------------------------------------------

def class_of ( object ):
    if type( object ) == StringType:
        name = object
    else:
        name = object.__class__.__name__
    if name[:1].lower() in 'aeiou':
        return 'an ' + name
    return 'a ' + name

#-------------------------------------------------------------------------------
#  Define a mapping from types to coercion functions:
#-------------------------------------------------------------------------------

CoercableFuncs = { IntType:     int,
                   LongType:    long,
                   FloatType:   float,
                   ComplexType: complex,
                   StringType:  strx,
                   UnicodeType: unicode }

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

    if trait_editors_module is None:
        if trait_editors_module_name is None:
            try:
                __import__( 'wxPython' )
                trait_editors_module_name = 'traits.wxtrait_sheet'
            except ImportError:
                try:
                    __import__( 'Tkinter' )
                    trait_editors_module_name = 'traits.tktrait_sheet'
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
#  'UndefinedObject' class:
#-------------------------------------------------------------------------------

class UndefinedObject:

    def __repr__ ( self ):
        return '<undefined>'

#-------------------------------------------------------------------------------
#  'TraitError' class:
#-------------------------------------------------------------------------------

class TraitError ( exceptions.Exception ):

    def __init__ ( self, args = None, name = None, info = None, value = None ):
        if name is None:
            self.args = args
        else:
            # Save the information, in case the 'args' object is not the correct
            # one, and we need to regenerate the message later:
            self.name  = name
            self.info  = info
            self.value = value
            self.desc  = None
            self.set_desc( None, args )

    def set_desc ( self, desc, object = None ):
        if not hasattr( self, 'desc' ):
            return
        if desc is not None:
            self.desc = desc
        if object is not None:
            self.object = object
        if self.desc is None:
            extra = ''
        else:
            extra = ' specifies %s and' % self.desc
        self.args = ( "The '%s' trait of %s instance%s must be %s, "
                      "but a value of %s was specified." % (
                      self.name, class_of( self.object ), extra,
                      self.info, self.value ) )

#-------------------------------------------------------------------------------
#  'DelegationError' class:
#-------------------------------------------------------------------------------

class DelegationError ( TraitError ):

    def __init__ ( self, args ):
        self.args = args

#-------------------------------------------------------------------------------
#  'Trait' class:
#-------------------------------------------------------------------------------

class Trait:

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, default_value, *value_type, **keywords ):
        setter     = None
        isDelegate = isinstance( default_value, TraitDelegate )
        if isDelegate:
            getter        = default_value
            default_value = Undefined
        else:
            getter = self
        if (len( value_type ) == 0) and (type( default_value ) in SequenceTypes):
            default_value, value_type = default_value[0], default_value
        if len( value_type ) == 0:
            if isDelegate:
                setter = getter
            elif isinstance( default_value, Trait ):
                dic = default_value.__dict__.copy()
                dic.update( keywords )
                keywords = dic
            elif isinstance( default_value, TraitHandler ):
                setter        = default_value
                default_value = None
            else:
                typeValue = type( default_value )
                if typeValue == ClassType:
                    setter        = TraitInstance( default_value )
                    default_value = None
                elif typeValue == InstanceType:
                    setter = TraitInstance( default_value.__class__ )
                else:
                    setter = TraitType( typeValue )
        else:
            enum  = []
            other = []
            map   = {}
            self.do_list( value_type, enum, map, other )
            if len( enum ) > 0:
                other.append( TraitEnum( enum ) )
            if len( map ) > 0:
                other.append( TraitMap( map ) )
            if len( other ) == 0:
                setter = TraitHandler()
            elif len( other ) == 1:
                setter = other[0]
                if isinstance( setter, TraitDelegate ):
                    getter = setter
                elif isinstance( setter, Trait ):
                    dic = setter.__dict__.copy()
                    dic.update( keywords )
                    dic[ 'default_value' ] = default_value
                    keywords = dic
            else:
                setter = TraitComplex( other )

        # Save the results as traits:
        self.default_value = default_value
        self.getter        = getter
        self.setter        = setter

        # Copy all the keyword values into the trait definition:
        for name, value in keywords.items():
            setattr( self, name, value )

    #----------------------------------------------------------------------------
    #  Return 'None' for undefined traits of the Trait object:
    #----------------------------------------------------------------------------

    def __getattr__ ( self, name ):
        if name[0:2] == '__':
            raise AttributeError, "%s instance has no attribute '%s'" % (
                                  self.__class__.__name__, name )
        return None

    #----------------------------------------------------------------------------
    #  Determine the correct TraitHandler for each item in a list:
    #----------------------------------------------------------------------------

    def do_list ( self, list, enum, map, other ):
        for item in list:
            if item in PythonTypes:
                other.append( TraitType( item ) )
            else:
                typeItem = type( item )
                if typeItem in ConstantTypes:
                    enum.append( item )
                elif typeItem in SequenceTypes:
                    self.do_list( item, enum, map, other )
                elif typeItem == DictType:
                    map.update( item )
                elif typeItem == ClassType:
                    other.append( TraitInstance( item ) )
                elif typeItem in CallableTypes:
                    other.append( TraitFunction( item ) )
                elif (isinstance( item, TraitHandler )  or
                      isinstance( item, TraitDelegate ) or
                      isinstance( item, Trait )):
                    other.append( item )
                else:
                    other.append( TraitHandler() )

    #----------------------------------------------------------------------------
    #  Default handler for initializing trait values:
    #----------------------------------------------------------------------------

    def getattr ( self, object, name, value ):
        return object.__setattr__( name, value )

    #----------------------------------------------------------------------------
    #  Return the trait editor associated with this trait:
    #----------------------------------------------------------------------------

    def get_editor ( self ):
        if self.editor is None:
            try:
                self.editor = self.setter.get_editor( self )
            except:
                pass
        return self.editor

#-------------------------------------------------------------------------------
#  'PythonTrait' class:
#-------------------------------------------------------------------------------

class PythonTrait ( Trait ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self ):
        self.default_value = None
        self.getter        = self
        self.setter        = self

    #----------------------------------------------------------------------------
    #  Default handler for initializing trait values:
    #----------------------------------------------------------------------------

    def getattr ( self, object, name, value ):
        raise AttributeError, "%s instance has no attribute '%s'" % (
                              object.__class__.__name__, name )

    #----------------------------------------------------------------------------
    #  Validate the value for a particular object delegate's trait:
    #----------------------------------------------------------------------------

    def setattr ( self, object, name, value, default ):
        object.__dict__[ name ] = value
        return value

#-------------------------------------------------------------------------------
#  'MappedTrait' class:
#-------------------------------------------------------------------------------

class MappedTrait ( Trait ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self ):
        self.default_value = None
        self.getter        = self
        self.setter        = AnyValue

    #----------------------------------------------------------------------------
    #  Default handler for initializing trait values:
    #----------------------------------------------------------------------------

    def getattr ( self, object, name, value ):
        getattr( object, name[:-1] )
        return object.__dict__[ name ]

#-------------------------------------------------------------------------------
#  'TraitDelegate' class:
#-------------------------------------------------------------------------------

class TraitDelegate:

    def __init__ ( self, delegate = None, mutate_or_prefix = FALSE ):
        self._delegate = delegate
        self.delegate  = self.get_delegate
        self.getattr   = self.getattr_method
        self.setattr   = self.setattr # Performance hack!
        self.prefix    = ''
        self.mutate    = FALSE
        if type( mutate_or_prefix ) == StringType:
            self.prefix = mutate_or_prefix
            self.name   = self.replace_name
            if mutate_or_prefix[-1:] == '*':
                self.prefix = mutate_or_prefix[:-1]
                self.name   = self.prefix_name
                if mutate_or_prefix == '*':
                    self.name = self.classprefix_name
        else:
            self.mutate = mutate_or_prefix

        # Handle the special case of a delegate that disallows everything:
        if delegate is None:
            self.getattr = self.getattr_locked
            self.setattr = self.setattr_locked

    #----------------------------------------------------------------------------
    #  Return the delegate for a specified object:
    #----------------------------------------------------------------------------

    def get_delegate ( self, object ):
        if hasattr( object, self._delegate ):
            delegate = getattr( object, self._delegate )
            if type( delegate ) == MethodType:
                self.getattr  = self.getattr_method
                self.delegate = delegate
                return delegate( object )
        self.getattr  = getattr( self, 'getattr_trait_' + self.name.__name__ )
        self.delegate = self._getattr
        return self._getattr( object )

    #----------------------------------------------------------------------------
    #  Return a delegate stored as an object trait:
    #----------------------------------------------------------------------------

    def _getattr ( self, object ):
        return getattr( object, self._delegate )

    #----------------------------------------------------------------------------
    #  Define variations on creating the trait name to delegate to:
    #----------------------------------------------------------------------------

    def name ( self, object, name ):
        return name

    def replace_name ( self, object, name ):
        return self.prefix

    def prefix_name ( self, object, name ):
        return self.prefix + name

    def classprefix_name ( self, object, name ):
        return object.__prefix__ + name

    #----------------------------------------------------------------------------
    #  Return an object delegate's value for a specified trait:
    #  (for the case where the delegate is a trait)
    #
    #  Note: There is one variant for each type of delegate 'name'. They are
    #        expanded out this way to eliminate a method call to 'self.name'
    #        and thus optimize the delegation 'getattr' code path.
    #----------------------------------------------------------------------------

    def getattr_trait_name ( self, object, name, value ):
        try:
            return getattr( getattr( object, self._delegate ), name )
        except:
            return self.getattr_exception( object, name, value )

    def getattr_trait_replace_name ( self, object, name, value ):
        try:
            return getattr( getattr( object, self._delegate ), self.prefix )
        except:
            return self.getattr_exception( object, name, value )

    def getattr_trait_prefix_name ( self, object, name, value ):
        try:
            return getattr( getattr( object, self._delegate ), self.prefix + name )
        except:
            return self.getattr_exception( object, name, value )

    def getattr_trait_classprefix_name ( self, object, name, value ):
        try:
            return getattr( getattr( object, self._delegate ),
                            object.__prefix__ + name )
        except:
            return self.getattr_exception( object, name, value )

    #----------------------------------------------------------------------------
    #  Common exception handler for a trait delegate:
    #----------------------------------------------------------------------------

    def getattr_exception ( self, object, name, value ):
        if getattr( object, self._delegate ) is None:
            if value != Undefined:
                return value
            raise DelegationError, (
                     "Attempted to get the '%s' trait of %s instance, "
                     "but its '%s' delegate is not defined." % (
                     name, class_of( object ), self._delegate ) )
        else:
            raise DelegationError, (
                     "Attempted to get the '%s' trait of %s instance, "
                     "but its '%s' delegate does not have the trait defined."
                     % ( name, class_of( object ), self._delegate ) )

    #----------------------------------------------------------------------------
    #  Return an object delegate's value for a specified trait:
    #  (for the case where the delegate is a method)
    #----------------------------------------------------------------------------

    def getattr_method ( self, object, name, value ):
        delegate = self.delegate( object )
        try:
            return getattr( delegate, self.name( object, name ) )
        except:
            if delegate is None:
                if value != Undefined:
                    return value
                raise DelegationError, (
                         "Attempted to get the '%s' trait of %s instance, "
                         "but its '%s' delegate is not defined." % (
                         name, class_of( object ), self._delegate ) )
            else:
                raise DelegationError, (
                         "Attempted to get the '%s' trait of %s instance, "
                         "but its '%s' delegate does not have the trait defined."
                         % ( name, class_of( object ), self._delegate ) )

    #----------------------------------------------------------------------------
    #  Throw an exception when a 'locked' trait is referenced:
    #----------------------------------------------------------------------------

    def getattr_locked ( self, object, name, value ):
        raise AttributeError, "%s instance has no attribute '%s'" % (
                              object.__class__.__name__, name )

    #-----------------------------------------------------------------------------
    #  Validate the value for a particular object delegate's trait:
    #-----------------------------------------------------------------------------

    def setattr ( self, object, name, value, default ):
        try:
            delegate_name = self.name( object, name )
            delegate      = self.delegate( object )
            while TRUE:
                handler = delegate._trait( delegate_name ).setter
                if not isinstance( handler, TraitDelegate ):
                    break
                delegate = handler.delegate( delegate )
        except AttributeError:
            if delegate is None:
                raise DelegationError, (
                         "Attempted to set the '%s' trait of %s instance, "
                         "but its '%s' delegate is not defined." % (
                         name, class_of( object ), self._delegate ) )
            else:
                raise DelegationError, (
                         "Attempted to set the '%s' trait of %s instance, but "
                         "its '%s' delegate does not have any traits defined." %
                         ( name, class_of( object ), self._delegate ) )
        except:
            raise DelegationError, (
                     "Attempted to set the '%s' trait of %s instance, but its "
                     "'%s' delegate does not have a trait with that name." % (
                     name, class_of( object ), self._delegate ) )

        if self.mutate:
            # Modify the delegate object:
            try:
                return handler.setattr( delegate, delegate_name, value, default )
            except TraitError, excp:
                # The exception is for the wrong object. Fix it, then pass it on:
                excp.set_desc( delegate._trait( delegate_name ).desc, object )
                raise excp
        else:
            # Modify the original object:
            try:
                return handler.setattr( object, name, value, default )
            except TraitError, excp:
                # Add the trait description to the exception:
                excp.set_desc( delegate._trait( delegate_name ).desc )
                raise excp

    #----------------------------------------------------------------------------
    #  Throw an exception when a 'locked' trait is set:
    #----------------------------------------------------------------------------

    def setattr_locked ( self, object, name, value, default ):
        raise TraitError, "%s instance does not have a '%s' trait" % (
                          class_of( object ).capitalize(), name )

    #----------------------------------------------------------------------------
    #  Get the base trait for a particular object delegate's trait:
    #----------------------------------------------------------------------------

    def base_trait ( self, object, name ):
        try:
            delegate = self.delegate( object )
            while TRUE:
                trait   = delegate._trait( self.name( object, name ) )
                handler = trait.setter
                if not isinstance( handler, TraitDelegate ):
                    break
                delegate = handler.delegate( delegate )
            return trait
        except AttributeError:
            if delegate is None:
                raise DelegationError, (
                         "Attempted to get the underlying '%s' trait of a "
                         "%s instance, but its '%s' delegate is not defined." % (
                         name, object.__class__.__name__, self._delegate ) )
            else:
                raise DelegationError, (
                         "Attempted to get the underlying '%s' trait of a "
                         "%s instance, but its '%s' delegate does not have any "
                         "traits defined." %
                         ( name, object.__class__.__name__, self._delegate ) )
        except:
            raise DelegationError, (
                     "Attempted to get the underlying '%s' trait of a "
                     "%s instance, but its '%s' delegate does not have a "
                     "trait with that name." % (
                     name, object.__class__.__name__, self._delegate ) )

#-------------------------------------------------------------------------------
#  'TraitDelegateSynched' class:
#-------------------------------------------------------------------------------

class TraitDelegateSynched ( TraitDelegate ):

    def __init__ ( self, delegate = None, mutate_or_prefix = FALSE ):
        self.original_setattr = self.setattr
        TraitDelegate.__init__( self, delegate, mutate_or_prefix )
        if delegate is not None:
            self.getattr = self.getattr_init
            self.setattr = self.setattr_init

    #----------------------------------------------------------------------------
    #  Decide which type of attribute getter to use the very first time this
    #  trait is used to get the trait's value:
    #----------------------------------------------------------------------------

    def getattr_init ( self, object, name, value ):
        self.delegate_init( object )
        return self.getattr( object, name, value )

    #----------------------------------------------------------------------------
    #  Validate the value for a particular object delegate's trait:
    #----------------------------------------------------------------------------

    def setattr_init ( self, object, name, value, default ):
        self.delegate_init( object )
        self.setattr( object, name, value, default )

    #----------------------------------------------------------------------------
    #  Bind the getattr/settatr methods the first time delegation occurs:
    #----------------------------------------------------------------------------

    def delegate_init ( self, object ):
        self.get_delegate( object )
        self.setattr = self.original_setattr
        if self.getattr != self.getattr_method:
            self.getattr = self.getattr_synched
            if not self.mutate:
                self.setattr = self.setattr_synched

    #----------------------------------------------------------------------------
    #  Return an object delegate's value for a specified trait:
    #  (for the case where the delegate is an attribute)
    #----------------------------------------------------------------------------

    def getattr_synched ( self, object, name, value ):
        delegate = self.delegate( object )
        try:
            delegate_name = self.name( object, name )
            value         = getattr( delegate, delegate_name )
            if not isinstance( delegate, HasTraits ):
                return value
            dict         = object.__dict__
            dict[ name ] = value
            delegates    = dict.get( '__delegates__', None )
            if delegates is None:
                dict[ '__delegates__' ] = delegates = {}
            handlers = delegates.get( self._delegate, None )
            if handlers is None:
                delegates[ self._delegate ] = handlers = {}
                object.on_trait_change( self.delegate_changed, self._delegate )
            handler = lambda v: object._set_trait_value( object, name, v, None )
            handlers[ name ] = ( handler, delegate_name )
            delegate.on_trait_change( handler, delegate_name )
            return value
        except:
            return self.getattr_exception( object, name, value )

    #----------------------------------------------------------------------------
    #  Handle one of an object's delegates being assigned a new value:
    #----------------------------------------------------------------------------

    def delegate_changed ( self, object, trait_name, old, new ):
        handlers = object.__delegates__[ trait_name ]
        for name, info in handlers.items():
            handler, delegate_name = info
            old.on_trait_change( handler, delegate_name, TRUE )
            new.on_trait_change( handler, delegate_name )
            object._set_trait_value( object, name,
                                     getattr( new, delegate_name ), None )

    #----------------------------------------------------------------------------
    #  Validate the value for a particular object delegate's trait:
    #----------------------------------------------------------------------------

    def setattr_synched ( self, object, name, value, default ):
        TraitDelegate.setattr( self, object, name, value, default )
        handlers = object.__dict__.get( '__delegates__', EmptyDict ).get(
                                        self._delegate, EmptyDict )
        info     = handlers.get( name, None )
        if info is not None:
            handler, delegate_name = info
            self.delegate( object ).on_trait_change( handler, delegate_name,
                                                     TRUE )
            del handlers[ name ]

#-------------------------------------------------------------------------------
#  'TraitHandler' class (base class for all trait handlers):
#-------------------------------------------------------------------------------

class TraitHandler:

    def setattr ( self, object, name, value, default ):
        raise TraitError, (
                 "The '%s' trait of %s instance has an unknown type. "
                 "Contact the developer to correct the problem." % (
                 name, class_of( object ) ) )

    def error ( self, object, name, value ):
        raise TraitError, ( object, name, self.info(), value )

    def info ( self ):
        return 'a legal value'

    def repr ( self, value ):
        if type( value ) == InstanceType:
            return 'class '  + value.__class__.__name__
        return repr( value )

    def is_mapped ( self ):
        return FALSE

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorText()

#-------------------------------------------------------------------------------
#  'TraitRange' class:
#-------------------------------------------------------------------------------

class TraitRange ( TraitHandler ):

    def __init__ ( self, low, high ):
        self.range( low, high )
        self.coerce, self.type_desc = NumericFuncs[ type( low ) ]

    def range ( self, low = None, high = None ):
        if low is not None:
            self.low  = low
        if high is not None:
            self.high = high
        return ( self.low, self.high )

    def setattr ( self, object, name, value, default ):
        try:
            cvalue = self.coerce( value )
            if (cvalue >= self.low) and (cvalue <= self.high):
                return object._set_trait_value( object, name, cvalue, default )
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return '%s in the range from %s to %s' % (
               self.type_desc, self.low, self.high )

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorRange(
                  self, cols = trait.cols or 3 )

#--------------------------------------------------------------------------------
#  'TraitType' class:
#--------------------------------------------------------------------------------

class TraitType ( TraitHandler ):

    def __init__ ( self, aType ):
        if type( aType ) != TypeType:
            aType = type( aType )
        self.aType = aType
        try:
            self.coerce = CoercableFuncs[ aType ]
        except:
            self.coerce = self.identity

    def setattr ( self, object, name, value, default ):
        try:
            return object._set_trait_value( object, name, self.coerce( value ),
                                            default )
        except:
            pass
        if type( value ) == InstanceType:
            kind = class_of( value )
        else:
            kind = repr( value )
        self.error( object, name, '%s (i.e. %s)' % (
                         str( type( value ) )[1:-1], kind ) )

    def info ( self ):
        return 'of %s' % str( self.aType )[1:-1]

    def identity ( self, value ):
        if type( value ) == self.aType:
            return value
        raise TraitError

#--------------------------------------------------------------------------------
#  'TraitInstance' class:
#--------------------------------------------------------------------------------

class TraitInstance ( TraitHandler ):

    def __init__ ( self, aClass ):
        if type( aClass ) == InstanceType:
            aClass = aClass.__class__
        self.aClass = aClass

    def setattr ( self, object, name, value, default ):
        if isinstance( value, self.aClass ):
            return object._set_trait_value( object, name, value, default )

        kind = type( value )
        if kind == InstanceType:
            msg = 'class %s' % value.__class__.__name__
        else:
            msg = '%s (i.e. %s)' % ( str( kind )[1:-1], repr( value ) )
        self.error( object, name, msg )

    def info ( self ):
        return class_of( self.aClass.__name__ )

#--------------------------------------------------------------------------------
#  'TraitFunction' class:
#--------------------------------------------------------------------------------

class TraitFunction ( TraitHandler ):

    def __init__ ( self, aFunc ):
        self.aFunc = aFunc

    def setattr ( self, object, name, value, default ):
        try:
            return object._set_trait_value( object, name,
                          self.aFunc( object, name, value ), default )
        except TraitError:
            pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        try:
            return self.aFunc.info
        except:
            if self.aFunc.__doc__:
                return self.aFunc.__doc__
            return 'a legal value'

#--------------------------------------------------------------------------------
#  'TraitEnum' class:
#--------------------------------------------------------------------------------

class TraitEnum ( TraitHandler ):

    def __init__ ( self, *values ):
        if (len( values ) == 1) and (type( values[0] ) in SequenceTypes):
            values = values[0]
        self.values = values

    def setattr ( self, object, name, value, default ):
        if value in self.values:
            return object._set_trait_value( object, name, value, default )
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return ' or '.join( map( lambda x: repr( x ), self.values ) )

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorEnum( self, cols = trait.cols or 3  )

#-------------------------------------------------------------------------------
#  'TraitPrefixList' class:
#-------------------------------------------------------------------------------

class TraitPrefixList ( TraitHandler ):

    def __init__ ( self, *values ):
        if (len( values ) == 1) and (type( values[0] ) in SequenceTypes):
            values = values[0]
        self.values = values
        self.map    = map = {}
        for key in values:
            map[ key ] = key

    def setattr ( self, object, name, value, default ):
        if not self.map.has_key( value ):
            match = None
            try:
                n = len( value )
                for key in self.values:
                    if value == key[:n]:
                        if match is not None:
                            match = None
                            break
                        match = key
            except:
                pass
            if match is None:
                self.error( object, name, self.repr( value ) )
            self.map[ value ] = match
        return object._set_trait_value( object, name, self.map[ value ],
                                        default )

    def info ( self ):
        return ' or '.join( map( repr, self.values ) ) + ' (or any unique prefix)'

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorEnum( self, cols = trait.cols or 3  )

#-------------------------------------------------------------------------------
#  'TraitMap' class:
#-------------------------------------------------------------------------------

class TraitMap ( TraitHandler ):

    def __init__ ( self, map ):
        self.map = map

    def setattr ( self, object, name, value, default ):
        if self.map.has_key( value ):
            old_value = object.__dict__.get( name, Undefined )
            object._set_trait_value( object, name, value, default )
            if value != old_value:
                setattr( object, name + '_', self.map[ value ] )
            return value
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        keys = map( lambda x: repr( x ), self.map.keys() )
        keys.sort()
        return ' or '.join( keys )

    def reverse ( self ):
        if not hasattr( self, '_reverse' ):
            self._reverse = TraitMapReverse( self.map )
        return self._reverse

    def is_mapped ( self ):
        return TRUE

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorEnum( self, cols = trait.cols or 3  )

#-------------------------------------------------------------------------------
#  'TraitPrefixMap' class:
#-------------------------------------------------------------------------------

class TraitPrefixMap ( TraitMap ):

    def __init__ ( self, map ):
        self.map  = map
        self._map = _map = {}
        for key in map.keys():
            _map[ key ] = key

    def setattr ( self, object, name, value, default ):
        if not self._map.has_key( value ):
            match = None
            try:
                n = len( value )
                for key in self.map.keys():
                    if value == key[:n]:
                        if match is not None:
                            match = None
                            break
                        match = key
            except:
                pass
            if match is None:
                self.error( object, name, self.repr( value ) )
            self._map[ value ] = match
        value     = self._map[ value ]
        old_value = object.__dict__.get( name, Undefined )
        object._set_trait_value( object, name, value, default )
        if value != old_value:
            setattr( object, name + '_', self.map[ value ] )
        return value

    def info ( self ):
        return TraitMap.info( self ) + ' (or any unique prefix)'

#-------------------------------------------------------------------------------
#  'TraitMapReverse' class:
#-------------------------------------------------------------------------------

class TraitMapReverse ( TraitHandler ):

    def __init__ ( self, map ):
        self.map = rmap = {}
        for key, value in map.items():
            if not rmap.has_key( value ):
                rmap[ value ] = [ key ]
            else:
                rmap[ value ].append( key )

    def setattr ( self, object, name, value, default ):
        if self.map.has_key( value ):
            object._set_trait_value( object, name, value, default )
            if object.__dict__.get( name[:-1], Undefined ) not in self.map[value]:
                setattr( object, name[:-1], self.map[ value ][0] )
            return value
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        keys = map( lambda x: repr( x ), self.map.keys() )
        keys.sort()
        return ' or '.join( keys )

#-------------------------------------------------------------------------------
#  'TraitComplex' class:
#-------------------------------------------------------------------------------

class TraitComplex ( TraitHandler ):

    def __init__ ( self, *handlers ):
        if (len( handlers ) == 1) and (type( handlers[0] ) in SequenceTypes):
            handlers = handlers[0]
        self.handlers = []
        for handler in handlers:
            if isinstance( handler, Trait ):
                handler = handler.setter
            if handler.is_mapped():
                if not hasattr( self, 'map_handlers' ):
                    self.map_handlers = []
                    self.setattr      = self.mapped_setattr
                    self.info         = self.mapped_info
                    self.is_mapped    = self.mapped_is_mapped
                self.map_handlers.append( handler )
            else:
                self.handlers.append( handler )

    def setattr ( self, object, name, value, default ):
        for handler in self.handlers:
            try:
                return handler.setattr( object, name, value, default )
            except TraitError:
                pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return ' or '.join( map( lambda x: x.info(), self.handlers ) )

    def mapped_setattr ( self, object, name, value, default ):
        for handler in self.handlers:
            try:
                result = handler.setattr( object, name, value, default )
                setattr( object, name + '_', result )
                return result
            except:
                pass
        for handler in self.map_handlers:
            try:
                return handler.setattr( object, name, value, default )
            except:
                pass
        self.error( object, name, self.repr( value ) )

    def mapped_info ( self ):
        return ' or '.join( map( lambda x: x.info(),
                                 self.handlers + self.map_handlers ) )

    def mapped_is_mapped ( self ):
        return TRUE

    def get_editor ( self, trait ):
        the_editors = map( lambda x: x.get_editor( trait ), self.handlers )
        if self.is_mapped():
            the_editors.extend( map( lambda x: x.get_editor( trait ),
                                     self.map_handlers ) )
        text_editor = trait_editors().TraitEditorText
        count       = 0
        editors     = []
        for editor in the_editors:
            if issubclass( text_editor, editor.__class__ ):
                count += 1
                if count > 1:
                    continue
            editors.append( editor )
        editors.sort( lambda x, y: cmp( id( x.__class__ ), id( y.__class__ ) ) )
        return trait_editors().TraitEditorComplex( editors )

#-------------------------------------------------------------------------------
#  'TraitAny' class:
#-------------------------------------------------------------------------------

class TraitAny ( TraitHandler ):

    def setattr ( self, object, name, value, default ):
        return object._set_trait_value( object, name, value, default )

    def info ( self ):
        return 'any value'

#-------------------------------------------------------------------------------
#  'TraitReadOnly' class:
#-------------------------------------------------------------------------------

class TraitReadOnly ( TraitHandler ):

    def setattr ( self, object, name, value, default ):
        no_key = (not object.__dict__.has_key( name ))
        if no_key:
            return object._set_trait_value( object, name, value, default )
        if ((value != Undefined) and
            (no_key or (getattr( object, name ) == Undefined))):
            return object._set_trait_value( object, name, value, default )
        raise TraitError, (
                 "The '%s' trait of %s instance is read only." % (
                 name, class_of( object ) ) )

    def info ( self ):
        return 'any value'

#-------------------------------------------------------------------------------
#  Create singleton-like instances (so users don't have to):
#-------------------------------------------------------------------------------

Undefined          = UndefinedObject() # Undefined trait name and/or value
AnyValue           = TraitAny()        # Allow any value for a trait
Disallow           = TraitDelegate()   # Disallow getting and/or setting a trait
ReadOnly           = Trait( Undefined, TraitReadOnly() ) # Read-only trait
DefaultPythonTrait = PythonTrait()     # Trait w/standard Python semantics

# This special trait is only for internal use:
TheMappedTrait = MappedTrait()  # Create the (unique) mapped trait

#-------------------------------------------------------------------------------
#  'TraitChangeNotifier' class:
#-------------------------------------------------------------------------------

class TraitChangeNotifier:

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, object, name, anytrait_notifiers ):
        self.trait_notifiers    = []
        self.anytrait_notifiers = anytrait_notifiers
        cls                     = object.__class__
        self.trait_notifier     = getattr( cls, name + '_changed',  None )
        self.anytrait_notifier  = getattr( cls, 'anytrait_changed', None )

    #----------------------------------------------------------------------------
    #  Add a new handler:
    #----------------------------------------------------------------------------

    def add ( self, handler ):
        trait_notifiers = self.trait_notifiers
        for cur_notifier in trait_notifiers:
            if handler == cur_notifier.handler:
                return 0
        trait_notifiers.append( TraitChangeNotifyWrapper( handler ) )
        return 1

    #----------------------------------------------------------------------------
    #  Remove an existing handler:
    #----------------------------------------------------------------------------

    def remove ( self, handler ):
        trait_notifiers = self.trait_notifiers
        for cur_notifier in trait_notifiers:
            if handler == cur_notifier.handler:
                trait_notifiers.remove( cur_notifier )
                return 1
        return 0

    #----------------------------------------------------------------------------
    #  Set a new value on the object:
    #----------------------------------------------------------------------------

    def __call__ ( self, object, name, value, default ):
        obj_dict  = object.__dict__
        old_value = obj_dict.get( name, default )
        try:
            if old_value != value:
                obj_dict[ name ] = value
                if self.trait_notifier is not None:
                    self.trait_notifier( object, old_value, value )
                for notifier in self.trait_notifiers:
                    notifier( object, name, old_value, value )
                if self.anytrait_notifier is not None:
                    self.anytrait_notifier( object, name, old_value, value )
                for notifier in self.anytrait_notifiers:
                    notifier( object, name, old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            if self.trait_notifier is not None:
                self.trait_notifier( object, old_value, value )
            for notifier in self.trait_notifiers:
                notifier( object, name, old_value, value )
            if self.anytrait_notifier is not None:
                self.anytrait_notifier( object, name, old_value, value )
            for notifier in self.anytrait_notifiers:
                notifier( object, name, old_value, value )
            return value

#-------------------------------------------------------------------------------
#  'TraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class TraitChangeNotifyWrapper:

    def __init__ ( self, handler ):
        self.handler = handler
        adjust       = 0
        func         = handler
        if type( handler ) == MethodType:
            func   = handler.im_func
            adjust = 1
        self.__call__ = getattr( self, 'call_%d' %
                                 (func.func_code.co_argcount - adjust) )

    def call_0 ( self, object, trait_name, old, new ):
        self.handler()

    def call_1 ( self, object, trait_name, old, new ):
        self.handler( new )

    def call_2 ( self, object, trait_name, old, new ):
        self.handler( trait_name, new )

    def call_3 ( self, object, trait_name, old, new ):
        self.handler( object, trait_name, new )

    def call_4 ( self, object, trait_name, old, new ):
        self.handler( object, trait_name, old, new )

#-------------------------------------------------------------------------------
#  'InstanceTraitNotifier' class:
#-------------------------------------------------------------------------------

class InstanceTraitNotifier:

    def __init__ ( self, object ):
        self.object             = object
        self.active_notifiers   = 0
        self.notifiers          = {}
        self.anytrait_notifiers = []

    def _set_trait_value ( self, object, name, value, default ):
        return self.notifiers.get( name, self.bind )(
                    object, name, value, default )

    def bind ( self, object, name, value, default ):
        if len( self.anytrait_notifiers ) == 0:
            notifier = self.object._notifier_for( name )
        else:
            notifier = TraitChangeNotifier( self.object, name,
                                            self.anytrait_notifiers )
        self.notifiers[ name ] = notifier
        return notifier( object, name, value, default )

    def add ( self, handler, name ):
        if name == 'anytrait':
            anytrait_notifiers = self.anytrait_notifiers
            if len( anytrait_notifiers ) == 0:
                notifiers = self.notifiers
                for name, notifier in notifiers.items():
                    if not isinstance( notifier, TraitChangeNotifier ):
                        notifiers[ name ] =  TraitChangeNotifier(
                                   self.object, name, anytrait_notifiers )
            anytrait_notifiers.append( TraitChangeNotifyWrapper( handler ) )
        else:
            notifier = self.notifiers.get( name, None )
            if not isinstance( notifier, TraitChangeNotifier ):
                self.notifiers[ name ] = notifier = TraitChangeNotifier(
                                     self.object, name, self.anytrait_notifiers )
            self.active_notifiers += notifier.add( handler )

    def remove ( self, handler, name ):
        if name == 'anytrait':
            anytrait_notifiers = self.anytrait_notifiers
            for notifier in anytrait_notifiers:
                if handler == notifier.handler:
                    anytrait_notifiers.remove( notifier )
                    if len( anytrait_notifiers ) == 0:
                        object = self.object
                        if self.active_notifiers == 0:
                            object._reset_trait_value()
                            delattr( object, TraitNotifier )
                            self.object = self.notifiers = None
                        else:
                            notifiers = self.notifiers
                            for name, notifier in notifiers.items():
                                if len( notifier.trait_notifiers ) == 0:
                                    notifiers[ name ] = object._notifier_for( name )
        else:
            notifiers = self.notifiers
            notifier  = notifiers.get( name, None )
            if isinstance( notifier, TraitChangeNotifier ):
                self.active_notifiers -= notifier.remove( handler )
                if ((len( notifier.trait_notifiers ) == 0) and
                    (len( self.anytrait_notifiers )  == 0)):
                    object = self.object
                    notifiers[ name ] = object._notifier_for( name )
                    if self.active_notifiers == 0:
                        object._reset_trait_value()
                        delattr( object, TraitNotifier )
                        self.object = self.notifiers = None

#-------------------------------------------------------------------------------
#  'ClassTraitNotifier' class:
#-------------------------------------------------------------------------------

def set_trait_value ( object, name, value, default ):
    object.__dict__[ name ] = value
    return value

class ClassTraitNotifier:

    def __init__ ( self, cls ):
        self.cls               = cls
        self.notifiers         = {}
        self.bind_factory      = self.no_anytrait_changed
        self.anytrait_notifier = getattr( cls, 'anytrait_changed', None )
        if self.anytrait_notifier is not None:
            self.bind_factory = self.has_anytrait_changed

    def _set_trait_value ( self, object, name, value, default ):
        return self.notifiers.get( name, self.bind )(
                    object, name, value, default )

    def bind ( self, object, name, value, default ):
        self.notifiers[ name ] = notifier = self.bind_factory( name )
        return notifier( object, name, value, default )

    def notifier_for ( self, name ):
        notifier = self.notifiers.get( name, None )
        if notifier is None:
            self.notifiers[ name ] = notifier = self.bind_factory( name )
        return notifier

    def no_anytrait_changed ( self, name ):
        notifier = getattr( self.cls, name + '_changed', None )
        if notifier is None:
            return set_trait_value
        return SpecificTraitNotifier( notifier )

    def has_anytrait_changed ( self, name ):
        notifier = getattr( self.cls, name + '_changed', None )
        if notifier is None:
            return self.anytrait_changed
        return AnyAndSpecificTraitNotifier( self.anytrait_notifier, notifier )

    def anytrait_changed ( self, object, name, value, default ):
        obj_dict  = object.__dict__
        old_value = obj_dict.get( name, default )
        try:
            if old_value != value:
                obj_dict[ name ] = value
                self.anytrait_notifier( object, name, old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            self.anytrait_notifier( object, name, old_value, value )
            return value

#-------------------------------------------------------------------------------
#  'SpecificTraitNotifier' class:
#-------------------------------------------------------------------------------

class SpecificTraitNotifier:

    def __init__ ( self, notifier ):
        self.notifier = notifier

    def __call__ ( self, object, name, value, default ):
        obj_dict  = object.__dict__
        old_value = obj_dict.get( name, default )
        try:
            if old_value != value:
                obj_dict[ name ] = value
                self.notifier( object, old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            self.notifier( object, old_value, value )
            return value

#-------------------------------------------------------------------------------
#  'AnyAndSpecificTraitNotifier' class:
#-------------------------------------------------------------------------------

class AnyAndSpecificTraitNotifier:

    def __init__ ( self, anytrait_notifier, trait_notifier ):
        self.anytrait_notifier = anytrait_notifier
        self.trait_notifier    = trait_notifier

    def __call__ ( self, object, name, value, default ):
        obj_dict  = object.__dict__
        old_value = obj_dict.get( name, default )
        try:
            if old_value != value:
                obj_dict[ name ] = value
                self.trait_notifier( object, old_value, value )
                self.anytrait_notifier( object, name, old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            self.trait_notifier( object, old_value, value )
            self.anytrait_notifier( object, name, old_value, value )
            return value

#-------------------------------------------------------------------------------
#  'HasTraits' class:
#-------------------------------------------------------------------------------

class HasTraits:

    #----------------------------------------------------------------------------
    #  Define default traits that mimic standard Python behavior:
    #----------------------------------------------------------------------------

    __traits__ = { '*': DefaultPythonTrait }

    #----------------------------------------------------------------------------
    #  Initialize the trait values of an object (optional):
    #----------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        for name, value in traits.items():
            setattr( self, name, value )

    #----------------------------------------------------------------------------
    #  Handle fetching an undefined trait or attribute:
    #
    #  Note: This code is written 'strangely' in order to optimize the case of
    #        delegated traits, which are the worst case scenario for
    #        traits. Simplifying the code may therefore have a negative
    #        performance impact on this important sub-case.
    #----------------------------------------------------------------------------

    def __getattr__ ( self, name ):
        try:
            trait = self.__traits__[ name ]
            return trait.getter.getattr( self, name, trait.default_value )
        except (AttributeError, KeyError):
            trait = self._trait( name )
            try:
                return trait.getter.getattr( self, name, trait.default_value )
            except DelegationError, excp:
                raise DelegationError, excp
            except TraitError, excp:
                raise TraitError, '%s %s' % ( str( excp )[:-1],
                         'as the default value. The trait must be assigned a '
                         'valid value before being used.' )
        except DelegationError, excp:
            raise DelegationError, excp
        except TraitError, excp:
            raise TraitError, '%s %s' % ( str( excp )[:-1],
                     'as the default value. The trait must be assigned a '
                     'valid value before being used.' )

    #----------------------------------------------------------------------------
    #  Handle setting a trait or normal attribute:
    #----------------------------------------------------------------------------

    def __setattr__ ( self, name, value ):
        try:
            trait = self.__traits__[ name ]
            return trait.setter.setattr( self, name, value, trait.default_value )
        except (AttributeError, KeyError):
            trait = self._trait( name )
            try:
                return trait.setter.setattr( self, name, value,
                                             trait.default_value )
            except TraitError, excp:
                excp.set_desc( trait.desc )
                raise TraitError, excp
        except TraitError, excp:
            excp.set_desc( trait.desc )
            raise TraitError, excp

    #----------------------------------------------------------------------------
    #  Return the string representation of an object with traits:
    #----------------------------------------------------------------------------

    def __call__ ( self, showHelp = 0 ):
        names = self._trait_names()
        if len( names ) == 0:
            return ''
        result = []
        pad    = max( map( lambda x: len( x ), names ) ) + 1
        maxval = 78 - pad
        names.sort()
        for name in names:
            try:
                value = repr( getattr( self, name ) ).replace( '\n', '\\n' )
                if len( value ) > maxval:
                    value = '%s...%s' % ( value[: (maxval - 2) / 2 ],
                                          value[ -((maxval - 3) / 2): ] )
            except:
                value = '<undefined>'
            lname = (name + ':').ljust( pad )
            if showHelp:
                result.append( '%s %s\n   The value must be %s.' % (
                      lname, value, self._base_trait( name ).setter.info() ) )
            else:
                result.append( '%s %s' % ( lname, value ) )
        print '\n'.join( result )

    #----------------------------------------------------------------------------
    #  Shortcut for setting object traits:
    #----------------------------------------------------------------------------

    def set ( self, **traits ):
        for name, value in traits.items():
            setattr( self, name, value )

    #----------------------------------------------------------------------------
    #  Reset some or all of an object's traits to their default values:
    #----------------------------------------------------------------------------

    def reset_traits ( self, *names ):
        if len( names ) == 0:
            names = self._trait_names()
        for name in names:
            try:
                delattr( self, name )
            except AttributeError:
                pass

    #----------------------------------------------------------------------------
    #  Set the object's traits based upon those of another object:
    #----------------------------------------------------------------------------

    def clone_traits ( self, other ):
        for name in self._trait_names():
            try:
                setattr( self, name, getattr( other, name ) )
            except AttributeError:
                pass

    #----------------------------------------------------------------------------
    #  Edit the object's traits:
    #----------------------------------------------------------------------------

    def edit_traits ( self, traits = None ):
        trait_editors().TraitSheetDialog( self, traits )

    #----------------------------------------------------------------------------
    #  Configure the object's traits:
    #----------------------------------------------------------------------------

    def configure_traits ( self, filename = None, edit = TRUE, traits = None ):
        if filename is not None:
            fd = None
            try:
                import cPickle
                fd = open( filename, 'rb' )
                self.clone_traits( cPickle.Unpickler( fd ).load() )
            except:
                if fd is not None:
                    fd.close()

        if edit:
            try:
                clone = self.__class__()
                clone.clone_traits( self )
            except:
                clone = None
            app = trait_editors().TraitSheetApp( self, traits )
            if (not app.save_ok) and (clone is not None):
                self.clone_traits( clone )
            elif (filename is not None) and app.save_ok:
                fd = None
                try:
                    import cPickle
                    fd = open( filename, 'wb' )
                    cPickle.Pickler( fd, TRUE ).dump( self )
                except:
                    if fd is not None:
                        fd.close()
                    return FALSE

        return TRUE

    #----------------------------------------------------------------------------
    #  Return the list of editable traits:
    #----------------------------------------------------------------------------

    def editable_traits ( self ):
        try:
            # Use the object's specified editable traits:
            return self.__editable_traits__
        except:
            # Otherwise, derive it from all of the object's traits:
            names = self._trait_names()
            names.sort()
            return names

    #----------------------------------------------------------------------------
    #  Add a new trait:
    #----------------------------------------------------------------------------

    def add_trait ( self, name, trait ):
        self.__traits__[ name ] = trait

        # If it is a trait 'class' definition, rebuild the class list:
        if name[-1:] == '*':
            self._init_class_list()

    #----------------------------------------------------------------------------
    #  Synchronize the value of two traits:
    #----------------------------------------------------------------------------

    def sync_trait ( self, trait_name, object, alias = None, mutual = TRUE ):
        if alias is None:
            alias = trait_name
        self.on_trait_change( lambda value: setattr( object, alias, value ),
                              trait_name )
        if mutual:
            object.on_trait_change(
                      lambda value: setattr( self, trait_name, value ), alias )

    #----------------------------------------------------------------------------
    #  Add/Remove a handler for a specified trait being changed:
    #
    #  If no trait_name is specified, the handler will be invoked for any trait
    #  change.
    #----------------------------------------------------------------------------

    def on_trait_change ( self, handler, trait_name = None, remove = FALSE ):
        trait_name = trait_name or 'anytrait'
        dict       = self.__dict__
        notifiers  = dict.get( TraitNotifier, None )

        # Handle a trait notifier being removed:
        if remove:
            if notifiers is not None:
                notifiers.remove( handler, trait_name )
            return

        # Handle a trait notifier being added:
        if notifiers is None:
            dict[ TraitNotifier ] = notifiers = InstanceTraitNotifier( self )
            dict[ '_set_trait_value' ] = notifiers._set_trait_value
        notifiers.add( handler, trait_name )

    #----------------------------------------------------------------------------
    #  Return the class-level notifier for a specific trait:
    #----------------------------------------------------------------------------

    def _notifier_for ( self, name ):
        return self._class_notifier().notifier_for( name )

    #-------------------------------------------------------------------------------
    #  Return the object's class-level notifier manager object:
    #-------------------------------------------------------------------------------

    def _class_notifier ( self ):
        cls_dict = self.__class__.__dict__
        notifier = cls_dict.get( TraitNotifier, None )
        if notifier is None:
            cls_dict[ TraitNotifier ] = notifier = ClassTraitNotifier(
                                                        self.__class__ )
        return notifier

    #----------------------------------------------------------------------------
    #  Set up the class-level 'set trait value' handler:
    #----------------------------------------------------------------------------

    def _reset_trait_value ( self ):
        self.__dict__[ '_set_trait_value' ] = self._class_notifier(
                                                   )._set_trait_value

    #----------------------------------------------------------------------------
    #  Change an object trait:
    #----------------------------------------------------------------------------

    def _set_trait_value ( self, object, name, value, default ):
        self._reset_trait_value()
        return self._set_trait_value( object, name, value, default )

    #----------------------------------------------------------------------------
    #  Get the list of all valid trait names:
    #----------------------------------------------------------------------------

    def _trait_names ( self ):
        return filter( lambda x: (x[-1:] != '*') and
                                 (x[-1:] != '_') and
                                 (x[0:2] != '__'),
                       self.__traits__.keys() )

    #----------------------------------------------------------------------------
    #  Get the Trait object for a specified trait:
    #----------------------------------------------------------------------------

    def _trait ( self, name ):
        # Get the information associated with the trait:
        trait = self.__traits__.get( name )
        if isinstance( trait, Trait ):
            return trait

        # If the trait does not have any information yet, get it:
        traits = self.__traits__
        if trait is None:

            # Get the list of trait 'classes':
            # Note: If the class list has not been constructed yet, then build
            # it, and also 'inherit' traits from superclasses. This is why
            # the recursive call to '_trait' is made, since the traits
            # dictionary may have been updated by inheritance.
            class_list = traits.get( '**' )
            if class_list is None:
                self._init_class_list()
                return self._trait( name )

            # Handle the special case of 'mapped' traits:
            if name[-1:] == '_':
                trait  = self._trait( name[:-1] )
                setter = trait.setter
                if isinstance( setter, TraitDelegate ):
                    traits[ name ] = trait
                    return trait
                if isinstance( setter, TraitHandler ) and setter.is_mapped():
                    try:
                        traits[ name ] = trait = Trait(
                                             setter.map.get( trait.default_value ),
                                             setter.reverse() )
                    except:
                        traits[ name ] = trait = TheMappedTrait
                    return trait

            # Find the first (i.e. longest) matching class prefix in the list:
            for cls in class_list:

                # If it does, map it to the same trait as that class:
                if cls == name[ : len( cls ) ]:
                    trait = traits[ cls + '*' ]
                    break

        # else trait is not in the right form, convert it:
        else:
            trait = Trait( trait )

        # Cache the trait information for the next time:
        traits[ name ] = trait

        # Return the trait information (a 'Trait' object):
        return trait

    #----------------------------------------------------------------------------
    #  Get the base Trait object for a specified trait:
    #----------------------------------------------------------------------------

    def _base_trait ( self, name ):
        trait = self._trait( name )
        if isinstance( trait.setter, TraitDelegate ):
            return trait.setter.base_trait( self, name )
        return trait

    #----------------------------------------------------------------------------
    #  Initialize the class list for all currently defined traits:
    #----------------------------------------------------------------------------

    def _init_class_list ( self ):

        # Get the current set of traits:
        traits = self.__traits__

        # Make sure that trait inheritance has already been performed:
        class_list = traits.get( '**' )
        if class_list is None:
            # Get all traits defined by subclasses that implement HasTraits:
            traits = {}
            self._inherit( self.__class__, traits )

            # Now replace our class traits dictionary with the updated one:
            self.__class__.__traits__ = traits

        # Initialize the class list:
        class_list = []

        # Make sure there is a definition for the 'undefined' trait:
        trait = traits.get( '*' )
        if trait is None:
            traits[ '*' ] = trait = DefaultPythonTrait
        elif not isinstance( trait, Trait ):
            traits[ '*' ] = trait = Trait( trait )

        # Always make traits that look like '__system__' traits have
        # the standard Python behavior:
        traits[ '__*' ] = DefaultPythonTrait

        # Now check every defined trait:
        for key, trait in traits.items():

            # If the trait name ends in a 'wildcard' (*), add it:
            if key[-1:] == '*':
                class_list.append( key[:-1] )
                if not isinstance( trait, Trait ):
                    traits[ key ] = Trait( trait )

        # Make sure the prefixes are sorted from longest to shortest:
        class_list.sort( lambda x, y: len( y ) - len( x ) )

        # Save the result:
        traits[ '**' ] = class_list

    #----------------------------------------------------------------------------
    #  Inherit traits from a specified class:
    #----------------------------------------------------------------------------

    def _inherit ( self, cls, traits ):
        has_traits = (cls == HasTraits)
        if not has_traits:
            for base in cls.__bases__:
                has_traits |= self._inherit( base, traits )
        if has_traits:
            traits.update( cls.__traits__ )
        return has_traits

#-------------------------------------------------------------------------------
#  'HasDynamicTraits' class:
#-------------------------------------------------------------------------------

class HasDynamicTraits ( HasTraits ):

    #----------------------------------------------------------------------------
    #  Initialize the trait values of an object (optional):
    #----------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        self.__traits__ = { '*': DefaultPythonTrait, '**': [ '' ] }
        HasTraits.__init__( self, **traits )
