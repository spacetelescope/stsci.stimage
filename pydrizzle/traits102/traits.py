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
#  Symbols defined: Trait
#                   HasTraits
#                   HasDynamicTraits
#                   Disallow
#                   ReadOnly
#                   DefaultPythonTrait
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
from __future__ import division # confidence medium
from __future__ import nested_scopes

from types import NoneType, IntType, LongType, FloatType, ComplexType,    \
                  StringType, UnicodeType, ListType, TupleType, DictType, \
                  FunctionType, ClassType, MethodType, InstanceType, TypeType, \
                  BooleanType

from trait_base      import SequenceTypes, Undefined, Self, trait_editors, \
                            class_of, TraitNotifier
from trait_errors    import TraitError, DelegationError
from trait_handlers  import TraitHandler, TraitReadOnly, TraitInstance, \
                            TraitFunction, TraitType, TraitEnum, TraitComplex, \
                            TraitMap, TraitString, AnyValue, TraitThisClass
from trait_delegates import TraitGetterSetter, TraitDelegate
from trait_notifiers import InstanceTraitNotifier, ClassTraitNotifier

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

ConstantTypes = ( NoneType, IntType, LongType, FloatType, ComplexType,
                  StringType, UnicodeType )

PythonTypes   = ( StringType,   UnicodeType,  IntType,    LongType,
                  FloatType,    ComplexType,  ListType,   TupleType,
                  DictType,     FunctionType, MethodType, ClassType,
                  InstanceType, TypeType,     NoneType )

TypeTypes     = ( StringType,   UnicodeType,  IntType,    LongType,
                  FloatType,    ComplexType,  ListType,   TupleType,
                  DictType,     BooleanType )

ClassTypes    = ( ClassType, TypeType )

CallableTypes = ( FunctionType, MethodType )

CopyTypes     = ( ListType, DictType )

try: False
except NameError: False,True = 0,1

#-------------------------------------------------------------------------------
#  'Trait' class:
#-------------------------------------------------------------------------------

class Trait:

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, default_value, *value_type, **keywords ):
        setter = None
        is_getter_setter = isinstance( default_value, TraitGetterSetter )
        if is_getter_setter:
            getter        = default_value
            default_value = Undefined
        else:
            getter = self
        if (len( value_type ) == 0) and (type( default_value ) in SequenceTypes):
            default_value, value_type = default_value[0], default_value
        if len( value_type ) == 0:
            if is_getter_setter:
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
                if typeValue is ClassType:
                    if default_value is TraitThisClass:
                        setter        = TraitThisClass()
                        default_value = None
                    else:
                        setter        = TraitInstance( default_value )
                        default_value = None
                elif typeValue is InstanceType:
                    setter = TraitInstance( default_value.__class__ )
                elif typeValue is StringType:
                    setter = TraitString( keywords )
                elif typeValue in TypeTypes:
                    setter = TraitType( typeValue )
                elif type( typeValue ) is TypeType:
                    setter        = TraitInstance( default_value )
                    default_value = None
                else:
                    setter = TraitInstance( default_value.__class__ )
        else:
            enum  = []
            other = []
            map   = {}
            self.do_list( value_type, enum, map, other )
            if ((default_value is None) and
               ((len( enum )  == 1) and (enum[0] is None)) and
               ((len( other ) == 1) and isinstance( other[0], TraitInstance ))):
                enum = []
            if len( enum ) > 0:
                other.append( TraitEnum( enum ) )
            if len( map ) > 0:
                other.append( TraitMap( map ) )
            if len( other ) == 0:
                setter = TraitHandler()
            elif len( other ) == 1:
                setter = other[0]
                if isinstance( setter, TraitGetterSetter ):
                    getter = setter
                elif isinstance( setter, Trait ):
                    dic = setter.__dict__.copy()
                    dic.update( keywords )
                    dic[ 'default_value' ] = default_value
                    keywords = dic
                elif ((default_value is None) and
                      isinstance( setter, TraitInstance )):
                    setter.allow_none()
            else:
                setter = TraitComplex( other )

        # Save the results as traits:
        self.default_value = default_value
        self.getter        = getter
        self.setter        = setter

        # Copy all the metadata into the trait definition:
        for name, value in keywords.items():
            setattr( self, name, value )

        # If the setter has any metadata, copy it also:
        if ((setter is not None) and
            (getattr( setter, 'metadata', None ) is not None)):
            for name, value in setter.metadata().items():
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
                elif typeItem is DictType:
                    map.update( item )
                elif typeItem in CallableTypes:
                    other.append( TraitFunction( item ) )
                elif item is TraitThisClass:
                    other.append( TraitThisClass() )
                elif (isinstance( item, TraitHandler )      or
                      isinstance( item, TraitGetterSetter ) or
                      isinstance( item, Trait )):
                    other.append( item )
                elif typeItem in ClassTypes:
                    other.append( TraitInstance( item ) )
                else:
                    other.append( TraitHandler() )

    #----------------------------------------------------------------------------
    #  Default handler for initializing trait values:
    #----------------------------------------------------------------------------

    def getattr ( self, object, name, value ):
        if value is Self:
            return object
        if type( value ) not in CopyTypes:
            return object.__setattr__( name, value )
        if type( value ) is ListType:
            return object.__setattr__( name, value[:] )
        return object.__setattr__( name, value.copy() )

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
#  Create singleton-like instances (so users don't have to):
#-------------------------------------------------------------------------------

Disallow           = TraitDelegate()   # Disallow getting and/or setting a trait
ReadOnly           = Trait( Undefined, TraitReadOnly() ) # Read-only trait
DefaultPythonTrait = PythonTrait()     # Trait w/standard Python semantics

# This special trait is only for internal use:
TheMappedTrait = MappedTrait()  # Create the (unique) mapped trait

#-------------------------------------------------------------------------------
#  'SimpleTest' class:
#-------------------------------------------------------------------------------

class SimpleTest:
    def __init__ ( self, value ): self.value = value
    def __call__ ( self, test  ): return test == self.value

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

    #---------------------------------------------------------------------------
    #  Handle a trait or attribute being deleted:
    #---------------------------------------------------------------------------

    def __delattr__ ( self, name ):
        try:
            del self.__dict__[ name ]
            self.__getattr__( name )
        except KeyError:
            return

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
            trait   = self.__traits__[ name ]
            getattr = trait.getter.getattr
        except (AttributeError, KeyError):
            trait   = self._trait( name )
            getattr = trait.getter.getattr
        try:
            return getattr( self, name, trait.default_value )
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
        pad    = max( [ len( x ) for x in names ] ) + 1
        maxval = 78 - pad
        names.sort()
        for name in names:
            try:
                value = repr( getattr( self, name ) ).replace( '\n', '\\n' )
                if len( value ) > maxval:
                    value = '%s...%s' % ( value[: (maxval - 2) // 2 ],
                                          value[ -((maxval - 3) // 2): ] )
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
        return self

    #----------------------------------------------------------------------------
    #  Reset some or all of an object's traits to their default values:
    #----------------------------------------------------------------------------

    def reset_traits ( self, traits = None ):
        unresetable = []
        if traits is None:
            traits = self._trait_names()
        for name in traits:
            try:
                delattr( self, name )
            except AttributeError:
                unresetable.append( name )
        return unresetable

    #----------------------------------------------------------------------------
    #  Set the object's traits based upon those of another object:
    #----------------------------------------------------------------------------

    def clone_traits ( self, other, traits = None ):
        unassignable = []
        if traits is None:
            traits = self._trait_names()
        for name in traits:
            try:
                setattr( self, name, getattr( other, name ) )
            except:
                unassignable.append( name )
        return unassignable

    #---------------------------------------------------------------------------
    #  Return a list of all trait names which match a set of metadata:
    #---------------------------------------------------------------------------

    def traits ( self, **metadata ):
        result = []
        for meta_name, meta_eval in metadata.items():
            if type( meta_eval ) is not FunctionType:
                metadata[ meta_name ] = SimpleTest( meta_eval )
        for name in self._trait_names():
            trait = self._trait( name )
            for meta_name, meta_eval in metadata.items():
                if not meta_eval( getattr( trait, meta_name ) ):
                    break
            else:
                result.append( name )
        return result

    #----------------------------------------------------------------------------
    #  Edit the object's traits:
    #----------------------------------------------------------------------------

    def edit_traits ( self, traits = None ):
        trait_editors().TraitSheetDialog( self, traits )

    #----------------------------------------------------------------------------
    #  Configure the object's traits:
    #----------------------------------------------------------------------------

    def configure_traits ( self, filename = None, edit = True, traits = None ):
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
                    cPickle.Pickler( fd, True ).dump( self )
                except:
                    if fd is not None:
                        fd.close()
                    return False

        return True

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
    #  Return the definition of a specified trait:
    #----------------------------------------------------------------------------

    def get_trait ( self, name ):
        return self._trait( name )

    #----------------------------------------------------------------------------
    #  Synchronize the value of two traits:
    #----------------------------------------------------------------------------

    def sync_trait ( self, trait_name, object, alias = None, mutual = True ):
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

    def on_trait_change ( self, handler, trait_name = None, remove = False ):
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
            dict[ TraitNotifier ] = notifiers = InstanceTraitNotifier( self,
                                    self.__class__.__dict__.get( TraitNotifier ) )
            notifiers.reset_trait_value( self )
        notifiers.add( handler, trait_name )

    #---------------------------------------------------------------------------
    #  Allow trait change notifications to be deferred to later or be
    #  processed immediately:
    #---------------------------------------------------------------------------

    def defer_trait_change ( self, defer = True ):
        self._object_notifier().defer_trait_change( self, defer )

    #---------------------------------------------------------------------------
    #  Get the object or class level notifier manager:
    #---------------------------------------------------------------------------

    def _object_notifier ( self ):
        notifier = self.__dict__.get( TraitNotifier )
        if notifier is not None:
            return notifier
        return self._class_notifier()

    #----------------------------------------------------------------------------
    #  Return the class-level notifier for a specific trait:
    #----------------------------------------------------------------------------

    def _notifier_for ( self, name ):
        return self._class_notifier().notifier_for( name )

    #----------------------------------------------------------------------------
    #  Return the class-level event notifier for a specific trait:
    #----------------------------------------------------------------------------

    def _event_notifier_for ( self, name ):
        return self._class_notifier().event_notifier_for( name )

    #----------------------------------------------------------------------------
    #  Return the object's class-level notifier manager:
    #----------------------------------------------------------------------------

    def _class_notifier ( self ):
        cls      = self.__class__
        notifier = cls.__dict__.get( TraitNotifier )
        if notifier is None:
            notifier = ClassTraitNotifier( cls )
            setattr( cls, TraitNotifier, notifier )
        return notifier

    #----------------------------------------------------------------------------
    #  Set up the class-level 'set trait value' handler:
    #----------------------------------------------------------------------------

    def _reset_trait_value ( self ):
        self._object_notifier().reset_trait_value( self )

    #----------------------------------------------------------------------------
    #  Change an object trait:
    #----------------------------------------------------------------------------

    def _set_trait_value ( self, object, name, value, default ):
        self._reset_trait_value()
        return self._set_trait_value( object, name, value, default )

    #----------------------------------------------------------------------------
    #  Change an object 'event' trait:
    #----------------------------------------------------------------------------

    def _set_event_value ( self, object, name, value, default ):
        self._reset_trait_value()
        return self._set_event_value( object, name, value, default )

    #----------------------------------------------------------------------------
    #  Get the list of all valid trait names:
    #----------------------------------------------------------------------------

    def _trait_names ( self ):
        return [ x for x in self.__traits__.keys() if
                 (x[-1:] not in '*_') and (x[:1] != '_') ]

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
        has_traits = (cls is HasTraits)
        if not has_traits:
            for base in cls.__bases__:
                has_traits |= self._inherit( base, traits )
        if has_traits:
            traits.update( cls.__traits__ )
        return has_traits

#------------------------------------------------------------------------------
#  'MetaTraits' class:
#------------------------------------------------------------------------------

class _MetaTraits ( type ):

    def __new__ ( cls, name, bases, classdict ):
        traits = classdict.get( '__traits__' )
        for name, value in classdict.items():
            if isinstance( value, Trait ):
                if traits is None:
                    classdict[ '__traits__' ] = traits = {}
                traits[ name ] = value
                del classdict[ name ]
        return super( _MetaTraits, cls ).__new__( cls, name, bases, classdict )

#-------------------------------------------------------------------------------
#  'HasObjectTraits' class:
#-------------------------------------------------------------------------------

class HasObjectTraits ( HasTraits, object ):
    __metaclass__ = _MetaTraits

#-------------------------------------------------------------------------------
#  'HasDynamicTraits' class:
#-------------------------------------------------------------------------------

class HasDynamicTraits ( HasTraits ):

    #----------------------------------------------------------------------------
    #  Initialize the trait values of an object (optional):
    #----------------------------------------------------------------------------

    def __init__ ( self, **traits ):
        if self.__traits__.get( '**' ) is None:
            self._init_class_list()
        self.__traits__ = self.__traits__.copy()
        HasTraits.__init__( self, **traits )

#-------------------------------------------------------------------------------
#  'HasDynamicObjectTraits' class:
#-------------------------------------------------------------------------------

class HasDynamicObjectTraits ( HasDynamicTraits, object ):

    __metaclass__ = _MetaTraits

#-------------------------------------------------------------------------------
#  'TraitProxy' class:
#-------------------------------------------------------------------------------

class TraitProxy ( HasDynamicTraits ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, object, *trait_names ):
        HasDynamicTraits.__init__( self )
        self._object = object
        delegate     = TraitDelegate( '_object', True )
        for trait_name in trait_names:
            self.add_trait( trait_name, delegate )

    def anytrait_changed ( self, trait_name, old, new ):
        setattr( self._object, trait_name, new )

#-------------------------------------------------------------------------------
#  Handle circular module dependencies:
#-------------------------------------------------------------------------------

import trait_handlers
trait_handlers.Trait = Trait

import trait_delegates
trait_delegates.HasTraits = HasTraits
