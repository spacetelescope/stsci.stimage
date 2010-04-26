#-------------------------------------------------------------------------------
#
#  Define the classes to handled 'synched' and 'unsynched' delegated traits.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  Refactored into a separate module: 07/04/2003
#
#  Symbols defined: TraitGetterSetter
#                   TraitDelegate
#                   TraitDelegateSynched
#                   TraitEvent
#                   TraitProperty
#
#  (c) Copyright 2002, 2003 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
from __future__ import division # confidence high

from __future__   import nested_scopes
from types        import StringType, MethodType, InstanceType, ClassType, \
                         TypeType
from trait_base   import Undefined, CoercableFuncs, class_of, trait_editors
from trait_errors import TraitError, DelegationError

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

EmptyDict = {}

HasTraits = None  # Patched by 'traits.py' once class is defined!

#-------------------------------------------------------------------------------
#  'TraitGetterSetter' class  (Abstract class):
#-------------------------------------------------------------------------------

class TraitGetterSetter:

    def metadata ( self ):
        return getattr( self, '__traits_metadata__', {} )

#-------------------------------------------------------------------------------
#  'TraitDelegate' class:
#-------------------------------------------------------------------------------

class TraitDelegate (TraitGetterSetter ):

    __traits_metadata__ = {
       'type': 'delegate'
    }

    def __init__ ( self, delegate = None, mutate_or_prefix = False ):
        self._delegate = delegate
        self.delegate  = self.get_delegate
        self.getattr   = self.getattr_method
        self.setattr   = self.setattr # Performance hack!
        self.prefix    = ''
        self.mutate    = False
        if type( mutate_or_prefix ) is StringType:
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
            if type( delegate ) is MethodType:
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
            return getattr( getattr( object, self._delegate ),
                            self.prefix + name )
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
            if value is not Undefined:
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
                if value is not Undefined:
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

    #----------------------------------------------------------------------------
    #  Validate the value for a particular object delegate's trait:
    #----------------------------------------------------------------------------

    def setattr ( self, object, name, value, default ):
        try:
            delegate_name = self.name( object, name )
            delegate      = self.delegate( object )
            while True:
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
            while True:
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

    def __init__ ( self, delegate = None, mutate_or_prefix = False ):
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
            setattr( object, name, value )
            dict         = object.__dict__
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
            old.on_trait_change( handler, delegate_name, True )
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
                                                     True )
            del handlers[ name ]

#-------------------------------------------------------------------------------
#  'TraitEvent' class:
#-------------------------------------------------------------------------------

class TraitEvent ( TraitGetterSetter ):

    __traits_metadata__ = {
       'type': 'event'
    }

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, klass = None ):
        if klass is None:
            self.validate = self.any_value_validate
        else:
            self.kind = klass
            kind      = type( klass )
            if kind is not ClassType:
                self.validate = self.type_validate
                if kind is not TypeType:
                    self.kind = kind
                try:
                    self.coerce = CoercableFuncs[ kind ]
                except:
                    self.coerce = self.identity

    def validate ( self, object, name, value ):
        if isinstance( value, self.kind ):
            return value
        raise TraitError, ( object, name,
                          '%s instance' % class_of( self.kind.__name__ ), value )

    def any_value_validate ( self, object, name, value ):
        return value

    def type_validate ( self, object, name, value ):
        try:
            return self.coerce( value )
        except:
            pass
        if type( value ) is InstanceType:
            kind = class_of( value )
        else:
            kind = repr( value )
        raise TraitError, ( object, name, 'of %s' % str( self.kind )[1:-1],
                          '%s (i.e. %s)' % ( str( type( value ) )[1:-1], kind ) )

    def identity ( self, value ):
        if type( value ) is self.kind:
            return value
        raise TraitError

    def setattr ( self, object, name, value, default ):
        return object._set_event_value( object, name,
                                  self.validate( object, name, value ), default )

    def getattr ( self, object, name, value ):
        raise AttributeError, ( "the %s trait of %s instance is an 'event', "
              "which is write only" % ( name, class_of( object ) ) )

#-------------------------------------------------------------------------------
#  'TraitProperty' class:
#-------------------------------------------------------------------------------

class TraitProperty ( TraitGetterSetter ):

    __traits_metadata__ = {
       'type': 'property'
    }

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, fget = None, fset = None ):
        if fget is not None:
            self.getattr = PropertyGetWrapper( fget )

        if fset is not None:
            self.setattr    = PropertySetWrapper( fset )
            self.get_editor = self._get_editor

    def setattr ( self, object, name, value, default ):
        raise AttributeError, "the %s trait of %s instance is read only" % (
                              name, class_of( object ) )

    def getattr ( self, object, name, value ):
        raise AttributeError, "the %s trait of %s instance is write only" % (
                              name, class_of( object ) )

    def _get_editor ( self, trait ):
        return trait_editors().TraitEditorText( evaluate = True,
                                                auto_set = False )

#-------------------------------------------------------------------------------
#  'PropertySetWrapper' class:
#-------------------------------------------------------------------------------

class PropertySetWrapper:

    def __init__ ( self, handler ):
        self.handler  = handler
        self.__call__ = getattr( self, 'call_%d' %
                                       handler.func_code.co_argcount )

    def call_0 ( self, object, trait_name, value, default ):
        return self.handler()

    def call_1 ( self, object, trait_name, value, default ):
        return self.handler( object )

    def call_2 ( self, object, trait_name, value, default ):
        return self.handler( object, value )

    def call_3 ( self, object, trait_name, value, default ):
        return self.handler( object, trait_name, value )

#-------------------------------------------------------------------------------
#  'PropertyGetWrapper' class:
#-------------------------------------------------------------------------------

class PropertyGetWrapper:

    def __init__ ( self, handler ):
        self.handler  = handler
        self.__call__ = getattr( self, 'call_%d' %
                                       handler.func_code.co_argcount )

    def call_0 ( self, object, trait_name, default ):
        return self.handler()

    def call_1 ( self, object, trait_name, default ):
        return self.handler( object )

    def call_2 ( self, object, trait_name, default ):
        return self.handler( object, trait_name )
