#-------------------------------------------------------------------------------
#
#  Define the classes needed to implement and support the trait change
#  notification mechanism.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  Refactored into a separate module: 07/04/2003
#
#  Symbols defined: TraitChangeNotifier
#                   EventChangeNotify
#                   TraitChangeNotifyWrapper
#                   StaticAnyTraitChangeNotifyWrapper
#                   StaticTraitChangeNotifyWrapper
#                   InstanceTraitNotifier
#                   ClassTraitNotifier
#                   SpecificTraitNotifier
#                   SpecificEventNotifier
#                   AnyAndSpecificTraitNotifier
#                   AnyAndSpecificEventNotifier
#
#  (c) Copyright 2002, 2003 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
from __future__ import division # confidence high

import traceback

from types           import MethodType
from trait_base      import TraitNotifier
from trait_delegates import TraitEvent

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
        cls_notifier            = cls.__dict__.get( TraitNotifier )
        self.trait_notifier     = self.anytrait_notifier = None
        trait_notifier          = getattr( cls, name + '_changed',  None )
        if trait_notifier is not None:
            if cls_notifier is not None:
                notifier = cls_notifier.notifiers.get( name )
                if notifier is not None:
                    self.trait_notifier = notifier.notifier
            if self.trait_notifier is None:
                self.trait_notifier = StaticTraitChangeNotifyWrapper(
                                         trait_notifier )
        anytrait_notifier = getattr( cls, 'anytrait_changed', None )
        if anytrait_notifier is not None:
            if cls_notifier is not None:
                self.anytrait_notifier = cls_notifier.anytrait_notifier
            else:
                self.anytrait_notifier = StaticAnyTraitChangeNotifyWrapper(
                                            anytrait_notifier )

    #----------------------------------------------------------------------------
    #  Add a new handler:
    #----------------------------------------------------------------------------

    def add ( self, handler ):
        trait_notifiers = self.trait_notifiers
        for cur_notifier in trait_notifiers:
            # NOTE: 'is' seems like it should work, but it doesn't:
            #if handler is cur_notifier.handler:
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
            # NOTE: 'is' seems like it should work, but it doesn't:
            #if handler is cur_notifier.handler:
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
                    self.trait_notifier( object, name, old_value, value )
                for notifier in self.trait_notifiers[:]:
                    notifier( object, name, old_value, value )
                if self.anytrait_notifier is not None:
                    self.anytrait_notifier( object, name, old_value, value )
                for notifier in self.anytrait_notifiers[:]:
                    notifier( object, name, old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            if self.trait_notifier is not None:
                self.trait_notifier( object, name, old_value, value )
            for notifier in self.trait_notifiers[:]:
                notifier( object, name, old_value, value )
            if self.anytrait_notifier is not None:
                self.anytrait_notifier( object, name, old_value, value )
            for notifier in self.anytrait_notifiers[:]:
                notifier( object, name, old_value, value )
            return value

    #----------------------------------------------------------------------------
    #  Set a new value on the object (with deferred notification):
    #----------------------------------------------------------------------------

    def deferred ( self, object, name, value, default ):
        obj_dict  = object.__dict__
        old_value = obj_dict.get( name, default )
        try:
            if old_value != value:
                obj_dict[ name ] = value
                tnotifier = getattr( object, TraitNotifier )
                if self.trait_notifier is not None:
                    tnotifier.defer_notify( self.trait_notifier, object, name,
                                            old_value, value )
                for notifier in self.trait_notifiers:
                    tnotifier.defer_notify( notifier, object, name,
                                            old_value, value )
                if self.anytrait_notifier is not None:
                    tnotifier.defer_notify( self.anytrait_notifier, object, name,
                                            old_value, value )
                for notifier in self.anytrait_notifiers:
                    tnotifier.defer_notify( notifier, object, name,
                                            old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            tnotifier = getattr( object, TraitNotifier )
            if self.trait_notifier is not None:
                tnotifier.defer_notify( self.trait_notifier, object, name,
                                        old_value, value )
            for notifier in self.trait_notifiers:
                tnotifier.defer_notify( notifier, object, name,
                                        old_value, value )
            if self.anytrait_notifier is not None:
                tnotifier.defer_notify( self.anytrait_notifier, object, name,
                                        old_value, value )
            for notifier in self.anytrait_notifiers:
                tnotifier.defer_notify( notifier, object, name,
                                        old_value, value )
            return value

#-------------------------------------------------------------------------------
#  'EventChangeNotifier' class:
#-------------------------------------------------------------------------------

class EventChangeNotifier ( TraitChangeNotifier ):

    #----------------------------------------------------------------------------
    #  Set a new value on the object:
    #----------------------------------------------------------------------------

    def __call__ ( self, object, name, value, default ):
        if self.trait_notifier is not None:
            self.trait_notifier( object, name, None, value )
        for notifier in self.trait_notifiers[:]:
            notifier( object, name, None, value )
        if self.anytrait_notifier is not None:
            self.anytrait_notifier( object, name, None, value )
        for notifier in self.anytrait_notifiers[:]:
            notifier( object, name, None, value )
        return value

    def deferred ( self, object, name, value, default ):
        tnotifier = getattr( object, TraitNotifier )
        if self.trait_notifier is not None:
            tnotifier.defer_notify( self.trait_notifier, object, name,
                                    None, value )
        for notifier in self.trait_notifiers:
            tnotifier.defer_notify( notifier, object, name, None, value )
        if self.anytrait_notifier is not None:
            tnotifier.defer_notify( self.anytrait_notifier, object, name,
                                    None, value )
        for notifier in self.anytrait_notifiers:
            tnotifier.defer_notify( notifier, object, name, None, value )
        return value

#-------------------------------------------------------------------------------
#  'TraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class TraitChangeNotifyWrapper:

    def __init__ ( self, handler ):
        self.handler = handler
        adjust       = 0
        func         = handler
        if type( handler ) is MethodType:
            func   = handler.im_func
            adjust = 1
        self.__call__ = getattr( self, 'call_%d' %
                                 (func.func_code.co_argcount - adjust) )

    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.handler()
        except:
            traceback.print_exc()

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.handler( new )
        except:
            traceback.print_exc()

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.handler( trait_name, new )
        except:
            traceback.print_exc()

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, new )
        except:
            traceback.print_exc()

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, old, new )
        except:
            traceback.print_exc()

#-------------------------------------------------------------------------------
#  'StaticAnyTraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class StaticAnyTraitChangeNotifyWrapper:

    def __init__ ( self, handler ):
        self.handler  = handler
        self.__call__ = getattr( self, 'call_%d' %
                                       handler.func_code.co_argcount )

    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.handler()
        except:
            traceback.print_exc()

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.handler( object )
        except:
            traceback.print_exc()

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name )
        except:
            traceback.print_exc()

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, new )
        except:
            traceback.print_exc()

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, old, new )
        except:
            traceback.print_exc()

#-------------------------------------------------------------------------------
#  'StaticTraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class StaticTraitChangeNotifyWrapper:

    def __init__ ( self, handler ):
        self.handler  = handler
        self.__call__ = getattr( self, 'call_%d' %
                                       handler.func_code.co_argcount )

    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.handler()
        except:
            traceback.print_exc()

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.handler( object )
        except:
            traceback.print_exc()

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, new )
        except:
            traceback.print_exc()

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, old, new )
        except:
            traceback.print_exc()

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, old, new )
        except:
            traceback.print_exc()

#-------------------------------------------------------------------------------
#  'InstanceTraitNotifier' class:
#-------------------------------------------------------------------------------

class InstanceTraitNotifier:

    def __init__ ( self, object, class_notifier ):
        TraitNotifier.__init__ ( self )
        self.object             = object
        self.deferrals          = None
        self.deferral_level     = 0
        self.active_notifiers   = 0
        self.notifiers          = {}
        self.anytrait_notifiers = []
        self.binder             = InstanceTraitNotifierBinder(
                                     self, '_notifier_for',
                                     TraitChangeNotifier )
        self.event_binder       = InstanceTraitNotifierBinder(
                                     self, '_event_notifier_for',
                                     EventChangeNotifier )
        if class_notifier is not None:
            obj_id = id( object )
            info   = class_notifier.deferrals.get( obj_id )
            if info is not None:
                self.deferral_level, deferrals = info
                self.deferrals = {}
                for trait_name in deferrals.keys():
                    notifiers, old_value, new_value = deferrals[ trait_name ]
                    for notifier in notifiers.values():
                        self.defer_notify( notifier, object, trait_name,
                                           old_value, new_value )
                del class_notifier.deferrals[ obj_id ]

    def _set_trait_value ( self, object, name, value, default ):
        return self.notifiers.get( name, self.binder )(
                                   object, name, value, default )

    def _set_trait_value_deferred ( self, object, name, value, default ):
        return self.notifiers.get( name, self.binder ).deferred(
                                   object, name, value, default )

    def _set_event_value ( self, object, name, value, default ):
        return self.notifiers.get( name, self.event_binder )(
                                   object, name, value, default )

    def _set_event_value_deferred ( self, object, name, value, default ):
        return self.notifiers.get( name, self.event_binder ).deferred(
                                   object, name, value, default )

    def add ( self, handler, name ):
        if name == 'anytrait':
            anytrait_notifiers = self.anytrait_notifiers
            if len( anytrait_notifiers ) == 0:
                notifiers = self.notifiers
                for name, notifier in notifiers.items():
                    if not isinstance( notifier, TraitChangeNotifier ):
                        mutates_to = TraitChangeNotifier
                        if isinstance( self.object._trait( name ).setter,
                                       TraitEvent ):
                            mutates_to = EventChangeNotifier
                        notifiers[ name ] = mutates_to(
                                   self.object, name, anytrait_notifiers )
            anytrait_notifiers.append( TraitChangeNotifyWrapper( handler ) )
        else:
            notifier = self.notifiers.get( name, None )
            if not isinstance( notifier, TraitChangeNotifier ):
                mutates_to = TraitChangeNotifier
                if isinstance( self.object._trait( name ).setter, TraitEvent ):
                    mutates_to = EventChangeNotifier
                self.notifiers[ name ] = notifier = mutates_to(
                                     self.object, name, self.anytrait_notifiers )
            self.active_notifiers += notifier.add( handler )

    def remove ( self, handler, name ):
        if name == 'anytrait':
            anytrait_notifiers = self.anytrait_notifiers
            for notifier in anytrait_notifiers:
                # NOTE: 'is' seems like it should work, but it doesn't:
                #if handler is notifier.handler:
                if handler == notifier.handler:
                    anytrait_notifiers.remove( notifier )
                    if len( anytrait_notifiers ) == 0:
                        object = self.object
                        if self.active_notifiers == 0:
                            self.move_deferrals_to_class()
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
                        self.move_deferrals_to_class()

    def reset_trait_value ( self, object ):
        obj_dict = object.__dict__
        if self.deferral_level == 0:
            obj_dict[ '_set_trait_value' ] = self._set_trait_value
            obj_dict[ '_set_event_value' ] = self._set_event_value
        else:
            obj_dict[ '_set_trait_value' ] = self._set_trait_value_deferred
            obj_dict[ '_set_event_value' ] = self._set_event_value_deferred

    def defer_trait_change ( self, object, defer = True ):
        if defer:
            self.deferral_level += 1
            if self.deferral_level == 1:
                self.deferrals = {}
                object._reset_trait_value()
        else:
            self.deferral_level -= 1
            if self.deferral_level == 0:
                deferrals = self.deferrals
                for trait_name in deferrals.keys():
                    notifiers, old_value, new_value = deferrals[ trait_name ]
                    for notifier in notifiers.values():
                        notifier( object, trait_name, old_value, new_value )
                self.deferrals = None
                object._reset_trait_value()

    def defer_notify ( self, notifier, object, trait_name, old, new ):
        info = self.deferrals.setdefault( trait_name, [ {}, old, new ] )
        info[0].setdefault( id( notifier ), notifier )
        info[2] = new

    def move_deferrals_to_class ( self ):
        object = self.object
        del object.__dict__[ TraitNotifier ]
        deferrals = self.deferrals
        if deferrals is not None:
            cls_notifier = object._class_notifier()
            info    = cls_notifier.deferrals.setdefault( id( object ), [ 0, {} ] )
            info[0] = self.deferral_level
            for trait_name in deferrals.keys():
                notifiers, old_value, new_value = deferrals[ trait_name ]
                for notifier in notifiers.values():
                    cls_notifier.defer_notify( notifier, object, trait_name,
                                               old_value, new_value )
            self.deferrals = None
        self.object = self.notifiers = None
        object._reset_trait_value()

#-------------------------------------------------------------------------------
#  'InstanceTraitNotifierBinder' class:
#-------------------------------------------------------------------------------

class InstanceTraitNotifierBinder:

    def __init__ ( self, tnotifier, notifier_for, notifier_factory ):
        self.tnotifier        = tnotifier
        self.notifier_for     = getattr( tnotifier.object, notifier_for )
        self.notifier_factory = notifier_factory

    def __call__ ( self, object, name, value, default ):
        tnotifier = self.tnotifier
        if len( tnotifier.anytrait_notifiers ) == 0:
            notifier = self.notifier_for( name )
        else:
            notifier = self.notifier_factory( object, name,
                                              tnotifier.anytrait_notifiers )
        tnotifier.notifiers[ name ] = notifier
        return notifier( object, name, value, default )

    def deferred ( self, object, name, value, default ):
        tnotifier = self.tnotifier
        if len( tnotifier.anytrait_notifiers ) == 0:
            notifier = self.notifier_for( name )
        else:
            notifier = self.notifier_factory( object, name,
                                              tnotifier.anytrait_notifiers )
        tnotifier.notifiers[ name ] = notifier
        return notifier.deferred( object, name, value, default )

#-------------------------------------------------------------------------------
#  'ClassTraitNotifier' class:
#-------------------------------------------------------------------------------

class ClassTraitNotifier:

    def __init__ ( self, cls ):
        TraitNotifier.__init__( self )
        self.cls                = cls
        self.notifiers          = {}
        self.deferrals          = {}
        self.bind_factory       = self.no_anytrait_changed
        self.event_bind_factory = self.event_no_anytrait_changed
        handler                 = getattr( cls, 'anytrait_changed', None )
        if handler is not None:
            self.anytrait_notifier  = StaticAnyTraitChangeNotifyWrapper( handler )
            self.bind_factory       = self.has_anytrait_changed
            self.event_bind_factory = self.event_has_anytrait_changed
        self.binder       = ClassTraitNotifierBinder( self.notifiers,
                                                      self.bind_factory )
        self.event_binder = ClassTraitNotifierBinder( self.notifiers,
                                                      self.event_bind_factory )

    def _set_trait_value ( self, object, name, value, default ):
        return self.notifiers.get( name, self.binder )(
                    object, name, value, default )

    def _set_trait_value_deferred ( self, object, name, value, default ):
        return self.notifiers.get( name, self.binder ).deferred(
                    object, name, value, default )

    def notifier_for ( self, name ):
        notifier = self.notifiers.get( name, None )
        if notifier is None:
            self.notifiers[ name ] = notifier = self.bind_factory( name )
        return notifier

    def no_anytrait_changed ( self, name ):
        notifier = getattr( self.cls, name + '_changed', None )
        if notifier is None:
            return simple_set_trait_value
        return SpecificTraitNotifier( StaticTraitChangeNotifyWrapper( notifier ))

    def has_anytrait_changed ( self, name ):
        notifier = getattr( self.cls, name + '_changed', None )
        if notifier is None:
            return SpecificTraitNotifier( self.anytrait_notifier )
        return AnyAndSpecificTraitNotifier( self.anytrait_notifier, notifier )

    def _set_event_value ( self, object, name, value, default ):
        return self.notifiers.get( name, self.event_binder )(
                                   object, name, value, default )

    def _set_event_value_deferred ( self, object, name, value, default ):
        return self.notifiers.get( name, self.event_binder ).deferred(
                                   object, name, value, default )

    def event_notifier_for ( self, name ):
        notifier = self.notifiers.get( name, None )
        if notifier is None:
            self.notifiers[ name ] = notifier = self.event_bind_factory( name )
        return notifier

    def event_no_anytrait_changed ( self, name ):
        notifier = getattr( self.cls, name + '_changed', None )
        if notifier is None:
            return ignore_set_trait_value
        return SpecificEventNotifier( StaticTraitChangeNotifyWrapper( notifier ))

    def event_has_anytrait_changed ( self, name ):
        notifier = getattr( self.cls, name + '_changed', None )
        if notifier is None:
            return SpecificEventNotifier( self.anytrait_notifier )
        return AnyAndSpecificEventNotifier( self.anytrait_notifier, notifier )

    def event_anytrait_changed ( self, object, name, value, default ):
        self.anytrait_notifier( object, name, None, value )
        return value

    def reset_trait_value ( self, object ):
        obj_dict = object.__dict__
        if self.deferrals.get( id( object ) ) is None:
            obj_dict[ '_set_trait_value' ] = self._set_trait_value
            obj_dict[ '_set_event_value' ] = self._set_event_value
        else:
            obj_dict[ '_set_trait_value' ] = self._set_trait_value_deferred
            obj_dict[ '_set_event_value' ] = self._set_event_value_deferred

    def defer_trait_change ( self, object, defer = True ):
        obj_id = id( object )
        if defer:
            info     = self.deferrals.setdefault( obj_id, [ 0, {} ] )
            info[0] += 1
            if info[0] == 1:
                object._reset_trait_value()
        else:
            info = self.deferrals.get( obj_id )
            if info is not None:
                info[0] -= 1
                if info[0] == 0:
                    deferrals = info[1]
                    for trait_name in deferrals.keys():
                        notifiers, old_value, new_value = deferrals[ trait_name ]
                        for notifier in notifiers.values():
                            notifier( object, trait_name, old_value, new_value )
                    del self.deferrals[ obj_id ]
                    object._reset_trait_value()

    def defer_notify ( self, notifier, object, trait_name, old, new ):
        info = self.deferrals[ id( object ) ][1].setdefault( trait_name,
                                                             [ {}, old, new ] )
        info[0].setdefault( id( notifier ), notifier )
        info[2] = new

#-------------------------------------------------------------------------------
#  'ClassTraitNotifierBinder' class:
#-------------------------------------------------------------------------------

class ClassTraitNotifierBinder:

    def __init__ ( self, notifiers, bind_factory ):
        self.notifiers    = notifiers
        self.bind_factory = bind_factory

    def __call__ ( self, object, name, value, default ):
        self.notifiers[ name ] = notifier = self.bind_factory( name )
        return notifier( object, name, value, default )

    def deferred ( self, object, name, value, default ):
        self.notifiers[ name ] = notifier = self.bind_factory( name )
        return notifier.deferred( object, name, value, default )

#-------------------------------------------------------------------------------
#  'SimpleSetTraitValue' class:
#-------------------------------------------------------------------------------

class SimpleSetTraitValue:

    def __call__ ( self, object, name, value, default ):
        object.__dict__[ name ] = value
        return value

    def deferred ( self, object, name, value, default ):
        object.__dict__[ name ] = value
        return value

simple_set_trait_value = SimpleSetTraitValue()

#-------------------------------------------------------------------------------
#  'IgnoreSetTraitValue' class:
#-------------------------------------------------------------------------------

class IgnoreSetTraitValue:

    def __call__ ( self, object, name, value, default ):
        pass

    def deferred ( self, object, name, value, default ):
        pass

ignore_set_trait_value = IgnoreSetTraitValue()

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
                self.notifier( object, name, old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            self.notifier( object, name, old_value, value )
            return value

    def deferred  ( self, object, name, value, default ):
        obj_dict  = object.__dict__
        old_value = obj_dict.get( name, default )
        try:
            if old_value != value:
                obj_dict[ name ] = value
                getattr( object.__class__, TraitNotifier ).defer_notify(
                   self.notifier, object, name, old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            getattr( object.__class__, TraitNotifier ).defer_notify(
               self.notifier, object, name, old_value, value )
            return value

#-------------------------------------------------------------------------------
#  'SpecificEventNotifier' class:
#-------------------------------------------------------------------------------

class SpecificEventNotifier:

    def __init__ ( self, notifier ):
        self.notifier = notifier

    def __call__ ( self, object, name, value, default ):
        self.notifier( object, name, None, value )
        return value

    def deferred ( self, object, name, value, default ):
        getattr( object.__class__, TraitNotifier ).defer_notify(
                 self.notifier, object, name, None, value )
        return value

#-------------------------------------------------------------------------------
#  'AnyAndSpecificTraitNotifier' class:
#-------------------------------------------------------------------------------

class AnyAndSpecificTraitNotifier:

    def __init__ ( self, anytrait_notifier, notifier ):
        self.anytrait_notifier = anytrait_notifier
        self.notifier          = StaticTraitChangeNotifyWrapper( notifier )

    def __call__ ( self, object, name, value, default ):
        obj_dict  = object.__dict__
        old_value = obj_dict.get( name, default )
        try:
            if old_value != value:
                obj_dict[ name ] = value
                self.notifier( object, name, old_value, value )
                self.anytrait_notifier( object, name, old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            self.notifier( object, name, old_value, value )
            self.anytrait_notifier( object, name, old_value, value )
            return value

    def deferred ( self, object, name, value, default ):
        obj_dict  = object.__dict__
        old_value = obj_dict.get( name, default )
        try:
            if old_value != value:
                obj_dict[ name ] = value
                tnotifier = getattr( object.__class__, TraitNotifier )
                tnotifier.defer_notify( self.notifier, object, name,
                                        old_value, value )
                tnotifier.defer_notify( self.anytrait_notifier, object, name,
                                        old_value, value )
                return value
            else:
                obj_dict[ name ] = value
                return value
        except:
            obj_dict[ name ] = value
            tnotifier = getattr( object.__class__, TraitNotifier )
            tnotifier.defer_notify( self.notifier, object, name,
                                    old_value, value )
            tnotifier.defer_notify( self.anytrait_notifier, object, name,
                                    old_value, value )
            return value

#-------------------------------------------------------------------------------
#  'AnyAndSpecificEventNotifier' class:
#-------------------------------------------------------------------------------

class AnyAndSpecificEventNotifier:

    def __init__ ( self, anytrait_notifier, notifier ):
        self.anytrait_notifier = anytrait_notifier
        self.notifier          = StaticTraitChangeNotifyWrapper( notifier )

    def __call__ ( self, object, name, value, default ):
        self.notifier( object, name, None, value )
        self.anytrait_notifier( object, name, None, value )
        return value

    def deferred ( self, object, name, value, default ):
        tnotifier = getattr( object.__class__, TraitNotifier )
        tnotifier.defer_notify( self.notifier, object, name, None, value )
        tnotifier.defer_notify( self.anytrait_notifier, object, name,
                                None, value )
        return value
