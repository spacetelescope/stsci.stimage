#-------------------------------------------------------------------------------
#
#  Define the base 'TraitHandler' class and a standard set of TraitHandler
#  subclasses for use with the 'traits' package.
#
#  A trait handler mediates the assignment of values to object traits. It
#  verifies (through the TraitHandler's 'validate' method) that a specified
#  value is consistent with the object trait, and generates a 'TraitError'
#  exception if not.
#
#  Written by: David C. Morrill
#
#  Date: 06/21/2002
#
#  Refactored into a separate module: 07/04/2003
#
#  Symbols defined: TraitHandler
#                   TraitRange
#                   TraitType
#                   TraitString
#                   TraitInstance
#                   TraitFunction
#                   TraitEnum
#                   TraitPrefixList
#                   TraitMap
#                   TraitPrefixMap
#                   TraitMapReverse
#                   TraitComplex
#                   TraitReadOnly
#                   TraitList
#
#  (c) Copyright 2002, 2003 by Enthought, Inc.
#
#-------------------------------------------------------------------------------
from __future__ import division # confidence high

try: False
except NameError: False,True = 0,1

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import sys
import re

from types           import StringType, InstanceType, TypeType
from trait_base      import strx, SequenceTypes, Undefined, NumericFuncs, \
                            StringTypes, CoercableFuncs, trait_editors, class_of
from trait_errors    import TraitError
from trait_delegates import TraitEvent

Trait = None  # Patched by 'traits.py' once class is defined!

#-------------------------------------------------------------------------------
#  'TraitHandler' class (base class for all trait handlers):
#-------------------------------------------------------------------------------

class TraitHandler:

    __traits_metadata__ = {
       'type': 'trait'
    }

    def validate ( self, object, name, value ):
        raise TraitError, (
                 "The '%s' trait of %s instance has an unknown type. "
                 "Contact the developer to correct the problem." % (
                 name, class_of( object ) ) )

    def setattr ( self, object, name, value, default ):
        return object._set_trait_value( object, name,
                           self.validate( object, name, value ), default )

    def error ( self, object, name, value ):
        raise TraitError, ( object, name, self.info(), value )

    def info ( self ):
        return 'a legal value'

    def repr ( self, value ):
        if type( value ) is InstanceType:
            return 'class '  + value.__class__.__name__
        return repr( value )

    def is_mapped ( self ):
        return False

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorText()

    def metadata ( self ):
        return getattr( self, '__traits_metadata__', {} )

#-------------------------------------------------------------------------------
#  'TraitRange' class:
#-------------------------------------------------------------------------------

class TraitRange ( TraitHandler ):

    def __init__ ( self, low, high ):
        self.low = self.high = None
        self.range( low, high )
        self.coerce, self.type_desc = NumericFuncs[ type( low ) ]

    def range ( self, low = None, high = None ):
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high
        return ( self.low, self.high )

    def validate ( self, object, name, value ):
        try:
            cvalue = self.coerce( value )
            if (((self.low  is None) or (self.low  <= cvalue)) and
                ((self.high is None) or (self.high >= cvalue))):
                return cvalue
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        if self.low is None:
            if self.high is None:
                return self.type_desc
            return '%s <= %s' % ( self.type_desc, self.high )
        elif self.high is None:
            return  '%s >= %s' % ( self.type_desc, self.low )
        return '%s in the range from %s to %s' % (
               self.type_desc, self.low, self.high )

    def get_editor ( self, trait ):
        auto_set = trait.auto_set
        if auto_set is None:
            auto_set = True
        return trait_editors().TraitEditorRange( self,
                     cols     = trait.cols or 3,
                     auto_set = auto_set )
#-------------------------------------------------------------------------------
#  'TraitString' class:
#-------------------------------------------------------------------------------

class TraitString ( TraitHandler ):

    def __init__ ( self, dic = {}, **keywords ):
        self.minlen = dic.get( 'minlen', None )
        if self.minlen is None:
            self.minlen = max( 0, keywords.get( 'minlen', 0 ) )
        self.maxlen = max( self.minlen, (dic.get( 'maxlen', 0 ) or
                                         keywords.get( 'maxlen', sys.maxint )) )
        self.regex  = (dic.get( 'regex', '' ) or keywords.get( 'regex', '' ) )
        if self.regex != '':
            self.match = re.compile( self.regex ).match

    def match ( self, value ):
        return True

    def validate ( self, object, name, value ):
        try:
            value = strx( value )
            if ((self.minlen <= len( value ) <= self.maxlen) and
                (self.match( value ) is not None)):
                return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        msg = ''
        if (self.minlen != 0) and (self.maxlen != sys.maxint):
            msg = ' between %d and %d characters long' % (
                  self.minlen, self.maxlen )
        elif self.maxlen != sys.maxint:
            msg = ' <= %d characters long' % self.maxlen
        elif self.minlen != 0:
            msg = ' >= %d characters long' % self.minlen
        if self.regex != '':
            if msg != '':
                msg += ' and'
            msg += (" matching the pattern '%s'" % self.regex)
        return 'a string' + msg

#-------------------------------------------------------------------------------
#  'TraitType' class:
#-------------------------------------------------------------------------------

class TraitType ( TraitHandler ):

    def __init__ ( self, aType ):
        if type( aType ) is not TypeType:
            aType = type( aType )
        self.aType = aType
        try:
            self.coerce = CoercableFuncs[ aType ]
        except:
            self.coerce = self.identity

    def validate ( self, object, name, value ):
        try:
            return self.coerce( value )
        except:
            pass
        if type( value ) is InstanceType:
            kind = class_of( value )
        else:
            kind = repr( value )
        self.error( object, name, '%s (i.e. %s)' % (
                         str( type( value ) )[1:-1], kind ) )

    def info ( self ):
        return 'of %s' % str( self.aType )[1:-1]

    def identity ( self, value ):
        if type( value ) is self.aType:
            return value
        raise TraitError

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorText( evaluate = True )

#-------------------------------------------------------------------------------
#  'TraitInstance' class:
#-------------------------------------------------------------------------------

class TraitInstance ( TraitHandler ):

    def __init__ ( self, aClass, or_none = 0 ):
        if aClass is None:
            aClass, or_none = or_none, aClass
        if type( aClass ) is InstanceType:
            aClass = aClass.__class__
        self.aClass = aClass
        if or_none is None:
            self.allow_none()

    def allow_none ( self ):
        self.validate = self.validate_none
        self.info     = self.info_none

    def validate ( self, object, name, value ):
        if isinstance( value, self.aClass ):
            return value
        self.validate_failed( object, name, value )

    def info ( self ):
        return class_of( self.aClass.__name__ )

    def validate_none ( self, object, name, value ):
        if isinstance( value, self.aClass ) or (value is None):
            return value
        self.validate_failed( object, name, value )

    def info_none ( self ):
        return class_of( self.aClass.__name__ ) + ' or None'

    def validate_failed ( self, object, name, value ):
        kind = type( value )
        if kind is InstanceType:
            msg = 'class %s' % value.__class__.__name__
        else:
            msg = '%s (i.e. %s)' % ( str( kind )[1:-1], repr( value ) )
        self.error( object, name, msg )

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorInstance()

#-------------------------------------------------------------------------------
#  'TraitThisClass' class:
#-------------------------------------------------------------------------------

class TraitThisClass ( TraitInstance ):

    def __init__ ( self, or_none = 0 ):
        if or_none is None:
            self.allow_none()

    def validate ( self, object, name, value ):
        if isinstance( value, object.__class__ ):
            return value
        self.validate_failed( object, name, value )

    def validate_none ( self, object, name, value ):
        if isinstance( value, object.__class__ ) or (value is None):
            return value
        self.validate_failed( object, name, value )

    def info ( self ):
        return 'an instance of the same type as the receiver'

    def info_none ( self ):
        return 'an instance of the same type as the receiver or None'

#-------------------------------------------------------------------------------
#  'TraitClass' class:
#-------------------------------------------------------------------------------

class TraitClass ( TraitHandler ):

    def __init__ ( self, aClass ):
        if type( aClass ) is InstanceType:
            aClass = aClass.__class__
        self.aClass = aClass

    def validate ( self, object, name, value ):
        try:
            if type( value ) == StringType:
                value = value.strip()
                col   = value.rfind( '.' )
                if col >= 0:
                    module_name = value[:col]
                    class_name  = value[col + 1:]
                    module      = sys.modules.get( module_name )
                    if module is None:
                        exec( 'import ' + module_name )
                        module = sys.modules[ module_name ]
                    value = getattr( module, class_name )
                else:
                    value = globals().get( value )

            if issubclass( value, self.aClass ):
                return value
        except:
            pass

        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return 'a subclass of ' + self.aClass.__name__

#-------------------------------------------------------------------------------
#  'TraitFunction' class:
#-------------------------------------------------------------------------------

class TraitFunction ( TraitHandler ):

    def __init__ ( self, aFunc ):
        self.aFunc = aFunc

    def validate ( self, object, name, value ):
        try:
            return self.aFunc( object, name, value )
        except TraitError:
            self.error( object, name, self.repr( value ) )

    def info ( self ):
        try:
            return self.aFunc.info
        except:
            if self.aFunc.__doc__:
                return self.aFunc.__doc__
            return 'a legal value'

#-------------------------------------------------------------------------------
#  'TraitEnum' class:
#-------------------------------------------------------------------------------

class TraitEnum ( TraitHandler ):

    def __init__ ( self, *values ):
        if (len( values ) == 1) and (type( values[0] ) in SequenceTypes):
            values = values[0]
        self.values = values

    def validate ( self, object, name, value ):
        if value in self.values:
            return value
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return ' or '.join( [ repr( x ) for x in self.values ] )

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorEnum( self, cols = trait.cols or 3  )

#-------------------------------------------------------------------------------
#  'TraitPrefixList' class:
#-------------------------------------------------------------------------------

class TraitPrefixList ( TraitHandler ):

    def __init__ ( self, *values ):
        if (len( values ) == 1) and (type( values[0] ) in SequenceTypes):
            values = values[0]
        self.values  = values
        self.values_ = values_ = {}
        for key in values:
            values_[ key ] = key

    def validate ( self, object, name, value ):
        try:
            if not self.values_.has_key( value ):
                match = None
                n     = len( value )
                for key in self.values:
                    if value == key[:n]:
                        if match is not None:
                            match = None
                            break
                        match = key
                if match is None:
                    self.error( object, name, self.repr( value ) )
                self.values_[ value ] = match
            return self.values_[ value ]
        except:
            self.error( object, name, self.repr( value ) )

    def info ( self ):
        return (' or '.join( [ repr( x ) for x in self.values ] ) +
                ' (or any unique prefix)')

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorEnum( self, cols = trait.cols or 3  )

#-------------------------------------------------------------------------------
#  'TraitMap' class:
#-------------------------------------------------------------------------------

class TraitMap ( TraitHandler ):

    def __init__ ( self, map ):
        self.map = map

    def validate ( self, object, name, value ):
        try:
            if self.map.has_key( value ):
                return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def setattr ( self, object, name, value, default ):
        old_value = object.__dict__.get( name, Undefined )
        value     = self.validate( object, name, value )
        if value != old_value:
            object.__dict__[ name + '_' ] = self.map[ value ]
        object._set_trait_value( object, name, value, default )
        return value

    def info ( self ):
        keys = [ repr( x ) for x in self.map.keys() ]
        keys.sort()
        return ' or '.join( keys )

    def reverse ( self ):
        if not hasattr( self, '_reverse' ):
            self._reverse = TraitMapReverse( self.map )
        return self._reverse

    def is_mapped ( self ):
        return True

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

    def validate ( self, object, name, value ):
        try:
            if not self._map.has_key( value ):
                match = None
                n     = len( value )
                for key in self.map.keys():
                    if value == key[:n]:
                        if match is not None:
                            match = None
                            break
                        match = key
                if match is None:
                    self.error( object, name, self.repr( value ) )
                self._map[ value ] = match
            return self._map[ value ]
        except:
            self.error( object, name, self.repr( value ) )

    def setattr ( self, object, name, value, default ):
        value     = self.validate( object, name, value )
        old_value = object.__dict__.get( name, Undefined )
        if value != old_value:
            object.__dict__[ name + '_' ] = self.map[ value ]
        object._set_trait_value( object, name, value, default )
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

    def validate ( self, object, name, value ):
        try:
            if self.map.has_key( value ):
                return value
        except:
            pass
        self.error( object, name, self.repr( value ) )

    def setattr ( self, object, name, value, default ):
        value = self.validate( object, name, value )
        object.__dict__[ name ] = value
        if object.__dict__.get( name[:-1], Undefined ) not in self.map[ value ]:
            object._set_trait_value( object, name[:-1], self.map[ value ][0],
                                     Undefined )
        return value

    def info ( self ):
        keys = [ repr( x ) for x in self.map.keys() ]
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

    def validate ( self, object, name, value ):
        for handler in self.handlers:
            try:
                return handler.validate( object, name, value )
            except TraitError:
                pass
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        return ' or '.join( [ x.info() for x in self.handlers ] )

    def mapped_setattr ( self, object, name, value, default ):
        for handler in self.handlers:
            try:
                result = handler.setattr( object, name, value, default )
                object.__dict__[ name + '_' ] = result
                return result
            except TraitError:
                pass
        for handler in self.map_handlers:
            try:
                return handler.setattr( object, name, value, default )
            except TraitError:
                pass
        self.error( object, name, self.repr( value ) )

    def mapped_info ( self ):
        return ' or '.join( [ x.info() for x in
                              self.handlers + self.map_handlers ] )

    def mapped_is_mapped ( self ):
        return True

    def get_editor ( self, trait ):
        the_editors = [ x.get_editor( trait ) for x in self.handlers ]
        if self.is_mapped():
            the_editors.extend( [ x.get_editor( trait ) for x in
                                  self.map_handlers ] )
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

    def validate ( self, object, name, value ):
        return value

    def info ( self ):
        return 'any value'

#-------------------------------------------------------------------------------
#  Create a singleton-like instance (so users don't have to):
#-------------------------------------------------------------------------------

AnyValue = TraitAny()        # Allow any value for a trait

#-------------------------------------------------------------------------------
#  'TraitReadOnly' class:
#-------------------------------------------------------------------------------

class TraitReadOnly ( TraitHandler ):

    def validate ( self, object, name, value ):
        no_key = (not object.__dict__.has_key( name ))
        if no_key:
            return value
        if ((value is not Undefined) and
            (no_key or (getattr( object, name ) is Undefined))):
            return value
        raise TraitError, (
                 "The '%s' trait of %s instance is read only." % (
                 name, class_of( object ) ) )

    def info ( self ):
        return 'any value'

#-------------------------------------------------------------------------------
#  'TraitList' class:
#-------------------------------------------------------------------------------

class TraitList ( TraitHandler ):

    info_trait = None

    def __init__ ( self, trait = None, min_items = None, max_items = None,
                         items = True ):
        if trait is None:
            trait = AnyValue
        if not isinstance( trait, Trait ):
            trait = Trait( trait )
        self.item_trait = trait
        self.min_items  = min_items or 0
        self.max_items  = max_items or sys.maxint
        self.items      = items

    def validate ( self, object, name, value ):
        if (isinstance( value, list ) and
           (self.min_items <= len( value ) <= self.max_items)):
            if self.items and (not hasattr( object, name + '_items' )):
                if self.info_trait is None:
                    self.info_trait = TraitEvent( 0 )
                object.add_trait( name + '_items', self.info_trait )
            list_value    = TraitListObject( self, object, name, self.items )
            list_value[:] = value
            return list_value
        self.error( object, name, self.repr( value ) )

    def info ( self ):
        if self.min_items == 0:
            if self.max_items == sys.maxint:
                size = 'of items'
            else:
                size = ' of at most %d items' % self.max_items
        else:
            if self.max_items == sys.maxint:
                size = ' of at least %d items' % self.min_items
            else:
                size = ' of from %d to %d items' % (
                       self.min_items, self.max_items )
        return 'a list%s which are %s' % ( size, self.item_trait.setter.info() )

    def get_editor ( self, trait ):
        return trait_editors().TraitEditorList( self, trait.rows or 5 )

#-------------------------------------------------------------------------------
#  'TraitListObject' class:
#-------------------------------------------------------------------------------

class TraitListObject ( list ):

    def __init__ ( self, trait, object, name, items ):
        self.trait      = trait
        self.object     = object
        self.name       = name
        self.name_items = None
        if items:
            self.name_items = name + '_items'

    def __setitem__ ( self, key, value ):
        try:
            list.__setitem__( self, key, self.trait.item_trait.setter.validate(
                                            self.object, self.name, value )  )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 0 )
        except TraitError, excp:
            excp.set_prefix( 'Each element of the' )
            raise excp

    def __setslice__ ( self, i, j, values ):
        delta = len( values ) - (min( j, len( self ) ) - max( 0, i ))
        if self.trait.min_items <= (len( self ) + delta) <= self.trait.max_items:
            try:
                object   = self.object
                name     = self.name
                trait    = self.trait.item_trait
                validate = trait.setter.validate
                list.__setslice__( self, i, j,
                          [ validate( object, name, value ) for value in values ] )
                if self.name_items is not None:
                    setattr( self.object, self.name_items, 0 )
                return
            except TraitError, excp:
                excp.set_prefix( 'Each element of the' )
                raise excp
        self.len_error( len( self ) + delta )

    def __delitem__ ( self, key ):
        if self.trait.min_items <= (len( self ) - 1):
            list.__delitem__( self, key )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 0 )
            return
        self.len_error( len( self ) - 1 )

    def __delslice__ ( self, i, j ):
        delta = min( j, len( self ) ) - max( 0, i )
        if self.trait.min_items <= (len( self ) - delta):
            list.__delslice__( self, i, j )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 0 )
            return
        self.len_error( len( self ) - delta )

    def append ( self, value ):
        if self.trait.min_items <= (len( self ) + 1) <= self.trait.max_items:
            try:
                list.append( self, self.trait.item_trait.setter.validate(
                                   self.object, self.name, value ) )
                if self.name_items is not None:
                    setattr( self.object, self.name_items, 0 )
                return
            except TraitError, excp:
                excp.set_prefix( 'Each element of the' )
                raise excp
        self.len_error( len( self ) + 1 )

    def extend ( self, xlist ):
        if (self.trait.min_items <= (len( self ) + len( xlist )) <=
            self.trait.max_items):
            object   = self.object
            name     = self.name
            validate = self.trait.item_trait.setter.validate
            try:
                list.extend( self, [ validate( object, name, value )
                                     for value in xlist ] )
                if self.name_items is not None:
                    setattr( self.object, self.name_items, 0 )
                return
            except TraitError, excp:
                excp.set_prefix( 'The elements of the' )
                raise excp
        self.len_error( len( self ) + len( xlist ) )

    def remove ( self, value ):
        if self.trait.min_items < len( self ):
            list.remove( self, value )
            if self.name_items is not None:
                setattr( self.object, self.name_items, 0 )
        else:
            self.len_error( len( self ) - 1 )

    def len_error ( self, len ):
        raise TraitError, ( "The '%s' trait of %s instance must be %s, "
                  "but you attempted to change its length to %d element%s." % (
                  self.name, class_of( self.object ), self.trait.info(),
                  len, 's'[ len == 1: ] ) )
