#-------------------------------------------------------------------------------
#
#  Define the set of abstract classes needed to define the notion of a
#  graphical 'trait sheet' for use with the 'traits' module.
#
#  Note: This module provides abstract definitions only. Concrete implementaions
#  are GUI toolkit specific and are provided by the following modules:
#
#     - wxtrait_sheet.py: wxPython
#     - tktrait_sheet.py: Tkinter
#
#  Written by: David C. Morrill
#
#  Date: 10/07/2002
#
#  (c) Copyright 2002 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
from __future__ import division # confidence high

from types          import StringType, ListType, TupleType
from trait_base     import SequenceTypes
from trait_handlers import TraitPrefixList
from traits         import Trait, HasTraits, ReadOnly
from string         import lowercase, uppercase

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

none_or_string = Trait( None, None, StringType )

true_trait  = Trait( 'true', {
                     'true':  1, 't': 1, 'yes': 1, 'y': 1, 'on':  1, 1: 1,
                     'false': 0, 'f': 0, 'no':  0, 'n': 0, 'off': 0, 0: 0
                     } )

style_trait = Trait( None, None,
                     TraitPrefixList( 'simple', 'custom', 'text', 'readonly' ) )

object_trait = Trait( None, None, HasTraits )

basic_sequence_types = [ ListType, TupleType ]

#-------------------------------------------------------------------------------
#  'TraitSheetHandler' class:
#-------------------------------------------------------------------------------

class TraitSheetHandler:

    #----------------------------------------------------------------------------
    #  Set the initial position of a trait sheet:
    #----------------------------------------------------------------------------

    def position ( self, trait_sheet, object ):
        return 0

    #----------------------------------------------------------------------------
    #  Notification that a trait sheet is being closed:
    #----------------------------------------------------------------------------

    def close ( self, trait_sheet, object ):
        return 1

    #----------------------------------------------------------------------------
    #  Notification that a trait sheet has modified a trait of its
    #  associated object:
    #----------------------------------------------------------------------------

    def changed ( self, object, trait_name, new_value, old_value, is_set ):
        pass

    #----------------------------------------------------------------------------
    #  Create extra content to add to the trait sheet:
    #----------------------------------------------------------------------------

    def init ( self, trait_sheet, object ):
        return None

# Create a default TraitSheetHandler:
default_trait_sheet_handler = TraitSheetHandler()

#-------------------------------------------------------------------------------
#  'TraitEditor' class:
#-------------------------------------------------------------------------------

class TraitEditor:

    #----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' trait of 'object' in a
    #  self-contained dialog:
    #----------------------------------------------------------------------------

#  def popup_editor ( self, object, trait_name, description, handler,
#                           parent = None ):
#      return 0

    #----------------------------------------------------------------------------
    #  Create a simple, imbeddable text view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def text_editor ( self, object, trait_name, description, handler,
                            parent ):
        raise NotImplementedError

    #----------------------------------------------------------------------------
    #  Create a simple, imbeddable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        raise NotImplementedError

    #----------------------------------------------------------------------------
    #  Create a custom, imbeddable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler,
                              parent ):
        return self.simple_editor( object, trait_name, description, handler,
                                   parent )

    #----------------------------------------------------------------------------
    #  Set a specified object trait value:
    #----------------------------------------------------------------------------

    def set ( self, object, trait_name, value, handler ):
        original_value = getattr( object, trait_name )
        setattr( object, trait_name, value )
        handler.changed( object, trait_name, value, original_value, True )

    #----------------------------------------------------------------------------
    #  Return the text representation of the 'trait' trait of 'object':
    #----------------------------------------------------------------------------

    def str ( self, object, trait_name ):
        return self.str_value( getattr( object, trait_name ) )

    #----------------------------------------------------------------------------
    #  Return the text representation of a specified object trait value:
    #----------------------------------------------------------------------------

    def str_value ( self, value ):
        return str( value )

#-------------------------------------------------------------------------------
#  'TraitMonitor' class:
#-------------------------------------------------------------------------------

class TraitMonitor:

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, object, trait_name, control, on_trait_change_handler ):
        self.control                 = control
        self.on_trait_change_handler = on_trait_change_handler
        object.on_trait_change( self.on_trait_change, trait_name )

    #----------------------------------------------------------------------------
    #  Handle an object trait being changed:
    #----------------------------------------------------------------------------

    def on_trait_change ( self, object, trait_name, new ):
        try:
            self.on_trait_change_handler( self.control, new )
        except:
            # NOTE: This code handles the case where a trait editor window has
            # been destroyed, but we don't know about it. Attempting to update the
            # now non-existent control generates an exception. We catch it here,
            # then disconnect the handler so it doesn't happen again:
            object.on_trait_change( self.on_trait_change, trait_name,
                                    remove = True )

#-------------------------------------------------------------------------------
#  'TraitGroupItem' class:
#-------------------------------------------------------------------------------

class TraitGroupItem ( HasTraits ):

    __traits__ = {
       'name':   none_or_string,
       'label':  none_or_string,
       'style':  style_trait,
       'editor': Trait( None, None, TraitEditor ),
       'object': object_trait
    }

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, *value, **traits ):
        HasTraits.__init__( self, **traits )
        if (len( value ) == 1) and (type( value[0] ) in SequenceTypes):
            value = value[0]
        for data in value:
            if type( data ) is StringType:
                if self.name is None:
                    self.name = data
                elif self.label is None:
                    self.label = data
                else:
                    self.style = data
            elif isinstance( data, TraitEditor ):
                self.editor = data
            else:
                self.object = data

    #---------------------------------------------------------------------------
    #  Create a clone of the object:
    #---------------------------------------------------------------------------

    def clone ( self ):
        clone = self.__class__()
        clone.clone_traits( self )
        return clone

    #----------------------------------------------------------------------------
    #  Return the user interface label for a specified object's trait:
    #----------------------------------------------------------------------------

    def label_for ( self, object ):
        return ( self.label or
                 (self.object or object)._base_trait( self.name ).label or
                 self.user_name_for( self.name ) )

    #----------------------------------------------------------------------------
    #  Return a 'user-friendly' name for a specified trait:
    #----------------------------------------------------------------------------

    def user_name_for ( self, name ):
        name       = name.replace( '_', ' ' ).capitalize()
        result     = ''
        last_lower = 0
        for c in name:
            if (c in uppercase) and last_lower:
                result += ' '
            last_lower = (c in lowercase)
            result    += c
        return result

    #----------------------------------------------------------------------------
    #  Return the TraitEditor object for a specified object's trait:
    #----------------------------------------------------------------------------

    def editor_for ( self, object ):
        return (self.editor or
                (self.object or object)._base_trait( self.name ).editor)

#-------------------------------------------------------------------------------
#  'TraitGroup' class:
#-------------------------------------------------------------------------------

class TraitGroup ( HasTraits ):

    __traits__ = {
       'values':      ReadOnly,
       'label':       none_or_string,
       'style':       style_trait,
       'orientation': Trait( 'vertical',
                             TraitPrefixList( 'vertical', 'horizontal' ) ),
       'show_border': true_trait,
       'show_labels': true_trait,
       'object':      object_trait
    }

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, *values, **traits ):
        HasTraits.__init__( self, **traits )
        _values = []
        for value in values:
            if (isinstance( value, TraitGroup ) or
                isinstance( value, TraitGroupItem )):
                _values.append( value )
            else:
                _values.append( TraitGroupItem( value ) )
        self.values = _values

    #---------------------------------------------------------------------------
    #  Create a clone of the object:
    #---------------------------------------------------------------------------

    def clone ( self ):
        clone = self.__class__()
        clone.clone_traits( self,
           [ 'label', 'style', 'orientation', 'show_border', 'show_labels' ] )
        clone_values_append = clone.values.append
        for value in self.values:
            clone_values_append( value.clone() )
        return clone

    #---------------------------------------------------------------------------
    #  Handle merging a TraitGroup with other editable traitsL
    #---------------------------------------------------------------------------

    def __add__ ( self, other ):
        return merge_trait_groups( self, other )

#-------------------------------------------------------------------------------
#  'TraitGroupList' class:
#-------------------------------------------------------------------------------

class TraitGroupList ( list ):

    #---------------------------------------------------------------------------
    #  Handle merging a TraitGroup with other editable traitsL
    #---------------------------------------------------------------------------

    def __add__ ( self, other ):
        return merge_trait_groups( self, other )

#-------------------------------------------------------------------------------
#  'MergeTraitGroups' class:
#-------------------------------------------------------------------------------

class MergeTraitGroups:

    #---------------------------------------------------------------------------
    #  Merge two trait groups:
    #---------------------------------------------------------------------------

    def __call__ ( self, group1, group2 ):
        return getattr( self, '%s_%s' % (
                        self._kind( group1 ), self._kind( group2 ) ) )(
                        group1, group2 )

    #---------------------------------------------------------------------------
    #  Return a string describing the kind of group specified:
    #---------------------------------------------------------------------------

    def _kind ( self, group ):
        if isinstance( group, TraitGroup ):
            return 'tg'
        if (isinstance( group, TraitGroupList ) or
            (type( group ) in basic_sequence_types)):
            if (len( group ) == 0) or (type( group[0] ) is StringType):
                return 'strl'
            return 'tgl'
        return 'str'

    #---------------------------------------------------------------------------
    #  Merge one TraitGroup into another:
    #---------------------------------------------------------------------------

    def _merge ( self, dest_group, src_group ):
        values = dest_group.values
        n      = len( values )
        for value in src_group.values:
            if isinstance( value, TraitGroupItem ) or (value.label is None):
                values.append( value )
            else:
                label = value.label
                for i in range( n ):
                    merge_item = values[i]
                    if (isinstance( merge_item, TraitGroup ) and
                        (label == merge_item.label)):
                        self._merge( merge_item, value )
                        break
                else:
                    values.append( value )

    #---------------------------------------------------------------------------
    #  Handle the various combinations of arguments:
    #---------------------------------------------------------------------------

    def str_str ( self, group1, group2 ):
        return TraitGroupList( [ group1, group2 ] )

    def str_strl ( self, group1, group2 ):
        return TraitGroupList( [ group1 ] + group2 )

    def str_tg ( self, group1, group2 ):
        return TraitGroupList( [ TraitGroup( group1, label = 'Main' ),
                                 group2 ] )

    def str_tgl ( self, group1, group2 ):
        return TraitGroupList( [ TraitGroup( group1, label = 'Main' ) ] +
                               group2 )

    def strl_str ( self, group1, group2 ):
        return TraitGroupList( group1 + [ group2 ] )

    def strl_strl ( self, group1, group2 ):
        return TraitGroupList( group1 + group2 )

    def strl_tg ( self, group1, group2 ):
        return TraitGroupList( [ TraitGroup( label = 'Main', *group1 ),
                                 group2 ] )

    def strl_tgl ( self, group1, group2 ):
        return TraitGroupList( [ TraitGroup( label = 'Main', *group1 ) ] +
                               group2 )

    def tg_str ( self, group1, group2 ):
        return TraitGroupList( [ group1,
                                 TraitGroup( group2, label = 'Other' ) ] )

    def tg_strl ( self, group1, group2 ):
        return TraitGroupList( [ group1,
                               TraitGroup( label = 'Other', *group2 ) ] )

    def tg_tg ( self, group1, group2 ):
        return self.tgl_tgl( [ group1 ], [ group2 ] )

    def tg_tgl ( self, group1, group2 ):
        return self.tgl_tgl( [ group1 ], group2 )

    def tgl_str ( self, group1, group2 ):
        return TraitGroupList( [ group1,
                               TraitGroup( group2, name = 'Other' ) ] )

    def tgl_strl ( self, group1, group2 ):
        return TraitGroupList( [ group1,
                                 TraitGroup( name = 'Other', *group2 ) ] )

    def tgl_tg ( self, group1, group2 ):
        return self.tgl_tgl( group1, [ group2 ] )

    def tgl_tgl ( self, group1, group2 ):
        result = TraitGroupList()
        page   = 0
        for group in group1:
            group = group.clone()
            if group.label is None:
                page += 1
                group.label = 'Page %d' % page
            result.append( group )
        for group in group2:
            label = group.label
            if label is None:
                page += 1
                group.label = 'Page %d' % page
                result.append( group )
            else:
                for merge_group in result:
                    if label == merge_group.label:
                        self._merge( merge_group, group )
                        break
                else:
                    result.append( group )
        return result

# Create a singleton instance which can be used to merge trait groups:
merge_trait_groups = MergeTraitGroups()

"""
A trait_element is:
  - A string (specifying a trait name)
  - A tuple containing 1 to 3 elements:
      - 1 string:  trait name
      - 2 strings: trait name and UI label
      - 1 string and 1 TraitEditor: trait name and editor
      - 2 strings and 1 TraitEditor: trait name, UI label and editor

A Trait Sheet description can be:
  - A string  (edit the single trait whose name is specified by the string)
  - A list of trait_elements: (simple non-tabbed trait sheet, vertically
           oriented, two-column using UI labels and simple editor)
  - A TraitGroup (simple non-tabbed trait sheet using layout specified by
           the TraitGroup contents)
  - A list of TraitGroup's (Each TraitGroup defines a notebook tab, the
      contents of which are defined by the TraitGroup's contents)

Each element passed to a TraitGroup constructor can be:
  - A trait_element (defines a single trait to be editor)
  - A TraitGroup (defines a nested group of traits to be edited)

Examples of Trait Sheet descriptions:

[ 'foo', 'bar', 'baz' ]

[ TraitGroup( 'foo', 'bar', 'baz',        label = 'Main' ),
  TraitGroup( 'color', 'style', 'weight', label = 'Border' )
]

[ TraitGroup( ( 'foo', 'Enter Foo' ),
          ( 'bar', 'Enter Bar', TraitEditBar() ),
            'baz', label = 'Main' ),
  TraitGroup( 'color', 'style', 'weight', label = 'Border' )
]

[ TraitGroup(
        TraitGroup( 'foo', 'bar', 'baz',
                  label = 'Group 1' ),
        TraitGroup( 'color', 'style', 'weight',
                  label       = 'Line',
                  border      = 'no',
                  orientation = 'horizontal',
                  show_labels = 'no',
                  style       = 'custom' ),
     label = 'Main' ),
"""
