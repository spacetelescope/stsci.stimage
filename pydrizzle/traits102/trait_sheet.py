#--------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from types  import StringType
from traits import Trait, HasTraits, ReadOnly, TraitPrefixList, SequenceTypes
from string import lowercase, uppercase

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

none_or_string = Trait( None, None, StringType )

true_trait  = Trait( 'true', {
                     'true':  1, 't': 1, 'yes': 1, 'y': 1, 'on':  1, 1: 1,
                     'false': 0, 'f': 0, 'no':  0, 'n': 0, 'off': 0, 0: 0
                     } )

style_trait = Trait( None, None, TraitPrefixList( 'simple', 'custom' ) )

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

    #-----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' trait of 'object' in a
    #  self-contained dialog:
    #-----------------------------------------------------------------------------

#  def popup_editor ( self, object, trait_name, description, handler,
#                           parent = None ):
#      return 0

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
        handler.changed( object, trait_name, value, original_value, 1 )

    #-----------------------------------------------------------------------------
    #  Return the text representation of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def str ( self, object, trait_name ):
        return str( getattr( object, trait_name ) )

#-------------------------------------------------------------------------------
#  'TraitGroupItem' class:
#-------------------------------------------------------------------------------

class TraitGroupItem ( HasTraits ):

    __traits__ = {
       'name':   none_or_string,
       'label':  none_or_string,
       'style':  style_trait,
       'editor': Trait( None, None, TraitEditor )
    }

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, *value, **traits ):
        HasTraits.__init__( self, **traits )
        if (len( value ) == 1) and (type( value[0] ) in SequenceTypes):
            value = value[0]
        for data in value:
            if type( data ) == StringType:
                if self.name is None:
                    self.name = data
                elif self.label is None:
                    self.label = data
                else:
                    self.style = data
            else:
                self.editor = data

    #----------------------------------------------------------------------------
    #  Return the user interface label for a specified object's trait:
    #----------------------------------------------------------------------------

    def label_for ( self, object ):
        return ( self.label or
                 object._base_trait( self.name ).label or
                 self.user_name_for( self.name ) )

    #----------------------------------------------------------------------------
    #  Return a 'user-friendly' name for a specified trait:
    #----------------------------------------------------------------------------

    def user_name_for ( self, name ):
        name       = name.replace( '_', ' ' )
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
        return self.editor or object._base_trait( self.name ).editor

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
       'object':      Trait( None, None, HasTraits )
    }

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, *values, **traits ):
        HasTraits.__init__( self, **traits )
        _values = []
        for value in values:
            if isinstance( value, TraitGroup ):
                _values.append( value )
            else:
                _values.append( TraitGroupItem( value ) )
        self.values = _values

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
