#--------------------------------------------------------------------------------
#
#  Define a wxPython based trait sheet mechanism for visually editing the
#  values of traits.
#
#  Written by: David C. Morrill
#
#  Date: 07/10/2002
#
#  (c) Copyright 2002 by Enthought, Inc.
#
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
#  Imports:
#--------------------------------------------------------------------------------

import sys
import os.path
import re

from wxPython    import wx
from traits      import Trait, TraitError, trait_editors
from trait_sheet import TraitEditor, TraitSheetHandler, \
                         TraitGroup, default_trait_sheet_handler
from types       import DictType, ListType, TupleType, ModuleType, \
                         StringType, FloatType
from math        import log10

#-------------------------------------------------------------------------------
#  Module initialization:
#-------------------------------------------------------------------------------

trait_editors( __name__ )

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Boolean values:
TRUE  = 1
FALSE = 0

# Basic sequence types:
basic_sequence_types = [ ListType, TupleType ]

# Standard width of an image bitmap:
standard_bitmap_width = 120

# Standard color samples:
color_choices = ( 0, 128, 192, 255 )
color_samples = [ None ] * 48
i             = 0
for r in color_choices:
    for g in color_choices:
        for b in ( 0, 128, 255 ):
            color_samples[i] = wx.wxColour( r, g, b )
            i += 1

# List of available font facenames:
facenames = None

# Standard font point sizes:
point_sizes = [
   '8',  '9', '10', '11', '12', '14', '16', '18',
  '20', '22', '24', '26', '28', '36', '48', '72'
]

# Global switch governing whether or not tooltips are displayed in trait
# sheet dialogs:
tooltips_enabled = TRUE

# Pattern of all digits:
all_digits = re.compile( r'\d+' )

# Color used to highlight input errors:
error_color = wx.wxColour( 255, 192, 192 )

#-------------------------------------------------------------------------------
#  'TraitSheetAppHandler' class:
#-------------------------------------------------------------------------------

class TraitSheetAppHandler ( TraitSheetHandler ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, app ):
        self.app      = app
        self.modified = FALSE

    #----------------------------------------------------------------------------
    #  Notification that a trait sheet has been closed:
    #----------------------------------------------------------------------------

    def close ( self, trait_sheet, object ):
        rc = TRUE
        if self.modified:
            dlg = wx.wxMessageDialog( trait_sheet,
                       'Changes have been made.\nDiscard changes?',
                       '%s Traits' % object.__class__.__name__  )
            result = dlg.ShowModal()
            dlg.Destroy()
            rc = (result == wx.wxID_OK)
        return rc

    #----------------------------------------------------------------------------
    #  Notification that a trait sheet has modified a trait of its
    #  associated object:
    #----------------------------------------------------------------------------

    def changed ( self, object, trait_name, new_value, old_value, is_set ):
        self.modified = TRUE
        self.save_button.Enable( TRUE )

    #----------------------------------------------------------------------------
    #  Create extra content to add to the trait sheet:
    #----------------------------------------------------------------------------

    def init ( self, trait_sheet, object ):
        self.sheet = trait_sheet
        vsizer     = wx.wxBoxSizer( wx.wxVERTICAL )
        hsizer     = wx.wxBoxSizer( wx.wxHORIZONTAL )
        vsizer.Add( wx.wxStaticLine( trait_sheet, -1 ), 1,
                    wx.wxEXPAND | wx.wxTOP, 4 )
        vsizer.Add( hsizer, 0, wx.wxALIGN_RIGHT )
        self.save_button = button = wx.wxButton( trait_sheet, -1, 'Save changes' )
        hsizer.Add( button, 0, wx.wxALL, 4 )
        wx.EVT_BUTTON( trait_sheet, button.GetId(), self.save )
        button.Enable( FALSE )
        button = wx.wxButton( trait_sheet, -1, 'Cancel' )
        hsizer.Add( button, 0, wx.wxALL, 4 )
        wx.EVT_BUTTON( trait_sheet, button.GetId(), trait_sheet.close_page )
        return vsizer

    #----------------------------------------------------------------------------
    #  Handle the user requesting that all changes to the object be saved:
    #----------------------------------------------------------------------------

    def save ( self, event ):
        self.modified    = FALSE
        self.app.save_ok = TRUE
        self.sheet.close_page()

#-------------------------------------------------------------------------------
#  'TraitSheetApp' class:
#-------------------------------------------------------------------------------

class TraitSheetApp ( wx.wxApp ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, object, traits = None ):
        self.object  = object
        self.traits  = traits
        self.save_ok = FALSE
        wx.wxApp.__init__( self, 1, 'debug.log' )
        self.MainLoop()

    #----------------------------------------------------------------------------
    #  Handle application initialization:
    #----------------------------------------------------------------------------

    def OnInit ( self ):
        self.SetTopWindow( TraitSheetDialog( self.object, self.traits,
                                             TraitSheetAppHandler( self ) ) )
        return TRUE

#--------------------------------------------------------------------------------
#  'TraitSheetDialog' class:
#--------------------------------------------------------------------------------

class TraitSheetDialog ( wx.wxDialog ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, object,
                         traits  = None,
                         handler = default_trait_sheet_handler,
                         parent  = None,
                         title   = None ):
        if title is None:
            title = '%s Traits' % object.__class__.__name__
        wx.wxDialog.__init__( self, parent, -1, title )
        wx.EVT_CLOSE( self, self.close_page )
        wx.EVT_CHAR(  self, self.on_key )

        self.object  = object
        self.handler = handler

        # Create the actual trait sheet panel:
        sizer = wx.wxBoxSizer( wx.wxVERTICAL )
        sizer.Add( TraitSheet( self, object, traits, handler ) )
        extra = handler.init( self, object )
        if extra is not None:
            sizer.Add( extra, 1, wx.wxEXPAND )
        self.SetSizer( sizer )
        sizer.Fit( self )

        # Find a nice place on the screen to display the trait sheet so
        # that it overlays the object as little as possible:
        if not handler.position( self, object ):
            self.Centre( wx.wxBOTH )

        self.Show( TRUE )

    #-----------------------------------------------------------------------------
    #  Close the trait sheet window:
    #-----------------------------------------------------------------------------

    def close_page ( self, event = None ):
        if self.handler.close( self, self.object ):
            self.Destroy()

    #----------------------------------------------------------------------------
    #  Handle the user hitting the 'Esc'ape key:
    #----------------------------------------------------------------------------

    def on_key ( self, event ):
        if event.GetKeyCode() == 0x1B:
            self.on_close_page( event )

#--------------------------------------------------------------------------------
#  'TraitPanel' class:
#--------------------------------------------------------------------------------

class TraitPanel ( wx.wxPanel ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, parent ):
        wx.wxPanel.__init__( self, parent, -1, style = wx.wxSUNKEN_BORDER )
        self.SetSizer( wx.wxBoxSizer( wx.wxVERTICAL ) )
        self._size        = None
        self._need_layout = TRUE

    #----------------------------------------------------------------------------
    #  Add a TraitSheet to the panel:
    #----------------------------------------------------------------------------

    def add ( self, sheet ):
        self.GetSizer().Add( sheet, 0, wx.wxEXPAND )
        self._size        = None
        self._need_layout = TRUE

    #----------------------------------------------------------------------------
    #  Remove a TraitSheet from the panel:
    #----------------------------------------------------------------------------

    def remove ( self, sheet ):
        sheet.Destroy()
        self._size        = None
        self._need_layout = TRUE

    #----------------------------------------------------------------------------
    #  Get the size of the panel:
    #----------------------------------------------------------------------------

    def size ( self ):
        if self._size is None:
            self._size = self.GetSizer().GetMinSize()
        return self._size

    #----------------------------------------------------------------------------
    #  Set the size and position of the panel:
    #----------------------------------------------------------------------------

    def position ( self, x, y, dx, dy ):
        self.SetDimensions( x, y, dx, dy )
        if self._need_layout:
            self._need_layout = FALSE
            self.Layout()

    #----------------------------------------------------------------------------
    #  Destroy the panel:
    #----------------------------------------------------------------------------

    def destroy ( self ):
        self.Destroy()

#--------------------------------------------------------------------------------
#  'TraitSheet' class:
#--------------------------------------------------------------------------------

class TraitSheet ( wx.wxPanel ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, parent,
                         object,
                         traits  = None,
                         handler = default_trait_sheet_handler ):
        wx.wxPanel.__init__( self, parent, -1 )

        self.object  = object
        self.handler = handler

        # If no traits were specified:
        if traits is None:
            # Get them from the specified object:
            traits = object.editable_traits()

        # Try to make sure that we now have either a single TraitGroup, or
        # a list of TraitGroups:
        kind = type( traits )
        if kind == StringType:
            # Convert the single trait name into a TraitGroup:
            traits = TraitGroup( traits, show_border = FALSE, style = 'custom' )
        elif ((kind in basic_sequence_types) and (len( traits ) > 0) and
              (not isinstance( traits[0], TraitGroup ))):
            # Convert a simple list of trait elements into a single,
            # TraitGroup, possibly containing multiple items:
            traits = TraitGroup( show_border = FALSE, style = 'custom', *traits )

        # Create the requested style of trait sheet editor:
        if isinstance( traits, TraitGroup ):
            # Single page dialog:
            sizer = self.add_page( traits, self )
            self.SetAutoLayout( TRUE )
            self.SetSizer( sizer )
            sizer.Fit( self )
        else:
            # Multi-tab notebook:
            self.add_tabs( traits )

    #-----------------------------------------------------------------------------
    #  Create a tab (i.e. notebook) style trait editor:
    #-----------------------------------------------------------------------------

    def add_tabs ( self, traits ):
        # Create the notebook, add it to the the dialog's sizer:
        nb    = wx.wxNotebook( self, -1 )
        nbs   = wx.wxNotebookSizer( nb )
        count = 0

        for pg in traits:
            # Create the new notebook page and add it to the notebook:
            panel     = wx.wxPanel( nb, -1 )
            page_name = pg.label
            if page_name is None:
                count    += 1
                page_name = 'Page %d' % count
            nb.AddPage( panel, page_name )
            sizer = self.add_page( pg, panel )
            panel.SetAutoLayout( TRUE )
            panel.SetSizer( sizer )

        nbs.Fit( nb )
        dx, dy = nb.GetSizeTuple()
        size   = wx.wxSize( max( len( traits ) * 54, 260, dx ), dy )
        nb.SetSize(   size )
        self.SetSize( size )

    #-----------------------------------------------------------------------------
    #  Create a single trait editor page:
    #-----------------------------------------------------------------------------

    def add_page ( self, pg, parent, box = None, default_style = 'simple' ):
        default_style = pg.style  or default_style
        object        = pg.object or self.object
        if pg.orientation == 'horizontal':
            rsizer = wx.wxBoxSizer( wx.wxVERTICAL )
            if box is not None:
                psizer = wx.wxStaticBoxSizer( box, wx.wxHORIZONTAL )
            else:
                psizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
            rsizer.Add( psizer, 0, wx.wxEXPAND )
        elif box is not None:
            rsizer = psizer = wx.wxStaticBoxSizer( box, wx.wxVERTICAL )
        else:
            rsizer = psizer = wx.wxBoxSizer( wx.wxVERTICAL )
        sizer       = None
        show_labels = pg.show_labels_
        for pge in pg.values:
            if isinstance( pge, TraitGroup ):
                box = None
                if pge.show_border_:
                    box = wx.wxStaticBox( parent, -1, pge.label or '' )
                psizer.Add( self.add_page( pge, parent, box, default_style ), 0,
                   wx.wxEXPAND | wx.wxALL, 6 )
            else:
                if sizer is None:
                    cols  = 1 + show_labels
                    sizer = wx.wxFlexGridSizer( 0, cols, 3, 6 )
                    if show_labels:
                        sizer.AddGrowableCol( 1 )

                name = pge.name or ' '

                if name == '-':
                    for i in range( cols ):
                        sizer.Add( wx.wxStaticLine( parent, -1 ), 0,
                                   wx.wxTOP | wx.wxBOTTOM | wx.wxEXPAND, 5 )
                    continue

                if name == ' ':
                    name = '5'
                if all_digits.match( name ):
                    n = int( name )
                    for i in range( cols ):
                        sizer.Add( n, n )
                    continue

                editor = pge.editor
                style  = pge.style or default_style
                if editor is None:
                    try:
                        editor = object._base_trait( name ).get_editor()
                    except:
                        pass
                if editor is None:
                    continue
                label = None
                if show_labels:
                    label = pge.label_for( object )
                self.add_trait( parent, sizer, object, name, label, editor,
                                style == 'simple' )

        if sizer is not None:
            psizer.Add( sizer, 1, wx.wxALL | wx.wxEXPAND, 6 )

        if rsizer != psizer:
            rsizer.Add( wx.wxPanel( parent, -1 ), 1, wx.wxEXPAND )

        return rsizer

    #-----------------------------------------------------------------------------
    #  Add a trait to the trait sheet:
    #-----------------------------------------------------------------------------

    def add_trait ( self, parent, sizer, object, trait_name, description, editor,
                          is_simple ):
        if description is not None:
            suffix = ':'[ description[-1:] == '?': ]
            label  = wx.wxStaticText( parent, -1,
                                      (description + suffix).capitalize(),
                                      style = wx.wxALIGN_RIGHT )
            sizer.Add( label, 0, wx.wxEXPAND )
            desc = object._base_trait( trait_name ).desc
            if desc is not None:
                label.SetToolTip( wx.wxToolTip( 'Specifies ' + desc ) )
                wx.EVT_RIGHT_UP( label, self.on_toggle_help )
        if is_simple:
            control = editor.simple_editor( object, trait_name, description,
                                            self.handler, parent )
            wx.EVT_RIGHT_UP( control, editor.on_restore )
        else:
            control = editor.custom_editor( object, trait_name, description,
                                            self.handler, parent )
        control.object         = object
        control.trait_name     = trait_name
        control.description    = description
        control.original_value = getattr( object, trait_name )
        control.handler        = self.handler
        sizer.Add( control, 1, editor.layout_style() )
        return control

    #-----------------------------------------------------------------------------
    #  Display a help message to the user indicating the purpose of a trait:
    #-----------------------------------------------------------------------------

    def on_toggle_help ( self, event ):
        global tooltips_enabled
        tooltips_enabled = not tooltips_enabled
        wx.wxToolTip_Enable( tooltips_enabled )

#--------------------------------------------------------------------------------
#  'wxTraitEditor' class:
#--------------------------------------------------------------------------------

class wxTraitEditor ( TraitEditor ):

    #-----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxTextCtrl( parent, -1, self.str( object, trait_name ),
                                 style = wx.wxTE_READONLY )
        wx.EVT_LEFT_UP( control, self.on_popup )
        return control

    #-----------------------------------------------------------------------------
    #  Invoke the pop-up editor for an object trait:
    #-----------------------------------------------------------------------------

    def on_popup ( self, event ):
        control = event.GetEventObject()
        if (hasattr( self, 'popup_editor' ) and
            self.popup_editor( control.object, control.trait_name,
                               control.description, control.handler, control )):
            self.update( control.object, control.trait_name, control )

    #-----------------------------------------------------------------------------
    #  Restore the original value of the object's trait:
    #-----------------------------------------------------------------------------

    def on_restore ( self, event ):
        control    = event.GetEventObject()
        object     = control.object
        trait_name = control.trait_name
        old_value  = getattr( object, trait_name )
        new_value  = control.original_value
        setattr( object, trait_name, new_value )
        control.handler.changed( object, trait_name, new_value, old_value,
                                 FALSE )
        self.update( object, trait_name, control )

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        control.SetValue( self.str( object, trait_name ) )

    #-----------------------------------------------------------------------------
    #  Handle a 'TraitError' exception:
    #-----------------------------------------------------------------------------

    def error ( self, description, excp, parent ):
        dlg = wx.wxMessageDialog( parent, str( excp ),
                                  description + ' value error',
                                  wx.wxOK | wx.wxICON_INFORMATION )
        dlg.ShowModal()
        dlg.Destroy()

    #----------------------------------------------------------------------------
    #  Return the layout style for this trait editor's control:
    #----------------------------------------------------------------------------

    def layout_style ( self ):
        return wx.wxEXPAND

#--------------------------------------------------------------------------------
#  'TraitEditorText' class:
#--------------------------------------------------------------------------------

class TraitEditorText ( wxTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, dic = {}, auto_set = TRUE ):
        self.dic      = dic
        self.auto_set = auto_set

    #-----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' text trait of 'object':
    #-----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             parent = None ):
        while TRUE:
            dialog = wx.wxTextEntryDialog( parent or object.window,
                          'Enter the new %s value:' % trait_name,
                          defaultValue = getattr( object, trait_name ) )
            if dialog.ShowModal() != wx.wxID_OK:
                return
            try:
                self.set( object, trait_name, self.get_value( dialog ),
                          handler )
                return TRUE
            except TraitError, excp:
                self.error( description, excp, object.window )
                return FALSE

    #-----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxTextCtrl( parent, -1, self.str( object, trait_name ),
                                 style = wx.wxTE_PROCESS_ENTER )
        wx.EVT_TEXT_ENTER( parent, control.GetId(), self.on_enter )
        if self.auto_set:
            wx.EVT_TEXT( parent, control.GetId(), self.on_key )
        return control

    #-----------------------------------------------------------------------------
    #  Handle the user pressing the 'Enter' key in the edit control:
    #-----------------------------------------------------------------------------

    def on_enter ( self, event ):
        control = event.GetEventObject()
        try:
            self.set( control.object, control.trait_name,
                      self.get_value( control ), control.handler )
        except TraitError, excp:
            self.error( control.description, excp, control )

    #-----------------------------------------------------------------------------
    #  Handle the user changing the contents of the edit control:
    #-----------------------------------------------------------------------------

    def on_key ( self, event ):
        owner = control = event.GetEventObject()
        if not hasattr( owner, 'object' ):
            owner = control.GetParent()
        try:
            setattr( owner.object, owner.trait_name, self.get_value( control ) )
            color = wx.wxWHITE
        except:
            color = error_color
        control.SetBackgroundColour( color )
        control.Refresh()

    #-----------------------------------------------------------------------------
    #  Get the actual value corresponding to what the user typed:
    #-----------------------------------------------------------------------------

    def get_value ( self, control ):
        value = control.GetValue().strip()
        if not self.dic.has_key( value ):
            return value
        return self.dic[ value ]

#--------------------------------------------------------------------------------
#  'TraitEditorEnum' class:
#--------------------------------------------------------------------------------

class TraitEditorEnum ( wxTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, values, cols = 1 ):
        self.cols   = cols
        self.mapped = (type( values ) == DictType)
        if self.mapped:
            sorted = values.values()
            sorted.sort()
            col = sorted[0].find( ':' ) + 1
            if col > 0:
                self.sorted = map( lambda x, c = col: x[ col: ], sorted )
                for n, v in values.items():
                    values[n] = v[ col: ]
            else:
                self.sorted = sorted
            self.values = values
        else:
            if not type( values ) in basic_sequence_types:
                handler = values
                if isinstance( handler, Trait ):
                    handler = handler.setter
                if hasattr( handler, 'map' ):
                    values = handler.map.keys()
                    values.sort()
                else:
                    values = handler.values
            self.values = map( str, values )

    #-----------------------------------------------------------------------------
    #  Create an in-place simple view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxChoice( parent, -1,
                     wx.wxPoint( 0, 0 ), wx.wxSize( 20, 20 ),
                     self.all_values() )
        try:
            control.SetStringSelection( self.current_value( object, trait_name ) )
        except:
            pass
        wx.EVT_CHOICE( parent, control.GetId(), self.on_value_changed )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        # Create a panel to hold all of the radio buttons:
        panel = wx.wxPanel( parent, -1 )

        # Get the current trait value:
        cur_value = self.current_value( object, trait_name )

        # Create a sizer to manage the radio buttons:
        values = self.all_values()
        n      = len( values )
        cols   = self.cols
        rows   = (n + cols - 1) / cols
        incr    = [ n / cols ] * cols
        rem     = n % cols
        for i in range( cols ):
            incr[i] += (rem > i)
        incr[-1] = -(reduce( lambda x, y: x + y, incr[:-1], 0 ) - 1)
        if cols > 1:
            sizer = wx.wxGridSizer( 0, cols, 2, 4 )
        else:
            sizer = wx.wxBoxSizer( wx.wxVERTICAL )

        # Add the set of all possible choices:
        style = wx.wxRB_GROUP
        index = 0
        for i in range( rows ):
            for j in range( cols ):
                if n > 0:
                    value   = values[ index ]
                    control = wx.wxRadioButton( panel, -1, value.capitalize(),
                                                style = style )
                    control.value = value
                    style         = 0
                    control.SetValue( value == cur_value )
                    wx.EVT_RADIOBUTTON( panel, control.GetId(), self.on_click )
                    index += incr[j]
                    n     -= 1
                else:
                    control = wx.wxRadioButton( panel, -1, '' )
                    control.value = ''
                    control.Show( FALSE )
                sizer.Add( control, 0, wx.wxNORTH, 5 )

        # Set-up the layout:
        panel.SetAutoLayout( TRUE )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Return the set of all possible values:
    #----------------------------------------------------------------------------

    def all_values ( self ):
        if self.mapped:
            return self.sorted
        return self.values

    #----------------------------------------------------------------------------
    #  Return the current value of the object trait:
    #----------------------------------------------------------------------------

    def current_value ( self, object, trait_name ):
        if self.mapped:
            return self.values[ getattr( object, trait_name + '_' ) ]
        return str( getattr( object, trait_name ) )

    #-----------------------------------------------------------------------------
    #  Handle the user selecting a new value from the combo box:
    #-----------------------------------------------------------------------------

    def on_value_changed ( self, event ):
        control = event.GetEventObject()
        value   = event.GetString()
        try:
            value = int( value )
        except:
            pass
        self.set( control.object, control.trait_name, value, control.handler )

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the 'custom' radio buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, event ):
        control = event.GetEventObject()
        parent  = control.GetParent()
        value   = control.value
        try:
            value = int( value )
        except:
            pass
        self.set( parent.object, parent.trait_name, value, parent.handler )

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        cur_value = self.current_value( object, trait_name )
        if isinstance( control, wx.wxPanel ):
            for button in control.GetChildren():
                button.SetValue( button.value == cur_value )
        else:
            try:
                control.SetStringSelection( cur_value )
            except:
                pass

#--------------------------------------------------------------------------------
#  'TraitEditorImageEnum' class:
#--------------------------------------------------------------------------------

class TraitEditorImageEnum ( TraitEditorEnum ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, values, suffix = '', cols = 1, path = None ):
        TraitEditorEnum.__init__( self, values )
        self.suffix = suffix
        self.cols   = cols
        if type( path ) == ModuleType:
            path = os.path.join( os.path.dirname( path.__file__ ), 'images' )
        self.path = path

    #-----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             parent = None ):
        return TraitEditorImageDialog( object, trait_name, description,
                                            parent, handler, self ).ShowModal()

    #-----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxBitmapButton( parent, -1,
                       bitmap_cache( self.current_value( object, trait_name ) +
                                     self.suffix, TRUE, self.path ) )
        control.SetBackgroundColour( wx.wxWHITE )
        wx.EVT_BUTTON( parent, control.GetId(), self.on_popup )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        # Create a panel to hold all of the radio buttons:
        panel = wx.wxPanel( parent, -1 )

        # Add the image buttons to the panel:
        self.create_image_grid( panel, self.on_click )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Populate a specified window with a grid of image buttons:
    #----------------------------------------------------------------------------

    def create_image_grid ( self, parent, handler ):
        # Create the main sizer:
        if self.cols > 1:
            sizer = wx.wxGridSizer( 0, self.cols, 2, 4 )
        else:
            sizer = wx.wxBoxSizer( wx.wxVERTICAL )

        # Add the set of all possible choices:
        for value in self.all_values():
            control = wx.wxBitmapButton( parent, -1,
                                         bitmap_cache( value + self.suffix,
                                                       FALSE, self.path ) )
            control.value = value
            wx.EVT_BUTTON( parent, control.GetId(), handler )
            sizer.Add( control, 0, wx.wxNORTH, 5 )

        # Finish setting up the control layout:
        parent.SetAutoLayout( TRUE )
        parent.SetSizer( sizer )
        sizer.Fit( parent )

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        control.SetBitmapLabel(
                bitmap_cache( self.current_value( object, trait_name ) +
                              self.suffix, TRUE, self.path ) )
        control.Refresh()

    #----------------------------------------------------------------------------
    #  Return the layout style for this trait editor's control:
    #----------------------------------------------------------------------------

    def layout_style ( self ):
        return 0

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the image buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, event ):
        control = event.GetEventObject()
        parent  = control.GetParent()
        self.set( parent.object, parent.trait_name, control.value,
                  parent.handler )

#--------------------------------------------------------------------------------
#  'TraitEditorList' class:
#--------------------------------------------------------------------------------

class TraitEditorList ( wxTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, values, cols = 1 ):
        self.cols   = cols
        self.values = values
        if type( values[0] ) == StringType:
            self.values = map( lambda x: ( x, x.capitalize() ), values )
        self.mapping = mapping = {}
        for value, key in self.values:
            mapping[ key ] = value

    #-----------------------------------------------------------------------------
    #  Create an in-place simple view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxChoice( parent, -1,
                     wx.wxPoint( 0, 0 ), wx.wxSize( 20, 20 ),
                     self.all_labels() )
        try:
            control.SetSelection( self.all_values().index(
                    self.current_value( object, trait_name )[0] ) )
        except:
            pass
        wx.EVT_CHOICE( parent, control.GetId(), self.on_value_changed )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        # Create a panel to hold all of the radio buttons:
        panel = wx.wxPanel( parent, -1 )

        # Get the current trait value:
        cur_value = self.current_value( object, trait_name )

        # Create a sizer to manage the radio buttons:
        labels = self.all_labels()
        values = self.all_values()
        n      = len( values )
        cols   = self.cols
        rows   = (n + cols - 1) / cols
        incr    = [ n / cols ] * cols
        rem     = n % cols
        for i in range( cols ):
            incr[i] += (rem > i)
        incr[-1] = -(reduce( lambda x, y: x + y, incr[:-1], 0 ) - 1)
        if cols > 1:
            sizer = wx.wxGridSizer( 0, cols, 2, 4 )
        else:
            sizer = wx.wxBoxSizer( wx.wxVERTICAL )

        # Add the set of all possible choices:
        index = 0
        for i in range( rows ):
            for j in range( cols ):
                if n > 0:
                    control = wx.wxCheckBox( panel, -1, labels[ index ] )
                    control.value = value = values[ index ]
                    control.SetValue( value in cur_value )
                    wx.EVT_CHECKBOX( panel, control.GetId(), self.on_click )
                    index += incr[j]
                    n     -= 1
                else:
                    control = wx.wxCheckBox( panel, -1, '' )
                    control.Show( FALSE )
                sizer.Add( control, 0, wx.wxNORTH, 5 )

        # Set-up the layout:
        panel.SetAutoLayout( TRUE )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Return the set of all possible labels:
    #----------------------------------------------------------------------------

    def all_labels ( self ):
        return map( lambda x: x[1], self.values )

    #----------------------------------------------------------------------------
    #  Return the set of all possible values:
    #----------------------------------------------------------------------------

    def all_values ( self ):
        return map( lambda x: x[0], self.values )

    #----------------------------------------------------------------------------
    #  Return whether or not the current value is a string or not:
    #----------------------------------------------------------------------------

    def is_string ( self, object, trait_name ):
        return (type( getattr( object, trait_name ) ) == StringType)

    #----------------------------------------------------------------------------
    #  Return the current value of the object trait:
    #----------------------------------------------------------------------------

    def current_value ( self, object, trait_name ):
        value = getattr( object, trait_name )
        if value is None:
            return []
        if type( value ) != StringType:
            return value
        return map( lambda x: x.strip(), value.split( ',' ) )

    #-----------------------------------------------------------------------------
    #  Handle the user selecting a new value from the combo box:
    #-----------------------------------------------------------------------------

    def on_value_changed ( self, event ):
        control = event.GetEventObject()
        value   = self.mapping[ event.GetString() ]
        if not self.is_string( control.object, control.trait_name ):
            value = [ value ]
        self.set( control.object, control.trait_name, value, control.handler )

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the 'custom' check boxes:
    #----------------------------------------------------------------------------

    def on_click ( self, event ):
        control   = event.GetEventObject()
        parent    = control.GetParent()
        value     = control.value
        cur_value = self.current_value( parent.object, parent.trait_name )
        if control.GetValue():
            cur_value.append( value )
        else:
            cur_value.remove( value )
        if self.is_string(  parent.object, parent.trait_name ):
            cur_value = ','.join( cur_value )
        self.set( parent.object, parent.trait_name, cur_value, parent.handler )

#--------------------------------------------------------------------------------
#  'TraitEditorBoolean' class:
#--------------------------------------------------------------------------------

class TraitEditorBoolean ( wxTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxCheckBox( parent, -1, '' )
        try:
            value = getattr( object, trait_name + '_' )
        except:
            value = getattr( object, trait_name )
        control.SetValue( value )
        wx.EVT_CHECKBOX( parent, control.GetId(), self.on_value_changed )
        return control

    #-----------------------------------------------------------------------------
    #  Handle the user clicking on the checkbox:
    #-----------------------------------------------------------------------------

    def on_value_changed ( self, event ):
        control = event.GetEventObject()
        self.set( control.object, control.trait_name, control.GetValue(),
                  control.handler )

#--------------------------------------------------------------------------------
#  'TraitEditorRange class:
#--------------------------------------------------------------------------------

class TraitEditorRange ( TraitEditorEnum ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, handler, cols = 1 ):
        if isinstance( handler, Trait ):
            handler = handler.setter
        self.low  = handler.low
        self.high = handler.high
        self.cols = cols

    #-----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        if type( self.low ) == FloatType:
            self.format = '%%.%df' % max( 0,
                                     4 - int( log10( self.high - self.low ) ) )
            panel       = wx.wxPanel( parent, -1 )
            sizer       = wx.wxBoxSizer( wx.wxHORIZONTAL )
            fvalue      = getattr( object, trait_name )
            try:
                fvalue_text = self.format % fvalue
                1 / (self.low <= fvalue <= self.high)
            except:
                fvalue_text = ''
                fvalue      = self.low
            ivalue   = int( ((fvalue - self.low) / (self.high - self.low))
                            * 10000 )
            label_lo = wx.wxStaticText( panel, -1, '999.999',
                                        style = wx.wxALIGN_RIGHT )
            sizer.Add( label_lo )
            panel.slider = slider = wx.wxSlider( panel, -1, ivalue, 0, 10000,
                           style = wx.wxSL_HORIZONTAL | wx.wxSL_AUTOTICKS )
            slider.SetTickFreq( 1000, 1 )
            slider.SetPageSize( 1000 )
            slider.SetLineSize( 100 )
            wx.EVT_SCROLL( slider, self.on_scroll )
            sizer.Add( slider, 1 )
            label_hi = wx.wxStaticText( panel, -1, '999.999' )
            sizer.Add( label_hi )
            panel.text = text = wx.wxTextCtrl( panel, -1, fvalue_text,
                                               style = wx.wxTE_PROCESS_ENTER )
            wx.EVT_TEXT_ENTER( panel, text.GetId(), self.on_enter )
            sizer.Add( text, 0, wx.wxLEFT, 8 )

            # Set-up the layout:
            panel.SetAutoLayout( TRUE )
            panel.SetSizer( sizer )
            sizer.Fit( panel )
            label_lo.SetLabel( str( self.low ) )
            label_hi.SetLabel( str( self.high ) )
            panel.is_enum = FALSE

            # Return the panel as the result:
            return panel

        if abs( self.high - self.low ) > 1000:
            control = wx.wxTextCtrl( parent, -1, self.str( object, trait_name ),
                                     style = wx.wxTE_PROCESS_ENTER )
            wx.EVT_TEXT_ENTER( parent, control.GetId(), self.on_enter )
            wx.EVT_TEXT(       parent, control.GetId(), self.on_key )

        else:
            value   = getattr( object, trait_name )
            control = wx.wxSpinCtrl( parent, -1, str( value ),
                                     min     = self.low,
                                     max     = self.high,
                                     initial = value )
            wx.EVT_SPINCTRL( parent, control.GetId(), self.on_value_changed )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        if ((type( self.low ) == FloatType) or
            (abs( self.high - self.low ) > 15)):
            return self.simple_editor( object, trait_name, description,
                                       handler, parent )
        control = TraitEditorEnum.custom_editor( self, object, trait_name,
                                                 description, handler, parent )
        control.is_enum = TRUE
        return control

    #----------------------------------------------------------------------------
    #  Return the set of all possible values:
    #----------------------------------------------------------------------------

    def all_values ( self ):
        return map( str, range( self.low, self.high + 1 ) )

    #----------------------------------------------------------------------------
    #  Return the current value of the object trait:
    #----------------------------------------------------------------------------

    def current_value ( self, object, trait_name ):
        return str( getattr( object, trait_name ) )

    #-----------------------------------------------------------------------------
    #  Handle the user selecting a new value from the spin control:
    #-----------------------------------------------------------------------------

    def on_value_changed ( self, event ):
        control = event.GetEventObject()
        self.set( control.object, control.trait_name, control.GetValue(),
                  control.handler )

    #-----------------------------------------------------------------------------
    #  Handle the user pressing the 'Enter' key in the edit control:
    #-----------------------------------------------------------------------------

    def on_enter ( self, event ):
        control = text = event.GetEventObject()
        slider  = None
        if not hasattr( control, 'object' ):
            control = control.GetParent()
            slider  = control.slider
        try:
            value = text.GetValue().strip()
            self.set( control.object, control.trait_name, value,
                      control.handler )
            if slider is not None:
                slider.SetValue( int( ((float( value ) - self.low) /
                                      (self.high - self.low)) * 10000 ) )
        except TraitError, excp:
            self.error( control.description, excp, control )

    #-----------------------------------------------------------------------------
    #  Handle the user changing the contents of the edit control:
    #-----------------------------------------------------------------------------

    def on_key ( self, event ):
        control = event.GetEventObject()
        try:
            setattr( control.object, control.trait_name,
                     control.GetValue().strip() )
            color = wx.wxWHITE
        except:
            color = error_color
        control.SetBackgroundColour( color )
        control.Refresh()

    #----------------------------------------------------------------------------
    #  Handle the user changing the current slider value:
    #----------------------------------------------------------------------------

    def on_scroll ( self, event ):
        control = event.GetEventObject().GetParent()
        value   = self.low + ((float( event.GetPosition() ) / 10000.0) *
                            (self.high - self.low))
        control.text.SetValue( self.format % value )
        if event.GetEventType() == wx.wxEVT_SCROLL_ENDSCROLL:
            self.set( control.object, control.trait_name, value,
                      control.handler )

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        if isinstance( control, wx.wxTextCtrl ):
            control.SetValue( self.str( object, trait_name ) )
        elif isinstance( control, wx.wxSpinCtrl ):
            try:
                control.SetValue( int( getattr( object, trait_name ) ) )
            except:
                pass
        elif control.is_enum:
            TraitEditorEnum.update( self, object, trait_name, control )
        else:
            fvalue = getattr( object, trait_name )
            try:
                fvalue_text = self.format % fvalue
                1 / (self.low <= fvalue <= self.high)
            except:
                fvalue_text = ''
                fvalue      = self.low
            ivalue = int( ((fvalue - self.low) / (self.high - self.low)) * 10000 )
            control.text.SetValue( fvalue_text )
            control.slider.SetValue( ivalue )

#--------------------------------------------------------------------------------
#  'TraitEditorComplex class:
#--------------------------------------------------------------------------------

class TraitEditorComplex ( TraitEditorText ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, editors, auto_set = TRUE ):
        TraitEditorText.__init__( self, auto_set = auto_set )
        self.editors = editors

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        # Create a panel to hold all of the component trait editors:
        panel = wx.wxPanel( parent, -1 )
        sizer = wx.wxBoxSizer( wx.wxVERTICAL )

        # Add all of the component trait editors:
        original_value  = getattr( object, trait_name )
        complex_handler = TraitEditorComplexHandler( handler )
        notifiers       = []
        for editor in self.editors:
            control = editor.custom_editor( object, trait_name, description,
                                            handler, panel )
            control.object         = object
            control.trait_name     = trait_name
            control.description    = description
            control.original_value = original_value
            control.handler        = complex_handler
            sizer.Add( control, 1, editor.layout_style() )
            notifiers.append( ( editor, control ) )

        # Save the list of notifiers in the complex handler:
        complex_handler.notifiers( notifiers )

        # Set-up the layout:
        panel.SetAutoLayout( TRUE )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        # Return the panel as the result:
        return panel

#-------------------------------------------------------------------------------
#  'TraitEditorComplexHandler' class:
#-------------------------------------------------------------------------------

class TraitEditorComplexHandler ( TraitSheetHandler ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, handler ):
        self.handler = handler

    #----------------------------------------------------------------------------
    #  Set the list of associated notifiers:
    #----------------------------------------------------------------------------

    def notifiers ( self, notifiers ):
        self._notifiers = notifiers

    #----------------------------------------------------------------------------
    #  Notification that a trait sheet has modified a trait of its
    #  associated object:
    #----------------------------------------------------------------------------

    def changed ( self, object, trait_name, new_value, old_value, is_set ):
        for editor, control in self._notifiers:
            editor.update( object, trait_name, control )
        self.handler.changed( object, trait_name, new_value, old_value, is_set )

#--------------------------------------------------------------------------------
#  'TraitEditorFile' class:
#--------------------------------------------------------------------------------

class TraitEditorFile ( TraitEditorText ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, filter = None ):
        TraitEditorText.__init__( self )
        self.filter = filter

    #-----------------------------------------------------------------------------
    #  Create an in-place simple view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        panel   = wx.wxPanel( parent, -1 )
        sizer   = wx.wxBoxSizer( wx.wxHORIZONTAL )
        control = wx.wxTextCtrl( panel, -1,
                                 self.str( object, trait_name ),
                                 style = wx.wxTE_PROCESS_ENTER )
        wx.EVT_TEXT_ENTER( panel, control.GetId(), self.on_enter )
        wx.EVT_TEXT(       panel, control.GetId(), self.on_key )
        sizer.Add( control, 1, wx.wxEXPAND )
        button          = wx.wxButton( panel, -1, 'Browse...' )
        button.filename = control
        sizer.Add( button, 0, wx.wxLEFT, 8 )
        wx.EVT_BUTTON( panel, button.GetId(), self.on_browse )
        panel.SetAutoLayout( TRUE )
        panel.SetSizer( sizer )
        sizer.Fit( panel )
        return panel

    #-----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' color trait of 'object':
    #-----------------------------------------------------------------------------

    def on_browse ( self, event ):
        control = event.GetEventObject()
        parent  = control.GetParent()
        dlg     = wx.wxFileDialog( control, message = 'Select a file' )
        dlg.SetFilename( control.filename.GetValue() )
        if self.filter is not None:
            dlg.SetWildcard( self.filter )
        rc       = (dlg.ShowModal() == wx.wxID_OK)
        filename = os.path.abspath( dlg.GetPath() )
        dlg.Destroy()
        if rc:
            control.filename.SetValue( filename )
            self.set( parent.object, parent.trait_name, filename, parent.handler )

#-------------------------------------------------------------------------------
#  'TraitEditorImageDialog' class:
#-------------------------------------------------------------------------------

class TraitEditorImageDialog ( wx.wxDialog ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, object, trait_name, description, parent, handler,
                         editor ):
        wx.wxDialog.__init__( self, parent, -1, 'Choose ' + description )
        wx.EVT_CLOSE( self, self.on_close_dialog )

        # Initialize instance data:
        self.object     = object
        self.trait_name = trait_name
        self.handler    = handler
        self.editor     = editor

        # Create the grid of image buttons:
        editor.create_image_grid( self, self.on_click )

        # Position the dialog:
        self.Centre( wx.wxBOTH )

    #-----------------------------------------------------------------------------
    #  Close the dialog:
    #-----------------------------------------------------------------------------

    def on_close_dialog ( self, event, rc = FALSE ):
        self.EndModal( rc )
        self.Destroy()

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the choice buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, event ):
        self.editor.set( self.object, self.trait_name,
                         event.GetEventObject().value, self.handler )
        self.on_close_dialog( event, TRUE )

#--------------------------------------------------------------------------------
#  Convert an image file name to a cached bitmap:
#--------------------------------------------------------------------------------

# Bitmap cache dictionary (indexed by filename):
_bitmap_cache = {}

### NOTE: This needs major improvements:
app_path        = None
traits_path = None

def bitmap_cache ( name, standard_size, path = None ):
    global app_path, traits_path
    if path is None:
        if traits_path is None:
            import traits
            traits_path = os.path.join(
                  os.path.dirname( traits.__file__ ), 'images' )
        path = traits_path
    elif path == '':
        if app_path is None:
            app_path = os.path.join( os.path.dirname( sys.argv[0] ),
                                       '..', 'images' )
        path = app_path
    filename = os.path.abspath( os.path.join( path,
                  name.replace( ' ', '_' ).lower() + '.gif' ) )
    bitmap   = _bitmap_cache.get( filename + ('*'[ not standard_size: ]) )
    if bitmap is not None:
        return bitmap
    std_bitmap = bitmap = wx.wxBitmapFromImage( wx.wxImage( filename ) )
    _bitmap_cache[ filename ] = bitmap
    dx = bitmap.GetWidth()
    if dx < standard_bitmap_width:
        dy = bitmap.GetHeight()
        std_bitmap = wx.wxEmptyBitmap( standard_bitmap_width, dy )
        dc1 = wx.wxMemoryDC()
        dc2 = wx.wxMemoryDC()
        dc1.SelectObject( std_bitmap )
        dc2.SelectObject( bitmap )
        dc1.SetPen( wx.wxTRANSPARENT_PEN )
        dc1.SetBrush( wx.wxWHITE_BRUSH )
        dc1.DrawRectangle( 0, 0, standard_bitmap_width, dy )
        dc1.Blit( (standard_bitmap_width - dx) / 2, 0, dx, dy, dc2, 0, 0 )
    _bitmap_cache[ filename + '*' ] = std_bitmap
    if standard_size:
        return std_bitmap
    return bitmap

#-------------------------------------------------------------------------------
#  Standard colors:
#-------------------------------------------------------------------------------

standard_colors = {
   'aquamarine':          wx.wxNamedColour( 'aquamarine' ),
   'black':               wx.wxNamedColour( 'black' ),
   'blue':                wx.wxNamedColour( 'blue' ),
   'blue violet':         wx.wxNamedColour( 'blue violet' ),
   'brown':               wx.wxNamedColour( 'brown' ),
   'cadet blue':          wx.wxNamedColour( 'cadet blue' ),
   'coral':               wx.wxNamedColour( 'coral' ),
   'cornflower blue':     wx.wxNamedColour( 'cornflower blue' ),
   'cyan':                wx.wxNamedColour( 'cyan' ),
   'dark grey':           wx.wxNamedColour( 'dark grey' ),
   'dark green':          wx.wxNamedColour( 'dark green' ),
   'dark olive green':    wx.wxNamedColour( 'dark olive green' ),
   'dark orchid':         wx.wxNamedColour( 'dark orchid' ),
   'dark slate blue':     wx.wxNamedColour( 'dark slate blue' ),
   'dark slate grey':     wx.wxNamedColour( 'dark slate grey' ),
   'dark turquoise':      wx.wxNamedColour( 'dark turquoise' ),
   'dim grey':            wx.wxNamedColour( 'dim grey' ),
   'firebrick':           wx.wxNamedColour( 'firebrick' ),
   'forest green':        wx.wxNamedColour( 'forest green' ),
   'gold':                wx.wxNamedColour( 'gold' ),
   'goldenrod':           wx.wxNamedColour( 'goldenrod' ),
   'grey':                wx.wxNamedColour( 'grey' ),
   'green':               wx.wxNamedColour( 'green' ),
   'green yellow':        wx.wxNamedColour( 'green yellow' ),
   'indian red':          wx.wxNamedColour( 'indian red' ),
   'khaki':               wx.wxNamedColour( 'khaki' ),
   'light blue':          wx.wxNamedColour( 'light blue' ),
   'light grey':          wx.wxNamedColour( 'light grey' ),
   'light steel':         wx.wxNamedColour( 'light steel' ),
   'blue':                wx.wxNamedColour( 'blue' ),
   'lime green':          wx.wxNamedColour( 'lime green' ),
   'magenta':             wx.wxNamedColour( 'magenta' ),
   'maroon':              wx.wxNamedColour( 'maroon' ),
   'medium aquamarine':   wx.wxNamedColour( 'medium aquamarine' ),
   'medium blue':         wx.wxNamedColour( 'medium blue' ),
   'medium forest green': wx.wxNamedColour( 'medium forest green' ),
   'medium goldenrod':    wx.wxNamedColour( 'medium goldenrod' ),
   'medium orchid':       wx.wxNamedColour( 'medium orchid' ),
   'medium sea green':    wx.wxNamedColour( 'medium sea green' ),
   'medium slate blue':   wx.wxNamedColour( 'medium slate blue' ),
   'medium spring green': wx.wxNamedColour( 'medium spring green' ),
   'medium turquoise':    wx.wxNamedColour( 'medium turquoise' ),
   'medium violet red':   wx.wxNamedColour( 'medium violet red' ),
   'midnight blue':       wx.wxNamedColour( 'midnight blue' ),
   'navy':                wx.wxNamedColour( 'navy' ),
   'orange':              wx.wxNamedColour( 'orange' ),
   'orange red':          wx.wxNamedColour( 'orange red' ),
   'orchid':              wx.wxNamedColour( 'orchid' ),
   'pale green':          wx.wxNamedColour( 'pale green' ),
   'pink':                wx.wxNamedColour( 'pink' ),
   'plum':                wx.wxNamedColour( 'plum' ),
   'purple':              wx.wxNamedColour( 'purple' ),
   'red':                 wx.wxNamedColour( 'red' ),
   'salmon':              wx.wxNamedColour( 'salmon' ),
   'sea green':           wx.wxNamedColour( 'sea green' ),
   'sienna':              wx.wxNamedColour( 'sienna' ),
   'sky blue':            wx.wxNamedColour( 'sky blue' ),
   'slate blue':          wx.wxNamedColour( 'slate blue' ),
   'spring green':        wx.wxNamedColour( 'spring green' ),
   'steel blue':          wx.wxNamedColour( 'steel blue' ),
   'tan':                 wx.wxNamedColour( 'tan' ),
   'thistle':             wx.wxNamedColour( 'thistle' ),
   'turquoise':           wx.wxNamedColour( 'turquoise' ),
   'violet':              wx.wxNamedColour( 'violet' ),
   'violet red':          wx.wxNamedColour( 'violet red' ),
   'wheat':               wx.wxNamedColour( 'wheat' ),
   'white':               wx.wxNamedColour( 'white' ),
   'yellow':              wx.wxNamedColour( 'yellow' ),
   'yellow green':        wx.wxNamedColour( 'yellow green' ),
}

#--------------------------------------------------------------------------------
#  Convert a number into a wxColour object:
#--------------------------------------------------------------------------------

def num_to_color ( object, name, value ):
    num = int( value )
    return wx.wxColour( num / 0x10000, (num / 0x100) & 0xFF, num & 0xFF )

num_to_color.info = ('a number, which in hex is of the form 0xRRGGBB, where '
                     'RR is red, GG is green, and BB is blue')

#--------------------------------------------------------------------------------
#  'TraitEditorColor' class:
#--------------------------------------------------------------------------------

class TraitEditorColor ( wxTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self ):
        self.color_data = wx.wxColourData()

    #-----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' color trait of 'object':
    #-----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             parent = None ):
        self.color_data.SetColour( self.to_wx_color( object, trait_name ) )
        self.color_data.SetChooseFull( TRUE )
        dialog = wx.wxColourDialog( parent or object.window, self.color_data )
        if dialog.ShowModal() == wx.wxID_OK:
            self.set( object, trait_name,
                      self.from_wx_color( dialog.GetColourData().GetColour() ),
                      handler )
            return TRUE
        return FALSE

    #-----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wxTraitEditor.simple_editor( self, object, trait_name,
                                                  description, handler, parent )
        self.update_color( object, trait_name, control )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        # Create a panel to hold all of the buttons:
        panel   = wx.wxPanel( parent, -1 )
        sizer   = wx.wxBoxSizer( wx.wxHORIZONTAL )
        control = panel.color = self.simple_editor( object, trait_name,
                                                    description, handler, panel )
        control.SetSize( wx.wxSize( 72, 72 ) )
        sizer.Add( control, 0, wx.wxEXPAND | wx.wxRIGHT, 4 )

        # Add all of the color choice buttons:
        sizer2 = wx.wxGridSizer( 0, 12, 0, 0 )

        for i in range( len( color_samples ) ):
            control = wx.wxButton( panel, -1, '',
                                   size = wx.wxSize( 18, 18 ) )
            control.SetBackgroundColour( color_samples[i] )
            wx.EVT_BUTTON( panel, control.GetId(), self.on_click )
            sizer2.Add( control )

        sizer.Add( sizer2 )

        # Set-up the layout:
        panel.SetAutoLayout( TRUE )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the 'custom' radio buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, event ):
        control = event.GetEventObject()
        parent  = control.GetParent()
        self.set( parent.object, parent.trait_name,
                  self.from_wx_color( control.GetBackgroundColour() ),
                  parent.handler )
        self.update( parent.object, parent.trait_name, parent.color )

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        wxTraitEditor.update( self, object, trait_name, control )
        self.update_color( object, trait_name, control )

    #-------------------------------------------------------------------------------
    #  Update the foreground/background colors of the control widget:
    #-------------------------------------------------------------------------------

    def update_color ( self, object, trait_name, control ):
        cur_color = self.to_wx_color( object, trait_name )
        control.SetBackgroundColour( cur_color )
        if ((cur_color.Red()   > 192) or
            (cur_color.Blue()  > 192) or
            (cur_color.Green() > 192)):
            control.SetForegroundColour( wx.wxBLACK )
        else:
            control.SetForegroundColour( wx.wxWHITE )

    #-------------------------------------------------------------------------------
    #  Get the wxPython color equivalent of the object trait:
    #-------------------------------------------------------------------------------

    def to_wx_color ( self, object, trait_name ):
        try:
            cur_color = getattr( object, trait_name + '_' )
        except:
            cur_color = getattr( object, trait_name )
        if cur_color is None:
            return wx.wxWHITE
        return cur_color

    #-------------------------------------------------------------------------------
    #  Get the application equivalent of a wxPython value:
    #-------------------------------------------------------------------------------

    def from_wx_color ( self, color ):
        return color

#-------------------------------------------------------------------------------
#  Define wxPython specific color traits:
#-------------------------------------------------------------------------------

# Create a singleton color editor:
color_editor = TraitEditorColor()

# Color traits:
color_trait       = Trait( 'white', wx.wxColour, wx.wxColourPtr,
                                 standard_colors, num_to_color,
                                 editor = color_editor )
clear_color_trait = Trait( 'clear', None, wx.wxColour,
                                 wx.wxColourPtr, standard_colors,
                                 { 'clear': None }, num_to_color,
                                 editor = color_editor )

#--------------------------------------------------------------------------------
#  Convert a string into a valid 'wxFont' object (if possible):
#--------------------------------------------------------------------------------

font_families = {
   'default':    wx.wxDEFAULT,
   'decorative': wx.wxDECORATIVE,
   'roman':      wx.wxROMAN,
   'script':     wx.wxSCRIPT,
   'swiss':      wx.wxSWISS,
   'modern':     wx.wxMODERN
}

font_styles = {
   'slant':  wx.wxSLANT,
   'italic': wx.wxITALIC
}

font_weights = {
   'light': wx.wxLIGHT,
   'bold':  wx.wxBOLD
}

font_noise = [ 'pt', 'point', 'family' ]

def str_to_font ( object, name, value ):
    try:
        point_size = 10
        family     = wx.wxDEFAULT
        style      = wx.wxNORMAL
        weight     = wx.wxNORMAL
        underline  = 0
        facename   = []
        for word in value.split():
            lword = word.lower()
            if font_families.has_key( lword ):
                family = font_families[ lword ]
            elif font_styles.has_key( lword ):
                style = font_styles[ lword ]
            elif font_weights.has_key( lword ):
                weight = font_weights[ lword ]
            elif lword == 'underline':
                underline = 1
            elif lword not in font_noise:
                try:
                    point_size = int( lword )
                except:
                    facename.append( word )
        return wx.wxFont( point_size, family, style, weight, underline,
                          ' '.join( facename ) )
    except:
        pass
    raise TraitError, ( object, name, 'a font descriptor string',
                           repr( value ) )

str_to_font.info = ( "a string describing a font (e.g. '12 pt bold italic "
                     "swiss family Arial' or 'default 12')" )

#--------------------------------------------------------------------------------
#  'TraitEditorFont' class:
#--------------------------------------------------------------------------------

class TraitEditorFont ( wxTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' font trait of 'object':
    #-----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             parent = None ):
        font_data = wx.wxFontData()
        font_data.SetInitialFont( self.to_wx_font( object, trait_name ) )
        dialog = wx.wxFontDialog( parent or object.window, font_data )
        if dialog.ShowModal() == wx.wxID_OK:
            self.set( object, trait_name,
                      self.from_wx_font( dialog.GetFontData().GetChosenFont() ),
                      handler )
            return TRUE
        return FALSE

    #-----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wxTraitEditor.simple_editor( self, object, trait_name,
                                               description, handler, parent )
        control.is_custom = FALSE
        self.update_font( object, trait_name, control )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        # Create a panel to hold all of the buttons:
        panel = wx.wxPanel( parent, -1 )
        sizer = wx.wxBoxSizer( wx.wxVERTICAL )

        # Add the standard font control:
        font = panel.font = self.simple_editor( object, trait_name,
                                                description, handler, panel )
        font.is_custom = TRUE
        sizer.Add( font, 0, wx.wxEXPAND | wx.wxBOTTOM, 3 )

        # Add all of the font choice controls:
        sizer2  = wx.wxBoxSizer( wx.wxHORIZONTAL )
        control = panel.facename = wx.wxChoice( panel, -1, wx.wxPoint( 0, 0 ),
                                      wx.wxSize( 20, 20 ), self.all_facenames() )
        sizer2.Add( control, 4, wx.wxEXPAND )
        wx.EVT_CHOICE( panel, control.GetId(), self.on_value_changed )

        control = panel.pointsize = wx.wxChoice( panel, -1, wx.wxPoint( 0, 0 ),
                                       wx.wxSize( 20, 20 ), point_sizes )
        sizer2.Add( control, 1, wx.wxEXPAND | wx.wxRIGHT, 3 )
        wx.EVT_CHOICE( panel, control.GetId(), self.on_value_changed )

        sizer.Add( sizer2, 0, wx.wxEXPAND )

        # Initialize the control's with the object's current trait value:
        self.update_font( object, trait_name, font )

        # Set-up the layout:
        panel.SetAutoLayout( TRUE )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        # Return the panel as the result:
        return panel

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        wxTraitEditor.update( self, object, trait_name, control )
        self.update_font( object, trait_name, control )

    #-------------------------------------------------------------------------------
    #  Update the font of the control widget:
    #-------------------------------------------------------------------------------

    def update_font ( self, object, trait_name, control ):
        cur_font = self.to_wx_font( object, trait_name )
        if control.is_custom:
            panel = control.GetParent()
            try:
                panel.facename.SetStringSelection( cur_font.GetFaceName() )
            except:
                panel.facename.SetSelection( 0 )
            try:
                panel.pointsize.SetStringSelection(
                         str( cur_font.GetPointSize() ) )
            except:
                panel.pointsize.SetSelection( 0 )
        cur_font.SetPointSize( min( 10, cur_font.GetPointSize() ) )
        control.SetFont( cur_font )

    #-----------------------------------------------------------------------------
    #  Return the text representation of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def str ( self, object, trait_name ):
        font   = getattr( object, trait_name )
        weight = { wx.wxLIGHT: ' Light',
                   wx.wxBOLD:  ' Bold'   }.get( font.GetWeight(), '' )
        style  = { wx.wxSLANT: ' Slant',
                   wx.wxITALIC:' Italic' }.get( font.GetStyle(), '' )
        return '%s point %s%s%s' % (
               font.GetPointSize(), font.GetFaceName(), style, weight )

    #----------------------------------------------------------------------------
    #  Return a list of all available font facenames:
    #----------------------------------------------------------------------------

    def all_facenames ( self ):
        global facenames
        if facenames is None:
            facenames = FontEnumerator().facenames()
            facenames.sort()
        return facenames

    #----------------------------------------------------------------------------
    #  Handle the user selecting a new facename or point size:
    #----------------------------------------------------------------------------

    def on_value_changed ( self, event ):
        control   = event.GetEventObject().GetParent()
        pointsize = int( control.pointsize.GetLabel() )
        facename  = control.facename.GetLabel()
        font      = wx.wxFont( pointsize, wx.wxDEFAULT, wx.wxNORMAL, wx.wxNORMAL,
                               faceName = facename )
        self.set( control.object, control.trait_name,
                  self.from_wx_font( font ), control.handler )
        self.update( control.object, control.trait_name, control.font )

    #-------------------------------------------------------------------------------
    #  Return a wxFont object corresponding to a specified object's font trait:
    #-------------------------------------------------------------------------------

    def to_wx_font ( self, object, trait_name ):
        return getattr( object, trait_name )

    #-------------------------------------------------------------------------------
    #  Get the application equivalent of a wxPython value:
    #-------------------------------------------------------------------------------

    def from_wx_font ( self, font ):
        return font

#-------------------------------------------------------------------------------
#  Define a wxPython specific font trait:
#-------------------------------------------------------------------------------

font_trait = Trait( 'Arial 10', wx.wxFont, wx.wxFontPtr, str_to_font,
                          editor = TraitEditorFont() )

#-------------------------------------------------------------------------------
#  'FontEnumerator' class:
#-------------------------------------------------------------------------------

class FontEnumerator ( wx.wxFontEnumerator ):

    #----------------------------------------------------------------------------
    #  Return a list of all available font facenames:
    #----------------------------------------------------------------------------

    def facenames ( self ):
        self._facenames = []
        self.EnumerateFacenames()
        return self._facenames

    #----------------------------------------------------------------------------
    #  Add a facename to the list of facenames:
    #----------------------------------------------------------------------------

    def OnFacename ( self, facename ):
        self._facenames.append( facename )
        return TRUE
