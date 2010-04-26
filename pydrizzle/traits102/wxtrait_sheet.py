#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------
from __future__ import division # confidence medium

import sys
import os.path
import re

from string      import ascii_lowercase
from wxPython    import wx
from wxmenu      import MakeMenu
from traits      import Trait, HasTraits, TraitError, HasDynamicTraits, \
                        trait_editors
from trait_sheet import TraitEditor, TraitSheetHandler, TraitMonitor, \
                        TraitGroup, TraitGroupList, default_trait_sheet_handler
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
tooltips_enabled = True

# Pattern of all digits:
all_digits = re.compile( r'\d+' )

# Color used to highlight input errors:
error_color = wx.wxColour( 255, 192, 192 )

# Width of a scrollbar:
scrollbar_dx = wx.wxSystemSettings_GetSystemMetric( wx.wxSYS_VSCROLL_X )

# Screen size:
screen_dx    = wx.wxSystemSettings_GetSystemMetric( wx.wxSYS_SCREEN_X )
screen_dy    = wx.wxSystemSettings_GetSystemMetric( wx.wxSYS_SCREEN_Y )

#-------------------------------------------------------------------------------
#  Position one window near another:
#-------------------------------------------------------------------------------

def position_near ( origin, target ):
    x, y   = origin.ClientToScreenXY( 0, 0 )
    y     -= 30   # Approximate adjustment for window title bar
    dx, dy = target.GetSizeTuple()
    if (x + dx) > screen_dx:
        x = screen_dx - dx
    if x < 0:
        x = 0
    if (y + dy) > screen_dy:
        y = screen_dy - dy
    if y < 0:
        y = 0
    target.SetPosition( wx.wxPoint( x, y ) )

#-------------------------------------------------------------------------------
#  Initialize an editor control:
#-------------------------------------------------------------------------------

class Undefined: pass

def init_control ( control, object, trait_name, description, handler,
                   original_value = Undefined ):
    control.object      = object
    control.trait_name  = trait_name
    control.description = description
    control.handler     = handler
    if original_value is Undefined:
        control.original_value = getattr( object, trait_name )
    else:
        control.original_value = original_value

#-------------------------------------------------------------------------------
#  'TraitSheetAppHandler' class:
#-------------------------------------------------------------------------------

class TraitSheetAppHandler ( TraitSheetHandler ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, app ):
        self.app      = app
        self.modified = False

    #----------------------------------------------------------------------------
    #  Notification that a trait sheet has been closed:
    #----------------------------------------------------------------------------

    def close ( self, trait_sheet, object ):
        rc = True
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
        self.modified = True
        self.save_button.Enable( True )

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
        self.save_button = button = wx.wxButton( trait_sheet, -1,
                                                 'Save changes' )
        hsizer.Add( button, 0, wx.wxALL, 4 )
        wx.EVT_BUTTON( trait_sheet, button.GetId(), self.save )
        button.Enable( False )
        button = wx.wxButton( trait_sheet, -1, 'Cancel' )
        hsizer.Add( button, 0, wx.wxALL, 4 )
        wx.EVT_BUTTON( trait_sheet, button.GetId(), trait_sheet.close_page )
        return vsizer

    #----------------------------------------------------------------------------
    #  Handle the user requesting that all changes to the object be saved:
    #----------------------------------------------------------------------------

    def save ( self, event ):
        self.modified    = False
        self.app.save_ok = True
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
        self.save_ok = False
        wx.wxInitAllImageHandlers()
        wx.wxApp.__init__( self, 1, 'debug.log' )
        self.MainLoop()

    #----------------------------------------------------------------------------
    #  Handle application initialization:
    #----------------------------------------------------------------------------

    def OnInit ( self ):
        self.SetTopWindow( TraitSheetDialog( self.object, self.traits,
                                             TraitSheetAppHandler( self ) ) )
        return True

#-------------------------------------------------------------------------------
#  'TraitSheetDialog' class:
#-------------------------------------------------------------------------------

class TraitSheetDialog ( wx.wxDialog ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

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
        sizer       = wx.wxBoxSizer( wx.wxVERTICAL )
        sw          = wx.wxScrolledWindow( self )
        trait_sheet = TraitSheet( sw, object, traits, handler )
        sizer.Add( trait_sheet, 0, wx.wxALL, 4 )
        tsdx, tsdy = trait_sheet.GetSizeTuple()
        tsdx += 8
        tsdy += 8
        extra = handler.init( self, object )
        if extra is not None:
            sizer.Add( extra, 1, wx.wxEXPAND )

        max_dy = (2 * screen_dy) // 3
        sw.SetAutoLayout( True )
        sw.SetSizer( sizer )
        sw.SetSize( wx.wxSize( tsdx + ((tsdy > max_dy) * scrollbar_dx),
                               min( tsdy, max_dy ) ) )
        sw.SetScrollRate( 0, 1 )

        sw_sizer = wx.wxBoxSizer( wx.wxVERTICAL )
        sw_sizer.Add( sw )
        sw_sizer.Fit( self )

        # Find a nice place on the screen to display the trait sheet so
        # that it overlays the object as little as possible:
        if not handler.position( self, object ):
            if parent is None:
                self.Centre( wx.wxBOTH )
            else:
                position_near( parent, self )

        self.Show( True )

    #----------------------------------------------------------------------------
    #  Close the trait sheet window:
    #----------------------------------------------------------------------------

    def close_page ( self, event = None ):
        if self.handler.close( self, self.object ):
            self.Destroy()

    #----------------------------------------------------------------------------
    #  Handle the user hitting the 'Esc'ape key:
    #----------------------------------------------------------------------------

    def on_key ( self, event ):
        if event.GetKeyCode() == 0x1B:
            self.on_close_page( event )

#-------------------------------------------------------------------------------
#  'TraitPanel' class:
#-------------------------------------------------------------------------------

class TraitPanel ( wx.wxPanel ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, parent ):
#       wx.wxPanel.__init__( self, parent, -1, style = wx.wxSUNKEN_BORDER )
        wx.wxPanel.__init__( self, parent, -1 )
        self.SetSizer( wx.wxBoxSizer( wx.wxVERTICAL ) )
        self._size        = None
        self._need_layout = True

    #----------------------------------------------------------------------------
    #  Add a TraitSheet to the panel:
    #----------------------------------------------------------------------------

    def add ( self, sheet ):
        self.GetSizer().Add( sheet, 0, wx.wxEXPAND )
        self._size        = None
        self._need_layout = True

    #----------------------------------------------------------------------------
    #  Remove a TraitSheet from the panel:
    #----------------------------------------------------------------------------

    def remove ( self, sheet ):
        sheet.Destroy()
        self._size        = None
        self._need_layout = True

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
            self._need_layout = False
            self.Layout()

    #----------------------------------------------------------------------------
    #  Destroy the panel:
    #----------------------------------------------------------------------------

    def destroy ( self ):
        self.Destroy()

#-------------------------------------------------------------------------------
#  'TraitSheet' class:
#-------------------------------------------------------------------------------

class TraitSheet ( wx.wxPanel ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

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
        if kind is StringType:
            # Convert the single trait name into a TraitGroup:
            traits = TraitGroup( traits, show_border = False, style = 'custom' )
        elif ((kind in basic_sequence_types) or
              isinstance( traits, TraitGroupList )):
            if len( traits ) == 0:
                # Empty trait list, leave the panel empty:
                return
            if not isinstance( traits[0], TraitGroup ):
                # Convert a simple list of trait elements into a single,
                # TraitGroup, possibly containing multiple items:
                traits = TraitGroup( show_border = False,
                                     style       = 'custom', *traits )

        # Create the requested style of trait sheet editor:
        if isinstance( traits, TraitGroup ):
            # Single page dialog:
            sizer = self.add_page( traits, self )
            self.SetAutoLayout( True )
            self.SetSizer( sizer )
            sizer.Fit( self )
        else:
            # Multi-tab notebook:
            self.add_tabs( traits )

    #----------------------------------------------------------------------------
    #  Create a tab (i.e. notebook) style trait editor:
    #----------------------------------------------------------------------------

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
            panel.SetAutoLayout( True )
            panel.SetSizer( sizer )

        nbs.Fit( nb )
        dx, dy = nb.GetSizeTuple()
        size   = wx.wxSize( max( len( traits ) * 54, 260, dx ), dy )
        nb.SetSize(   size )
        self.SetSize( size )

    #----------------------------------------------------------------------------
    #  Create a single trait editor page:
    #----------------------------------------------------------------------------

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
                   wx.wxEXPAND | wx.wxALL, 2 )
            else:
                if sizer is None:
                    cols  = 1 + show_labels
                    sizer = wx.wxFlexGridSizer( 0, cols, 2, 4 )
                    if show_labels:
                        sizer.AddGrowableCol( 1 )

                name = pge.name or ' '

                if name == '-':
                    for i in range( cols ):
                        sizer.Add( wx.wxStaticLine( parent, -1 ), 0,
                                   wx.wxTOP | wx.wxBOTTOM | wx.wxEXPAND, 2 )
                    continue

                if name == ' ':
                    name = '5'
                if all_digits.match( name ):
                    n = int( name )
                    for i in range( cols ):
                        sizer.Add( n, n )
                    continue

                editor     = pge.editor
                style      = pge.style or default_style
                pge_object = pge.object or object
                if editor is None:
                    try:
                        editor = pge_object._base_trait( name ).get_editor()
                    except:
                        pass
                if editor is None:
                    continue
                label = None
                if show_labels:
                    label = pge.label_for( object )
                self.add_trait( parent, sizer, pge_object, name, label, editor,
                                style )

        if sizer is not None:
            psizer.Add( sizer, 1, wx.wxALL | wx.wxEXPAND, 1 )

        if rsizer is not psizer:
            rsizer.Add( wx.wxPanel( parent, -1 ), 1, wx.wxEXPAND )

        return rsizer

    #----------------------------------------------------------------------------
    #  Add a trait to the trait sheet:
    #----------------------------------------------------------------------------

    def add_trait ( self, parent, sizer, object, trait_name, description, editor,
                          style ):
        if description is not None:
            suffix = ':'[ description[-1:] == '?': ]
            label  = wx.wxStaticText( parent, -1,
                                      description + suffix,
                                      style = wx.wxALIGN_RIGHT )
            sizer.Add( label, 0, wx.wxEXPAND | wx.wxALIGN_CENTER )
            desc = object._base_trait( trait_name ).desc
            if desc is not None:
                label.SetToolTip( wx.wxToolTip( 'Specifies ' + desc ) )
                wx.EVT_RIGHT_UP( label, self.on_toggle_help )
        if style == 'custom':
            control = editor.custom_editor( object, trait_name, description,
                                            self.handler, parent )
        elif style == 'readonly':
            control = editor.readonly_editor( object, trait_name, description,
                                              self.handler, parent )
        else:
            if style == 'simple':
                control = editor.simple_editor( object, trait_name, description,
                                                self.handler, parent )
            else:
                control = editor.text_editor( object, trait_name, description,
                                              self.handler, parent )
            wx.EVT_RIGHT_UP( control, editor.on_restore )

        init_control( control, object, trait_name, description, self.handler )
        sizer.Add( control, 1, editor.layout_style() )
        return control

    #----------------------------------------------------------------------------
    #  Display a help message to the user indicating the purpose of a trait:
    #----------------------------------------------------------------------------

    def on_toggle_help ( self, event ):
        global tooltips_enabled
        tooltips_enabled = not tooltips_enabled
        wx.wxToolTip_Enable( tooltips_enabled )

#-------------------------------------------------------------------------------
#  'wxTraitEditor' class:
#-------------------------------------------------------------------------------

class wxTraitEditor ( TraitEditor ):

    #----------------------------------------------------------------------------
    #  Create an in-place read only view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def readonly_editor ( self, object, trait_name, description, handler,
                                parent ):
        control = wx.wxTextCtrl( parent, -1, self.str( object, trait_name ),
                                 style = wx.wxTE_READONLY )
        TraitMonitor( object, trait_name, control, self.on_trait_change_text )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place text editable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def text_editor ( self, object, trait_name, description, handler,
                            parent ):
        control = wx.wxTextCtrl( parent, -1, self.str( object, trait_name ),
                                 style = wx.wxTE_PROCESS_ENTER )
        wx.EVT_KILL_FOCUS( control, self.on_text_enter )
        wx.EVT_TEXT_ENTER( parent, control.GetId(), self.on_text_enter )
        TraitMonitor( object, trait_name, control, self.on_trait_change_text )
        return control

    #----------------------------------------------------------------------------
    #  Handle the user pressing the 'Enter' key in the edit control:
    #----------------------------------------------------------------------------

    def on_text_enter ( self, event ):
        control = event.GetEventObject()
        try:
            self.set( control.object, control.trait_name,
                      control.GetValue(), control.handler )
            return True
        except TraitError, excp:
            self.error( control.description, excp, control )
            return False

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_text ( self, control, new ):
        if not hasattr( control, 'updating' ):
            new_value = self.str_value( new )
            if control.GetValue() != new_value:
                control.SetValue( new_value )

    #----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxTextCtrl( parent, -1, self.str( object, trait_name ),
                                 style = wx.wxTE_READONLY )
        wx.EVT_LEFT_UP( control, self.on_popup )
        TraitMonitor( object, trait_name, control, self.on_trait_change_text )
        return control

    #----------------------------------------------------------------------------
    #  Invoke the pop-up editor for an object trait:
    #----------------------------------------------------------------------------

    def on_popup ( self, event ):
        self.on_popup_control( event.GetEventObject() )

    def on_popup_control ( self, control ):
        if hasattr( self, 'popup_editor' ):
            self.popup_editor( control.object, control.trait_name,
                               control.description, control.handler, control )

    #----------------------------------------------------------------------------
    #  Restore the original value of the object's trait:
    #----------------------------------------------------------------------------

    def on_restore ( self, event ):
        control    = event.GetEventObject()
        object     = control.object
        trait_name = control.trait_name
        old_value  = getattr( object, trait_name )
        new_value  = control.original_value
        setattr( object, trait_name, new_value )
        control.handler.changed( object, trait_name, new_value, old_value,
                                 False )

    #----------------------------------------------------------------------------
    #  Handle a 'TraitError' exception:
    #----------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------
#  'TraitEditorInstance' class:
#-------------------------------------------------------------------------------

class TraitEditorInstance ( wxTraitEditor ):

    #----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        value = getattr( object, trait_name )
        if value is None:
            control = wx.wxButton( parent, -1, 'None' )
            control.Enable( False )
        else:
            control = wx.wxButton( parent, -1, value.__class__.__name__ )
        wx.EVT_BUTTON( parent, control.GetId(), self.on_popup )
        TraitMonitor( object, trait_name, control, self.on_trait_change_text )
        return control

    #----------------------------------------------------------------------------
    #  Invoke the pop-up editor for an object trait:
    #----------------------------------------------------------------------------

    def on_popup_control ( self, control ):
        object = getattr( control.object, control.trait_name )
        if isinstance( object, HasTraits ):
            traits = object.editable_traits()
            try:
                wrap = (not isinstance( traits, TraitGroup ))
                if wrap:
                    wrap = (not isinstance( traits[0], TraitGroup ))
            except:
                wrap = True
            if wrap:
                traits = TraitGroup( show_border = False,
                                     style       = 'simple',
                                     *traits )
            TraitSheetDialog( object, traits, control.handler, control )

#-------------------------------------------------------------------------------
#  'TraitEditorText' class:
#-------------------------------------------------------------------------------

class TraitEditorText ( wxTraitEditor ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, dic = {}, auto_set = True, evaluate = False ):
        self.dic      = dic
        self.auto_set = auto_set
        self.evaluate = evaluate

    #----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' text trait of 'object':
    #----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             parent = None ):
        while True:
            dialog = wx.wxTextEntryDialog( parent or object.window,
                          'Enter the new %s value:' % trait_name,
                          defaultValue = getattr( object, trait_name ) )
            if dialog.ShowModal() != wx.wxID_OK:
                return
            try:
                self.set( object, trait_name, self.get_value( dialog ),
                          handler )
                return True
            except TraitError, excp:
                self.error( description, excp, object.window )
                return False

    #----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxTextCtrl( parent, -1, self.str( object, trait_name ),
                                 style = wx.wxTE_PROCESS_ENTER )
        wx.EVT_KILL_FOCUS( control, self.on_enter )
        wx.EVT_TEXT_ENTER( parent, control.GetId(), self.on_enter )
        if self.auto_set:
            wx.EVT_TEXT( parent, control.GetId(), self.on_key )
        TraitMonitor( object, trait_name, control, self.on_trait_change_text )
        return control

    #----------------------------------------------------------------------------
    #  Handle the user pressing the 'Enter' key in the edit control:
    #----------------------------------------------------------------------------

    def on_enter ( self, event ):
        control = event.GetEventObject()
        try:
            self.set( control.object, control.trait_name,
                      self.get_value( control ), control.handler )
        except TraitError, excp:
            self.error( control.description, excp, control )

    #----------------------------------------------------------------------------
    #  Handle the user changing the contents of the edit control:
    #----------------------------------------------------------------------------

    def on_key ( self, event ):
        owner = control = event.GetEventObject()
        control.updating = True
        if not hasattr( owner, 'object' ):
            owner = control.GetParent()
        try:
            self.set( owner.object, owner.trait_name,
                      self.get_value( control ), owner.handler )
            color = wx.wxWHITE
        except:
            color = error_color
        del control.updating
        control.SetBackgroundColour( color )
        control.Refresh()

    #----------------------------------------------------------------------------
    #  Get the actual value corresponding to what the user typed:
    #----------------------------------------------------------------------------

    def get_value ( self, control ):
        value = control.GetValue().strip()
        if self.evaluate:
            value = eval( value )
        if not self.dic.has_key( value ):
            return value
        return self.dic[ value ]

#-------------------------------------------------------------------------------
#  'TraitEditorEnum' class:
#-------------------------------------------------------------------------------

class TraitEditorEnum ( wxTraitEditor ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, values, cols = 1 ):
        self.cols   = cols
        self.mapped = (type( values ) is DictType)
        if self.mapped:
            sorted = values.values()
            sorted.sort()
            col = sorted[0].find( ':' ) + 1
            if col > 0:
                self.sorted = [ x[ col: ] for x in sorted ]
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
            self.values = [ str( x ) for x in values ]

    #----------------------------------------------------------------------------
    #  Create an in-place simple view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxChoice( parent, -1,
                     wx.wxPoint( 0, 0 ), wx.wxSize( 100, 20 ),
                     self.all_values() )
        try:
            control.SetStringSelection( self.current_value( object, trait_name ) )
        except:
            pass
        wx.EVT_CHOICE( parent, control.GetId(), self.on_value_changed )
        TraitMonitor( object, trait_name, control, self.on_trait_change_choice )
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
        rows   = (n + cols - 1) // cols
        incr   = [ n // cols ] * cols
        rem    = n % cols
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
                    value = label = values[ index ]
                    if label[:1] in ascii_lowercase:
                        label = label.capitalize()
                    control = wx.wxRadioButton( panel, -1, label, style = style )
                    control.value = value
                    style         = 0
                    control.SetValue( value == cur_value )
                    wx.EVT_RADIOBUTTON( panel, control.GetId(), self.on_click )
                    index += incr[j]
                    n     -= 1
                else:
                    control = wx.wxRadioButton( panel, -1, '' )
                    control.value = ''
                    control.Show( False )
                sizer.Add( control, 0, wx.wxNORTH, 5 )

        # Set-up the layout:
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        if self.mapped:
            trait_name += '_'
        TraitMonitor( object, trait_name, panel, self.on_trait_change_radio )

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

    #----------------------------------------------------------------------------
    #  Handle the user selecting a new value from the combo box:
    #----------------------------------------------------------------------------

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

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_choice ( self, control, new ):
        try:
            control.SetStringSelection( str( new ) )
        except:
            pass

    def on_trait_change_radio ( self, control, new ):
        for button in control.GetChildren():
            button.SetValue( button.value == new )

#-------------------------------------------------------------------------------
#  'TraitEditorImageEnum' class:
#-------------------------------------------------------------------------------

class TraitEditorImageEnum ( TraitEditorEnum ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, values, suffix = '', cols = 1, path = None ):
        TraitEditorEnum.__init__( self, values )
        self.suffix = suffix
        self.cols   = cols
        if type( path ) is ModuleType:
            path = os.path.join( os.path.dirname( path.__file__ ), 'images' )
        self.path = path

    #----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             parent = None ):
        return TraitEditorImageDialog( object, trait_name, description,
                                       parent, handler, self ).ShowModal()

    #----------------------------------------------------------------------------
    #  Create an in-place read only view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def readonly_editor ( self, object, trait_name, description, handler,
                                parent ):
        control = ImageControl( parent,
                     bitmap_cache( self.current_value( object, trait_name ) +
                                   self.suffix, False, self.path ) )
        #control.SetBackgroundColour( wx.wxWHITE )
        TraitMonitor( object, trait_name, control, self.on_trait_change_image )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = self.readonly_editor( object, trait_name, description, handler,
                                        parent )
        control.Selected( True )
        control.Handler( self.on_popup_control )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        # Create a panel to hold all of the radio buttons:
        panel = wx.wxPanel( parent, -1 )

        # Add the image buttons to the panel:
        self.create_image_grid( panel, getattr( object, trait_name ),
                                self.on_click )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Populate a specified window with a grid of image buttons:
    #----------------------------------------------------------------------------

    def create_image_grid ( self, parent, cur_value, handler ):
        # Create the main sizer:
        if self.cols > 1:
            sizer = wx.wxGridSizer( 0, self.cols, 0, 0 )
        else:
            sizer = wx.wxBoxSizer( wx.wxVERTICAL )

        # Add the set of all possible choices:
        cur_value = str( cur_value )
        for value in self.all_values():
            control = ImageControl( parent,
                         bitmap_cache( value + self.suffix, False, self.path ),
                         value == cur_value, handler )
            control.value = value
            sizer.Add( control, 0, wx.wxALL, 2 )

        # Finish setting up the control layout:
        parent.SetAutoLayout( True )
        parent.SetSizer( sizer )
        sizer.Fit( parent )

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_image ( self, control, new ):
        control.Bitmap( bitmap_cache( new + self.suffix, False, self.path ) )

    #----------------------------------------------------------------------------
    #  Return the layout style for this trait editor's control:
    #----------------------------------------------------------------------------

    def layout_style ( self ):
        return 0

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the image buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, control ):
        parent= control.GetParent()
        self.set( parent.object, parent.trait_name, control.value,
                  parent.handler )

#-------------------------------------------------------------------------------
#  'TraitEditorCheckList' class:
#-------------------------------------------------------------------------------

class TraitEditorCheckList ( wxTraitEditor ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, values, cols = 1 ):
        self.cols   = cols
        self.values = values
        if type( values[0] ) is StringType:
            self.values = [ ( x, x.capitalize() ) for x in values ]
        self.mapping = mapping = {}
        for value, key in self.values:
            mapping[ key ] = value

    #----------------------------------------------------------------------------
    #  Create an in-place simple view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxChoice( parent, -1,
                     wx.wxPoint( 0, 0 ), wx.wxSize( 100, 20 ),
                     self.all_labels() )
        try:
            control.SetSelection( self.all_values().index(
                    self.current_value( object, trait_name )[0] ) )
        except:
            pass
        wx.EVT_CHOICE( parent, control.GetId(), self.on_value_changed )
        TraitMonitor( object, trait_name, control, on_trait_change_choice )
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
        rows   = (n + cols - 1) // cols
        incr    = [ n // cols ] * cols
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
                    control.Show( False )
                sizer.Add( control, 0, wx.wxNORTH, 5 )

        # Set-up the layout:
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        TraitMonitor( object, trait_name, panel, self.on_trait_change_list )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Return the set of all possible labels:
    #----------------------------------------------------------------------------

    def all_labels ( self ):
        return [ x[1] for x in self.values ]

    #----------------------------------------------------------------------------
    #  Return the set of all possible values:
    #----------------------------------------------------------------------------

    def all_values ( self ):
        return [ x[0] for x in self.values ]

    #----------------------------------------------------------------------------
    #  Return whether or not the current value is a string or not:
    #----------------------------------------------------------------------------

    def is_string ( self, object, trait_name ):
        return (type( getattr( object, trait_name ) ) is StringType)

    #----------------------------------------------------------------------------
    #  Return the current value of the object trait:
    #----------------------------------------------------------------------------

    def current_value ( self, object, trait_name ):
        return self.parse_value( getattr( object, trait_name ) )

    #----------------------------------------------------------------------------
    #  Parse a value into a list:
    #----------------------------------------------------------------------------

    def parse_value ( self, value ):
        if value is None:
            return []
        if type( value ) is not StringType:
            return value
        return [ x.strip() for x in value.split( ',' ) ]

    #----------------------------------------------------------------------------
    #  Handle the user selecting a new value from the combo box:
    #----------------------------------------------------------------------------

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

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_choice ( self, control, new ):
        try:
            control.SetSelection( self.all_values().index(
                                       self.parse_value( new )[0] ) )
        except:
            pass

    def on_trait_change_list ( self, panel, new ):
        new_values = self.parse_value( new )
        for control in panel.GetChildren():
            if control.IsShown():
                control.SetValue( control.value in new_values )

#-------------------------------------------------------------------------------
#  'TraitEditorBoolean' class:
#-------------------------------------------------------------------------------

class TraitEditorBoolean ( wxTraitEditor ):

    #----------------------------------------------------------------------------
    #  Create an in-place read only view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def readonly_editor ( self, object, trait_name, description, handler,
                                parent ):
        control = wx.wxTextCtrl( parent, -1, '', style = wx.wxTE_READONLY )
        try:
            value       = getattr( object, trait_name + '_' )
            trait_name += '_'
        except:
            value = getattr( object, trait_name )
        TraitMonitor( object, trait_name, control,
                      self.on_trait_change_readonly )
        self.on_trait_change_readonly( control, value )
        return control

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_readonly ( self, control, new ):
        if new:
            control.SetValue( 'True' )
        else:
            control.SetValue( 'False' )

    #----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = wx.wxCheckBox( parent, -1, '' )
        try:
            value       = getattr( object, trait_name + '_' )
            trait_name += '_'
        except:
            value = getattr( object, trait_name )
        control.SetValue( value )
        wx.EVT_CHECKBOX( parent, control.GetId(), self.on_value_changed )
        TraitMonitor( object, trait_name, control, self.on_trait_change_check )
        return control

    #----------------------------------------------------------------------------
    #  Handle the user clicking on the checkbox:
    #----------------------------------------------------------------------------

    def on_value_changed ( self, event ):
        control = event.GetEventObject()
        self.set( control.object, control.trait_name, control.GetValue(),
                  control.handler )

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_check ( self, control, new ):
        control.SetValue( new )

#-------------------------------------------------------------------------------
#  'TraitEditorRange class:
#-------------------------------------------------------------------------------

class TraitEditorRange ( TraitEditorEnum ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, handler, cols = 1, auto_set = True ):
        if isinstance( handler, Trait ):
            handler = handler.setter
        self.low      = handler.low
        self.high     = handler.high
        self.is_float = (type( self.low ) is FloatType)
        self.cols     = cols
        self.auto_set = auto_set
        self.mapped   = False

    #----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        if self.is_float or (abs( self.high - self.low ) > 100):
            self.format = '%d'
            if self.is_float:
                self.format = '%%.%df' % max( 0,
                                         4 - int( log10( self.high - self.low ) ) )
            panel  = wx.wxPanel( parent, -1 )
            sizer  = wx.wxBoxSizer( wx.wxHORIZONTAL )
            fvalue = getattr( object, trait_name )
            try:
                fvalue_text = self.format % fvalue
                1 / (self.low <= fvalue <= self.high)
            except:
                fvalue_text = ''
                fvalue      = self.low
            ivalue   = int( (float( fvalue - self.low ) / (self.high - self.low))
                            * 10000 )
            label_lo = wx.wxStaticText( panel, -1, '999.999',
                          style = wx.wxALIGN_RIGHT | wx.wxST_NO_AUTORESIZE  )
            sizer.Add( label_lo, 0, wx.wxALIGN_CENTER )
            panel.slider = slider = wx.wxSlider( panel, -1, ivalue, 0, 10000,
                           size   = wx.wxSize( 100, 20 ),
                           style  = wx.wxSL_HORIZONTAL | wx.wxSL_AUTOTICKS )
            slider.SetTickFreq( 1000, 1 )
            slider.SetPageSize( 1000 )
            slider.SetLineSize( 100 )
            wx.EVT_SCROLL( slider, self.on_scroll )
            sizer.Add( slider, 1, wx.wxEXPAND )
            label_hi = wx.wxStaticText( panel, -1, '999.999' )
            sizer.Add( label_hi, 0, wx.wxALIGN_CENTER )
            panel.text = text = wx.wxTextCtrl( panel, -1, fvalue_text,
                                               size  = wx.wxSize( 60, 20 ),
                                               style = wx.wxTE_PROCESS_ENTER )
            wx.EVT_KILL_FOCUS( text, self.on_enter )
            wx.EVT_TEXT_ENTER( panel, text.GetId(), self.on_enter )
            sizer.Add( text, 0, wx.wxLEFT | wx.wxEXPAND, 8 )

            # Set-up the layout:
            panel.SetAutoLayout( True )
            panel.SetSizer( sizer )
            sizer.Fit( panel )
            label_lo.SetLabel( str( self.low ) )
            label_hi.SetLabel( str( self.high ) )
            panel.is_enum = False
            TraitMonitor( object, trait_name, panel, self.on_trait_change_slider )

            # Return the panel as the result:
            return panel
        else:
            value   = getattr( object, trait_name )
            control = wx.wxSpinCtrl( parent, -1, str( value ),
                                     min     = self.low,
                                     max     = self.high,
                                     initial = value )
            wx.EVT_SPINCTRL( parent, control.GetId(), self.on_value_changed )
            TraitMonitor( object, trait_name, control,
                          self.on_trait_change_spinner )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        if ((type( self.low ) is FloatType) or
            (abs( self.high - self.low ) > 15)):
            return self.simple_editor( object, trait_name, description,
                                       handler, parent )
        control = TraitEditorEnum.custom_editor( self, object, trait_name,
                                                 description, handler, parent )
        control.is_enum = True
        return control

    #----------------------------------------------------------------------------
    #  Return the set of all possible values:
    #----------------------------------------------------------------------------

    def all_values ( self ):
        return [ str( x ) for x in xrange( self.low, self.high + 1 ) ]

    #----------------------------------------------------------------------------
    #  Return the current value of the object trait:
    #----------------------------------------------------------------------------

    def current_value ( self, object, trait_name ):
        return str( getattr( object, trait_name ) )

    #----------------------------------------------------------------------------
    #  Handle the user selecting a new value from the spin control:
    #----------------------------------------------------------------------------

    def on_value_changed ( self, event ):
        control = event.GetEventObject()
        self.set( control.object, control.trait_name, control.GetValue(),
                  control.handler )

    #----------------------------------------------------------------------------
    #  Handle the user pressing the 'Enter' key in the edit control:
    #----------------------------------------------------------------------------

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

    #----------------------------------------------------------------------------
    #  Handle the user changing the contents of the edit control:
    #----------------------------------------------------------------------------

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
        event_type = event.GetEventType()
        if ((event_type == wx.wxEVT_SCROLL_ENDSCROLL) or
            (self.auto_set and (event_type == wx.wxEVT_SCROLL_THUMBTRACK))):
            self.set( control.object, control.trait_name, value,
                      control.handler )

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_spinner ( self, control, new ):
        try:
            control.SetValue( int( new ) )
        except:
            pass

    def on_trait_change_slider ( self, control, new ):
        try:
            new_text = self.format % new
            1 / (self.low <= new <= self.high)
        except:
            new_text = ''
            new      = self.low
        ivalue = int( (float(new - self.low) / (self.high - self.low)) * 10000.0)
        control.text.SetValue( new_text )
        control.slider.SetValue( ivalue )

#-------------------------------------------------------------------------------
#  'TraitEditorComplex class:
#-------------------------------------------------------------------------------

class TraitEditorComplex ( TraitEditorText ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, editors, auto_set = True ):
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
        for editor in self.editors:
            control = editor.custom_editor( object, trait_name, description,
                                            handler, panel )
            init_control( control, object, trait_name, description, handler )
            sizer.Add( control, 1, editor.layout_style() )

        # Set-up the layout:
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        # Return the panel as the result:
        return panel

#-------------------------------------------------------------------------------
#  'TraitEditorList class:
#-------------------------------------------------------------------------------

class TraitEditorList ( wxTraitEditor ):

    # Normal list item menu:
    list_menu = """
       Add before     [menu_before]: self.add_before()
       Add after      [menu_after]:  self.add_after()
       ---
       Delete         [menu_delete]: self.delete_item()
       ---
       Move up        [menu_up]:     self.move_up()
       Move down      [menu_down]:   self.move_down()
       Move to top    [menu_top]:    self.move_top()
       Move to bottom [menu_bottom]: self.move_bottom()
    """

    # Empty list item menu:
    empty_list_menu = """
       Add: self.add_empty()
    """

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, trait_handler, rows = 5 ):
        self.trait_handler = trait_handler
        self.rows          = rows

    #----------------------------------------------------------------------------
    #  Create an in-place read only view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def readonly_editor ( self, object, trait_name, description, handler,
                                parent ):
        return self.list_editor( object, trait_name, description, handler,
                                 parent, 'readonly_editor' )

    #----------------------------------------------------------------------------
    #  Create an in-place text editable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def text_editor ( self, object, trait_name, description, handler,
                            parent ):
        return self.list_editor( object, trait_name, description, handler,
                                 parent, 'text_editor' )

    #----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        return self.list_editor( object, trait_name, description, handler,
                                 parent, 'simple_editor' )

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler, parent ):
        return self.list_editor( object, trait_name, description, handler,
                                 parent, 'custom_editor' )

    #----------------------------------------------------------------------------
    #  Create an editable list view of the current value of the 'trait_name'
    #  trait of 'object':
    #----------------------------------------------------------------------------

    def list_editor ( self, object, trait_name, description, handler, parent,
                            type ):
        # Create all of the list item trait editors:
        trait_handler         = self.trait_handler
        resizable = ((trait_handler.min_items != trait_handler.max_items) and
                     (type != 'readonly_editor'))
        item_trait            = trait_handler.item_trait
        list_pane             = wx.wxScrolledWindow( parent, -1 )
        init_control( list_pane, object, trait_name, description, handler, None )
        list_pane.editor      = editor = getattr( item_trait.get_editor(), type )
        list_sizer            = wx.wxFlexGridSizer( 0, 1 + resizable, 0, 0 )
        list_sizer.AddGrowableCol( resizable )
        values        = getattr( object, trait_name )
        index         = 0
        width, height = 100, 18
        is_fake = (resizable and (len( values ) == 0))
        if is_fake:
            values = [ item_trait.default_value ]
        for value in values:
            width = height = 0
            if resizable:
                control = ImageControl( list_pane,
                                        bitmap_cache( 'list_editor', False ),
                                        -1, self.popup_menu )
                width, height = control.GetSize()
                width += 4
            try:
                proxy    = ListItemProxy( object, trait_name, index,
                                          item_trait, value )
                pcontrol = editor( proxy, 'value', description, handler,
                                          list_pane )
                if resizable:
                    control.pcontrol = pcontrol
                init_control( pcontrol, proxy, 'value', description, handler,
                              value )
            except:
                if not is_fake:
                    raise
                pcontrol = wx.wxButton( list_pane, -1, 'sample' )
            width2, height2 = pcontrol.GetSize()
            width += width2
            height = max( height, height2 )
            if resizable:
                list_sizer.Add( control, 0, wx.wxLEFT | wx.wxRIGHT, 2 )
            list_sizer.Add( pcontrol, 1, wx.wxEXPAND )
            index += 1
        list_pane.SetAutoLayout( True )
        list_pane.SetSizer( list_sizer )
        if is_fake:
            self.cur_control = control
            self.empty_list( list_pane )
            control.Destroy()
            pcontrol.Destroy()
        rows = [ self.rows, 1 ][ type == 'simple_editor' ]
        list_pane.SetSize( wx.wxSize(
             width + ((trait_handler.max_items > rows) * scrollbar_dx),
             height * rows ) )
        list_pane.SetScrollRate( 0, height )
        return list_pane

    #----------------------------------------------------------------------------
    #  Add a new value at the specified list index:
    #----------------------------------------------------------------------------

    def add_item ( self, offset ):
        controls    = self.get_controls()
        list, index = self.get_info()
        index      += offset
        item_trait  = self.trait_handler.item_trait
        value       = item_trait.default_value
        list[ index: index ] = [ value ]
        list_pane   = self.cur_control.GetParent()
        control     = ImageControl( list_pane,
                                    bitmap_cache( 'list_editor', False ),
                                    -1, self.popup_menu )
        proxy    = ListItemProxy( list_pane.object, list_pane.trait_name, index,
                                  item_trait, value )
        pcontrol = list_pane.editor( proxy, 'value', list_pane.description,
                                     list_pane.handler, list_pane )
        control.pcontrol         = pcontrol
        init_control( pcontrol, proxy, 'value', list_pane.description,
                      list_pane.handler, value )
        controls[ index: index ] = [ ( control, pcontrol ) ]
        self.reload_sizer( controls, -2 )

    #----------------------------------------------------------------------------
    #  Create an empty list entry (so the user can add a new item):
    #----------------------------------------------------------------------------

    def empty_list ( self, list_pane ):
        control = ImageControl( list_pane, bitmap_cache( 'list_editor', False ),
                                -1, self.popup_empty_menu )
        control.is_empty = True
        proxy    = ListItemProxy( list_pane.object, list_pane.trait_name, -1,
                                  None, None )
        pcontrol = wx.wxStaticText( list_pane, -1, '   (Empty List)' )
        control.pcontrol = pcontrol
        pcontrol.object  = proxy
        self.reload_sizer( [ ( control, pcontrol ) ] )

    #----------------------------------------------------------------------------
    #  Display the empty list editor popup menu:
    #----------------------------------------------------------------------------

    def popup_empty_menu ( self, control ):
        self.cur_control = control
        control.PopupMenuXY( MakeMenu( self.empty_list_menu, self, True,
                                       control ).menu, 0, 0 )

    #----------------------------------------------------------------------------
    #  Display the list editor popup menu:
    #----------------------------------------------------------------------------

    def popup_menu ( self, control ):
        self.cur_control = control
        proxy    = control.pcontrol.object
        index    = proxy.index
        menu     = MakeMenu( self.list_menu, self, True, control ).menu
        len_list = len( proxy.list() )
        not_full = (len_list < self.trait_handler.max_items)
        self.menu_before.enabled( not_full )
        self.menu_after.enabled(  not_full )
        self.menu_delete.enabled( len_list > self.trait_handler.min_items )
        self.menu_up.enabled(  index > 0 )
        self.menu_top.enabled( index > 0 )
        self.menu_down.enabled(   index < (len_list - 1) )
        self.menu_bottom.enabled( index < (len_list - 1) )
        control.PopupMenuXY( menu, 0, 0 )

    #----------------------------------------------------------------------------
    #  Insert a new item before the current item:
    #----------------------------------------------------------------------------

    def add_before ( self ):
        self.add_item( 0 )

    #----------------------------------------------------------------------------
    #  Insert a new item after the current item:
    #----------------------------------------------------------------------------

    def add_after ( self ):
        self.add_item( 1 )

    #----------------------------------------------------------------------------
    #  Add a new item when the list is empty:
    #----------------------------------------------------------------------------

    def add_empty ( self ):
        self.add_item( 0 )
        self.delete_item()

    #----------------------------------------------------------------------------
    #  Delete the current item:
    #----------------------------------------------------------------------------

    def delete_item ( self ):
        controls          = self.get_controls()
        list, index       = self.get_info()
        control, pcontrol = controls[ index ]
        del controls[ index ]
        if hasattr( control, 'is_empty' ):
            self.reload_sizer( controls, 2 )
        else:
            del list[ index ]
            if len( list ) == 0:
                self.empty_list( control.GetParent() )
            else:
                self.reload_sizer( controls, 2 )
        control.Destroy()
        pcontrol.Destroy()

    #----------------------------------------------------------------------------
    #  Move the current item up one in the list:
    #----------------------------------------------------------------------------

    def move_up ( self ):
        controls    = self.get_controls()
        list, index = self.get_info()
        controls[index-1], controls[index] = controls[index], controls[index-1]
        list[index-1], list[index] = list[index], list[index-1]
        self.reload_sizer( controls )

    #----------------------------------------------------------------------------
    #  Move the current item down one in the list:
    #----------------------------------------------------------------------------

    def move_down ( self ):
        controls    = self.get_controls()
        list, index = self.get_info()
        controls[index+1], controls[index] = controls[index], controls[index+1]
        list[index+1], list[index] = list[index], list[index+1]
        self.reload_sizer( controls )

    #----------------------------------------------------------------------------
    #  Move the current item to the top of the list:
    #----------------------------------------------------------------------------

    def move_top ( self ):
        controls    = self.get_controls()
        list, index = self.get_info()
        control     = [ controls[ index ] ]
        item        = [ list[ index ] ]
        del controls[ index ]
        del list[ index ]
        controls[0:0] = control
        list[0:0]     = item
        self.reload_sizer( controls )

    #----------------------------------------------------------------------------
    #  Move the current item to the bottom of the list:
    #----------------------------------------------------------------------------

    def move_bottom ( self ):
        controls    = self.get_controls()
        list, index = self.get_info()
        control     = controls[ index ]
        item        = list[ index ]
        del controls[ index ]
        del list[ index ]
        controls.append( control )
        list.append( item )
        self.reload_sizer( controls )

    #----------------------------------------------------------------------------
    #  Return the associated object list and current item index:
    #----------------------------------------------------------------------------

    def get_info ( self ):
        cur_control = self.cur_control
        proxy = self.cur_control.pcontrol.object
        return ( proxy.list(), proxy.index )

    #----------------------------------------------------------------------------
    #  Return all controls in the correct index order as ( button, proxy ) pairs:
    #----------------------------------------------------------------------------

    def get_controls ( self ):
        cur_control = self.cur_control
        result      = []
        controls = cur_control.GetParent().GetChildren()
        for i in xrange( 0, len( controls ), 2 ):
            result.append( ( controls[i], controls[i+1] ) )
        result.sort( lambda x, y: cmp( x[1].object.index, y[1].object.index ) )
        return result

    #----------------------------------------------------------------------------
    #  Reload the layout from the specified list of ( button, proxy ) pairs:
    #----------------------------------------------------------------------------

    def reload_sizer ( self, controls, extra = 0 ):
        list  = self.cur_control.GetParent()
        sizer = list.GetSizer()
        for i in xrange( 2 * len( controls ) + extra ):
            sizer.Remove( 0 )
        index = 0
        for control, pcontrol in controls:
            sizer.Add( control,  0, wx.wxLEFT | wx.wxRIGHT, 2 )
            sizer.Add( pcontrol, 1, wx.wxEXPAND )
            pcontrol.object.index = index
            index += 1
        sizer.Layout()
        list.SetVirtualSize( sizer.GetMinSize() )

#-------------------------------------------------------------------------------
#  'ListItemProxy' class:
#-------------------------------------------------------------------------------

class ListItemProxy ( HasDynamicTraits ):

    def __init__ ( self, object, trait_name, index, trait, value ):
        HasDynamicTraits.__init__( self )
        self.add_trait( 'value', trait )
        self.object     = object
        self.trait_name = trait_name
        self.index      = index
        self.value      = value

    def list ( self ):
        return getattr( self.object, self.trait_name )

    def value_changed ( self, old_value, new_value ):
        self.list()[ self.index ] = new_value

#-------------------------------------------------------------------------------
#  'TraitEditorFile' class:
#-------------------------------------------------------------------------------

class TraitEditorFile ( TraitEditorText ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, filter = None ):
        TraitEditorText.__init__( self )
        self.filter = filter

    #----------------------------------------------------------------------------
    #  Create an in-place simple view of the current value of the
    #  'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        panel   = wx.wxPanel( parent, -1 )
        sizer   = wx.wxBoxSizer( wx.wxHORIZONTAL )
        control = wx.wxTextCtrl( panel, -1,
                                 self.str( object, trait_name ),
                                 style = wx.wxTE_PROCESS_ENTER )
        init_control( control, object, trait_name, description, handler )
        wx.EVT_KILL_FOCUS( control, self.on_enter )
        wx.EVT_TEXT_ENTER( panel, control.GetId(), self.on_enter )
        wx.EVT_TEXT(       panel, control.GetId(), self.on_key )
        sizer.Add( control, 1, wx.wxEXPAND | wx.wxALIGN_CENTER )
        button          = wx.wxButton( panel, -1, 'Browse...' )
        button.filename = control
        sizer.Add( button, 0, wx.wxLEFT | wx.wxALIGN_CENTER, 8 )
        wx.EVT_BUTTON( panel, button.GetId(), self.on_browse )
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )
        TraitMonitor( object, trait_name, control, self.on_trait_change_text )
        return panel

    #----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' color trait of 'object':
    #----------------------------------------------------------------------------

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

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

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
        if editor.mapped:
            trait_name == '_'
        editor.create_image_grid( self, getattr( object, trait_name ),
                                  self.on_click )

        # Position the dialog:
        position_near( parent, self )

    #----------------------------------------------------------------------------
    #  Close the dialog:
    #----------------------------------------------------------------------------

    def on_close_dialog ( self, event, rc = False ):
        self.EndModal( rc )
        self.Destroy()

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the choice buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, control ):
        self.editor.set( self.object, self.trait_name, control.value,
                         self.handler )

#-------------------------------------------------------------------------------
#  Convert an image file name to a cached bitmap:
#-------------------------------------------------------------------------------

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
        dc1.Blit( (standard_bitmap_width - dx) // 2, 0, dx, dy, dc2, 0, 0 )
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

#-------------------------------------------------------------------------------
#  Convert a number into a wxColour object:
#-------------------------------------------------------------------------------

def num_to_color ( object, name, value ):
    num = int( value )
    return wx.wxColour( num // 0x10000, (num // 0x100) & 0xFF, num & 0xFF )

num_to_color.info = ('a number, which in hex is of the form 0xRRGGBB, where '
                     'RR is red, GG is green, and BB is blue')

#-------------------------------------------------------------------------------
#  'TraitEditorColor' class:
#-------------------------------------------------------------------------------

class TraitEditorColor ( wxTraitEditor ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self ):
        self.color_data = wx.wxColourData()

    #----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' color trait of 'object':
    #----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             parent = None ):
        if not hasattr( parent, 'is_custom' ):
            # Fixes a problem with the edit field having the focus:
            parent.ReleaseMouse()
            return TraitEditorColorDialog( object, trait_name, description,
                                           parent, handler, self ).ShowModal()

        self.color_data.SetColour( self.to_wx_color( object, trait_name ) )
        self.color_data.SetChooseFull( True )
        dialog = wx.wxColourDialog( parent or object.window, self.color_data )
        if dialog.ShowModal() == wx.wxID_OK:
            panel = parent
            if panel is not None:
                panel = panel.GetParent()
            self.set( object, trait_name,
                      self.from_wx_color( dialog.GetColourData().GetColour(),
                                          panel ),
                      handler )
            return True
        return False

    #----------------------------------------------------------------------------
    #  Create a view of the current value of the 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def create_editor ( self, object, trait_name, description, handler,
                              parent, factory ):
        control = factory( self, object, trait_name, description, handler,
                           parent )
        self.set_color( control, self.to_wx_color( object, trait_name ) )
        try:
            getattr( object, trait_name + '_' )
            TraitMonitor( object, trait_name, control, self.on_trait_change_text )
            TraitMonitor( object, trait_name + '_', control,
                          self.on_trait_change_color )
        except:
            TraitMonitor( object, trait_name, control, self.on_trait_change_both )
        return control

    #----------------------------------------------------------------------------
    #  Create a read only view of the current value of the 'trait_name'
    #  trait of 'object':
    #----------------------------------------------------------------------------

    def readonly_editor ( self, object, trait_name, description, handler,
                                parent ):
        return self.create_editor( object, trait_name, description, handler,
                                   parent, wxTraitEditor.readonly_editor )

    #----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        return self.create_editor( object, trait_name, description, handler,
                                   parent, wxTraitEditor.simple_editor )

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
        control.is_custom = True
        init_control( control, object, trait_name, description, handler, None )
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
        panel.SetAutoLayout( True )
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
                  self.from_wx_color( control.GetBackgroundColour(), parent ),
                  parent.handler )

    #----------------------------------------------------------------------------
    #   Set the color of the 'current color' control:
    #----------------------------------------------------------------------------

    def set_color ( self, control, color ):
        control.SetBackgroundColour( color )
        if ((color.Red()   > 192) or
            (color.Blue()  > 192) or
            (color.Green() > 192)):
            control.SetForegroundColour( wx.wxBLACK )
        else:
            control.SetForegroundColour( wx.wxWHITE )

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_color ( self, control, new ):
        self.set_color( control, self.to_wx_color( new ) )

    def on_trait_change_both ( self, control, new ):
        self.on_trait_change_text(  control, new )
        self.on_trait_change_color( control, new )

    #----------------------------------------------------------------------------
    #  Get the current object trait color:
    #----------------------------------------------------------------------------

    def get_cur_color ( self, object, trait_name ):
        try:
            return getattr( object, trait_name + '_' )
        except:
            return getattr( object, trait_name )

    #----------------------------------------------------------------------------
    #  Get the wxPython color equivalent of the object trait:
    #----------------------------------------------------------------------------

    def to_wx_color ( self, object, trait_name = None ):
        if trait_name is None:
            color = object
        else:
            color = self.get_cur_color( object, trait_name )
        if color is None:
            color = wx.wxWHITE
        return color

    #----------------------------------------------------------------------------
    #  Get the application equivalent of a wxPython value:
    #----------------------------------------------------------------------------

    def from_wx_color ( self, color, panel = None ):
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

#-------------------------------------------------------------------------------
#  'TraitEditorColorDialog' class:
#-------------------------------------------------------------------------------

class TraitEditorColorDialog ( wx.wxDialog ):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, object, trait_name, description, parent, handler,
                         editor ):
        wx.wxDialog.__init__( self, parent, -1, 'Choose ' + description )
        wx.EVT_CLOSE( self, self.on_close_dialog )
        panel = editor.custom_editor( object, trait_name, description, handler,
                                      self )
        # Initialize instance data:
        panel.object     = object
        panel.trait_name = trait_name
        panel.handler    = handler
        panel.editor     = editor

        sizer = wx.wxBoxSizer( wx.wxVERTICAL )
        sizer.Add( panel )
        self.SetAutoLayout( True )
        self.SetSizer( sizer )
        sizer.Fit( self )
        position_near( parent, self )

    #----------------------------------------------------------------------------
    #  Close the dialog:
    #----------------------------------------------------------------------------

    def on_close_dialog ( self, event, rc = False ):
        self.EndModal( rc )
        self.Destroy()

#-------------------------------------------------------------------------------
#  Convert a string into a valid 'wxFont' object (if possible):
#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------
#  'TraitEditorFont' class:
#-------------------------------------------------------------------------------

class TraitEditorFont ( wxTraitEditor ):

    #----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' font trait of 'object':
    #----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             parent = None ):
        font_data = wx.wxFontData()
        font_data.SetInitialFont( self.to_wx_font( object, trait_name ) )
        dialog = wx.wxFontDialog( parent or object.window, font_data )
        if dialog.ShowModal() == wx.wxID_OK:
            self.set( object, trait_name,
                      self.from_wx_font( dialog.GetFontData().GetChosenFont() ),
                      handler )
            return True
        return False

    #----------------------------------------------------------------------------
    #  Create a view of the current value of the 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def create_editor ( self, object, trait_name, description, handler,
                              parent, factory ):
        control = factory( self, object, trait_name, description, handler,
                           parent )
        self.on_trait_change_simple( control, getattr( object, trait_name ) )
        TraitMonitor( object, trait_name, control, self.on_trait_change_simple )
        return control

    #----------------------------------------------------------------------------
    #  Create a read only view of the current value of the 'trait_name'
    #  trait of 'object':
    #----------------------------------------------------------------------------

    def readonly_editor ( self, object, trait_name, description, handler,
                                parent ):
        return self.create_editor( object, trait_name, description, handler,
                                   parent, wxTraitEditor.readonly_editor )

    #----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        return self.create_editor( object, trait_name, description, handler,
                                   parent, wxTraitEditor.simple_editor )

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
        init_control( font, object, trait_name, description, handler, None )
        sizer.Add( font, 0, wx.wxEXPAND | wx.wxBOTTOM, 3 )

        # Add all of the font choice controls:
        sizer2  = wx.wxBoxSizer( wx.wxHORIZONTAL )
        control = panel.facename = wx.wxChoice( panel, -1, wx.wxPoint( 0, 0 ),
                                     wx.wxSize( 100, 20 ), self.all_facenames() )
        sizer2.Add( control, 2, wx.wxEXPAND )
        wx.EVT_CHOICE( panel, control.GetId(), self.on_value_changed )

        control = panel.pointsize = wx.wxChoice( panel, -1, wx.wxPoint( 0, 0 ),
                                       wx.wxSize( 30, 20 ), point_sizes )
        sizer2.Add( control, 1, wx.wxEXPAND | wx.wxRIGHT, 3 )
        wx.EVT_CHOICE( panel, control.GetId(), self.on_value_changed )

        sizer.Add( sizer2, 0, wx.wxEXPAND )

        # Initialize the control's with the object's current trait value:
        self.on_trait_change_custom( font, getattr( object, trait_name ) )

        # Set-up the layout:
        panel.SetAutoLayout( True )
        panel.SetSizer( sizer )
        sizer.Fit( panel )

        TraitMonitor( object, trait_name, font, self.on_trait_change_custom )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Handle the object trait changing value outside the editor:
    #----------------------------------------------------------------------------

    def on_trait_change_simple ( self, control, new ):
        self.on_trait_change_text( control, new )
        font = self.to_wx_font( new )
        font.SetPointSize( min( 10, font.GetPointSize() ) )
        control.SetFont( font )

    def on_trait_change_custom ( self, control, new ):
        self.on_trait_change_text(  control, new )
        font  = self.to_wx_font( new )
        panel = control.GetParent()
        try:
            panel.facename.SetStringSelection( font.GetFaceName() )
        except:
            panel.facename.SetSelection( 0 )
        try:
            panel.pointsize.SetStringSelection( str( font.GetPointSize() ) )
        except:
            panel.pointsize.SetSelection( 0 )
        font.SetPointSize( min( 10, font.GetPointSize() ) )
        control.SetFont( font )

    #----------------------------------------------------------------------------
    #  Return the text representation of the specified object trait value:
    #----------------------------------------------------------------------------

    def str_value ( self, font ):
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

    #----------------------------------------------------------------------------
    #  Return a wxFont object corresponding to a specified object's font trait:
    #----------------------------------------------------------------------------

    def to_wx_font ( self, object, trait_name = None ):
        if trait_name is None:
            return object
        return getattr( object, trait_name )

    #----------------------------------------------------------------------------
    #  Get the application equivalent of a wxPython value:
    #----------------------------------------------------------------------------

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
        return True

#-------------------------------------------------------------------------------
#  'ImageControl' class:
#-------------------------------------------------------------------------------

class ImageControl ( wx.wxWindow ):

    # Pens used to draw the 'selection' marker:
    _selectedPenDark = wx.wxPen(
       wx.wxSystemSettings_GetColour( wx.wxSYS_COLOUR_3DSHADOW ), 1,
       wx.wxSOLID )
    _selectedPenLight = wx.wxPen(
       wx.wxSystemSettings_GetColour( wx.wxSYS_COLOUR_3DHIGHLIGHT ), 1,
       wx.wxSOLID )

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, parent, bitmap, selected = None, handler = None ):
        wx.wxWindow.__init__( self, parent, -1,
                               size = wx.wxSize( bitmap.GetWidth()  + 10,
                                                 bitmap.GetHeight() + 10 ) )
        self._bitmap      = bitmap
        self._selected    = selected
        self._handler     = handler
        self._mouse_over  = False
        self._button_down = False

        # Set up the 'paint' event handler:
        wx.EVT_PAINT( self, self._on_paint )

        # Set up mouse event handlers:
        wx.EVT_LEFT_DOWN(    self, self._on_left_down )
        wx.EVT_LEFT_UP(      self, self._on_left_up )
        wx.EVT_ENTER_WINDOW( self, self._on_enter )
        wx.EVT_LEAVE_WINDOW( self, self._on_leave )

    #----------------------------------------------------------------------------
    #  Get/Set the current selection state of the image:
    #----------------------------------------------------------------------------

    def Selected ( self, selected = None ):
        if selected is not None:
            selected = (selected != 0)
            if selected != self._selected:
                if selected:
                    for control in self.GetParent().GetChildren():
                        if (isinstance( control, ImageControl ) and
                           control.Selected()):
                            control.Selected( False )
                            break
                self._selected = selected
                self.Refresh()
        return self._selected

    #----------------------------------------------------------------------------
    #  Get/Set the current bitmap image:
    #----------------------------------------------------------------------------

    def Bitmap ( self, bitmap = None ):
        if bitmap is not None:
            if bitmap != self._bitmap:
                self._bitmap = bitmap
                self.Refresh()
        return self._bitmap

    #----------------------------------------------------------------------------
    #  Get/Set the current click handler:
    #----------------------------------------------------------------------------

    def Handler ( self, handler = None ):
        if handler is not None:
            if handler != self._handler:
                self._handler = handler
                self.Refresh()
        return self._handler

    #----------------------------------------------------------------------------
    #  Handle the mouse entering the control:
    #----------------------------------------------------------------------------

    def _on_enter ( self, event = None ):
        if self._selected is not None:
            self._mouse_over = True
            self.Refresh()

    #----------------------------------------------------------------------------
    #  Handle the mouse leaving the control:
    #----------------------------------------------------------------------------

    def _on_leave ( self, event = None ):
        if self._mouse_over:
            self._mouse_over = False
            self.Refresh()

    #----------------------------------------------------------------------------
    #  Handle the user pressing the mouse button:
    #----------------------------------------------------------------------------

    def _on_left_down ( self, event = None ):
        if self._selected is not None:
            self.CaptureMouse()
            self._button_down = True
            self.Refresh()

    #----------------------------------------------------------------------------
    #  Handle the user clicking the control:
    #----------------------------------------------------------------------------

    def _on_left_up ( self, event = None ):
        need_refresh = self._button_down
        if need_refresh:
            self.ReleaseMouse()
            self._button_down = False

        if self._selected is not None:
            wdx, wdy = self.GetClientSizeTuple()
            x        = event.GetX()
            y        = event.GetY()
            if (0 <= x < wdx) and (0 <= y < wdy):
                if self._selected != -1:
                    self.Selected( True )
                    need_refresh = False
                if self._handler is not None:
                    self._handler( self )

        if need_refresh:
            self.Refresh()

    #----------------------------------------------------------------------------
    #  Handle the control being re-painted:
    #----------------------------------------------------------------------------

    def _on_paint ( self, event = None ):
        wdc      = wx.wxPaintDC( self )
        wdx, wdy = self.GetClientSizeTuple()
        bitmap   = self._bitmap
        bdx      = bitmap.GetWidth()
        bdy      = bitmap.GetHeight()
        wdc.DrawBitmap( bitmap, (wdx - bdx) // 2, (wdy - bdy) // 2, True )

        pens = [ self._selectedPenLight, self._selectedPenDark ]
        bd   = self._button_down
        if self._mouse_over:
            wdc.SetBrush( wx.wxTRANSPARENT_BRUSH )
            wdc.SetPen( pens[ bd ] )
            wdc.DrawLine( 0, 0, wdx, 0 )
            wdc.DrawLine( 0, 1, 0, wdy )
            wdc.SetPen( pens[ 1 - bd ] )
            wdc.DrawLine( wdx - 1, 1, wdx - 1, wdy )
            wdc.DrawLine( 1, wdy - 1, wdx - 1, wdy - 1 )

        if self._selected == True:
            wdc.SetBrush( wx.wxTRANSPARENT_BRUSH )
            wdc.SetPen( pens[ bd ] )
            wdc.DrawLine( 1, 1, wdx - 1, 1 )
            wdc.DrawLine( 1, 1, 1, wdy - 1 )
            wdc.DrawLine( 2, 2, wdx - 2, 2 )
            wdc.DrawLine( 2, 2, 2, wdy - 2 )
            wdc.SetPen( pens[ 1 - bd ] )
            wdc.DrawLine( wdx - 2, 2, wdx - 2, wdy - 1 )
            wdc.DrawLine( 2, wdy - 2, wdx - 2, wdy - 2 )
            wdc.DrawLine( wdx - 3, 3, wdx - 3, wdy - 2 )
            wdc.DrawLine( 3, wdy - 3, wdx - 3, wdy - 3 )
