#--------------------------------------------------------------------------------
#
#  Define a Tkinter based trait sheet mechanism for visually editing the
#  values of traits.
#
#  Written by: David C. Morrill
#
#  Date: 10/15/2002
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

import Tkinter        as tk
import tkMessageBox   as mb
import tkSimpleDialog as sd
import tkColorChooser as cc
import Pmw
import tkFont

from traits      import Trait, TraitError, trait_editors
from trait_sheet import TraitEditor, TraitSheetHandler, \
                        TraitGroup, default_trait_sheet_handler
from types       import DictType, ListType, TupleType, ModuleType, StringType

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

# Standard colors:
WHITE = '#FFFFFF'

# Standard color samples:
color_choices = ( 0, 128, 192, 255 )
color_samples = [ None ] * 48
i             = 0
for r in color_choices:
    for g in color_choices:
        for b in ( 0, 128, 255 ):
            color_samples[i] = '#%02X%02X%02X' % ( r, g, b )
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
        self.dlg = None

    #----------------------------------------------------------------------------
    #  Notification that a trait sheet has been closed:
    #----------------------------------------------------------------------------

    def close ( self, trait_sheet, object ):
        rc = TRUE
        print 'TraitSheetHandler.close() called...'
        print 'modified = ',self.modified

        if self.modified:
            self.dlg = Pmw.MessageDialog( trait_sheet, title="Discard changes?",
                       message_text ='Changes have been made.\n Discard changes for %s Traits?' % object.__class__.__name__,
                       buttons = ('Yes','No'), defaultbutton=1, command=self.discardChanges)
            result = self.dlg.activate()

            if result == 'Yes': rc = TRUE
            else: rc = FALSE
        return rc

    def discardChanges(self, result):
        if result == 'Yes':
            self.save(result)
        if self.dlg:
            self.dlg.deactivate(result)

    #----------------------------------------------------------------------------
    #  Notification that a trait sheet has modified a trait of its
    #  associated object:
    #----------------------------------------------------------------------------

    def changed ( self, object, trait_name, new_value, old_value, is_set ):
        self.modified = TRUE
        self.save_enable = tk.ACTIVE

    #----------------------------------------------------------------------------
    #  Create extra content to add to the trait sheet:
    #     specifically, the SAVE and CANCEL buttons
    #----------------------------------------------------------------------------

    def init ( self, trait_sheet,  object ):
        self.sheet = trait_sheet
        vsizer = tk.Frame()
        vsizer.pack()
        self.save_enable = tk.DISABLED
        #self.save_enable = tk.ACTIVE
        save_button = tk.Button( vsizer, text='Save changes',
                                               state= self.save_enable,
                                               command=self.save).pack(side=tk.LEFT)
        cancel_button = tk.Button( vsizer, text='Cancel', state = tk.ACTIVE,
                            command=self.cancel).pack(side=tk.RIGHT)
        return self.sheet

    #----------------------------------------------------------------------------
    #  Handle the user requesting that all changes to the object be saved:
    #----------------------------------------------------------------------------

    def save ( self):
        event = 'Save'
        #self.modified    = FALSE
        self.app.save_ok = TRUE
        self.sheet.on_close_page(event)

    def cancel (self):
        event = 'Cancel'
        self.modified  = FALSE
        self.sheet.on_close_page(event)

#-------------------------------------------------------------------------------
#  'TraitSheetApp' class:
#-------------------------------------------------------------------------------

class TraitSheetApp (tk.Frame):

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, object, traits = None ):
        self.object  = object
        self.traits  = traits
        self.save_ok = FALSE
        tk.Frame.__init__(self)
        self.OnInit()

    #----------------------------------------------------------------------------
    #  Handle application initialization:
    #----------------------------------------------------------------------------

    def OnInit ( self ):

        self.sheet = TraitSheetDialog( self.object, self.traits,
                          TraitSheetAppHandler( self ), self )
        return self.sheet

#--------------------------------------------------------------------------------
#  'TraitSheetDialog' class:
#--------------------------------------------------------------------------------

class TraitSheetDialog ( tk.Frame ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, object,
                         traits = None,
                         handler    = default_trait_sheet_handler,
                         parent     = None,
                         title      = None ):
        if title is None:
            title = '%s Traits' % object.__class__.__name__
        tk.Frame.__init__( self)
        self.pack(expand=tk.YES, fill=tk.BOTH)
        if not parent: parent = self
        parent.master.title(title)
        parent.bind( '<Destroy>', self.on_close_page )
        parent.bind( '<Escape>',  self.on_close_key )

        self.object  = object
        self.handler = handler

        # Create the actual trait sheet panel:
        TraitSheet( self, object, traits, handler ).grid( row = 0 )
        self.handler.init(self, object)

        # Find a nice place on the screen to display the trait sheet so
        # that it overlays the object as little as possible:
        if not handler.position( self, object ):
#???      self.Centre( wx.wxBOTH )
            pass

        self.mainloop()

    #-----------------------------------------------------------------------------
    #  Close the trait sheet window:
    #-----------------------------------------------------------------------------

    def on_close_page ( self, event = None):
        print 'Running on_close_page with event = ',event
        self.handler.close( self, self.object )

    #----------------------------------------------------------------------------
    #  Handle the user hitting the 'Esc'ape key:
    #----------------------------------------------------------------------------

    def on_close_key ( self, event ):
        self.on_close_page( event )
        self.destroy()

#--------------------------------------------------------------------------------
#  'TraitPanel' class:
#--------------------------------------------------------------------------------

class TraitPanel ( tk.Frame ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, parent ):
        tk.Frame.__init__( self, parent )
        pass   ### NOT IMPLEMENTED YET

    #----------------------------------------------------------------------------
    #  Add a TraitSheet to the panel:
    #----------------------------------------------------------------------------

    def add ( self, sheet ):
        pass   ### NOT IMPLEMENTED YET

    #----------------------------------------------------------------------------
    #  Remove a TraitSheet from the panel:
    #----------------------------------------------------------------------------

    def remove ( self, sheet ):
        pass   ### NOT IMPLEMENTED YET

    #----------------------------------------------------------------------------
    #  Get the size of the panel:
    #----------------------------------------------------------------------------

    def size ( self ):
        return ( 0, 0 )   ### NOT IMPLEMENTED YET

    #----------------------------------------------------------------------------
    #  Set the size and position of the panel:
    #----------------------------------------------------------------------------

    def position ( self, x, y, dx, dy ):
        pass   ### NOT IMPLEMENTED YET

    #----------------------------------------------------------------------------
    #  Destroy the panel:
    #----------------------------------------------------------------------------

    def destroy ( self ):
        pass   ### NOT IMPLEMENTED YET

#--------------------------------------------------------------------------------
#  'TraitSheet' class:
#--------------------------------------------------------------------------------

class TraitSheet ( tk.Frame ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, parent,
                         object,
                         traits = None,
                         handler    = default_trait_sheet_handler ):
        tk.Frame.__init__( self, parent )

        self.object  = object
        self.handler = handler
        self.tooltip = None

        # If no traits were specified:
        if traits is None:
            # Get them from the specified object:
            traits = object.editable_traits()

        # Try to make sure that we now have either a single TraitGroup, or
        # a list of TraitGroups:
        kind = type( traits )
        if kind == StringType:
            # Convert the single trait name into a TraitGroup:
            traits = TraitGroup( traits, show_border = FALSE )
        elif ((kind in basic_sequence_types) and (len( traits ) > 0) and
              (not isinstance( traits[0], TraitGroup ))):
            # Convert a simple list of trait elements into a single,
            # TraitGroup, possibly containing multiple items:
            traits = TraitGroup( show_border = FALSE, *traits )

        # Create the requested style of trait sheet editor:
        if isinstance( traits, TraitGroup ):
            # Single page dialog:
            self.add_page( traits, self )
        else:
            # Multi-tab notebook:
            self.add_tabs( traits )

    #-----------------------------------------------------------------------------
    #  Create a tab (i.e. notebook) style trait editor:
    #-----------------------------------------------------------------------------

    def add_tabs ( self, traits ):
        # Create the notebook:
        nb = Pmw.NoteBook( self )
        nb.grid( row = 0, col = 0, sticky = 'new' )

        count = 0
        for pg in traits:
            # Create the new notebook page:
            page_name = pg.label
            if page_name is None:
                count    += 1
                page_name = 'Page %d' % count
            self.add_page( pg, nb.add( page_name ) )

        # Size the notebook to fit the pages it contains:
        nb.setnaturalsize()

    #-----------------------------------------------------------------------------
    #  Create a single trait editor page:
    #-----------------------------------------------------------------------------

    def add_page ( self, pg, parent, default_style = 'simple' ):
        default_style = pg.style  or default_style
        object        = pg.object or self.object
        row_incr      = col_incr = 0
        if pg.orientation == 'horizontal':
            col_incr = 1
        else:
            row_incr = 1
        show_labels = pg.show_labels_
        row         = col = 0
        cols        = 1 + show_labels
        for pge in pg.values:
            if isinstance( pge, TraitGroup ):
                if pge.show_border_:
                    box = Pmw.Group( parent, tag_text = pge.label or '' )
                    box.grid( row = row, col = col, sticky = 'new',
                              padx = 4 )
                    frame = tk.Frame( box.interior() )
                    frame.grid( row = 0, col = 0, sticky = 'news',
                                padx = 4, pady = 3 )
                    box.interior().columnconfigure( 0, weight = 1 )
                    self.add_page( pge, frame, default_style )
                else:
                    self.add_page( pge, parent )
                if row_incr:
                    parent.columnconfigure( 0, weight = 1 )
                row += row_incr
                col += col_incr
            else:
                name = pge.name or ' '

                if name == '-':
                    tk.Frame( parent, bg = '#A0A0A0' ).grid(
                              row = row, col = 0, columnspan = cols, sticky = 'ew' )
                    parent.rowconfigure( row, minsize = 9 )
                    row += 1
                    continue

                if name == ' ':
                    name = '5'
                if all_digits.match( name ):
                    parent.rowconfigure( row, minsize = int( name ) )
                    row += 1
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
                self.add_trait( parent, row, object, name, label, editor,
                                   style == 'simple' )
                parent.columnconfigure( 1, weight = 1 )

                # Advance to the next row in the grid:
                row += 1

        # Allocate any extra space to an imaginary row following last one used:
        parent.rowconfigure( row, weight = 1 )

    #-----------------------------------------------------------------------------
    #  Add a trait to the trait sheet:
    #-----------------------------------------------------------------------------

    def add_trait ( self, parent, row, object, trait_name, description, editor,
                             is_simple ):
        global tooltips_enabled

        col = 0
        if description is not None:
            suffix = ':'[ description[-1:] == '?': ]
            label  = tk.Label( parent,
                               text   = (description + suffix).capitalize(),
                               anchor = 'e' )
            label.grid( row = row, sticky = 'new', padx = 0, pady = 2 )
            col  = 1
            desc = object._base_trait( trait_name ).desc
            if (desc is not None) and tooltips_enabled:
                if self.tooltip is None:
                    self.tooltip = Pmw.Balloon( self, state = 'balloon' )
                self.tooltip.bind( label, 'Specifies ' + desc )
                label.bind( '<B2-ButtonRelease>', self.on_toggle_help )
        if is_simple:
            control = editor.simple_editor( object, trait_name, description,
                                            self.handler, parent )
            control.bind( '<B2-ButtonRelease>', editor.on_restore )
        else:
            control = editor.custom_editor( object, trait_name, description,
                                            self.handler, parent )
        control.grid( row = row, col = col, sticky = 'ew', padx = 4, pady = 2 )
        control.object         = object
        control.trait_name  = trait_name
        control.description    = description
        control.original_value = getattr( object, trait_name )
        control.handler        = self.handler
        return control

    #-----------------------------------------------------------------------------
    #  Toggle whether user sees tooltips or not:
    #-----------------------------------------------------------------------------

    def on_toggle_help ( self, event ):
        global tooltips_enabled
        tooltips_enabled = not tooltips_enabled
        if self.tooltip is not None:
            self.tooltip.configure(
                 state = ( 'none', 'balloon' )[ tooltips_enabled ] )

#--------------------------------------------------------------------------------
#  'tkTraitEditor' class:
#--------------------------------------------------------------------------------

class tkTraitEditor ( TraitEditor ):

    #-----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = tk.Label( parent, text   = self.str( object, trait_name ),
                                    relief = tk.SUNKEN )
        control.object         = object
        control.trait_name     = trait_name
        control.description    = description
        control.handler        = handler
        control.original_value = getattr( object, trait_name )
        control.bind( '<B1-ButtonRelease>', self.on_popup )
        control.bind( '<3>',                self.on_restore )
        return control

    #-----------------------------------------------------------------------------
    #  Invoke the pop-up editor for an object trait:
    #-----------------------------------------------------------------------------

    def on_popup ( self, event ):
        control = event.widget
        if hasattr( self, 'popup_editor' ):
            self.popup_editor( control.object, control.trait_name,
                               control.description, control.handler, control )

    #-----------------------------------------------------------------------------
    #  Restore the original value of the object's trait:
    #-----------------------------------------------------------------------------

    def on_restore ( self, event ):
        control    = event.widget
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
    #  value of the 'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        control.configure( text = self.str( object, trait_name ) )

    #-----------------------------------------------------------------------------
    #  Handle a 'TraitError' exception:
    #-----------------------------------------------------------------------------

    def error ( self, description, excp, parent ):
        mb.showerror( description + ' value error', str( excp ) )

#--------------------------------------------------------------------------------
#  'TraitEditorText' class:
#--------------------------------------------------------------------------------

class TraitEditorText ( tkTraitEditor ):

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
                             control ):
        while TRUE:
            value = sd.askstring( 'Prompt',
                                  'Enter the new %s value:' % trait_name,
                                  initialvalue = getattr( object, trait_name ) )
            if value is None:
                return
            if self.dic.has_key( value ):
                value = self.dic[ value ]
            try:
                self.set( object, trait_name, value, handler )
                self.update( object, trait_name, control )
            except TraitError, excp:
                self.error( description, excp, object.window )

    #-----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        var = tk.StringVar()
        var.set( self.str( object, trait_name ) )
        control       = tk.Entry( parent, textvariable = var )
        control.var   = var
        control.value = self.get_value( control )
        control.bind( '<Return>', self.on_enter )
        control.bind( '<3>',      self.on_restore )
        if self.auto_set:
            control.bind( '<KeyRelease>', self.on_key )
        return control

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        control.var.set( self.str( object, trait_name ) )
        control.configure( bg = WHITE )

    #-----------------------------------------------------------------------------
    #  Handle the user pressing the 'Enter' key in the edit control:
    #-----------------------------------------------------------------------------

    def on_enter ( self, event ):
        control = event.widget
        try:
            self.set( control.object, control.trait_name,
                      self.get_value( control ), control.handler )
        except TraitError, excp:
            self.error( control.description, excp, control )

    #-----------------------------------------------------------------------------
    #  Handle the user releasing a key in the edit control:
    #-----------------------------------------------------------------------------

    def on_key ( self, event ):
        control = event.widget
        value   = self.get_value( control )
        if value != control.value:
            control.value = value
            try:
                setattr( control.object, control.trait_name, value )
                color = WHITE
            except:
                color = '#FFC0C0'
            control.configure( bg = color )

    #-----------------------------------------------------------------------------
    #  Get the actual value corresponding to what the user typed:
    #-----------------------------------------------------------------------------

    def get_value ( self, control ):
        value = control.var.get().strip()
        if not self.dic.has_key( value ):
            return value
        return self.dic[ value ]

#--------------------------------------------------------------------------------
#  'TraitEditorEnum' class:
#--------------------------------------------------------------------------------

class TraitEditorEnum ( tkTraitEditor ):

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
        delegate = tkDelegate( self.on_value_changed )
        values   = self.all_values()
        control  = Pmw.ComboBox( parent,
                                 dropdown           = TRUE,
                                 selectioncommand   = delegate(),
                                 scrolledlist_items = values,
                                 listheight = min( 150, len( values ) * 24 ) )
        delegate.control = control
        control.selectitem( self.current_value( object, trait_name ) )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler,
                              parent ):
        # Create a panel to hold all of the radio buttons:
        panel = tk.Frame( parent )

        # Get the current trait value:
        cur_value = self.current_value( object, trait_name )

        # Initialize loop data:
        values   = self.all_values()
        n        = len( values )
        cols     = self.cols
        rows     = (n + cols - 1) / cols
        var      = tk.StringVar()
        delegate = tkDelegate( self.on_click, var = var, panel = panel )
        incr     = [ n / cols ] * cols
        rem      = n % cols
        for i in range( cols ):
            incr[i] += (rem > i)
        incr[-1] = -(reduce( lambda x, y: x + y, incr[:-1], 0 ) - 1)

        # Add the set of all possible choices to the panel:
        index = 0
        for i in range( rows ):
            for j in range( cols ):
                value   = values[ index ]
                control = tk.Radiobutton( panel,
                                          text     = value.capitalize(),
                                          value    = value,
                                          variable = var,
                                          command  = delegate() )
                control.grid( row = i, col = j, sticky = 'w' )
                if value == cur_value:
                    var.set( value )
                index += incr[j]
                n     -= 1
                if n <= 0:
                    break

        for j in range( cols ):
            panel.columnconfigure( j, weight = 1 )

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

    def on_value_changed ( self, delegate, value ):
        control = delegate.control
        try:
            value = int( value )
        except:
            pass
        self.set( control.object, control.trait_name, value, control.handler )

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the 'custom' radio buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, delegate ):
        panel = delegate.panel
        value = delegate.var.get()
        try:
            value = int( value )
        except:
            pass
        self.set( panel.object, panel.trait_name, value, panel.handler )

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
                             control ):
        TraitEditorImageDialog( object, trait_name, description,
                                   control, handler, self )

    #-----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = tk.Button( parent,
                       image   =  bitmap_cache(
                                    self.current_value( object, trait_name ) +
                                    self.suffix, self.path ),
                       bg      = WHITE )
        control.configure( command = tkDelegate( self.on_popup,
                                                 widget = control )() )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler,
                              parent ):
        # Create a panel to hold all of the radio buttons:
        panel = tk.Frame( parent )

        # Save the information the event handler needs:
        panel.object        = object
        panel.trait_name = trait_name
        panel.handler       = handler

        # Add the image buttons to the panel:
        self.create_image_grid( panel, self.on_click )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Populate a specified window with a grid of image buttons:
    #----------------------------------------------------------------------------

    def create_image_grid ( self, parent, handler ):
        # Add the set of all possible choices:
        i    = 0
        cols = self.cols
        for value in self.all_values():
            control = tk.Button( parent,
                         image   = bitmap_cache( value + self.suffix,
                                                 self.path ),
                         command = tkDelegate( handler,
                                      parent = parent,
                                      value  = value )() )
            control.grid( row = i / cols, col = i % cols, sticky = 'w',
                          padx = 2, pady = 2 )
            i += 1

        parent.columnconfigure( cols, weight = 1 )

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        control.configure( image = bitmap_cache(
                              self.current_value( object, trait_name ) +
                              self.suffix, self.path ) )

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the image buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, delegate ):
        parent = delegate.parent
        self.set( parent.object, parent.trait_name, delegate.value,
                  parent.handler )

#--------------------------------------------------------------------------------
#  'TraitEditorList' class:
#--------------------------------------------------------------------------------

class TraitEditorList ( tkTraitEditor ):

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
        delegate = tkDelegate( self.on_value_changed )
        labels   = self.all_labels()
        control  = Pmw.ComboBox( parent,
                                 dropdown           = TRUE,
                                 selectioncommand   = delegate(),
                                 scrolledlist_items = labels,
                                 listheight = min( 150, len( labels ) * 24 ) )
        delegate.control = control
        try:
            control.selectitem( self.all_values().index(
                    self.current_value( object, trait_name )[0] ) )
        except:
            pass
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler,
                              parent ):
        # Create a panel to hold all of the radio buttons:
        panel = tk.Frame( parent )

        # Get the current trait value:
        cur_value = self.current_value( object, trait_name )

        # Create a sizer to manage the radio buttons:
        labels   = self.all_labels()
        values   = self.all_values()
        n        = len( values )
        cols     = self.cols
        rows     = (n + cols - 1) / cols
        incr     = [ n / cols ] * cols
        rem      = n % cols
        for i in range( cols ):
            incr[i] += (rem > i)
        incr[-1] = -(reduce( lambda x, y: x + y, incr[:-1], 0 ) - 1)

        # Add the set of all possible choices:
        index = 0
        for i in range( rows ):
            for j in range( cols ):
                if n > 0:
                    value    = values[ index ]
                    var      = tk.IntVar()
                    delegate = tkDelegate( self.on_click,
                                           var   = var,
                                           panel = panel,
                                           value = value )
                    control  = tk.Checkbutton( panel,
                                              text     = labels[ index ],
                                              variable = var,
                                              command  = delegate() )
                    control.grid( row = i, col = j, sticky = 'w' )
                    var.set( value in cur_value )
                    index += incr[j]
                    n     -= 1
                    if n <= 0:
                        break

        for j in range( cols ):
            panel.columnconfigure( j, weight = 1 )

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

    def on_value_changed ( self, delegate, value ):
        control = delegate.control
        value   = self.mapping[ value ]
        if not self.is_string( control.object, control.trait_name ):
            value = [ value ]
        self.set( control.object, control.trait_name, value, control.handler )

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the 'custom' radio buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, delegate ):
        panel     = delegate.panel
        value     = delegate.value
        cur_value = self.current_value( panel.object, panel.trait_name )
        if delegate.var.get():
            cur_value.append( value )
        else:
            cur_value.remove( value )
        if self.is_string(  panel.object, panel.trait_name ):
            cur_value = ','.join( cur_value )
        self.set( panel.object, panel.trait_name, cur_value, panel.handler )

#--------------------------------------------------------------------------------
#  'TraitEditorBoolean' class:
#--------------------------------------------------------------------------------

class TraitEditorBoolean ( tkTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Create an in-place editable view of the current value of the
    #  'trait_name' trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        var     = tk.IntVar()
        control = tk.Checkbutton( parent,
                                  text     = '',
                                  variable = var,
                                  anchor   = 'w' )
        control.configure( command = tkDelegate( self.on_value_changed,
                                                 control = control,
                                                 var     = var )() )
        try:
            value = getattr( object, trait_name + '_' )
        except:
            value = getattr( object, trait_name )
        var.set( value )
        return control

    #-----------------------------------------------------------------------------
    #  Handle the user clicking on the checkbox:
    #-----------------------------------------------------------------------------

    def on_value_changed ( self, delegate ):
        control = delegate.control
        self.set( control.object, control.trait_name, delegate.var.get(),
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
        value = self.str( object, trait_name )
        if abs( self.high - self.low ) > 1000:
            var = tk.StringVar()
            var.set( value )
            control       = tk.Entry( parent, textvariable = var )
            control.var   = var
            control.value = value
            control.bind( '<Return>',     self.on_enter )
            control.bind( '<KeyRelease>', self.on_key )
            control.bind( '<3>',          self.on_restore )
        else:
            control = Pmw.Counter( parent )
            control.configure( datatype = {
                                  'counter':   self.on_value_changed,
                                  'control':   control,
                                  'min_value': self.low,
                                  'max_value': self.high } )
            control._counterEntry.setentry( str( value ) )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler,
                              parent ):
        if abs( self.high - self.low ) > 15:
            return self.simple_editor( object, trait_name, description,
                                       handler, parent )
        return TraitEditorEnum.custom_editor( self, object, trait_name,
                                                 description, handler, parent )

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
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        if isinstance( control, tk.Entry ):
            control.var.set( self.str( object, trait_name ) )
            control.configure( bg = WHITE )
        else:
            control.entryfield.configure(
                text = str( getattr( object, trait_name ) ) )

    #-----------------------------------------------------------------------------
    #  Handle the user selecting a new value from the spin control:
    #-----------------------------------------------------------------------------

    def on_value_changed ( self, value, factor, incr,
                                 control   = None,
                                 min_value = None,
                                 max_value = None ):
        value = min( max_value,
                     max( min_value, int( value ) + factor * incr ) )
        try:
            self.set( control.object, control.trait_name, value,
                      control.handler )
            return str( value )
        except:
            raise ValueError

    #-----------------------------------------------------------------------------
    #  Handle the user pressing the 'Enter' key in the edit control:
    #-----------------------------------------------------------------------------

    def on_enter ( self, event ):
        control = event.widget
        try:
            self.set( control.object, control.trait_name,
                      control.var.get().strip(), control.handler )
        except TraitError, excp:
            self.error( control.description, excp, control )

    #-----------------------------------------------------------------------------
    #  Handle the user releasing a key in the edit control:
    #-----------------------------------------------------------------------------

    def on_key ( self, event ):
        control = event.widget
        value   = control.var.get()
        if value != control.value:
            control.value = value
            try:
                setattr( control.object, control.trait_name, value )
                color = WHITE
            except:
                color = '#FFC0C0'
            control.configure( bg = color )

#-------------------------------------------------------------------------------
#  'TraitEditorImageDialog' class:
#-------------------------------------------------------------------------------

class TraitEditorImageDialog ( tk.Toplevel ):

    #-----------------------------------------------------------------------------
    #  Initialize the object:
    #-----------------------------------------------------------------------------

    def __init__ ( self, object, trait_name, description, control, handler,
                         editor ):
        tk.Toplevel.__init__( self, control )
        self.title( 'Choose ' + description )
        self.bind( '<Escape>',  self.on_close_key )

        # Initialize instance data:
        self.object        = object
        self.trait_name = trait_name
        self.control       = control
        self.handler       = handler
        self.editor        = editor

        # Create the grid of image buttons:
        editor.create_image_grid( self, self.on_click )

    #----------------------------------------------------------------------------
    #  Handle the user hitting the 'Esc'ape key:
    #----------------------------------------------------------------------------

    def on_close_key ( self ):
        self.destroy()

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the choice buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, delegate ):
        self.editor.set( self.object, self.trait_name,
                         delegate.value, self.handler )
        self.editor.update( self.object, self.trait_name, self.control )
        self.destroy()

#--------------------------------------------------------------------------------
#  Convert an image file name to a cached bitmap:
#--------------------------------------------------------------------------------

# Bitmap cache dictionary (indexed by filename):
_bitmap_cache = {}

### NOTE: This needs major improvements:
app_path        = None
traits_path = None

def bitmap_cache ( name, path = None ):
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
    bitmap   = _bitmap_cache.get( filename )
    if bitmap is None:
        bitmap = _bitmap_cache[ filename ] = tk.PhotoImage( file = filename )
    return bitmap

#-------------------------------------------------------------------------------
#  Standard colors:
#-------------------------------------------------------------------------------

standard_colors = {
   'aquamarine':          '#70DB93',
   'black':               '#000000',
   'blue':                '#0000FF',
   'blue violet':         '#9F5F9F',
   'brown':               '#A52A2A',
   'cadet blue':          '#5F9F9F',
   'coral':               '#FF7F00',
   'cornflower blue':     '#42426F',
   'cyan':                '#00FFFF',
   'dark green':          '#2F4F2F',
   'dark grey':           '#2F2F2F',
   'dark olive green':    '#4F4F2F',
   'dark orchid':         '#9932CC',
   'dark slate blue':     '#6B238E',
   'dark slate grey':     '#2F4F4F',
   'dark turquoise':      '#7093DB',
   'dim grey':            '#545454',
   'firebrick':           '#8E2323',
   'forest green':        '#238E23',
   'gold':                '#CC7F32',
   'goldenrod':           '#DBDB70',
   'green':               '#00FF00',
   'green yellow':        '#93DB70',
   'grey':                '#808080',
   'indian red':          '#4F2F2F',
   'khaki':               '#9F9F5F',
   'light blue':          '#BFD8D8',
   'light grey':          '#C0C0C0',
   'light steel':         '#000000',
   'lime green':          '#32CC32',
   'magenta':             '#FF00FF',
   'maroon':              '#8E236B',
   'medium aquamarine':   '#32CC99',
   'medium blue':         '#3232CC',
   'medium forest green': '#6B8E23',
   'medium goldenrod':    '#EAEAAD',
   'medium orchid':       '#9370DB',
   'medium sea green':    '#426F42',
   'medium slate blue':   '#7F00FF',
   'medium spring green': '#7FFF00',
   'medium turquoise':    '#70DBDB',
   'medium violet red':   '#DB7093',
   'midnight blue':       '#2F2F4F',
   'navy':                '#23238E',
   'orange':              '#CC3232',
   'orange red':          '#FF007F',
   'orchid':              '#DB70DB',
   'pale green':          '#8FBC8F',
   'pink':                '#BC8FEA',
   'plum':                '#EAADEA',
   'purple':              '#B000FF',
   'red':                 '#FF0000',
   'salmon':              '#6F4242',
   'sea green':           '#238E6B',
   'sienna':              '#8E6B23',
   'sky blue':            '#3299CC',
   'slate blue':          '#007FFF',
   'spring green':        '#00FF7F',
   'steel blue':          '#236B8E',
   'tan':                 '#DB9370',
   'thistle':             '#D8BFD8',
   'turquoise':           '#ADEAEA',
   'violet':              '#4F2F4F',
   'violet red':          '#CC3299',
   'wheat':               '#D8D8BF',
   'white':               '#FFFFFF',
   'yellow':              '#FFFF00',
   'yellow green':        '#99CC32'
}

#--------------------------------------------------------------------------------
#  Convert a number into a Tkinter color string:
#--------------------------------------------------------------------------------

def num_to_color ( object, name, value ):
    if type( value ) == StringType:
        if ((len( value ) == 7) and (value[0] == '#') and
            (eval( '0x' + value[1:] ) >= 0)):
            return value
        raise TraitError
    return '#%06X' % int( value )

num_to_color.info = ("a string of the form '#RRGGBB' or a number, which in "
                     "hex is of the form 0xRRGGBB, where RR is red, GG is "
                     "green, and BB is blue")

#--------------------------------------------------------------------------------
#  'TraitEditorColor' class:
#--------------------------------------------------------------------------------

class TraitEditorColor ( tkTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Interactively edit the 'trait_name' color trait of 'object':
    #-----------------------------------------------------------------------------

    def popup_editor ( self, object, trait_name, description, handler,
                             control ):
        color = cc.askcolor( self.to_tk_color( object,
                                               trait_name ) )[1].upper()
        if color is not None:
            self.set( object, trait_name, self.from_tk_color( color ),
                      handler )
            self.update( object, trait_name, control )

    #-----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = tkTraitEditor.simple_editor( self, object, trait_name,
                                                  description, handler, parent )
        self.update_color( object, trait_name, control )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler,
                              parent ):
        # Create a panel to hold all of the buttons:
        panel       = tk.Frame( parent )
        panel.color = self.simple_editor( object, trait_name, description,
                                          handler, panel )
        panel.color.grid( row = 0, col = 0, sticky = 'nesw' )

        # Add all of the color choice buttons:
        panel2 = tk.Frame( panel )
        for i in range( len( color_samples ) ):
            tk.Button( panel2,
                       bg      = color_samples[i],
                       font    = 'Helvetica 1',
                       command = tkDelegate( self.on_click,
                                    panel = panel,
                                    color = color_samples[i] )() ).grid(
                       row = i / 12, col = i % 12, sticky = 'ew' )
        for i in range( 12 ):
            panel2.columnconfigure( i, weight = 1 )

        panel2.grid( row = 0, col = 1, sticky = 'nesw', padx = 4 )
        panel.columnconfigure( 0, minsize = 70 )
        panel.columnconfigure( 1, weight  =  1 )

        # Return the panel as the result:
        return panel

    #----------------------------------------------------------------------------
    #  Handle the user clicking one of the 'custom' color buttons:
    #----------------------------------------------------------------------------

    def on_click ( self, delegate ):
        panel = delegate.panel
        self.set( panel.object, panel.trait_name,
                  self.from_tk_color( delegate.color ),
                  panel.handler )
        self.update( panel.object, panel.trait_name, panel.color )

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        tkTraitEditor.update( self, object, trait_name, control )
        self.update_color( object, trait_name, control )

    #-------------------------------------------------------------------------------
    #  Update the foreground/background colors of the control widget:
    #-------------------------------------------------------------------------------

    def update_color ( self, object, trait_name, control ):
        bg_color = self.to_tk_color( object, trait_name )
        red      = eval( '0x%s' % bg_color[1:3] )
        green    = eval( '0x%s' % bg_color[3:5] )
        blue     = eval( '0x%s' % bg_color[5:7] )
        fg_color = ( '#FFFFFF', '#000000' )[
           (red > 192) or(green > 192) or(blue > 192) ]
        control.configure( bg = bg_color, fg = fg_color )

    #-------------------------------------------------------------------------------
    #  Get the Tkinter color equivalent of the object trait:
    #-------------------------------------------------------------------------------

    def to_tk_color ( self, object, trait_name ):
        try:
            cur_color = getattr( object, trait_name + '_' )
        except:
            cur_color = getattr( object, trait_name )
        if cur_color is None:
            return WHITE
        return cur_color

    #-------------------------------------------------------------------------------
    #  Get the application equivalent of a Tkinter value:
    #-------------------------------------------------------------------------------

    def from_tk_color ( self, color ):
        return color

#-------------------------------------------------------------------------------
#  Define Tkinter specific color traits:
#-------------------------------------------------------------------------------

# Create a singleton color editor:
color_editor = TraitEditorColor()

# Color traits:
color_trait       = Trait( 'white', standard_colors, num_to_color,
                                 editor = color_editor )
clear_color_trait = Trait( 'clear', None, standard_colors,
                                 { 'clear': None }, num_to_color,
                                 editor = color_editor )

#--------------------------------------------------------------------------------
#  Convert a string into a valid Tkinter font (if possible):
#--------------------------------------------------------------------------------

slant_types  = [ 'roman', 'italic' ]
weight_types = [ 'bold' ]
font_noise   = [ 'pt', 'point', 'family' ]

def str_to_font ( object, name, value ):
    try:
        size      = 10
        family    = []
        slant     = 'roman'
        weight    = 'normal'
        underline = overstrike = 0
        for word in value.split():
            lword = word.lower()
            if lword in slant_types:
                slant = lword
            elif lword in weight_types:
                weight = lword
            elif lword == 'underline':
                underline = 1
            elif lword == 'overstrike':
                overstrike = 1
            elif lword not in font_noise:
                try:
                    size = int( lword )
                except:
                    family.append( word )
        if len( family ) == 0:
            family = [ 'Helvetica' ]
        return tkFont.Font( family     = ' '.join( family ),
                            size       = size,
                            weight     = weight,
                            slant      = slant,
                            underline  = underline,
                            overstrike = overstrike )
    except:
        pass
    raise TraitError

str_to_font.info = ( "a string describing a font (e.g. '12 pt bold italic' "
                     "or 'Arial bold 14 point')" )

#--------------------------------------------------------------------------------
#  'TraitEditorFont' class:
#--------------------------------------------------------------------------------

class TraitEditorFont ( tkTraitEditor ):

    #-----------------------------------------------------------------------------
    #  Create a static view of the current value of the 'trait_name'
    #  trait of 'object':
    #-----------------------------------------------------------------------------

    def simple_editor ( self, object, trait_name, description, handler,
                              parent ):
        control = tkTraitEditor.simple_editor( self, object, trait_name,
                                                  description, handler, parent )
        control.is_custom = FALSE
        self.update_font( object, trait_name, control )
        return control

    #----------------------------------------------------------------------------
    #  Create an in-place custom view of the current value of the
    # 'trait_name' trait of 'object':
    #----------------------------------------------------------------------------

    def custom_editor ( self, object, trait_name, description, handler,
                              parent ):
        # Create a panel to hold all of the buttons:
        panel = tk.Frame( parent )

        # Add the standard font control:
        font = panel.font = self.simple_editor( object, trait_name,
                                                description, handler, panel )
        font.configure( anchor = 'w' )
        font.is_custom = TRUE
        font.grid( row = 0, col = 0, columnspan = 2, sticky = 'ew', pady = 3 )

        # Add all of the font choice controls:
        delegate = tkDelegate( self.on_value_changed, panel = panel )
        values   = self.all_facenames()
        panel.facename = control = Pmw.ComboBox( panel,
                         dropdown           = TRUE,
                         selectioncommand   = delegate(),
                         scrolledlist_items = values,
                         listheight         = min( 150, len( values ) * 24 ) )
        control.grid( row = 1, col = 0 )

        panel.pointsize = control = Pmw.ComboBox( panel,
                          dropdown           = TRUE,
                          selectioncommand   = delegate(),
                          scrolledlist_items = point_sizes )
        control.grid( row = 1, col = 1, padx = 4 )

        # Initialize the control's with the object's current trait value:
        self.update_font( object, trait_name, font )

        # Return the panel as the result:
        return panel

    #-----------------------------------------------------------------------------
    #  Update the contents of a previously created viewer control with the new
    #  value of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def update ( self, object, trait_name, control ):
        tkTraitEditor.update( self, object, trait_name, control )
        self.update_font( object, trait_name, control )

    #-------------------------------------------------------------------------------
    #  Update the font of the control widget:
    #-------------------------------------------------------------------------------

    def update_font ( self, object, trait_name, control ):
        cur_font = self.to_tk_font( object, trait_name )
        size     = cur_font.cget( 'size' )
        if control.is_custom:
            panel = control.master
            try:
                panel.facename.selectitem( cur_font.cget( 'family' ) )
            except:
                panel.facename.selectitem( 0 )
            try:
                panel.pointsize.selectitem( size )
            except:
                panel.pointsize.selectitem( '10' )
        cur_font.configure( size = min( 10, size ) )
        control.configure( font = cur_font )

    #-----------------------------------------------------------------------------
    #  Return the text representation of the 'trait' trait of 'object':
    #-----------------------------------------------------------------------------

    def str ( self, object, trait_name ):
        font   = getattr( object, trait_name )
        size   = font.cget( 'size' )
        family = font.cget( 'family' )
        slant  = font.cget( 'slant' )
        weight = font.cget( 'weight' )
        return '%s point %s %s %s' % ( size, family, slant.capitalize(),
                                       weight.capitalize() )

    #----------------------------------------------------------------------------
    #  Return a list of all available font facenames:
    #----------------------------------------------------------------------------

    def all_facenames ( self ):
        return ( 'Courier',
                 'Times',
                 'Helvetica',
                 'Arial',
                 'Verdana' )

    #----------------------------------------------------------------------------
    #  Handle the user selecting a new facename or point size:
    #----------------------------------------------------------------------------

    def on_value_changed ( self, delegate, unused ):
        panel  = delegate.panel
        size   = panel.pointsize.get()
        family = panel.facename.get()
        font   = tkFont.Font( family = family, size = size )
        self.set( panel.object, panel.trait_name,
                  self.from_tk_font( font ), panel.handler )
        self.update( panel.object, panel.trait_name, panel.font )

    #-------------------------------------------------------------------------------
    #  Return a Tkinter Font object corresponding to a specified object's font
    #  trait:
    #-------------------------------------------------------------------------------

    def to_tk_font ( self, object, trait_name ):
        return getattr( object, trait_name )

    #-------------------------------------------------------------------------------
    #  Get the application equivalent of a Tkinter Font value:
    #-------------------------------------------------------------------------------

    def from_tk_font ( self, font ):
        return font

#-------------------------------------------------------------------------------
#  Define a Tkinter specific font trait:
#-------------------------------------------------------------------------------

font_trait = Trait( 'Arial 10', tkFont.Font, str_to_font,
                          editor = TraitEditorFont() )

#-------------------------------------------------------------------------------
#  'tkDelegate' class:
#-------------------------------------------------------------------------------

class tkDelegate:

    #----------------------------------------------------------------------------
    #  Initialize the object:
    #----------------------------------------------------------------------------

    def __init__ ( self, delegate = None, **kw ):
        self.delegate = delegate
        for name, value in kw.items():
            setattr( self, name, value )

    #----------------------------------------------------------------------------
    #  Return the handle method for the delegate:
    #----------------------------------------------------------------------------

    def __call__ ( self ):
        return self.on_event

    #----------------------------------------------------------------------------
    #  Handle an event:
    #----------------------------------------------------------------------------

    def on_event ( self, *args ):
        self.delegate( self, *args )
