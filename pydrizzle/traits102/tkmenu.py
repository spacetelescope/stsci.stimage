#===============================================================================
#
#  Dynamically construct Tkinter Menus from a supplied string string
#  description of the menu.
#
#  Written By: David C. Morrill
#
#  Date: 10/15/2002
#
#  Version: 0.1
#
#  (c) Copyright 2002 by Enthought, Inc.
#
#===============================================================================
#
#  Menu Description Syntax:
#
#     submenu_label {help_string}
#        menuitem_label | accelerator {help_string} [~/-name]: code
#        -
#
#  where:
#     submenu_label  = Label of a sub menu
#     menuitem_label = Label of a menu item
#     help_string    = Help string to display on the status line (optional)
#     accelerator    = Accelerator key (e.g. Ctrl-C) (| and key are optional)
#     [~]            = Menu item checkable, but not checked initially (optional)
#     [/]            = Menu item checkable, and checked initially (optional)
#     [-]            = Menu item disabled initially (optional)
#     [name]         = Symbolic name used to refer to menu item (optional)
#     code           = Python code invoked when menu item is selected
#
#===============================================================================

#===============================================================================
#  Imports:
#===============================================================================
from __future__ import division # confidence high

import Tkinter as tk
import re
import string

from types  import StringType

#===============================================================================
#  Constants:
#===============================================================================

TRUE  = 1
FALSE = 0

DEBUG = TRUE

help_pat    = re.compile( r'(.*){(.*)}(.*)' )
options_pat = re.compile( r'(.*)\[(.*)\](.*)' )

#===============================================================================
#  'MakeMenu' class:
#===============================================================================

class MakeMenu:

    def __init__ ( self, desc, owner, popup = FALSE, window = None ):
        self.owner = owner
        if window is None:
            window = owner
        self.window  = window
        self.desc    = desc.split( '\n' )
        self.index   = 0
        self.menus   = []
        if popup:
            self.menu = menu = tk.Menu( window, tearoff = FALSE )
            self.parse( menu, -1 )
        else:
            self.menubar = tk.Frame( window, relief = tk.RAISED, borderwidth = 1 )
            self.parse( None, -1 )
            self.menubar.tk_menuBar( *self.menus )

    #============================================================================
    #  Recursively parse menu items from the description:
    #============================================================================

    def parse ( self, menu, indent ):

        cur_id = 0
        while TRUE:

            # Make sure we have not reached the end of the menu description yet:
            if self.index >= len( self.desc ):
                return

            # Get the next menu description line and check its indentation:
            dline    = self.desc[ self.index ]
            line     = dline.lstrip()
            indented = len( dline ) - len( line )
            if indented <= indent:
                return

            # Indicate that the current line has been processed:
            self.index += 1

            # Check for a blank or comment line:
            if (line == '') or (line[0:1] == '#'):
                continue

            # Check for a menu separator:
            if line[0:1] == '-':
                menu.add_separator()
                cur_id += 1
                continue

            # Extract the help string (if any):
            help  = ''
            match = help_pat.search( line )
            if match:
                help = ' ' + match.group(2).strip()
                line = match.group(1) + match.group(3)

            # Check for a menu item:
            col = line.find( ':' )
            if col >= 0:
                handler = line[ col + 1: ].strip()
                if handler != '':
                    try:
                        exec 'def handler(event=None,self=self.owner):\n %s\n' % handler
                    except:
                        handler = null_handler
                else:
                    try:
                        exec 'def handler(event=None,self=self.owner):\n%s\n' % (
                             self.get_body( indented ), ) in globals()
                    except:
                        handler = null_handler
                not_checked = checked = disabled = FALSE
                line       = line[ : col ]
                match      = options_pat.search( line )
                if match:
                    line = match.group(1) + match.group(3)
                    not_checked, checked, disabled, name = option_check( '~/-',
                                                           match.group(2).strip() )
                    check_var = None
                    if not_checked or checked:
                        check_var = tk.IntVar()
                        check_var.set( checked )
                    if name != '':
                        setattr( self.owner, name,
                                 MakeMenuItem( menu, cur_id, check_var ) )
                label = line.strip().replace( '&', '' )
                col   = label.find( '|' )
                if col >= 0:
                    key_name = label[ col + 1: ].strip()
                    label    = label[ : col ].strip()
                    self.window.bind( self.binding_for( key_name ), handler )
                if checked or not_checked:
                    menu.add_checkbutton( label    = label,
                                          command  = handler,
                                          variable = check_var )
                else:
                    menu.add_command( label   = label,
                                      command = handler )
                if col >= 0:
                    menu.entryconfig( cur_id, accelerator = key_name )
                if disabled:
                    menu.entryconfig( cur_id,  state =  tk.DISABLED )
                cur_id += 1
                continue

            # Else must be the start of a sub menu:
            label = line.strip().replace( '&', '' )
            if menu is None:
                menubar = tk.Menubutton( self.menubar, text = label )
                self.menus.append( menubar )
                menubar.pack( side = tk.LEFT )
                menubar[ 'menu' ] = submenu = tk.Menu( menubar, tearoff = FALSE )
            else:
                submenu = tk.Menu( menu, tearoff = FALSE )
                menu.add_cascade( label = label, menu = submenu )

            # Recursively parse the sub-menu:
            self.parse( submenu, indented )

            cur_id += 1

    #============================================================================
    #  Return the body of an inline method:
    #============================================================================

    def get_body ( self, indent ):
        result = []
        while self.index < len( self.desc ):
            line = self.desc[ self.index ]
            if (len( line ) - len( line.lstrip() )) <= indent:
                break
            result.append( line )
            self.index += 1
        result = string.join( result, '\n' ).rstrip()
        if result != '':
            return result
        return '  pass'

    #----------------------------------------------------------------------------
    #  Return the correct Tk binding for a specified key:
    #----------------------------------------------------------------------------

    def binding_for ( self, key_name ):
        key_name = key_name.replace( 'Ctrl-', 'Control-' )
        if key_name[-2:-1] == '-':
            key_name = key_name[:-1] + key_name[-1].lower()
        return '<%s>' % key_name

#===============================================================================
#  'MakeMenuItem' class:
#===============================================================================

class MakeMenuItem:

    def __init__ ( self, menu, id, var ):
        self.menu = menu
        self.id   = id
        self.var  = var

    def checked ( self, check = None ):
        if check is not None:
            self.var.set( check )
            return check
        return self.var.get()

    def toggle ( self ):
        checked = not self.checked()
        self.checked( checked )
        return checked

    def enabled ( self, enable = None ):
        if enable is not None:
            self.menu.entryconfig( self.id,
                           state = ( tk.DISABLED, tk.NORMAL )[ enable ] )
            return enable
        return (self.menu.entrycget( self.id, 'state' ) == tk.NORMAL)

    def label ( self, label = None ):
        if label is not None:
            self.menu.entryconfig( self.id, label = label )
            return label
        return self.menu.entrycget( self.id, 'label' )

#===============================================================================
#  Determine whether a string contains any specified option characters, and
#  remove them if it does:
#===============================================================================

def option_check ( test, string ):
    result = []
    for char in test:
        col = string.find( char )
        result.append( col >= 0 )
        if col >= 0:
            string = string[ : col ] + string[ col + 1: ]
    return result + [ string.strip() ]

#===============================================================================
#  Null menu option selection handler:
#===============================================================================

def null_handler ( event ):
    print 'null_handler invoked'
    pass
