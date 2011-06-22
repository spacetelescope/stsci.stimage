#----------------------------------------------------------------------------
#
# module:  opusutil
#
# General purpose OPUS-Python functions
#
# 01/20/05  52348  Sherbert Remove "MSG_DEGUG" to obviate its use
# 12/15/05  54226  Sontag   Added UsingLvl() and __all_valid_levels
#---------------------------------------------------------------------------
from __future__ import division # confidence high
import os
import sys
import string
import time

# Read the environment setting for the PrintMsg report level
# or default if not defined
#
# This section is done only once, during the "import opusutil" statement
#
try:
  _report_level_val = os.environ['MSG_REPORT_LEVEL']
except KeyError:
  _report_level_val = "MSG_INFO"
#
# define the active levels for message reporting
if _report_level_val == "MSG_NONE":
  _active_levels = []
elif _report_level_val == "MSG_FATAL":
  _active_levels = ["F"]
elif _report_level_val == "MSG_ERROR":
  _active_levels = ["F","E"]
elif _report_level_val[0:8] == "MSG_WARN":
  _active_levels = ["F","E","W"]
elif (_report_level_val == "MSG_DIAG"  or
      _report_level_val == "MSG_ALL" ):
  _active_levels = ["F","E","W","I","D"]
else: # default
  _active_levels = ["F","E","W","I"]
#
# define the list of all possible levels
__all_valid_levels = ["F","E","W","I","D"]
#
# set a default message trailer device
_trl_fd = None

#--------------------------------------------------------------------------
#
# name:    OpenTrl
#
# purpose: Assigns a file object for trailer messages.
#
# input:   filespec - name of trailer file
#
# return:  none
#
# 01/13/03  46981  MSwam    first version
#---------------------------------------------------------------------------
def OpenTrl (filespec):
  global _trl_fd
  _trl_fd = open(filespec, 'a')

#--------------------------------------------------------------------------
#
# name:    CloseTrl
#
# purpose: Closes an open trailer file.
#
# input:   none
#
# return:  none
#
# 01/13/03  46981  MSwam    first version
#---------------------------------------------------------------------------
def CloseTrl ():
  try:
    _trl_fd.close()
  except:
    # failure to close is not an error
    pass

#--------------------------------------------------------------------------
#
# name:    UsingLvl
#
# purpose: States whether the input level will be used or ignored.
#
# input:   level - msg level (see PrintMsg)
#
# return:  True if it will be used, False if not
#
# 12/15/05  54226  Sontag   first version
#---------------------------------------------------------------------------
def UsingLvl (level):
   # input check
   assert level in __all_valid_levels, "Invalid message level: "+level

   # if level is not active say so
   if level in _active_levels:
      return True
   else:
      return False

#--------------------------------------------------------------------------
#
# name:    PrintMsg
#
# purpose: Prints a datetime-stamped string to STDOUT, along with a
#          severity-level indicator.
#
# input:   level - msg level (e.g. D,I,W,E,F for diag,info,warning,error,fatal)
#          msg   - text string to print
#          modulename - optional name of module text string to print
#
# return:  none
#
# 01/13/03  46981  MSwam    first version
# 10/21/04  52107  Sherbert On a hint from Jim Hare: add stdout flush so that
#                           messages are closer to where erros actually happen
# 01/20/05  52348  Sherbert Change "DEGUG" to "DIAG" to be consistent
# 12/15/05  54226  Sontag   Shift some logic to UsingLvl()
#---------------------------------------------------------------------------
def PrintMsg (level, msg, module_name=""):
    # if level is not active, return
    if not UsingLvl(level):
      return

    # get the current date/time and format the message
    current_time = time.strftime("%Y%j%H%M%S", time.localtime(time.time()))

    if level == "W":
      leveltxt = "WARNING"
    elif level == "E":
      leveltxt = "ERROR"
    elif level == "F":
      leveltxt = "FATAL"
    elif level == "D":
      leveltxt = "DIAG"
    else:
      leveltxt = "INFO"

    # Want a consistent format if module_name is or isn't provided
    if len(module_name) != 0:
      module_name = module_name + '-'

    # write to stdout and the trailer, if opened
    print current_time+'-'+level+'-'+leveltxt+'-'+module_name+msg
    if _trl_fd:
      try:
        _trl_fd.write(current_time+'-'+level+'-'+leveltxt+'-'+module_name+msg+'\n')
      except:
        # failed trailer writes are not considered errors
        pass
    #
    sys.stdout.flush()
    return

#--------------------------------------------------------------------------
#
# name:    RemoveIfThere
#
# purpose: Removes a file from  disk, if it is there.
#          No error or warning is given if the file is not present.
#          Other errors, like an illegal filespec, result in an
#          exception.
#
# input:   filename
#
# return:  none
#
# 01/13/03  46981  MSwam    first version
#---------------------------------------------------------------------------
def RemoveIfThere (filename):

    if os.path.exists(filename):
       os.remove(filename)

    return

#--------------------------------------------------------------------------
#
# name:    StretchFile
#
# purpose: Locates a file on disk given a filename with a
#          VMS-style "stretched" variable name, if one exists.
#          (e.g. OPUS_DEFINITIONS_DIR:my.path)
#
# description:  Uses the OPUS tool "osfile_stretch_file" by issuing
#               a subprocess to invoke the tool.  If the stretched
#               filename argument given resolves to a disk file,
#               the name of the disk file is returned.  If not, the
#               input stretched filename input is returned as the output.
#
# input:   stretched_filename
#
# return:  resolved filename or an echo of the input filename
#
# 01/13/03  46981  MSwam    first version
#---------------------------------------------------------------------------
def StretchFile (stretched_filename):
    #
    # use the osfile_stretch_file OPUS tool to resolve the stretch
    (Stdin, Stdouterr) = os.popen4("osfile_stretch_file "+stretched_filename)
    os.wait()
    #
    # read the tool output, trim newline from filename output
    result = Stdouterr.readline()[:-1]
    Stdin.close()
    Stdouterr.close()
    #
    return result

#--------------------------------------------------------------------------
#
# name:    FileToList
#
# purpose: Reads a file and converts each line to a list entry.
#
# input:   filename - name of file to read
#
# return:  list of tokens from file
#
# 02/13/03  xxxxx  MSwam    first version
#---------------------------------------------------------------------------
def FileToList (filename):
  theList = []
  for line in open(filename, 'r').readlines():
    # trim newline and save in list
    theList.append(line[:-1])
  #
  return theList

#--------------------------------------------------------------------------
#
# name:    ResourceToMap
#
# purpose: Reads a file of keyword = value lines and converts each line to a
#          map entry keyed on "keyword".  Ignores lines that start with !
#
# input:   filename - name of file to read
#
# return:  map of tokens from file
#
# 01/15/04  49551  MSwam    first version
#---------------------------------------------------------------------------
def ResourceToMap (filename):
  theMap = {}
  for line in open(filename, 'r').readlines():
    # trim comment (find ! and remove from there until EOS)
    # and if line now empty, read next line
    theLine = string.split(line,'!')
    if len(theLine[0]) == 0:
      continue
    # work with theLine[0] now
    # parse "keyword = value" into a map entry by splitting on =
    # then trimming leading/trailing blanks and newline from parts
    parts = string.split(theLine[0],"=")
    if len(parts) > 1:
      # non-blank line, use it
      key = string.strip(parts[0])
      value = string.strip(parts[1])
      theMap[key] = value
  #
  return theMap

#========================================================================
# TEST
# % python opusutil.py
#========================================================================
if __name__ == "__main__":
  fname="TestTrailer.tst"
  OpenTrl(fname)
  print "################################"
  PrintMsg("F","test msg  1")
  PrintMsg("F","test msg  2",__name__)
  PrintMsg("E","test msg  3")
  PrintMsg("E","test msg  4",__name__)
  PrintMsg("W","test msg  5")
  PrintMsg("W","test msg  6",__name__)
  PrintMsg("I","test msg  7")
  PrintMsg("I","test msg  8",__name__)
  PrintMsg("D","test msg  9")
  PrintMsg("D","test msg 10",__name__)
  CloseTrl()
  print "================================"
  print "now here is the trailer content:"
  print "==========Beg trl==============="
  for line in open(fname, 'r').readlines():
    print line,
  print "==========End trl==============="
  aList = FileToList(fname)
  print "after reading to a list:"+str(aList)
  RemoveIfThere(fname)
  print "stretching OPUS_DEFINITIONS_DIR:null.path = "+StretchFile ("OPUS_DEFINITIONS_DIR:null.path")
