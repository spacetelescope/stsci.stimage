#  Program: stis_assoc_support.py
#  Author:  Christopher Hanley
#  History:
#   Version 0.1, 12/06/2004: Initial Creation -- CJH
__version__ = '0.1 (12/06/2004)'
__author__  = 'Christopher Hanley'


# import external modules
import pyfits

def parseSTIS(inputfile):
    """
    FUNCTION:   parseSTIS
    PURPOSE :   the parse STIS function is used to convert STIS assocation files into
                single exposure STIS files.  The primary header of the original assocation
                will populate the primary header of the newly created file.  The first extension
                of the new file will contain a copy of the individual header, sci, err, dq, 
                extension of input assocation.  A STIS assocation of N individual exposures will
                be used to create N individual files.
    INPUT   :   a Python list object populated by strings.  Each string represents the name of a
                file on disk.
    OUTPUT  :   newfilelist - a Python list object populated by strings.  Each string represents 
                the name of a newly created STIS single exposure file on disk.
                
                specialparse - a Boolean value used to indicated if a STIS assocation was split
                into separate files.  
    """
    # Define return values
    
    newfilelist = []    # Python list containing strings representing the
                        # names of the files to be processed
                        
    specialparse = False # Boolean flag used to indicated that a STIS
                         # assocation file has been split into multiple
                         # new files

    # First determine if the input file has multiple science extensions.
    # If there are multiple science extensions, then we have an association
    # file.
    
    # Open the input file
    try :
        img = pyfits.open(inputfile)
    except:
        errstr  = "#################################\n"
        errstr += "#                               #\n"
        errstr += "# ERROR: Unable to open file:   #\n"
        errstr += str(inputfile) 
        errstr += "#                               #\n"
        errstr += "#################################\n"
        raise RuntimeError,errstr
        
    # Count the number of sci extensions
    scicount = 0
    for extension in img:
        if extension.header.has_key('extname'):
            if (itemheader['extname'].upper() == 'SCI'):
                scicount += 1
    
    # If there is only 1 science extension, populate the return list with
    # the input file name for return to the calling program
    if (scicount == 1):
        newfilelist.append(inputfile)
    else:
        # We now assume that we have a STIS association file.  We need to
        # split the association file into separate files for each
        # exposure.  A single exposure file will contain a copy of the 
        # primary header and the 'sci','err', and 'dq' extension header
        # and data information.
        specialparse = True
        
        # Determine the number of new files to be created
        numNewFiles = int(scicount / 3)
        
        # We need to build a new rootname for the output file
        rootname = inputfile[0:inputfile.rfind('.')] + '_extn'
        
        # Populate the 'newfilelist' with the names of the files we will
        # create while creating the file on disk
        extnumber = 0
        for count in numNewFiles:
            # Increment the extension counter
            extnumber += 1
            
            # Create the new filename and add it to the return list
            newfilename = rootname + str(extnumber) + ".fits"
            newfilelist.append(newfilename)
            
            # Creat the new file on disk
            
            # Create the fitsobject
            fitsobj = pyfits.HDUList()
            # Copy the primary header
            hdu = img[0].copy()
            fitsobj.append(hdu)
            # Copy the science extension
            hdu = img['sci',extnumber].copy()
            fitsobj.append(hdu)
            # Copy the err extension
            hdu = img['err',extnumber].copy()
            fitsobj.append(hdu)
            # Copy the dq extension
            hdu = img['dq',extnumber].copy()
            fitsobj.append(hdu)
            # Write out the new file
            fitsobj.writeto(newfilename)
            # Clean up
            delete(fitsobj)
            delete(hdu)
        
    # Return the list of file names to be processed
    return newfilelist, specialparse
