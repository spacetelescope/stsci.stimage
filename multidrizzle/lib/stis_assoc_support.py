#  Program: stis_assoc_support.py
#  Author:  Christopher Hanley
#  History:
#   Version 0.1.0, 12/06/2004: Initial Creation -- CJH
#   Version 0.2.0, 12/09/2004: Added method for supporting IVM file for
#       association data.  -- CJH
#   Version 0.3.0, 01/03/2004 -- Added check for zero exposure time input
#       files.  Those with exposure times of zero not included in the
#       newfilelist list. -- CJH
#   Version 0.4.0, 01/26/05 -- Modifed the EXPTIME zero check to be less than
#       or equal to zero.

__version__ = '0.4 (01/26/2005)'
__author__  = 'Christopher Hanley'


# import external modules
import pyfits
import os

def parseSTIS(inputfile):
    """
    FUNCTION:   parseSTIS
    PURPOSE :   the parse STIS function is used to convert STIS assocation files into
                single exposure STIS files.  The primary header of the original assocation
                will populate the primary header of the newly created file.  The first extension
                of the new file will contain a copy of the individual header, sci, err, dq, 
                extension of input assocation.  A STIS assocation of N individual exposures will
                be used to create N individual files.

    INPUT   :   inputfile - a Python list object populated by strings.  Each string represents 
                the name of a file on disk.

    OUTPUT  :   newfilelist - a Python list object populated by strings.  Each string represents 
                the name of a newly created STIS single exposure file on disk.
                
                excludedlist - a Python list object populated by strings.  Each string represents
                the name of a STIS exposure that has been excluded from processing because the
                EXPTIME header keyword value is zero.
                
                specialparse - a Boolean value used to indicated if a STIS assocation was split
                into separate files.  
    """
    # Define return values
    
    newfilelist = []    # Python list containing strings representing the
                        # names of the files to be processed
                        
    excludedList = []   # Python list containing strings representing the
                        # names of the files that have been excluded from
                        # processing because they have an EXPTIME
                        # keyword value of zero.
                        
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
        errstr += "      " + str(inputfile) + "\n" 
        errstr += "#                               #\n"
        errstr += "#################################\n"
        raise RuntimeError,errstr
        
    # Count the number of sci extensions
    scicount = 0
    for extension in img:
        if extension.header.has_key('extname'):
            if (extension.header['extname'].upper() == 'SCI'):
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
                
        # We need to build a new rootname for the output file
        rootname = inputfile[0:inputfile.rfind('.')] + '_extn'
        
        # Populate the 'newfilelist' with the names of the files we will
        # create while creating the file on disk
        
        for count in range(1,scicount+1):
            # Create the new filename and add it to the return list
            newfilename = rootname + str(count) + ".fits"
            
            # Creat the new file on disk
            
            # Create the fitsobject
            fitsobj = pyfits.HDUList()
            # Copy the primary header
            hdu = img[0].copy()
            fitsobj.append(hdu)
            
            # Modify the 'NEXTEND' keyword of the primary header to 3 for the 
            #'sci, err, and dq' extensions of the newly created file.  Programs
            # such as MAKEWCS look at this keyword.
            fitsobj[0].header['NEXTEND'] = 3
            
            # Copy the science extension
            hdu = img['sci',count].copy()
            fitsobj.append(hdu)
            try:
                # Verify error array exists
                if img['err',count].data == None:
                    raise ValueError
                # Verify dq array exists
                if img['dq',count].data == None:
                    raise ValueError
                # Copy the err extension
                hdu = img['err',count].copy()
                fitsobj.append(hdu)
                # Copy the dq extension
                hdu = img['dq',count].copy()
                fitsobj.append(hdu)
            except:
                errorstr =  "###############################\n"
                errorstr += "#                             #\n"
                errorstr += "# ERROR:                      #\n"
                errorstr += "#  The input image:           #\n"
                errorstr += "      " + str(inputfile) +"\n"
                errorstr += "#  does not contain required  #\n"
                errorstr += "#  image extensions.  Each    #\n"
                errorstr += "#  must contain populated sci,#\n"
                errorstr += "#  dq, and err arrays.        #\n"
                errorstr += "#                             #\n"
                errorstr += "###############################\n"
                raise ValueError, errorstr

            # Update the 'EXTNER' keyword to indicate the new extnesion number
            # for the single exposure files.
            fitsobj[1].header['EXTVER'] = 1
            fitsobj[2].header['EXTVER'] = 1
            fitsobj[3].header['EXTVER'] = 1
                
            # Determine if the file you wish to create already exists on the disk.
            # If the file does exist, replace it.
            dirfiles = os.listdir(os.curdir)
            if (dirfiles.count(newfilename) > 0):
                os.remove(newfilename)
                print "       Replacing "+newfilename+"..."
    
            # Write out the new file
            fitsobj.writeto(newfilename)

            # Determine if the newly created extension image will be included in the
            # list of Multidrizzle images to process.
            #
            # If the EXPTIME value for the image is zero, add it to the exclusion list.
            # Otwerwise the image should be included for processing by Multidrizzle.
            if (fitsobj[1].header['EXPTIME'] <= 0.0):
                excludedList.append(newfilename)
            else:
                newfilelist.append(newfilename)

            # Clean up
            del(fitsobj)
            del(hdu)
        
    # Return the list of file names to be processed
    return newfilelist, excludedList, specialparse


def parseSTISIVM(inputfile):
    """
    FUNCTION:   parseSTISIVM
    PURPOSE :   the parseSTISIVM function is used to convert STIS IVM assocation files into
                single STIS IVM files.  The primary header of the original assocation
                will populate the primary header of the newly created file.  The first extension
                of the new file will contain a copy of the individual header and IVM  
                extension of input assocation.  A STIS assocation of N individual exposures will
                be used to create N individual files.
    INPUT   :   a Python list object populated by strings.  Each string represents the name of a
                file on disk.
    OUTPUT  :   newfilelist - a Python list object populated by strings.  Each string represents 
                the name of a newly created STIS single IVM file on disk.
                
    """
    # Define return values
    
    newfilelist = []    # Python list containing strings representing the
                        # names of the files to be processed
                        

    # First determine if the input file has multiple IVM extensions.
    # If there are multiple science extensions, then we have an association
    # file.
    
    # Open the input file
    try :
        img = pyfits.open(inputfile)
    except:
        errstr  = "#################################\n"
        errstr += "#                               #\n"
        errstr += "# ERROR: Unable to open file:   #\n"
        errstr += "       " + str(inputfile) + '\n' 
        errstr += "#                               #\n"
        errstr += "#################################\n"
        raise RuntimeError,errstr
        
    # Count the number of sci extensions
    ivmcount = 0
    for extension in img:
        if extension.header.has_key('extname'):
            if (extension.header['extname'].upper() == 'IVM'):
                ivmcount += 1
    
    # If there is only 1 science extension, populate the return list with
    # the input file name for return to the calling program
    if (ivmcount == 1):
        newfilelist.append(inputfile)
    # Check to verify that there are actual IVM arrays to apply
    elif (ivmcount == 0):
        errorstr =  "########################################\n"
        errorstr += "# Error:                               #\n"
        errorstr += "#  No valid 'IVM' extensions found in: #\n"
        errorstr += "       " + str(inputfile) + '\n'
        errorstr += "#                                      #\n"
        errorstr += "########################################\n"
        raise ValueError, errorstr
        
    else:
        # We now assume that we have a STIS association file.  We need to
        # split the association file into separate files for each
        # IVM extension.  A single exposure file will contain a copy of the 
        # primary header and the 'IVM' data and extension header.
                
        # We need to build a new rootname for the output file
        rootname = inputfile[0:inputfile.rfind('.')] + '_extn'
        
        # Populate the 'newfilelist' with the names of the files we will
        # create while creating the file on disk
        for count in range(1,ivmcount+1):
            # Create the new filename and add it to the return list
            newfilename = rootname + str(count) + ".fits"
            newfilelist.append(newfilename)
            
            # Creat the new file on disk
            
            # Create the fitsobject
            fitsobj = pyfits.HDUList()
            # Copy the primary header
            hdu = img[0].copy()
            fitsobj.append(hdu)

            # Modify the 'NEXTEND' keyword of the primary header to 1 for the 
            #'IVM' extension of the newly created file.
            fitsobj[0].header['NEXTEND'] = 1

            # Copy the IVM extension
            hdu = img['IVM',count].copy()
            fitsobj.append(hdu)

            # Update the 'EXTNER' keyword to indicate the new extnesion number
            # for the single exposure files.
            fitsobj[1].header['EXTVER'] = 1
            fitsobj[2].header['EXTVER'] = 1
            fitsobj[3].header['EXTVER'] = 1

            # Write out the new file
            fitsobj.writeto(newfilename)
            # Clean up
            del(fitsobj)
            del(hdu)
        
    # Return the list of file names to be processed
    return newfilelist
