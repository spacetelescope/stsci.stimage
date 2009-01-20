/*
*******************************************************************************
*
* Name:		OPUS.H
*
* Purpose:	provide typedefs and constants for all OPUS software
*
* History:
* Date	   OPR	    Who       Reason
* -------- -------- --------  -----------------------------------------------
* 03/13/97 33528.1  MSwam     initial version
* 03/14/97 33528.2  MSwam     add stdio.h
* 03/26/97 33528.3  MSwam     if TRUE,FALSE already defined, cancel and
*                             use OPUS definitions
* 10/26/97 35291    Heller    Add OPUS fatal error number
* 05/04/98 36862    MSwam     Adjust fatal error range for UNIX and VMS use
* 06/17/98 37169    MSwam     Add default location and file extension for path
* 01/19/99 38230    MSwam     Remove bool type for CXX 6.0
* 05/21/07 57369    MSwam     Add isBigEndian()
*
*******************************************************************************
*/
#ifndef __OPUS_H
#define __OPUS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

/* Constants ----------------------------------------------------------- */

#undef	FALSE		/* undefine TRUE,FALSE to force use of OPUS versions */
#undef	TRUE
#define	FALSE		(0)
#define	TRUE		(1)

#define	EOS		'\0'

#define OPUS_PATHNAME_DEFAULT_LOCATION  "OPUS_DEFINITIONS_DIR:"
#define OPUS_PATHNAME_DEFAULT_EXTENSION ".path"

/* OPUS errors      ---------------------------------------------------- */
/*	Error numbers 100-127 are reserved as OPUS fatal errors.
*       When returned by a process to XPOLL, XPOLL will immediately exit.
*       Users should stick to odd numbers under VMS to avoid the
*       operating system spitting out an error message that has nothing
*       to do with the real error (e.g. exit 104 will report "device already
*       mounted").
*/
#define FATAL_ERROR_START_RANGE	100
#define FATAL_ERROR_END_RANGE	127
#define FATAL_DB_ERROR	101

/* Data Definitions ---------------------------------------------------- */

#ifdef ITS_VAX_VMS
typedef int     bool;           /* Boolean type for use with TRUE,FALSE */
#endif 

/* Macros -------------------------------------------------------------- */

#define	OUT_OF_MEMORY(p)	if (p == NULL) { \
				  fprintf(stderr,"OPUS: Out of memory\n"); \
				  exit(EXIT_FAILURE); }

/* DON'T PUT ANYTHING BELOW THIS ENDIF!!! */
#ifdef __cplusplus
}
#endif
#endif
