/*
*******************************************************************************
*
* Name: GC_private.h
*
* Description:
*       These are the constants and structures which are used by the Data
*	Validation (GC) package.
*
* History:
* Date     OPR      Who       Reason
* -------- -------- --------  -----------------------------------------------
* 08/21/95 28179    Rose      Initial version
* 04/04/97 33528    MSwam     prototype for GC_parse_targ, GC_set_version
* 04/15/97 33742    WMiller   Change prototype for GC_exptime/add UT_CLOCK
* 07/24/97 34728    WMiller   Add coordinate conversion prototypes
* 11/17/97 26710    MSwam     New syntax for GC_clockcoeff
* 11/24/97 26710.1  MSwam     Add GC_upd_file_times,GC_read_pkt_time
* 12/16/97 35746    MSwam     Add tacq_mode to GC_STRUCT for target acqs
* 12/29/97 35746.1  MSwam     Resolve variant for 9.0
* 02/05/98 36241    MSwam     Add archclass to gc_upd_file_times
* 07/06/98 36865    Heller    GC port to Alpha
* 08/10/98 29818    MSwam     Change clock_ct to type double,remove gc_exptime
* 11/10/98 37344.2  Rose      TLM_PATTERN is obsolete
* 01/10/99 37623    Heller    Additions for FOC fits port
* 02/26/99 31648    Heller    Set fill keywords
* 05/21/99 36235    MSwam     Add rule classes to GC_init()
* 01/31/01 43165    MSwam     Add GC_keyword_repairs
* 11/06/01 44435    Rose      Add constants & generic functions 
* 12/17/01 44675    MSwam     Add GC_cal_tables()
* 09/26/02 46503    J.Baum    Add GC_vel_aberr_scale, etc, and new include file
*                                for velocity aberration scale factor 
* 12/14/04 52075    MSwam     Add GC_datamax_bzero prototype
* 12/14/05 53808    Heller    Add GC_sgstar
* 04/20/06 55102    MSwam     Add GC_pdq_summary
* 06/10/06 53948    Heller    HST clock rollover - Add year value for
*                               trans_time window.
* 05/21/07 57570    MSwam       Address GCC 411 warnings
* 05/30/08 59548    Sherbert  Remove gc_construct_edd.c prototype
* 10/13/08 61008    Heller    Clock resets can happen < 7 years apart
* 11/17/08 61211    MSwam     back out 61008
*
******************************************************************************
*/
#ifndef __GC_PRIVATE_H_LOADED
#define __GC_PRIVATE_H_LOADED 1
#ifdef __cplusplus
extern "C" {
#endif

/* Return Statuses ----------------------------------------------------- */

#define         GC_OK           1
#define         GC_ERROR        -1

/* Constants ----------------------------------------------------------- */

#define         GC_VALID        0x4A696D52      /* valid bit pattern    */

#define		GC_SPT		15	/* support FITS file		*/
#define		GC_RAW		16	/* raw FITS data file		*/
#define		GC_TTAG		17      /* timetag FITS data file	*/

#define		GC_SHH		9	/* primary SPT FITS header	*/
#define		GC_UDL		10	/* SPT image extension headers  */

#define 	GC_SHP_FILE	1
#define 	GC_UDL_FILE	2
#define 	GC_PKX_FILE	3
#define 	GC_DQX_FILE	4

#define		GC_MAX_GROUP	100
#define		GC_BLANKS	"                    "

#define GC_ACCUM        10      /* accumulation mode            */
#define GC_ACQ          30      /* target acquisition mode      */
#define GC_PEAKUP       40      /* target acq/peakup mode       */
#define GC_LRC          60      /* local rate check image       */
#define GC_ENG_DIAG     91      /* Engineering diagnostic       */
#define GC_MEM_DUMP     92      /* Memory dump                  */

#define GC_TRANS_TIME_WINDOW  7  /* Window for veh_time_coeff.trans_time*/

/* Data Definitions ---------------------------------------------------- */

#include "gc_opus_private.h"


void GC_radec2elatlon (
      double *ra,        /* I: Right Ascension (deg 2000) */
      double *dec,       /* I: Declination (deg 2000)     */
      double *elon,      /* O: Ecliptic longitude (deg)   */
      double *elat);     /* O: Ecliptic latitude (deg)    */

void GC_radec2glatlon (
      double *ra,        /* I: Right Ascension (deg 2000) */
      double *dec,       /* I: Declination (deg 2000)     */
      double *glon,      /* O: Galactic longitude (deg)   */
      double *glat);     /* O: Galactic latitude (deg)    */


#ifdef __cplusplus
}
#endif
#endif
