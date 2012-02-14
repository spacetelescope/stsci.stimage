/*
*****************************************************************************
*
* Name: GC_coords_pkg
*
* Description:
*       This package contains coordinate conversion routines need by
*       generic conversion.
*
* History:
* Date     OPR      Who         Reason
* -------- -------- ----------  ---------------------------------------------
* 07/22/97 34728    WMiller     Initial code
*
*****************************************************************************
*/

#define _USE_MATH_DEFINES	/* needed for MS Windows to define M_PI */
#include <math.h>
#include "gc_opus_private.h"
#include "gc_coords_private.h"

#define DEG_TO_RAD (2.0 * M_PI / 360.0) /* degrees to radians conversion */

/*
*****************************************************************************
*
* Name: GC_radec2glatlon
*
* Prototype:
*       gc_coords_private.h
*
* Description:
*       Converts (right ascension, declination) to Galactic (longitude,
*       latitude), System II assuming J2000 equinox. This routine was
*       rewritten (OPR.52126) to use the most precise calculation based on
*       definitional parameters for the Galactic Coordinate system II found
*       on the Web at http://ledas-cxc.star.le.ac/uk/udocs/PG/html/node76.html.
*       The values obtained from this html file were:
*       Galactic center (0,0) in J2000 RA and DEC is at 
*       (17 45 37.20 -28 56 10.22) which when converted to degrees is
*       (266.4050000 -28.9361722). The Galactic N Pole in J2000 RA and DEC is
*       (12 51 26.28 +27 07 41.70), which when converted to degrees is
*       (192.8595000 27.1282500). These two sets of coordinates can be 
*       converted to orthogonal unit vectors and the crossproduct provides the
*       vector for the third direction. Because of rounding error for the input
*       values the normalized crossproduct and the galactic center vector can be
*       used in another cross-product that gives an accurate orthonormal set of
*       vectors within the full precision of these calculations. This conversion
*       matrix has been calulated and the result is recorded to 14 digits (in 
*       excess of the accuracy of the input data). Any imprecision is due to the
*       input parameters not the conversion matrix. 
*
* Return:
*
* Usage:
*       GC_radec2glatlon (&ra, &dec, &glon, &glat);
*
* History:
* Date     OPR      Who         Reason
* -------- -------- ----------  ---------------------------------------------
* 07/22/97 34728    WMiller     Initial code
* 11/29/04 52126    J.Baum      Rewrite for better precession using matrix
*                               method.
*
*****************************************************************************
*/

void GC_radec2glatlon (
                       double *ra,        /* I: Right Ascension (deg 2000) */
                       double *dec,       /* I: Declination (deg 2000)     */
                       double *glon,      /* O: Galactic longitude (deg)   */
                       double *glat)      /* O: Galactic latitude (deg)    */
{
    double deg2rad = DEG_TO_RAD;
    double ra_rad,dec_rad;
    double cel_vtr[3];
    double gal_vtr[3];
    int i,j;
        
    /* This conversion matrix consists of three orthogonal vectors which point,
    ** respectively, to the galactic center (gal_x), the direction perpendicular
    ** to both the center and the pole (gal_y), and the pole (gal_z).
    */ 
    double cvtmat[3][3] = { 
           { -0.05487548209352,  -0.87343711004849,  -0.48383498866363},
           {  0.49410957812535,  -0.44482953746567,   0.74698220019146},
           { -0.86766606840984,  -0.19807649378008,   0.45598387728295}
                          };

    /* convert input ra and dec to a celestial unit vector */
    ra_rad  = (*ra)  * deg2rad;
    dec_rad = (*dec) * deg2rad;
    cel_vtr[0] = cos(ra_rad) * cos(dec_rad);
    cel_vtr[1] = sin(ra_rad) * cos(dec_rad);
    cel_vtr[2] = sin(dec_rad);

    /* apply the conversion matrix to the celestial vector to get the
    ** galactic vector 
    */
    for ( i = 0; i < 3; i++ ) {
       gal_vtr[i] = 0.0;
       
       for ( j = 0; j < 3; j++) {
          gal_vtr[i] += cvtmat[i][j] * cel_vtr[j];
       }
    }
    /* convert the galactic vector to spherical coordinates */
    *glat = asin ( gal_vtr[2]) / deg2rad;
    *glon = atan2( gal_vtr[1], gal_vtr[0]) / deg2rad;

    if (*glon < 0.0) {
       *glon += 360.0;  /* no negative longitudes */
    }
    return;
}

/*
*****************************************************************************
*
* Name: GC_radec2elatlon
*
* Prototype:
*       gc_coords_private.h
*
* Description:
*       Converts (right ascension, declination) to ecliptic (longitude,
*       latitude) assuming J2000 equinox.
*
*       The calculation depends on only one number, the obliquity of the
*       ecliptic plane. This is a time dependent value. The epoch that is
*       chosen is the year 2000 since all HST position use J2000 positions.
*       The value found in the Astronomical Almanac is 23deg,26'21.448",
*       which has been converted to fractional degrees below.  
*
* Return:
*
* Usage:
*       GC_radec2latlon (&ra, &dec, &elon, &elat);
*
* History:
* Date     OPR      Who         Reason
* -------- -------- ----------  ---------------------------------------------
* 07/22/97 34728    WMiller     Initial code
* 11/30/04 52126    J.Baum      Correct obliquity from 23.439486 to
*                               the Almanac value 23.439291111. Don't
*                               recalculate DEG_TO_RAD, use local value.
*
*****************************************************************************
*/

#define EPSILON		23.439291111      /* obliquity (degrees) for J2000.0 */


void GC_radec2elatlon (
                       double *ra,        /* I: Right Ascension (deg 2000) */
                       double *dec,       /* I: Declination (deg 2000)     */
                       double *elon,      /* O: Ecliptic longitude (deg)   */
                       double *elat)      /* O: Ecliptic latitude (deg)    */
{
   static int first = TRUE;
   static double se, ce;
   double cdec, sdec, sra, cra, selon, celon, selat;
   double deg2rad = DEG_TO_RAD;
   
   if (first) {
      first = FALSE;
      se    = sin (EPSILON * deg2rad);
      ce    = cos (EPSILON * deg2rad);
   }

   /*
    * RA/DEC to ELON/ELAT conversion based on formulae:
    *
    *     cos DEC * cos RA = cos ELAT * cos ELON
    *     cos DEC * sin RA = cos ELAT * sin ELON * cos e - sin ELAT * sin e
    *     sin DEC = cos ELAT * sin ELON * sin e + sin ELAT * cos e
    *     cos ELAT * sin ELON = cos DEC * sin RA * cos e + sin DEC * sin e
    *     sin ELAT = sin DEC * cos e - cos DEC * sin RA * sin e
    */

    sra  = sin (*ra  * deg2rad);
    cra  = cos (*ra  * deg2rad);
    sdec = sin (*dec * deg2rad);
    cdec = cos (*dec * deg2rad);

    selat = sdec * ce - cdec * sra * se;
    *elat = asin (selat) / deg2rad;

    selon = cdec * sra * ce + sdec * se;
    celon = cra * cdec;
    *elon = atan2 (selon, celon) / deg2rad;

    if (*elon < 0.0) *elon += 360.0;

    return;
}

