#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "immatch/geomap.h"
#include "test.h"

#define INITIALIZED 0xbadbeef

// ----------------------------------------------------------------------
struct coord_list {
    int initialized;
    coord_t * coords;
    int alloc;
    int used;
};

void print_coord_list(struct coord_list * coords, const char * label, int ret)
{
    int k=0;

    if (!coords || INITIALIZED!=coords->initialized) {
        return;
    }

    if (label) {
        printf("%s = \n", label);
    }

    printf("    .alloc = %d\n", coords->alloc);
    printf("    .used  = %d\n", coords->used);

    printf("    .coords = {\n        (%"DBL", %"DBL"), ", coords->coords[k].x, coords->coords[k].y);
    for (k=1; k<coords->used; ++k) {
        if (0==(k%5)) {
            printf("\n        ");
        }
        printf("(%"DBL", %"DBL"), ", coords->coords[k].x, coords->coords[k].y);
    }
    printf("\n    }");
    ret = (ret < 1) ? 1 : ret;
    ret = (ret > 2) ? 2 : ret;
    while (ret > 0) {
        printf("\n");
        ret--;
    }
}
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
const int ncoords = 20;
const int nbase = 15;
const int nextra = 5;

void base_coords(struct coord_list * coords)
{
    int k;

    double x[nbase] = { -303.84609, -246.00600,  420.31010, -407.02170,   47.92501,
                         449.99497,  317.32335,  -78.08896,  345.70714,  150.20207,
                         458.55107, -343.01210, -306.99155, -145.12925,  183.15312};

    double y[nbase] = { -305.98824, -464.58525, -146.35067,  185.79842,  253.11053,
                         300.24990, -262.25222,  235.43031,  320.96704, -139.03478,
                         233.38413, -316.93320,  202.85420, -252.86989,  382.40623};

    for (k=0; k<nbase; ++k)
    {
        coords->coords[k].x = x[k];
        coords->coords[k].y = y[k];
    }
}
void get_input(struct coord_list * coords)
{
    double x[nextra] = {  426.96851, -388.88402,  108.22633,  170.85110,  435.57264};
    double y[nextra] = { -423.38909, -447.35147,  255.00272,  -47.67573,  492.17773};
    int k;

    // Same fifteen coordinates
    base_coords(coords);

    // Random five different pixels.
    for (k=0; k<nextra; ++k)
    {
        coords->coords[nbase+k].x = x[k];
        coords->coords[nbase+k].y = y[k];
    }
    coords->used = ncoords;

    return;
}

void get_reference(struct coord_list * coords)
{
    double x[nextra] = {  166.09336, -137.12012,  131.59430,  469.28326,  375.83601};
    double y[nextra] = {  219.05374,  269.97358, -432.54307,  -53.96007,  341.31046};
    int k;

    // Same fifteen coordinates
    base_coords(coords);

    // Random five different pixels.
    for (k=0; k<nextra; ++k)
    {
        coords->coords[nbase+k].x = x[k];
        coords->coords[nbase+k].y = y[k];
    }
    coords->used = ncoords;

    return;
}

// Computes radians from degrees.
static inline double deg_to_rad(double deg)
{
    return deg * M_PI / 180.0;
}

// Computes a rotation matrix for a given input of degrees.
void rotation_matrix(coord_t * coord, double deg)
{
    double rad = deg_to_rad(deg);

    coord->x = cos(rad);
    coord->y = sin(rad);

    return;
}

// Rotates a set of coordinates given a ratation matrix.
void coords_rotate(struct coord_list * coords, coord_t * rotate)
{
    int k;
    double x, y;

    /*
     *  The rotation matrix by t radians is:
     *
     *        R = [cos(t)  -sin(t)]
     *            [sin[t)   cos(t)]
     *
     *       rotate.x = cos(t)
     *       rotate.y = sin(t)
     */
    for (k=0; k<coords->used; ++k)
    {
        x = coords->coords[k].x * rotate->x - coords->coords[k].y * rotate->y;
        y = coords->coords[k].x * rotate->y + coords->coords[k].y * rotate->x;
        coords->coords[k].x = x;
        coords->coords[k].y = y;
    }

    return;
}

// Translate a set of points by the translate point.
void coords_translate(struct coord_list * coords, coord_t * translate)
{
    int k;

    for (k=0; k<coords->used; ++k)
    {
        coords->coords[k].x += translate->x;
        coords->coords[k].y += translate->y;
    }

    return;
}

// Magnfiy a set of points by magnification.
void coords_magnify(struct coord_list * coords, double magnify)
{
    int k;

    for (k=0; k<coords->used; ++k)
    {
        coords->coords[k].x *= magnify;
        coords->coords[k].y *= magnify;
    }

    return;
}

// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Tests
#if 0
const int ncoords = 20;
const int nbase = 15;
const int nextra = 5;

struct coord_list {
    int initialized;
    coord_t * coords;
    int alloc;
    int used;
};
#endif

int
check_coord(coord_t result, coord_t check, double tol, const char * label)
{
    double xd, yd;

    xd = fabs(result.x - check.x);
    yd = fabs(result.y - check.y);
    if ((xd>tol) || (yd>tol))
    {
        printf("Fail: %s\n", label);
        return 1;
    }
    return 0;
}

int
check_integer(int result, int check, const char * label)
{
    if (result != check)
    {
        printf("Fail: %s\n", label);
        return 1;
    }
    return 0;
}

void
result_check(geomap_result_t * result, geomap_result_t * check, const char * tname)
{
    double tol = 1.0e-4;
    int fail = 0;

    printf("    Test: %s\n", tname);

    if (result->fit_geometry != check->fit_geometry)
    {
        printf("Fail: Fit Geometry\n");
        fail++;
    }
    if (result->function != check->function)
    {
        printf("Fail: Function\n");
        fail++;
    }

    fail += check_coord(result->rms, check->rms, tol, "Root Mean Square");
    fail += check_coord(result->mean_input, check->mean_input, tol, "Mean Input");
    fail += check_coord(result->mean_ref, check->mean_ref, tol, "Mean Reference");
    fail += check_coord(result->shift, check->shift, tol, "Shift");
    fail += check_coord(result->mag, check->mag, tol, "Magnitude");
    fail += check_coord(result->rotation, check->rotation, tol, "rotation");
    fail += check_integer(result->nxcoeff, check->nxcoeff, "nxcoeff");
    fail += check_integer(result->nycoeff, check->nycoeff, "nycoeff");

    if (0==fail)
    {
        printf(".... Pass\n");
    } else {
        printf(".... Failed %d checks\n", fail);
    }
    printf("\n");
}

int base_test(
        const char * test_name,
        struct coord_list * in,             /* Input list */
        struct coord_list * ref,            /* Reference list */
        const geomap_fit_e fit_geometry,    /* Choice of geometry for fitting */
        const surface_type_e function,      /* type of surface to use */
        size_t * noutput,                   /* The number of output records */
        geomap_output_t* const output,      /* Array of output records */
        geomap_result_t* const result,      /* The fit found */
        stimage_error_t* const error)       /* Error structure */
{
    bbox_t bbox;
    int status = 1;

    bbox_init(&bbox);

    status = geomap(
            in->used, in->coords,
            ref->used, ref->coords,
            &bbox,
            fit_geometry,
            function,
            2, 2, 2, 2, // xx, yy, xy, yx orders
            xterms_half, xterms_half,
            0, 0, // maxiter, reject
            noutput, output,
            result,
            error);

    printf("Test: %s - Return status = %d\n", test_name, status);

    return 0;
}

void
same_result(geomap_result_t * result)
{
    memset(result, 0, sizeof(*result));

    result->fit_geometry = geomap_fit_general;
    result->function = surface_type_polynomial;

    result->rms.x = 645.265145;
    result->rms.y = 859.820076;

    result->mean_input.x = 64.790287;
    result->mean_input.y = 2.747533;

    result->mean_ref.x = 64.790287;
    result->mean_ref.y = 2.747533;

    result->shift.x = -18641051.962848;
    result->shift.y = -8745863.542405;

    result->mag.x = 3.132711;
    result->mag.y = 0.162293;

    result->rotation.x = 135.000000;
    result->rotation.y = 37.222253;

    result->nxcoeff = 3;
    result->nycoeff = 3;
}

int test_same(void)
{
    coord_t in_arr[ncoords] = {0};
    coord_t ref_arr[ncoords] = {0};
    struct coord_list in = {INITIALIZED, in_arr, ncoords, 0};
    struct coord_list ref = {INITIALIZED, ref_arr, ncoords, 0};
    geomap_fit_e fgeom = geomap_fit_general;
    surface_type_e stype = surface_type_polynomial;
    size_t noutput;
    geomap_output_t output[ncoords];
    geomap_result_t result;
    geomap_result_t check;
    stimage_error_t error;

    stimage_error_init(&error);
    geomap_result_init(&result);
    geomap_result_init(&check);
    same_result(&check);

    get_input(&in);
    get_input(&ref);

    base_test(__FUNCTION__, &in, &ref, fgeom, stype, &noutput, output, &result, &error);
    result_check(&result, &check, __FUNCTION__);

    geomap_result_free(&result);

    return 0;
}

void
rotation_30_result(geomap_result_t * result)
{
    memset(result, 0, sizeof(*result));

    result->fit_geometry = geomap_fit_rotate;
    result->function = surface_type_polynomial;

    result->rms.x = 386.205840;
    result->rms.y = 336.900542;

    result->mean_input.x = 64.790287;
    result->mean_input.y = 2.747533;

    result->mean_ref.x = 54.736268;
    result->mean_ref.y = 34.774577;

    result->shift.x = 71.040012;
    result->shift.y = -60.235758;

    result->mag.x = 0.866809;
    result->mag.y = 1.000000;

    result->rotation.x = 135.000000;
    result->rotation.y = 52.198604;

    result->nxcoeff = 3;
    result->nycoeff = 3;
}

void
rotation_45_result(geomap_result_t * result)
{
    memset(result, 0, sizeof(*result));

    result->fit_geometry = geomap_fit_rotate;
    result->function = surface_type_polynomial;

    result->rms.x = 356.934127;
    result->rms.y = 363.123531;

    result->mean_input.x = 64.790287;
    result->mean_input.y = 2.747533;

    result->mean_ref.x = 43.870852;
    result->mean_ref.y = 47.756451;

    result->shift.x = 60.953716;
    result->shift.y = -60.547634;

    result->mag.x = 0.982662;
    result->mag.y = 1.000000;

    result->rotation.x = 135.000000;
    result->rotation.y = 45.984971;

    result->nxcoeff = 3;
    result->nycoeff = 3;
}

void
rotation_90_result(geomap_result_t * result)
{
    memset(result, 0, sizeof(*result));

    result->fit_geometry = geomap_fit_rotate;
    result->function = surface_type_polynomial;

    result->rms.x = 49.354636;
    result->rms.y = 48.276783;

    result->mean_input.x = 64.790287;
    result->mean_input.y = 2.747533;

    result->mean_ref.x = -2.747534;
    result->mean_ref.y = 64.790287;

    result->shift.x = 2.323053;
    result->shift.y = -9.874069;

    result->mag.x = 0.216232;
    result->mag.y = 1.000000;

    result->rotation.x = 315.000000;
    result->rotation.y = 81.205030;

    result->nxcoeff = 3;
    result->nycoeff = 3;
}

int
rotation_result(int deg, geomap_result_t * result)
{
    switch(deg)
    {
        case 30:
            rotation_30_result(result);
            break;
        case 45:
            rotation_45_result(result);
            break;
        case 90:
            rotation_90_result(result);
            break;
        default:
            printf("Error: Invalid angle for rotation test\n");
            return 1;
            break;
    }
    return 0;
}

int base_rotation_deg(int ideg)
{
    coord_t in_arr[ncoords] = {0};
    coord_t ref_arr[ncoords] = {0};
    struct coord_list in = {INITIALIZED, in_arr, ncoords, 0};
    struct coord_list ref = {INITIALIZED, ref_arr, ncoords, 0};
    geomap_fit_e fgeom = geomap_fit_rotate;  // rotate
    surface_type_e stype = surface_type_polynomial;
    size_t noutput;
    geomap_output_t output[ncoords];
    geomap_result_t result;
    geomap_result_t check;
    stimage_error_t error;
    coord_t rotate;
    double deg = (double)ideg;

    print_delim('-', 40, 1);
    printf("TEST: %s (Degree: %.2f)\n", __FUNCTION__, deg);
    print_delim('-', 40, 1);

    stimage_error_init(&error);
    geomap_result_init(&result);
    geomap_result_init(&check);
    rotation_result(ideg, &check);


    get_input(&in);
    get_input(&ref);

    // Rotate 
    rotation_matrix(&rotate, deg);
    coords_rotate(&ref, &rotate);

    base_test(__FUNCTION__, &in, &ref, fgeom, stype, &noutput, output, &result, &error);
    result_check(&result, &check, __FUNCTION__);

    // geomap_result_print(&result);
    geomap_result_free(&result);

    return 0;
}

int test_rotation(void)
{
    base_rotation_deg(30);
    base_rotation_deg(45);
    base_rotation_deg(90);

    return 0;
}

void
translate_result(geomap_result_t * result)
{
    memset(result, 0, sizeof(*result));

    result->fit_geometry = geomap_fit_shift;
    result->function = surface_type_polynomial;

    result->rms.x = 5.129892;
    result->rms.y = 9.233805;

    result->mean_input.x = 64.790287;
    result->mean_input.y = 2.747533;

    result->mean_ref.x = 69.790287;
    result->mean_ref.y = 11.747533;

    result->shift.x = -8.000000;
    result->shift.y = -3.000000;

    result->mag.x = 0.000000;
    result->mag.y = 1.000000;


    result->rotation.x = 0.000000;
    result->rotation.y = 90.000000;

    result->nxcoeff = 3;
    result->nycoeff = 3;
}

int test_translation(void)
{
    coord_t in_arr[ncoords] = {0};
    coord_t ref_arr[ncoords] = {0};
    struct coord_list in = {INITIALIZED, in_arr, ncoords, 0};
    struct coord_list ref = {INITIALIZED, ref_arr, ncoords, 0};
    geomap_fit_e fgeom = geomap_fit_shift; // Translate
    surface_type_e stype = surface_type_polynomial;
    size_t noutput;
    geomap_output_t output[ncoords];
    geomap_result_t result;
    geomap_result_t check;
    stimage_error_t error;
    coord_t translate = {.x=5.0, .y=9.0};

    stimage_error_init(&error);
    geomap_result_init(&result);
    geomap_result_init(&check);
    translate_result(&check);

    get_input(&in);
    get_input(&ref);

    coords_translate(&ref, &translate);

    base_test(__FUNCTION__, &in, &ref, fgeom, stype, &noutput, output, &result, &error);
    result_check(&result, &check, __FUNCTION__);
    //geomap_result_print(&result);
    geomap_result_free(&result);

    return 0;
    return 0;
}

int test_mag_fit_params(geomap_fit_e fgeom, surface_type_e stype, geomap_result_t * check, const char * label)
{
    coord_t in_arr[ncoords] = {0};
    coord_t ref_arr[ncoords] = {0};
    struct coord_list in = {INITIALIZED, in_arr, ncoords, 0};
    struct coord_list ref = {INITIALIZED, ref_arr, ncoords, 0};

    size_t noutput;
    geomap_output_t output[ncoords];
    geomap_result_t result;
    stimage_error_t error;
    double magnify = 5.0;

    stimage_error_init(&error);
    geomap_result_init(&result);

    get_input(&in);
    get_input(&ref);

    coords_magnify(&ref, magnify);

    print_delim('-', 40, 1);
    printf("TEST: %s - type: %s\n", __FUNCTION__, label);
    print_delim('-', 40, 1);

    base_test(__FUNCTION__, &in, &ref, fgeom, stype, &noutput, output, &result, &error);
    result_check(&result, check, __FUNCTION__);
    // geomap_result_print(&result);
    geomap_result_free(&result);

    return 0;
}

void
mag_poly_result(geomap_result_t * result)
{
    memset(result, 0, sizeof(*result));

    result->fit_geometry = geomap_fit_rscale;
    result->function = surface_type_polynomial;

    result->rms.x = 0.000000;
    result->rms.y = 0.000000;

    result->mean_input.x = 64.790287;
    result->mean_input.y = 2.747533;

    result->mean_ref.x = 323.951435;
    result->mean_ref.y = 13.737667;

    result->shift.x = 0.200000;
    result->shift.y = -0.000000;

    result->mag.x = 0.282843;
    result->mag.y = 0.200000;

    result->rotation.x = 315.000000;
    result->rotation.y = 360.000000;

    result->nxcoeff = 3;
    result->nycoeff = 3;
}

void
mag_legendre_result(geomap_result_t * result)
{
    memset(result, 0, sizeof(*result));


    result->fit_geometry = geomap_fit_rscale;
    result->function = surface_type_legendre;

    result->rms.x = 700111.549164;
    result->rms.y = 748496.099203;

    result->mean_input.x = 64.790287;
    result->mean_input.y = 2.747533;

    result->mean_ref.x = 323.951435;
    result->mean_ref.y = 13.737667;

    result->shift.x = -407.121700;
    result->shift.y = 13.796240;

    result->mag.x = 0.282843;
    result->mag.y = 0.200000;

    result->rotation.x = 315.000000;
    result->rotation.y = 360.000000;

    result->nxcoeff = 3;
    result->nycoeff = 3;
}

void
mag_chebyshev_result(geomap_result_t * result)
{
    memset(result, 0, sizeof(*result));

    result->fit_geometry = geomap_fit_rscale;
    result->function = surface_type_chebyshev;

    result->rms.x = 700111.549164;
    result->rms.y = 748496.099203;

    result->mean_input.x = 64.790287;
    result->mean_input.y = 2.747533;

    result->mean_ref.x = 323.951435;
    result->mean_ref.y = 13.737667;

    result->shift.x = -407.121700;
    result->shift.y = 13.796240;

    result->mag.x = 0.282843;
    result->mag.y = 0.200000;

    result->rotation.x = 315.000000;
    result->rotation.y = 360.000000;

    result->nxcoeff = 3;
    result->nycoeff = 3;
}

int test_magnification(void)
{
    geomap_fit_e fgeom = geomap_fit_rscale;
    geomap_result_t check;

    print_delim('-', 30, 1);
    mag_poly_result(&check);
    test_mag_fit_params(fgeom, surface_type_polynomial, &check, "poly");
    print_delim('-', 30, 1);

    mag_legendre_result(&check);
    test_mag_fit_params(fgeom, surface_type_legendre, &check, "legendre");
    print_delim('-', 30, 1);

    mag_chebyshev_result(&check);
    test_mag_fit_params(fgeom, surface_type_chebyshev, &check, "chebyshev");
    print_delim('-', 30, 1);

    return 0;
}

// ----------------------------------------------------------------------

typedef enum {
    same_test=0,
    rotation_test=1,
    translate_test=2,
    magnification_test=3,
    all_tests=4
} test_t;

test_t which_tests(const char * in)
{
    int k;
    const int tlen = 5;
    char * test_types[tlen] = {"same", "rotate", "shift", "mag", "all"};

    for (k=0; k<tlen; ++k)
    {
        if (0==strncmp(in, test_types[k], strlen(test_types[k])))
        {
            return k;
        }
    }

    return -1;
}

// ----------------------------------------------------------------------
// MAIN
int main(int argc, char * argv[])
{
    test_t tests = all_tests;
    if (argc > 1)
    {
        tests = which_tests(argv[1]);
    }

    PRINT_DELIM;
    if (same_test==tests || all_tests==tests)
    {
        test_same();
        PRINT_DELIM;
    } 

    if (rotation_test==tests || all_tests==tests)
    {
        test_rotation();
        PRINT_DELIM;
    } 

    if (translate_test==tests || all_tests==tests)
    {
        test_translation();
        PRINT_DELIM;
    } 

    if (magnification_test==tests || all_tests==tests)
    {
        test_magnification();
        PRINT_DELIM;
    }

    if (tests<same_test || tests>all_tests)
    {
        printf("Invalid geomap testing\n");
        PRINT_DELIM;
    }

    return 0;
}
