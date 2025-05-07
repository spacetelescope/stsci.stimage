#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "immatch/geomap.h"
#include "test.h"

int
main(void)
{
    #define ncoords 64
    coord_t ref[ncoords];
    coord_t input[ncoords];
    bbox_t bbox;
    geomap_output_t output[ncoords];
    size_t noutput = ncoords;
    geomap_result_t result;
    stimage_error_t error;
    size_t i = 0;
    int status = 1;

    stimage_error_init(&error);
    bbox_init(&bbox);
    geomap_result_init(&result);

    /* TEST 1 */
    dbg_print("Test 1\n");
    return 0;

    srand48(0);

    for (i = 0; i < ncoords; ++i) {
        ref[i].x = input[i].x = drand48();
        ref[i].y = input[i].y = drand48();
    }

    status = geomap(
            ncoords, input,
            ncoords, ref,
            &bbox,
            geomap_fit_general,
            surface_type_polynomial,
            2, 2, 2, 2, // xx, yy, xy, yx orders
            xterms_half, xterms_half,
            0, 0, // maxiter, reject
            &noutput, output,
            &result,
            &error);
    dbg_print("End Test 1\n");
    // geomap_result_print(&result);
    geomap_result_free(&result);

    /* TEST 2: SHIFT */
    srand48(0);

    for (i = 0; i < ncoords; ++i) {
        ref[i].x = drand48();
        ref[i].y = drand48();
        input[i].x = ref[i].x + 1.5;
        input[i].y = ref[i].y + 1.25;
    }

    status = geomap(
            ncoords, input,
            ncoords, ref,
            &bbox,
            geomap_fit_shift,
            surface_type_polynomial,
            2, 2, 2, 2, // xx, yy, xy, yx orders
            xterms_none, xterms_none,
            0, 0, // maxiter, reject
            &noutput, output,
            &result,
            &error);
    dbg_print("End Test 2\n");
    // geomap_result_print(&result);
    geomap_result_free(&result);

    /* /\* TEST 3: SCALE *\/ */
    /* srand48(0); */

    /* for (i = 0; i < ncoords; ++i) { */
    /*     ref[i].x = drand48(); */
    /*     ref[i].y = drand48(); */
    /*     input[i].x = ref[i].x * 5.5; */
    /*     input[i].y = ref[i].y * 1.25; */
    /* } */

    /* status = geomap( */
    /*         ncoords, input, */
    /*         ncoords, ref, */
    /*         &bbox, */
    /*         geomap_fit_xyscale, */
    /*         surface_type_polynomial, */
    /*         2, 2, 2, 2, */
    /*         xterms_none, xterms_none, */
    /*         0, 0, */
    /*         &noutput, output, */
    /*         &result, */
    /*         &error); */
    /* geomap_result_print(&result); */
    /* geomap_result_free(&result); */

    return status;
}
#if 0
union dbl_bytes {
    double x;
    uint8_t u8[16];
};

void print_u8_arr(uint8_t * arr, int len) {
    printf("(");
    for (int k=0; k<len; ++k) {
        printf("%02x", arr[k]);
    }
    printf(")");
}

int main(void) {
    double x, nx;
    int ex;
    union dbl_bytes dbl_x, dbl_m;

    dbl_x.x = x;
    nx = frexp(x, &ex);
    dbl_m.x = nx;
    printf("dbl_x = ");
    print_u8_arr(dbl_x.u8, 16);
    printf("\ndbl_m = ");
    print_u8_arr(dbl_m.u8, 16);
    printf("\n");
}
#endif
