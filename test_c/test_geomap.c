#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "immatch/geomap.h"
#include "test.h"

int
main(int argc, char** argv) {
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
            2, 2, 2, 2,
            xterms_half, xterms_half,
            0, 0,
            &noutput, output,
            &result,
            &error);
    geomap_result_print(&result);
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
            2, 2, 2, 2,
            xterms_none, xterms_none,
            0, 0,
            &noutput, output,
            &result,
            &error);
    geomap_result_print(&result);
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
