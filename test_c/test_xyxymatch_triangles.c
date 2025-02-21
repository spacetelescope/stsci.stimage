#include <stdio.h>
#include <stdlib.h>

#include "immatch/xyxymatch.h"
#include "lib/lintransform.h"
#include "test.h"

int compare(const size_t ncoords,
            const coord_t* const ref,
            const coord_t* const input,
            xyxymatch_output_t* output) {
    int status;
    const coord_t origin = {0.0, 0.0};
    const coord_t mag = {1.0, 1.0};
    const coord_t rot = {0.0, 0.0};
    const coord_t ref_origin = {0.0, 0.0};
    const double tolerance = 0.0001;
    const double max_ratio = 10.0;
    const size_t max_points = 40;
    const size_t nreject = 10;
    size_t noutput = ncoords;
    stimage_error_t error;
    size_t i = 0;

    stimage_error_init(&error);

    status = xyxymatch(
            ncoords, input,
            ncoords, ref,
            &noutput, output,
            &origin, &mag, &rot, &ref_origin,
            xyxymatch_algo_triangles,
            tolerance, 0.0, max_points, max_ratio, nreject,
            &error);

    if (status) {
        printf(stimage_error_get_message(&error));
        return status;
    }

    if (noutput != max_points) {
        printf("Expected %lu pairs, got %lu\n",
               (unsigned long)max_points,
               (unsigned long)noutput);
    }

    for (i = 0; i < noutput; ++i) {
        if (output[i].coord_idx != output[i].ref_idx) {
            printf("Mismatched indicies\n");
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    #define ncoords 4098
    coord_t ref[ncoords];
    coord_t input[ncoords];
    xyxymatch_output_t output[ncoords];
    lintransform_t trans;
    coord_t in = {0.0, 0.0};
    coord_t mag = {1.002, 1.003};
    coord_t rot = {2.0, 2.0};
    coord_t out = {1.0, 3.0};

    size_t i = 0;

    srand48(0);

    for (i = 0; i < ncoords; ++i) {
        ref[i].x = drand48() - 0.5;
        ref[i].y = drand48() - 0.5;
    }

    /* Test identity transform */
    printf("Identity\n");

    for (i = 0; i < ncoords; ++i) {
        input[i].x = ref[i].x;
        input[i].y = ref[i].y;
    }

    if (compare(ncoords, ref, input, output)) {
        return 1;
    }

    /* Test translation */
    printf("Translation\n");

    for (i = 0; i < ncoords; ++i) {
        input[i].x = ref[i].x + 24;
        input[i].y = ref[i].y + 42;
    }

    if (compare(ncoords, ref, input, output)) {
        return 1;
    }

    /* Test scale + translation */
    printf("Scale and Translation\n");

    for (i = 0; i < ncoords; ++i) {
        input[i].x = ref[i].x * 1.002 + 24;
        input[i].y = ref[i].y * 1.003 + 42;
    }

    if (compare(ncoords, ref, input, output)) {
        return 1;
    }

    /* rotation */
    printf("Rotation\n");

    compute_lintransform(in, mag, rot, out, &trans);
    apply_lintransform(&trans, ncoords, ref, input);

    if (compare(ncoords, ref, input, output)) {
        return 1;
    }

    return 0;
}
