#include <stdio.h>
#include <stdlib.h>

#include "immatch/xyxymatch.h"
#include "test.h"

int main(int argc, char** argv) {
    #define ncoords 512
    coord_t ref[ncoords];
    coord_t input[ncoords];
    xyxymatch_output_t output[ncoords];
    size_t noutput = ncoords;
    coord_t origin = {0.0, 0.0};
    coord_t mag = {1.0, 1.0};
    coord_t rot = {0.0, 0.0};
    coord_t ref_origin = {0.0, 0.0};
    stimage_error_t error;
    double x0, y0, x1, y1;
    double dx, dy;
    double distance;
    const double tolerance = 0.01;
    int status;

    size_t i = 0;

    stimage_error_init(&error);

    srand48(0);

    for (i = 0; i < ncoords; ++i) {
        ref[i].x = input[i].x = drand48();
        ref[i].y = input[i].y = drand48();
    }

    status = xyxymatch(ncoords, input,
                       ncoords, ref,
                       &noutput, output,
                       &origin, &mag, &rot, &ref_origin,
                       xyxymatch_algo_tolerance,
                       tolerance, 0.0, 0, 0.0, 0,
                       &error);

    if (status) {
        printf(stimage_error_get_message(&error));
        return status;
    }

    if (noutput != ncoords) {
        return 1;
    }

    for (i = 0; i < noutput; ++i) {
        x0 = output[i].coord.x;
        y0 = output[i].coord.y;
        x1 = output[i].ref.x;
        y1 = output[i].ref.y;
        dx = x1 - x0;
        dy = y1 - y0;
        distance = dx*dx + dy*dy;
        if (distance > tolerance*tolerance) {
            printf("Match beyond tolerance\n");
            return 1;
        }
        if (output[i].coord_idx > ncoords ||
            output[i].ref_idx > ncoords) {
            printf("Out of range indices\n");
            return 1;
        }
    }

    /* Now with different values in input and ref */

    for (i = 0; i < ncoords; ++i) {
        input[i].x = drand48();
        input[i].y = drand48();
        ref[i].x = drand48();
        ref[i].y = drand48();
    }

    status = xyxymatch(ncoords, input,
                       ncoords, ref,
                       &noutput, output,
                       &origin, &mag, &rot, &ref_origin,
                       xyxymatch_algo_tolerance,
                       tolerance, 0.0, 0, 0.0, 0,
                       &error);

    if (status) {
        printf(stimage_error_get_message(&error));
        return status;
    }

    if (noutput == 0 || noutput == ncoords) {
        return 1;
    }

    for (i = 0; i < noutput; ++i) {
        x0 = output[i].coord.x;
        y0 = output[i].coord.y;
        x1 = output[i].ref.x;
        y1 = output[i].ref.y;
        dx = x1 - x0;
        dy = y1 - y0;
        distance = dx*dx + dy*dy;
        if (distance > tolerance*tolerance) {
            return 1;
        }
        if (output[i].coord_idx > ncoords ||
            output[i].ref_idx > ncoords) {
            printf("Out of range indices\n");
            return 1;
        }
    }

    return status;
}
