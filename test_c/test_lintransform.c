#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "lib/lintransform.h"
#include "test.h"

void
print_array(const size_t ncoords, const coord_t* data, const char* name) {
    size_t i;

    printf("%s = [ ", name);
    for (i = 0; i < ncoords; ++i) {
        printf("[%f, %f], ", data[i].x, data[i].y);
    }
    printf("]\n");
}

int main(int argv, char** argc) {
    #define ncoords 512
    coord_t data[ncoords];
    coord_t data_trans[ncoords];
    lintransform_t transform;
    coord_t in = {0.0, 0.0};
    coord_t mag = {1.0, 1.0};
    coord_t rot = {0.0, 0.0};
    coord_t out = {0.0, 0.0};
    size_t i = 0;

    srand48(0);

    for (i = 0; i < ncoords; ++i) {
        data[i].x = drand48();
        data[i].y = drand48();
    }

    print_array(ncoords, data, "orig");

    /* First, test identity */
    compute_lintransform(in, mag, rot, out, &transform);
    apply_lintransform(&transform, ncoords, data, data_trans);

    for (i = 0; i < ncoords; ++i) {
        if (data[i].x != data_trans[i].x ||
            data[i].y != data_trans[i].y) {
            return 1;
        }
    }

    /* Now, a magnification X 2 */
    mag.x = 2.0;
    mag.y = 2.0;

    compute_lintransform(in, mag, rot, out, &transform);
    apply_lintransform(&transform, ncoords, data, data_trans);

    for (i = 0; i < ncoords; ++i) {
        if (data[i].x * 2 != data_trans[i].x ||
            data[i].y * 2 != data_trans[i].y) {
            return 1;
        }
    }

    print_array(ncoords, data_trans, "mag");

    /* Now, a rotation */
    mag.x = 1.0;
    mag.y = 1.0;
    rot.x = 2.0;
    rot.y = 2.0;

    compute_lintransform(in, mag, rot, out, &transform);
    apply_lintransform(&transform, ncoords, data, data_trans);

    /* for (i = 0; i < ncoords; ++i) { */
    /*     if (data[i].x * 2 != data_trans[i].x || */
    /*         data[i].y * 2 != data_trans[i].y) { */
    /*         return 1; */
    /*     } */
    /* } */

    print_array(ncoords, data_trans, "rot");

    printf("\n\n");
    fflush(stdout);

    return 0;
}
