#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "lib/xysort.h"
#include "test.h"

int main(int argv, char** argc) {
    #define ncoords 512
    coord_t data[ncoords];
    const coord_t* ptr[ncoords];
    size_t i = 0;
    double lastx = 0.0;
    double lasty = 0.0;
    double x = 0.0;
    double y = 0.0;

    srand48(0);

    for (i = 0; i < ncoords; ++i) {
        data[i].x = drand48();
        data[i].y = drand48();
    }

    xysort(ncoords, data, ptr);

    lastx = ptr[0]->x;
    lasty = ptr[0]->y;
    for (i = 1; i < ncoords; ++i) {
        x = ptr[i]->x;
        y = ptr[i]->y;

        if (y < lasty || (y == lasty && x < lastx)) {
            return 1;
        }

        lastx = x;
        lasty = y;
    }

    return 0;
}
