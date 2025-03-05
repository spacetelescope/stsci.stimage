#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "lib/xysort.h"
#include "lib/xycoincide.h"
#include "test.h"

int main(int argv, char** argc) {
    #define ncoords 512
    coord_t data[ncoords];
    const coord_t* ptr[ncoords];
    size_t i = 0;
    size_t j = 0;
    size_t nunique = 0;
    double x0, y0, x1, y1;
    double dx, dy;
    double distance2;
    const double tolerance = 0.1;
    const double tolerance2 = tolerance*tolerance;

    FILE *data_handle = get_test_data_handle(TEST_DATA_FILE);

    for (i = 0; i < ncoords; ++i) {
        data[i].x = iter_test_data(&data_handle);
        data[i].y = iter_test_data(&data_handle);
    }

    xysort(ncoords, data, ptr);

    nunique = xycoincide(ncoords, ptr, ptr, tolerance);

    for (i = 0; i < nunique; ++i) {
        for (j = 0; j < nunique; ++j) {
            if (i == j) continue;

            x0 = ptr[i]->x;
            y0 = ptr[i]->y;
            x1 = ptr[j]->x;
            y1 = ptr[j]->y;
            dx = x1 - x0;
            dy = y1 - y0;
            distance2 = dx*dx + dy*dy;
            if (distance2 < tolerance2) {
                printf("Found distance of %f between (%f, %f) [%lu] and (%f, %f) [%lu]\n",
                       distance2, x0, y0, (unsigned long)i, x1, y1, (unsigned long)j);
                return 1;
            }
        }
    }

    fclose(data_handle);
    return 0;
}
