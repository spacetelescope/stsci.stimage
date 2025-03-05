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

    FILE *data_handle = get_test_data_handle(TEST_DATA_FILE);

    for (i = 0; i < ncoords; ++i) {
        data[i].x = iter_test_data(&data_handle);
        data[i].y = iter_test_data(&data_handle);
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

    fclose(data_handle);
    return 0;
}
