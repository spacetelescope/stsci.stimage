#ifndef STSCI_STIMAGE_TEST_H
#define STSCI_STIMAGE_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <float.h>

#define TEST_DATA_FMT "%.8lf"
#define TEST_DATA_FILE "drand48_linux"
#define TEST_DATA_PATH_MAX 1024

extern inline const char *get_test_data_dir() {
    const char *datadir = getenv("STIMAGE_TEST_DATA");
    if (!datadir) {
        fprintf(stderr, "error: STIMAGE_TEST_DATA environment variable must be set\n");
        exit(1);
    }
    return datadir;
}

extern inline double iter_test_data(FILE **fp) {
    char value_buf[255] = {0};
    if (fgets(value_buf, sizeof(value_buf) - 1, *fp) != NULL) {
        // truncate line feeds
        value_buf[strcspn(value_buf, "\r\n")] = 0;

        char *value_ptr = NULL;
        double value = strtod(value_buf, &value_ptr);
        if (!value_ptr && !value) {
            fprintf(stderr, "invalid value in data file\n");
            exit(1);
        }
        return value;
    }
    fprintf(stderr, "test data exhausted\n");
    exit(1);
}

extern inline FILE *get_test_data_handle(const char *filename) {
    const char *datadir = get_test_data_dir();

    char path[TEST_DATA_PATH_MAX] = {0};
    if (sprintf(path, "%s/%s", datadir, filename) < 2) {
        perror("path string creation failed");
        exit(1);
    }

    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        exit(1);
    }
    return fp;

}

#endif // STSCI_STIMAGE_TEST_H
