#include <math.h>
#include "test.h"

#define fname_max 1024
#define fname_default "numbers.txt"

char *get_basename(char *s) {
    const char *dirseps = "/\\";
    size_t len = strlen(s);
    do {
       for (size_t i = 0; i < strlen(dirseps); i++) {
           char ch = s[len];
           char sep = dirseps[i];
           char *tmp = &s[len];
           if (ch == sep) {
               return &tmp[len + strlen(tmp) ? 1 : 0];
           }
       }
    } while (len--);
    return s;
}

void usage(char *argv0) {
    char *name = get_basename(argv0);
    printf("usage: %s {nelem} [filename]\n", name);
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
    printf("srand48 and drand48 are not supported on windows.");
    exit(1);
#else
    if (argc < 2) {
        fprintf(stderr, "Number of elements required\n");
        usage(argv[0]);
        exit(1);
    }

    char *nelem_ptr = NULL;
    size_t nelem = strtoul(argv[1], &nelem_ptr, 10);
    if (nelem_ptr && !nelem) {
        fprintf(stderr, "Invalid number of elements: '%s'\n", nelem_ptr);
        usage(argv[0]);
        exit(1);
    }

    char outfile[fname_max] = {0};
    if (argc < 3) {
        strcpy(outfile, fname_default);
    } else {
        strcpy(outfile, argv[2]);
    }

    FILE *fp = fopen(outfile, "w+");
    if (!fp) {
        perror(outfile);
        exit(1);
    }

    printf("Writing %zu elements to %s\n", nelem, outfile);
    srand48(0);
    for (size_t i = 0; i < nelem; i++) {
        fprintf(fp, TEST_DATA_FMT "\n", drand48());
    }

    fclose(fp);
#endif
    return 0;
}