#ifndef STSCI_STIMAGE_TEST_H
#define STSCI_STIMAGE_TEST_H

#ifdef _WIN32

#ifdef _MSC_VER

#include <stdlib.h>

#ifndef srand48
#define srand48 srand
#endif // srand48

#ifndef drand48
#define drand48 rand
#endif // drand48

#endif // _MSC_VER

#endif // _WIN32

#endif // STSCI_STIMAGE_TEST_H
