/*
Copyright (C) 2008-2025 Association of Universities for Research in Astronomy (AURA)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of AURA and its representatives may not be used to
      endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

/*
 Author: Michael Droettboom
*/

#ifndef _STIMAGE_ERROR_H_
#define _STIMAGE_ERROR_H_

#define STIMAGE_MAX_ERROR_LEN 512

#include <limits.h>
#include <stdio.h>

#define PARAM_UNUSED(X) (void)(X)  // Used for unused parameter warnings

int base_print_function(FILE * fd, int line, const char * filename, const char * label);
int base_print_function2(
        FILE * fd, int line, const char * filename, const char * label, const char * fmt, ...);

#define print_base(F, L) base_print_function(F, __LINE__, __FILE__, L);
#define dbg_print(...) do{print_base(stdout, "DEBUG"); fprintf(stdout,__VA_ARGS__);}while(0)
#define err_print(...) do{print_base(stderr, "ERROR"); fprintf(stdout,__VA_ARGS__);}while(0)

#if 0
// For some reason this doesn't work.
#define dbg_print2(FMT, ...) do{base_print_function2(stdout, __LINE__, __FILE__, "DEBUG", FMT, __VA_ARGS__while(0)
#endif

/* Message structure */
typedef struct {
  char message[STIMAGE_MAX_ERROR_LEN];
} stimage_error_t;

/*
 Initialize the error buffer
*/
void
stimage_error_init(stimage_error_t* const error);

/**
 Set the message in the error object to the given string.
 */
void
stimage_error_set_message(stimage_error_t* error, const char* message);

/**
 Set the message in the error object using printf-style formatting
 */
void
stimage_error_format_message(stimage_error_t* error, const char* format, ...);

/**
 Get the current message in the error object
 */
const char*
stimage_error_get_message(stimage_error_t* error);

/**
 Returns non-zero if an error message has been set
 */
int
stimage_error_is_set(const stimage_error_t* const error);

/**
 Remove the error message from the object
 */
void
stimage_error_unset(stimage_error_t* error);

#endif /* _STIMAGE_ERROR_H_ */
