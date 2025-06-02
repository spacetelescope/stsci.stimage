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

#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "lib/error.h"

#define MAX_MSG 4096

/*
 * Prints a line of delimiters for visual cues.
 *
 * Primarily used for debugging.
 */
void
print_delim(
        char c,     /**< The character to use as a delimiter  */
        int nlen,   /**< The number of c to print. */
        int nret)   /**< The number of returns after the line of delimiters. */
{
    char delim_char = '=';  /* The default character */
    int k;

    if (nlen < 1) {
        return;
    }
    if (isprint(c)) {
        delim_char = c;
    }
    for (k=0; k<nlen; ++k) {
        printf("%c", delim_char);
    }
    if (nret < 1) {
        return;
    }
    if (nret > 5) {
        nret = 5;
    }
    for (k=0; k<nret; ++k) {
        printf("\n");
    }
}

void
stimage_error_init(
    stimage_error_t* const error) {

    size_t i;

    for (i = 0; i < STIMAGE_MAX_ERROR_LEN; ++i) {
        error->message[i] = '\0';
    }
}

void
stimage_error_set_message_func(char * fname, int line, stimage_error_t* error, const char* message) {
    char msg[STIMAGE_MAX_ERROR_LEN];

    assert(error);
    assert(message);

    memset(msg, 0, STIMAGE_MAX_ERROR_LEN);
    snprintf(msg, STIMAGE_MAX_ERROR_LEN-1, "[%s:%d] %s", fname, line, message);

    strncpy(error->message, msg, STIMAGE_MAX_ERROR_LEN);

#if DEBUG
    printf("ERROR RAISED:\n%s\n", error->message);
    assert(0);
#endif
}

void
stimage_error_format_message(
    stimage_error_t* error,
    const char* format,
    ...) {

  /* 
   * See http://c-faq.com/varargs/vprintf.html
   * for an explanation of how all this variable length argument list stuff
   * works.
   */
  va_list argp;

  assert(error);
  assert(format);

  va_start(argp, format);
  (void)vsnprintf(error->message, STIMAGE_MAX_ERROR_LEN, format, argp);
  va_end(argp);
}

const char*
stimage_error_get_message(
    stimage_error_t* error) {

  assert(error);

  return error->message;
}

int
stimage_error_is_set(
    const stimage_error_t* error) {

  assert(error);

  return error->message[0] != 0;
}

void
stimage_error_unset(
    stimage_error_t* error) {

  assert(error);

  error->message[0] = 0;
}
