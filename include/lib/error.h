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

#ifndef _XY_COORD_MATCH_ERROR_H_
#define _XY_COORD_MATCH_ERROR_H_

#define XY_COORD_MATCH_MAX_ERROR_LEN 512

typedef struct {
    char message[XY_COORD_MATCH_MAX_ERROR_LEN];
} xy_coord_match_error_t;

/*
 Initialize the error buffer
*/
void
xy_coord_match_error_init(xy_coord_match_error_t *const error);

/**
 Set the message in the error object to the given string.
 */
void
xy_coord_match_error_set_message(xy_coord_match_error_t *error, const char *message);

/**
 Set the message in the error object using printf-style formatting
 */
void
xy_coord_match_error_format_message(xy_coord_match_error_t *error, const char *format, ...);

/**
 Get the current message in the error object
 */
const char *
xy_coord_match_error_get_message(xy_coord_match_error_t *error);

/**
 Returns non-zero if an error message has been set
 */
int
xy_coord_match_error_is_set(const xy_coord_match_error_t *const error);

/**
 Remove the error message from the object
 */
void
xy_coord_match_error_unset(xy_coord_match_error_t *error);

#endif /* _XY_COORD_MATCH_ERROR_H_ */
