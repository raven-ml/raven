/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What every worker shares: a program that runs compiles in a process of its
   own, embedded in tolk and spawned by Compiler_worker. A worker serves
   compiles until its input ends, and ends during a compile if the parent's
   end of its input closes.

     worker LIBRARY TEMP

   LIBRARY is the vendor's library, TEMP a directory of the worker's own for
   the library's temporary files (TMPDIR, or TMP and TEMP on Windows). The
   worker removes it when it ends if it is empty, and the parent removes it,
   with what a crashed compile left there, when it finds the worker ended. A
   request, on standard input, is N\n then
   N bytes: the worker's fields, each NUL-terminated, then the source. The
   reply goes to standard output, once per request:

     ok N\n<N bytes of binary>     or     error N\n<N bytes of message>

   so that a reply is complete iff it is N bytes past its header. A worker
   reads a whole request before it replies. Diagnostics (the library's) go to
   standard error.

   Each worker's program is compiled with compiler_worker.c. */

#define _GNU_SOURCE

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The library [path], or NULL, and the function [name] of it. */
void *open_library(const char *path);
void *find_symbol(void *lib, const char *name);

/* Why [path] did not load, naming it. */
const char *library_error(const char *path);

/* The reply */

/* Replies with [kind], "ok" or "error", and the [size] bytes of [data]. */
void reply_with(const char *kind, const char *data, size_t size);

/* Replies with the error [fmt] formats. */
void reply_error(const char *fmt, ...);

/* The request */

/* The next NUL-terminated field of [*at], before [end], or NULL. */
const char *field(const char **at, const char *end);

struct options {
  const char **list;
  size_t length;
};

/* A count, then that many fields, in a list [o] owns. */
bool option_list(const char **at, const char *end, struct options *o);

/* Appends [later] to [*o], whose list it replaces: false if out of memory. */
bool join(struct options *o, struct options later);

/* Serving */

/* Starts the worker that serves the requests of its standard input and
   replies on its standard output, whose library is called [library] in its
   usage: resets the process, and points standard output at standard error, so
   that what the library prints reaches the diagnostics. Exits with 2 on a
   usage error. */
void serve(int argc, char **argv, const char *library);

/* The next request, [*size] bytes NUL-terminated past them, or NULL when the
   input ends before one starts. Exits with 1, saying why, if the input ends
   inside one or it is malformed or too large to hold. */
char *next_request(size_t *size);
