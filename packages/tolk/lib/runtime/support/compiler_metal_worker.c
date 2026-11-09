/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* The program that runs MTLCompiler's compiles in a process of its own,
   embedded in tolk and spawned by Compiler_metal. MTLCompiler reads its
   options into LLVM's, which are global to a process, at every compile: a
   process per worker lets compiles run at once. It serves compiles until its
   input ends, so that loading MTLCompiler is paid once per process.

     worker MTLCOMPILER

   MTLCOMPILER is MTLCompiler's file, which macOS keeps in its shared cache of
   libraries. A request is MTLCompiler's own: the sizes of the source and of
   the options, each a 64-bit little-endian integer, then both, each padded
   with NULs. A reply holds MTLCompiler's reply, whose Metal library follows a
   header and the warnings, as compiler_worker.h says. Elsewhere than on
   macOS, MTLCompiler never loads. */

#include "compiler_worker.h"

/* Why MTLCompiler did not load, or "" once it has. */
static char load_error[4096];

#ifdef __APPLE__

/* The request type that compiles source to a Metal library. It is not
   documented, and the framework has no symbol for it. tolk's table of Metal
   libraries holds neither it nor the service's name: changing either changes
   Compiler_metal's table. */
#define REQUEST_TYPE_COMPILE 13
#define SERVICE_NAME "tolk"

typedef void (^reply_t)(int32_t error, const void *data, size_t size,
                        const char *message);
static void *(*create)(const char *);
static void (*build)(void *, void *, int32_t, const void *, size_t, reply_t);
static void *service;

/* Loads MTLCompiler and makes its service, or says why it cannot in
   [load_error]: a library that changed under a macOS update replies so to each
   compile, and never crashes. */
static bool load(const char *path) {
  void *lib = open_library(path);
  if (lib == NULL) {
    snprintf(load_error, sizeof load_error,
             "failed to load library MTLCompiler: %s", library_error(path));
    return false;
  }
  const char *missing = NULL;
  if ((create = find_symbol(lib, "MTLCodeGenServiceCreate")) == NULL)
    missing = "MTLCodeGenServiceCreate";
  else if ((build = find_symbol(lib, "MTLCodeGenServiceBuildRequest")) == NULL)
    missing = "MTLCodeGenServiceBuildRequest";
  if (missing != NULL) {
    snprintf(load_error, sizeof load_error,
             "failed to load library MTLCompiler: %s: undefined symbol %s",
             path, missing);
    return false;
  }
  if ((service = create(SERVICE_NAME)) == NULL) {
    snprintf(load_error, sizeof load_error,
             "failed to load library MTLCompiler: %s: MTLCodeGenServiceCreate "
             "made no service",
             path);
    return false;
  }
  return true;
}

/* Compiles the request [request], [size] bytes, and replies with
   MTLCompiler's reply or its message. The reply comes through the block,
   before the request returns. */
static void compile(const char *request, size_t size) {
  __block char *reply = NULL, *message = NULL;
  __block size_t reply_size = 0;
  __block bool called = false;
  build(service, NULL, REQUEST_TYPE_COMPILE, request, size,
        ^(int32_t error, const void *data, size_t data_size, const char *msg) {
          called = true;
          if (error != 0)
            message = strdup(msg != NULL ? msg : "");
          else if ((reply = malloc(data_size ? data_size : 1)) != NULL)
            memcpy(reply, data, reply_size = data_size);
        });
  if (!called)
    reply_error(
        "MTLCodeGenServiceBuildRequest returned without calling the callback");
  else if (message != NULL)
    reply_error("%s", message);
  else if (reply == NULL)
    reply_error("out of memory");
  else
    reply_with("ok", reply, reply_size);
  free(reply);
  free(message);
}

#else

static bool load(const char *path) {
  (void)path;
  snprintf(load_error, sizeof load_error,
           "failed to load library MTLCompiler: MTLCompiler is macOS's");
  return false;
}

static void compile(const char *request, size_t size) {
  (void)request;
  (void)size;
  reply_error("%s", load_error);
}

#endif

int main(int argc, char **argv) {
  serve(argc, argv, "MTLCOMPILER");
  bool loaded = load(argv[1]);
  size_t size;
  for (char *request; (request = next_request(&size)) != NULL;) {
    if (loaded)
      compile(request, size);
    else
      reply_error("%s", load_error);
    free(request);
  }
  return 0;
}
