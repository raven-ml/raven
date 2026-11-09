/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* The version and the build of macOS, which pick the language MTLCompiler
   compiles and name it. Elsewhere than on macOS, they are 0 and "". */

#include <caml/alloc.h>
#include <caml/mlvalues.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <sys/sysctl.h>

value caml_tolk_metal_macos_major(value unit) {
  (void)unit;
  char version[32];
  size_t size = sizeof version;
  if (sysctlbyname("kern.osproductversion", version, &size, NULL, 0) != 0)
    return Val_int(0);
  return Val_int(atoi(version));
}

/* The build of macOS, as 25D125: MTLCompiler is part of it. */
value caml_tolk_metal_macos_build(value unit) {
  (void)unit;
  char build[64];
  size_t size = sizeof build;
  if (sysctlbyname("kern.osversion", build, &size, NULL, 0) != 0)
    return caml_copy_string("");
  return caml_copy_string(build);
}

#else

value caml_tolk_metal_macos_major(value unit) {
  (void)unit;
  return Val_int(0);
}

value caml_tolk_metal_macos_build(value unit) {
  (void)unit;
  return caml_copy_string("");
}

#endif
