/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/mlvalues.h>

#if defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

/* Whether macOS lists half-precision arithmetic among the host processor's
   features (hw.optional.arm.FEAT_FP16); false elsewhere. */
value caml_tolk_cpu_mac_fp16(value unit) {
  (void)unit;
#if defined(__APPLE__) && defined(__aarch64__)
  int on = 0;
  size_t size = sizeof on;
  if (sysctlbyname("hw.optional.arm.FEAT_FP16", &on, &size, NULL, 0) != 0)
    return Val_false;
  return Val_bool(on);
#else
  return Val_false;
#endif
}
