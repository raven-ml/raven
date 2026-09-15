/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Loading a vendor library at first use, on every platform. A runtime that
   resolves CUDA, NVRTC, or comgr this way builds and loads on machines
   without them; Windows spells the same two calls differently. */

#ifndef TOLK_DL_H
#define TOLK_DL_H

#if defined(_WIN32)
#include <windows.h>

static inline void *tolk_dlopen(const char *name) {
  return (void *)LoadLibraryA(name);
}

static inline void *tolk_dlsym(void *handle, const char *symbol) {
  return (void *)GetProcAddress((HMODULE)handle, symbol);
}
#else
#include <dlfcn.h>

static inline void *tolk_dlopen(const char *name) {
  return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}

static inline void *tolk_dlsym(void *handle, const char *symbol) {
  return dlsym(handle, symbol);
}
#endif

#endif /* TOLK_DL_H */
