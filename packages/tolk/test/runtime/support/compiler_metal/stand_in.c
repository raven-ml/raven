/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A library that stands in for MTLCompiler. Built with NO_SERVICE, its
   MTLCodeGenServiceCreate makes no service and its
   MTLCodeGenServiceBuildRequest aborts, which a compile must never reach.
   Built without, it has neither function. */

#include <stdlib.h>

#ifdef NO_SERVICE
void *MTLCodeGenServiceCreate(const char *name) {
  (void)name;
  return NULL;
}

void MTLCodeGenServiceBuildRequest(void) { abort(); }
#else
int tolk_stand_in = 0;
#endif
