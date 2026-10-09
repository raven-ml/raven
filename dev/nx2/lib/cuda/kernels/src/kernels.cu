/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx.cuda's cubin: every family in one unit, each kernel defined from its
   row of kernels.h's NX_CUDA_KERNELS by its family's macro. */

#include "contract.cu"

#include "nx_kinds.h"
#include "nx_spec.h"

#include "kinds.cuh"

#include "fold.cu"

#define DEFINE(name, FAMILY, ...) FAMILY(name, __VA_ARGS__)
NX_CUDA_KERNELS(DEFINE)
