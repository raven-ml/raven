/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The C library the kernels see: the exact-width integers, as AMD GPUs lay
   them out, and the limits the kernels use. */

#ifndef NX_AMD_STDINT_H
#define NX_AMD_STDINT_H

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

#define UINT8_MAX 0xff
#define INT64_MAX 0x7fffffffffffffffL
#define INT64_MIN (-INT64_MAX - 1)
#define UINT64_MAX 0xffffffffffffffffUL

#endif
