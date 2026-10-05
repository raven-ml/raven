/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The functions of OCML, ROCm's device math library, that the kernels call,
   at binary32 and binary64. comgr links the library with the kernels. */

#define OCML(name)                                                             \
  extern "C" __attribute__((device, const)) float __ocml_##name##_f32(float);  \
  extern "C" __attribute__((device, const)) double __ocml_##name##_f64(double);
#define OCML2(name)                                                            \
  extern "C" __attribute__((device, const)) float __ocml_##name##_f32(float,   \
                                                                      float);  \
  extern "C" __attribute__((device, const)) double __ocml_##name##_f64(        \
      double, double);

OCML(exp)
OCML(log)
OCML(log1p)
OCML(expm1)
OCML(sin)
OCML(cos)
OCML(tan)
OCML(asin)
OCML(acos)
OCML(atan)
OCML(sinh)
OCML(cosh)
OCML(tanh)
OCML(erf)
OCML2(pow)
OCML2(atan2)
OCML2(fmod)
