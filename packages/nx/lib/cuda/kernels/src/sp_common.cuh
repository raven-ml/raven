/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the spectral kernels share: the four formats they compute in
   (float, double and their complex numbers), scalar arithmetic that reads
   the same on each, the formats' constants, and the scan that finds a
   matrix's largest magnitude and whether it is finite.

   Matrices are row-major, element (i, j) of an r × c matrix at
   x[i·c + j], one after another along the batch. A kernel specialises by
   format alone: T is the element, R its real part. */

#ifndef SPECTRAL_COMMON_CUH
#define SPECTRAL_COMMON_CUH


#define D __device__ __forceinline__

template <class R> struct cx {
  R re, im;
};
typedef cx<float> cf;
typedef cx<double> cd;

template <class T> struct real_of {
  typedef T R;
  static const bool cplx = false;
};
template <class Q> struct real_of<cx<Q>> {
  typedef Q R;
  static const bool cplx = true;
};

/* Scalars. A real T reads as a complex one of zero imaginary part. */
template <class R> D R re(R x) { return x; }
template <class R> D R re(cx<R> x) { return x.re; }
template <class R> D R im(R) { return 0; }
template <class R> D R im(cx<R> x) { return x.im; }
template <class R> D R conj(R x) { return x; }
template <class R> D cx<R> conj(cx<R> x) { return {x.re, -x.im}; }
template <class R> D R abs2(R x) { return x * x; }
template <class R> D R abs2(cx<R> x) { return x.re * x.re + x.im * x.im; }
template <class R> D R mag(R x) { return fabs(x); }
template <class R> D R mag(cx<R> x) { return hypot(x.re, x.im); }
/* |re| + |im|, LAPACK's cheap magnitude. */
template <class R> D R mag1(R x) { return fabs(x); }
template <class R> D R mag1(cx<R> x) { return fabs(x.re) + fabs(x.im); }
template <class R> D cx<R> operator+(cx<R> a, cx<R> b) {
  return {a.re + b.re, a.im + b.im};
}
template <class R> D cx<R> operator-(cx<R> a, cx<R> b) {
  return {a.re - b.re, a.im - b.im};
}
template <class R> D cx<R> operator-(cx<R> a) { return {-a.re, -a.im}; }
template <class R> D cx<R> operator*(cx<R> a, cx<R> b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <class R> D cx<R> operator*(R s, cx<R> a) {
  return {s * a.re, s * a.im};
}
template <class R> D cx<R> operator*(cx<R> a, R s) {
  return {s * a.re, s * a.im};
}
template <class R> D cx<R> operator/(cx<R> a, R s) {
  return {a.re / s, a.im / s};
}
template <class R> D cx<R> &operator+=(cx<R> &a, cx<R> b) {
  return a = a + b;
}
template <class R> D cx<R> &operator-=(cx<R> &a, cx<R> b) {
  return a = a - b;
}
template <class R> D bool operator==(cx<R> a, cx<R> b) {
  return a.re == b.re && a.im == b.im;
}
template <class R> D bool operator!=(cx<R> a, cx<R> b) { return !(a == b); }

/* a / b by Smith's algorithm, which overflows only where the quotient
   does. */
template <class R> D R cdiv(R a, R b) { return a / b; }
template <class R> D cx<R> cdiv(cx<R> a, cx<R> b) {
  if (fabs(b.re) >= fabs(b.im)) {
    R r = b.im / b.re, d = b.re + b.im * r;
    return {(a.re + a.im * r) / d, (a.im - a.re * r) / d};
  }
  R r = b.re / b.im, d = b.re * r + b.im;
  return {(a.re * r + a.im) / d, (a.im * r - a.re) / d};
}
template <class R> D cx<R> cdiv(cx<R> a, R b) { return a / b; }
template <class R> D cx<R> operator/(cx<R> a, cx<R> b) { return cdiv(a, b); }

/* The element of real part [r] and imaginary part [i], [i] dropped on a
   real T. */
template <class T> struct mk {
  template <class R> static D T of(R r, R) { return r; }
};
template <class Q> struct mk<cx<Q>> {
  template <class R> static D cx<Q> of(R r, R i) { return {(Q)r, (Q)i}; }
};
template <class T, class R> D T elt(R r, R i = 0) { return mk<T>::of(r, i); }

/* The unit roundoff, the least normal number, the least positive
   subnormal. */
template <class R> struct lim;
template <> struct lim<float> {
  static D float eps() { return FLT_EPSILON / 2; }
  static D float tiny() { return FLT_MIN; }
  static D float eta() { return 1.40129846e-45f; }
};
template <> struct lim<double> {
  static D double eps() { return DBL_EPSILON / 2; }
  static D double tiny() { return DBL_MIN; }
  static D double eta() { return 4.9406564584124654e-324; }
};

/* A sum of squares that carries its rounding errors along: Neumaier's
   compensated sum, each square's own error taken exactly by a fused
   multiply-add. Within a few u of the exact sum whatever the count. A
   plain running sum loses every term below half its last bit, so it
   comes out low, and a vector divided by its root comes out long. */
template <class R> struct sumsq {
  R s = 0, c = 0;
  D void add(R t) {
    R y = s + t;
    c += fabs(s) >= fabs(t) ? (s - y) + t : (t - y) + s;
    s = y;
  }
  D void sq(R x) {
    R p = x * x;
    add(p);
    c += fma(x, x, -p);
  }
  D void sq(cx<R> x) { sq(x.re), sq(x.im); }
  D R get() const { return s + c; }
};

/* The exponent e whose 2^e brings [big], a matrix's largest magnitude,
   into [1, 2) where it lies outside [lo, hi], 0 where it lies inside or
   is 0. Scaling by ldexp is exact both ways for every element that stays
   normal. */
template <class R> D int scale_exp(R big, R lo, R hi) {
  if (big == 0 || (big >= lo && big <= hi)) return 0;
  return -ilogb(big);
}
template <class R> D R scaled(R x, int e) { return ldexp(x, e); }
template <class R> D cx<R> scaled(cx<R> x, int e) {
  return {ldexp(x.re, e), ldexp(x.im, e)};
}

#endif
