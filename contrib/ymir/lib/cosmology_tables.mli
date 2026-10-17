(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The tables {!Cosmology} computes with, one per compute dtype, which
    [test/gen/cosmology.py] writes. *)

type t = {
  distance_nodes : float array;
  distance_weights : float array;
  age_nodes : float array;
  age_weights : float array;
  f_series_below : float;
  f_series : float array;
  f_series_log : float;
  f_log_floor : float;
  f_cheb_mid : float;
  f_cheb_scale : float;
  f_cheb : float array;
  f_asymptotic_from : float;
  f_asymptotic : float array;
  sinc_taylor : float array;
  volume_taylor : float array;
}
(** The type for a dtype's tables. [distance_*] and [age_*] are
    Gauss-Legendre rules on \[0, 1\], nodes ascending, weights summing to 1.

    F, the neutrinos' energy function, is
    [1 + y² (P(y) + f_series_log y² log (max y f_log_floor))] up to
    [f_series_below], P's coefficients [f_series] from the leading one; the
    Chebyshev sum [Σ_j f_cheb.(j) T_j(t)] with
    [t = (log y - f_cheb_mid) f_cheb_scale] between; and [y Q(1/y²)] from
    [f_asymptotic_from], Q's coefficients [f_asymptotic] from the leading
    one.

    [sinc_taylor] and [volume_taylor] are the Taylor polynomials of S and W
    ({!Ymir.Cosmology.transverse}, {!Ymir.Cosmology.comoving_volume}) on
    \[-1, 1\], from the leading coefficient. Callers must not change the
    arrays. *)

val float32 : t
val float64 : t
