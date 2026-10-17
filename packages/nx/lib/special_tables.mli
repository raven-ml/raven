(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The coefficients and region bounds of {!Special}'s functions.

    [packages/nx/test/gen/special.py] generates this interface and its
    implementation, whose comments give each table's form and source. *)

(** {1:erf The error function}

    fdlibm's [s_erf.c]: P/Q and R/S coefficients and the bounds of their
    regions in [|x|]. *)

val erx : float
val erf_small_p : float array
val erf_small_q : float array
val erf_near_p : float array
val erf_near_q : float array
val erfc_mid_p : float array
val erfc_mid_q : float array
val erfc_far_p : float array
val erfc_far_q : float array
val erfc_small_split : float
val erf_small_below : float
val erf_near_below : float
val erfc_mid_below : float
val erfc_far_from : float

(** {1:ndtri The normal quantile}

    The bounds of AS 241's regions. *)

val ndtri_central_below : float
val ndtri_central_r : float
val ndtri_near_below : float
val ndtri_near_shift : float
val ndtri_far_shift : float

(** {1:lgamma The log-gamma function}

    fdlibm's [e_lgamma_r.c]: its coefficients and the bounds of its
    regions. *)

val lgamma_tc : float
val lgamma_tf : float
val lgamma_tt : float
val lgamma_a : float array
val lgamma_t : float array
val lgamma_u : float array
val lgamma_v : float array
val lgamma_s : float array
val lgamma_r : float array
val lgamma_w0 : float
val lgamma_w : float array
val lgamma_small : float
val lgamma_c : float
val lgamma_b : float
val lgamma_f : float
val lgamma_e : float
val lgamma_shift_from : float
val lgamma_stirling_from : float

(** {1:bessel Scaled Bessel functions}

    The bounds of i0e's and i1e's regions. *)

val bessel_split : float
val bessel_weight : float

(** {1:igamma The incomplete gamma function}

    The bounds of the incomplete gamma function's regions, and its inverse's
    first guess. *)

val igamma_corner : float
val igamma_temme_from : float
val igamma_temme_width : float
val igamma_near_v : float
val erfcx_far : float
val igamma_guess_p : float array
val igamma_guess_q : float array
val igamma_guess_terms : float

(** {1:beta The incomplete beta function}

    The bounds of the incomplete beta function's regions, as TOMS 708 splits
    them. *)

val beta_bpser_bx : float
val beta_bup_x : float
val beta_power_bx : float
val beta_power_x : float
val beta_a_small : float
val beta_x_far : float
val beta_x_near : float
val beta_bgrat_b : float
val beta_b_small : float
val beta_frac_a : float
val beta_frac_lambda : float
val beta_bup_terms : int
val beta_bup_most : int
val beta_large : float
val rlog1_series : float

(** {1:dtypes Tables per dtype} *)

(** The type for the tables whose values differ between dtypes. *)
type t = {
  erfinv_central : float array;
  erfinv_central_below : float;
  erfinv_central_shift : float;
  erfinv_near : float array;
  erfinv_near_shift : float;
  erfinv_far : float array;
  erfinv_far_from : float;
  erfinv_far_shift : float;
  erfinv_newton_split : float;
  sqrt1_2_hi : float;
  sqrt1_2_lo : float;
  ndtri_central_p : float array;
  ndtri_central_q : float array;
  ndtri_near_p : float array;
  ndtri_near_q : float array;
  ndtri_far_p : float array;
  ndtri_far_q : float array;
  log_ndtr_below : float;
  log_ndtr_series : float array;
  digamma_root_hi : float;
  digamma_root_lo : float;
  digamma_core : float array;
  digamma_from : float;
  sinpi : float array;
  digamma_series : float array;
  i0e_near : float array;
  i1e_near : float array;
  i0e_far : float array;
  i1e_far : float array;
  igamma_corner_terms : float;
  igamma_series_terms : float;
  igamma_cf_depth : float;
  igamma_atanh : float array;
  igamma_shift_terms : float array;
  igamma_temme : float array;
  igamma_temme_degrees : float array;
  igamma_halley_steps : float;
  erfcx_series : float array;
  beta_eps : float;
  rlog1 : float array;
  bpser_terms : int;
  bfrac_depth : int;
  basym : float array array;
  bgrat : float array array;
}

val float32 : t
val float64 : t
