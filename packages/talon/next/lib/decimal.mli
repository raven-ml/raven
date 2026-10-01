(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Decimal numbers.

    A decimal is the exact number [unscaled] divided by 10{^ [scale]}, where
    [unscaled] is an integer of at most 18 digits and [scale] counts the digits
    after the decimal point. Decimal columns ({!Type.decimal}) read as {!t}. *)

type t
(** The type for decimals. *)

val v : unscaled:int64 -> scale:int -> t
(** [v ~unscaled ~scale] is [unscaled] divided by 10{^ [scale]}.

    Raises [Invalid_argument] if [scale] is not in \[[0];[18]\] or if [unscaled]
    is not in \[[-999_999_999_999_999_999L];[999_999_999_999_999_999L]\]. *)

val unscaled : t -> int64
(** [unscaled d] is the integer [d] was made with. *)

val scale : t -> int
(** [scale d] is the scale [d] was made with. *)

val equal : t -> t -> bool
(** [equal d0 d1] is [true] iff [d0] and [d1] are the same number. Equal
    decimals may differ in scale: [v ~unscaled:10L ~scale:1] and
    [v ~unscaled:1L ~scale:0] are equal. *)

val compare : t -> t -> int
(** [compare d0 d1] orders decimals by their value. It is exact for every pair
    of scales and compatible with {!equal}. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf d] formats [d] in decimal notation with exactly [scale d] digits
    after the point, and no point when the scale is [0]: [12.34], [-0.050], [7].
*)
