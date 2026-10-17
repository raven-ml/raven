(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FITS numbers compared as the exact decimals they write.

    [32768], [3.2768E4] and [32768.0] are one number, and
    [9223372036854775807] is not 2{^ 63}. *)

val equal : string -> string -> bool
(** [equal a b] is [true] iff the real texts [a] and [b] (FITS 4.0 §4.2.4)
    write the same decimal. It is [false] if either is not real text or has
    an exponent past [int]'s range. *)
