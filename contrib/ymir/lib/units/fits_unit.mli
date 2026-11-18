(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FITS unit strings: the implementation of {!Ymir_units.Fits_unit}. *)

val vocabulary : Vocabulary.t
val parse : ?scope:string -> string -> (Unit.t, string) result
val print : Unit.t -> (string, string) result
