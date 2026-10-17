(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FITS unit strings: the implementation of {!Ymir_fits.Fits.Unit}. *)

val vocabulary : Ymir_units.Vocabulary.t
val parse : ?scope:string -> string -> (Ymir_units.Unit.t, string) result
val print : Ymir_units.Unit.t -> (string, string) result
