(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The rows of NIST's tables of the CODATA adjustments: each constant's name,
    its value with its uncertainty in {!Constant.v}'s text, and its unit. *)

val v2018 : (string * string * Unit.t) list
val v2022 : (string * string * Unit.t) list
