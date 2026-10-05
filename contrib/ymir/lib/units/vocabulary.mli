(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Unit vocabularies, as {!Ymir_units.Vocabulary} documents them. *)

type t
type prefixing = Prefixable | Bare

val v : (string * prefixing * Unit.t) list -> t
val si : t
val union : t -> t -> t
val lookup : t -> string -> Unit.t option

type word = { prefix : int; symbol : string; num : int; den : int }
type spelling = { decade : int; words : word list }

val spell : t -> Unit.t -> spelling option
val pp : t -> Format.formatter -> Unit.t -> unit
