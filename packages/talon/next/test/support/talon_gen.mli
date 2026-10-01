(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Generators of types and values for talon's suites. *)

open Talon_next
open Windtrap

val type_ : Type.any Gen.t
(** [type_] draws a type: every scalar type at its edges of precision, scale,
    unit and dictionary, extensions over them, and lists and records nested two
    deep. [string], an extension, and a record of a record with an extension
    field are each drawn in at least one case of seven. *)

val value : 'a Type.t -> 'a Gen.t option
(** [value ty] draws values [ty] holds, its bounds included: [min_int], [-0.],
    NaN, infinities, the first and last instants, empty and non-ASCII text. It
    is [None] for a type with no value that OCaml can build: an extension, and a
    record with a field that is or holds one. *)

val options : 'a Type.t -> 'a option array Gen.t
(** [options ty] draws up to forty rows of [ty], some of them null, and no row
    in at least one case of five. Every row is null when [value ty] is [None].
*)

(** The type for a type and rows of it. *)
type sample = Sample : 'a Type.t * 'a option array -> sample

val sample : sample Gen.t
(** [sample] is {!options} of a {!type_}. *)

val pp_sample : Format.formatter -> sample -> unit
(** [pp_sample] formats a sample's type and rows. *)

val split : Talon_next.t -> Talon_next.t Gen.t
(** [split t] draws [t] cut into batches: {!Talon_next.of_batches} of runs of
    its rows, empty runs and runs of one row included. *)

val witness : 'a Type.t -> 'a Testable.t
(** [witness ty] compares values of [ty] by {!Type.compare_value}, and floats by
    their bits, all NaNs being equal: [-0.] and [0.] differ. *)
