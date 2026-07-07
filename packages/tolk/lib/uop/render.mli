(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tinygrad-shaped UOp graph rendering.

    This module mirrors tinygrad's [tinygrad/uop/render.py] organization for
    stable graph listings. The output is intended for debugging and golden
    tests, not for parsing. *)

val uops_list_to_string : Uop.t list -> string
(** [uops_list_to_string uops] is a tinygrad-shaped debug listing of [uops] in
    the supplied order. Rows contain the row index, operation, live range
    column, dtype, sources, and payload. Sources that refer to constants in
    [uops] are printed as quoted constant values; other sources in [uops] are
    printed as row indexes. The output is deterministic and intended for golden
    tests. Side {!Uop.metadata} is not printed. *)

val pp_uops : Format.formatter -> Uop.t list -> unit
(** [pp_uops ppf uops] formats {!uops_list_to_string} [uops] on [ppf]. *)

val uops_to_string : ?label:string -> Uop.t -> string
(** [uops_to_string ?label root] is {!uops_list_to_string} over
    {!Uop.toposort}[ root]. When supplied, [label] is printed as a
    ["=== label ==="] header before the rows. *)
