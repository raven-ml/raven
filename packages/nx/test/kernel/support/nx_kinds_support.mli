(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx_kinds.h's kinds, called by name through loops compiled as nx.cpu's
    rows are: each call runs the loop nx.cpu would run on this CPU. *)

val f32 : string -> int array -> int
(** [f32 kind args] is the f32 kind [kind] ([exp], [add], [fma], …) of the
    operands whose binary32 bits are [args], as binary32 bits. Comparisons
    give 1.0 or 0.0; [where] selects on a non-zero first operand. Raises
    [Invalid_argument] on an unknown kind or arity. *)

val f64 : string -> float array -> float
(** [f64 kind args] is {!f32} at f64, operands and result as floats. *)

val int : string -> string -> int64 array -> int64
(** [int ty kind args] is the integer kind [kind] at [ty], one of ["i32"],
    ["u32"], ["i64"] and ["u64"], of operands of that type, each the [int64]
    whose low bits are its bits; the result likewise, sign-extended from a
    signed 32-bit type and zero-extended from an unsigned one. Comparisons give
    1 or 0. *)

val threefry : int64 -> int64 -> int64
(** [threefry counter key] is [nx_threefry_u64]. *)

val c64 : string -> int array -> int array
(** [c64 kind args] is the c64 kind [kind] ([add], [fdiv], [less], …) of the
    operands whose parts' binary32 bits are [args], each operand's real part
    first: [[| re; im |]] as bits. A comparison gives 1.0 or 0.0 as its real
    part. Raises [Invalid_argument] on an unknown kind or arity. *)

val c128 : string -> float array -> float array
(** [c128 kind args] is {!c64} at c128, parts as floats. *)

val run :
  string ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  unit
(** [run kind x z y] writes [kind] of [x]'s elements, and [z]'s for a kind of
    two operands, into [y]: float32 or float64 vectors of one length. *)
