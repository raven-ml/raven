(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx_kinds.h's kinds, called by name through per-target loops compiled with
    dev/nx2's flags: each call runs the loop nx.cpu would run on this CPU. *)

(** {1 Calls} *)

val f32 : string -> int array -> int
(** [f32 kind args] is the f32 kind [kind] ([exp], [add], [fma], …) of the
    operands whose binary32 bits are [args], as binary32 bits. Comparisons give
    1.0 or 0.0; [where] selects on a non-zero first operand. Raises
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

val run :
  string ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  unit
(** [run kind x z y] writes [kind] of [x]'s elements, and [z]'s for a kind of
    two operands, into [y]: float32 or float64 vectors of one length, the loop
    the best target runs. *)

(** {1 Errors of the f32 transcendentals}

    Against the C library's f64 function rounded to binary32, in ulps. Where the
    reference is too close to a tie to decide between two floats, the nearer
    counts. A NaN against a number and a zero of the wrong sign count as [2^31].
    A symmetric kind is checked at each pattern's magnitude, and at the negated
    magnitude against its value there, bit for bit. *)

type worst = { errors : (int * int) array; ambiguous : int }
(** The type for a check's worst points: up to 64 [(error, pattern)] pairs, the
    largest first, the earliest pattern first among equal errors, and the number
    of ambiguous references. *)

val sweep_range : string -> int -> int -> int -> worst
(** [sweep_range kind lo hi step] checks [kind] at the bit patterns [lo],
    [lo + step], ... below [hi]; a symmetric kind only at the positive ones,
    which cover the negative. It releases the runtime. *)

val sweep : ?step:int -> ?offset:int -> string -> worst
(** [sweep ~step ~offset kind] is {!sweep_range} at every bit pattern congruent
    to [offset] modulo [step] (default [1] and [0]: all of them), split across
    the domains. Raises [Invalid_argument] unless [step] divides [2^24] and
    [0 <= offset < step]. *)

val points : string -> int array -> worst
(** [points kind ps] checks [kind] at the bit patterns [ps]. *)

val strata : string -> (string * int * worst) array
(** [strata kind] checks [kind] at the f32 strata of nx_kinds_strata.h, by
    region: its name, its size, its worst points. *)

val narrow : string -> int -> int * int
(** [narrow kind code] is the largest error of [kind] at every value of the
    narrow float dtype [code] (nx_dtype.h's codes), computed in f32 and rounded
    once to the dtype, against the C library's f64 function rounded once to it,
    in the dtype's ulps; and a code where it occurs. *)

val binary : string -> seed:int -> int -> int * int * int
(** [binary kind ~seed n] checks f32 [pow] or [atan2] at [n] pairs drawn from
    [seed], a sixth from all patterns and the rest where the kind is hard, its
    reductions' boundaries among them, against the C library's f64 function: the
    largest error and a pair of operands, as bits, where it occurs. *)

val f32_bounds : (string * int) list
(** [f32_bounds] is each f32 transcendental kind of one operand with the bound
    nx_kinds.h states for it, in ulps. *)

(** {1 Digests} *)

val targets : unit -> string list
(** [targets ()] is the targets this CPU runs: ["base"], and ["v3"] on an x86-64
    with AVX2 and FMA. *)

val digest : target:string -> string -> string -> string
(** [digest ~target kind ty] is the digest of [kind] at [ty] (["f32"], ["f64"]
    or ["int"]) over its strata on [target], every NaN one NaN, as 16 hex
    digits: what every library computing the kind gives. *)

val kinds : (string * string list) list
(** [kinds] is every kind with the types it has a loop at. *)

(** {1 The full sweep's record}

    Paths relative to dev/nx2/test/array, in the source tree and the build. *)

val headers : string list
(** [headers] is the paths of nx_kinds.h and nx_kinds_real.h, which define the
    kinds. nx_dtype.h's bit helpers, which they call, are left out: a change
    there that moves a result moves the digests of results. *)

val record : string
(** [record] is the record's path. *)

val header_digest : unit -> string
(** [header_digest ()] is the hex MD5 of the MD5s of {!headers}. *)

val write_record : (string * worst) list -> unit
(** [write_record sweeps] writes {!record}: {!header_digest}, then each kind's
    largest error over all arguments and its worst patterns. *)

val read_record : unit -> string * (string * int * int list) list
(** [read_record ()] is the digest {!record} holds and each kind's largest error
    and worst patterns. Raises [Failure] on a malformed record. *)
