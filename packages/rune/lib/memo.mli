(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Plans derived from scalar programs, kept per domain.

    A rule derives a {e plan} from a scalar program ([Nx_kernel.Prog.t]), a
    mask of its operands and a few codes of the operation that applies it, as
    a map's derivative from its program and which operands have a tangent. A
    memo keeps the plans it made. Each domain has its own: the last key it
    found, which a chain of one kind of operation finds again without hashing,
    then a table, emptied when it holds {!entries} plans. The systhreads of a
    domain share its memo, and each reads it whole: a thread that takes over
    while another adds a plan finds the table before or after the addition. *)

module P := Nx_kernel.Prog

type key = { prog : P.t; mask : int; codes : int array }
(** The type for keys: [prog], bit [j] of [mask] for operand [j] of [prog],
    and [codes], which the rule defines. A plan is a function of its key. *)

type 'v t
(** The type for memos of plans of type ['v]. *)

val entries : int
(** [entries] is the number of plans a domain's table holds before it is
    emptied. *)

val fits : int -> bool
(** [fits n] is whether a mask holds a bit for each of [n] operands. A rule
    over more derives its plan at each call. *)

val operands : key -> bool array
(** [operands k] is bit [j] of [k.mask] for each operand [j] of [k.prog]. *)

val create : unit -> 'v t
(** [create ()] is a memo whose every domain's table is empty. *)

val find :
  'v t ->
  P.t ->
  int ->
  same:(int array -> 'a -> bool) ->
  codes:('a -> int array) ->
  make:(key -> 'a -> 'v) ->
  'a ->
  'v
(** [find t prog mask ~same ~codes ~make x] is the plan of the key of [prog],
    [mask] and [codes x] in the calling domain's table of [t]. [same c x] is
    whether [c] is [codes x], and [make k x] the plan of [k], the same for every
    [x] whose codes are [k.codes]. Where the table has no plan of the key,
    [make] makes one, which is added. An exception [make] raises passes through
    [find], and nothing is added. [find] allocates nothing when [prog] is
    physically the last program it found with [mask] and [same] holds, and the
    functions it is given are closed. *)

val plain : 'v t -> P.t -> int -> make:(key -> 'v) -> 'v
(** [plain t prog mask ~make] is [find] of a key with no codes. *)
