(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A range gather ({!Nx.Prim.Gather_ranges}) as a gather at the positions
    its runs read, each operation through a given evaluator. *)

type eval = { eval : 'r. 'r Nx.Prim.t -> 'r } [@@unboxed]
(** The type for evaluators of operations. *)

val positions :
  eval -> size:int -> 'd Nx.int64_t -> 'd Nx.int64_t -> 'd Nx.int64_t
(** [positions e ~size starts ends] is, for [starts] and [ends] of
    [[lead…; k]], the [[lead…; size]] positions along the ranged axis that a
    range gather of [size] elements reads: [starts.{j} + p - ends.{j - 1}]
    in [int64], modulo [2{^64}], at a position [p] of run [j], and [-1] at
    one past every end. Each operation runs through [e]. *)

val compose :
  eval ->
  size:int ->
  'd Nx.int64_t ->
  'd Nx.int64_t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [compose e ~size starts ends x] is the range gather of [x] as a
    {!Nx.Prim.Gather} at {!positions}, broadcast over the cells: its bits,
    each operation run through [e]. *)
