(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Attention as other operations.

    Rune's differentiation gives {!Nx.Prim.Attention} this meaning: two
    contractions, a map of the pair program, a log-sum-exp and a map for
    the weights, each an operation it already differentiates. A compiled
    call lowers the operation whole ({!Lower_attention}). It stores three arrays of [[lead…; hq; tq; tk]] in the
    accumulator; with lengths it copies the values with the keys
    past each row's length zeroed, and paged keys and values are gathered
    into dense ones. A NaN or an infinity at a key hidden from a row within
    its length makes the row's [o] NaN. *)

type eval = { eval : 'r. 'r Nx.Prim.t -> 'r } [@@unboxed]
(** How the composition's operations are given their meaning: by
    {!Nx.Prim.eval} where a differentiation's code runs. *)

val compose :
  eval ->
  prog:Nx_kernel.Prog.t ->
  scale:float ->
  acc:(float, 'a) Nx.dtype ->
  out:('w, 'r) Nx.dtype ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.kv ->
  'd Nx.int64_t option ->
  'd Nx.Prim.any iarray ->
  ('w, 'r, 'd) Nx.t * (float, 'a, 'd) Nx.t
(** [compose e ~prog ~scale ~acc ~out q kv lengths loads] is the attention
    {!Nx.Prim.Attention} of these fields states, [(o, lse)], by contractions,
    maps, a log-sum-exp, movements and, for paged keys, gathers, each given
    its meaning by [e]. *)
