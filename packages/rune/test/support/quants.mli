(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Quantised weights of every nx.quant format and the bound of their
    products. *)

type format = Q8_0 | Q6_K | Q4_K | Mxfp4 | Mxfp4_blocks

val formats : format list
(** [formats] is every format. *)

val name : format -> string
(** [name f] is [f]'s name, as ["q8_0"]. *)

val block_values : format -> int
(** [block_values f] is the values of a block of [f] along [k]. *)

val weight : format -> rows:int -> blocks:int -> Nx.host Nx_quant.t
(** [weight f ~rows ~blocks] is a weight of [rows] rows of [blocks] blocks of
    [f], on the host, its bytes a fixed pattern, its float16 fields of either
    sign in [[2^-9, 2^-4)] and its MXFP4 scale bytes in [[118, 136]], so that
    every value is finite. *)

val reference :
  (float, 'b, Nx.host) Nx.t ->
  Nx.host Nx_quant.t ->
  Nx.host Nx.float64_t * Nx.host Nx.float64_t
(** [reference x w] is the product of [x] with [w]'s values transposed,
    computed at float64, and how far {!Nx_quant.apply}'s result may lie from
    it at each element: {!Nx.contract}'s bound [γ(k + 1, 2u) Σ|x||v|] in its
    accumulator, float64's for a float64 [x] and float32's otherwise; the
    reference's own distance from the exact product, {!Nx.contract}'s bound
    for a float64 {!Nx.matmul}, [γ(k + 1, 2{^-52}) Σ|x||v|]; and, below
    float32, half an ulp of [x]'s dtype at the sum and a subnormal step. *)
