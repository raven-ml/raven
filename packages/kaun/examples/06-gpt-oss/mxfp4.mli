(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The decode of an MXFP4 {!Nx_quant.t} as a composition of {!Nx} functions,
    which traces under [Rune.jit]. {!Nx_quant.dequant} is the same values; rune
    does not lower it yet, so the model's compiled products decode here.

    Every value a float32 holds is exact at float32 and bfloat16, subnormal ones
    included, where the device keeps subnormal numbers. Metal flushes them to
    zero: there a group whose scale byte is 0 is zero, and one whose scale byte
    is 1 loses its halves. The scale byte 255 is the format's NaN: its whole
    group is NaN. *)

val dequant : Nx_quant.t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t
(** [dequant w dtype] is the values of [w], of shape [Nx_quant.shape w]. *)

val dequant_rows :
  Nx_quant.t ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.dtype ->
  (float, 'b) Nx.t
(** [dequant_rows w ids dtype] is the values of the rows [ids] selects along the
    first axis, of [ids]'s shape followed by the rest of [Nx_quant.shape w]. The
    packed rows are gathered first, so only the selected rows are dequantised:
    with [w] of shape [[| experts; out; inputs |]] and [ids] of shape
    [[| tokens; k |]] the result has shape [[| tokens; k; out; inputs |]].

    Indices lie in \[[0], first extent), as for {!Nx.take}. *)
