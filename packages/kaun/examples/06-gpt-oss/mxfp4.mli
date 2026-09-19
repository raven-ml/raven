(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** MXFP4 weights (OCP Microscaling Formats, 2023), as gpt-oss checkpoints store
    them.

    A packed weight is two uint8 tensors. [blocks] has shape
    [[| ...; groups; 16 |]]: each byte holds two 4-bit codes, the low nibble
    first, so a group of 16 bytes is 32 consecutive values. [scales] has shape
    [[| ...; groups |]]: one biased power-of-two exponent per group. A code is a
    sign bit over a 3-bit magnitude naming one of 0, 0.5, 1, 1.5, 2, 3, 4 and 6,
    and a value is its code's number times [2 ^ (scale - 127)].

    Dequantisation is a composition of {!Nx} functions: it runs eagerly and
    traces under [Rune.jit]. *)

type blocks = (int, Nx.uint8_elt) Nx.t
(** The type for packed codes, of shape [[| ...; groups; 16 |]]. *)

type scales = (int, Nx.uint8_elt) Nx.t
(** The type for group exponents, of shape [[| ...; groups |]]. *)

val dequant : blocks -> scales -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t
(** [dequant blocks scales dtype] is the values of a packed weight, of shape
    [[| ...; groups * 32 |]].

    Every value a float32 holds is exact at float32 and bfloat16, subnormal ones
    included, where the device keeps subnormal numbers. Metal flushes them to
    zero: there a group whose scale byte is 0 is zero, and one whose scale byte
    is 1 loses its halves. The scale byte 255 is the format's NaN: its whole
    group is NaN.

    Raises [Invalid_argument] if [blocks] does not have shape
    [[| ...; groups; 16 |]] or if [scales] does not have [blocks]'s shape
    without its last axis. *)

val dequant_rows :
  blocks ->
  scales ->
  (int32, Nx.int32_elt) Nx.t ->
  (float, 'b) Nx.dtype ->
  (float, 'b) Nx.t
(** [dequant_rows blocks scales ids dtype] is the values of the rows [ids]
    selects along the first axis, of [ids]'s shape followed by
    [[| ...; groups * 32 |]]. The packed rows are gathered first, so only the
    selected rows are dequantised: with [blocks] of shape
    [[| experts; out; groups; 16 |]] and [ids] of shape [[| tokens; k |]] the
    result has shape [[| tokens; k; out; groups * 32 |]].

    Indices lie in \[[0], first extent), as for {!Nx.take}. *)
