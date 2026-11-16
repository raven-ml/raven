(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** MXFP4 products, eager and compiled, against their exact value. *)

type t = {
  dtype : Nx.Dtype.any;  (** [x]'s: [float32], [bfloat16] or [float16]. *)
  m : int;  (** [x]'s rows. *)
  k : int;  (** The inputs, a multiple of 32. *)
  n : int;  (** The outputs, at least 10. *)
  seed : int;  (** [x]'s draw. *)
}
(** The type for cases: [Nx_quant.apply w x] of an MXFP4 weight [w] of [n] by
    [k] and an [x] of [m] by [k]. Output row [r] has the scale byte
    [r mod 10]-th of 0, 1, 2, 126, 127, 128, 252, 253, 254 and 255 in every
    block, and each block of 32 of its values holds each of the 16 codes
    twice, or at bytes 253 and 254, where the larger codes overflow, each
    positive code four times. [x]'s elements lie in [\[0, 2{^-8}\]], so that
    a row's infinities keep one sign and no sum of finite products overflows
    [float32]. *)

val pp : Format.formatter -> t -> unit

val cases : t list
(** [cases] holds each dtype at extents that 16 divides, as a tensor core
    takes them, and at extents it does not divide, [m] of 1 among them. *)

val law : ?flushes:bool -> (module Nx.Devices) -> t -> unit
(** [law ~flushes s c] checks that [Nx_quant.apply w x], eagerly on the host
    and compiled on [s]'s first device with no call evaluated, is within
    {!Nx.contract}'s bound of the exact product of [x] by
    [Nx_quant.dequant Nx.float32 w] transposed, computed in [float64]:
    [γ(K + 1, 2u) Σ|x w|] for [K = k] and float32's [u], plus [K 2{^-149}]
    for products below [float32]'s least normal, then rounded once to [x]'s
    dtype.
    A NaN of the exact product is a NaN, and an infinity that infinity.

    With [~flushes:true] (default [false]), for a device that flushes
    subnormals to zero, the compiled product is checked only in the outputs
    of weight rows of scale bytes 126 and above, where no value it computes
    is subnormal. {!Nx.contract}'s bound excludes underflow, which a row of a
    smaller byte holds. *)

(** {1:decodes Decodes} *)

type decode = {
  tokens : int;  (** [x]'s rows. *)
  experts : bool;
      (** Whether the product runs through [Nx_quant.apply_experts], each
          token's id choosing one of two equal experts, or [Nx_quant.apply]. *)
}
(** The type for decode cases: an MXFP4 weight of [112] rows by [256]
    inputs, row [16 b + c] holding the code [c] at each input and the [b]-th
    of the scale bytes 0, 1, 126, 127, 128, 254 and 255 in each block, by
    [bfloat16] rows of [2{^-8}]. *)

val pp_decode : Format.formatter -> decode -> unit

val decodes : decode list
(** [decodes] holds 1 and 64 tokens, each through [apply] and through
    [apply_experts]. *)

val decode_law : ?flushes:bool -> (module Nx.Devices) -> decode -> unit
(** [decode_law ~flushes s d] checks that the product compiled on [s]'s first
    device gives, in every token's output [16 b + c], the [float32] value of
    code [c] at its scale byte, rounded to [bfloat16], bit for bit: [+0] for
    both zeros, an infinity where the value passes [float32]'s range, a NaN
    at byte 255. Its [256] terms are equal, so the sum is exact.

    With [~flushes:true] (default [false]) only the rows of bytes 126 and
    above are checked, whose values are normal. *)
