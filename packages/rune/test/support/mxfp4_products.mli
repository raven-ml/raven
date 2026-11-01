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

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks that [Nx_quant.apply w x], eagerly on the host and
    compiled on [s]'s first device with no call evaluated, is within
    {!Nx.contract}'s bound of the exact product of [x] by
    [Nx_quant.dequant Nx.float32 w] transposed, computed in [float64]:
    [γ(K + 1, 2u) Σ|x w|] for [K = k] and float32's [u], plus [K 2{^-149}]
    for products below [float32]'s least normal, then rounded once to [x]'s
    dtype.
    A NaN of the exact product is a NaN, and an infinity that infinity. *)
