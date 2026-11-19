(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Integers and booleans cast to floats and complex numbers, compiled on a
    set's first device, against eager's on the host. nx_kernel's [Prog] states a
    cast of an integer as its exact value rounded once, to nearest with ties to
    even. *)

type t =
  | Case : {
      src : ('v, 's) Nx.dtype;  (** An integer or boolean dtype. *)
      dst : Nx.Dtype.any;  (** A float or complex dtype. *)
      xs : int64 array;
          (** Each taken modulo [src]'s width, or as [true] where not [0]. *)
    }
      -> t  (** The type for cases: [xs] cast to [dst]. *)

val pp : Format.formatter -> t -> unit

val gen : t Windtrap.Gen.t
(** [gen] draws each integer or boolean source and each destination, with 64
    elements, each one of the source's extremes, their neighbours, a sum of up
    to three small integers shifted anywhere, of either sign, or any value. *)

val examples : t list
(** [examples] holds, for each source of 32 or 64 bits and each destination, and
    for each narrower source and the 16-bit floats and [float32], the extremes,
    their neighbours, [-1] to [1], [2{^24}] and [2{^53}] and their neighbours,
    and at each exponent the destination's ties, at one bit past its precision,
    and their neighbours, of both signs. *)

val dst64 : t -> bool
(** [dst64 c] is [true] iff [c]'s destination has [float64] parts. *)

val law : (module Nx.Devices) -> t -> unit
(** [law s c] checks that on [s]'s first device the compiled cast gives eager's
    bits, with no call evaluated. It covers each source of 32 or 64 bits, each
    destination of [float32] or narrower parts, a negative element, one that
    [float32] does not hold, one past [2{^53}] that [float64] does not hold, and
    a destination's tie with a bit below [float32]'s precision, which a cast
    that rounds twice gets wrong. *)

(** {1:bf16 Every integer to bfloat16} *)

val to_bf16_sources : Nx.Dtype.any list
(** [to_bf16_sources] is every integer and boolean dtype. *)

val to_bf16 : (module Nx.Devices) -> Nx.Dtype.any -> unit
(** [to_bf16 s src] checks that on [s]'s first device the compiled cast of
    [src] to [bfloat16] gives eager's bits, with no call evaluated: at every
    value of a [src] of at most 16 bits, and for a wider one at the 4096
    integers around [0], [2{^w-1}] for its width [w], and, for each power
    [2{^e}] from [2{^8}] that [src] holds, around [2{^e}] and three of
    [bfloat16]'s ties above it, [2{^e} + 2{^e-8}], which rounds down,
    [2{^e} + 3·2{^e-8}], which rounds up, and [2{^e+1} - 2{^e-8}], which
    rounds up to [2{^e+1}]; each of these negated too, the whole taken
    modulo [2{^w}]. For [int32] it checks too the cast of these integers'
    low 24 bits, which the compiled call computes, an integer within
    [2{^24}]. *)
