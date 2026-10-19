(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The complex kinds on [complex64] and [complex128] numbers, as nx's kernels
    compute them.

    A number is a pair of nodes of one shape, both [float32] or both [float64]:
    its real part, then its imaginary part. Each function computes its kind by
    the algorithm nx's kinds write for its compute type ([nx_kinds.h] and
    [nx_kinds_complex.h] of {!Nx_kernel}), from {!Lower_real}'s float kinds: the
    same operations in the same order, with a fused multiply-add where they
    write one and nowhere else, so that a node gives the bits nx's kernels give
    on a target that rounds as those headers assume. Every NaN part of a result
    but [Neg]'s, [Add]'s, [Sub]'s and an extreme's is the quiet NaN of a clear
    sign bit. *)

open Tolk

val unary : Nx_kernel.Prog.unary -> Ops.t * Ops.t -> Ops.t * Ops.t
(** [unary k z] is [k] of each element of [z].

    Raises [Invalid_argument] for a kind complex numbers do not take. *)

val binary :
  Nx_kernel.Prog.binary -> Ops.t * Ops.t -> Ops.t * Ops.t -> Ops.t * Ops.t
(** [binary k a b] is [k] of the elements of [a] and [b].

    Raises [Invalid_argument] for a kind complex numbers do not take. *)

val compare : Nx_kernel.Prog.compare -> Ops.t * Ops.t -> Ops.t * Ops.t -> Ops.t
(** [compare k a b] is the boolean comparison [k] of the elements of [a] and
    [b]: by real part, then imaginary part, false where a part of either is a
    NaN but for [Not_equal]. *)
