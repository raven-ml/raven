(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The float kinds on [float32] and [float64] nodes, as nx's kernels compute
    them.

    Each function computes its kind by the algorithm nx's kinds write for its
    compute type ([nx_kinds.h] and [nx_kinds_real.h] of {!Nx_kernel}): the same
    operations in the same order, with a fused multiply-add where they write one
    and nowhere else, so that a node gives the bits nx's kernels give on a
    target that rounds as those headers assume, NaN operands included. *)

open Tolk

val unary : Nx_kernel.Prog.unary -> Ops.t -> Ops.t
(** [unary k x] is [k] of each element of the [float32] or [float64] node [x].
*)

val binary : Nx_kernel.Prog.binary -> Ops.t -> Ops.t -> Ops.t
(** [binary k x y] is [k] of the elements of [x] and [y], two nodes of one float
    type. Raises [Invalid_argument] for a kind floats do not take. *)

val compare : Nx_kernel.Prog.compare -> Ops.t -> Ops.t -> Ops.t
(** [compare k x y] is the boolean comparison [k] of the elements of [x] and
    [y]. *)

val fma : Ops.t -> Ops.t -> Ops.t -> Ops.t
(** [fma a b c] is [a * b + c] rounded once. *)
