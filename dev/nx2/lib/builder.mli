(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar programs of one node, kept per domain. *)

val single : Nx_kernel.Prog.node -> Nx_array.Dtype.any array -> Nx_kernel.Prog.t
(** [single n ins] is the program of the one node [n] over operands of dtypes
    [ins], [In i] being operand [i]. Each domain keeps the programs it made: a
    call equal to an earlier one on the domain makes none.

    Raises [Invalid_argument] as {!Nx_kernel.Prog.v} does. *)
