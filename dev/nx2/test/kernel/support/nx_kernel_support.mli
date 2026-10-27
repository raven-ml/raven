(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What nx.kernel's suite needs beyond the library's interface. *)

val prog : Nx_kernel.Prog.t -> string
(** [prog p] is [p] read by C through [nx_kernel.h]'s codes, one line per node:
    its tag ([in], [coord], [const], [op1], [op2], [op3]), then its fields in
    order, a kind by its name in snake case, a dtype by its code and a
    constant's bits in hex; then a line [ins] with the operands' dtype codes
    and a line [outs] with the output nodes. *)

val op1 : Nx_kernel.Prog.op1 -> int
(** [op1 k] is the code C reads of [k] through [nx_kernel.h]'s [NX_OP1_CODE]. *)

val op2 : Nx_kernel.Prog.op2 -> int
(** [op2 k] is the code C reads of [k] through [nx_kernel.h]'s [NX_OP2_CODE]. *)

val tag : 'a -> int
(** [tag c] is the tag C reads of the constructor [c]: its index among its
    type's constant constructors, or among those with an argument. *)

val code : string -> int
(** [code name] is the code [nx_kernel.h] names [name].

    Raises [Invalid_argument] for a name it does not declare. *)
