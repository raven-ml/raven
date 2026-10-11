(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What nx.kernel's suite needs beyond the library's interface. *)

val view_fields : Nx_kernel.Contract_view.t -> int array
(** [view_fields v] is [v] read by C through [nx_contract_view.h]'s
    [nx_contract_view]: its four extents and four offsets, then each
    operand's four strides, by operand then axis. *)

val prog : Nx_kernel.Prog.t -> string
(** [prog p] is [p] read by C through [nx_prog.h]'s [nx_prog], one line per
    node: its tag ([in], [coord], [const], [op1], [op2], [op3]), its kind's
    name in snake case or [-], its dtype's code, its three operand fields and
    its sixteen bytes of constant bits in hex; then a line [ins] with the
    operands' dtype codes and a line [outs] with the output nodes. *)

val tag : 'a -> int
(** [tag c] is the tag C reads of the constructor [c]: its index among its
    type's constant constructors, or among those with an argument. *)

val code : string -> int
(** [code name] is the code [nx_kernel.h] names [name].

    Raises [Invalid_argument] for a name it does not declare. *)
