(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What nx.kernel's suite needs beyond the library's interface. *)

val contract_fields : Nx_kernel.Spec.contract Nx_kernel.Spec.t -> int array
(** [contract_fields s] is [s] read by C through [nx_spec.h]'s
    [nx_spec_contract]: the family, [acc]'s and [out]'s codes, [init], the
    counts of batch and contracting pairs, then each pair's two axes, batch
    pairs first. *)

val view_fields : Nx_kernel.Spec.Contract_view.t -> int array
(** [view_fields v] is [v] read by C through [nx_spec.h]'s [nx_contract_view]:
    its four extents and four offsets, then each operand's four strides, by
    operand then axis. *)

val prog : Nx_kernel.Prog.t -> string
(** [prog p] is [p] read by C through [nx_spec.h]'s [nx_prog], one line per
    node: its tag ([in], [coord], [const], [op1], [op2], [op3]), its kind's
    name in snake case or [-], its dtype's code, its three operand fields and
    its sixteen bytes of constant bits in hex; then a line [ins] with the
    operands' dtype codes and a line [outs] with the output nodes. *)

val map : Nx_kernel.Spec.map Nx_kernel.Spec.t -> string
(** [map s] is [s] read by C through [nx_spec.h]'s [nx_spec_map] and
    [nx_spec_pad]: [family f prog h], [h] its program's bytes in hex, then a
    line per load, [plain], or [padded r w f] followed by [lo], [hi],
    [interior] and each window's axis, size, step and dilation, [f] the fill's
    sixteen bytes in hex. *)
