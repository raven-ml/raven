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

val loop : [< `Map | `Reduce | `Scan ] Nx_kernel.Spec.t -> string
(** [loop s] is [s] read by C through [nx_spec.h]'s [nx_spec_loop] and
    [nx_spec_pad]: [family f prog h], [h] its program's bytes in hex; a line
    [axes] followed by its axes; a line [reduction c k d] per reduction, its
    kind's code, output and dtype's code; then a line per load, [plain], or
    [padded r w f] followed by [lo], [hi], [interior] and each window's axis,
    size, step and dilation, [f] the fill's sixteen bytes in hex. *)

val axis_fields : [< `Gather | `Scatter | `Sort ] Nx_kernel.Spec.t -> int array
(** [axis_fields s] is [s] read by C through [nx_spec.h]'s [nx_spec_axis]:
    the family, the axis, the combine or a sort's direction, [unique] and
    [k]. *)

val shaped : [< `Assemble | `Fold ] Nx_kernel.Spec.t -> string
(** [shaped s] is [s] read by C through [nx_spec.h]'s [nx_spec_shaped]:
    [family f fill h], [h] the fill's bytes in hex; a line [shape] followed
    by its extents; a line [piece] per piece followed by each axis's start,
    count and step; then for a fold a line [pad r w] followed by [lo], [hi],
    [interior] and each window's axis, size, step and dilation. *)

val fft_fields : Nx_kernel.Spec.fft Nx_kernel.Spec.t -> int array
(** [fft_fields s] is [s] read by C through [nx_spec.h]'s [nx_spec_fft]: the
    family, the transform's code, the count of axes, [n], then each axis. *)

val linalg_fields : Nx_kernel.Spec.linalg Nx_kernel.Spec.t -> int array
(** [linalg_fields s] is [s] read by C through [nx_spec.h]'s
    [nx_spec_linalg]: the family, the routine's code, [upper], [factors],
    [vectors], [transpose] and [unit_diagonal]. *)
