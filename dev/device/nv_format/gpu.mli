(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPUs, as their formats depend on them.

    A GPU's launch descriptors follow its compute engine's class, its kernels'
    machine code its streaming multiprocessors' version, and its local memory
    the number of its parts. A driver reads them from the GPU at open: the
    classes from the GPU's class list, the rest from its graphics engine's
    information. *)

type t = {
  compute_class : int;
      (** The class of its compute engine, such as [0xc9c0] for Ada's. *)
  sass_version : int;
      (** The version of the machine code its multiprocessors run. *)
  gpcs : int;  (** Its graphics processing clusters. *)
  tpcs_per_gpc : int;  (** The texture processing clusters of one. *)
  sms_per_tpc : int;  (** The streaming multiprocessors of one of those. *)
  warps_per_sm : int;  (** The most warps a multiprocessor runs at once. *)
}
(** The type for GPUs. Every count is positive. *)
