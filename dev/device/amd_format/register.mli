(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The registers of a GPU's GC.

    A register is a 32-bit word of fields, at an offset in one of its block's
    segments. Where a segment starts depends on the GPU: PM4 packets address a
    register from the base its GC's generation gives the segment ({!address}),
    and a driver that programs the block directly from the bases the GPU's
    discovery table lists.

    The registers are those of the latest GC version at or before the GPU's of
    its major version, as the GC's headers define them: those that compute
    queues, thread traces, performance counters and bringing the GPU up read and
    write. *)

type t = {
  name : string;  (** Its name in the GC's headers, as ["regGRBM_GFX_INDEX"]. *)
  offset : int;  (** Its offset in its segment, in 32-bit words. *)
  segment : int;  (** Its segment. *)
  fields : (string * (int * int)) list;
      (** Its fields by name, each as its lowest and highest bit. *)
}
(** The type for registers. *)

val registers : Gpu.t -> t list
(** [registers g] is the registers of [g]'s GC. It is [[]] for a GC whose major
    version has no version at or before [g.gc] with definitions: the GPU's
    queues cannot be encoded. *)

val find : Gpu.t -> string -> t option
(** [find g name] is the register of {!registers}[ g] named [name], if any. *)

val address : Gpu.t -> t -> int
(** [address g r] is [r]'s address in the register space PM4 packets address, in
    32-bit words: its offset from the base [g]'s GC generation gives its
    segment.

    Raises [Invalid_argument] if that generation gives [r]'s segment no base. *)

val encode : t -> (string * int) list -> int
(** [encode r fs] is the word of [r] with each field of [fs] set to its value,
    cut to the field's width, and its other bits zero.

    Raises [Invalid_argument] if [r] has no field of one of [fs]. *)
