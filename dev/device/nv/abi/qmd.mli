(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Launch descriptors.

    A launch runs from a descriptor in memory, a QMD, which the compute engine
    reads when the launch is scheduled ({!Method.schedule}): the kernel's code,
    its constant banks, the sizes of its grid and blocks, and what to do when it
    completes. Compute classes from Blackwell's on read version 5, earlier ones
    version 3.

    A descriptor is a {!Structure.t}: fields known when encoding are bytes, and
    each field a value fills is a hole. Setters return a new descriptor and
    leave their argument as it was. *)

type 'v t
(** The type for launch descriptors around values of type ['v]. *)

val make : Launch.t -> 'v t
(** [make l] is the descriptor of a launch of [l], without its sizes and
    addresses. *)

(** {1:sizes Sizes} *)

(** The type for the axes of a launch's grid and blocks. *)
type axis = X | Y | Z

(** The type for the sizes of a launch. *)
type dim =
  | Grid of axis  (** The grid's blocks along an axis. *)
  | Block of axis  (** A block's threads along an axis. *)

val set_dim : 'v t -> dim -> int -> 'v t
(** [set_dim q d n] is [q] with the size [d] of the launch [n].

    Raises [Invalid_argument] if [n] is negative or does not fit the field. *)

val patch_dim : 'v t -> dim -> 'v -> 'v t
(** [patch_dim q d v] is [q] with the size [d] of the launch [v], which fits the
    field. *)

(** {1:addresses Addresses} *)

val set_program : 'v t -> 'v -> 'v t
(** [set_program q addr] is [q] with the kernel's first instruction at [addr],
    256-byte aligned. *)

val set_bank : 'v t -> int -> 'v -> 'v t
(** [set_bank q i addr] is [q] with the constant bank [i] at [addr], 64-byte
    aligned.

    Raises [Invalid_argument] if [i] is not one of the banks of [q]'s launch
    ({!Launch.banks}). *)

val set_local_memory : 'v t -> 'v -> 'v t
(** [set_local_memory q bytes] is [q] with [bytes] of local memory for each
    thread, a multiple of 16 and at least the launch's {!Launch.local_bytes}:
    the share of each thread in the memory {!Method.local_memory} gives. *)

(** {1:completion Completion} *)

val release : 'v t -> 'v -> 'v -> 'v t option
(** [release q addr v] is [q] with a launch that writes the 64-bit [v] at
    [addr], 8-byte aligned, once the launch completes; or [None] if both of
    [q]'s two releases are taken.

    The write orders nothing but this launch: work the channel runs after the
    launch's scheduling may complete before it, and a launch scheduled later may
    run at the same time. A channel's own progress is written by
    {!Method.release}. *)

val release_stamp : 'v t -> 'v -> 'v -> 'v t option
(** [release_stamp q addr v] is {!release}, also writing the GPU's timer, in
    nanoseconds, into the 8 bytes at [addr + 8]. [addr] is 16-byte aligned. *)

val chain : 'v t -> 'v -> 'v t
(** [chain q addr] is [q] followed by the launch of the descriptor at [addr],
    256-byte aligned: scheduling [q] schedules it too, and it starts once [q]'s
    launch completes. *)

(** {1:layout Layout} *)

val structure : 'v t -> 'v Structure.t
(** [structure q] is [q] laid out in memory. *)
