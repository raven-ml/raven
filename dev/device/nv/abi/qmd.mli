(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Launch descriptors.

    A launch runs from a descriptor in memory, a QMD, which the compute engine
    reads when the launch is scheduled ({!Method.schedule}): the kernel's code,
    its constant banks, the sizes of its grid and blocks, and what to do when it
    completes. Blackwell's compute class reads version 5, the others version 3
    ({!Gpu.t}).

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

val max_size : dim -> int
(** [max_size d] is the largest size [d] takes: [2{^31}-1] blocks along a grid's
    [X], [65535] along [Y] and [Z]; [1024] threads along a block's [X] and [Y],
    [64] along [Z]. *)

val set_dim : dim -> int -> 'v t -> 'v t
(** [set_dim d n q] is [q] with the size [d] of the launch [n], in place of a
    value {!patch_dim} gave it.

    Raises [Invalid_argument] if [n] is outside \[[0];[max_size d]\]. *)

val patch_dim : dim -> 'v -> 'v t -> 'v t
(** [patch_dim d v q] is [q] with the size [d] of the launch [v], which is at
    most [max_size d], in place of a size {!set_dim} gave it: a hole of the
    structure ({!structure}). *)

(** {1:addresses Addresses} *)

val set_program : 'v -> 'v t -> 'v t
(** [set_program addr q] is [q] with the kernel's first instruction at [addr],
    256-byte aligned. *)

val set_bank : int -> 'v -> 'v t -> 'v t
(** [set_bank i addr q] is [q] with the constant bank [i] at [addr], 64-byte
    aligned.

    Raises [Invalid_argument] if [i] is not one of the banks of [q]'s launch
    ({!Launch.banks}). *)

val set_local_memory : 'v -> 'v t -> 'v t
(** [set_local_memory bytes q] is [q] with [bytes] of local memory for each
    thread, a multiple of 16 and at least the launch's {!Launch.local_bytes}:
    the share of each thread in the memory {!Method.local_memory} gives. *)

(** {1:completion Completion} *)

val release : Packet.scope -> 'v -> 'v -> 'v t -> 'v t option
(** [release s addr v q] is [q] with a launch that writes the 64-bit [v] at
    [addr], 8-byte aligned, once the launch completes, so that readers of scope
    [s] who see [v] see the launch's writes; or [None] if both of [q]'s two
    releases are taken.

    The write orders nothing but this launch: work the channel runs after the
    launch's scheduling may complete before it, and a launch scheduled later may
    run at the same time. A channel's own progress is written by
    {!Method.release}. *)

val release_stamp : Packet.scope -> 'v -> 'v -> 'v t -> 'v t option
(** [release_stamp s addr v q] is [release s addr v q], also writing the GPU's
    timer, in nanoseconds, into the 8 bytes at [addr + 8]. [addr] is 16-byte
    aligned. *)

val chain : 'v -> 'v t -> 'v t
(** [chain addr q] is [q] followed by the launch of the descriptor at [addr],
    256-byte aligned: scheduling [q] schedules it too, and it starts once [q]'s
    launch completes. *)

(** {1:layout Layout} *)

val structure : 'v t -> 'v Structure.t
(** [structure q] is [q] laid out in memory. *)
