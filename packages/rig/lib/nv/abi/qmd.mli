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

    A descriptor is made from a launch ({!Launch.t}) and its values ({!t}) as a
    {!Template.t}: a value known when encoding is in its bytes, and each slot a
    patch.

    {b References.} NVIDIA's descriptor headers in open-gpu-doc: [clc7c0qmd.h]
    (version 3) and [clcec0qmd.h] (version 5). *)

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
    [64] along [Z]. The caller also keeps a block's threads, the product of its
    three sizes, at most {!Launch.max_threads}. *)

(** {1:descriptors Descriptors} *)

(** The type for what a launch writes when it completes: the 64-bit value at an
    address, 8-byte aligned, so that any reader who sees the value, the host and
    other devices included, sees the launch's writes. The write orders nothing
    but this launch: work the channel runs after the launch's scheduling may
    complete before it, and a launch scheduled later may run at the same time. A
    channel's own progress is written by {!Method.release}. *)
type release =
  | Release of Template.operand * Template.operand
      (** [Release (addr, v)] writes [v] at [addr]. *)
  | Stamped of Template.operand * Template.operand
      (** [Stamped (addr, v)] also writes a stamp at [addr + 8], [addr] 16-byte
          aligned ({!Method.section-host}). *)

type t = {
  program : Template.operand;
      (** The address of the kernel's first instruction, 256-byte aligned. *)
  local : Template.operand;
      (** The bytes of local memory each thread takes of the memory
          {!Method.local_memory} gives: a multiple of 16, at least the launch's
          {!Launch.local_bytes}, such as the {!Local_memory.field-per_thread}
          the memory was sized with. *)
  banks : (int * Template.operand) list;
      (** The addresses of the launch's constant banks, 64-byte aligned, by
          index ({!Launch.banks}). A bank left out is at [0]. *)
  grid : Template.operand * Template.operand * Template.operand;
      (** The grid's blocks along [X], [Y] and [Z], each at most its
          {!max_size}. *)
  block : Template.operand * Template.operand * Template.operand;
      (** A block's threads along [X], [Y] and [Z], each at most its
          {!max_size}. *)
  shared : Template.operand option;
      (** [Some v] if each block takes [v] bytes of dynamic shared memory beside
          its own: a multiple of 128, at most {!Launch.dynamic_shared}. The
          multiprocessors then run the launch in their configuration of the most
          shared memory, which holds any such [v]. *)
  chain : Template.operand option;
      (** [Some addr] if the launch of the descriptor at [addr], 256-byte
          aligned, follows: scheduling this one schedules it too, and it starts
          once this launch completes. *)
  releases : release list;  (** The launch's releases, in order, at most two. *)
}
(** The type for a launch's values, as its descriptor takes them. *)

val make : Launch.t -> t -> Template.t
(** [make l d] is the descriptor of a launch of [l] with the values [d]: 256
    bytes for version 3 and 384 for version 5.

    Raises [Invalid_argument] if a bank of [d] is not one of [l]'s, if [d] has
    more than two releases, or if a known value is negative or does not fit its
    field. *)

(** {1:parameters The driver's parameters} *)

val parameters : Launch.t -> t -> Template.t
(** [parameters l d] is the start of constant bank [0] for the launch of [l]
    with the values [d], which the caller writes before the kernel's parameters.
    Its length is the kernel's [params_offset], or the length of the class's
    parameters if that is longer. It holds the sizes of [d]'s block and grid
    along [X], [Y] and [Z], as six unsigned 32-bit words where CUDA's code reads
    [blockDim] and [gridDim]; the GPU's shared and local memory windows
    ({!Gpu.t}) and the stack limit; the address CUDA uses for grid-constant
    parameters, as 64 bits; and zeros elsewhere. That address is bank [0]'s on
    Ampere and Ada, and bank [0]'s plus the kernel's [params_offset] on
    Blackwell, or zero if [d] leaves bank [0] out.

    Raises [Invalid_argument] if a known size is negative or does not fit 32
    bits. *)
