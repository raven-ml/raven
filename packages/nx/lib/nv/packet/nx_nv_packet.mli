(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** NVIDIA GPU command packets: the methods a channel runs, and the launch
    descriptors (QMDs) its compute engine reads.

    A channel runs segments of 32-bit words: a header names an engine's
    subchannel, a method and how many words follow, which the methods from that
    one on take in turn. A launch runs from a launch descriptor in memory, whose
    layout depends on the compute engine's class.

    Packets are words around values of the caller's type: integers for a runtime
    that writes them now, or values computed later for a library that encodes
    work ahead of time. The encoders compute on values only through {!VALUE}, so
    an encoding evaluated at given integers is the encoding of those integers.
*)

(** {1:values Values} *)

(** The operations the encoders apply to values. *)
module type VALUE = sig
  type t
  (** The type for values: 64-bit unsigned integers, or what stands for them. *)

  val add : t -> int -> t
  (** [add v n] is [v + n]. *)

  val shift_right : t -> int -> t
  (** [shift_right v n] is [v] shifted right by [n] bits. *)
end

module Int : VALUE with type t = int
(** Values known now, as OCaml integers. *)

(** The type for the words of a packet. *)
type 'v word =
  | Dword of int  (** A word known when encoding: the integer's low 32 bits. *)
  | W32 of 'v  (** A value's low 32 bits. *)
  | W64 of 'v  (** A value's 64 bits, as two words, low first. *)

val dwords : int word list -> int list
(** [dwords ws] is the 32-bit words of [ws]. *)

(** {1:methods Methods} *)

(** The type for the subchannels of a channel, each bound to an engine. *)
type subchannel =
  | Host  (** The channel itself: semaphores and interrupts. *)
  | Compute  (** The compute engine. *)
  | Copy  (** The copy engine. *)

(** Methods. A semaphore of the host is a 64-bit word; the copy engine's are
    32-bit words. Addresses are of memory the GPU addresses. *)
module Methods (V : VALUE) : sig
  val acquire : V.t -> V.t -> V.t word list
  (** [acquire addr v] waits until the semaphore at [addr] is at least [v],
      compared circularly. *)

  val release : V.t -> V.t -> V.t word list
  (** [release addr v] writes [v] at [addr] once the channel's earlier work is
      done, then raises a non-stalling interrupt. *)

  val release_stamp : V.t -> V.t -> V.t word list
  (** [release_stamp addr v] writes [v] at [addr] and the GPU's timer, in
      nanoseconds, at [addr + 8], once the channel's earlier work is done. *)

  val set_object : subchannel -> int -> V.t word list
  (** [set_object s cls] binds the engine of class [cls] to [s]. *)

  val local_memory_window : V.t -> V.t word list
  (** [local_memory_window addr] makes kernels' local memory appear at [addr].
  *)

  val shared_memory_window : V.t -> V.t word list
  (** [shared_memory_window addr] makes kernels' shared memory appear at [addr].
  *)

  val local_memory : V.t -> per_tpc:V.t -> V.t word list
  (** [local_memory addr ~per_tpc] gives kernels the local memory at [addr],
      [per_tpc] bytes for each texture processing cluster, on every streaming
      multiprocessor. *)

  val invalidate_caches : V.t word list
  (** [invalidate_caches] invalidates the compute engine's instruction, data and
      constant caches, without waiting for its work to finish. *)

  val schedule : V.t -> V.t word list
  (** [schedule addr] schedules the launch descriptor at [addr], 256-byte
      aligned, and the descriptors that depend on it ({!Qmd.chain}). *)

  val copy : dst:V.t -> src:V.t -> int -> V.t word list
  (** [copy ~dst ~src n] copies [n] bytes from [src] to [dst] on the copy
      engine, in lines of at most 2 GiB. *)

  val copy_release : V.t -> V.t -> V.t word list
  (** [copy_release addr v] writes [v]'s low 32 bits at [addr] on the copy
      engine, once its earlier copies are done. *)

  val copy_stamp : V.t -> V.t word list
  (** [copy_stamp addr] writes [0] into the 8 bytes at [addr] and the GPU's
      timer, in nanoseconds, into the 8 after, on the copy engine. *)
end

(** {1:gpfifo Channel rings} *)

(** A channel's ring (its GPFIFO) holds 64-bit entries, each naming a segment of
    words. *)
module Gpfifo (V : VALUE) : sig
  val max_words : int
  (** [max_words] is the most words an entry's segment holds, [2{^21} - 1]. *)

  val entry : V.t -> offset:int -> words:int -> V.t
  (** [entry addr ~offset ~words] is the entry of the segment of [words] words
      at [addr + offset], 4-byte aligned and below [2{^40}], which the channel
      runs as a subroutine.

      Raises [Invalid_argument] if [words] is negative or more than
      {!max_words}. *)
end

(** {1:launches Launches} *)

(** What a launch of a kernel needs on a GPU: the layout of its descriptor and
    its constant bank 0, and the GPU's limits for it. *)
module Program : sig
  type t
  (** The type for kernels as a GPU launches them. *)

  val make :
    compute_class:int ->
    sass_version:int ->
    shared_window:int ->
    local_window:int ->
    Nx_nv_cubin.kernel ->
    (t, string) result
  (** [make ~compute_class ~sass_version ~shared_window ~local_window k] is the
      kernel [k] launched by the compute engine of class [compute_class], whose
      streaming multiprocessors run machine code of version [sass_version] and
      show kernels their shared and local memory at the addresses
      [shared_window] and [local_window]. It is [Error] if [k] declares more
      shared memory than a launch configures, 100 KiB with the 1 KiB the driver
      reserves. *)

  val code : t -> int
  (** [code p] is the offset of the kernel's first instruction in its cubin's
      image. *)

  val banks : t -> Nx_nv_cubin.bank list
  (** [banks p] is the constant banks a launch addresses: the cubin's, with a
      bank [0] of 352 bytes at offset [0] if it has none. A launch writes bank 0
      anew: {!driver_parameters}, then the kernel's parameters. *)

  val driver_parameters : t -> string
  (** [driver_parameters p] is the start of constant bank 0, the driver's
      parameters: the shared and local memory windows, and the stack limit. The
      kernel's parameters follow it. *)

  val local_bytes : t -> int
  (** [local_bytes p] is the local memory each thread of a launch needs: the
      kernel's stack and the 576 bytes the driver reserves. *)

  val max_threads : t -> int
  (** [max_threads p] is the most threads a block may have for the registers its
      threads use. *)
end

type 'v hole = {
  at : int;  (** The byte offset of the hole. *)
  bytes : int;  (** Its size: 1, 2, 4 or 8. *)
  value : 'v;  (** The value whose low bytes fill it, little-endian. *)
}
(** The type for words of a structure that values fill. *)

type 'v structure = {
  bytes : string;
      (** The structure's bytes, of which its holes replace those they cover. *)
  holes : 'v hole list;  (** Its holes, in increasing offsets. *)
}
(** The type for structures in memory around values. *)

val fill : int structure -> string
(** [fill s] is the bytes of [s], its holes filled. *)

(** The type for the axes of a launch's grid and blocks. *)
type axis = X | Y | Z

(** The type for the sizes of a launch. *)
type dim =
  | Grid of axis  (** The grid's blocks along an axis. *)
  | Block of axis  (** A block's threads along an axis. *)

(** Launch descriptors: version 5 for the compute classes from Blackwell's on,
    version 3 before. A descriptor is mutable; its fields are known integers, or
    holes for values. *)
module Qmd (V : VALUE) : sig
  type t
  (** The type for launch descriptors being encoded. *)

  val make : Program.t -> t
  (** [make p] is the descriptor of a launch of [p], without its sizes and
      addresses. *)

  val copy : t -> t
  (** [copy q] is a descriptor with [q]'s fields, which changes independently of
      [q]. *)

  val set_dim : t -> dim -> int -> unit
  (** [set_dim q d n] sets the size [d] of the launch to [n].

      Raises [Invalid_argument] if [n] does not fit the field. *)

  val patch_dim : t -> dim -> V.t -> unit
  (** [patch_dim q d v] sets the size [d] of the launch to [v]. *)

  val set_program : t -> V.t -> unit
  (** [set_program q addr] sets the address of the kernel's first instruction to
      [addr], 256-byte aligned. *)

  val set_bank : t -> int -> V.t -> unit
  (** [set_bank q i addr] sets the address of the constant bank [i] to [addr],
      64-byte aligned. *)

  val set_local_memory : t -> V.t -> unit
  (** [set_local_memory q bytes] sets the local memory each thread has to
      [bytes], a multiple of 16. *)

  val release : t -> V.t -> V.t -> bool
  (** [release q addr v] makes the launch write the 64-bit [v] at [addr] once it
      completes, and is [true], if one of the descriptor's two releases is free,
      and is [false] otherwise. *)

  val release_stamp : t -> V.t -> V.t -> bool
  (** [release_stamp q addr v] is {!release}, also writing the GPU's timer at
      [addr + 8]. *)

  val chain : t -> V.t -> unit
  (** [chain q addr] makes the launch of the descriptor at [addr], 256-byte
      aligned, start once [q]'s completes, scheduled with [q]. *)

  val structure : t -> V.t structure
  (** [structure q] is [q] laid out. Its holes are the widest unsigned words
      within their fields. *)
end
