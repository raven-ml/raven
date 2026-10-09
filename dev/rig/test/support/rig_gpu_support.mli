(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the GPU suites and benches share: host memory by address, and a GPU
    of one driver opened through rig. *)

(** {1:host Host memory} *)

(** Host memory the suites own, by address. *)
module Host : sig
  val page : int
  (** [page] is the host's page size, in bytes. *)

  val pages : ?read_only:bool -> int -> int
  (** [pages n] is the address of [n] zeroed bytes from a page boundary,
      read-only iff [read_only] (defaults to [false]). *)

  val free_pages : int -> int -> unit
  (** [free_pages a n] gives back the [n] bytes {!pages} gave at [a]. *)

  val get8 : int -> int
  (** [get8 a] is the byte at [a]. *)

  val set8 : int -> int -> unit
  (** [set8 a x] stores [x] as the byte at [a]. *)

  val get32 : int -> int
  (** [get32 a] is the unsigned 32-bit word at [a]. *)

  val set32 : int -> int -> unit
  (** [set32 a x] stores [x] as the 32-bit word at [a]. *)

  val get64 : int -> int
  (** [get64 a] is the 64-bit word at [a], read with acquire order. *)

  val set64 : int -> int -> unit
  (** [set64 a x] stores [x] in the 64-bit word at [a] with release order. *)

  val read : int -> int -> string
  (** [read a n] is the [n] bytes at [a]. *)

  val write : int -> string -> unit
  (** [write a s] stores [s] at [a]. *)
end

val still :
  ?msg:string -> 'a Windtrap.testable -> 'a -> (unit -> 'a) -> ms:int -> unit
(** [still w x f ~ms] reads [f ()] for [ms] milliseconds of the monotonic
    clock ({!Rig.Profile.now}), the clock a device's hang bound counts, and
    fails the test if a read is ever other than [x] under [w]. *)

(** {1:gpu A GPU} *)

(** The GPU a suite acts on. *)
module type Gpu = sig
  module D : Rig.Driver

  val class_ : string
  (** [class_] names the GPU, as ["CUDA"]: the skip's reason and rig's name of
      the device. *)

  val present : unit -> bool
  (** [present ()] is [true] iff the machine has the GPU, from its files alone:
      it starts no vendor library. *)

  val open_ : unit -> (D.t, string) result
  (** [open_ ()] opens the driver's device of the GPU. *)
end

(** A GPU's device, opened through rig or by its driver alone. *)
module type S = sig
  type gpu
  (** The type for the driver's devices. *)

  type t = { d : Rig.t; g : gpu }
  (** The type for an open GPU: [d] as programs reach it, [g] its driver's
      device, which rig owns. *)

  val class_ : string
  (** [class_] names the GPU, as ["CUDA"]. *)

  val present : unit -> bool
  (** [present ()] is [true] iff the machine has the GPU, from its files alone:
      it starts no vendor library. *)

  val hold : unit -> unit
  (** [hold ()] is {!Rig_gpu_lock.hold} if {!present}[ ()]. A suite calls it
      before [Windtrap.run], so that the wait counts against no test's
      timeout. *)

  val open_ : unit -> t
  (** [open_ ()] is the GPU opened by its driver and handed to rig under a
      name of the GPU's, while the process holds the machine's GPU lock. It
      first ends what the last [open_] or {!driver} made ({!release}). It skips
      the test if the machine has no such GPU, and fails it if the GPU does not
      open. *)

  val close : t -> unit
  (** [close t] is {!Rig.close}[ t.d]. *)

  val with_ : (t -> 'a) -> 'a
  (** [with_ f] is [f t], [t] the {!open_}ed GPU, closed after [f] returns or
      raises. *)

  val driver : unit -> gpu
  (** [driver ()] is the GPU opened by its driver alone, which rig never takes,
      as {!open_} opens it. The caller stops it with {!stop_driver}. *)

  val stop_driver : gpu -> unit
  (** [stop_driver g] stops [g], which {!driver} opened. *)

  val with_driver : (gpu -> 'a) -> 'a
  (** [with_driver f] is [f g], [g] {!driver}[ ()], stopped after [f] returns
      or raises. *)

  val release : unit -> unit
  (** [release ()] ends what the last {!open_} or {!driver} made, if a failed
      test left it: it closes a device in rig, and stops one opened alone. *)

  val submit : t -> Rig.Submission.part array -> int
  (** [submit t ps] submits [ps] through rig on [t] and is their value. *)

  val wait : t -> int -> unit
  (** [wait t v] is {!Rig.wait}[ t.d v]. *)
end

(** [Make (G)] opens [G]'s GPU under the name [G.class_ ^ ":test"]. *)
module Make (G : Gpu) : S with type gpu = G.D.t

(** {1:conformance Conformance}

    What the conformance suite needs of a GPU beyond its contract: a binary
    and work on the device's first queue, which a driver's support makes from
    its suite's fixtures, found from the directory a suite runs in. Each work
    comes with the buffer of the device it reads its arguments from, which the
    submission reads, so that rig keeps it until the work ran. *)
module type Conformance = sig
  module D : Rig.Driver
  include S with type gpu = D.t

  val binary : unit -> string * string list
  (** [binary ()] is a binary of the driver's fixtures and the kernels it
      holds. *)

  val second : unit -> (D.t, string) result option
  (** [second ()] opens another device of the driver beside the one {!open_}
      gave, while that one is open, or is [None] where the machine has none.
      The caller stops it. *)

  val copy_words :
    t -> dst:Rig.Buffer.t -> src:Rig.Buffer.t -> Rig.Submission.part * Rig.Buffer.t
  (** [copy_words t ~dst ~src] is work on [t]'s first queue that copies [src]'s
      bytes into [dst], buffers of [t] of one length, a positive multiple of 4,
      and the buffer of [t] it reads its arguments from. *)

  val spin : t -> ns:int -> Rig.Submission.part * Rig.Buffer.t
  (** [spin t ~ns] is work on [t]'s first queue that runs at least [ns]
      nanoseconds, and the buffer of [t] it reads its arguments from. *)
end

val loader : (unit -> string) -> Rig.t -> Rig.Image.t
(** [loader bin] is [bin ()] loaded on a device, once per device while it is
    not lost. *)

val arguments : Rig.t -> string -> Rig.Buffer.t
(** [arguments d s] is new [Pinned] memory of [d] holding [s]. *)
