(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NV suites share. *)

(** {1:gpu The GPU} *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] returns once the process holds the machine's GPU lock, which
    it keeps until it exits, or at once if the machine has no NVIDIA GPU. The
    lock is [flock] on [/tmp/raven-rig-gpu.lock], the file every suite that
    acts on a GPU of the machine locks; its holder writes its executable and
    process id into it. A suite calls [hold_gpu] before [Windtrap.run], so that
    the wait counts against no test's timeout; {!gpu} calls it again.

    Raises [Failure] naming the holder if another process still holds the lock
    after 300 s, or naming the errno if the file cannot be locked. *)

type dev = { d : Rig.t; g : Rig_nv.t }
(** The type for an open GPU: [d] as programs reach it, [g] its driver's device.
*)

val gpu : unit -> dev
(** [gpu ()] is GPU [0], opened through {!Rig_nv_nvidia} and
    {!Rig.open_} under a name of its own, after stopping the driver
    device an earlier {!gpu} or {!driver} opened if no {!stop} stopped it, as a
    failed test leaves it. It holds the machine's GPU lock ({!hold_gpu}), and
    skips the test if the machine has no NVIDIA GPU. *)

val driver : unit -> Rig_nv.t
(** [driver ()] is {!gpu}'s driver device alone, for work handed over at the C
    edge ({!edge_submit}). *)

val stop : Rig_nv.t -> unit
(** [stop g] is [Rig_nv.stop g]. Tests stop the devices {!gpu} and {!driver}
    opened through it. *)

val close : dev -> unit
(** [close t] stops [t]'s driver device, once the core gave back what it mapped
    of collected memory, unless the core lost [t], which stopped it. *)

val with_gpu : (dev -> 'a) -> 'a
(** [with_gpu f] is [f t], [t] the {!gpu} opened for [f] and its driver device
    stopped after it, whether it returns or raises, unless [f] stopped it or the
    core lost it, which stops it. *)

val with_driver : (Rig_nv.t -> 'a) -> 'a
(** [with_driver f] is {!with_gpu} for {!driver}. *)

(** {1:work Work through the core} *)

module Sub := Rig.Submission

val submit : dev -> Sub.part array -> int
(** [submit t ps] is the value {!Rig.submit} gave [ps] on [t]. *)

val run : dev -> Sub.part array -> unit
(** [run t ps] submits [ps] and waits for their value. *)

val words : ?after:int array -> int array -> Sub.part
(** [words ~after ws] is the ring words [ws] on ["COMPUTE:0"]. *)

val copy :
  ?after:int array ->
  dst:Rig.Buffer.t ->
  Rig.Buffer.t ->
  Sub.part
(** [copy ~after ~dst src] is a copy of [src] into [dst] on ["COPY:0"]. *)

val shared : dev -> int -> Rig.Buffer.t * int
(** [shared t n] is [(b, a)]: [n] bytes of host memory at [a], which [b], a
    buffer of [t]'s, borrows. *)

(** {1:edge Work at the C edge} *)

type part = {
  queue : int;  (** [0] for ["COMPUTE:0"], [1] for ["COPY:0"]. *)
  after : int array;
  work : [ `Words of int array | `Copy of int * int * int | `Fill of int * int ];
      (** Ring words; a copy [(dst, src, n)] between addresses; a fill of ring
          units and segment bytes. *)
}
(** The type for parts as [rig_edge.h] describes them. *)

val edge_room : Rig_nv.t -> part array -> int
(** [edge_room g ps] is what [rig_nv_room] answers for [ps]: [0] RIG_FITS, [1]
    RIG_LATER, [2] RIG_NEVER. *)

val edge_submit :
  Rig_nv.t -> v:int -> waits:(int * int) array -> part array -> unit
(** [edge_submit g ~v ~waits ps] hands [ps] to [rig_nv_submit] as the value
    [v], after the waits [(address, value)]. It fails the test unless the answer
    is RIG_OK. *)

val wait : Rig_nv.t -> int -> unit
(** [wait g v] returns once [g]'s word, read as host memory, reaches [v]: it
    spins for 200 ms, then sleeps in {!Rig_nv.sleep} between reads, which
    raises the device's {!Rig_nv.Fault}. *)

val still :
  ?msg:string -> 'a Windtrap.testable -> 'a -> (unit -> 'a) -> ms:int -> unit
(** [still w x f ~ms] reads [f ()] for about [ms] milliseconds of CPU time, and
    fails the test if it is ever other than [x]. *)

val watchdog : string -> (unit -> 'a) -> 'a
(** [watchdog what f] is [f ()]. If [f] has not returned after 10 seconds, it
    prints [what] and ends the process: [f] blocks in C, where no timeout of the
    test reaches it. *)

(** {1:host Host memory} *)

val page : int
(** [page] is the host's page size, in bytes. *)

val pages : int -> int
(** [pages n] is the address of [n] zeroed writable bytes from a page boundary.
*)

val free_pages : int -> int -> unit
(** [free_pages a n] returns the [n] bytes {!pages} gave at [a]. *)

val host : Rig_nv.region -> int
(** [host r] is [r]'s host address. Fails the test if the host does not address
    [r]. *)

val address : Rig_nv.region -> int
(** [address r] is [r]'s GPU address. *)

val get64 : int -> int
(** [get64 a] is the 64-bit word at [a], read with acquire order. *)

val set64 : int -> int -> unit
(** [set64 a x] stores [x] in the 64-bit word at [a] with release order. *)

val get32 : int -> int -> int
(** [get32 a i] is the unsigned 32-bit word [i] at [a]. *)

val read : int -> int -> string
(** [read a n] is the [n] bytes at [a]. *)

val write : int -> string -> unit
(** [write a s] stores [s] at [a]. *)

val pattern : int -> int -> int -> unit
(** [pattern a n seed] writes the [n] bytes at [a] with the pattern of [seed]:
    byte [i] a hash of [i] and [seed], so that a byte shifted, repeated or left
    from another pattern differs. *)

val mismatch : int -> int -> int -> int
(** [mismatch a n seed] is the index of the first of the [n] bytes at [a] that
    differs from the pattern of [seed], or [-1] if none does. *)

(** {1:launches Kernels and launches} *)

val fixture : string -> string
(** [fixture f] is the contents of the file [f] of ["fixtures"]. *)

type kernels
(** The type for a cubin loaded on a device. *)

val kernels : ?file:string -> dev -> kernels
(** [kernels ~file t] is the cubin [file] (defaults to ["kernels_sm89.cubin"])
    of the fixtures, loaded on [t] by {!Rig.Program.load}.
    ["kernels_sm89.cubin"] holds the kernels of ["kernels.cu"]. *)

val image : Rig_nv.t -> string -> Rig_nv.image * Rig_nv.region * string
(** [image g bin] is the cubin [bin] loaded on [g] by the driver alone: the
    image, its new [`Device] code region and the bytes to write there. *)

type launches
(** The type for driver memory that holds launches: their descriptors, constant
    banks and ring segments. *)

val launches : Rig_nv.t -> launches
(** [launches g] is new memory for launches on [g]. *)

val launch :
  launches -> kernels -> string -> blocks:int -> int list -> int array
(** [launch l k f ~blocks args] is the ring entry, as two words, of a segment
    that schedules the kernel [f] of [k] over [blocks] blocks of 256 threads,
    with the 64-bit parameters [args], after making the local memory of [l]'s
    device serve it. The memory it uses is [l]'s until {!reset}. *)

val launch_at :
  launches ->
  code:int ->
  string ->
  string ->
  blocks:int ->
  int list ->
  int array
(** [launch_at l ~code bin f ~blocks args] is {!launch} for the kernel [f] of
    the cubin [bin] whose image the caller placed at the address [code]. *)

val segment : launches -> int Rig_nv_abi.Packet.t -> int array
(** [segment l p] is the ring entry, as two words, of a segment of the words
    [p]. *)

val reset : launches -> unit
(** [reset l] makes [l]'s memory free for new launches. The caller resets [l]
    once no work uses its launches. *)

val free_launches : launches -> unit
(** [free_launches l] frees [l]'s memory. *)
