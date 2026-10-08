(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NV suites share. *)

(** {1:gpu The GPU} *)

include Rig_gpu_support.S with type gpu = Rig_nv.t
(** GPU [0], opened through {!Rig_nv_nvidia}. *)

val driver : unit -> Rig_nv.t
(** [driver ()] is GPU [0]'s driver device alone, opened through
    {!Rig_nv_nvidia} while the process holds the machine's GPU lock, after
    closing the device {!open_} made ({!release}) and stopping the one an
    earlier [driver] opened if no {!stop} stopped it, as a failed test leaves
    them: the GPU has one device at a time. {!open_} stops it too. It skips
    the test if the machine has no NVIDIA GPU. *)

val stop : Rig_nv.t -> unit
(** [stop g] is [Rig_nv.stop g]. Tests stop the devices {!driver} opened
    through it. *)

val with_driver : (Rig_nv.t -> 'a) -> 'a
(** [with_driver f] is [f g], [g] the {!driver} opened for [f] and stopped
    after it, whether it returns or raises. *)

(** {1:work Work through rig} *)

module Sub := Rig.Submission

val run : t -> Sub.part array -> unit
(** [run t ps] submits [ps] and waits for their value. *)

val words : ?after:int array -> int array -> Sub.part
(** [words ~after ws] is the ring words [ws] on ["COMPUTE:0"]. *)

val copy :
  ?after:int array ->
  dst:Rig.Buffer.t ->
  Rig.Buffer.t ->
  Sub.part
(** [copy ~after ~dst src] is a copy of [src] into [dst] on ["COPY:0"]. *)

val shared : t -> int -> Rig.Buffer.t * int
(** [shared t n] is [(b, a)]: [n] bytes of host memory at [a], which [b], a
    buffer of [t]'s, borrows. *)

val watchdog : string -> (unit -> 'a) -> 'a
(** [watchdog what f] is [f ()]. If [f] has not returned after 10 seconds, it
    prints [what] and ends the process: [f] blocks in C, where no timeout of the
    test reaches it. *)

(** {1:host Host memory} *)

val host : Rig_nv.region -> int
(** [host r] is [r]'s host address. Fails the test if the host does not address
    [r]. *)

val address : Rig_nv.region -> int
(** [address r] is [r]'s GPU address. *)

val pattern : int -> int -> int -> unit
(** [pattern a n seed] writes the [n] bytes at [a] with the pattern of [seed]:
    byte [i] a hash of [i] and [seed], so that a byte shifted, repeated or left
    from another pattern differs. *)

val mismatch : int -> int -> int -> int
(** [mismatch a n seed] is the index of the first of the [n] bytes at [a] that
    differs from the pattern of [seed], or [-1] if none does. *)

(** {1:launches Kernels and launches} *)

val fixture : ?dir:string -> string -> string
(** [fixture ~dir f] is the contents of the file [f] of [dir] (defaults to
    ["fixtures"]). *)

type kernels
(** The type for a cubin loaded on a device. *)

val kernels : ?dir:string -> ?file:string -> t -> kernels
(** [kernels ~dir ~file t] is the cubin [file] (defaults to
    ["kernels_sm89.cubin"]) of [dir] ({!fixture}), loaded on [t] by
    {!Rig.Image.load}.
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
