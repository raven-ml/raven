(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the NV suites share. *)

(** {1:gpu The GPU} *)

val gpu_lock : string
(** [gpu_lock] is the environment variable that names the machine's GPU lock
    file: ["DEVICE_NV_TEST_GPU_LOCK"]. *)

val gpu : unit -> Device_nv.t
(** [gpu ()] is GPU [0], opened through {!Device_nv_nvidia}, after stopping the
    device an earlier {!gpu} opened if no {!stop} stopped it, as a failed test
    leaves it. It skips the test if the machine has no NVIDIA GPU, if
    {!gpu_lock} names no file, or if another process holds the lock, which this
    process keeps until it exits once it took it. *)

val stop : Device_nv.t -> unit
(** [stop g] is [Device_nv.stop g]. Tests stop the devices {!gpu} opened through
    it. *)

val with_gpu : (Device_nv.t -> 'a) -> 'a
(** [with_gpu f] is [f g], [g] the {!gpu} opened for [f] and stopped after it,
    whether it returns or raises, unless [f] stopped it. *)

(** {1:values Values} *)

val last : Device_nv.t -> int
(** [last g] is the last value {!submit} gave [g], [0] before the first. A test
    that calls {!Device_nv.submit} itself gives [last g + 1] and then calls
    {!given}. *)

val given : Device_nv.t -> int -> unit
(** [given g v] records that [v] was given to [g]. *)

val submit :
  ?waits:([ `Word | `Object ] * int * int) array ->
  Device_nv.t ->
  Device_nv.part array ->
  int
(** [submit ~waits g ps] is the value [v] after {!last}[ g], once
    {!Device_nv.submit} took [ps] as [v] with [waits] (defaults to none). While
    {!Device_nv.room} answers [`Later] it waits for {!last}[ g]. It fails the
    test if room answers [`Never] or the submission [`Failed]. *)

val wait : Device_nv.t -> int -> unit
(** [wait g v] returns once [g]'s word, read as host memory, reaches [v]. It
    fails the test after 10 seconds of CPU time. *)

val answer : [ `Ok | `Failed of string ] Windtrap.testable
(** [answer] prints and compares what {!Device_nv.submit} answers. *)

val still :
  ?msg:string -> 'a Windtrap.testable -> 'a -> (unit -> 'a) -> ms:int -> unit
(** [still w x f ~ms] reads [f ()] for about [ms] milliseconds of CPU time, and
    fails the test if it is ever other than [x]. *)

val watchdog : string -> (unit -> 'a) -> 'a
(** [watchdog what f] is [f ()]. If [f] has not returned after 10 seconds, it
    prints [what] and ends the process: [f] blocks in C, where no timeout of the
    test reaches it. *)

val room :
  Device_nv.t ->
  queue:int ->
  words:int ->
  ?fill:bool ->
  ?units:int ->
  ?bytes:int ->
  ?copy:bool ->
  int array ->
  int
(** [room g ~queue ~words ~fill ~units ~bytes ~copy after] is what
    [device_nv_room] answers for one part on the queue at index [queue]: [words]
    zero words, a fill iff [fill], [units] ring units, [bytes] segment bytes, a
    copy of one byte iff [copy], and the indices [after]. [0] is NX_FITS, [1]
    NX_LATER, [2] NX_NEVER. Options default to [false] and [0]. *)

(** {1:host Host memory} *)

val page : int
(** [page] is the host's page size, in bytes. *)

val pages : int -> int
(** [pages n] is the address of [n] zeroed writable bytes from a page boundary.
*)

val free_pages : int -> int -> unit
(** [free_pages a n] returns the [n] bytes {!pages} gave at [a]. *)

val host : Device_nv.region -> int
(** [host r] is [r]'s host address. Fails the test if the host does not address
    [r]. *)

val address : Device_nv.region -> int
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
(** The type for a cubin loaded on a device and uploaded. *)

val kernels : ?file:string -> Device_nv.t -> kernels
(** [kernels ~file g] is the cubin [file] (defaults to ["kernels_sm89.cubin"])
    of the fixtures loaded on [g] over a new [`Device] region, its image copied
    there by a submission that completed. ["kernels_sm89.cubin"] holds the
    kernels of ["kernels.cu"]. *)

val image : kernels -> Device_nv.image
(** [image k] is [k]'s image. *)

val code : kernels -> Device_nv.region
(** [code k] is [k]'s code region, which {!unload} frees. *)

val unload : Device_nv.t -> kernels -> unit
(** [unload g k] unloads [k]'s image, then frees its code region. *)

type launches
(** The type for memory that holds launches: their descriptors, constant banks
    and ring segments. *)

val launches : Device_nv.t -> launches
(** [launches g] is new memory for launches on [g]. *)

val launch :
  launches -> kernels -> string -> blocks:int -> int list -> int array
(** [launch l k f ~blocks args] is the ring entry, as two words, of a segment
    that schedules the kernel [f] of [k] over [blocks] blocks of 256 threads,
    with the 64-bit parameters [args], after making the local memory of [l]'s
    device serve it. The memory it uses is [l]'s until {!reset}. *)

val release : launches -> int -> int -> int array
(** [release l a x] is the ring entry, as two words, of a segment that writes
    the 64-bit [x] at the address [a] once the channel's earlier work completed.
*)

val reset : launches -> unit
(** [reset l] makes [l]'s memory free for new launches. The caller resets [l]
    once no work uses its launches. *)

val free_launches : Device_nv.t -> launches -> unit
(** [free_launches g l] frees [l]'s memory. *)
