(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the CUDA suite and bench share. *)

(** {1:gpu The GPU} *)

val gpu_lock : string
(** [gpu_lock] is the environment variable that names the machine's GPU lock
    file: ["DEVICE_CUDA_TEST_GPU_LOCK"]. *)

val gpu : unit -> Device_cuda.t
(** [gpu ()] is GPU [0], opened, after stopping the device an earlier {!gpu}
    opened if no {!stop} stopped it, as a failed test leaves it. It skips the
    test if CUDA sees no GPU, if {!gpu_lock} names no file, or if another
    process holds the lock, which this process keeps until it exits once it took
    it. *)

val stop : Device_cuda.t -> unit
(** [stop g] is [Device_cuda.stop g]. Tests stop the devices {!gpu} opened
    through it. *)

val bind : Device_cuda.t -> unit
(** [bind g] makes the CUDA functions of [g]'s capability those {!attribute},
    {!current}, {!locked}, {!register}, {!unregister}, {!read_gpu},
    {!write_gpu}, {!launch} and {!delayed} call. {!gpu} binds them. *)

val with_gpu : (Device_cuda.t -> 'a) -> 'a
(** [with_gpu f] is [f g], [g] the {!gpu} opened for [f] and stopped after it,
    whether it returns or raises, unless [f] stopped it. *)

val attribute : int -> int
(** [attribute a] is the value of CUDA's device attribute [a] of CUDA's device
    [0], after a {!with_gpu}. *)

val current : unit -> nativeint
(** [current ()] is the calling thread's current CUDA context, [0n] for none,
    after a {!with_gpu}. *)

val locked : int -> bool
(** [locked a] is [true] iff CUDA holds the host memory at [a] page-locked and
    mapped for its devices, after a {!with_gpu}. *)

val register : int -> int -> unit
(** [register a n] page-locks the [n] bytes of host memory at [a] for every CUDA
    device and maps them, as another library would, after a {!with_gpu}. *)

val unregister : int -> unit
(** [unregister a] ends what {!register} page-locked at [a]. *)

val wait : Device_cuda.t -> int -> unit
(** [wait g v] returns once [g]'s word, read as host memory, reaches [v]. It
    fails the test after 10 seconds of CPU time. *)

(** {1:checks Checks} *)

val answer : [ `Ok | `Failed of string ] Windtrap.testable
(** [answer] prints and compares what {!Device_cuda.submit} answers. *)

val still :
  ?msg:string -> 'a Windtrap.testable -> 'a -> (unit -> 'a) -> ms:int -> unit
(** [still w x f ~ms] reads [f ()] for about [ms] milliseconds of CPU time, and
    fails the test if it is ever other than [x]. *)

(** {1:host Host memory} *)

val page : int
(** [page] is the host's page size, in bytes. *)

val pages : ?read_only:bool -> int -> int
(** [pages n] is the address of [n] zeroed bytes from a page boundary, read-only
    iff [read_only] (defaults to [false]). *)

val free_pages : int -> int -> unit
(** [free_pages a n] returns the [n] bytes {!pages} gave at [a]. *)

val get64 : int -> int
(** [get64 a] is the 64-bit word at [a], read with acquire order. *)

val set64 : int -> int -> unit
(** [set64 a x] stores [x] in the 64-bit word at [a] with release order. *)

val read : int -> int -> string
(** [read a n] is the [n] bytes at [a]. *)

val write : int -> string -> unit
(** [write a s] stores [s] at [a]. *)

val get32 : int -> int -> int
(** [get32 a i] is the unsigned 32-bit word [i] at [a]. *)

val read_gpu : nativeint -> int -> string
(** [read_gpu a n] is the [n] bytes of GPU memory at the address [a], read by
    CUDA, after a {!with_gpu}. The caller reads it once no work writes it. *)

val write_gpu : nativeint -> string -> unit
(** [write_gpu a s] stores [s] in the GPU memory at the address [a], through
    CUDA, after a {!with_gpu}. *)

(** {1:fills Fills} *)

type fill
(** The type for fills and their arguments. *)

val part :
  Device_cuda.t -> queue:string -> ?after:int array -> fill -> Device_cuda.part
(** [part g ~queue ~after f] is [f] as a part of [g]'s [queue]. The caller keeps
    [f] until the part's submission returned. *)

val failing : int -> fill
(** [failing code] returns [code] and enqueues nothing. *)

val launch : ?count:int -> int -> grid:int -> block:int -> int -> int -> fill
(** [launch ~count f ~grid ~block a b] launches the kernel [f] [count] times
    (defaults to [1]) over [grid] blocks of [block] threads with the 64-bit
    parameters [a] and [b], through the device's capability. *)

val delayed :
  spin:int -> flag:int -> ns:int -> dst:int -> src:int -> int -> fill
(** [delayed ~spin ~flag ~ns ~dst ~src n] runs the kernel [spin] of
    ["kernels.ptx"] with [flag] and [ns], then copies [n] bytes from the address
    [src] to the address [dst]: a copy that starts at least [ns] nanoseconds
    late while the 32-bit word at [flag] is [0]. *)

val seen : fill -> nativeint
(** [seen f] is the context current while the {!launch} fill [f] last ran. *)

val room :
  Device_cuda.t ->
  queue:int ->
  words:bool ->
  units:int ->
  bytes:int ->
  after:int array ->
  int
(** [room g ~queue ~words ~units ~bytes ~after] is what [device_cuda_room]
    answers for one fill on the queue at index [queue], with one ring word iff
    [words], [units] ring units, [bytes] segment bytes, and [after]: [0] for
    NX_FITS, [2] for NX_NEVER. *)

(** {1:kernels Kernels} *)

val fixture : ?dir:string -> string -> string
(** [fixture ~dir f] is the file [f] of [dir] (defaults to ["fixtures"]):
    ["kernels.ptx"] holds the kernels [empty], [double_index out n] (the 32-bit
    word [i] at [out] is [2i] for [i < n]), [spin flag ns] (runs until the
    32-bit word at [flag] is not [0] or for [ns] nanoseconds) and [fault]
    (stores to address [0]), each of two 64-bit parameters; ["kernels.cubin"] is
    them compiled for [sm_89]. *)

val loaded :
  [ `Loaded of Device_cuda.image
  | `Place of int * (Device_cuda.region -> Device_cuda.image * string) ] ->
  Device_cuda.image
(** [loaded i] is the image of {!Device_cuda.image}'s answer [i].

    Raises [Failure] if [i] is [`Place _]. *)

val kernels :
  ?dir:string -> Device_cuda.t -> Device_cuda.image * (string -> int)
(** [kernels ~dir g] is ["kernels.ptx"] of [dir] loaded on [g], and its kernels
    by name. *)
