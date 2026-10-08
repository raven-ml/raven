(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the CUDA suite and bench share. *)

(** {1:gpu The GPU} *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] returns once the process holds the machine's GPU lock, which
    it keeps until it exits, or at once if the machine has no NVIDIA driver
    ([/dev/nvidiactl]). The lock is [flock] on [/tmp/raven-rig-gpu.lock], the
    file every suite and bench that acts on a GPU of the machine locks; its
    holder writes its executable and process id into it. A suite calls
    [hold_gpu] before [Windtrap.run], so that the wait counts against no test's
    timeout, and {!gpu} calls it again. A bench calls it before [Thumper.run],
    so that the workers it forks run under the lock: [hold_gpu] starts no vendor
    library, which a process must not start before it forks. It returns at once, taking nothing, if the variable
    [RIG_GPU_LOCK_HELD] is set: the process that started this one holds the
    lock for it, as a timing run takes it before the host's timing locks.

    Raises [Failure] naming the holder if another process still holds the lock
    after 300 s, or naming the errno if the file cannot be locked. *)

val gpu : unit -> Rig_cuda.t
(** [gpu ()] is GPU [0], opened, after stopping the device an earlier {!gpu}
    opened if no {!stop} stopped it, as a failed test leaves it, while the
    process holds the machine's GPU lock ({!hold_gpu}). It skips the test if
    CUDA sees no GPU. *)

val stop : Rig_cuda.t -> unit
(** [stop g] is [Rig_cuda.stop g]. Tests stop the devices {!gpu} opened through
    it. *)

val core : Rig_cuda.t -> Rig.t
(** [core g] is rig's device over [g], which {!gpu} opened under a name of its
    own.

    Raises [Invalid_argument] if [g] was stopped, lost or opened otherwise. *)

val submit : Rig_cuda.t -> Rig.Submission.part array -> int
(** [submit g ps] submits [ps] through rig on [g] and is their value. Raises
    what {!Rig.submit} raises; once it raised {!Rig.Lost}, rig stopped [g], and
    {!with_gpu} does not stop it again. *)

val bind : Rig_cuda.t -> unit
(** [bind g] makes the CUDA functions of [g]'s capability those {!attribute},
    {!current}, {!locked}, {!register}, {!unregister}, {!read_gpu},
    {!write_gpu}, {!free_memory}, {!launch} and {!delayed} call. {!gpu} binds
    them. *)

val with_gpu : (Rig_cuda.t -> 'a) -> 'a
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

val free_memory : unit -> int
(** [free_memory ()] is the bytes of GPU [0]'s memory CUDA reports free
    ([cuMemGetInfo]), after a {!with_gpu}. *)

val wait : Rig_cuda.t -> int -> unit
(** [wait g v] returns once [g]'s word, read as host memory, reaches [v]. It
    fails the test after 10 seconds of the host's monotonic clock. *)

(** {1:checks Checks} *)

val answer : [ `Ok | `Failed of string ] Windtrap.testable
(** [answer] prints and compares what {!Rig_cuda.submit} answers. *)

val still :
  ?msg:string -> 'a Windtrap.testable -> 'a -> (unit -> 'a) -> ms:int -> unit
(** [still w x f ~ms] reads [f ()] for about [ms] milliseconds of the host's
    monotonic clock, and fails the test if it is ever other than [x]. *)

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

val part : queue:string -> ?after:int array -> fill -> Rig.Submission.part
(** [part ~queue ~after f] is [f] as a part on [queue] after the parts [after]
    (defaults to [[||]]), its argument a host buffer. *)

val copy :
  queue:string ->
  ?after:int array ->
  dst:Rig.Buffer.t ->
  Rig.Buffer.t ->
  Rig.Submission.part
(** [copy ~queue ~after ~dst src] is a copy of [src] into [dst] on [queue]. *)

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
  Rig_cuda.t ->
  queue:int ->
  words:bool ->
  units:int ->
  bytes:int ->
  after:int array ->
  int
(** [room g ~queue ~words ~units ~bytes ~after] is what [rig_cuda_room] answers
    for one fill on the queue at index [queue], with one ring word iff [words],
    [units] ring units, [bytes] segment bytes, and [after]: [0] for RIG_FITS,
    [2] for RIG_NEVER. *)

type copy_c = {
  queue : int;
  dst : int;
  src : int;
  bytes : int;
  after : int array;
}
(** The type for copies handed to the C submit: [bytes] bytes from the address
    [src] to [dst], on the queue at index [queue], after the parts [after]. *)

val copies :
  Rig_cuda.t ->
  v:int ->
  waits:(int * int) array ->
  copy_c array ->
  [ `Ok | `Failed of string ]
(** [copies g ~v ~waits cs] is what [rig_cuda_submit] answers for [cs] as [g]'s
    value [v], after each wait [(a, w)]: the 64-bit word at [a] holds at least
    [w]. It reaches the waits rig cannot make, on any word. *)

(** {1:kernels Kernels} *)

val fixture : ?dir:string -> string -> string
(** [fixture ~dir f] is the file [f] of [dir] (defaults to ["fixtures"]):
    ["kernels.ptx"] holds the kernels [empty], [double_index out n] (the 32-bit
    word [i] at [out] is [2i] for [i < n]), [spin flag ns] (runs until the
    32-bit word at [flag] is not [0] or for [ns] nanoseconds) and [fault]
    (stores to address [0]), each of two 64-bit parameters; ["kernels.cubin"] is
    them compiled for [sm_89]; ["global.ptx"] holds [touch], of the same
    parameters, which stores to a global of 256 MiB. *)

val loaded :
  [ `Loaded of Rig_cuda.image
  | `Place of int * (Rig_cuda.region -> Rig_cuda.image * string) ] ->
  Rig_cuda.image
(** [loaded i] is the image of {!Rig_cuda.image}'s answer [i].

    Raises [Failure] if [i] is [`Place _]. *)

val kernels : ?dir:string -> Rig_cuda.t -> Rig_cuda.image * (string -> int)
(** [kernels ~dir g] is ["kernels.ptx"] of [dir] loaded on [g], and its kernels
    by name. *)
