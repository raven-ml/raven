(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the CUDA suite and bench share: the GPU, the harness's kernels and runs
    of them as rig's launches, nx.cuda's contractions, their device time, and
    operands drawn on the device. *)

(** {1:gpu The GPU} *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] is {!Rig_gpu_lock.hold} if the machine has an NVIDIA driver
    ([/dev/nvidiactl]). A suite calls it before [Windtrap.run], a bench before
    [Thumper.run]. *)

type gpu
(** The type for the GPU the suite runs on: CUDA's GPU 0 through rig. *)

val gpu : unit -> gpu
(** [gpu ()] is GPU 0, opened at the first call and kept for the process, while
    the process holds the GPU lock. It skips the test if CUDA sees no GPU.

    Raises [Failure] if nx.cuda does not compute on it. *)

val device : gpu -> Rig.t
(** [device g] is [g]'s device. *)

val arch : gpu -> string
(** [arch g] is [g]'s compute capability, as ["sm_89"]. *)

val sms : gpu -> int
(** [sms g] is [g]'s count of multiprocessors. *)

(** {1:images Images} *)

type image
(** The type for loaded cubins. *)

val harness : gpu -> image
(** [harness g] is the harness's cubin (harness.h) loaded on [g] once. *)

(** {1:launches Launches} *)

(** The type for a kernel's parameters, laid out in order as 8-byte words: its
    addresses first. *)
type param =
  | A of Rig.Buffer.t  (** The address of a buffer's first byte. *)
  | W of int  (** A 64-bit word. *)
  | D of int * int  (** Two 32-bit words. *)

type launch
(** The type for launches of a kernel, by name. *)

val launch :
  string ->
  grid:int * int * int ->
  block:int ->
  ?shared:int ->
  param list ->
  launch
(** [launch k ~grid ~block ~shared ps] is a launch of the kernel [k] over [grid]
    blocks of [block] threads along X, with [shared] dynamic shared bytes
    (defaults to [0]) and the parameters [ps].

    Raises [Invalid_argument] if an address follows another parameter. *)

type run
(** The type for runs: launches of an image's kernels, a driver's copy, or a
    call of nx.cuda's contraction. A run keeps the buffers it names. *)

val record : image -> launch list -> run
(** [record i ls] is the run of [ls], in order, as one submission's launches,
    every buffer they address written.

    Raises [Invalid_argument] if [i] has no kernel a launch names. *)

val driver_copy : src:Rig.Buffer.t -> dst:Rig.Buffer.t -> run
(** [driver_copy ~src ~dst] is the driver's copy of [src]'s bytes into [dst], a
    {!Rig.Submission.Copy} on ["COPY:0"]. *)

(** {1:runs Runs} *)

val run : gpu -> run -> unit
(** [run g r] runs [r], launches on [g]'s queue ["COMPUTE:0"], a driver copy on
    ["COPY:0"], and returns once it is done. *)

val enqueue : gpu -> count:int -> run -> unit
(** [enqueue g ~count r] enqueues [r] [count] times behind a hold, releases the
    hold once they are queued, and returns without waiting for them: the host's
    share of [count] runs. *)

val call : run -> unit
(** [call r] calls the contraction [r] once and returns once its work is queued.
    It allocates nothing beyond what {!Nx_cuda.contract} allocates.

    Raises [Invalid_argument] if [r] is not {!contract}'s, and [Failure] if
    nx.cuda declines it. *)

val issue : gpu -> count:int -> run -> unit
(** [issue g ~count r] calls the contraction [r] [count] times behind a hold,
    releases the hold, and returns once their work is done. With a warm [r] it
    allocates nothing beyond what the calls allocate.

    Raises [Invalid_argument] if [r] is not {!contract}'s. *)

val device_time : gpu -> run -> count:int -> float
(** [device_time g r ~count] runs [r] [count] times and is the GPU's time per
    run, in seconds: the span between two timer stamps around the runs on
    ["COMPUTE:0"], behind a kernel that holds it until every run is queued, so
    that the host never starves the GPU. A long [count] is cut into rounds of at
    most 512 launches, their spans summed.

    Raises [Failure] if a round's launches outgrew the stream before its hold
    ran out. *)

val sm_clock : gpu -> float
(** [sm_clock g] is the clock of one of [g]'s multiprocessors, in MHz, over a
    million of its cycles. Measured right after work, it is the clock the work
    ended at: the GPU moves its clock over milliseconds. *)

(** {1:floors Floors} *)

val floor_copy : ins:Rig.Buffer.t list -> out:Rig.Buffer.t -> launch
(** [floor_copy ~ins ~out], a launch of the {!harness}, writes to each 16-byte
    vector of [out] the XOR of the vectors at its index in the one to three
    [ins] that reach it: the copy floor of work that reads [ins] and writes
    [out].

    Raises [Invalid_argument] if [ins] has none or more than three buffers, or a
    buffer's length is not a multiple of 16. *)

val floor_read : gpu -> Rig.Buffer.t -> launch
(** [floor_read g b], a launch of the {!harness}, reads [b]'s 16-byte vectors
    and writes a word per block: the floor of a reduction over [b].

    Raises [Invalid_argument] if [b]'s length is not a multiple of 16. *)

(** {1:contract Contractions} *)

type operand = {
  buffer : Rig.Buffer.t;
  dtype : int;  (** Its dtype's code. *)
  shape : int array;
  strides : int array;  (** In elements. *)
  first : int;  (** Bytes from the buffer's start to element 0. *)
}
(** The type for operands of nx.cuda's contractions. *)

val contract :
  gpu ->
  a:operand ->
  b:operand ->
  ?init:operand ->
  y:operand ->
  batch:(int * int) list ->
  contracting:(int * int) list ->
  acc:int ->
  unit ->
  run option
(** [contract g ~a ~b ~init ~y ~batch ~contracting ~acc ()] computes the
    contraction with {!Nx_cuda.contract} into [y], its work queued on [g], and
    is the run that computes it again; or [None] if nx.cuda declines it. [batch]
    and [contracting] pair an axis of [a] with one of [b].

    Raises what {!Nx_cuda.contract} raises, and [Invalid_argument] for a
    refusal. *)

(** {1:memory Memory} *)

val buffer : gpu -> int -> Rig.Buffer.t
(** [buffer g n] is [n] bytes of [g]'s memory. *)

val read : Rig.Buffer.t -> string
(** [read b] is [b]'s bytes, copied to the host. *)

val write : Rig.Buffer.t -> string -> unit
(** [write b s] copies [s] into [b], of [s]'s length. *)

(** How {!generate} draws values ([harness.h]'s [nx_harness_draw]). *)
type draw =
  | Uniform  (** Floats in [\[-1, 1)], integers over their range. *)
  | Wide of int
      (** Floats of a random sign and an exponent uniform in [[-e, e]]; integers
          as [Uniform]. *)
  | Small  (** Integers in [\[-8, 8)], or [\[0, 16)] unsigned. *)

val generate :
  gpu -> Rig.Buffer.t -> ('v, 's) Nx_array.Dtype.t -> draw -> seed:int -> unit
(** [generate g b dt d ~seed] fills [b] with elements of [dt] drawn by [d] from
    [seed], each a function of [seed] and its index: as many as [b] holds.

    Raises [Invalid_argument] if [dt] is complex or narrower than a byte: it
    draws the floats from float64 to float8, the integers and booleans. *)
