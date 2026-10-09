(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the AMD suite and bench share: the GPU, loaded code objects, launch
    records and the runs of them, their device time, and operands drawn on the
    device. *)

(** {1:gpu The GPU} *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] is {!Rig_gpu_lock.hold} if the machine has an AMD GPU
    ({!Rig_amd_amdgpu.count}). A suite calls it before [Windtrap.run], a bench
    before [Thumper.run]. *)

type gpu
(** The type for the GPU the suite runs on: AMD GPU 0, opened through the amdgpu
    driver and rig. *)

val gpu : unit -> gpu
(** [gpu ()] is GPU 0, opened at the first call and kept for the process, while
    the process holds the GPU lock. It skips the test if the machine has no AMD
    GPU.

    Raises [Failure] if the GPU's compute queue reads AQL packets: the fill
    places PM4. *)

val arch : gpu -> string
(** [arch g] is the processor [g] runs code objects of, as ["gfx1201"]. *)

val wgps : gpu -> int
(** [wgps g] is [g]'s count of work-group processors that run work. *)

val cus : gpu -> int
(** [cus g] is [g]'s count of compute units that run work, two a work-group
    processor. *)

(** {1:images Images} *)

type image
(** The type for loaded code objects and the table of their kernels' dispatches
    a fill reads. Its kernels that take scratch memory share one buffer of the
    device's, made with the image. *)

val harness : gpu -> image
(** [harness g] is the harness's code object (harness.h) loaded on [g] once. *)

val threads : int
(** [threads] is the work-items of a workgroup of the harness's [generate],
    peaks and probes. *)

(** {1:records Records} *)

(** The type for a kernel's parameters, laid out in order as 8-byte words: its
    addresses first. *)
type param =
  | A of Rig.Buffer.t  (** The address of a buffer's first byte. *)
  | W of int  (** A 64-bit word. *)
  | D of int * int  (** Two 32-bit words. *)

type launch
(** The type for launches of a kernel, by name. *)

val launch :
  string -> groups:int * int * int -> threads:int -> param list -> launch
(** [launch k ~groups ~threads ps] is a launch of the kernel [k] over [groups]
    workgroups of [threads] work-items along X, with the parameters [ps]. *)

type run
(** The type for runs: launch records ([kernels.h]) of an image's kernels, which
    keep the buffers they address alive, or a driver's copy. *)

val record : image -> launch list -> run
(** [record i ls] is the run of the records of [ls], in order, which
    [nx_amd_add] appends to one run.

    Raises [Invalid_argument] if [i] has no kernel a launch names, an address
    follows another parameter, or [nx_amd_add] refuses a record. *)

val driver_copy : src:Rig.Buffer.t -> dst:Rig.Buffer.t -> run
(** [driver_copy ~src ~dst] is the driver's copy of [src]'s bytes into [dst], a
    {!Rig.Submission.Copy} on ["COPY:0"]. *)

(** {1:runs Runs} *)

type hog
(** The type for hogs: runs that hold half of a GPU's work-group processors. *)

val hog : gpu -> hog
(** [hog g] holds half of [g]'s work-group processors from a second device of
    the GPU while a run beside it lasts ({!run}). *)

val held_wgps : hog -> int list
(** [held_wgps h] is the work-group processors [h] held when it last ran, as
    the harness's [where] names them. *)

val run : ?beside:hog -> gpu -> run -> unit
(** [run ~beside g r] runs [r] on [g]'s queue ["COMPUTE:0"], a driver copy on
    ["COPY:0"], and returns once it is done. Beside a hog, [r] starts once the
    hog holds every processor it holds, and the hog lets them go once [r] is
    done, so that [r]'s workgroups run only on the others.

    Raises [Invalid_argument] if [r] is a driver copy beside a hog or a record's
    parameters are not the bytes its kernel reads, and [Failure] if the hog's
    workgroups do not all start within 2 s or the hog lets go after 2 s before
    [r] is done. *)

val enqueue : gpu -> count:int -> run -> unit
(** [enqueue g ~count r] submits [r] [count] times, in submissions of at most
    1,024 launches or 256 driver copies, and returns without waiting for them:
    the host's share of [count] runs. The caller keeps [r] until a later {!run}
    or {!device_time} returns. *)

val device_time : gpu -> run -> count:int -> float
(** [device_time g r ~count] runs [r] [count] times and is the GPU's time per
    run, in seconds: the span between two stamps of the GPU's clock that [r]'s
    queue writes around the runs. A long [count] is cut into rounds of at most
    1,024 launches or 256 driver copies, their spans summed. *)

val cu_clock : gpu -> float
(** [cu_clock g] is the clock of one of [g]'s compute units, in MHz, over ten
    million of its cycles timed as {!device_time} times. Measured right after
    work, it is the clock the work ended at: the GPU moves its clock over
    milliseconds. *)

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
    and writes a word per workgroup: the floor of a reduction over [b].

    Raises [Invalid_argument] if [b]'s length is not a multiple of 16. *)

(** {1:contract Contractions} *)

type operand = {
  buffer : Rig.Buffer.t;
  dtype : int;  (** Its dtype's code. *)
  shape : int array;
  strides : int array;  (** In elements. *)
  first : int;  (** Bytes from the buffer's start to element 0. *)
}
(** The type for operands of nx.amd's plans. *)

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
(** [contract g ~a ~b ~init ~y ~batch ~contracting ~acc ()] is nx.amd's plan of
    the contraction ([nx_amd_plan_contract]) on {!library}'s kernels, its
    scratch allocated on [g] and kept by the run, or [None] if the plan
    declines. [batch] and [contracting] pair an axis of [a] with one of [b].

    Raises [Out_of_memory] if the host's memory cannot hold the plan, and
    {!Rig.Out_of_memory} if [g]'s cannot hold its scratch. *)

val planner :
  a:operand ->
  b:operand ->
  ?init:operand ->
  y:operand ->
  batch:(int * int) list ->
  contracting:(int * int) list ->
  acc:int ->
  unit ->
  unit ->
  int
(** [planner ~a ~b ~init ~y ~batch ~contracting ~acc () ()] plans the
    contraction as {!contract} does, into records it keeps from one call to the
    next, and is the plan's count of launches: the planner's own cost, for the
    bench. *)

val library : gpu -> image
(** [library g] is nx.amd's code object for [g]'s processor loaded on [g] once,
    its table holding every kernel of [kernels.h].

    Raises [Failure] if nx.amd has no code object for [g]. *)

val library_size : gpu -> int * int
(** [library_size g] is the count of kernels and the bytes of {!library}'s code
    object: the instance budget's measure. *)

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
