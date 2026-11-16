(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the AMD suite and bench share: the GPU, the harness's kernels and runs
    of them as rig's launches, calls of nx.amd's kernels, their time on the GPU,
    and operands drawn on the device. *)

(** {1:gpu The GPU} *)

val device : unit -> Rig.t
(** [device ()] is the device of {!Nx_kernels_support.amd}[ ()]: the machine's
    AMD GPU 0, on which nx.amd computes. It skips the test where there is none.
*)

type gpu
(** The type for {!device}[ ()] with the host memory it maps for holds and
    stamps. *)

val gpu : unit -> gpu
(** [gpu ()] is the GPU, made at the first call and kept for the process. It
    skips the test where {!device} does.

    Raises [Failure] if the device maps no host memory. *)

val arch : gpu -> string
(** [arch g] is [g]'s processor, as ["gfx1201"]. *)

val compute_units : gpu -> int
(** [compute_units g] is the count of [g]'s compute units that run work. *)

(** {1:launches Launches} *)

(** The type for a harness kernel's parameters, laid out in order as 8-byte
    words. *)
type word =
  | A of Rig.Buffer.t  (** The address of a buffer's first byte. *)
  | W of int  (** A 64-bit word. *)

val launch :
  string -> groups:int * int * int -> threads:int -> word list -> unit
(** [launch k ~groups ~threads ws] runs the harness kernel [k] over [groups]
    groups of [threads] work-items along x, with parameters [ws], every buffer
    it addresses written, on {!device}[ ()], and returns once it is done.

    Raises [Invalid_argument] if the harness has no kernel [k]. *)

type dispatch
(** The type for dispatches of a harness kernel, by name. *)

val dispatch :
  string -> groups:int * int * int -> threads:int -> word list -> dispatch
(** [dispatch k ~groups ~threads ws] is a dispatch of the harness kernel [k]
    over [groups] groups of [threads] work-items along x, with the parameters
    [ws], each buffer it addresses written.

    Raises [Invalid_argument] if the harness has no kernel [k]. *)

(** {1:runs Runs} *)

type run
(** The type for runs: dispatches of the harness, a driver's copy, or a call of
    one of nx.amd's kernels ({!section-calls}). A run keeps the buffers it
    names. *)

val record : dispatch list -> run
(** [record ds] is the run of [ds] as one submission's launches, in order. A
    launch addresses every buffer as written, so it follows each earlier launch
    of the run, and of an earlier repetition, that addresses a buffer it
    addresses; launches that share none may run concurrently. *)

val driver_copy : src:Rig.Buffer.t -> dst:Rig.Buffer.t -> run
(** [driver_copy ~src ~dst] is the driver's copy of [src]'s bytes into [dst], a
    {!Rig.Submission.copy} part, which runs on ["COPY:0"]. *)

val run : gpu -> run -> unit
(** [run g r] runs [r] once and returns once it is done. *)

val call : run -> unit
(** [call r] calls the kernel of the call [r] once and returns once its work is
    submitted. It allocates nothing beyond what nx.amd's kernel allocates.

    Raises [Invalid_argument] if [r] is not a call, and [Failure] if nx.amd
    declines it. *)

val issue : gpu -> count:int -> run -> unit
(** [issue g ~count r] calls the kernel of the call [r] [count] times behind a
    hold, releases the hold, and returns once their work is done.

    Raises [Invalid_argument] if [r] is not a call, and [Failure] if the hold
    ran out first. *)

val busy : gpu -> run -> count:int -> unit
(** [busy g r ~count] runs [r] [count] times with no hold and returns once they
    are done: work that brings the GPU's clock to where [r] keeps it. *)

val device_time : gpu -> run -> count:int -> float
(** [device_time g r ~count] runs [r] [count] times and is the GPU's time per
    run, in seconds: the span between two stamps of the GPU's clock around the
    runs, behind a kernel that holds the GPU until every run is submitted, so
    that the host never starves it. The stamps follow the hold, and the runs
    follow the first stamp and precede the second, by rig's order alone: by
    parts that follow every earlier part, by a buffer the first stamp addresses
    as written for a driver's copy, and by device order for calls. A call is a
    submission of its own, so its time counts the cache writeback and
    invalidation that starts each submission. A long [count] is cut into rounds
    of at most 512 launches and 512 KiB of their arguments, a launch of nx.amd's
    counted as 4 KiB, their spans summed. A round's hold waits for the host to
    make the round's calls, 2 s at most.

    Raises [Failure] if a round's calls outlasted its hold, as when their
    launches' arguments outgrow the device's argument segment. *)

val clock : gpu -> float
(** [clock g] is the clock of one of [g]'s shader engines, in MHz, over a
    million of its cycles. Measured right after work, it is the clock the work
    ended at: the GPU moves its clock over milliseconds. *)

(** {1:floors Floors} *)

val floor_copy : ins:Rig.Buffer.t list -> out:Rig.Buffer.t -> dispatch
(** [floor_copy ~ins ~out] writes to each 16-byte vector of [out] the XOR of the
    vectors at its index in the one to three [ins] that reach it: the copy floor
    of work that reads [ins] and writes [out].

    Raises [Invalid_argument] if [ins] has none or more than three buffers, or a
    buffer's length is not a multiple of 16. *)

val floor_read : ?out:Rig.Buffer.t -> gpu -> Rig.Buffer.t -> dispatch
(** [floor_read ~out g b] reads [b]'s 16-byte vectors over groups of 256
    work-items, at most 65,535 of them, and writes to the 32-bit word [i] of
    [out] the XOR of the 32-bit words that group [i] read: the floor of a
    reduction over [b]. Every vector is read by one group, so the XOR of [out]'s
    words written is the XOR of [b]'s. [out] (defaults to a buffer of its own)
    receives a word for each of the [min 65535 (ceil (n / 1024))] groups of [n]
    vectors.

    Raises [Invalid_argument] if [b]'s length is not a multiple of 16, or [out]
    holds fewer words than the groups. *)

val peak : gpu -> string -> rounds:int -> dispatch * int
(** [peak g k ~rounds] is a dispatch of the harness's peak kernel [k] in 4
    groups of 8 waves a compute unit, each wave [rounds] rounds of 16 steps, and
    the count of its waves' steps. The kernels are ["peak_wmma_f16"],
    ["peak_wmma_bf16"], ["peak_wmma_fp8"] and ["peak_wmma_iu8"], whose step is a
    16x16x16 matrix instruction, and ["peak_fma_f32"] and ["peak_fma_f64"],
    whose step is an fma of each element of a 32-byte vector a work-item.

    Raises [Invalid_argument] if the harness has no kernel [k]. *)

(** {1:calls Calls of nx.amd's kernels} *)

type operand = {
  buffer : Rig.Buffer.t;
  dtype : int;  (** Its dtype's code. *)
  shape : int array;
  strides : int array;  (** In elements. *)
  first : int;  (** Bytes from the buffer's start to element 0. *)
}
(** The type for operands of nx.amd's kernels. *)

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
    contraction with {!Nx_amd.contract} into [y], its work submitted, and is the
    run that computes it again; or [None] if nx.amd declines it. [batch] and
    [contracting] pair an axis of [a] with one of [b].

    Raises what {!Nx_amd.contract} raises, and [Invalid_argument] for a refusal
    or an [init] of another dtype than [y]'s. *)

val reduce :
  gpu ->
  Nx_kernel.monoid ->
  axes:int list ->
  x:operand ->
  y:operand ->
  run option
(** [reduce g m ~axes ~x ~y] reduces [x] by [m] along [axes] into [y] with
    {!Nx_amd.reduce}, a load of [x] read plain, its work submitted, and is the
    run that computes it again; or [None] if nx.amd declines it.

    Raises what {!Nx_amd.reduce} raises, and [Invalid_argument] for a refusal.
*)

val scan :
  gpu -> Nx_kernel.monoid -> axis:int -> x:operand -> y:operand -> run option
(** [scan g m ~axis ~x ~y] scans [x] by [m] along [axis] into [y] with
    {!Nx_amd.scan}, as {!reduce} reduces. *)

val array : operand -> Nx_array.any
(** [array o] is [o] as an array over its buffer. *)

val loop_launches : int
(** [loop_launches] is the most launches a call of nx.amd's [map], [reduce] or
    [scan] submits: 3. *)

val kernel_call :
  what:string ->
  launches:int ->
  Nx_array.any list ->
  (unit -> Nx_array.answer) ->
  run option
(** [kernel_call ~what ~launches arrays f] calls [f], a call of nx.amd's kernel
    [what] over [arrays] that submits at most [launches] launches, and is the
    run that calls it again; or [None] if nx.amd declines it.

    Raises [Invalid_argument] for a refusal. *)

val empty_call : gpu -> run
(** [empty_call g] is a call that submits one launch of a kernel that does
    nothing, in a submission of its own: the least a call of nx.amd's kernels
    takes on the GPU. *)

(** {1:memory Memory} *)

val buffer : gpu -> int -> Rig.Buffer.t
(** [buffer g n] is [n] bytes of [g]'s memory. *)

val read : Rig.Buffer.t -> string
(** [read b] is [b]'s bytes, copied to the host. *)

val write : Rig.Buffer.t -> string -> unit
(** [write b s] copies [s] into [b], of [s]'s length. *)

(** How {!generate} draws values. *)
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
    Floats round to [dt] from the draw's double, a value that rounds to a
    subnormal is zero, and a [Uniform] float that rounds to a magnitude of 1 is
    [dt]'s next value toward zero, so it lies in [\[-1, 1)] at every float
    dtype. [Small] draws floats as [Uniform].

    Raises [Invalid_argument] if [dt] is complex or narrower than a byte. *)
