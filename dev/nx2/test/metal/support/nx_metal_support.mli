(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Metal suite and bench share: the Mac's GPU opened through rig with
    the harness's metallib, operands in its memory, and runs of the harness's
    launches and of nx.metal's contractions. *)

(** {1:gpu The machine's GPU lock} *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] is {!Rig_gpu_lock.hold} if the machine has the Metal
    framework. A suite calls it before [Windtrap.run], a bench before
    [Thumper.run]. *)

(** {1:devices Devices} *)

type t
(** The type for the GPU opened with the harness's metallib loaded. *)

val open_ : unit -> t option
(** [open_ ()] opens the Mac's GPU as the rig device ["METAL"] and loads the
    harness's metallib; [None] on a machine with no Metal GPU.

    Raises [Failure] if nx.metal does not compute on it. *)

val rig : t -> Rig.t
(** [rig t] is [t]'s device. *)

(** {1:operands Operands} *)

type operand
(** The type for device memory the host reads and writes. *)

val operand : t -> int -> operand
(** [operand t n] is [n] bytes of [t]'s memory, [n >= 1], with unspecified
    contents. *)

val buffer : operand -> Rig.Buffer.t
(** [buffer o] is [o]'s memory as a buffer. *)

val view :
  ('a, 'b) Bigarray.kind ->
  operand ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
(** [view k o] is [o]'s bytes as elements of kind [k]; it keeps [o] alive. The
    host reads what the GPU wrote once the run that wrote it returned. *)

val generate :
  ?spread:int ->
  t ->
  operand ->
  ('v, 's) Nx_array.Dtype.t ->
  int ->
  seed:int ->
  unit
(** [generate ~spread t o dt n ~seed] writes into [o] the first [n] elements of
    the deterministic sequence of [dt] from [seed], and returns once they are
    there: a float has a random sign and significand and an exponent drawn in
    [[-spread, spread]] (defaults to [0]), rounded to [dt]; an integer has
    random bits.

    Raises [Invalid_argument] naming [dt] unless it is float64, float32,
    float16, bfloat16 or an integer of 8 to 64 bits. *)

(** {1:runs Runs} *)

type run
(** The type for runs: launches of the harness's kernels and calls of
    nx.metal's contraction, in order. A run keeps the operands it names. *)

val launch :
  ?groups:int * int * int ->
  ?threads:int * int * int ->
  string ->
  addrs:operand list ->
  words:int list ->
  run
(** [launch ~groups ~threads k ~addrs ~words] is a launch of the harness's
    kernel [k] over [groups] threadgroups (defaults to [(1, 1, 1)]) of
    [threads] threads (defaults to the harness's 256), with its parameters:
    the 64-bit addresses of [addrs]' first bytes, then the 32-bit [words],
    padded to a multiple of 8 bytes. A run of launches writes every operand
    they address.

    Raises [Invalid_argument] if the harness has no kernel [k]. *)

val groups : int -> int * int * int
(** [groups n] is the threadgroups of 256 threads that cover [n] threads. *)

val seq : run list -> run
(** [seq rs] runs the launches of [rs] in order. *)

val prepare : t -> run -> unit -> int
(** [prepare t r] makes the submissions of [r]'s launches, each run of
    consecutive launches one submission, and is a function that submits them
    and calls [r]'s contractions in order, waits for their work and answers
    the wall time it took, in nanoseconds.

    The function raises [Failure] if nx.metal declines a contraction, and
    [Invalid_argument] for a refusal. *)

val run : t -> run -> int
(** [run t r] is [prepare t r ()]. *)

val call : run -> unit
(** [call r] calls the contraction [r] once and returns once its work is queued.
    It allocates nothing beyond what {!Nx_metal.contract} allocates.

    Raises [Invalid_argument] if [r] is not one contraction of
    {!plan_contract}, and [Failure] if nx.metal declines it. *)

val issue : t -> count:int -> run -> unit
(** [issue t ~count r] calls the contraction [r] [count] times and returns once
    their work is done. With a warm [r] it allocates nothing beyond what the
    calls allocate.

    Raises as {!call} does. *)

(** {1:probes Probes} *)

type floats = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
type words = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

val probe : ?dtype:int -> t -> string -> operand -> which:int -> int -> operand
(** [probe ~dtype t k in_ ~which n] runs the probe kernel [k] over the first [n]
    operand tuples of [in_], with the format [dtype] (an nx_dtype.h code,
    defaults to [0]), and is its [n] 32-bit results. *)

(** The checks below compare the GPU's results with the host's and count the
    differences. NaN equals NaN. A [show]ing check prints its first [show] wrong
    cases (defaults to [0]). *)

val probe_contract : ?show:int -> floats -> floats -> floats -> int * int * int
(** [probe_contract ~show abc one fused] counts, over the triples [abc], the
    results [one] of [a·b + c] that differ from the product and the sum rounded
    apart, those that differ from the fused result, and the results [fused] of
    [fma a b c] that differ from it, float32 subnormals flushed. *)

val probe_div_sqrt : ?show:int -> floats -> floats -> floats -> int * int * int
(** [probe_div_sqrt ~show xy div sqrt] counts, over the pairs [xy], the
    quotients [div] that differ from the correctly rounded [x / y], the roots
    [sqrt] that differ from the correctly rounded root of [x], float32
    subnormals flushed, and the correct quotients that are subnormal. *)

val probe_half : words -> words -> words -> words -> int * int * int
(** [probe_half xs to_ from sum] counts, over the words [xs], the binary16 codes
    [to_] of [x] read as float32 that differ from [nx_float_to_f16]'s, the
    float32 bits [from] of [x]'s low half read as binary16 that differ from
    [nx_f16_to_float]'s, and the codes [sum] of [h + h], [h] that low half, that
    differ from the exact sum rounded to binary16. *)

val probe_codec :
  ('v, 's) Nx_array.Dtype.t ->
  words ->
  words ->
  words ->
  words ->
  int * int * int
(** [probe_codec dt codes decoded floats encoded] counts, for the narrow float
    format [dt], the GPU's decodings [decoded] of [codes] and its encodings
    [encoded] of the float32 bits [floats] that differ from nx_dtype.h's on the
    host. *)

(** {1:contract Contract} *)

type arg
(** The type for operands of a call: memory, a dtype and strides in elements
    over the call's three axes. *)

val arg :
  ?first:int -> operand -> ('v, 's) Nx_array.Dtype.t -> int * int * int -> arg
(** [arg ~first o dt strides] is [o] read as [dt] with [strides], its first
    element [first] elements into [o] (defaults to [0]). *)

val arg_operand : arg -> operand
(** [arg_operand a] is [a]'s memory. *)

val plan_contract :
  ?init:arg ->
  ?acc:Nx_array.Dtype.any ->
  t ->
  int * int * int * int ->
  a:arg ->
  b:arg ->
  out:arg ->
  run option
(** [plan_contract ~init ~acc t (batch, m, n, k) ~a ~b ~out] computes the
    contraction of a [(batch, m, k)] and b [(batch, k, n)] into out
    [(batch, m, n)], C-contiguous, plus [init], accumulated in [acc] (defaults
    to float32), with {!Nx_metal.contract}, waits for it, and is the run that
    computes it again; or [None] if nx.metal declines it.

    Raises what {!Nx_metal.contract} raises, and [Invalid_argument] for a
    refusal. *)

val contract_wrong :
  ?init:arg ->
  ?samples:int ->
  acc:Nx_array.Dtype.any ->
  int * int * int * int ->
  a:arg ->
  b:arg ->
  out:arg ->
  int * int
(** [contract_wrong ~acc] is, for an integer contraction accumulated in [acc],
    once its run returned, the number of outputs that differ from the sum
    wrapped to [acc]'s width then to out's, and the first one's index, or [-1].
    With [samples], it reads that many outputs spread evenly, the last
    included; all of them by default. *)

val contract_error :
  ?init:arg ->
  ?samples:int ->
  int * int * int * int ->
  a:arg ->
  b:arg ->
  out:arg ->
  float * int
(** [contract_error] is, for a float contraction, once its run returned, the
    largest distance of an output to the exact result as a fraction of the
    distance the contraction's bound allows, float32 subnormals flushed, with
    that output's index: at most [1.] when every output is within it. With
    [samples], it reads that many outputs spread evenly, the last included;
    all of them by default.

    Raises [Failure] if an output's ratio is NaN, which no bound orders. *)
