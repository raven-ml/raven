(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Metal suite and bench share: the Mac's GPU opened through rig with
    the harness's metallib, operands in its memory, and runs of launch records
    submitted through nx.metal's fill. *)

(** {1:gpu The machine's GPU lock} *)

val hold_gpu : unit -> unit
(** [hold_gpu ()] returns once the process holds the machine's GPU lock, which
    it keeps until it exits, or at once if the machine has no Metal framework.
    The lock is [flock] on [/tmp/raven-rig-gpu.lock], which every suite and
    bench that acts on a GPU of the machine takes; its holder writes its
    executable and process id into it. It returns at once, taking nothing, if
    the variable [RIG_GPU_LOCK_HELD] is set: the process that started this one
    holds the lock for it.

    Raises [Failure] naming the holder if another process still holds the lock
    after 300 s, or naming the errno if the file cannot be locked. *)

(** {1:devices Devices} *)

type t
(** The type for the GPU opened with the harness's metallib loaded. *)

val open_ : unit -> t option
(** [open_ ()] opens the Mac's GPU as the rig device ["METAL"] and loads the
    harness; [None] on a machine with no Metal GPU. A second call answers the
    same device. *)

val rig : t -> Rig.t
(** [rig t] is [t]'s device. *)

val kernels : string array
(** [kernels] is the harness's kernels, by their enum. *)

(** {1:operands Operands} *)

type operand
(** The type for device memory the host reads and writes. *)

val operand : t -> int -> operand
(** [operand t n] is [n] bytes of [t]'s memory, [n >= 1], with unspecified
    contents. *)

val address : operand -> int
(** [address o] is [o]'s GPU address. *)

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
(** The type for runs of launches. *)

val launch :
  ?groups:int * int * int ->
  ?threads:int * int * int ->
  string ->
  addrs:int list ->
  words:int list ->
  run
(** [launch ~groups ~threads k ~addrs ~words] is a launch of the harness's
    kernel [k] over [groups] threadgroups (defaults to [(1, 1, 1)]) of [threads]
    threads (defaults to the harness's 256), with its parameters: the 64-bit
    addresses [addrs], then the 32-bit [words], padded to a multiple of 8 bytes.

    Raises [Invalid_argument] if [k] is no kernel. *)

val groups : int -> int * int * int
(** [groups n] is the threadgroups of 256 threads that cover [n] threads. *)

val seq : run list -> run
(** [seq rs] runs the launches of [rs] in order. *)

val prepare : t -> run -> unit -> int
(** [prepare t r] makes the pipelines of [r]'s kernels and is a function that
    submits [r] as one command buffer, waits for it and answers its GPU time in
    nanoseconds. *)

val run : t -> run -> int
(** [run t r] is [prepare t r ()]. *)

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
