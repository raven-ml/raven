(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the CUDA suite and bench share. *)

(** {1:gpu The GPU} *)

include Rig_gpu_support.Conformance with module D = Rig_cuda
(** GPU [0], its CUDA functions bound ({!bind}); the second device is GPU
    [1]. *)

val bind : Rig_cuda.t -> unit
(** [bind g] makes the CUDA functions of [g]'s capability those {!attribute},
    {!current}, {!locked}, {!register}, {!unregister}, {!read_gpu},
    {!write_gpu}, {!free_memory}, {!functions_loaded}, {!stall}, {!context}
    and {!graph_launch} call. {!open_} binds them. *)

val attribute : int -> int
(** [attribute a] is the value of CUDA's device attribute [a] of CUDA's device
    [0], after a {!with_}. *)

val current : unit -> nativeint
(** [current ()] is the calling thread's current CUDA context, [0n] for none,
    after a {!with_}. *)

val locked : int -> bool
(** [locked a] is [true] iff CUDA holds the host memory at [a] page-locked and
    mapped for its devices, after a {!with_}. *)

val register : int -> int -> unit
(** [register a n] page-locks the [n] bytes of host memory at [a] for every CUDA
    device and maps them, as another library would, after a {!with_}. *)

val unregister : int -> unit
(** [unregister a] ends what {!register} page-locked at [a]. *)

val free_memory : unit -> int
(** [free_memory ()] is the bytes of GPU [0]'s memory CUDA reports free
    ([cuMemGetInfo]), after a {!with_}. *)

val functions_loaded : int -> int * int
(** [functions_loaded f] is [(n, k)]: the image of the function [f], an
    entry's [CUfunction], has [n] functions, and CUDA holds the code of [k] of
    them loaded on the GPU ([cuFuncIsLoaded]), after a {!with_}. *)

val read_gpu : nativeint -> int -> string
(** [read_gpu a n] is the [n] bytes of GPU memory at the address [a], read by
    CUDA, after a {!with_}. The caller reads it once no work writes it. *)

val write_gpu : nativeint -> string -> unit
(** [write_gpu a s] stores [s] in the GPU memory at the address [a], through
    CUDA, after a {!with_}: the bytes are there when it returns, after every
    work of CUDA's default stream ({!stall}). *)

val stall : int -> flag:int -> ns:int -> unit
(** [stall spin ~flag ~ns] runs the kernel [spin] of ["kernels.ptx"] with [flag]
    and [ns] on CUDA's default stream, which rig's streams do not wait for,
    after a {!with_}. *)

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

val context : unit -> fill
(** [context ()] records the context current while it runs ({!seen}). *)

val seen : fill -> nativeint
(** [seen f] is the context current while the {!context} fill [f] last ran. *)

val kernel : ?grid:int -> ?block:int -> int -> int -> int -> Rig_cuda_abi.kernel
(** [kernel ~grid ~block f a b] is a launch of the kernel [f] over [grid] blocks
    (defaults to [1]) of [block] threads (defaults to [1]) with the 64-bit
    parameters [a] and [b] as its argument block, no shared memory. *)

val graph_launch :
  Rig_cuda_abi.graph -> (int * Rig_cuda_abi.kernel) array -> fill
(** [graph_launch g us] updates node [i] of [g] to the kernel [k] for each
    [(i, k)] of [us], in order, then launches [g], through the device's
    capability. Each argument block holds at most 64 bytes. *)

(** {1:kernels Kernels} *)

val launch : t -> string -> int -> int -> Rig_gpu_support.work
(** [launch t f a b] is a launch on [t]'s ["COMPUTE:0"] of the kernel [f] of
    ["kernels.ptx"], loaded by rig, over one thread, with the 64-bit parameters
    [a] and [b]. *)

val fixture : ?dir:string -> string -> string
(** [fixture ~dir f] is the file [f] of [dir] (defaults to ["fixtures"]):
    ["kernels.ptx"] holds the kernels [empty], [double_index out n] (the 32-bit
    word [i] at [out] is [2i] for [i < n]), [spin flag ns] (runs until the
    32-bit word at [flag] is not [0] or for [ns] nanoseconds), [step out i]
    (stores [i + 1] into the 64-bit word at [out] if it holds [i]) and [fault]
    (stores to address [0]), each of two 64-bit parameters; ["kernels.cubin"] is
    them compiled for [sm_89]; ["global.ptx"] holds [touch], of the same
    parameters, which stores to a global of 256 MiB; ["launch.ptx"] holds the
    conformance suite's launch kernels, [copy dst src n] (copies [n] 32-bit
    words, one a thread) and [rotate out k], which stores into the 32-bit word
    [i] at [out], [i] a thread's index in the grid, the value of the next
    thread of its group (of the first for the last): [k + 3j + 7g] for [j]
    that thread's index and [g] its group's, through the group's dynamic
    shared memory of at least 4 bytes a thread. *)

val loaded : (Rig_cuda.region, Rig_cuda.image) Rig_edge.code -> Rig_cuda.image
(** [loaded i] is the image of {!Rig_cuda.image}'s answer [i].

    Raises [Failure] if [i] is [Place _]. *)

val kernels : ?dir:string -> Rig_cuda.t -> Rig_cuda.image * (string -> int)
(** [kernels ~dir g] is ["kernels.ptx"] of [dir] loaded on [g], and its kernels
    by name. *)
