(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the CUDA suite and bench share. *)

(** {1:gpu The GPU} *)

include Rig_gpu_support.Conformance with module D = Rig_cuda
(** GPU [0], its CUDA functions bound ({!bind}); the second device is GPU [1].
*)

val bind : Rig_cuda.t -> unit
(** [bind g] makes the CUDA functions of [g]'s capability those {!attribute},
    {!current}, {!locked}, {!register}, {!unregister}, {!read_gpu},
    {!write_gpu}, {!free_memory}, {!functions_loaded}, {!context_flags},
    {!set_context_flags}, {!limit_shared}, {!stall} and {!context} call.
    {!open_} binds them. *)

val attribute : int -> int
(** [attribute a] is the value of CUDA's device attribute [a] of CUDA's device
    [0], after a {!with_}. *)

val context_flags : unit -> int
(** [context_flags ()] is the flags of CUDA's device [0]'s primary context
    ([cuDevicePrimaryCtxGetState]), after a {!with_}. *)

val set_context_flags : int -> unit
(** [set_context_flags f] sets the flags of CUDA's device [0]'s primary context
    to [f] ([cuDevicePrimaryCtxSetFlags]), as another library would, after a
    {!with_}. The flags are the process's. *)

val limit_shared : int -> int -> unit
(** [limit_shared f n] sets the most dynamic shared memory a launch of the
    function [f], an entry's [CUfunction], may take to [n] bytes
    ([cuFuncSetAttribute]), behind its device's back, after a {!with_}. *)

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
(** [functions_loaded f] is [(n, k)]: the image of the function [f], an entry's
    [CUfunction], has [n] functions, and CUDA holds the code of [k] of them
    loaded on the GPU ([cuFuncIsLoaded]), after a {!with_}. *)

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

val fn : fill -> nativeint
(** [fn f] is the address of [f]'s C function. *)

val parameters : fill -> string
(** [parameters f] is the argument block of [f], while [f] is reachable. *)

val part : Rig.t -> fill -> Rig_gpu_support.work
(** [part d f] is [f] as work on [d], with its parameters, the values its
    submits pass, and private state. *)

val failing : ?bytes:int -> int -> fill
(** [failing ~bytes code] returns [code] and enqueues nothing. Its argument
    holds [code] in its first 4 bytes, then zeros up to [bytes] (defaults to
    4). *)

val context : unit -> fill
(** [context ()] records the context current while it runs ({!seen}). *)

val seen : fill -> nativeint
(** [seen f] is the context current while the {!context} fill [f] last ran. *)

val spinning : int -> flag:int -> ns:int -> fill
(** [spinning spin ~flag ~ns] runs the kernel [spin] as {!stall} does, on the
    stream of its part's queue. *)

(** {1:entry The C entry} *)

val submit_edge : Rig_cuda.t -> v:int -> (int * fill) list -> string option
(** [submit_edge g ~v parts] is what [g]'s C submit answers for [parts], each
    a fill on the queue at its index, as the value [v] with no waits: [None],
    or [Some why] for a failure. [g] is a device opened outside rig, so that
    no other call runs meanwhile. *)

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
    [i] at [out], [i] a thread's index in the grid, the value of the next thread
    of its group (of the first for the last): [k + 3j + 7g] for [j] that
    thread's index and [g] its group's, through the group's dynamic shared
    memory of at least 4 bytes a thread; [mark], of no parameters, which adds
    1 to a count the image holds; [marks out], which stores that count into
    the 32-bit word at [out] and sets it to [0]; and [page out], of 4,096
    bytes of parameters, which stores its 32-bit words at bytes 8 and 4,092
    into words 0 and 1 at [out]. ["wide.ptx"] holds the conformance suite's
    [wide] and [widest out], of 32,764 bytes of parameters, which stores its
    words at bytes 8, 16,384 and 32,760 into words 0 to 2 at [out]. *)

val loaded : (Rig_cuda.region, Rig_cuda.image) Rig_edge.code -> Rig_cuda.image
(** [loaded i] is the image of {!Rig_cuda.image}'s answer [i].

    Raises [Failure] if [i] is [Place _]. *)

val kernels : ?dir:string -> Rig_cuda.t -> Rig_cuda.image * (string -> int)
(** [kernels ~dir g] is ["kernels.ptx"] of [dir] loaded on [g], and its kernels
    by name. *)
