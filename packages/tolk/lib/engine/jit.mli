(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** JIT compilation and replay.

    A {e JIT} ({!Tiny_jit}) wraps a function and transparently captures
    its computation schedule on the second call, then replays it on all
    subsequent calls.  Three phases:

    {ul
    {- {e Warmup} (cnt=0): execute eagerly.}
    {- {e Capture} (cnt=1): record the schedule, compile kernels, plan
       memory, execute, and store the result as a {!Captured_jit}.}
    {- {e Exec} (cnt>=2): validate inputs, substitute fresh buffers,
       and replay the compiled schedule.}}

    On the first replay, the schedule may be condensed into
    {!Graph_runner} executors when the device supports batched
    dispatch.

    See also {!Realize} for the underlying runner and exec-item
    types. *)

(** {1:exceptions Exceptions} *)

exception Graph_exn of string
(** Raised when graph batching fails for a batch of kernels.
    The string describes why (e.g. too few kernels, unsupported
    device). *)

exception Jit_error of string
(** Raised for JIT-specific errors: nested capture, empty capture,
    input mismatch on replay. *)

(** {1:types Types} *)

(** Runner kind.

    Discriminates the runner attached to an {!exec_item} so that
    JIT internals can dispatch on runner kind (compiled kernel vs
    buffer copy vs graph batch) without runtime type introspection. *)
type prg =
  | Compiled of Realize.Compiled_runner.t
      (** Compiled kernel.  Carries the full {!Program_spec.t} via
          {!Realize.Compiled_runner.p}. *)
  | View_op of Realize.Runner.t
      (** Buffer view (zero-copy reshape). *)
  | Buffer_copy of Realize.Runner.t
      (** Host-bounce buffer copy. *)
  | Buffer_xfer of Realize.Runner.t
      (** Device-to-device transfer. *)
  | Enc_dec of Realize.Runner.t
      (** Hardware encode/decode. *)
  | Graph of graph_runner
      (** Batched graph executor. *)

(** Execution item with mutable buffer slots.

    Buffer slots are stored as an array so that input substitution
    on replay is O(1).  The {!uid} field provides stable identity
    across list rebuilds (e.g. after graph batching). *)
and exec_item = {
  uid : int;
  bufs : Device.Buffer.t option array;
  prg : prg;
  fixedvars : string list;
      (** Variable names bound at schedule time.  These are excluded
          from runtime substitution in the {!Graph_runner}. *)
}

(** Graph runner.

    Batches multiple kernels for accelerated dispatch on devices
    that support graph APIs (e.g. CUDA graphs, Metal command
    buffers).  Precomputes replacement tables for variables and
    launch dimensions so the device graph only needs to update the
    values that actually change between calls.

    Device-specific graph implementations construct this via
    {!create_graph_runner} and override the runner's call function
    to perform the actual graph dispatch. *)
and graph_runner = {
  gr_cache : exec_item list;
      (** Exec items in this batch (kept alive for the graph). *)
  gr_input_replace : ((int * int), int) Hashtbl.t;
      (** [(j, i) -> k]: buffer slot [i] of cache entry [j] is
          input buffer [k]. *)
  gr_var_replace : (int, (int * int) list) Hashtbl.t;
      (** [j -> \[(prog_var_idx, global_var_idx); ...\]]: for
          cache entry [j], which program variables need runtime
          substitution and their index into {!field-gr_vars}. *)
  gr_dims_replace : (int, int option * int option) Hashtbl.t;
      (** [j -> (global_sym_idx, local_sym_idx)]: for cache entry
          [j], indices into {!field-gr_sym_dims} for symbolic
          launch dimensions.  [None] means the dimension is
          constant. *)
  gr_dims_base : (int, int array * int array) Hashtbl.t;
      (** [j -> (global, local)]: concrete base launch dimensions
          for cache entry [j], used as fallback for non-symbolic
          dimensions. *)
  gr_vars : string array;
      (** Sorted unique variable names across all kernels. *)
  gr_sym_dims : Tolk_ir.Kernel.t array list;
      (** Unique symbolic launch dimension vectors.  Evaluated
          via {!Tolk_ir.Kernel.sym_infer} at dispatch time. *)
  gr_w_dep : (int, (int * int * int) list) Hashtbl.t;
      (** Write dependency map for suballocated buffers.  Keyed
          by base buffer id; values are [(start, end, dep)]
          interval triples.  Populated by {!access_resources}. *)
  gr_r_dep : (int, (int * int * int) list) Hashtbl.t;
      (** Read dependency map.  Same structure as
          {!field-gr_w_dep}. *)
  gr_runner : Realize.Runner.t;
      (** Underlying runner for dispatch.  Device graph
          implementations provide the call function. *)
}

(** View input descriptor.

    Records a sub-buffer relationship so that view inputs can be
    reconstructed from base input buffers on every replay call. *)
type view_input = {
  vi_base_idx : int;  (** Index of the base buffer in the input array. *)
  vi_offset : int;  (** Byte offset from the base. *)
  vi_device : string;  (** Device name. *)
  vi_size : int;  (** Element count. *)
  vi_dtype : Tolk_ir.Dtype.t;  (** Element type. *)
}

(** Input validation descriptor.

    Captured at the end of the capture phase and checked on every
    replay call to ensure inputs have not changed shape, dtype, or
    device. *)
type input_info = {
  ii_size : int;  (** Element count. *)
  ii_dtype : Tolk_ir.Dtype.t;  (** Element type. *)
  ii_device : string;  (** Device name. *)
}

(** {1:exec_items Exec items} *)

val runner_of_prg : prg -> Realize.Runner.t
(** [runner_of_prg prg] is the underlying {!Realize.Runner.t}
    for [prg], regardless of runner kind. *)

val run_ei :
  exec_item -> (string * int) list -> jit:bool -> unit
(** [run_ei ei var_vals ~jit] dispatches [ei] with variable
    bindings [var_vals].  Buffers are allocated on demand.
    When [jit] is [true], execution does not wait for
    completion. *)

val lower_realize_ei :
  device:Device.t ->
  get_program:(Tolk_ir.Kernel.t -> Program_spec.t) ->
  Realize.Exec_item.t ->
  exec_item
(** [lower_realize_ei ~device ~get_program rei] compiles [rei]
    and wraps the result as an {!exec_item} with the appropriate
    {!prg} variant.

    Kernel ASTs are compiled via {!Realize.get_runner}.
    Buffer views become {!View_op}; copies become
    {!Buffer_copy}.

    Raises [Failure] if the AST node is not a supported
    [Call] variant. *)

val get_out_buffers : exec_item -> Device.Buffer.t list
(** [get_out_buffers ei] is the list of buffers written by [ei].
    For compiled kernels, output parameters not also read; for
    copies, the destination buffer.  Empty for views and graph
    runners. *)

(** {1:dependencies Buffer dependencies} *)

(** Mutable set of buffers keyed by identity, with an optional
    [None] sentinel. *)
type buf_set = {
  mutable has_none : bool;
  tbl : (int, Device.Buffer.t) Hashtbl.t;
}

val buf_set : unit -> buf_set
(** [buf_set ()] is a fresh empty set. *)

val buf_set_mem : buf_set -> Device.Buffer.t option -> bool
(** [buf_set_mem s b] is [true] iff [b] is in [s].  [None]
    matches the sentinel. *)

val buf_set_add : buf_set -> Device.Buffer.t -> unit
(** [buf_set_add s b] adds [b] to [s]. *)

val update_depends : buf_set -> exec_item list -> unit
(** [update_depends depends cache] propagates buffer dependencies
    forward: for each exec item in [cache] whose inputs overlap
    [depends], the item's output buffers are added to [depends]. *)

val get_input_replace :
  exec_item list ->
  Device.Buffer.t array ->
  ?orig_valid_positions:(int, int list) Hashtbl.t ->
  unit ->
  ((int * int), int) Hashtbl.t
(** [get_input_replace cache input_bufs ?orig_valid_positions ()]
    maps input buffer positions in [cache].

    Returns a table where key [(j, i)] means buffer slot [i] of
    cache entry [j] holds input buffer at index [v].

    When [orig_valid_positions] is provided (keyed by
    {!exec_item.uid}), only positions present in that table are
    included.  This prevents aliasing bugs when graph batching
    reuses buffer slots. *)

(** {1:graph_runner Graph runner} *)

val create_graph_runner :
  exec_item list ->
  Device.Buffer.t array ->
  (string * int) list ->
  ?orig_valid_positions:(int, int list) Hashtbl.t ->
  unit ->
  graph_runner
(** [create_graph_runner cache input_bufs var_vals
    ?orig_valid_positions ()] builds a graph runner for [cache].

    Precomputes variable and launch-dimension replacement tables
    from the compiled kernels in [cache].  The base runner's call
    function is a no-op; device graph implementations should
    replace it. *)

val updated_vars :
  graph_runner -> (string * int) list -> (int * int * int) list
(** [updated_vars gr var_vals] is the list of
    [(cache_idx, program_var_idx, value)] triples for all
    variables in [gr] that need runtime substitution given
    [var_vals]. *)

val updated_launch_dims :
  graph_runner -> (string * int) list ->
  (int * int array * int array) list
(** [updated_launch_dims gr var_vals] is the list of
    [(cache_idx, global, local)] triples for all kernels in [gr]
    with symbolic launch dimensions, evaluated against
    [var_vals]. *)

val access_resources :
  graph_runner ->
  Device.Buffer.t array ->
  write:int list ->
  int ->
  int list
(** [access_resources gr bufs ~write new_dep] updates the
    interval-based read/write dependency maps in [gr] and returns
    the list of prior dependencies that [bufs] must wait on.

    [write] is the list of buffer indices (into [bufs]) that are
    written.  [new_dep] is the dependency handle for this
    dispatch. *)

val supports_exec_item : Device.t list -> exec_item -> bool
(** [supports_exec_item devs ei] is [true] iff [ei] is a
    compiled kernel and all devices in [devs] are the same. *)

val multi_supports_exec_item : Device.t list -> exec_item -> bool
(** [multi_supports_exec_item devs ei] is [true] iff [ei] is a
    compiled kernel or device transfer and all devices (from
    [devs] and [ei]'s buffers) share the same backend type. *)

(** {1:graph_batching Graph batching} *)

val apply_graph_to_jit :
  exec_item list ->
  Device.Buffer.t array ->
  (string * int) list ->
  ?orig_valid_positions:(int, int list) Hashtbl.t ->
  ?max_batch_size:int ->
  unit ->
  exec_item list
(** [apply_graph_to_jit cache input_bufs var_vals
    ?orig_valid_positions ?max_batch_size ()] splits [cache]
    into batches for graph execution.

    Consecutive compatible kernels are condensed into a single
    {!Graph} exec item when the device supports batched dispatch.
    The batch size doubles after each successful graph.

    Returns [cache] unchanged when no device graph support is
    available.

    [max_batch_size] defaults to [0] (unlimited). *)

(** {1:memory Memory planning} *)

val plan_jit_memory : exec_item list -> exec_item list
(** [plan_jit_memory cache] runs the internal memory planner over
    [cache], returning a new cache with optimized buffer
    assignments.  Buffers not reassigned by the planner keep
    their original allocation; reassigned buffers are allocated
    eagerly. *)

(** {1:captured Captured JIT} *)

(** A captured computation schedule ready for replay.

    Created at the end of the capture phase, a {!captured_jit}
    holds the compiled schedule, the input-to-buffer mapping, and
    precomputed read-after-write hazard tables.  On the first
    replay, graph batching is attempted; subsequent replays reuse
    the batched schedule. *)
type 'a captured_jit

val create_captured :
  'a ->
  exec_item list ->
  ((int * int), int) Hashtbl.t ->
  view_input list ->
  input_info array ->
  'a captured_jit
(** [create_captured ret cache input_replace views input_info]
    is a captured JIT holding return value [ret], schedule
    [cache], input mapping [input_replace], view input
    descriptors [views], and input validation info [input_info].

    Initializes hazard-detection tables and clears input buffer
    slots. *)

val clear_inputs : 'a captured_jit -> unit
(** [clear_inputs t] sets all input buffer slots to [None] so
    their memory can be freed or reused between calls. *)

val free_intermediates : 'a captured_jit -> unit
(** [free_intermediates t] deallocates all intermediate buffers
    reachable from cleared input slots and resets execution
    state.  The next replay will re-allocate intermediates and
    re-attempt graph batching. *)

val replan_buffers_memory_layout : 'a captured_jit -> unit
(** [replan_buffers_memory_layout t] re-runs the memory planner
    over [t]'s schedule with relaxed checks, remaps buffer
    assignments, copies data from old to new buffers, and resets
    execution state. *)

val exec_captured :
  'a captured_jit ->
  device:Device.t ->
  Device.Buffer.t array ->
  (string * int) list ->
  'a
(** [exec_captured t ~device input_bufs var_vals] executes the
    captured schedule with fresh [input_bufs] and [var_vals],
    returning the captured return value.

    On the first call, intermediates are allocated and graph
    batching is attempted.  Input buffer slots are cleared after
    execution.

    Raises {!Jit_error} if [input_bufs] does not match the
    captured input count, sizes, dtypes, or devices. *)

(** {1:capture Capture state} *)

val is_capturing : unit -> bool
(** [is_capturing ()] is [true] iff a {!Tiny_jit} capture is
    in progress. *)

val add_linear : Tolk_ir.Tensor.t -> unit
(** [add_linear linear] records [linear] into the active capture.

    Raises [Failure] if no capture is in progress. *)

(** {1:tiny_jit TinyJit} *)

(** The JIT wrapper.

    Wraps a function and transparently captures its computation
    schedule on the second call.  Subsequent calls replay the
    compiled schedule with fresh input buffers. *)
type 'a tiny_jit

val captured : 'a tiny_jit -> 'a captured_jit option
(** [captured t] is [t]'s captured schedule, or [None] if [t] has
    not yet completed the capture phase. *)

val jit_cache : 'a captured_jit -> exec_item array
(** [jit_cache t] is [t]'s compiled schedule.  Buffer slots in
    these items are updated in-place on each replay. *)

val create :
  device:Device.t ->
  get_program:(Tolk_ir.Kernel.t -> Program_spec.t) ->
  ?fxn:(Device.Buffer.t array -> (string * int) list -> 'a) ->
  ?captured:'a captured_jit ->
  ?prune:bool ->
  ?optimize:bool ->
  unit ->
  'a tiny_jit
(** [create ~device ~get_program ?fxn ?captured ?prune
    ?optimize ()] is a JIT wrapper.

    Provide either [fxn] (the function to JIT) or [captured]
    (a pre-captured schedule).  When [captured] is provided,
    execution starts at the replay phase (cnt=2).

    {ul
    {- [prune] removes kernels whose outputs are not reachable
       from the inputs.  Defaults to [false].}
    {- [optimize] re-runs the memory planner after capture for
       tighter allocation.  Defaults to [false].}}

    Raises [Invalid_argument] if neither [fxn] nor [captured]
    is provided. *)

val reset : 'a tiny_jit -> unit
(** [reset t] resets [t] to the warmup phase, discarding any
    captured schedule.

    Raises [Invalid_argument] if [t] was created without a
    function. *)

val call :
  'a tiny_jit ->
  Device.Buffer.t array ->
  (string * int) list ->
  buffers:(Tolk_ir.Tensor.t -> Device.Buffer.t option) ->
  'a
(** [call t input_bufs var_vals ~buffers] executes [t] with
    [input_bufs], variable bindings [var_vals], and tensor-to-buffer
    mapping [buffers].

    {ul
    {- {e Warmup} (cnt=0): calls the wrapped function eagerly.}
    {- {e Capture} (cnt=1): calls the function under the capture
       handler, converts recorded linears to a compiled schedule,
       runs memory planning, executes, and stores a
       {!captured_jit}.}
    {- {e Exec} (cnt>=2): validates inputs against the capture
       and replays via {!exec_captured}.}}

    [buffers] maps tensor IR nodes to device buffers.  It is used
    during the capture phase to resolve the schedule; ignored on
    warmup and replay.

    Raises {!Jit_error} if:
    {ul
    {- capture is attempted while another capture is in progress,}
    {- the capture produces no linears,}
    {- inputs mismatch on replay (count, size, dtype, or
       device).}} *)
