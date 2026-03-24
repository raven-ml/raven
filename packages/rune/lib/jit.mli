(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** JIT compilation via effect handler.

    Intercepts {!Nx} tensor operations to build a computation graph, compiles
    it into optimized machine code, and replays the compiled schedule on
    subsequent calls.

    {b Usage:}
    {[
      let f_jit = Rune.jit f in
      let y1 = f_jit x1 in   (* warmup: execute eagerly *)
      let y2 = f_jit x2 in   (* capture: compile computation graph *)
      let y3 = f_jit x3 in   (* replay: fast, no recompilation *)
    ]}

    When no device is provided, the JIT captures the graph but falls back to
    eager execution. Pass [~device] to enable compiled execution. *)

val trace :
  ?device:Tolk.Device.t ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t)
(** [trace ?device f] returns a JIT-compiled version of [f].

    The returned function has the same type as [f] but compiles the
    computation graph on the second call and replays the compiled schedule
    on subsequent calls.

    [device] selects the execution backend. When omitted, the computation
    graph is still captured but execution falls back to the C backend.

    Raises [Invalid_argument] if input shapes or dtypes change after capture. *)

(** {1:inspection Inspecting computation graphs} *)

type traced = {
  tensor_graph : Tolk_ir.Tensor.t;
      (** High-level operation graph before scheduling. *)
  kernel_graph : Tolk_ir.Tensor.t;
      (** Scheduled graph with [Call] nodes containing kernel ASTs. *)
  rendered_source : string list;
      (** Rendered source code for each kernel (one per [Call] node). *)
}
(** Result of tracing a function through the JIT capture handler. *)

val trace_graph :
  ?device:Tolk.Device.t ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  traced
(** [trace_graph ?device f x] traces [f] applied to [x], capturing the
    computation graph without executing it.

    Returns the tensor graph, kernel graph, and rendered source for each
    kernel. Useful for debugging what the JIT produces, inspecting gradient
    graphs, or comparing against reference implementations.

    When [device] is omitted, renders using the C backend (clang). *)

val reset : unit -> unit
(** [reset ()] clears the JIT cache, forcing recompilation on the next call. *)
