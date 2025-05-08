open Nx_core

type ('a, 'b) buffer =
  | Cpu_buffer : ('a, 'b) Nx_native.buffer -> ('a, 'b) buffer

type context = Cpu_context : Nx_native.context -> context

type ('a, 'b) t = {
  dtype : ('a, 'b) Dtype.dtype;
  buffer : ('a, 'b) buffer;
  view : View.view;
}

type _ Effect.t +=
  | Const :
      int array * ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
      -> ('a, 'b) t Effect.t
  | Buffer : ('a, 'b) t -> ('a, 'b) t Effect.t
  | Reshape : ('a, 'b) t * int array -> ('a, 'b) t Effect.t
  | Expand : ('a, 'b) t * int array -> ('a, 'b) t Effect.t
  | Add : ('a, 'b) t * ('a, 'b) t * ('a, 'b) t -> unit Effect.t
  | Reduce_sum : {
      x : ('a, 'b) t;
      axis : int array;
      keepdims : bool;
    }
      -> ('a, 'b) t Effect.t

let const context shape buffer =
  try Effect.perform (Const (shape, buffer))
  with Effect.Unhandled _ -> (
    match context with
    | Cpu_context ctx ->
        let t_native = Nx_native.const ctx shape buffer in
        {
          dtype = t_native.dtype;
          buffer = Cpu_buffer t_native.buffer;
          view = t_native.view;
        })

let reshape context t new_shape =
  try Effect.perform (Reshape (t, new_shape))
  with Effect.Unhandled _ -> (
    match context with
    | Cpu_context ctx ->
        let t_native = Nx_native.reshape ctx t new_shape in
        {
          dtype = t_native.dtype;
          buffer = Cpu_buffer t_native.buffer;
          view = t_native.view;
        })

let expand context t new_shape =
  try Effect.perform (Expand (t, new_shape))
  with Effect.Unhandled _ -> (
    match context with
    | Cpu_context ctx ->
        let t_native = Nx_native.expand ctx t new_shape in
        {
          dtype = t_native.dtype;
          buffer = Cpu_buffer t_native.buffer;
          view = t_native.view;
        })

let add context a b out =
  try Effect.perform (Add (a, b, out))
  with Effect.Unhandled _ -> (
    match (context, a, b, out) with
    | Cpu_context ctx, Cpu_buffer a_buf, Cpu_buffer b_buf, Cpu_buffer out_buf ->
        let a = { buffer = a_buf; dtype = a.dtype; view = a.view }
        and b = { buffer = b_buf; dtype = b.dtype; view = b.view }
        and out = { buffer = out_buf; dtype = out.dtype; view = out.view } in
        Nx_native.add ctx a b out;
        ())
