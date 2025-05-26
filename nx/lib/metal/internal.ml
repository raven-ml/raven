open Nx_core

type metal_buffer = { buffer : Metal.Buffer.t; size_bytes : int }

type kernel_cache = {
  mutable binary_f32 : (string, Metal.Function.t) Hashtbl.t;
  mutable binary_f64 : (string, Metal.Function.t) Hashtbl.t;
  mutable binary_i32 : (string, Metal.Function.t) Hashtbl.t;
  mutable binary_i64 : (string, Metal.Function.t) Hashtbl.t;
  mutable unary_f32 : (string, Metal.Function.t) Hashtbl.t;
  mutable unary_f64 : (string, Metal.Function.t) Hashtbl.t;
  mutable unary_i32 : (string, Metal.Function.t) Hashtbl.t;
  mutable unary_i64 : (string, Metal.Function.t) Hashtbl.t;
  mutable reduce : (string, Metal.Function.t) Hashtbl.t;
  mutable special : (string, Metal.Function.t) Hashtbl.t;
}

type context = {
  device : Metal.Device.t;
  queue : Metal.CommandQueue.t;
  library : Metal.Library.t;
  kernels : kernel_cache;
  mutable pool : Buffer_pool.t;
}

type ('a, 'b) t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : metal_buffer;
  view : View.t;
}

let dtype t = t.dtype
let view t = t.view
let shape t = View.shape t.view
let numel t = View.numel t.view
let buffer t = t.buffer

let create_kernel_cache () =
  {
    binary_f32 = Hashtbl.create 32;
    binary_f64 = Hashtbl.create 32;
    binary_i32 = Hashtbl.create 32;
    binary_i64 = Hashtbl.create 32;
    unary_f32 = Hashtbl.create 32;
    unary_f64 = Hashtbl.create 32;
    unary_i32 = Hashtbl.create 32;
    unary_i64 = Hashtbl.create 32;
    reduce = Hashtbl.create 32;
    special = Hashtbl.create 32;
  }

let dtype_to_metal_type : type a b. (a, b) Dtype.t -> string = function
  | Dtype.Float16 -> "half"
  | Dtype.Float32 -> "float"
  | Dtype.Float64 -> "double"
  | Dtype.Int32 -> "int"
  | Dtype.Int64 -> "long"
  | Dtype.UInt8 -> "uchar"
  | Dtype.UInt16 -> "ushort"
  | Dtype.Int8 -> "char"
  | Dtype.Int16 -> "short"
  | Dtype.Int -> "int"
  | Dtype.NativeInt -> "long"
  | Dtype.Complex32 | Dtype.Complex64 ->
      failwith "dtype_to_metal_type: complex types not supported"

let sizeof_dtype : type a b. (a, b) Dtype.t -> int = function
  | Dtype.Float16 -> 2
  | Dtype.Float32 -> 4
  | Dtype.Float64 -> 8
  | Dtype.Int32 -> 4
  | Dtype.Int64 -> 8
  | Dtype.UInt8 -> 1
  | Dtype.UInt16 -> 2
  | Dtype.Int8 -> 1
  | Dtype.Int16 -> 2
  | Dtype.Int -> Sys.word_size / 8
  | Dtype.NativeInt -> Sys.word_size / 8
  | Dtype.Complex32 -> 8
  | Dtype.Complex64 -> 16

let copy_from_bigarray : type a b.
    context ->
    metal_buffer ->
    (a, b, Bigarray.c_layout) Bigarray.Array1.t ->
    unit =
 fun ctx mbuf ba ->
  (* For shared memory buffers, we can directly copy using Bigarray *)
  let metal_buf : Metal.Buffer.t = mbuf.buffer in
  let contents = Metal.Buffer.contents metal_buf in
  let size = Bigarray.Array1.dim ba in
  let kind = Bigarray.Array1.kind ba in
  (* Create a bigarray view of the Metal buffer memory *)
  (* TODO: Remove Obj.magic once Metal bindings are fixed to return properly typed pointers *)
  let metal_ba =
    Ctypes.bigarray_of_ptr Ctypes.array1 size kind (Obj.magic contents)
  in
  (* Copy data from source to Metal buffer *)
  Bigarray.Array1.blit ba metal_ba

let copy_to_bigarray : type a b.
    (a, b) t -> (a, b, Bigarray.c_layout) Bigarray.Array1.t -> unit =
 fun t ba ->
  (* For shared memory buffers, we can directly copy using Bigarray *)
  let contents = Metal.Buffer.contents t.buffer.buffer in
  let size = Bigarray.Array1.dim ba in
  let kind = Bigarray.Array1.kind ba in
  (* Create a bigarray view of the Metal buffer memory *)
  (* TODO: Remove Obj.magic once Metal bindings are fixed to return properly typed pointers *)
  let metal_ba =
    Ctypes.bigarray_of_ptr Ctypes.array1 size kind (Obj.magic contents)
  in
  (* Copy data from Metal buffer to destination *)
  Bigarray.Array1.blit metal_ba ba

let with_command_buffer ctx f =
  let buffer = Metal.CommandBuffer.on_queue ctx.queue in
  let result = f buffer in
  Metal.CommandBuffer.commit buffer;
  Metal.CommandBuffer.wait_until_completed buffer;
  result

let compute_thread_groups numel =
  let threads_per_group = 256 in
  let thread_groups = (numel + threads_per_group - 1) / threads_per_group in
  (threads_per_group, thread_groups)
