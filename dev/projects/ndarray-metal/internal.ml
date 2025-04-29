open Bigarray
open Ndarray_core

type ('a, 'b) t = {
  buffer : Metal.buffer;
  host_buffer : ('a, 'b) buffer;
  descriptor : ('a, 'b) descriptor;
}

type context = Metal_context.t = {
  device : Metal.device;
  library : Metal.library;
  command_queue : Metal.command_queue;
  pipeline_cache : (string, Metal.pipeline_state) Hashtbl.t;
}

type metal_dtype = Float32 | Int32

let metal_dtype_to_suffix = function Float32 -> "float32" | Int32 -> "int32"
let metal_dtype_size_in_bytes = function Float32 -> 4 | Int32 -> 4

let is_contiguous shape strides =
  let ndim = Array.length shape in
  if ndim = 0 then true
  else
    let expected_strides = Array.make ndim 0 in
    let s = ref 1 in
    let contig = ref true in
    for i = ndim - 1 downto 0 do
      expected_strides.(i) <- !s;
      let current_dim_size = max 1 shape.(i) in
      s := !s * current_dim_size
    done;
    for i = 0 to ndim - 1 do
      if shape.(i) > 1 && strides.(i) <> expected_strides.(i) then
        contig := false
    done;
    !contig

let compute_broadcasted_strides input_shape input_strides output_shape =
  let n_out = Array.length output_shape in
  let n_in = Array.length input_shape in
  let broadcasted_strides = Array.make n_out 0 in
  for i = 0 to n_out - 1 do
    let j = i - (n_out - n_in) in
    if j >= 0 && input_shape.(j) = output_shape.(i) then
      broadcasted_strides.(i) <- input_strides.(j)
    else if j >= 0 && input_shape.(j) = 1 then broadcasted_strides.(i) <- 0
    else if j < 0 (* Implicit dimension of size 1 *) then
      broadcasted_strides.(i) <- 0
    else
      failwith
        "Invalid shape for broadcasting (should not happen if shapes are \
         pre-broadcasted)"
  done;
  broadcasted_strides

let create_int32_buffer ctx data =
  let data = Array.map Int32.of_int data in
  let len = Array.length data in
  if len = 0 then
    Metal.create_buffer ctx.device 1L [ Metal.Storage_Mode_Shared ]
  else
    let ba = Bigarray.Array1.of_array Bigarray.int32 Bigarray.c_layout data in
    Metal.create_buffer_with_data ctx.device ba [ Metal.Storage_Mode_Shared ]

let create_int64_buffer ctx data =
  let data = Array.map Int64.of_int data in
  let len = Array.length data in
  if len = 0 then
    Metal.create_buffer ctx.device 1L [ Metal.Storage_Mode_Shared ]
  else
    let ba = Bigarray.Array1.of_array Bigarray.int64 Bigarray.c_layout data in
    Metal.create_buffer_with_data ctx.device ba [ Metal.Storage_Mode_Shared ]

let create_bool_buffer ctx data =
  let len = Array.length data in
  if len = 0 then
    Metal.create_buffer ctx.device 1L [ Metal.Storage_Mode_Shared ]
  else
    let int_data = Array.map (fun b -> if b then 1 else 0) data in
    let ba =
      Bigarray.Array1.of_array Bigarray.int8_unsigned Bigarray.c_layout int_data
    in
    Metal.create_buffer_with_data ctx.device ba [ Metal.Storage_Mode_Shared ]

(** Helper to convert Ndarray dtype to Metal_ops dtype *)
let ndarray_dtype_to_metal_dtype : type a b.
    (a, b) Ndarray_core.dtype -> metal_dtype = function
  | Ndarray_core.Float32 -> Float32
  | Ndarray_core.Int32 -> Int32
  | _ -> failwith "Unsupported data type for Metal Ops"

let buffer t = t.buffer
let host_buffer t = t.host_buffer
let descriptor t = t.descriptor
let shape t = t.descriptor.shape
let dtype t = t.descriptor.dtype
let strides t = t.descriptor.strides
let offset t = t.descriptor.offset
let layout t = t.descriptor.layout

(** Helper to create a Metal buffer view (shared memory) onto an Ndarray *)
let metal_buffer_from_host ctx buffer =
  let nbytes = Bigarray.Array1.size_in_bytes buffer in
  if nbytes = 0 then
    Metal.create_buffer ctx.device 1L [ Metal.Storage_Mode_Shared ]
  else
    Metal.create_buffer_with_data ctx.device buffer
      [ Metal.Storage_Mode_Shared ]

let create : type a b.
    context -> (a, b) dtype -> int array -> a array -> (a, b) t =
 fun ctx dtype shape array ->
  let host_buffer : (a, b) buffer = create_buffer dtype (Array.length array) in
  let kind = kind_of_dtype dtype in
  let buffer =
    metal_buffer_from_host ctx (Bigarray.Array1.of_array kind c_layout array)
  in
  let strides = compute_c_strides shape in
  let descriptor =
    { dtype; shape; layout = C_contiguous; strides; offset = 0 }
  in
  { buffer; host_buffer; descriptor }

let empty : type a b. context -> (a, b) dtype -> int array -> (a, b) t =
 fun ctx dtype shape ->
  let size = Array.fold_left ( * ) 1 shape in
  let host_buffer : (a, b) buffer = create_buffer dtype size in
  let kind = kind_of_dtype dtype in
  let buffer =
    metal_buffer_from_host ctx (Bigarray.Array1.create kind c_layout size)
  in
  let strides = compute_c_strides shape in
  let descriptor =
    { dtype; shape; layout = C_contiguous; strides; offset = 0 }
  in
  { host_buffer; buffer; descriptor }

let full : type a b. context -> (a, b) dtype -> int array -> a -> (a, b) t =
 fun ctx dtype shape value ->
  let t = empty ctx dtype shape in
  let host_buffer = t.host_buffer in
  Array1.fill host_buffer value;
  t

let empty_like ctx t = empty ctx (dtype t) (shape t)
