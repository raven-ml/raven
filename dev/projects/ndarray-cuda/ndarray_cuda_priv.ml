type context = { device : Cuda.cu_device; context : Cuda.cu_context }

(** Initialize CUDA context and set it as current *)
let context =
  Cuda.cu_init 0;
  (* Initialize the CUDA driver API *)
  let device = Cuda.cu_device_get 0 in
  let context = Cuda.cu_ctx_create 0 device in
  (* cu_ctx_create sets the context as current, but we ensure it explicitly *)
  Cuda.cu_ctx_set_current context;
  { device; context }

(** Finalize CUDA context *)
let finalize_context ctx = Cuda.cu_ctx_destroy ctx.context

(** CUDA handles module *)
module Cuda_handles = struct
  let () =
    Printf.printf "PTX data length: %d\n%!" (String.length Cuda_kernels.data)

  (* Module and function handles initialized after context is set *)
  let module_handle = Cuda.cu_module_load_data_ex Cuda_kernels.data [||]

  let add_float32_handle =
    Cuda.cu_module_get_function module_handle "add_float32"
end

(** Calculate the number of elements from shape *)
let num_elements shape = Array.fold_left ( * ) 1 shape

(** Get element size based on dtype *)
let element_size : type a b. (a, b) dtype -> int = function
  | Float32 -> 4
  | _ -> failwith "Unsupported dtype"

(** Implementation module *)
module Impl = struct
  type ('a, 'b) t = {
    dtype : ('a, 'b) dtype; (* Data type of elements *)
    shape : int array; (* Array dimensions *)
    data : Cuda.cu_deviceptr; (* Pointer to GPU memory *)
  }

  (** Create an array on the GPU *)
  let create : type a b. (a, b) dtype -> int array -> a array -> (a, b) t =
   fun dtype shape data ->
    let n = num_elements shape in
    if Array.length data <> n then failwith "Data length mismatch";
    let element_size = element_size dtype in
    let bytes = Int64.of_int (n * element_size) in
    let data_ptr = Cuda.cu_mem_alloc bytes in
    let host_data =
      match dtype with
      | Float32 ->
          let ba =
            Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout data
          in
          let bytes_data = Bytes.create (n * element_size) in
          for i = 0 to n - 1 do
            let float_val = Bigarray.Array1.get ba i in
            let int_val = Int32.bits_of_float float_val in
            Bytes.set_int32_le bytes_data (i * 4) int_val
          done;
          bytes_data
      | _ -> failwith "Unsupported dtype"
    in
    Cuda.cu_memcpy_H_to_D data_ptr host_data bytes;
    let arr = { dtype; shape; data = data_ptr } in
    Gc.finalise (fun a -> Cuda.cu_mem_free a.data) arr;
    arr

  (** Create an empty array on the GPU *)
  let empty dtype shape =
    let n = num_elements shape in
    let bytes = Int64.of_int (n * element_size dtype) in
    let data = Cuda.cu_mem_alloc bytes in
    let arr = { dtype; shape; data } in
    Gc.finalise (fun a -> Cuda.cu_mem_free a.data) arr;
    arr

  let full _ _ _ = failwith "Not implemented"

  (** Properties *)
  let ndim arr = Array.length arr.shape

  let dim arr i = arr.shape.(i)
  let dtype arr = arr.dtype

  (** Element access *)
  let get _ = failwith "Not implemented"

  let set _ = failwith "Not implemented"

  (** Element-wise addition *)
  let add : type a b. (a, b) t -> (a, b) t -> (a, b) t =
   fun a b ->
    if a.shape <> b.shape then failwith "Shape mismatch";
    if a.dtype <> b.dtype then failwith "Dtype mismatch";
    match a.dtype with
    | Float32 ->
        let c = empty a.dtype a.shape in
        let n = num_elements a.shape in
        (* Allocate device memory for n *)
        let n_dev = Cuda.cu_mem_alloc 4L in
        (* 4 bytes for a 32-bit int *)
        let n_bytes = Bytes.create 4 in
        Bytes.set_int32_le n_bytes 0 (Int32.of_int n);
        Cuda.cu_memcpy_H_to_D n_dev n_bytes 4L;
        (* Copy n to device *)
        (* Prepare arguments as int64 device pointers *)
        let args_int64 = [| a.data; b.data; c.data; n_dev |] in
        (* Launch kernel *)
        let block_dim = 256 in
        let grid_dim = (n + block_dim - 1) / block_dim in
        Cuda.cu_launch_kernel Cuda_handles.add_float32_handle grid_dim 1
          1 (* Grid dimensions *)
          block_dim 1 1 (* Block dimensions *)
          args_int64;
        (* Free device memory for n *)
        Cuda.cu_mem_free n_dev;
        c
    | _ -> failwith "Unsupported dtype for add"

  let sub _ _ = failwith "Not implemented"
  let mul _ _ = failwith "Not implemented"
  let div _ _ = failwith "Not implemented"

  (** Reductions *)
  let sum ?axis:_ ?keepdims:_ _arr = failwith "Not implemented"

  (** Matrix operations *)
  let matmul _ _ = failwith "Not implemented"
end

open Impl
include Make (Impl)

(** Convert GPU array to host Ndarray_c.t *)
let to_host : type a b. (a, b) t -> (a, b) Ndarray_c.t =
 fun arr ->
  let n = num_elements arr.shape in
  let element_size = element_size arr.dtype in
  let bytes = n * element_size in
  let host_data = Bytes.create bytes in
  Cuda.cu_memcpy_D_to_H host_data arr.data (Int64.of_int bytes);
  let layout = Ndarray_c.C_contiguous in
  let byte_order = Ndarray_c.Little_endian in
  let access_mode = Ndarray_c.Read_only in
  (* TODO: That doesn't work, the memory will be gced when host_data is
     deleted *)
  Ndarray_c.create_from_external host_data arr.dtype arr.shape layout byte_order
    access_mode

(** Overriding functions *)
let pp fmt arr = Ndarray_c.pp fmt (to_host arr)

let print arr = Ndarray_c.print (to_host arr)
let to_string arr = Ndarray_c.to_string (to_host arr)
