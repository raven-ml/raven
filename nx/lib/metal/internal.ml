open Nx_core

type metal_buffer = { buffer : Metal.Buffer.t; size_bytes : int }

(* Nested hashtable structure: category -> kernel_name -> function *)
type kernel_cache = (string, (string, Metal.Function.t) Hashtbl.t) Hashtbl.t

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
  let cache = Hashtbl.create 8 in
  (* Initialize the main categories *)
  Hashtbl.add cache "binary" (Hashtbl.create 32);
  Hashtbl.add cache "unary" (Hashtbl.create 32);
  Hashtbl.add cache "reduce" (Hashtbl.create 32);
  Hashtbl.add cache "special" (Hashtbl.create 128);
  cache

type dtype_info = {
  metal_name : string;
  size_bytes : int;
      (* Add more fields as needed, e.g., initial values for reductions *)
}

let get_dtype_info : type a b. (a, b) Dtype.t -> dtype_info = function
  | Dtype.Float16 -> { metal_name = "half"; size_bytes = 2 }
  | Dtype.Float32 -> { metal_name = "float"; size_bytes = 4 }
  | Dtype.Float64 -> { metal_name = "double"; size_bytes = 8 }
  | Dtype.Int32 -> { metal_name = "int"; size_bytes = 4 }
  | Dtype.Int64 -> { metal_name = "long"; size_bytes = 8 }
  | Dtype.UInt8 -> { metal_name = "uchar"; size_bytes = 1 }
  | Dtype.UInt16 -> { metal_name = "ushort"; size_bytes = 2 }
  | Dtype.Int8 -> { metal_name = "char"; size_bytes = 1 }
  | Dtype.Int16 -> { metal_name = "short"; size_bytes = 2 }
  | Dtype.Int ->
      if Sys.word_size = 64 then { metal_name = "long"; size_bytes = 8 }
      else { metal_name = "int"; size_bytes = 4 }
  | Dtype.NativeInt -> { metal_name = "long"; size_bytes = Sys.word_size / 8 }
  | Dtype.Complex32 -> { metal_name = "float2"; size_bytes = 8 }
  | Dtype.Complex64 -> { metal_name = "float2"; size_bytes = 16 }
(* Keep OCaml size for compatibility *)

(* Convenience functions for backward compatibility *)
let dtype_to_metal_type dtype = (get_dtype_info dtype).metal_name
let sizeof_dtype dtype = (get_dtype_info dtype).size_bytes

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

  (* Special handling for Complex64 - convert from double to float *)
  match kind with
  | Bigarray.Complex64 ->
      (* Convert Complex64 (double precision) to float2 for Metal *)
      let float_array = Ctypes.(from_voidp float contents) in
      for i = 0 to size - 1 do
        let c = Bigarray.Array1.get ba i in
        (* Write as float precision to Metal buffer *)
        let re_val = c.Complex.re in
        let im_val = c.Complex.im in
        let re_ptr = Ctypes.(float_array +@ (i * 2)) in
        let im_ptr = Ctypes.(float_array +@ ((i * 2) + 1)) in

        Ctypes.(re_ptr <-@ re_val);
        Ctypes.(im_ptr <-@ im_val)
      done
  | _ ->
      (* For other types, direct copy *)
      let metal_ba =
        Ctypes.bigarray_of_ptr Ctypes.array1 size kind (Obj.magic contents)
      in
      (* Copy data from source to Metal buffer *)
      Bigarray.Array1.blit ba metal_ba

let copy_to_bigarray : type a b.
    (a, b) t -> (a, b, Bigarray.c_layout) Bigarray.Array1.t -> unit =
 fun t ba ->
  (* Handle views correctly by considering offset and strides *)
  let view = t.view in
  let contents = Metal.Buffer.contents t.buffer.buffer in
  let kind = Bigarray.Array1.kind ba in

  (* Special handling for Complex64 *)
  match kind with
  | Bigarray.Complex64 ->
      (* Convert from float2 (Metal) to Complex64 (OCaml) *)
      let float_ptr = Ctypes.(from_voidp float contents) in
      let view_size = View.numel view in
      let offset = View.offset view in

      if View.is_c_contiguous view then
        (* For contiguous views, convert directly *)
        for i = 0 to view_size - 1 do
          let idx = offset + i in
          let re_ptr = Ctypes.(float_ptr +@ (idx * 2)) in
          let im_ptr = Ctypes.(float_ptr +@ ((idx * 2) + 1)) in
          let re = Ctypes.( !@ ) re_ptr in
          let im = Ctypes.( !@ ) im_ptr in
          Bigarray.Array1.set ba i Complex.{ re; im }
        done
      else
        (* For non-contiguous views, handle strides *)
        let shape = View.shape view in
        let strides = View.strides view in
        let ndim = Array.length shape in
        let rec copy_elements indices pos dst_idx =
          if pos = ndim then (
            let src_idx = ref offset in
            for d = 0 to ndim - 1 do
              src_idx := !src_idx + (indices.(d) * strides.(d))
            done;
            let re_ptr = Ctypes.(float_ptr +@ (!src_idx * 2)) in
            let im_ptr = Ctypes.(float_ptr +@ ((!src_idx * 2) + 1)) in
            let re = Ctypes.( !@ ) re_ptr in
            let im = Ctypes.( !@ ) im_ptr in
            Bigarray.Array1.set ba dst_idx Complex.{ re; im })
          else
            for i = 0 to shape.(pos) - 1 do
              indices.(pos) <- i;
              copy_elements indices (pos + 1)
                (dst_idx
                + i
                  * Array.fold_left ( * ) 1
                      (Array.sub shape (pos + 1) (ndim - pos - 1)))
            done
        in
        copy_elements (Array.make ndim 0) 0 0
  | _ ->
      let elem_size = sizeof_dtype t.dtype in
      (* Create a bigarray view of the entire Metal buffer *)
      let buffer_size = t.buffer.size_bytes / elem_size in
      let metal_ba =
        Ctypes.bigarray_of_ptr Ctypes.array1 buffer_size kind
          (Obj.magic contents)
      in

      (* Check if the view is contiguous AND the buffer has enough elements *)
      let view_size = View.numel view in
      let required_size = View.offset view + view_size in

      if View.is_c_contiguous view && required_size <= buffer_size then (
        (* For contiguous views with sufficient buffer size, we can do a direct
           copy *)
        let offset = View.offset view in
        if view_size > 0 then
          (* Create sub-arrays and blit *)
          let src = Bigarray.Array1.sub metal_ba offset view_size in
          Bigarray.Array1.blit src ba)
      else
        (* For non-contiguous views, we need to copy element by element *)
        (* NOTE: For better performance, callers should use Ops_movement.make_contiguous
       before calling copy_to_bigarray on non-contiguous views *)
        let shape = View.shape view in
        let strides = View.strides view in
        let offset = View.offset view in
        let ndim = Array.length shape in

        (* Helper to convert multi-dimensional index to linear index *)
        let rec copy_elements indices pos =
          if pos = ndim then (
            (* Calculate the source index using strides *)
            let src_idx = ref offset in
            for i = 0 to ndim - 1 do
              (* Only add to index if stride is non-zero (handles broadcast
                 dimensions) *)
              if strides.(i) <> 0 then
                src_idx := !src_idx + (indices.(i) * strides.(i))
            done;

            (* Calculate the linear index in the destination *)
            let dst_idx = ref 0 in
            let dst_stride = ref 1 in
            for i = ndim - 1 downto 0 do
              dst_idx := !dst_idx + (indices.(i) * !dst_stride);
              dst_stride := !dst_stride * shape.(i)
            done;

            (* Copy the element *)
            if !src_idx >= buffer_size then
              failwith
                (Printf.sprintf
                   "Source index %d out of bounds (buffer size: %d)" !src_idx
                   buffer_size)
            else if !dst_idx >= Bigarray.Array1.dim ba then
              failwith
                (Printf.sprintf
                   "Destination index %d out of bounds (ba size: %d)" !dst_idx
                   (Bigarray.Array1.dim ba))
            else
              Bigarray.Array1.set ba !dst_idx
                (Bigarray.Array1.get metal_ba !src_idx))
          else
            for
              (* Iterate through this dimension *)
              i = 0 to shape.(pos) - 1
            do
              indices.(pos) <- i;
              copy_elements indices (pos + 1)
            done
        in

        if View.numel view > 0 then copy_elements (Array.make ndim 0) 0

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

let uint32_array_to_buffer arr =
  let len = Array.length arr in
  let open Ctypes in
  let ptr = allocate_n uint32_t ~count:len in
  Array.iteri (fun i v -> ptr +@ i <-@ Unsigned.UInt32.of_int v) arr;
  to_voidp ptr
