open Bigarray
open Ndarray_core

type ('a, 'b) t = {
  descriptor : ('a, 'b) Ndarray_core.descriptor;
  buffer : ('a, 'b) Ndarray_core.buffer;
}

type context = { pool : Parallel.pool }

let buffer t = t.buffer
let descriptor t = t.descriptor
let shape t = t.descriptor.shape
let dtype t = t.descriptor.dtype
let strides t = t.descriptor.strides
let offset t = t.descriptor.offset
let layout t = t.descriptor.layout

let size t =
  let n = Array.length t.descriptor.shape in
  if n = 0 then 1 else Array.fold_left ( * ) 1 t.descriptor.shape

let stride axis t =
  if axis < 0 then invalid_arg "axis must be non-negative";
  if axis >= Array.length t.descriptor.strides then
    invalid_arg "axis out of bounds";
  let stride = Array.unsafe_get t.descriptor.strides axis in
  stride

let dims t = t.descriptor.shape

let dim axis t =
  if axis < 0 then invalid_arg "axis must be non-negative";
  if axis >= Array.length t.descriptor.shape then
    invalid_arg "axis out of bounds";
  let dim = Array.unsafe_get t.descriptor.shape axis in
  dim

let ndim t = Array.length t.descriptor.shape
let is_c_contiguous t = Ndarray_core.is_c_contiguous t.descriptor

let create : type a b. (a, b) dtype -> int array -> a array -> (a, b) t =
 fun dtype shape array ->
  let buffer : (a, b) buffer = create_buffer dtype (Array.length array) in
  let strides = compute_c_strides shape in
  let descriptor =
    { dtype; shape; layout = C_contiguous; strides; offset = 0 }
  in
  { buffer; descriptor }

let empty : type a b. (a, b) dtype -> int array -> (a, b) t =
 fun dtype shape ->
  let size = Array.fold_left ( * ) 1 shape in
  let buffer : (a, b) buffer = create_buffer dtype size in
  let strides = compute_c_strides shape in
  let descriptor =
    { dtype; shape; layout = C_contiguous; strides; offset = 0 }
  in
  { buffer; descriptor }

let full : type a b. (a, b) dtype -> int array -> a -> (a, b) t =
 fun dtype shape value ->
  let t = empty dtype shape in
  let buffer = t.buffer in
  Array1.fill buffer value;
  t

let empty_like t = empty (dtype t) (shape t)

let copy : type a b. (a, b) t -> (a, b) t =
 fun t ->
  let total_size = Array.fold_left ( * ) 1 (shape t) in
  let new_buffer = create_buffer (dtype t) total_size in
  let new_shape = Array.copy (shape t) in
  let new_strides = compute_c_strides new_shape in
  let new_t =
    {
      buffer = new_buffer;
      descriptor =
        {
          dtype = dtype t;
          shape = new_shape;
          layout = C_contiguous;
          strides = new_strides;
          offset = 0;
        };
    }
  in
  if is_c_contiguous t && offset t = 0 then (
    Array1.blit t.buffer new_t.buffer;
    new_t)
  else
    let n_dims = Array.length (shape t) in
    let current_md_idx = Array.make n_dims 0 in
    let rec copy_slice dim =
      if dim = n_dims then
        let src_linear_idx =
          md_to_linear current_md_idx (strides t) + offset t
        in
        let dst_linear_idx =
          md_to_linear current_md_idx (strides new_t) + offset new_t
        in
        new_buffer.{dst_linear_idx} <- t.buffer.{src_linear_idx}
      else
        for i = 0 to (shape t).(dim) - 1 do
          current_md_idx.(dim) <- i;
          copy_slice (dim + 1)
        done
    in
    if total_size > 0 then copy_slice 0;
    new_t

let fill : type a b. a -> (a, b) t -> unit =
 fun value t ->
  (* This needs to respect strides/offset if not contiguous *)
  if is_c_contiguous t && offset t = 0 then
    let buffer = t.buffer in
    Array1.fill buffer value
  else
    (* Generic case: Iterate and fill element by element *)
    (* This is inefficient but correct for all layouts/offsets *)
    let total_size = size t in
    if total_size > 0 then
      let n_dims = Array.length (shape t) in
      let current_md_idx = Array.make n_dims 0 in
      let rec fill_slice dim =
        if dim = n_dims then
          let linear_idx = md_to_linear current_md_idx (strides t) + offset t in
          t.buffer.{linear_idx} <- value
        else
          for i = 0 to (shape t).(dim) - 1 do
            current_md_idx.(dim) <- i;
            fill_slice (dim + 1)
          done
      in
      fill_slice 0

let blit : type a b. (a, b) t -> (a, b) t -> unit =
 fun src dst ->
  let src_desc = descriptor src in
  let dst_desc = descriptor dst in
  let n_dims = Array.length src_desc.shape in

  if n_dims <> Array.length dst_desc.shape then
    invalid_arg "blit: tensors must have the same number of dimensions";
  if src_desc.shape <> dst_desc.shape then
    invalid_arg "blit: tensors must have the same shape";

  let total_size = size src in
  if total_size = 0 then ()
  else
    let src_buffer = buffer src in
    let dst_buffer = buffer dst in
    let src_strides = strides src in
    let dst_strides = strides dst in
    let src_offset = offset src in
    let dst_offset = offset dst in
    let shape = shape src in

    if n_dims = 0 then dst_buffer.{dst_offset} <- src_buffer.{src_offset}
    else
      let current_md_idx = Array.make n_dims 0 in
      let rec blit_slice dim =
        if dim = n_dims then (
          let src_linear_idx = ref src_offset in
          let dst_linear_idx = ref dst_offset in
          for i = 0 to n_dims - 1 do
            src_linear_idx :=
              !src_linear_idx + (current_md_idx.(i) * src_strides.(i));
            dst_linear_idx :=
              !dst_linear_idx + (current_md_idx.(i) * dst_strides.(i))
          done;
          dst_buffer.{!dst_linear_idx} <- src_buffer.{!src_linear_idx})
        else
          for i = 0 to shape.(dim) - 1 do
            current_md_idx.(dim) <- i;
            blit_slice (dim + 1)
          done
      in
      blit_slice 0

let zeros : type a b. (a, b) dtype -> int array -> (a, b) t =
 fun dtype shape ->
  let t = empty dtype shape in
  fill (zero t.descriptor.dtype) t;
  t

let ones : type a b. (a, b) dtype -> int array -> (a, b) t =
 fun dtype shape ->
  let t = empty dtype shape in
  fill (one t.descriptor.dtype) t;
  t

let astype : type a b c d. (c, d) dtype -> (a, b) t -> (c, d) t =
 fun new_dtype t ->
  let new_buffer = Ndarray_core.astype new_dtype (descriptor t) (buffer t) in
  { buffer = new_buffer; descriptor = { t.descriptor with dtype = new_dtype } }

let broadcast_to t new_shape =
  let new_descriptor = Ndarray_core.broadcast_to t.descriptor new_shape in
  { t with descriptor = new_descriptor }

let is_scalar t = Ndarray_core.is_scalar t.descriptor
