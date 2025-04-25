open Bigarray
open Ndarray_core
open Internal

let iter_slices shape axis f =
  let ndim = Array.length shape in
  let slice_shape =
    Array.init (ndim - 1) (fun i ->
        if i < axis then shape.(i) else shape.(i + 1))
  in
  iter_multi_indices slice_shape f

let base_offset_of_slice ~axis ~shape ~strides ~offset slice_idx =
  let ndim = Array.length shape in
  let full = Array.make ndim 0 in
  for i = 0 to ndim - 2 do
    let dim = if i < axis then i else i + 1 in
    full.(dim) <- slice_idx.(i)
  done;
  offset + md_to_linear full strides

let sort _context ~axis (t : ('a, 'b) t) (out : ('a, 'b) t) =
  let desc_t = descriptor t in
  let shape = desc_t.shape in
  let strides_t = desc_t.strides in
  let off_t = desc_t.offset in
  let desc_o = descriptor out in
  let strides_o = desc_o.strides in
  let off_o = desc_o.offset in
  let len = shape.(axis) in
  let step_in = strides_t.(axis) in
  let step_out = strides_o.(axis) in
  let buf_t = buffer t in
  let buf_o = buffer out in

  iter_slices shape axis (fun slice_idx ->
      let base_i =
        base_offset_of_slice ~axis ~shape ~strides:strides_t ~offset:off_t
          slice_idx
      in
      let base_o =
        base_offset_of_slice ~axis ~shape ~strides:strides_o ~offset:off_o
          slice_idx
      in
      let tmp =
        Array.init len (fun j ->
            Array1.unsafe_get buf_t (base_i + (j * step_in)))
      in
      Array.sort compare tmp;
      for j = 0 to len - 1 do
        Array1.unsafe_set buf_o (base_o + (j * step_out)) tmp.(j)
      done);

  ()

let argsort _context ~axis (t : ('a, 'b) t)
    (out : (int64, Bigarray.int64_elt) t) =
  let desc_t = descriptor t in
  let shape = desc_t.shape in
  let strides_t = desc_t.strides in
  let off_t = desc_t.offset in
  let desc_o = descriptor out in
  let strides_o = desc_o.strides in
  let off_o = desc_o.offset in
  let len = shape.(axis) in
  let step_in = strides_t.(axis) in
  let step_out = strides_o.(axis) in
  let buf_t = buffer t in
  let buf_o = buffer out in

  iter_slices shape axis (fun slice_idx ->
      let base_i =
        base_offset_of_slice ~axis ~shape ~strides:strides_t ~offset:off_t
          slice_idx
      in
      let base_o =
        base_offset_of_slice ~axis ~shape ~strides:strides_o ~offset:off_o
          slice_idx
      in

      let idxs = Array.init len Int64.of_int in
      Array.sort
        (fun i j ->
          let vi =
            Array1.unsafe_get buf_t (base_i + (Int64.to_int i * step_in))
          in
          let vj =
            Array1.unsafe_get buf_t (base_i + (Int64.to_int j * step_in))
          in
          compare vi vj)
        idxs;
      for k = 0 to len - 1 do
        Array1.unsafe_set buf_o (base_o + (k * step_out)) idxs.(k)
      done);

  ()

let arg_extreme ~is_max _context ~axis (t : ('a, 'b) t)
    (out : (int64, Bigarray.int64_elt) t) =
  let desc_t = descriptor t in
  let shape = desc_t.shape in
  let strides_t = desc_t.strides in
  let off_t = desc_t.offset in
  let desc_o = descriptor out in
  let strides_o = desc_o.strides in
  let off_o = desc_o.offset in
  let len = shape.(axis) in
  let step_in = strides_t.(axis) in
  let buf_t = buffer t in
  let buf_o = buffer out in

  iter_slices shape axis (fun slice_idx ->
      let base_i =
        base_offset_of_slice ~axis ~shape ~strides:strides_t ~offset:off_t
          slice_idx
      in
      let out_lin = off_o + md_to_linear slice_idx strides_o in
      let best_j = ref 0 in
      let best_v = ref (Array1.unsafe_get buf_t base_i) in
      for j = 1 to len - 1 do
        let v = Array1.unsafe_get buf_t (base_i + (j * step_in)) in
        if (is_max && v > !best_v) || ((not is_max) && v < !best_v) then (
          best_v := v;
          best_j := j)
      done;
      Array1.unsafe_set buf_o out_lin (Int64.of_int !best_j))

let argmax context ~axis t out = arg_extreme ~is_max:true context ~axis t out
let argmin context ~axis t out = arg_extreme ~is_max:false context ~axis t out
