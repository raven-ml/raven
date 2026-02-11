open Import

let op_pad_float64
    a_arr
    out_arr
    va
    vout
    padding
    start_idx
    end_idx
  =
  let in_offset = View.offset va in
  let out_offset = View.offset vout in

  let src_shape = shape va in
  let dst_strides = View.strides vout in
  let src_strides = View.strides va in

  let rank = Array.length src_shape in
  let md_index = Array.make rank 0 in

  for k = start_idx to end_idx - 1 do
    (* 1. Convert linear index -> source multi-index *)
    Shape.unravel_index_into k src_shape md_index;

    (* 2. Compute source linear index *)
    let src_lin = Shape.ravel_index md_index src_strides in
    let v =
      Array.unsafe_get a_arr (in_offset + src_lin)
    in

    (* 3. Shift coordinates by padding *)
    for d = 0 to rank - 1 do
      let before, _ = padding.(d) in
      md_index.(d) <- md_index.(d) + before
    done;

    (* 4. Compute destination linear index *)
    let dst_lin = Shape.ravel_index md_index dst_strides in

    (* 5. Write *)
    Array.unsafe_set out_arr (out_offset + dst_lin) v
  done
