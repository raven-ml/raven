(* nx_native.ml *)

open Nx_core

type ('a, 'b) buffer = ('a, 'b) Internal.buffer

type ('a, 'b) t = ('a, 'b) Internal.t = {
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t;
}

let view t = t.view
let dtype t = t.dtype
let buffer t = t.buffer
let with_view t view = { t with view }

type context = Internal.context

let create_context () = Internal.{ pool = Parallel.get_or_setup_pool () }

(* --- Backend Ops --- *)

let op_buffer _ctx dt size_in_elements =
  let kind = Dtype.kind_of_dtype dt in
  let ba = Bigarray.Array1.create kind Bigarray.c_layout size_in_elements in
  let initial_view =
    if size_in_elements = 0 then View.create [| 0 |]
      (* Consistent 0-element view *)
    else View.create [| size_in_elements |]
  in
  Internal.{ dtype = dt; buffer = ba; view = initial_view }

let op_const_scalar _ctx (value : 'a) dt =
  let kind = Dtype.kind_of_dtype dt in
  let ba = Bigarray.Array1.create kind Bigarray.c_layout 1 in
  Bigarray.Array1.set ba 0 value;
  let scalar_view = View.create [||] in
  (* 0-dim for scalar *)
  Internal.{ dtype = dt; buffer = ba; view = scalar_view }

(* Binary Ops *)

let op_add ctx a b =
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.add ctx a b out_tensor;
  out_tensor

let op_mul ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.mul ctx a b out_tensor;
  out_tensor

let op_idiv ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.idiv ctx a b out_tensor;
  out_tensor

let op_fdiv ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.fdiv ctx a b out_tensor;
  out_tensor

let op_max ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.max ctx a b out_tensor;
  out_tensor

let op_mod ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.modulo ctx a b out_tensor;
  out_tensor

let op_pow ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.pow ctx a b out_tensor;
  out_tensor

let op_cmplt ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx Dtype.uint8 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.cmplt ctx a b out_tensor;
  out_tensor

let op_cmpne ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx Dtype.uint8 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.cmpne ctx a b out_tensor;
  out_tensor

let op_xor ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.bit_xor ctx a b out_tensor;
  out_tensor

let op_or ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.bit_or ctx a b out_tensor;
  out_tensor

let op_and ctx a b =
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in
  let out_tensor =
    op_buffer ctx a.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_binary.bit_and ctx a b out_tensor;
  out_tensor

(* Unary Ops *)

let op_neg ctx input_t =
  let out_shape = Internal.shape input_t in
  let out_size = Internal.size input_t in
  let out_tensor =
    op_buffer ctx input_t.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.neg ctx input_t out_tensor;
  out_tensor

let op_log2 ctx input_t =
  let out_shape = Internal.shape input_t in
  let out_size = Internal.size input_t in
  let out_tensor =
    op_buffer ctx Dtype.float32 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.log2 ctx input_t out_tensor;
  out_tensor

let op_exp2 ctx input_t =
  let out_shape = Internal.shape input_t in
  let out_size = Internal.size input_t in
  let out_tensor =
    op_buffer ctx Dtype.float32 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.exp2 ctx input_t out_tensor;
  out_tensor

let op_sin ctx input_t =
  let out_shape = Internal.shape input_t in
  let out_size = Internal.size input_t in
  let out_tensor =
    op_buffer ctx Dtype.float32 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.sin ctx input_t out_tensor;
  out_tensor

let op_sqrt ctx input_t =
  let out_shape = Internal.shape input_t in
  let out_size = Internal.size input_t in
  let out_tensor =
    op_buffer ctx Dtype.float32 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.sqrt ctx input_t out_tensor;
  out_tensor

let op_recip ctx input_t =
  let out_shape = Internal.shape input_t in
  let out_size = Internal.size input_t in
  let out_tensor =
    op_buffer ctx Dtype.float32 out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_unary.recip ctx input_t out_tensor;
  out_tensor

(* Ternary Op *)
let op_where ctx cond if_true if_false =
  let out_shape = View.shape cond.view in
  let out_size = View.numel cond.view in
  let out_tensor =
    op_buffer ctx if_true.dtype out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  Ops_ternary.where ctx cond if_true if_false out_tensor;
  out_tensor

let fill_buffer_with_identity buf count identity_val =
  for i = 0 to count - 1 do
    Bigarray.Array1.unsafe_set buf i identity_val
  done

let op_sum ctx ~(axes : int array) ~(keepdims : bool) input_tensor =
  let input_shape = Internal.shape input_tensor in
  let input_rank = Array.length input_shape in
  let axes_to_reduce_normalized =
    Array.map (fun ax -> if ax < 0 then ax + input_rank else ax) axes
  in
  let axes_to_reduce =
    Array.of_list
      (List.sort_uniq Int.compare (Array.to_list axes_to_reduce_normalized))
  in

  let output_shape_logical =
    Array.mapi
      (fun i s ->
        if Array.mem i axes_to_reduce then
          if keepdims then 1 else -1 (* placeholder for removal *)
        else s)
      input_shape
    |> Array.to_list
    |> List.filter (( <> ) (-1))
    |> Array.of_list
  in
  let output_shape_final =
    if Array.length output_shape_logical = 0 then [||] else output_shape_logical
  in

  let output_numel = View.prod output_shape_final in
  let output_tensor =
    op_buffer ctx input_tensor.dtype output_numel |> fun t ->
    with_view t (View.create output_shape_final)
  in

  if output_numel > 0 then
    fill_buffer_with_identity
      (Internal.buffer output_tensor)
      output_numel
      (Dtype.zero input_tensor.dtype);

  Ops_reduce.sum ctx ~axes:axes_to_reduce ~keepdims input_tensor output_tensor;
  output_tensor

let op_reduce_max ctx ~(axes : int array) ~(keepdims : bool) input_tensor =
  let input_shape = Internal.shape input_tensor in
  let input_rank = Array.length input_shape in
  let axes_to_reduce_normalized =
    Array.map (fun ax -> if ax < 0 then ax + input_rank else ax) axes
  in
  let axes_to_reduce =
    Array.of_list
      (List.sort_uniq Int.compare (Array.to_list axes_to_reduce_normalized))
  in

  let output_shape_logical =
    Array.mapi
      (fun i s ->
        if Array.mem i axes_to_reduce then if keepdims then 1 else -1 else s)
      input_shape
    |> Array.to_list
    |> List.filter (( <> ) (-1))
    |> Array.of_list
  in
  let output_shape_final =
    if Array.length output_shape_logical = 0 then [||] else output_shape_logical
  in

  let output_numel = View.prod output_shape_final in
  let output_tensor =
    op_buffer ctx input_tensor.dtype output_numel |> fun t ->
    with_view t (View.create output_shape_final)
  in

  if output_numel > 0 then
    fill_buffer_with_identity
      (Internal.buffer output_tensor)
      output_numel
      (Dtype.min_val input_tensor.dtype);

  Ops_reduce.max ctx ~axes:axes_to_reduce ~keepdims input_tensor output_tensor;
  output_tensor

let op_reduce_prod ctx ~(axes : int array) ~(keepdims : bool) input_tensor =
  let input_shape = Internal.shape input_tensor in
  let input_rank = Array.length input_shape in
  let axes_to_reduce_normalized =
    Array.map (fun ax -> if ax < 0 then ax + input_rank else ax) axes
  in
  let axes_to_reduce =
    Array.of_list
      (List.sort_uniq Int.compare (Array.to_list axes_to_reduce_normalized))
  in

  let output_shape_logical =
    Array.mapi
      (fun i s ->
        if Array.mem i axes_to_reduce then if keepdims then 1 else -1 else s)
      input_shape
    |> Array.to_list
    |> List.filter (( <> ) (-1))
    |> Array.of_list
  in
  let output_shape_final =
    if Array.length output_shape_logical = 0 then [||] else output_shape_logical
  in

  let output_numel = View.prod output_shape_final in
  let output_tensor =
    op_buffer ctx input_tensor.dtype output_numel |> fun t ->
    with_view t (View.create output_shape_final)
  in

  if output_numel > 0 then
    fill_buffer_with_identity
      (Internal.buffer output_tensor)
      output_numel
      (Dtype.one input_tensor.dtype);

  Ops_reduce.prod ctx ~axes:axes_to_reduce ~keepdims input_tensor output_tensor;
  output_tensor

(* Movement Ops: These just update the view *)
let op_reshape _ctx t (new_shape : int array) =
  match View.reshape t.view new_shape with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_reshape: " ^ msg)
  | exception Failure msg -> failwith ("op_reshape: " ^ msg)

let op_expand _ctx t (new_target_shape : int array) =
  match View.expand t.view new_target_shape with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_expand: " ^ msg)

let op_permute _ctx t (axes : int array) =
  match View.permute t.view axes with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_permute: " ^ msg)

let op_pad _ctx t padding_config (fill_value : 'a) =
  (* Native backend's op_pad for eager mode needs to handle fill_value. The view
     operation itself doesn't store the fill_value. If the padding results in
     accessing outside the original buffer, a new buffer is needed and filled
     appropriately. For simplicity, if the view padding implies new elements,
     this op must create a new tensor. If all padding is negative (effectively a
     shrink), view update is fine. *)
  let old_view = t.view in
  let new_view_metadata_only = View.pad old_view padding_config in

  (* Check if padding is entirely "virtual" (i.e., all pad values are <= 0 for
     before, >= shape for after, or if the original data covers the new view
     completely due to negative padding) This is complex. For now, assume if any
     pad_before > 0 or pad_after > 0, we need a new buffer. *)
  let needs_new_buffer =
    Array.exists (fun (pb, pa) -> pb > 0 || pa > 0) padding_config
  in

  if not needs_new_buffer then { t with view = new_view_metadata_only }
  else
    (* Create new tensor, fill with fill_value, then copy old data into place *)
    let new_shape = View.shape new_view_metadata_only in
    let new_numel = View.numel new_view_metadata_only in
    let new_t =
      op_buffer _ctx t.dtype new_numel |> fun nt ->
      with_view nt (View.create new_shape)
    in

    (* Fill new_t with fill_value *)
    Bigarray.Array1.fill new_t.buffer fill_value;

    (* Copy original data into the "center" of the new tensor *)
    (* Define the slice in the new_t that corresponds to the original t *)
    let shrink_args_for_dst =
      Array.mapi
        (fun i (pb, _pa) -> (pb, pb + old_view.shape.(i)))
        padding_config
    in
    let dst_slice_view = View.shrink new_t.view shrink_args_for_dst in
    let dst_slice_tensor = { new_t with view = dst_slice_view } in

    Internal.blit t dst_slice_tensor;
    (* Use Internal.blit for view-aware copy *)
    new_t

let op_shrink _ctx t limits =
  match View.shrink t.view limits with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_shrink: " ^ msg)

let op_flip _ctx t axes_to_flip =
  match View.flip t.view axes_to_flip with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_flip: " ^ msg)

let op_cat ctx tensors axis =
  if List.length tensors = 0 then
    invalid_arg "op_cat: tensor list cannot be empty";
  let first_t = List.hd tensors in
  let dt_ref = Internal.dtype first_t in
  let rank = Internal.ndim first_t in
  let axis = if axis < 0 then axis + rank else axis in

  if axis < 0 || axis >= rank then invalid_arg "op_cat: axis out of bounds";

  let output_dim_size_at_axis =
    List.fold_left
      (fun acc t ->
        if not (Dtype.eq (Internal.dtype t) dt_ref) then
          failwith "op_cat: dtypes mismatch";
        if Internal.ndim t <> rank then failwith "op_cat: ranks mismatch";
        for i = 0 to rank - 1 do
          if i <> axis && (Internal.shape t).(i) <> (Internal.shape first_t).(i)
          then failwith "op_cat: non-axis dimensions mismatch"
        done;
        acc + (Internal.shape t).(axis))
      0 tensors
  in

  let output_shape = Array.copy (Internal.shape first_t) in
  output_shape.(axis) <- output_dim_size_at_axis;
  let output_numel = View.prod output_shape in
  let output_t =
    op_buffer ctx dt_ref output_numel |> fun t ->
    with_view t (View.create output_shape)
  in

  let current_offset_at_axis = ref 0 in
  List.iter
    (fun src_t ->
      let src_dim_size_at_axis = (Internal.shape src_t).(axis) in
      if src_dim_size_at_axis > 0 then (
        (* Only blit if there's data *)
        let slice_starts = Array.make rank 0 in
        let slice_ends = Array.copy (Internal.shape output_t) in
        (* or src_t's shape for non-axis dims *)

        slice_starts.(axis) <- !current_offset_at_axis;
        slice_ends.(axis) <- !current_offset_at_axis + src_dim_size_at_axis;

        (* Create a view into the output tensor for this part *)
        let shrink_args_for_dst =
          Array.init rank (fun i -> (slice_starts.(i), slice_ends.(i)))
        in
        let dst_slice_view = View.shrink output_t.view shrink_args_for_dst in
        let dst_slice_tensor = { output_t with view = dst_slice_view } in

        Internal.blit src_t dst_slice_tensor);
      current_offset_at_axis := !current_offset_at_axis + src_dim_size_at_axis)
    tensors;
  output_t

(* Other Ops *)
let op_cast ctx input_t target_dt =
  let out_shape = Internal.shape input_t in
  let out_size = Internal.size input_t in
  let out_tensor =
    op_buffer ctx target_dt out_size |> fun t ->
    with_view t (View.create out_shape)
  in
  (* For cast, we need a specific iteration that applies the Dtype.cast_element
     function *)
  let view_in = Internal.view input_t in
  let view_out = Internal.view out_tensor in
  let buf_in = Internal.buffer input_t in
  let buf_out = Internal.buffer out_tensor in
  let cast_fn = Dtype.cast_element (Internal.dtype input_t) target_dt in

  let iter_func logical_indices =
    let phys_idx_in =
      View.offset view_in
      + View.index_to_offset logical_indices (View.strides view_in)
    in
    let phys_idx_out =
      View.offset view_out
      + View.index_to_offset logical_indices (View.strides view_out)
    in
    let val_in = Bigarray.Array1.unsafe_get buf_in phys_idx_in in
    Bigarray.Array1.unsafe_set buf_out phys_idx_out (cast_fn val_in)
  in
  let n_dims = Array.length out_shape in
  (if View.numel view_out = 0 then ()
   else if n_dims = 0 then
     let val_in = Bigarray.Array1.unsafe_get buf_in (View.offset view_in) in
     Bigarray.Array1.unsafe_set buf_out (View.offset view_out) (cast_fn val_in)
   else
     let current_md_idx = Array.make n_dims 0 in
     let rec loop dim =
       if dim = n_dims then iter_func current_md_idx
       else
         for i = 0 to out_shape.(dim) - 1 do
           current_md_idx.(dim) <- i;
           loop (dim + 1)
         done
     in
     loop 0);
  out_tensor

let op_contiguous _ctx t =
  if Internal.is_contiguous t && View.offset t.view = 0 then t
    (* Already contiguous and offset 0 *)
  else Internal.copy t (* Internal.copy creates a new C-contiguous tensor *)

let op_copy _ctx t = Internal.copy t

let op_assign _ctx target_t source_t =
  (* Frontend ensures source_t is broadcast to target_t's shape and dtypes
     match. This means source_t's view matches target_t's view's shape. *)
  Internal.blit source_t target_t;
  target_t (* Return the modified target *)

let op_threefry ctx data_t seed_t =
  (* Placeholder for actual Threefry PRNG. This would involve bitwise
     operations, rotations, XORs, and additions, typically implemented in a
     custom kernel for performance. Simulating it with generic iter_binary_op
     would be very slow and complex due to the specific sequence of operations.
     For an eager backend, this would call a C/Fortran/Rust implementation. For
     a JIT backend, this would be a specific effect. *)
  if not (Internal.shape data_t = Internal.shape seed_t) then
    failwith "op_threefry: data and seed shapes must match after broadcasting";

  let out_shape = Internal.shape data_t in
  let out_size = Internal.size data_t in
  let out_tensor =
    op_buffer ctx Dtype.int32 out_size |> fun t ->
    with_view t (View.create out_shape)
  in

  (* Extremely simplified placeholder - NOT a real Threefry. This just XORs data
     with seed for demonstration. *)
  let placeholder_threefry d s = Int32.logxor d s in
  Ops_binary.iter_binary_op ctx data_t seed_t out_tensor placeholder_threefry;
  out_tensor
