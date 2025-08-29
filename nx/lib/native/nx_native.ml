(* nx_native.ml *)

open Nx_core
open Bigarray_ext

type ('a, 'b) buffer = ('a, 'b) Internal.buffer
type context = Internal.context

type ('a, 'b) t = ('a, 'b) Internal.t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : Lazy_view.t;
}

let view t = t.view
let dtype t = t.dtype
let data t = t.buffer
let context t = t.context
let with_view t view = { t with view }
let create_context () = Internal.{ pool = Parallel.get_or_setup_pool () }

(* --- Backend Ops --- *)

let op_buffer ctx dt size_in_elements =
  let kind = Dtype.to_bigarray_ext_kind dt in
  let ba = Array1.create kind c_layout size_in_elements in
  let initial_view =
    if size_in_elements = 0 then
      Lazy_view.create (Symbolic_shape.of_ints [| 0 |])
      (* Consistent 0-element view *)
    else Lazy_view.create (Symbolic_shape.of_ints [| size_in_elements |])
  in
  { context = ctx; dtype = dt; buffer = ba; view = initial_view }

let op_const_scalar ctx value dt =
  let kind = Dtype.to_bigarray_ext_kind dt in
  let ba = Array1.create kind c_layout 1 in
  Array1.set ba 0 value;
  let scalar_view = Lazy_view.create (Symbolic_shape.of_ints [||]) in
  (* 0-dim for scalar *)
  { context = ctx; dtype = dt; buffer = ba; view = scalar_view }

let op_const_array ctx bigarray =
  let dtype = Dtype.of_bigarray_ext_kind (Bigarray_ext.Array1.kind bigarray) in
  let size = Bigarray_ext.Array1.dim bigarray in
  let t = op_buffer ctx dtype size in
  Bigarray_ext.Array1.blit bigarray (data t);
  t

let op_contiguous t =
  if Internal.is_c_contiguous t && Internal.offset t = 0 then t
    (* Already contiguous and offset 0 *)
  else Internal.copy t (* Internal.copy creates a new C-contiguous tensor *)

let op_copy t = Internal.copy t

(* Helper to ensure tensors have materializable views for operations *)
let ensure_materializable t =
  (* If can't get strides (e.g., broadcast views), materialize *)
  if not (Lazy_view.can_get_strides t.view) then op_contiguous t
  else
    (* Can get strides - but double-check they're valid *)
    try
      let strides = Internal.strides t in
      let shape = Internal.shape t in
      if Array.length strides <> Array.length shape then
        (* Stride/shape mismatch - materialize *)
        op_contiguous t
      else t
    with _ ->
      (* Any error getting strides - materialize *)
      op_contiguous t

let op_assign target_t source_t = Internal.blit source_t target_t

(* Binary Ops *)

(* Helper for binary operations that ensures inputs are materializable first *)
let binary_op op_func a b =
  let ctx = a.context in
  let out_shape = Internal.shape a in
  let out_size = Internal.numel a in
  let out_dtype = a.dtype in
  let out_tensor =
    op_buffer ctx out_dtype out_size |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints out_shape))
  in
  op_func ctx a b out_tensor;
  out_tensor

(* Helper for binary comparison operations *)
let binary_cmp_op op_func a b =
  let a' = ensure_materializable a in
  let b' = ensure_materializable b in
  let ctx = a'.context in
  let out_shape = Internal.shape a' in
  let out_size = Internal.numel a' in
  let out_tensor =
    op_buffer ctx Dtype.uint8 out_size |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints out_shape))
  in
  op_func ctx a' b' out_tensor;
  out_tensor

let op_add a b = binary_op Ops_binary.add a b
let op_mul a b = binary_op Ops_binary.mul a b
let op_idiv a b = binary_op Ops_binary.idiv a b
let op_fdiv a b = binary_op Ops_binary.fdiv a b
let op_max a b = binary_op Ops_binary.max a b
let op_mod a b = binary_op Ops_binary.modulo a b
let op_pow a b = binary_op Ops_binary.pow a b
let op_cmplt a b = binary_cmp_op Ops_binary.cmplt a b
let op_cmpne a b = binary_cmp_op Ops_binary.cmpne a b
let op_xor a b = binary_op Ops_binary.bit_xor a b
let op_or a b = binary_op Ops_binary.bit_or a b
let op_and a b = binary_op Ops_binary.bit_and a b

(* Unary Ops *)

(* Helper for unary operations that ensures input is materializable first *)
let unary_op op_func x =
  let x' = ensure_materializable x in
  let ctx = x'.context in
  let out_shape = Internal.shape x' in
  let out_size = Internal.size x' in
  let out_tensor =
    op_buffer ctx x'.dtype out_size |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints out_shape))
  in
  op_func ctx x' out_tensor;
  out_tensor

let op_neg x = unary_op Ops_unary.neg x
let op_log2 x = unary_op Ops_unary.log2 x
let op_exp2 x = unary_op Ops_unary.exp2 x
let op_sin x = unary_op Ops_unary.sin x
let op_sqrt x = unary_op Ops_unary.sqrt x
let op_recip x = unary_op Ops_unary.recip x

(* Ternary Op *)
let op_where cond if_true if_false =
  (* Ensure all inputs are materializable first *)
  let cond' = ensure_materializable cond in
  let if_true' = ensure_materializable if_true in
  let if_false' = ensure_materializable if_false in
  let ctx = cond'.context in
  let out_shape = Internal.shape cond' in
  let out_size = Internal.numel cond' in
  let out_tensor =
    op_buffer ctx if_true'.dtype out_size |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints out_shape))
  in
  Ops_ternary.where ctx cond' if_true' if_false' out_tensor;
  out_tensor

let fill_buffer_with_identity buf count identity_val =
  for i = 0 to count - 1 do
    Array1.unsafe_set buf i identity_val
  done

let op_reduce_sum ~(axes : int array) ~(keepdims : bool) xensor =
  let xensor = ensure_materializable xensor in
  let ctx = xensor.context in
  let input_shape = Internal.shape xensor in
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

  let output_numel = Shape.numel output_shape_final in
  let output_tensor =
    op_buffer ctx xensor.dtype output_numel |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints output_shape_final))
  in

  if output_numel > 0 then
    fill_buffer_with_identity
      (Internal.buffer output_tensor)
      output_numel (Dtype.zero xensor.dtype);

  Ops_reduce.sum ctx ~axes:axes_to_reduce ~keepdims xensor output_tensor;
  output_tensor

let op_reduce_max ~(axes : int array) ~(keepdims : bool) xensor =
  let xensor = ensure_materializable xensor in
  let ctx = xensor.context in
  let input_shape = Internal.shape xensor in
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

  let output_numel = Shape.numel output_shape_final in
  let output_tensor =
    op_buffer ctx xensor.dtype output_numel |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints output_shape_final))
  in

  if output_numel > 0 then
    fill_buffer_with_identity
      (Internal.buffer output_tensor)
      output_numel
      (Dtype.min_value xensor.dtype);

  Ops_reduce.max ctx ~axes:axes_to_reduce ~keepdims xensor output_tensor;
  output_tensor

let op_reduce_prod ~(axes : int array) ~(keepdims : bool) xensor =
  let xensor = ensure_materializable xensor in
  let ctx = xensor.context in
  let input_shape = Internal.shape xensor in
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

  let output_numel = Shape.numel output_shape_final in
  let output_tensor =
    op_buffer ctx xensor.dtype output_numel |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints output_shape_final))
  in

  if output_numel > 0 then
    fill_buffer_with_identity
      (Internal.buffer output_tensor)
      output_numel (Dtype.one xensor.dtype);

  Ops_reduce.prod ctx ~axes:axes_to_reduce ~keepdims xensor output_tensor;
  output_tensor

(* Movement Ops: These just update the view *)
let op_reshape t (new_shape : Symbolic_shape.t) =
  let new_view = Lazy_view.reshape new_shape t.view in
  { t with view = new_view }

let op_expand t (new_target_shape : Symbolic_shape.t) =
  let new_view = Lazy_view.expand new_target_shape t.view in
  { t with view = new_view }

let op_permute t (axes : int array) =
  let new_view = Lazy_view.permute axes t.view in
  { t with view = new_view }

let op_pad t padding_config (fill_value : 'a) =
  (* Native backend's op_pad for eager mode needs to handle fill_value. The view
     operation itself doesn't store the fill_value. If the padding results in
     accessing outside the original buffer, a new buffer is needed and filled
     appropriately. For simplicity, if the view padding implies new elements,
     this op must create a new tensor. If all padding is negative (effectively a
     shrink), view update is fine. *)
  let old_view = t.view in
  let new_view_metadata_only = Lazy_view.pad padding_config old_view in

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
    let new_shape =
      match Symbolic_shape.eval (Lazy_view.shape new_view_metadata_only) with
      | Some arr -> arr
      | None ->
          Error.failed ~op:"op_pad" ~what:"cannot pad with symbolic dimensions"
            ()
    in
    let new_numel = Shape.numel new_shape in
    let new_t =
      op_buffer t.context t.dtype new_numel |> fun nt ->
      with_view nt (Lazy_view.create (Symbolic_shape.of_ints new_shape))
    in

    (* Fill new_t with fill_value *)
    Array1.fill new_t.buffer fill_value;

    (* Copy original data into the "center" of the new tensor *)
    (* Define the slice in the new_t that corresponds to the original t *)
    let shrink_args_for_dst =
      Array.mapi
        (fun i (pb, _pa) -> (pb, pb + (Internal.shape t).(i)))
        padding_config
    in
    let dst_slice_view = Lazy_view.shrink shrink_args_for_dst new_t.view in
    let dst_slice_tensor = { new_t with view = dst_slice_view } in

    Internal.blit t dst_slice_tensor;
    (* Use Internal.blit for view-aware copy *)
    new_t

let op_shrink t limits =
  let new_view = Lazy_view.shrink limits t.view in
  { t with view = new_view }

let op_flip t axes_to_flip =
  let new_view = Lazy_view.flip axes_to_flip t.view in
  { t with view = new_view }

let op_cat tensors axis =
  if List.length tensors = 0 then
    invalid_arg "op_cat: tensor list cannot be empty";
  let first_t = List.hd tensors in
  let ctx = first_t.context in
  let dt_ref = Internal.dtype first_t in
  let rank = Internal.ndim first_t in
  let axis = if axis < 0 then axis + rank else axis in

  if axis < 0 || axis >= rank then invalid_arg "op_cat: axis out of bounds";

  let output_dim_size_at_axis =
    List.fold_left
      (fun acc t ->
        if not (Dtype.equal (Internal.dtype t) dt_ref) then
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
  let output_numel = Shape.numel output_shape in
  let output_t =
    op_buffer ctx dt_ref output_numel |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints output_shape))
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
        let dst_slice_view =
          Lazy_view.shrink shrink_args_for_dst output_t.view
        in
        let dst_slice_tensor = { output_t with view = dst_slice_view } in

        Internal.blit src_t dst_slice_tensor);
      current_offset_at_axis := !current_offset_at_axis + src_dim_size_at_axis)
    tensors;
  output_t

(* Other Ops *)
let op_cast x target_dt =
  let ctx = x.context in
  let out_shape = Internal.shape x in
  let out_size = Internal.size x in
  let out_tensor =
    op_buffer ctx target_dt out_size |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints out_shape))
  in
  (* Use the optimized cast implementation from Ops_cast *)
  Ops_cast.cast ctx x out_tensor;
  out_tensor

let op_threefry data seed =
  let ctx = data.context in
  let out_shape = Internal.shape data in
  let out_size = Internal.size data in
  let out_tensor =
    op_buffer ctx Dtype.int32 out_size |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints out_shape))
  in
  Ops_random.threefry ctx data seed out_tensor;
  out_tensor

(* Element Access Ops *)

let op_gather data_t indices_t axis =
  let ctx = data_t.context in
  let data_shape = Internal.shape data_t in
  let indices_shape = Internal.shape indices_t in
  let data_rank = Array.length data_shape in
  let indices_rank = Array.length indices_shape in

  (* Validate inputs *)
  if data_rank <> indices_rank then
    invalid_arg
      (Printf.sprintf
         "op_gather: data rank (%d) and indices rank (%d) must match" data_rank
         indices_rank);

  let axis = if axis < 0 then axis + data_rank else axis in
  if axis < 0 || axis >= data_rank then
    invalid_arg
      (Printf.sprintf "op_gather: axis %d out of bounds for rank %d" axis
         data_rank);

  (* Validate shape compatibility *)
  for i = 0 to data_rank - 1 do
    if i <> axis && indices_shape.(i) > data_shape.(i) then
      invalid_arg
        (Printf.sprintf "op_gather: indices.shape[%d]=%d > data.shape[%d]=%d" i
           indices_shape.(i) i data_shape.(i))
  done;

  (* Create output tensor with shape of indices *)
  let output_shape = indices_shape in
  let output_numel = Shape.numel output_shape in
  let output_t =
    op_buffer ctx data_t.dtype output_numel |> fun t ->
    with_view t (Lazy_view.create (Symbolic_shape.of_ints output_shape))
  in

  if output_numel = 0 then output_t
  else
    let output_buffer = Internal.buffer output_t in
    (* Ensure tensors are materializable for gather operation *)
    let data_buffer, data_view =
      match Lazy_view.compose data_t.view with
      | Some v -> (Internal.buffer data_t, v)
      | None -> (
          let cont_data = op_contiguous data_t in
          match Lazy_view.compose cont_data.view with
          | Some v -> (Internal.buffer cont_data, v)
          | None ->
              Error.failed ~op:"op_gather" ~what:"cannot materialize data view"
                ())
    in
    let indices_buffer, indices_view =
      match Lazy_view.compose indices_t.view with
      | Some v -> (Internal.buffer indices_t, v)
      | None -> (
          let cont_indices = op_contiguous indices_t in
          match Lazy_view.compose cont_indices.view with
          | Some v -> (Internal.buffer cont_indices, v)
          | None ->
              Error.failed ~op:"op_gather"
                ~what:"cannot materialize indices view" ())
    in

    (* Pre-allocate work arrays *)
    let md_idx = Array.make (Array.length output_shape) 0 in
    let src_idx = Array.make (Array.length output_shape) 0 in

    (* Process each output element *)
    for linear_idx = 0 to output_numel - 1 do
      (* Get multi-dimensional index in output/indices *)
      Shape.unravel_index_into linear_idx output_shape md_idx;

      (* Read the index value from indices tensor *)
      let indices_offset = View.linear_index indices_view md_idx in
      let idx_value =
        Int32.to_int (Array1.unsafe_get indices_buffer indices_offset)
      in

      (* Handle negative indices and bounds checking *)
      let data_size_at_axis = data_shape.(axis) in
      let normalized_idx =
        if idx_value < 0 then idx_value + data_size_at_axis else idx_value
      in

      (* Clamp to valid range *)
      let clamped_idx = max 0 (min (data_size_at_axis - 1) normalized_idx) in

      (* Build source index for data tensor *)
      Array.blit md_idx 0 src_idx 0 (Array.length md_idx);
      src_idx.(axis) <- clamped_idx;

      (* Copy value from data to output *)
      if View.is_valid data_view src_idx then
        let data_offset = View.linear_index data_view src_idx in
        let value = Array1.unsafe_get data_buffer data_offset in
        Array1.unsafe_set output_buffer linear_idx value
      (* If invalid due to view mask, output remains at default (likely 0) *)
    done;
    output_t

let op_scatter (type a b) ?(mode = `Set) ?(unique_indices = false)
    (data_template_t : (a, b) Internal.t)
    (indices_t : (int32, Dtype.int32_elt) Internal.t)
    (updates_t : (a, b) Internal.t) axis : (a, b) Internal.t =
  let _ = unique_indices in
  (* TODO: use this hint for optimization *)
  let template_shape = Internal.shape data_template_t in
  let indices_shape = Internal.shape indices_t in
  let updates_shape = Internal.shape updates_t in
  let template_rank = Array.length template_shape in

  (* Validate inputs *)
  if not (Dtype.equal data_template_t.dtype updates_t.dtype) then
    invalid_arg "op_scatter: data_template and updates must have same dtype";

  if indices_shape <> updates_shape then
    invalid_arg "op_scatter: indices and updates must have same shape";

  let axis = if axis < 0 then axis + template_rank else axis in
  if axis < 0 || axis >= template_rank then
    invalid_arg
      (Printf.sprintf "op_scatter: axis %d out of bounds for rank %d" axis
         template_rank);

  (* Validate shape compatibility *)
  for i = 0 to template_rank - 1 do
    if i <> axis && updates_shape.(i) > template_shape.(i) then
      invalid_arg
        (Printf.sprintf
           "op_scatter: updates.shape[%d]=%d > template.shape[%d]=%d" i
           updates_shape.(i) i template_shape.(i))
  done;

  (* Create output as copy of template *)
  let output_t = Internal.copy data_template_t in

  let updates_numel = Shape.numel updates_shape in
  if updates_numel = 0 then output_t
  else
    let output_buffer = Internal.buffer output_t in
    (* Ensure views can be composed *)
    let output_view =
      match Lazy_view.compose output_t.view with
      | Some v -> v
      | None ->
          Error.failed ~op:"op_scatter" ~what:"cannot materialize output view"
            ()
    in
    let updates_buffer, updates_view =
      match Lazy_view.compose updates_t.view with
      | Some v -> (Internal.buffer updates_t, v)
      | None -> (
          let cont_updates = op_contiguous updates_t in
          match Lazy_view.compose cont_updates.view with
          | Some v -> (Internal.buffer cont_updates, v)
          | None ->
              Error.failed ~op:"op_scatter"
                ~what:"cannot materialize updates view" ())
    in
    let indices_buffer, indices_view =
      match Lazy_view.compose indices_t.view with
      | Some v -> (Internal.buffer indices_t, v)
      | None -> (
          let cont_indices = op_contiguous indices_t in
          match Lazy_view.compose cont_indices.view with
          | Some v -> (Internal.buffer cont_indices, v)
          | None ->
              Error.failed ~op:"op_scatter"
                ~what:"cannot materialize indices view" ())
    in

    (* Pre-allocate work arrays *)
    let md_idx = Array.make (Array.length updates_shape) 0 in
    let dst_idx = Array.make (Array.length updates_shape) 0 in

    (* Process each update *)
    for linear_idx = 0 to updates_numel - 1 do
      (* Get multi-dimensional index in updates/indices *)
      Shape.unravel_index_into linear_idx updates_shape md_idx;

      (* Read the target index from indices tensor *)
      let indices_offset = View.linear_index indices_view md_idx in
      let idx_value =
        Int32.to_int (Array1.unsafe_get indices_buffer indices_offset)
      in

      (* Handle negative indices *)
      let template_size_at_axis = template_shape.(axis) in
      let normalized_idx =
        if idx_value < 0 then idx_value + template_size_at_axis else idx_value
      in

      (* Check bounds *)
      if normalized_idx >= 0 && normalized_idx < template_size_at_axis then (
        (* Build destination index *)
        Array.blit md_idx 0 dst_idx 0 (Array.length md_idx);
        dst_idx.(axis) <- normalized_idx;

        (* Write update value to output if destination is valid *)
        if View.is_valid output_view dst_idx then
          let updates_offset = View.linear_index updates_view md_idx in
          let uv = Array1.unsafe_get updates_buffer updates_offset in
          let output_offset = View.linear_index output_view dst_idx in
          match mode with
          | `Set -> Array1.unsafe_set output_buffer output_offset uv
          | `Add ->
              let cv = Array1.unsafe_get output_buffer output_offset in
              let sum = Dtype.add data_template_t.dtype cv uv in
              Array1.unsafe_set output_buffer output_offset sum)
    done;
    output_t

let op_unfold x ~kernel_size ~stride ~dilation ~padding =
  Ops_window.unfold x.context x ~kernel_size ~stride ~dilation ~padding

let op_fold x ~output_size ~kernel_size ~stride ~dilation ~padding =
  Ops_window.fold x.context x ~output_size ~kernel_size ~stride ~dilation
    ~padding

let op_matmul a b = Ops_matmul.matmul a.context a b

(* FFT operations *)
let op_fft x ~axes ~s = Ops_fft.fft x.context x ~axes ~s
let op_ifft x ~axes ~s = Ops_fft.ifft x.context x ~axes ~s
let op_rfft x ~dtype ~axes ~s = Ops_fft.rfft x.context x ~dtype ~axes ~s
let op_irfft x ~dtype ~axes ~s = Ops_fft.irfft x.context x ~dtype ~axes ~s

(* Linear algebra operations *)

let op_cholesky ~upper x = Ops_linalg.cholesky ~upper x.context x
let op_qr ~reduced x = Ops_linalg.qr ~reduced x.context x
let op_svd ~full_matrices x = Ops_svd.svd ~full_matrices x.context x
let op_eig ~vectors x = Ops_linalg.eig ~vectors x.context x
let op_eigh ~vectors x = Ops_linalg.eigh ~vectors x.context x

let op_triangular_solve ~upper ~transpose ~unit_diag a b =
  Ops_linalg.triangular_solve ~upper ~transpose ~unit_diag a.context a b

let op_as_strided t new_shape new_strides_in_elements offset_in_elements =
  (* Native backend can implement this as zero-copy by manipulating the view *)

  (* Validate that the new view doesn't access out-of-bounds memory *)
  let buffer_size = Array1.dim t.buffer in
  let new_shape_arr =
    match Symbolic_shape.eval new_shape with
    | Some arr -> arr
    | None ->
        Error.failed ~op:"op_as_strided" ~what:"symbolic shapes not supported"
          ()
  in

  (* Calculate the maximum element accessed *)
  let max_element_accessed = ref offset_in_elements in
  Array.iteri
    (fun i dim ->
      if dim > 0 then
        max_element_accessed :=
          max !max_element_accessed
            (offset_in_elements + ((dim - 1) * new_strides_in_elements.(i))))
    new_shape_arr;

  if !max_element_accessed >= buffer_size then
    Error.failed ~op:"op_as_strided"
      ~what:"view would access out-of-bounds memory" ();

  (* Create a new view with custom strides and offset *)
  let new_view =
    Lazy_view.create_strided new_shape ~strides:new_strides_in_elements
      ~offset:offset_in_elements
  in

  (* Return tensor with the same buffer but new view *)
  { t with view = new_view }
