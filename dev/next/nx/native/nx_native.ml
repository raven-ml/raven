(* native/nx_native.ml *)

open Nx_core
module View = View (* Use the newly defined View *)

type ('a, 'b) buffer = ('a, 'b) Internal.buffer

type ('a, 'b) t = ('a, 'b) Internal.t = {
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t; (* Use the new View type *)
}

let view t = t.view
let dtype t = t.dtype
let buffer t = t.buffer
let with_view t view = { t with view } (* Helper to update view *)

(* *)

type context = Internal.context

let create_context () = Internal.{ pool = Parallel.get_or_setup_pool () }

(* --- Backend Ops --- *)

let op_buffer (_ctx : context) (dt : ('a, 'b) Dtype.t) (size_in_elements : int)
    : ('a, 'b) t =
  let kind = Dtype.kind_of_dtype dt in
  let ba = Bigarray.Array1.create kind Bigarray.c_layout size_in_elements in
  (* Initial view is a flat, contiguous array of the specified size *)
  let initial_view = View.create [| size_in_elements |] in
  Internal.{ dtype = dt; buffer = ba; view = initial_view }

let op_const_scalar (_ctx : context) (value : 'a) (dt : ('a, 'b) Dtype.t) :
    ('a, 'b) t =
  let kind = Dtype.kind_of_dtype dt in
  let ba = Bigarray.Array1.create kind Bigarray.c_layout 1 in
  Bigarray.Array1.set ba 0 value;
  (* Scalar constant has a 0-dimensional shape *)
  let scalar_view = View.create [||] in
  Internal.{ dtype = dt; buffer = ba; view = scalar_view }

(* Helper to extract a scalar integer from an index tensor. Index tensors are
   (int, Dtype.int32_elt) t as per op_load/op_store. *)
let get_scalar_int_from_index_tensor (idx_tensor : (int, Dtype.int32_elt) t) :
    int =
  if View.numel idx_tensor.view <> 1 then
    invalid_arg "Index tensor must be a scalar (numel=1) for op_load/op_store";

  let physical_offset =
    if View.ndim idx_tensor.view = 0 then
      (* 0-D tensor, e.g. shape [||] *)
      View.offset idx_tensor.view
    else if View.ndim idx_tensor.view = 1 && idx_tensor.view.shape.(0) = 1 then
      (* 1-D tensor of shape [|1|], logical index is [0] *)
      View.offset_of_indices idx_tensor.view [| 0 |]
    else
      invalid_arg
        "Index tensor has an unexpected shape for scalar extraction (must be \
         [] or [1])"
  in
  Bigarray.Array1.get idx_tensor.buffer physical_offset

let op_load (_ctx : context) ?valid_mask (buffer_source : ('a, 'b) t)
    (logical_indices_tensors : (int, Dtype.int32_elt) t list) : ('a, 'b) t =
  let indices_arr =
    Array.of_list
      (List.map get_scalar_int_from_index_tensor logical_indices_tensors)
  in

  if Array.length indices_arr <> View.ndim buffer_source.view then
    invalid_arg
      (Printf.sprintf
         "op_load: incorrect number of indices (%d) provided for tensor of \
          rank %d."
         (Array.length indices_arr)
         (View.ndim buffer_source.view));

  let final_validity =
    (* 1. Check if the indices are valid for the source tensor's view
       (mask/shape) *)
    let source_access_is_valid = View.is_valid buffer_source.view indices_arr in
    match valid_mask with
    | None -> source_access_is_valid
    | Some mask_tensor ->
        if not source_access_is_valid then false
          (* Access to source buffer is already invalid by its own view *)
        else
          let () =
            if
              (* 2. Check validity using the provided valid_mask tensor *)
              Array.length indices_arr <> View.ndim mask_tensor.view
            then
              invalid_arg
                (Printf.sprintf
                   "op_load: valid_mask rank (%d) mismatch with number of \
                    indices (%d)."
                   (View.ndim mask_tensor.view)
                   (Array.length indices_arr))
          in

          let mask_tensor_access_is_valid =
            View.is_valid mask_tensor.view indices_arr
          in
          if not mask_tensor_access_is_valid then false
            (* Indexing out of bounds of the valid_mask tensor itself *)
          else
            let physical_mask_offset =
              View.offset_of_indices mask_tensor.view indices_arr
            in
            let mask_value_uint8 =
              Bigarray.Array1.get mask_tensor.buffer physical_mask_offset
            in
            mask_value_uint8 <> 0
    (* Assuming 0 is false, non-zero is true for uint8 bool *)
  in

  let value_to_load =
    if final_validity then
      let physical_offset =
        View.offset_of_indices buffer_source.view indices_arr
      in
      Bigarray.Array1.get buffer_source.buffer physical_offset
    else
      Dtype.zero
        buffer_source.dtype (* Load default zero if access is invalid *)
  in
  (* Result of op_load is a new scalar tensor *)
  op_const_scalar _ctx value_to_load buffer_source.dtype

let op_store (_ctx : context) ?valid_mask (buffer_target : ('a, 'b) t)
    (logical_indices_tensors : (int, Dtype.int32_elt) t list)
    (scalar_value_to_store : ('a, 'b) t) : unit =
  let indices_arr =
    Array.of_list
      (List.map get_scalar_int_from_index_tensor logical_indices_tensors)
  in

  if Array.length indices_arr <> View.ndim buffer_target.view then
    invalid_arg
      (Printf.sprintf
         "op_store: incorrect number of indices (%d) provided for tensor of \
          rank %d."
         (Array.length indices_arr)
         (View.ndim buffer_target.view));

  if View.numel scalar_value_to_store.view <> 1 then
    invalid_arg "op_store: value_to_store must be a scalar tensor (numel=1).";

  let final_validity =
    let target_access_is_valid = View.is_valid buffer_target.view indices_arr in
    match valid_mask with
    | None -> target_access_is_valid
    | Some mask_tensor ->
        if not target_access_is_valid then false
        else if Array.length indices_arr <> View.ndim mask_tensor.view then
          invalid_arg
            (Printf.sprintf
               "op_store: valid_mask rank (%d) mismatch with number of indices \
                (%d)."
               (View.ndim mask_tensor.view)
               (Array.length indices_arr))
        else
          let mask_tensor_access_is_valid =
            View.is_valid mask_tensor.view indices_arr
          in
          if not mask_tensor_access_is_valid then false
          else
            let physical_mask_offset =
              View.offset_of_indices mask_tensor.view indices_arr
            in
            let mask_value_uint8 =
              Bigarray.Array1.get mask_tensor.buffer physical_mask_offset
            in
            mask_value_uint8 <> 0
  in

  if final_validity then
    let physical_target_offset =
      View.offset_of_indices buffer_target.view indices_arr
    in
    (* Extract the scalar value from the scalar_value_to_store tensor *)
    let value =
      let val_tensor_view = scalar_value_to_store.view in
      let val_tensor_buffer = scalar_value_to_store.buffer in
      let val_access_offset =
        if View.ndim val_tensor_view = 0 then View.offset val_tensor_view
        else if View.ndim val_tensor_view = 1 && val_tensor_view.shape.(0) = 1
        then View.offset_of_indices val_tensor_view [| 0 |]
        else
          invalid_arg
            "op_store: scalar_value_to_store has unexpected shape (must be [] \
             or [1])"
      in
      Bigarray.Array1.get val_tensor_buffer val_access_offset
    in
    Bigarray.Array1.set buffer_target.buffer physical_target_offset value

(* Binary Ops *)
let op_add ctx (a : ('a, 'b) t) (b : ('c, 'd) t) : ('e, 'f) t =
  (* NOTE: Type safety for ('a,'b) and ('c,'d) -> ('e,'f) isn't enforced here.
     The frontend or a type promotion layer should handle this. Assuming a and b
     have compatible types for addition for now. *)
  if not (View.shape a.view = View.shape b.view) then
    failwith
      "op_add: Shapes must match after broadcasting (responsibility of \
       frontend)";
  let out_shape = View.shape a.view in
  let out_size = View.numel a.view in

  let out_tensor =
    let unshaped_buffer = op_buffer ctx a.dtype out_size in
    with_view unshaped_buffer (View.create out_shape)
  in

  Ops_binary.add ctx a b out_tensor;
  out_tensor

let op_mul ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
  if not (Internal.shape a = Internal.shape b) then
    failwith
      "op_mul: Shapes must match after broadcasting (responsibility of \
       frontend)";
  let out_shape = Internal.shape a in
  let out_size = Internal.size a in

  let out_tensor =
    let unshaped_buffer = op_buffer ctx a.dtype out_size in
    with_view unshaped_buffer (View.create out_shape)
  in
  Ops_binary.mul ctx a b out_tensor;
  out_tensor

let fill_buffer_with_zero (type a b) (dt : (a, b) Dtype.t)
    (buf : (a, b) Internal.buffer) (count : int) =
  let zero_val = Dtype.zero dt in
  for i = 0 to count - 1 do
    Bigarray.Array1.set buf i zero_val
  done

let op_sum ctx ~(axes : int array) ~(keepdims : bool)
    (input_tensor : ('a, 'b) t) : ('a, 'b) t =
  let input_shape = Internal.shape input_tensor in
  let input_rank = Array.length input_shape in

  (* Normalize and validate axes. The `axis` array received might contain
     negative indices or duplicates. Frontend might do some normalization, but
     backend should be robust. *)
  let axes_to_reduce_normalized =
    Array.map
      (fun ax ->
        let ax' = if ax < 0 then ax + input_rank else ax in
        (* FIX 1 *)
        if ax' < 0 || ax' >= input_rank then
          invalid_arg
            (Printf.sprintf "op_sum: axis %d is out of bounds for rank %d" ax
               input_rank);
        ax')
      axes
  in
  (* Sort and unique the normalized axes *)
  let axes_to_reduce =
    if Array.length axes_to_reduce_normalized = 0 && input_rank > 0 then
      (* No axes to reduce means sum over nothing for each element if not
         reducing all *)
      [||]
      (* Or handle as sum over all if axis = [||] is supposed to mean that by
         convention? Interface says "axes to reduce along" *)
    else if Array.length axes_to_reduce_normalized = 0 && input_rank = 0 then
      (* scalar input, reducing no axes *)
      [||]
    else
      Array.of_list
        (List.sort_uniq Int.compare (Array.to_list axes_to_reduce_normalized))
    (* FIX 2 *)
  in

  (* Handle case where axis is empty: sum over all axes as per Nx Python convention if axis=None *)
  (* The frontend.ml handles axis=None by converting it to [0,1,...,rank-1].
   So, if `axes_to_reduce` is empty here, it means the original `axis` list/array was empty.
   Summing over an empty set of axes is an identity operation on each element,
   but usually reduction ops expect at least one axis or imply all if none given.
   If `axis` parameter to `op_sum` is truly empty `[||]`, what should happen?
   If input `axis` is `[||]`, `axes_to_reduce` becomes `[||]`.
   `Ops_reduce.sum` handles `axes_to_reduce_list = []` by blitting input to output.
   This implies summing over no axes results in the original tensor.
   This seems like a reasonable interpretation if an empty `axis` array is passed.
*)
  let output_shape_logical =
    let s = ref [] in
    if input_rank = 0 then (* Scalar input *)
      if keepdims then s := [ 1 ]
        (* keepdims makes it [1] from [] potentially *)
      else s := [] (* shape [] *)
    else
      for i = 0 to input_rank - 1 do
        if Array.mem i axes_to_reduce then (if keepdims then s := 1 :: !s)
        else s := input_shape.(i) :: !s
      done;
    if
      !s = [] && input_rank > 0
      && Array.length axes_to_reduce = input_rank
      && not keepdims
    then [||] (* Fully reduced to scalar *)
    else if !s = [] && input_rank = 0 && not keepdims then [||]
      (* Scalar input, not reducing, not keepdims *)
    else Array.of_list (List.rev !s)
  in

  let output_numel = View.prod output_shape_logical in
  let output_tensor =
    let unshaped_buffer = op_buffer ctx input_tensor.dtype output_numel in
    let final_view_shape =
      if output_numel = 0 && output_shape_logical = [||] then [| 0 |]
        (* buffer view hack for 0 elements*)
      else if output_numel = 1 && output_shape_logical = [||] then [||]
        (* scalar *)
      else output_shape_logical
    in
    with_view unshaped_buffer (View.create final_view_shape)
  in

  (* Initialize output tensor to zeros *)
  if output_numel > 0 then (* Avoid filling 0-sized buffer if numel is 0 *)
    fill_buffer_with_zero output_tensor.dtype
      (Internal.buffer output_tensor)
      output_numel;

  (* Delegate to specialized reduction function *)
  Ops_reduce.sum ctx ~axes:axes_to_reduce ~keepdims input_tensor output_tensor;
  (* FIX 3 *)
  output_tensor

(* Movement Ops: These just update the view *)
let op_reshape _ctx (t : ('a, 'b) t) (new_shape : int array) : ('a, 'b) t =
  match View.reshape t.view new_shape with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_reshape: " ^ msg)
  | exception Failure msg -> failwith ("op_reshape: " ^ msg)
(* Handle internal errors *)

let op_expand _ctx (t : ('a, 'b) t) (new_target_shape : int array) : ('a, 'b) t
    =
  match View.expand t.view new_target_shape with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_expand: " ^ msg)

let op_permute _ctx (t : ('a, 'b) t) (axes : int array) : ('a, 'b) t =
  match View.permute t.view axes with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_permute: " ^ msg)

let op_pad _ctx (t : ('a, 'b) t) (padding : (int * int) array) : ('a, 'b) t =
  match View.pad t.view padding with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_pad: " ^ msg)

let op_shrink _ctx (t : ('a, 'b) t) (limits : (int * int) array) : ('a, 'b) t =
  match View.shrink t.view limits with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_shrink: " ^ msg)

let op_flip _ctx (t : ('a, 'b) t) (axes_to_flip : bool array) : ('a, 'b) t =
  match View.flip t.view axes_to_flip with
  | new_view -> { t with view = new_view }
  | exception Invalid_argument msg -> invalid_arg ("op_flip: " ^ msg)

(* Add other binary ops similarly (sub, mul, div, etc.) *)
(* op_sub, op_mul, op_div ... *)

(* JIT Ops - Raise errors or are identity in eager mode *)
let op_define_global _ctx _name t = t (* Identity for eager *)

let op_range _ctx _name _bound =
  failwith "op_range not supported in eager native backend"

let op_special _ctx _name _kind =
  failwith "op_special not supported in eager native backend"
