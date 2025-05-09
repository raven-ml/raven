module Make (B : Backend_intf.S) = struct
  type ('a, 'b) t = ('a, 'b) B.t
  type context = B.context

  let create_context () = B.create_context ()
  let shape t = View.shape (B.view t)
  let size t = View.size (B.view t)
  let numel t = View.numel (B.view t)
  let dtype t = B.dtype t
  let view t = B.view t
  let ndim t = View.ndim (B.view t)

  let resolve_neg_one_shape current_shape new_shape_spec =
    let new_shape_spec_l = Array.to_list new_shape_spec in
    let current_numel = View.prod current_shape in
    (* Dummy tensor for numel *)
    let neg_one_count =
      new_shape_spec_l |> List.filter (( = ) (-1)) |> List.length
    in
    if neg_one_count > 1 then
      invalid_arg "Reshape target shape can only contain one -1"
    else if neg_one_count = 0 then new_shape_spec
    else
      let specified_numel =
        List.filter (( <> ) (-1)) new_shape_spec_l |> Array.of_list |> View.prod
      in
      if specified_numel = 0 then (* Handle case where 0 is present *)
        if current_numel = 0 then
          Array.map (fun x -> if x = -1 then 0 else x) new_shape_spec
        else
          invalid_arg
            "Reshape cannot infer -1 when other dimensions multiply to 0 but \
             total size is non-zero"
      else if current_numel mod specified_numel <> 0 then
        invalid_arg
          (Printf.sprintf
             "Reshape size mismatch: Cannot reshape %d elements into shape \
              with specified elements %d"
             current_numel specified_numel)
      else
        let inferred_dim = current_numel / specified_numel in
        Array.map (fun s -> if s = -1 then inferred_dim else s) new_shape_spec

  let reshape ctx (x : ('a, 'b) t) (shape_spec : int array) : ('a, 'b) t =
    let new_shape = resolve_neg_one_shape (shape x) shape_spec in
    if shape x = new_shape then x (* No-op *) else B.op_reshape ctx x new_shape

  (* Internal broadcast helper matching tinygrad's _broadcast_to *)
  let broadcast_to ctx (x : ('a, 'b) t) (new_shape : int array) : ('a, 'b) t =
    let current_shape = shape x in
    if current_shape = new_shape then x (* No-op *)
    else
      let rank_current = Array.length current_shape in
      let rank_new = Array.length new_shape in
      if rank_current > rank_new then
        invalid_arg
          (Printf.sprintf "Cannot broadcast tensor to fewer dimensions")
      else
        (* 1. Align shapes by padding with 1s *)
        let padded_shape =
          if rank_current < rank_new then
            Array.append (Array.make (rank_new - rank_current) 1) current_shape
          else current_shape
        in
        (* 2. Check broadcast compatibility *)
        let compatible = ref true in
        for i = 0 to rank_new - 1 do
          if not (padded_shape.(i) = new_shape.(i) || padded_shape.(i) = 1) then
            compatible := false
        done;
        if not !compatible then
          invalid_arg (Printf.sprintf "Cannot broadcast shape");

        (* 3. Reshape if needed to add leading 1s *)
        let x_reshaped =
          if padded_shape <> current_shape then
            (* Need reshape if rank increased *)
            reshape ctx x padded_shape
          else x
        in
        (* 4. Call backend expand *)
        B.op_expand ctx x_reshaped new_shape

  (* Internal helper matching tinygrad's _broadcasted *)
  let broadcasted ctx ?(reverse = false) x y =
    let a, b = if reverse then (y, x) else (x, y) in

    (* Actual Broadcasting via backend *)
    let broadcast_shape = View.broadcast_shapes (shape a) (shape b) in
    let a_broad = broadcast_to ctx a broadcast_shape in
    let b_broad = broadcast_to ctx b broadcast_shape in
    (a_broad, b_broad)

  let expand ctx (x : ('a, 'b) t) (shape_spec : int array) : ('a, 'b) t =
    (* Resolve -1 to keep original dimension size *)
    let current_shape = shape x in
    let rank_current = Array.length current_shape in
    let rank_spec = Array.length shape_spec in
    let rank_new = max rank_current rank_spec in

    (* Align shapes first *)
    let current_aligned =
      if rank_current < rank_new then
        Array.append (Array.make (rank_new - rank_current) 1) current_shape
      else current_shape
    in
    let spec_aligned =
      if rank_spec < rank_new then
        Array.append (Array.make (rank_new - rank_spec) (-1)) shape_spec
      else shape_spec
    in

    let new_shape =
      Array.mapi
        (fun i spec_dim ->
          if spec_dim = -1 then current_aligned.(i) else spec_dim)
        spec_aligned
    in
    broadcast_to ctx x new_shape

  let add ctx a b =
    let a', b' = broadcasted ctx a b in
    (* Assuming B.op_add handles type promotion or they are compatible *)
    B.op_add ctx a' b'

  let mul ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
    let a', b' = broadcasted ctx a b in
    B.op_mul ctx a' b'

  let sum ctx ?(axes : int array option) ?(keepdims = false) (x : ('a, 'b) t) :
      ('a, 'b) t =
    let current_shape = shape x in
    let rank = Array.length current_shape in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id (* Reduce all axes *)
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    (* Backend op_sum will handle further validation and logic *)
    B.op_sum ctx x ~axes:axes_to_reduce ~keepdims

  (* Add other binary ops: sub, mul, div, etc. *)
  (* let sub ctx a b = ... *)
  (* let mul ctx a b = ... *)
  (* let div ctx a b = ... *)

  let empty ctx ?(dtype = Dtype.float32) shape =
    let numel = View.prod shape in
    let buf = B.op_buffer ctx dtype numel in
    (* Give the buffer the correct logical shape *)
    reshape ctx buf shape

  let full ctx ~dtype:(dt : ('a, 'b) Dtype.t) (target_shape : int array)
      (fill_value : 'a) : ('a, 'b) t =
    let scalar_tensor = B.op_const_scalar ctx fill_value dt in
    let rank = Array.length target_shape in
    (* intermediate_shape will be [||] if rank is 0, e.g. for a scalar
       target_shape *)
    let intermediate_shape = Array.make rank 1 in
    (* Reshape scalar_tensor to intermediate_shape. If rank=0,
       intermediate_shape=[||], so this is a no-op if scalar_tensor is already
       0-D. *)
    let reshaped_scalar = B.op_reshape ctx scalar_tensor intermediate_shape in
    (* Expand to target_shape. If rank=0, target_shape=[||], so this is also a
       no-op. *)
    B.op_expand ctx reshaped_scalar target_shape

  let zeros ctx ~(dtype : ('a, 'b) Dtype.t) (shape_arr : int array) : ('a, 'b) t
      =
    let zero_val = Dtype.zero dtype in
    full ctx ~dtype shape_arr zero_val

  let ones ctx ~(dtype : ('a, 'b) Dtype.t) (shape_arr : int array) : ('a, 'b) t
      =
    let one_val = Dtype.one dtype in
    full ctx ~dtype shape_arr one_val

  let full_like ctx (x : ('a, 'b) t) (fill_value : 'c) : ('c, 'd) t =
    let target_shape = shape x in
    let self_dtype = B.dtype x in
    (* This is ('a, 'b) Dtype.t *)
    full ctx ~dtype:self_dtype target_shape fill_value

  let zeros_like ctx (x : ('a, 'b) t) : ('c, 'd) t =
    (* If opt_dt_param is None: - Result type of zeros_like is ('a, 'b) t (by
       unification 'c := 'a, 'd := 'b). - We need a fill_value of type 'a. - The
       dtype to use is `B.dtype x : ('a, 'b) Dtype.t`. *)
    let self_dtype = B.dtype x in
    let zero_val = Dtype.zero self_dtype in
    (* zero_val has type 'a *)
    (* Call `full_like ctx x zero_val` where `zero_val` is 'a. This
           specializes `full_like`'s 'c to 'a. Result is ('a, 'b) t. *)
    full_like ctx x zero_val (* ~dtype:None is implicit *)

  let ones_like ctx (x : ('a, 'b) t) : ('c, 'd) t =
    let self_dtype = B.dtype x in
    let one_val = Dtype.one self_dtype in
    (* one_val has type 'a *)
    full_like ctx x one_val (* ~dtype:None is implicit *)
end
