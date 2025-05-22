(* High-level tensor operations built on backend [B]. The type parameters ['a]
   and ['b] come from the backend tensor type. *)

module Make (B : Backend_intf.S) = struct
  type ('a, 'b) t = ('a, 'b) B.t
  type context = B.context

  let create_context () = B.create_context ()
  let shape t = View.shape (B.view t)
  let dtype t = B.dtype t
  let strides t = View.strides (B.view t)
  let stride i t = View.stride i (B.view t)
  let dims t = View.dims (B.view t)
  let dim i t = View.dim i (B.view t)
  let ndim t = View.ndim (B.view t)
  let itemsize t = Dtype.itemsize (B.dtype t)
  let size t = View.size (B.view t)
  let numel t = size t
  let nbytes t = numel t * itemsize t
  let offset t = View.offset (B.view t)
  let layout t = View.layout (B.view t)

  (* infer the dimension corresponding to [-1] in [new_shape_spec] *)
  let resolve_neg_one_shape current_shape new_shape_spec =
    let new_shape_spec_l = Array.to_list new_shape_spec in
    let current_numel = View.prod current_shape in
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
      (* when shape_spec includes zero dimensions *)
      if specified_numel = 0 then
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
    if shape x = new_shape then x else B.op_reshape ctx x new_shape

  (* reshape and expand [x] to [new_shape] following numpy-style rules *)
  let broadcast_to ctx (x : ('a, 'b) t) (new_shape : int array) : ('a, 'b) t =
    let current_shape = shape x in
    if current_shape = new_shape then x
    else
      let rank_current = Array.length current_shape in
      let rank_new = Array.length new_shape in
      if rank_current > rank_new then
        invalid_arg "Cannot broadcast tensor to fewer dimensions"
      else
        let padded_shape =
          if rank_current < rank_new then
            Array.append (Array.make (rank_new - rank_current) 1) current_shape
          else current_shape
        in
        let compatible = ref true in
        for i = 0 to rank_new - 1 do
          if not (padded_shape.(i) = new_shape.(i) || padded_shape.(i) = 1) then
            compatible := false
        done;
        if not !compatible then
          invalid_arg
            (Printf.sprintf "Cannot broadcast shape %s to %s (padded: %s)"
               (View.pp_int_array current_shape)
               (View.pp_int_array new_shape)
               (View.pp_int_array padded_shape));
        let x_reshaped =
          if padded_shape <> current_shape then reshape ctx x padded_shape
          else x
        in
        B.op_expand ctx x_reshaped new_shape

  (* return [x] and [y] broadcasted to a common shape *)
  let broadcasted ctx ?(reverse = false) x y =
    let a, b = if reverse then (y, x) else (x, y) in
    let broadcast_shape = View.broadcast_shapes (shape a) (shape b) in
    let a_broad = broadcast_to ctx a broadcast_shape in
    let b_broad = broadcast_to ctx b broadcast_shape in
    (a_broad, b_broad)

  (* like [broadcast_to] but [-1] keeps the original dimension *)
  let expand ctx (x : ('a, 'b) t) (shape_spec : int array) : ('a, 'b) t =
    let current_shape = shape x in
    let rank_current = Array.length current_shape in
    let rank_spec = Array.length shape_spec in
    let rank_new = max rank_current rank_spec in
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

  let cast ctx (x : ('a, 'b) t) (dt : ('c, 'd) Dtype.t) : ('c, 'd) t =
    B.op_cast ctx x dt

  (* ────────── element-wise ops ────────── *)

  let add ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
    let a', b' = broadcasted ctx a b in
    B.op_add ctx a' b'

  let mul ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
    let a', b' = broadcasted ctx a b in
    B.op_mul ctx a' b'

  let neg ctx (x : ('a, 'b) t) : ('a, 'b) t = B.op_neg ctx x

  let sub ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
    let a', b' = broadcasted ctx a b in
    let neg_b = B.op_neg ctx b' in
    B.op_add ctx a' neg_b

  let fdiv ctx (a : (float, 'b) t) (b : (float, 'b) t) : (float, 'b) t =
    let a_broad, b_broad = broadcasted ctx a b in
    B.op_fdiv ctx a_broad b_broad

  let idiv (type a b) ctx (a : (a, b) t) (b : (a, b) t) : (a, b) t =
    (match dtype a with
    | Dtype.Int8 | Dtype.UInt8 | Dtype.Int16 | Dtype.UInt16 | Dtype.Int32
    | Dtype.Int64 | Dtype.Int | Dtype.NativeInt ->
        ()
    | _ -> failwith "idiv: operands must be integer types.");
    let a_broad, b_broad = broadcasted ctx a b in
    B.op_idiv ctx a_broad b_broad

  let pow ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
    let a', b' = broadcasted ctx a b in
    B.op_pow ctx a' b'

  let maximum ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
    let a', b' = broadcasted ctx a b in
    B.op_max ctx a' b'

  let minimum ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
    let a_neg = neg ctx a in
    let b_neg = neg ctx b in
    let max_neg = maximum ctx a_neg b_neg in
    neg ctx max_neg

  let modulus ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
    let a', b' = broadcasted ctx a b in
    B.op_mod ctx a' b'

  (* ────────── comparison ops ────────── *)

  let cmplt ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : (int, Dtype.uint8_elt) t =
    if not (Dtype.eq (dtype a) (dtype b)) then
      failwith "cmplt: operands must have the same dtype. Cast explicitly.";
    let a', b' = broadcasted ctx a b in
    B.op_cmplt ctx a' b'

  let cmpne ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : (int, Dtype.uint8_elt) t =
    if not (Dtype.eq (dtype a) (dtype b)) then
      failwith "cmpne: operands must have the same dtype. Cast explicitly.";
    let a', b' = broadcasted ctx a b in
    B.op_cmpne ctx a' b'

  let logical_not ctx (a : (int, Dtype.uint8_elt) t) : (int, Dtype.uint8_elt) t
      =
    B.op_neg ctx a (* Assumes op_neg on uint8 means logical not (0->1, 1->0) *)

  let cmpeq ctx (a : ('a, 'b) t) (b : ('a, 'b) t) : (int, Dtype.uint8_elt) t =
    if not (Dtype.eq (dtype a) (dtype b)) then
      failwith "cmpeq: operands must have the same dtype. Cast explicitly.";
    let ne_result = cmpne ctx a b in
    logical_not ctx ne_result

  let cmpgt ctx a b = cmplt ctx b a
  let cmple ctx a b = logical_not ctx (cmpgt ctx a b)
  let cmpge ctx a b = logical_not ctx (cmplt ctx a b)

  (* ────────── logical ops ────────── *)

  let logical_and ctx (a : (int, Dtype.uint8_elt) t)
      (b : (int, Dtype.uint8_elt) t) : (int, Dtype.uint8_elt) t =
    let a_b, b_b = broadcasted ctx a b in
    B.op_and ctx a_b b_b

  let logical_or ctx (a : (int, Dtype.uint8_elt) t)
      (b : (int, Dtype.uint8_elt) t) : (int, Dtype.uint8_elt) t =
    let a_b, b_b = broadcasted ctx a b in
    B.op_or ctx a_b b_b

  let logical_xor ctx (a : (int, Dtype.uint8_elt) t)
      (b : (int, Dtype.uint8_elt) t) : (int, Dtype.uint8_elt) t =
    let a_b, b_b = broadcasted ctx a b in
    B.op_xor ctx a_b b_b

  (* ────────── float-only unary ops ────────── *)

  let log2 ctx x = B.op_log2 ctx x
  let exp2 ctx x = B.op_exp2 ctx x
  let sin ctx x = B.op_sin ctx x
  let sqrt ctx x = B.op_sqrt ctx x
  let recip ctx x = B.op_recip ctx x

  (* natural logarithm via [log2] *)
  let log ctx x =
    let log2_x = log2 ctx x in
    let ln_2_val = log 2.0 in
    let ln_2_tensor = B.op_const_scalar ctx ln_2_val Dtype.Float32 in
    let ln_2_b = broadcast_to ctx ln_2_tensor (shape log2_x) in
    B.op_mul ctx log2_x ln_2_b

  (* natural exponent via [exp2] *)
  let exp ctx x =
    let one_over_ln_2_val = 1.0 /. Stdlib.log 2.0 in
    (* scale input so that [exp2] matches [exp] *)
    let factor_tensor = B.op_const_scalar ctx one_over_ln_2_val (dtype x) in
    let factor_b = broadcast_to ctx factor_tensor (shape x) in
    let x_scaled = B.op_mul ctx x factor_b in
    B.op_exp2 ctx x_scaled

  (* cosine via [sin] shift *)
  let cos ctx x =
    let pi_half = Stdlib.acos 0.0 in
    let pi_half_tensor = B.op_const_scalar ctx pi_half (dtype x) in
    let pi_half_b = broadcast_to ctx pi_half_tensor (shape x) in
    let arg_to_sin = sub ctx pi_half_b x in
    B.op_sin ctx arg_to_sin

  (* tangent as sin/cos *)
  let tan ctx x =
    let sin_x = B.op_sin ctx x in
    let cos_x = cos ctx x in
    B.op_fdiv ctx sin_x cos_x

  (* zero out negative values *)
  let relu ctx (x : ('a, 'b) t) : ('a, 'b) t =
    let x_dt = dtype x in
    let zero_val = Dtype.zero x_dt in
    let zero_tensor = B.op_const_scalar ctx zero_val x_dt in
    let zero_b = broadcast_to ctx zero_tensor (shape x) in
    let cond = cmpgt ctx x zero_b in
    B.op_where ctx cond x zero_b

  (* ────────── reduction ops ────────── *)

  let sum ctx ?(axes : int array option) ?(keepdims = false) (x : ('a, 'b) t) :
      ('a, 'b) t =
    let current_shape = shape x in
    let rank = Array.length current_shape in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_sum ctx x ~axes:axes_to_reduce ~keepdims

  let max ctx ?(axes : int array option) ?(keepdims = false) (x : ('a, 'b) t) :
      ('a, 'b) t =
    let rank = Array.length (shape x) in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_max ctx x ~axes:axes_to_reduce ~keepdims

  let prod ctx ?(axes : int array option) ?(keepdims = false) (x : ('a, 'b) t) :
      ('a, 'b) t =
    let rank = Array.length (shape x) in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list
    in
    B.op_reduce_prod ctx x ~axes:axes_to_reduce ~keepdims

  (* ────────── movement ops ────────── *)
  let permute ctx (x : ('a, 'b) t) (axes_param : int array) : ('a, 'b) t =
    let rank = ndim x in
    let axes =
      Array.map (fun ax -> if ax < 0 then ax + rank else ax) axes_param
    in
    B.op_permute ctx x axes

  let pad ctx (x : ('a, 'b) t) (padding_config : (int * int) array)
      (fill_value : 'a) : ('a, 'b) t =
    B.op_pad ctx x padding_config fill_value

  let shrink ctx (x : ('a, 'b) t) (shrink_args : (int * int) array) : ('a, 'b) t
      =
    B.op_shrink ctx x shrink_args

  let flip ctx (x : ('a, 'b) t) (flip_axes_bools : bool array) : ('a, 'b) t =
    B.op_flip ctx x flip_axes_bools

  let contiguous ctx (x : ('a, 'b) t) : ('a, 'b) t = B.op_contiguous ctx x

  (* ────────── ternary ops ────────── *)
  (* select between [if_true] and [if_false] based on [cond] *)
  let where ctx (cond : (int, Dtype.uint8_elt) t) (if_true : ('a, 'b) t)
      (if_false : ('a, 'b) t) : ('a, 'b) t =
    if not (Dtype.eq (dtype if_true) (dtype if_false)) then
      failwith
        "where: if_true and if_false must have the same dtype. Cast explicitly.";

    let s_true = shape if_true in
    let s_false = shape if_false in
    let s_cond = shape cond in
    let target_shape_val = View.broadcast_shapes s_true s_false in
    let final_target_shape = View.broadcast_shapes target_shape_val s_cond in

    let cond_b = broadcast_to ctx cond final_target_shape in
    let if_true_b = broadcast_to ctx if_true final_target_shape in
    let if_false_b = broadcast_to ctx if_false final_target_shape in
    B.op_where ctx cond_b if_true_b if_false_b

  (* ────────── creation ops ────────── *)

  let empty ctx dtype shape_arr =
    let numel = View.prod shape_arr in
    let buf = B.op_buffer ctx dtype numel in
    reshape ctx buf shape_arr

  let full ctx (dt : ('a, 'b) Dtype.t) (target_shape : int array)
      (fill_value : 'a) : ('a, 'b) t =
    let scalar_tensor = B.op_const_scalar ctx fill_value dt in
    if Array.length target_shape = 0 then scalar_tensor
    else
      let rank = Array.length target_shape in
      let intermediate_shape = Array.make rank 1 in
      let reshaped_scalar = reshape ctx scalar_tensor intermediate_shape in
      expand ctx reshaped_scalar target_shape

  let zeros ctx (dtype : ('a, 'b) Dtype.t) (shape_arr : int array) : ('a, 'b) t
      =
    let zero_val = Dtype.zero dtype in
    full ctx dtype shape_arr zero_val

  let ones ctx (dtype : ('a, 'b) Dtype.t) (shape_arr : int array) : ('a, 'b) t =
    let one_val = Dtype.one dtype in
    full ctx dtype shape_arr one_val

  let full_like ctx (x_ref : ('a, 'b) t) (fill_value : 'a) : ('a, 'b) t =
    let target_shape = shape x_ref in
    let self_dtype = B.dtype x_ref in
    full ctx self_dtype target_shape fill_value

  let zeros_like ctx (x : ('a, 'b) t) : ('a, 'b) t =
    let self_dtype = B.dtype x in
    let zero_val = Dtype.zero self_dtype in
    full_like ctx x zero_val

  let ones_like ctx (x : ('a, 'b) t) : ('a, 'b) t =
    let self_dtype = B.dtype x in
    let one_val = Dtype.one self_dtype in
    full_like ctx x one_val

  (* collapse dimensions between [start_dim] and [end_dim] *)
  let flatten ctx ?(start_dim = 0) ?(end_dim = -1) (x : ('a, 'b) t) : ('a, 'b) t
      =
    let sh = shape x in
    let r = Array.length sh in
    let s_orig = start_dim in
    let e_orig = end_dim in
    let s = if s_orig < 0 then s_orig + r else s_orig in
    let e = if e_orig < 0 then e_orig + r else e_orig in

    if
      not
        ((s >= 0 && s < r && e >= 0 && e < r)
        || (r = 0 && (s = 0 || s_orig = 0) && (e = -1 || e_orig = -1)))
    then
      invalid_arg
        (Printf.sprintf
           "flatten: start_dim %d or end_dim %d out of bounds for rank %d"
           start_dim end_dim r);
    if s > e then invalid_arg "flatten: start_dim must be <= end_dim";

    let new_shape_list =
      if r = 0 then [ 1 ]
      else
        let pre = Array.to_list (Array.sub sh 0 s) in
        let mid_slice = Array.sub sh s (e - s + 1) in
        let mid_prod =
          if Array.length mid_slice = 0 then 1 else View.prod mid_slice
        in
        (* treat missing middle slice as scalar *)
        let post = Array.to_list (Array.sub sh (e + 1) (r - (e + 1))) in
        pre @ [ mid_prod ] @ post
    in
    reshape ctx x (Array.of_list new_shape_list)

  (* drop axes of size 1; [axis] restricts the squeeze *)
  let squeeze ctx ?(axis : int option) (x : ('a, 'b) t) : ('a, 'b) t =
    let sh = shape x in
    match axis with
    | None ->
        let new_shape_list = List.filter (( <> ) 1) (Array.to_list sh) in
        let new_shape = Array.of_list new_shape_list in
        if Array.length new_shape = 0 && Array.length sh > 0 then
          reshape ctx x [| 1 |]
        else if Array.length new_shape = 0 && Array.length sh = 0 then x
        else reshape ctx x new_shape
    | Some ax_val ->
        let r = Array.length sh in
        if r = 0 then x (* Cannot squeeze a scalar *)
        else
          let ax = if ax_val < 0 then ax_val + r else ax_val in
          if ax < 0 || ax >= r then invalid_arg "squeeze: axis out of bounds"
          else if sh.(ax) <> 1 then x
          else
            let sh_list = Array.to_list sh in
            let new_shape_list = List.filteri (fun i _ -> i <> ax) sh_list in
            let new_shape = Array.of_list new_shape_list in
            if Array.length new_shape = 0 && Array.length sh > 0 then
              reshape ctx x [| 1 |]
            else if Array.length new_shape = 0 && Array.length sh = 0 then x
            else reshape ctx x new_shape

  (* insert a size‑1 dimension at [axis] *)
  let unsqueeze ctx ?(axis : int option) (x : ('a, 'b) t) : ('a, 'b) t =
    let sh = shape x in
    let r = Array.length sh in
    let ax_val =
      match axis with
      | None -> invalid_arg "unsqueeze: axis must be specified"
      | Some ax_v -> ax_v
    in
    let ax = if ax_val < 0 then ax_val + r + 1 else ax_val in

    if ax < 0 || ax > r then
      invalid_arg
        (Printf.sprintf "unsqueeze: axis %d out of bounds for rank %d" ax_val r);

    let sh_list = Array.to_list sh in
    let rec insert_at_idx current_idx target_idx lst item =
      match lst with
      | [] ->
          if current_idx = target_idx then [ item ]
          else invalid_arg "insert_at_idx: target_idx too large"
      | h :: t ->
          if current_idx = target_idx then item :: lst
          else h :: insert_at_idx (current_idx + 1) target_idx t item
    in
    let new_shape_list = insert_at_idx 0 ax sh_list 1 in
    reshape ctx x (Array.of_list new_shape_list)

  (* swap two dimensions *)
  let transpose ctx ?(dim0 = -2) ?(dim1 = -1) (x : ('a, 'b) t) : ('a, 'b) t =
    let r = ndim x in
    if r < 2 then x
    else
      let d0 = if dim0 < 0 then dim0 + r else dim0 in
      let d1 = if dim1 < 0 then dim1 + r else dim1 in
      if d0 < 0 || d0 >= r || d1 < 0 || d1 >= r then
        invalid_arg "transpose: dims out of bounds";
      if d0 = d1 then x
      else
        let axes = Array.init r Fun.id in
        let temp = axes.(d0) in
        axes.(d0) <- axes.(d1);
        axes.(d1) <- temp;
        permute ctx x axes
end
