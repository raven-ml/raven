open Descriptor
open Views

module Make (B : Backend_intf.S) = struct
  module Creation = Backend_creation.Make (B)
  module Transform = Backend_transform.Make (B)

  let add context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.add context a_op b_op c;
    c

  let add_inplace context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    if shape a_desc <> out_shape && not (is_scalar a_desc) then
      invalid_arg
        "add_inplace: broadcasting in place requires 'a' to match output shape \
         or be scalar";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.add context a_op b_op a;
    a

  let add_scalar context a value =
    let a_desc = B.descriptor a in
    let scalar = Creation.scalar context (dtype a_desc) value in
    add context a scalar

  let sub context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.sub context a_op b_op c;
    c

  let sub_inplace context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    if shape a_desc <> out_shape && not (is_scalar a_desc) then
      invalid_arg
        "sub_inplace: broadcasting in place requires 'a' to match output shape \
         or be scalar";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.sub context a_op b_op a;
    a

  let sub_scalar context a value =
    let a_desc = B.descriptor a in
    let scalar = Creation.scalar context (dtype a_desc) value in
    sub context a scalar

  let mul context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.mul context a_op b_op c;
    c

  let mul_inplace context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    if shape a_desc <> out_shape && not (is_scalar a_desc) then
      invalid_arg
        "mul_inplace: broadcasting in place requires 'a' to match output shape \
         or be scalar";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.mul context a_op b_op a;
    a

  let mul_scalar context a value =
    let a_desc = B.descriptor a in
    let scalar = Creation.scalar context (dtype a_desc) value in
    mul context a scalar

  let div context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.div context a_op b_op c;
    c

  let div_inplace context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    if shape a_desc <> out_shape && not (is_scalar a_desc) then
      invalid_arg
        "div_inplace: broadcasting in place requires 'a' to match output shape \
         or be scalar";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.div context a_op b_op a;
    a

  let div_scalar context a value =
    let a_desc = B.descriptor a in
    let scalar = Creation.scalar context (dtype a_desc) value in
    div context a scalar

  let pow context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.pow context a_op b_op c;
    c

  let pow_inplace context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    if shape a_desc <> out_shape && not (is_scalar a_desc) then
      invalid_arg
        "pow_inplace: broadcasting in place requires 'a' to match output shape \
         or be scalar";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.pow context a_op b_op a;
    a

  let pow_scalar context a value =
    let a_desc = B.descriptor a in
    let scalar = Creation.scalar context (dtype a_desc) value in
    pow context a scalar

  let equal context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context uint8 out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.equal context a_op b_op c;
    c

  let rem context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context (dtype a_desc) out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.rem context a_op b_op c;
    c

  let rem_inplace context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    if shape a_desc <> out_shape && not (is_scalar a_desc) then
      invalid_arg
        "rem_inplace: broadcasting in place requires 'a' to match output shape \
         or be scalar";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.rem context a_op b_op a;
    a

  let rem_scalar context a value =
    let a_desc = B.descriptor a in
    let scalar = Creation.scalar context (dtype a_desc) value in
    rem context a scalar

  let maximum context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    let cond = B.empty context uint8 out_shape in
    B.greater context a_op b_op cond;
    let out = B.empty context (dtype a_desc) out_shape in
    B.where context cond a_op b_op out;
    out

  let maximum_inplace context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    if shape a_desc <> out_shape then
      invalid_arg "maximum_inplace: output shape must match broadcasted shape";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    let cond = B.empty context uint8 out_shape in
    B.greater context a_op b_op cond;
    B.where context cond a_op b_op a;
    a

  let maximum_scalar context a value =
    let a_desc = B.descriptor a in
    let scalar = Creation.scalar context (dtype a_desc) value in
    maximum context a scalar

  let minimum context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    let cond = B.empty context uint8 out_shape in
    B.less context a_op b_op cond;
    let out = B.empty context (dtype a_desc) out_shape in
    B.where context cond a_op b_op out;
    out

  let minimum_inplace context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    if shape a_desc <> out_shape then
      invalid_arg "minimum_inplace: output shape must match broadcasted shape";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    let cond = B.empty context uint8 out_shape in
    B.less context a_op b_op cond;
    B.where context cond a_op b_op a;
    a

  let minimum_scalar context a value =
    let a_desc = B.descriptor a in
    let scalar = Creation.scalar context (dtype a_desc) value in
    minimum context a scalar

  let greater context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context uint8 out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.greater context a_op b_op c;
    c

  let greater_equal context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let greater = B.empty context uint8 out_shape in
    let eq = B.empty context uint8 out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.greater context a_op b_op greater;
    B.equal context a_op b_op eq;
    let greater_equal = B.empty context uint8 out_shape in
    B.add context greater eq greater_equal;
    greater_equal

  let less context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let c = B.empty context uint8 out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.less context a_op b_op c;
    c

  let less_equal context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let less = B.empty context uint8 out_shape in
    let eq = B.empty context uint8 out_shape in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    B.less context a_op b_op less;
    B.equal context a_op b_op eq;
    let less_equal = B.empty context uint8 out_shape in
    B.add context less eq less_equal;
    less_equal

  let neg context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.neg context a out;
    out

  let neg_inplace context a =
    B.neg context a a;
    a

  let abs context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.abs context a out;
    out

  let abs_inplace context a =
    B.abs context a a;
    a

  let sign context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.sign context a out;
    out

  let sign_inplace context a =
    B.sign context a a;
    a

  let sqrt context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.sqrt context a out;
    out

  let sqrt_inplace context a =
    B.sqrt context a a;
    a

  let exp context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.exp context a out;
    out

  let exp_inplace context a =
    B.exp context a a;
    a

  let log context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.log context a out;
    out

  let log_inplace context a =
    B.log context a a;
    a

  let sin context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.sin context a out;
    out

  let sin_inplace context a =
    B.sin context a a;
    a

  let cos context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.cos context a out;
    out

  let cos_inplace context a =
    B.cos context a a;
    a

  let tan context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.tan context a out;
    out

  let tan_inplace context a =
    B.tan context a a;
    a

  let asin context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.asin context a out;
    out

  let asin_inplace context a =
    B.asin context a a;
    a

  let acos context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.acos context a out;
    out

  let acos_inplace context a =
    B.acos context a a;
    a

  let atan context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.atan context a out;
    out

  let atan_inplace context a =
    B.atan context a a;
    a

  let sinh context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.sinh context a out;
    out

  let sinh_inplace context a =
    B.sinh context a a;
    a

  let cosh context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.cosh context a out;
    out

  let cosh_inplace context a =
    B.cosh context a a;
    a

  let tanh context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.tanh context a out;
    out

  let tanh_inplace context a =
    B.tanh context a a;
    a

  let asinh context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.asinh context a out;
    out

  let asinh_inplace context a =
    B.asinh context a a;
    a

  let acosh context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.acosh context a out;
    out

  let acosh_inplace context a =
    B.acosh context a a;
    a

  let atanh context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.atanh context a out;
    out

  let atanh_inplace context a =
    B.atanh context a a;
    a

  let square context a =
    let desc = B.descriptor a in
    let out = B.empty context (dtype desc) (shape desc) in
    B.mul context a a out;
    out

  let square_inplace context a =
    B.mul context a a a;
    a

  let where context mask a b =
    let mask_desc = B.descriptor mask in
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    (* Compute output shape by broadcasting mask, a, and b shapes *)
    let out_shape =
      broadcast_shapes (shape mask_desc)
        (broadcast_shapes (shape a_desc) (shape b_desc))
    in
    let out_ndim = Array.length out_shape in
    let mask_shape = shape mask_desc in
    let mask_ndim = Array.length mask_shape in
    (* Prepare mask by adding dimensions on the right if needed *)
    let mask_op =
      if mask_ndim < out_ndim then
        (* Add ones on the right to match out_shape's dimensions *)
        let new_mask_shape =
          Array.append mask_shape (Array.make (out_ndim - mask_ndim) 1)
        in
        let mask_reshaped = Transform.reshape context new_mask_shape mask in
        B.view
          (broadcast_to (B.descriptor mask_reshaped) out_shape)
          mask_reshaped
      else B.view (broadcast_to mask_desc out_shape) mask
    in
    (* Broadcast a and b to out_shape *)
    let a_op = B.view (broadcast_to a_desc out_shape) a in
    let b_op = B.view (broadcast_to b_desc out_shape) b in
    let out = B.empty context (dtype a_desc) out_shape in
    B.where context mask_op a_op b_op out;
    out

  let nonzero context a =
    (* Extract tensor properties *)
    let desc_a = B.descriptor a in
    let shape_a = shape desc_a in
    let ndim = Array.length shape_a in
    let total = size desc_a in
    let buf_a = B.buffer a in
    let off_a = offset desc_a in
    let dtype_a = dtype desc_a in
    let zero_val = zero dtype_a in

    (* First pass: count nonzero elements *)
    let count = ref 0 in
    for k = 0 to total - 1 do
      let v = Bigarray.Array1.unsafe_get buf_a (off_a + k) in
      if v <> zero_val then incr count
    done;
    let m = !count in

    (* Create output tensor with shape (ndim, m) *)
    let out_shape = [| ndim; m |] in
    let out = B.empty context int64 out_shape in
    let desc_out = B.descriptor out in
    let buf_out = B.buffer out in
    let off_out = offset desc_out in

    (* Second pass: fill output with indices *)
    let count = ref 0 in
    for k = 0 to total - 1 do
      let v = Bigarray.Array1.unsafe_get buf_a (off_a + k) in
      if v <> zero_val then (
        let idxs = linear_to_md_c_contig k shape_a in
        for axis = 0 to ndim - 1 do
          let out_lin = off_out + (axis * m) + !count in
          Bigarray.Array1.unsafe_set buf_out out_lin (Int64.of_int idxs.(axis))
        done;
        incr count)
    done;
    out

  let sum context ?axes ?(keepdims = false) t =
    let desc = B.descriptor t in
    let ndim = Array.length (shape desc) in
    let axes_default = Array.init ndim Fun.id in
    let axes = Option.value ~default:axes_default axes in
    let to_reduce = Array.make ndim false in
    Array.iter
      (fun ax ->
        if ax >= 0 && ax < ndim then to_reduce.(ax) <- true
        else invalid_arg "axis out of bounds")
      axes;
    let out_shape =
      match keepdims with
      | true ->
          Array.mapi (fun i s -> if to_reduce.(i) then 1 else s) (shape desc)
      | false ->
          let out_shape = ref [] in
          for i = 0 to ndim - 1 do
            if not to_reduce.(i) then
              out_shape := (shape desc).(i) :: !out_shape
          done;
          Array.of_list (List.rev !out_shape)
    in
    let out = B.empty context (dtype desc) out_shape in
    B.sum context ~axes ~keepdims t out;
    out

  let prod context ?axes ?(keepdims = false) t =
    let desc = B.descriptor t in
    let ndim = Array.length (shape desc) in
    let axes_default = Array.init ndim Fun.id in
    let axes = Option.value ~default:axes_default axes in
    let to_reduce = Array.make ndim false in
    Array.iter
      (fun ax ->
        if ax >= 0 && ax < ndim then to_reduce.(ax) <- true
        else invalid_arg "axis out of bounds")
      axes;
    let out_shape =
      match keepdims with
      | true ->
          Array.mapi (fun i s -> if to_reduce.(i) then 1 else s) (shape desc)
      | false ->
          let out_shape = ref [] in
          for i = 0 to ndim - 1 do
            if not to_reduce.(i) then
              out_shape := (shape desc).(i) :: !out_shape
          done;
          Array.of_list (List.rev !out_shape)
    in
    let out = B.empty context (dtype desc) out_shape in
    B.prod context ~axes ~keepdims t out;
    out

  let mean context ?axes ?(keepdims = false) t =
    let desc = B.descriptor t in
    let ndim = Array.length (shape desc) in
    let axes_default = Array.init ndim Fun.id in
    let axes = Option.value ~default:axes_default axes in
    let sum_t = sum context ~axes ~keepdims t in
    let product =
      Array.fold_left (fun acc ax -> acc * (shape desc).(ax)) 1 axes
    in
    let num_elements = float_of_int product in
    let scalar_arr = B.empty context (dtype desc) [||] in
    B.fill context (1.0 /. num_elements) scalar_arr;
    let sum_desc = B.descriptor sum_t in
    let out_shape = shape sum_desc in
    let scalar_op_desc = broadcast_to (B.descriptor scalar_arr) out_shape in
    let scalar_op = B.view scalar_op_desc scalar_arr in
    let out = B.empty context (dtype desc) out_shape in
    B.mul context sum_t scalar_op out;
    out

  let max context ?axes ?(keepdims = false) t =
    let desc = B.descriptor t in
    let ndim = Array.length (shape desc) in
    let axes_default = Array.init ndim Fun.id in
    let axes = Option.value ~default:axes_default axes in
    let to_reduce = Array.make ndim false in
    Array.iter
      (fun ax ->
        if ax >= 0 && ax < ndim then to_reduce.(ax) <- true
        else invalid_arg "axis out of bounds")
      axes;
    let out_shape =
      match keepdims with
      | true ->
          Array.mapi (fun i s -> if to_reduce.(i) then 1 else s) (shape desc)
      | false ->
          let out_shape = ref [] in
          for i = 0 to ndim - 1 do
            if not to_reduce.(i) then
              out_shape := (shape desc).(i) :: !out_shape
          done;
          Array.of_list (List.rev !out_shape)
    in
    let out = B.empty context (dtype desc) out_shape in
    B.max context ~axes ~keepdims t out;
    out

  let min context ?axes ?(keepdims = false) t =
    let desc = B.descriptor t in
    let ndim = Array.length (shape desc) in
    let axes_default = Array.init ndim Fun.id in
    let axes = Option.value ~default:axes_default axes in
    let to_reduce = Array.make ndim false in
    Array.iter
      (fun ax ->
        if ax >= 0 && ax < ndim then to_reduce.(ax) <- true
        else invalid_arg "axis out of bounds")
      axes;
    let out_shape =
      match keepdims with
      | true ->
          Array.mapi (fun i s -> if to_reduce.(i) then 1 else s) (shape desc)
      | false ->
          let out_shape = ref [] in
          for i = 0 to ndim - 1 do
            if not to_reduce.(i) then
              out_shape := (shape desc).(i) :: !out_shape
          done;
          Array.of_list (List.rev !out_shape)
    in
    let out = B.empty context (dtype desc) out_shape in
    B.min context ~axes ~keepdims t out;
    out

  let var context ?axes ?(keepdims = false) t =
    (* 1) Determine axes to reduce and output shape *)
    let desc = B.descriptor t in
    let ndim = Array.length (shape desc) in
    let axes_default = Array.init ndim Fun.id in
    let axes = Option.value ~default:axes_default axes in
    let to_reduce = Array.make ndim false in
    Array.iter
      (fun ax ->
        let real_ax = if ax < 0 then ndim + ax else ax in
        if real_ax < 0 || real_ax >= ndim then
          invalid_arg "var: axis out of bounds"
        else to_reduce.(real_ax) <- true)
      axes;

    (* 2) Compute sums: sum1 = Σ x, sum2 = Σ x² *)
    let sum1 = sum context ~axes ~keepdims t in
    let sq = mul context t t in
    let sum2 = sum context ~axes ~keepdims sq in

    (* 3) Build divisor = number of elements reduced over as a broadcasted
       tensor *)
    let count = Array.fold_left (fun acc ax -> acc * desc.shape.(ax)) 1 axes in
    let scalar = B.empty context desc.dtype [||] in
    B.fill context (zero desc.dtype +. float_of_int count) scalar;
    let scalar_desc =
      broadcast_to (B.descriptor scalar) (shape (B.descriptor sum1))
    in
    let divisor = B.view scalar_desc scalar in

    (* 4) E[x²] – (E[x])² *)
    let mean2 = div context sum2 divisor in
    let mean1 = div context sum1 divisor in
    let mean1_sq = mul context mean1 mean1 in
    sub context mean2 mean1_sq

  let std context ?axes ?keepdims t =
    let v = var context ?axes ?keepdims t in
    sqrt context v

  let array_equal context a b =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    if shape a_desc <> shape b_desc then false
    else if size a_desc = 0 then true
    else
      let eq = equal context a b in
      let min_eq = min context eq in
      let scalar_min = B.buffer min_eq in
      Bigarray.Array1.get scalar_min 0 = 1

  let fma context a b c =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let c_desc = B.descriptor c in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let out_shape = broadcast_shapes out_shape (shape c_desc) in
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let c_op_desc = broadcast_to c_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    let c_op = B.view c_op_desc c in
    let out = B.empty context (dtype a_desc) out_shape in
    B.fma context a_op b_op c_op out;
    out

  let fma_inplace context a b c =
    let a_desc = B.descriptor a in
    let b_desc = B.descriptor b in
    let c_desc = B.descriptor c in
    let out_shape = broadcast_shapes (shape a_desc) (shape b_desc) in
    let out_shape = broadcast_shapes out_shape (shape c_desc) in
    if shape a_desc <> out_shape && not (is_scalar a_desc) then
      invalid_arg
        "fma_inplace: broadcasting in place requires 'a' to match output shape \
         or be scalar";
    let a_op_desc = broadcast_to a_desc out_shape in
    let b_op_desc = broadcast_to b_desc out_shape in
    let c_op_desc = broadcast_to c_desc out_shape in
    let a_op = B.view a_op_desc a in
    let b_op = B.view b_op_desc b in
    let c_op = B.view c_op_desc c in
    B.fma context a_op b_op c_op a;
    a

  let sort context ?(axis = 0) t =
    let d = B.descriptor t in
    let ndim = Array.length d.shape in
    let axis = if axis < 0 then ndim + axis else axis in
    if axis < 0 || axis >= ndim then invalid_arg "sort: axis out of bounds";
    let out = B.empty context d.dtype d.shape in
    B.sort context ~axis t out;
    out

  let argsort context ?(axis = 0) t =
    let d = B.descriptor t in
    let ndim = Array.length d.shape in
    let axis = if axis < 0 then ndim + axis else axis in
    if axis < 0 || axis >= ndim then invalid_arg "argsort: axis out of bounds";
    (* use Int64 for indices *)
    let out = B.empty context Int64 d.shape in
    B.argsort context ~axis t out;
    out

  let argmax context ?(axis = 0) t =
    let d = B.descriptor t in
    let ndim = Array.length d.shape in
    let axis = if axis < 0 then ndim + axis else axis in
    if axis < 0 || axis >= ndim then invalid_arg "argmax: axis out of bounds";
    (* output drops that dimension *)
    let out_shape =
      let new_ndim = ndim - 1 in
      if new_ndim = 0 then [||]
      else
        let out = Array.make new_ndim 0 in
        let j = ref 0 in
        for i = 0 to ndim - 1 do
          if i <> axis then (
            out.(!j) <- d.shape.(i);
            incr j)
        done;
        out
    in
    let out = B.empty context Int64 out_shape in
    B.argmax context ~axis t out;
    out

  let argmin context ?(axis = 0) t =
    let d = B.descriptor t in
    let ndim = Array.length d.shape in
    let axis = if axis < 0 then ndim + axis else axis in
    if axis < 0 || axis >= ndim then invalid_arg "argmin: axis out of bounds";
    let out_shape =
      let new_ndim = ndim - 1 in
      if new_ndim = 0 then [||]
      else
        let out = Array.make new_ndim 0 in
        let j = ref 0 in
        for i = 0 to ndim - 1 do
          if i <> axis then (
            out.(!j) <- d.shape.(i);
            incr j)
        done;
        out
    in
    let out = B.empty context Int64 out_shape in
    B.argmin context ~axis t out;
    out

  let round context t =
    let d = B.descriptor t in
    let out = B.empty context d.dtype d.shape in
    B.round context t out;
    out

  let around = round

  let floor context t =
    let d = B.descriptor t in
    let out = B.empty context d.dtype d.shape in
    B.floor context t out;
    out

  let ceil context t =
    let d = B.descriptor t in
    let out = B.empty context d.dtype d.shape in
    B.ceil context t out;
    out

  let clip (type a b) context ~(min : a) ~(max : a) (t : (a, b) B.b_t) :
      (a, b) B.b_t =
    let d = B.descriptor t in
    let shape = d.shape in
    (* create scalar buffers *)
    let s_min = B.empty context d.dtype [||] in
    B.fill context min s_min;
    let s_max = B.empty context d.dtype [||] in
    B.fill context max s_max;
    (* broadcast to t’s shape *)
    let min_op = B.view (broadcast_to (B.descriptor s_min) shape) s_min in
    let max_op = B.view (broadcast_to (B.descriptor s_max) shape) s_max in
    (* t_clamped = maximum(t, min) then minimum(t_clamped, max) *)
    let tmp = B.empty context UInt8 shape in
    B.greater context t min_op tmp;
    (* tmp = t > min *)
    let ge = B.empty context d.dtype shape in
    B.where context tmp t min_op ge;
    (* ge = max(t,min) *)
    let cond = B.empty context uint8 shape in
    B.less context ge max_op cond;
    (* cond = ge < max *)
    let out = B.empty context d.dtype shape in
    B.where context cond ge max_op out;
    (* out = min(ge,max) *)
    out

  let unique context t =
    (* flatten to 1-D *)
    let flat = Transform.flatten context t in
    let d = B.descriptor flat in
    let buf = B.buffer flat in
    let len = size d in
    (* pull into OCaml set *)
    let module S = Set.Make (struct
      type t = Obj.t

      let compare = compare
    end) in
    let s =
      let rec loop i acc =
        if i = len then acc
        else
          let x = Obj.repr (Bigarray.Array1.get buf i) in
          loop (i + 1) (S.add x acc)
      in
      loop 0 S.empty
    in
    (* build result array *)
    let uniq_list = S.elements s |> List.map Obj.obj in
    let n = List.length uniq_list in
    let out = B.empty context d.dtype [| n |] in
    let out_buf = B.buffer out in
    List.iteri (fun i v -> Bigarray.Array1.set out_buf i v) uniq_list;
    out
end
