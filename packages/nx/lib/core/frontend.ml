(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Make (B : Backend_intf.S) = struct
  module B = B

  let span ?attrs ~op () =
    let hook = !Instrumentation.current_hook in
    if hook.enabled then hook.with_span ~op ?attrs else fun f -> f ()

  let ( let@ ) m f = m f
  let err op fmt = Printf.ksprintf (fun msg -> invalid_arg (op ^ ": " ^ msg)) fmt

  (* ───── Core Types ───── *)

  type ('a, 'b) t = ('a, 'b) B.t
  type context = B.context
  type float16_elt = Nx_buffer.float16_elt
  type float32_elt = Nx_buffer.float32_elt
  type float64_elt = Nx_buffer.float64_elt
  type bfloat16_elt = Nx_buffer.bfloat16_elt
  type float8_e4m3_elt = Nx_buffer.float8_e4m3_elt
  type float8_e5m2_elt = Nx_buffer.float8_e5m2_elt
  type int4_elt = Nx_buffer.int4_signed_elt
  type uint4_elt = Nx_buffer.int4_unsigned_elt
  type int8_elt = Nx_buffer.int8_signed_elt
  type uint8_elt = Nx_buffer.int8_unsigned_elt
  type int16_elt = Nx_buffer.int16_signed_elt
  type uint16_elt = Nx_buffer.int16_unsigned_elt
  type int32_elt = Nx_buffer.int32_elt
  type uint32_elt = Nx_buffer.uint32_elt
  type int64_elt = Nx_buffer.int64_elt
  type uint64_elt = Nx_buffer.uint64_elt
  type complex32_elt = Nx_buffer.complex32_elt
  type complex64_elt = Nx_buffer.complex64_elt
  type bool_elt = Nx_buffer.bool_elt

  type ('a, 'b) dtype = ('a, 'b) Dtype.t =
    | Float16 : (float, float16_elt) dtype
    | Float32 : (float, float32_elt) dtype
    | Float64 : (float, float64_elt) dtype
    | BFloat16 : (float, bfloat16_elt) dtype
    | Float8_e4m3 : (float, float8_e4m3_elt) dtype
    | Float8_e5m2 : (float, float8_e5m2_elt) dtype
    | Int4 : (int, int4_elt) dtype
    | UInt4 : (int, uint4_elt) dtype
    | Int8 : (int, int8_elt) dtype
    | UInt8 : (int, uint8_elt) dtype
    | Int16 : (int, int16_elt) dtype
    | UInt16 : (int, uint16_elt) dtype
    | Int32 : (int32, int32_elt) dtype
    | UInt32 : (int32, uint32_elt) dtype
    | Int64 : (int64, int64_elt) dtype
    | UInt64 : (int64, uint64_elt) dtype
    | Complex64 : (Complex.t, complex32_elt) dtype
    | Complex128 : (Complex.t, complex64_elt) dtype
    | Bool : (bool, bool_elt) dtype

  type float16_t = (float, float16_elt) t
  type float32_t = (float, float32_elt) t
  type float64_t = (float, float64_elt) t
  type int8_t = (int, int8_elt) t
  type uint8_t = (int, uint8_elt) t
  type int16_t = (int, int16_elt) t
  type uint16_t = (int, uint16_elt) t
  type int32_t = (int32, int32_elt) t
  type int64_t = (int64, int64_elt) t
  type uint32_t = (int32, uint32_elt) t
  type uint64_t = (int64, uint64_elt) t
  type complex64_t = (Complex.t, complex32_elt) t
  type complex128_t = (Complex.t, complex64_elt) t
  type bool_t = (bool, bool_elt) t

  let float16 = Float16
  let float32 = Float32
  let float64 = Float64
  let bfloat16 = BFloat16
  let float8_e4m3 = Float8_e4m3
  let float8_e5m2 = Float8_e5m2
  let int4 = Int4
  let uint4 = UInt4
  let int8 = Int8
  let uint8 = UInt8
  let int16 = Int16
  let uint16 = UInt16
  let int32 = Int32
  let uint32 = UInt32
  let int64 = Int64
  let uint64 = UInt64
  let complex64 = Complex64
  let complex128 = Complex128
  let bool = Bool

  type index =
    | I of int
    | L of int list
    | R of int * int
    | Rs of int * int * int
    | A
    | M of (bool, bool_elt) t
    | N

  (* ───── Tensor Properties ───── *)

  let data x = B.to_host x

  let shape x =
    match Symbolic_shape.eval (View.shape (B.view x)) with
    | Some arr -> arr
    | None ->
        invalid_arg "shape: cannot get shape with unbound symbolic dimensions"

  let shape_symbolic x = View.shape (B.view x)
  let dtype x = B.dtype x
  let itemsize x = Dtype.itemsize (B.dtype x)

  let strides x =
    let view = B.view x in
    let itemsize = itemsize x in
    match View.strides_opt view with
    | Some elem_strides -> Array.map (fun s -> s * itemsize) elem_strides
    | None ->
        let reason =
          if not (View.is_materializable view) then
            "view has non-materializable layout"
          else if not (Symbolic_shape.is_static (View.shape view)) then
            "view has symbolic shape"
          else "view has complex striding pattern"
        in
        err "strides" "%s, call contiguous() to get a standard layout" reason

  let stride i x =
    let view = B.view x in
    let itemsize = itemsize x in
    match View.strides_opt view with
    | Some elem_strides ->
        let ndim = View.ndim view in
        let i = if i < 0 then i + ndim else i in
        if i < 0 || i >= ndim then
          err "stride" "axis %d out of bounds for %dD tensor" i ndim
        else elem_strides.(i) * itemsize
    | None ->
        err "stride" "stride for dimension %d, tensor does not have defined strides, call contiguous() first or check has_strides()" i

  let dims x =
    match Symbolic_shape.eval (View.shape (B.view x)) with
    | Some arr -> arr
    | None ->
        invalid_arg "dims: cannot get dimensions with unbound symbolic values"

  let dim i x =
    let shape = View.shape (B.view x) in
    let ndim = Symbolic_shape.rank shape in
    let i = if i < 0 then i + ndim else i in
    if i < 0 || i >= ndim then
      err "dim" "axis %d out of bounds for %dD tensor" i ndim
    else
      match Symbolic_shape.eval_dim shape.(i) with
      | Some n -> n
      | None ->
          invalid_arg "dim: cannot get dimension with unbound symbolic value"

  let ndim x = View.ndim (B.view x)

  let size x =
    match Symbolic_shape.eval_dim (View.numel (B.view x)) with
    | Some n -> n
    | None ->
        invalid_arg "size: cannot get size of tensor with symbolic shape, bind symbolic dimensions first"

  let numel x = size x
  let numel_symbolic x = View.numel (B.view x)

  let nbytes x =
    let itemsize = itemsize x in
    try numel x * itemsize
    with _ ->
      invalid_arg "nbytes: cannot compute bytes for symbolic tensor"

  let offset x = View.offset (B.view x)
  let is_c_contiguous x = View.is_c_contiguous (B.view x)

  (* ───── Internal Utilities ───── *)

  let array_prod arr = Array.fold_left ( * ) 1 arr

  module IntSet = Set.Make (Int)

  (* 2^shift_val for integer dtypes, used by lshift/rshift. *)
  let power_of_two : type a b. (a, b) Dtype.t -> int -> a =
   fun dtype shift_val ->
    if shift_val < 0 then
      err "power_of_two" "shift_val must be >= 0, got %d" shift_val;
    match dtype with
    | Int8 | UInt8 | Int16 | UInt16 -> (
        let power = 1 lsl shift_val in
        match dtype with
        | Int8 -> power
        | UInt8 -> power land 0xFF
        | Int16 -> power
        | UInt16 -> power land 0xFFFF
        | _ -> assert false)
    | Int32 -> Int32.shift_left Int32.one shift_val
    | UInt32 -> Int32.shift_left Int32.one shift_val
    | Int64 -> Int64.shift_left Int64.one shift_val
    | UInt64 -> Int64.shift_left Int64.one shift_val
    | _ ->
        err "power_of_two" "dtype %s, not an integer type" (Dtype.to_string dtype)

  let ensure_no_infer ~op shape =
    Array.iter
      (fun dim ->
        if Symbolic_shape.is_infer dim then
          err op "target shape cannot contain infer (-1) dimensions")
      shape

  let ensure_float_dtype fname x =
    if not (Dtype.is_float (dtype x)) then
      err fname "dtype %s, expected float type (Float16, Float32, or Float64)" (Dtype.to_string (dtype x))

  let ensure_int_dtype fname x =
    if not (Dtype.is_int (dtype x)) then
      invalid_arg (fname ^ ": dtype must be an integer type")

  let resolve_axis ?ndim_opt x (axis_opt : int option) =
    let ndim = match ndim_opt with Some n -> n | None -> ndim x in
    match axis_opt with
    | None -> Array.init ndim Fun.id
    | Some a ->
        let resolved_a = if a < 0 then a + ndim else a in
        [| resolved_a |]

  let resolve_single_axis ?ndim_opt x axis : int =
    let ndim = match ndim_opt with Some n -> n | None -> ndim x in
    if axis < 0 then axis + ndim else axis

  (* Normalize negative axes, validate bounds, sort, and deduplicate. *)
  let normalize_and_dedup_axes ~op ndim axes =
    let normalized =
      List.map
        (fun ax ->
          let axis = if ax < 0 then ndim + ax else ax in
          if axis < 0 || axis >= ndim then
            err op "axis %d out of bounds for %dD tensor" ax ndim;
          axis)
        axes
    in
    let sorted = List.sort compare normalized in
    let rec dedup prev acc = function
      | [] -> List.rev acc
      | h :: t when Some h = prev -> dedup prev acc t
      | h :: t -> dedup (Some h) (h :: acc) t
    in
    dedup None [] sorted

  (* Count elements across reduction axes. *)
  let reduction_element_count input_shape ?axes () =
    let rank = Array.length input_shape in
    let axes_arr =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.of_list
            (List.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list)
    in
    if Array.length axes_arr = 0 then 1
    else array_prod (Array.map (fun ax -> input_shape.(ax)) axes_arr)

  (* Write [result] into [?out] if provided, otherwise return [result]. *)
  let copy_to_out ?out result =
    match out with
    | Some o -> B.assign o result; o
    | None -> result

  (* ───── Shape Manipulation Helpers ───── *)

  let reshape_symbolic new_shape x =
    let current_shape = shape_symbolic x in
    let resolved =
      if Array.exists Symbolic_shape.is_infer new_shape then
        match
          Symbolic_shape.resolve_reshape ~from_shape:current_shape
            ~to_shape:new_shape
        with
        | Some s -> s
        | None ->
            invalid_arg "reshape: cannot infer dimension for symbolic reshape, bind symbolic dimensions or avoid using -1"
      else new_shape
    in
    (match
       (Symbolic_shape.eval current_shape, Symbolic_shape.eval resolved)
     with
    | Some old_arr, Some new_arr ->
        let old_numel = array_prod old_arr in
        let new_numel = array_prod new_arr in
        if old_numel <> new_numel && old_numel <> 0 && new_numel <> 0 then
          err "reshape" "cannot reshape %s to %s (%d\xe2\x86\x92%d elements)"
            (Shape.to_string old_arr) (Shape.to_string new_arr) old_numel new_numel
    | _ -> ());
    if Symbolic_shape.equal current_shape resolved then x
    else B.reshape x resolved

  let reshape shape_spec x =
    let@ _ = span ~op:"reshape" () in
    let infer_count = ref 0 in
    let target_shape =
      Array.map
        (fun dim ->
          if dim = -1 then (
            incr infer_count;
            Symbolic_shape.infer)
          else if dim < -1 then
            err "reshape" "shape specification, dimension %d < -1" dim
          else Symbolic_shape.static dim)
        shape_spec
    in
    if !infer_count > 1 then
      invalid_arg "reshape: shape specification, multiple -1 dimensions, can only specify one unknown dimension";
    reshape_symbolic target_shape x

  let broadcast_shapes shape_a shape_b =
    let rank_a = Symbolic_shape.rank shape_a in
    let rank_b = Symbolic_shape.rank shape_b in
    let rank_out = max rank_a rank_b in
    let static_one = Symbolic_shape.static 1 in
    let result = Array.make rank_out static_one in
    let dim_eq a b = Symbolic_shape.equal [| a |] [| b |] in
    for i = 0 to rank_out - 1 do
      let idx_a = rank_a - rank_out + i in
      let idx_b = rank_b - rank_out + i in
      let dim_a = if idx_a >= 0 then shape_a.(idx_a) else static_one in
      let dim_b = if idx_b >= 0 then shape_b.(idx_b) else static_one in
      let broadcast_err () =
        err "broadcast" "cannot broadcast dimensions %s and %s, bind symbolic dimensions first"
          (Symbolic_shape.to_string [| dim_a |])
          (Symbolic_shape.to_string [| dim_b |])
      in
      result.(i) <-
        (if dim_eq dim_a dim_b then dim_a
         else
           match
             (Symbolic_shape.eval_dim dim_a, Symbolic_shape.eval_dim dim_b)
           with
           | Some a, Some b ->
               if a = b then dim_a
               else if a = 1 then dim_b
               else if b = 1 then dim_a
               else (
                 match
                   (Symbolic_shape.eval shape_a, Symbolic_shape.eval shape_b)
                 with
                 | Some sa, Some sb ->
                     err "broadcast" "cannot broadcast %s with %s (dim %d: %d\xe2\x89\xa0%d)"
                       (Shape.to_string sa) (Shape.to_string sb) i a b
                 | _ -> broadcast_err ())
           | Some 1, None -> dim_b
           | None, Some 1 -> dim_a
           | _ -> broadcast_err ())
    done;
    result

  let broadcast_to_symbolic target_shape x =
    ensure_no_infer ~op:"broadcast_to" target_shape;
    let current_shape = shape_symbolic x in
    if Symbolic_shape.equal current_shape target_shape then x
    else
      let rank_current = Symbolic_shape.rank current_shape in
      let rank_target = Symbolic_shape.rank target_shape in
      if rank_current > rank_target then
        err "broadcast_to" "rank mismatch: source rank %d exceeds target rank %d, target shape must have at least as many dimensions as source"
          rank_current rank_target
      else
        let pad_count = rank_target - rank_current in
        let padded_shape =
          if pad_count <= 0 then current_shape
          else
            let arr = Array.make rank_target (Symbolic_shape.static 1) in
            Array.blit current_shape 0 arr pad_count rank_current;
            arr
        in
        let dim_eq a b = Symbolic_shape.equal [| a |] [| b |] in
        for i = 0 to rank_target - 1 do
          let curr_dim = padded_shape.(i) in
          let target_dim = target_shape.(i) in
          if dim_eq curr_dim target_dim then ()
          else
            match Symbolic_shape.eval_dim curr_dim with
            | Some 1 -> ()
            | Some curr_val -> (
                match Symbolic_shape.eval_dim target_dim with
                | Some target_val when curr_val = target_val -> ()
                | Some target_val ->
                    let to_ints sh =
                      Array.init rank_target (fun j ->
                          match Symbolic_shape.eval_dim sh.(j) with
                          | Some v -> v
                          | None -> -1)
                    in
                    err "broadcast_to" "cannot broadcast %s to %s (dim %d: %d\xe2\x89\xa0%d)"
                      (Shape.to_string (to_ints padded_shape))
                      (Shape.to_string (to_ints target_shape))
                      i curr_val target_val
                | None ->
                    err "broadcast_to" "dimension %d: cannot broadcast %s to %s"
                      i
                      (Symbolic_shape.to_string [| curr_dim |])
                      (Symbolic_shape.to_string [| target_dim |]))
            | None ->
                err "broadcast_to" "dimension %d: cannot broadcast %s to %s, bind symbolic dimensions first" i
                  (Symbolic_shape.to_string [| curr_dim |])
                  (Symbolic_shape.to_string [| target_dim |])
        done;
        let x_aligned =
          if pad_count <= 0 then x else reshape_symbolic padded_shape x
        in
        if Symbolic_shape.equal (shape_symbolic x_aligned) target_shape then
          x_aligned
        else B.expand x_aligned target_shape

  let broadcast_to new_shape x =
    let@ _ = span ~op:"broadcast_to" () in
    broadcast_to_symbolic
      (Array.map
         (fun dim ->
           if dim < 0 then
             err "broadcast_to" "target shape, dimension %d < 0" dim
           else Symbolic_shape.static dim)
         new_shape)
      x

  let broadcasted ?(reverse = false) x y =
    let a, b = if reverse then (y, x) else (x, y) in
    let broadcast_shape =
      broadcast_shapes (shape_symbolic a) (shape_symbolic b)
    in
    (broadcast_to_symbolic broadcast_shape a,
     broadcast_to_symbolic broadcast_shape b)

  (* Like [broadcast_to] but [-1] keeps the original dimension. *)
  let expand shape_spec x =
    let@ _ = span ~op:"expand" () in
    let current_shape = shape_symbolic x in
    let rank_current = Symbolic_shape.rank current_shape in
    let rank_spec = Array.length shape_spec in
    let rank_new = max rank_current rank_spec in
    let current_aligned =
      if rank_current = rank_new then current_shape
      else
        let arr = Array.make rank_new (Symbolic_shape.static 1) in
        Array.blit current_shape 0 arr (rank_new - rank_current) rank_current;
        arr
    in
    let target_shape =
      Array.init rank_new (fun i ->
          let spec_idx = i - (rank_new - rank_spec) in
          let spec_dim = if spec_idx < 0 then -1 else shape_spec.(spec_idx) in
          if spec_dim = -1 then current_aligned.(i)
          else if spec_dim < -1 then
            err "expand" "dimension %d, negative size %d" i spec_dim
          else Symbolic_shape.static spec_dim)
    in
    broadcast_to_symbolic target_shape x

  (* ───── Type Conversion and Tensor Creation ───── *)

  let cast (type a b c d) (dt : (c, d) Dtype.t) (x : (a, b) t) : (c, d) t =
    match Dtype.equal_witness (dtype x) dt with
    | Some Equal -> B.copy x
    | None ->
        let out = B.buffer (B.context x) dt (shape x) in
        B.cast ~out x;
        out

  let astype dt x = cast dt x
  let contiguous x = B.contiguous x

  let copy x =
    let@ _ = span ~op:"copy" () in
    B.copy x

  let blit src dst =
    let ss = shape src and ds = shape dst in
    if ss <> ds then
      err "blit" "shape mismatch %s vs %s, source and destination must have identical shapes"
        (Shape.to_string ss) (Shape.to_string ds);
    B.assign dst src

  let create ctx dtype shape arr =
    let n = Array.fold_left ( * ) 1 shape in
    if Array.length arr <> n then
      err "create" "array size, got %d elements, expected %d" (Array.length arr) n;
    let kind = Dtype.to_buffer_kind dtype in
    let bigarray = Nx_buffer.create kind n in
    for i = 0 to n - 1 do
      Nx_buffer.unsafe_set bigarray i arr.(i)
    done;
    let tensor_1d = B.from_host ctx bigarray in
    if Array.length shape = 1 && shape.(0) = n then tensor_1d
    else B.reshape tensor_1d (Symbolic_shape.of_ints shape)

  let init ctx dtype shape f =
    let size = Array.fold_left ( * ) 1 shape in
    let arr =
      Array.init size (fun i -> f (Shape.unravel_index i shape))
    in
    create ctx dtype shape arr

  let scalar ctx dt value = B.full ctx dt [||] value
  let scalar_like x_ref value = scalar (B.context x_ref) (B.dtype x_ref) value

  let fill value x =
    let copied = B.copy x in
    B.assign copied (broadcast_to (shape copied) (scalar_like copied value));
    copied

  let empty ctx dtype shape_arr = B.buffer ctx dtype shape_arr
  let zeros ctx dtype shape_arr = B.full ctx dtype shape_arr (Dtype.zero dtype)
  let ones ctx dtype shape_arr = B.full ctx dtype shape_arr (Dtype.one dtype)
  let full ctx dt target_shape fill_value = B.full ctx dt target_shape fill_value

  let create_like x_ref fill_fn =
    fill_fn (B.context x_ref) (B.dtype x_ref) (shape x_ref)

  let empty_like x_ref = create_like x_ref empty
  let full_like x_ref fill_value = create_like x_ref (fun ctx dt sh -> full ctx dt sh fill_value)
  let zeros_like x = full_like x (Dtype.zero (B.dtype x))
  let ones_like x = full_like x (Dtype.one (B.dtype x))

  let to_buffer x =
    let@ _ = span ~op:"to_buffer" () in
    let t =
      let t = if is_c_contiguous x && offset x = 0 then x else contiguous x in
      let buffer = data t in
      if Nx_buffer.length buffer = numel t then t else copy t
    in
    data t

  let to_bigarray x =
    let buf = to_buffer x in
    let _ = Dtype.to_bigarray_kind (B.dtype x) in
    let ga = Nx_buffer.to_genarray buf (shape x) in
    (Obj.magic ga : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t)

  let of_buffer ctx ~shape buf =
    reshape shape (B.from_host ctx buf)

  let of_bigarray ctx ba =
    let ga_ext : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t =
      Obj.magic ba
    in
    of_buffer ctx ~shape:(Bigarray.Genarray.dims ga_ext)
      (Nx_buffer.of_genarray ga_ext)

  let to_array x =
    let ba = data (contiguous x) in
    let n = numel x in
    Array.init n (fun i -> Nx_buffer.get ba i)

  (* ───── Element-wise Binary Operations ───── *)

  let binop ?out ?(op_name = "binop") op a b =
    let@ _ = span ~op:op_name () in
    let a', b' = broadcasted a b in
    let out =
      match out with
      | Some o -> o
      | None -> empty (B.context a') (B.dtype a') (shape a')
    in
    op ~out a' b';
    out

  let cmpop ?out ?(op_name = "cmpop") op a b =
    let@ _ = span ~op:op_name () in
    let a', b' = broadcasted a b in
    let out =
      match out with
      | Some o -> o
      | None -> empty (B.context a') Dtype.bool (shape a')
    in
    op ~out a' b';
    out

  let add ?out a b = binop ~op_name:"add" ?out B.add a b
  let add_s ?out t s = add ?out t (scalar_like t s)
  let radd_s ?out s t = add ?out (scalar_like t s) t

  let sub ?out a b = binop ~op_name:"sub" ?out B.sub a b
  let sub_s ?out t s = sub ?out t (scalar_like t s)
  let rsub_s ?out s t = sub ?out (scalar_like t s) t

  let mul ?out a b = binop ~op_name:"mul" ?out B.mul a b
  let mul_s ?out t s = mul ?out t (scalar_like t s)
  let rmul_s ?out s t = mul ?out (scalar_like t s) t

  let div ?out a b = binop ~op_name:"div" ?out B.div a b
  let div_s ?out t s = div ?out t (scalar_like t s)
  let rdiv_s ?out s t = div ?out (scalar_like t s) t

  let pow ?out a b = binop ~op_name:"pow" ?out B.pow a b
  let pow_s ?out t s = pow ?out t (scalar_like t s)
  let rpow_s ?out s t = pow ?out (scalar_like t s) t

  let maximum ?out a b = binop ~op_name:"maximum" ?out B.max a b
  let maximum_s ?out t s = maximum ?out t (scalar_like t s)
  let rmaximum_s ?out s t = maximum ?out (scalar_like t s) t

  let minimum ?out a b = binop ~op_name:"minimum" ?out B.min a b
  let minimum_s ?out t s = minimum ?out t (scalar_like t s)
  let rminimum_s ?out s t = minimum ?out (scalar_like t s) t

  let mod_ ?out a b = binop ~op_name:"mod" ?out B.mod_ a b
  let mod_s ?out t s = mod_ ?out t (scalar_like t s)
  let rmod_s ?out s t = mod_ ?out (scalar_like t s) t

  let bitwise_xor ?out a b = binop ~op_name:"bitwise_xor" ?out B.xor a b
  let bitwise_or ?out a b = binop ~op_name:"bitwise_or" ?out B.or_ a b
  let bitwise_and ?out a b = binop ~op_name:"bitwise_and" ?out B.and_ a b

  (* ───── Logical and Comparison Operations ───── *)

  let logical_and ?out a b = binop ?out B.and_ a b
  let logical_or ?out a b = binop ?out B.or_ a b
  let logical_xor ?out a b = binop ?out B.xor a b

  let logical_not (type a b) ?out (x : (a, b) t) : (a, b) t =
    let dt = dtype x in
    let one = full (B.context x) dt (shape x) (Dtype.one dt) in
    match dt with
    | Dtype.UInt8 | Dtype.Bool | Dtype.UInt4 -> binop ?out B.xor x one
    | _ -> sub ?out one x

  let cmpeq ?out a b = cmpop ~op_name:"equal" ?out B.cmpeq a b
  let cmpne ?out a b = cmpop ~op_name:"not_equal" ?out B.cmpne a b
  let cmplt ?out a b = cmpop ~op_name:"less" ?out B.cmplt a b
  let cmple ?out a b = cmpop ~op_name:"less_equal" ?out B.cmple a b
  let cmpgt ?out a b = cmplt ?out b a
  let cmpge ?out a b = cmple ?out b a

  let less = cmplt
  let less_equal = cmple
  let greater = cmpgt
  let greater_equal = cmpge
  let equal = cmpeq
  let not_equal = cmpne

  let equal_s ?out a s = equal ?out a (scalar_like a s)
  let not_equal_s ?out a s = not_equal ?out a (scalar_like a s)
  let less_s ?out a s = less ?out a (scalar_like a s)
  let greater_s ?out a s = greater ?out a (scalar_like a s)
  let less_equal_s ?out a s = less_equal ?out a (scalar_like a s)
  let greater_equal_s ?out a s = greater_equal ?out a (scalar_like a s)

  (* ───── Element-wise Unary Operations ───── *)

  let unaryop ?out ?(op_name = "unary") op x =
    let@ _ = span ~op:op_name () in
    let out =
      match out with
      | Some o -> o
      | None -> empty (B.context x) (B.dtype x) (shape x)
    in
    op ~out x;
    out

  let neg ?out x = unaryop ~op_name:"neg" ?out B.neg x

  let bitwise_not ?out x =
    let dt = dtype x in
    binop ?out B.xor x
      (broadcast_to (shape x) (B.full (B.context x) dt [||] (Dtype.minus_one dt)))

  let invert ?out x = bitwise_not ?out x

  let sin ?out x = unaryop ~op_name:"sin" ?out B.sin x
  let cos ?out x = unaryop ~op_name:"cos" ?out B.cos x
  let sqrt ?out x = unaryop ~op_name:"sqrt" ?out B.sqrt x
  let recip ?out x = unaryop ~op_name:"recip" ?out B.recip x
  let log ?out x = unaryop ~op_name:"log" ?out B.log x
  let exp ?out x = unaryop ~op_name:"exp" ?out B.exp x
  let abs ?out x = unaryop ~op_name:"abs" ?out B.abs x

  let log2 ?out x =
    mul ?out (log x)
      (broadcast_to (shape x)
         (scalar (B.context x) (dtype x) (Dtype.of_float (dtype x) (1.0 /. Stdlib.log 2.0))))

  let exp2 ?out x =
    exp ?out
      (mul x
         (broadcast_to (shape x)
            (scalar (B.context x) (dtype x) (Dtype.of_float (dtype x) (Stdlib.log 2.0)))))

  let tan ?out x = unaryop ~op_name:"tan" ?out B.tan x
  let square ?out x = mul ?out x x
  let sign ?out x = unaryop ~op_name:"sign" ?out B.sign x
  let relu ?out x = maximum ?out x (zeros_like x)

  let sigmoid ?out x =
    let dt = dtype x in
    let neg_one_over_log2 = B.full (B.context x) dt [||] (-1.0 /. Stdlib.log 2.0) in
    recip ?out (add (ones_like x) (exp2 (mul x neg_one_over_log2)))

  let rsqrt ?out x = recip ?out (sqrt x)
  let asin ?out x = unaryop ~op_name:"asin" ?out B.asin x
  let acos ?out x = unaryop ~op_name:"acos" ?out B.acos x
  let atan ?out x = unaryop ~op_name:"atan" ?out B.atan x
  let sinh ?out x = unaryop ~op_name:"sinh" ?out B.sinh x
  let cosh ?out x = unaryop ~op_name:"cosh" ?out B.cosh x
  let tanh ?out x = unaryop ~op_name:"tanh" ?out B.tanh x

  let asinh ?out x =
    let one_x = full (B.context x) (dtype x) (shape x) 1.0 in
    log ?out (add x (sqrt (add (square x) one_x)))

  let acosh ?out x =
    let one_x = full (B.context x) (dtype x) (shape x) 1.0 in
    log ?out (add x (sqrt (sub (square x) one_x)))

  let atanh ?out x =
    let one_x = full (B.context x) (dtype x) (shape x) 1.0 in
    let two_x = full (B.context x) (dtype x) (shape x) 2.0 in
    div ?out (log (div (add one_x x) (sub one_x x))) two_x

  let trunc ?out x = unaryop ~op_name:"trunc" ?out B.trunc x
  let ceil ?out x = unaryop ~op_name:"ceil" ?out B.ceil x
  let floor ?out x = unaryop ~op_name:"floor" ?out B.floor x
  let round ?out x = unaryop ~op_name:"round" ?out B.round x

  let isinf ?out x =
    if not (Dtype.is_float (dtype x)) then
      copy_to_out ?out (zeros (B.context x) Dtype.bool (shape x))
    else
      let dt = dtype x in
      let pos_inf = broadcast_to (shape x) (B.full (B.context x) dt [||] Float.infinity) in
      let neg_inf = broadcast_to (shape x) (B.full (B.context x) dt [||] Float.neg_infinity) in
      logical_or ?out (cmpeq x pos_inf) (cmpeq x neg_inf)

  let isnan ?out x =
    if not (Dtype.is_float (dtype x)) then
      copy_to_out ?out (zeros (B.context x) Dtype.bool (shape x))
    else cmpne ?out x x

  let isfinite ?out x =
    if not (Dtype.is_float (dtype x)) then
      copy_to_out ?out (ones (B.context x) Dtype.bool (shape x))
    else logical_not ?out (logical_or (isinf x) (isnan x))

  let lerp ?out start_tensor end_tensor weight =
    add ?out start_tensor (mul (sub end_tensor start_tensor) weight)

  let lerp_scalar_weight ?out start_tensor end_tensor weight_val =
    lerp ?out start_tensor end_tensor
      (full (B.context start_tensor) (dtype start_tensor)
         (shape start_tensor) weight_val)

  let shift_op ~op ~apply ?out x shift_val =
    let dt = dtype x in
    if not (Dtype.is_int dt) then
      err op "dtype %s, expected integer type" (Dtype.to_string dt);
    if shift_val < 0 then
      err op "shift_val must be >= 0, got %d" shift_val;
    if shift_val = 0 then copy_to_out ?out x
    else
      apply ?out x
        (broadcast_to (shape x)
           (B.full (B.context x) dt [||] (power_of_two dt shift_val)))

  let lshift ?out x shift_val =
    shift_op ~op:"lshift" ~apply:mul ?out x shift_val

  let rshift ?out x shift_val =
    shift_op ~op:"rshift"
      ~apply:(fun ?out a b -> binop ?out B.div a b)
      ?out x shift_val

  let clamp ?out ?min ?max x =
    let x =
      match min with
      | None -> x
      | Some min_v -> maximum x (full_like x min_v)
    in
    match max with
    | None -> copy_to_out ?out x
    | Some max_v -> minimum ?out x (full_like x max_v)

  let clip = clamp

  (* ───── Ternary Operations ───── *)

  let where ?out cond if_true if_false =
    let@ _ = span ~op:"where" () in
    let target = Shape.broadcast (shape if_true) (shape if_false) in
    let target = Shape.broadcast target (shape cond) in
    let cond_b = broadcast_to target cond in
    let if_true_b = broadcast_to target if_true in
    let if_false_b = broadcast_to target if_false in
    let out = match out with
      | Some o -> o
      | None -> empty (B.context if_true_b) (B.dtype if_true_b) target
    in
    B.where ~out cond_b if_true_b if_false_b;
    out

  (* ───── Binary Mathematical Functions ───── *)

  let atan2 ?out y x = binop ~op_name:"atan2" ?out B.atan2 y x

  (* sqrt(x² + y²) with overflow protection via max * sqrt(1 + (min/max)²) *)
  let hypot ?out x y =
    let x', y' = broadcasted x y in
    let x_abs = abs x' in
    let y_abs = abs y' in
    let max_val = maximum x_abs y_abs in
    let min_val = minimum x_abs y_abs in
    let both_zero =
      logical_and
        (cmpeq x_abs (zeros_like x_abs))
        (cmpeq y_abs (zeros_like y_abs))
    in
    let ratio = where both_zero (zeros_like min_val) (div min_val max_val) in
    let result = mul max_val (sqrt (add (ones_like ratio) (square ratio))) in
    where ?out both_zero (zeros_like result) result

  (* ───── Reduction Operations ───── *)

  let reduce_output_shape input_shape axes_to_reduce keepdims =
    if keepdims then
      Array.mapi
        (fun i dim -> if Array.exists (( = ) i) axes_to_reduce then 1 else dim)
        input_shape
    else
      let filtered = ref [] in
      Array.iteri
        (fun i dim ->
          if not (Array.exists (( = ) i) axes_to_reduce) then
            filtered := dim :: !filtered)
        input_shape;
      Array.of_list (List.rev !filtered)

  let reduce_op ?out backend_op ?axes ?(keepdims = false) x =
    let input_shape = shape x in
    let rank = Array.length input_shape in
    let axes_to_reduce = match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.of_list
            (List.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list)
    in
    Array.iter
      (fun ax ->
        if ax < 0 || ax >= rank then
          err "reduce" "axis %d out of bounds for %dD tensor" ax rank)
      axes_to_reduce;
    let out = match out with
      | Some o -> o
      | None ->
          empty (B.context x) (B.dtype x)
            (reduce_output_shape input_shape axes_to_reduce keepdims)
    in
    backend_op ~out ~axes:axes_to_reduce ~keepdims x;
    out

  let sum ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"sum" () in
    reduce_op ?out B.reduce_sum ?axes ~keepdims x

  let max ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"max" () in
    reduce_op ?out B.reduce_max ?axes ~keepdims x

  let min ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"min" () in
    reduce_op ?out B.reduce_min ?axes ~keepdims x

  let prod ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"prod" () in
    reduce_op ?out B.reduce_prod ?axes ~keepdims x

  let associative_scan ~axis op x =
    let x_shape = shape x in
    let rank = Array.length x_shape in
    if rank = 0 then
      let a = if axis < 0 then axis + 1 else axis in
      if a = 0 then x
      else
        err "associative_scan" "axis %d out of bounds for rank 0 tensor (only axis 0 valid)" axis
    else
      let a = if axis < 0 then axis + rank else axis in
      if a < 0 || a >= rank then
        err "associative_scan" "axis %d out of bounds for %dD tensor" axis rank
      else
        let out = empty (B.context x) (B.dtype x) x_shape in
        B.associative_scan ~out ~axis:a ~op x;
        out

  let cumulative_scan ?axis op x =
    let orig_shape = shape x in
    match axis with
    | Some axis -> associative_scan ~axis op x
    | None ->
        let flat = reshape [| array_prod orig_shape |] x in
        let scanned = associative_scan ~axis:0 op flat in
        if Array.length orig_shape = 0 then reshape [||] scanned
        else reshape orig_shape scanned

  let cumsum ?axis x =
    let@ _ = span ~op:"cumsum" () in
    cumulative_scan ?axis `Sum x

  let cumprod ?axis x =
    let@ _ = span ~op:"cumprod" () in
    cumulative_scan ?axis `Prod x

  let cummax ?axis x =
    let@ _ = span ~op:"cummax" () in
    cumulative_scan ?axis `Max x

  let cummin ?axis x =
    let@ _ = span ~op:"cummin" () in
    cumulative_scan ?axis `Min x

  let mean ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"mean" () in
    let dt = B.dtype x in
    let s = sum ?axes ~keepdims x in
    let n = reduction_element_count (shape x) ?axes () in
    let divisor = broadcast_to (shape s) (scalar (B.context x) dt
      (Dtype.of_float dt (float_of_int (Stdlib.max 1 n)))) in
    div ?out s divisor

  let var ?out ?axes ?(keepdims = false) ?(ddof = 0) x =
    let@ _ = span ~op:"var" () in
    let dt = B.dtype x in
    let mean_x = mean ?axes ~keepdims:true x in
    let sum_sq = sum ?axes ~keepdims (square (sub x mean_x)) in
    let n = reduction_element_count (shape x) ?axes () in
    let n_corr = float_of_int (Stdlib.max 0 (n - ddof)) in
    let divisor = broadcast_to (shape sum_sq) (scalar (B.context x) dt
      (Dtype.of_float dt n_corr)) in
    div ?out sum_sq divisor

  let std ?out ?axes ?(keepdims = false) ?(ddof = 0) x =
    let@ _ = span ~op:"std" () in
    sqrt ?out (var ?axes ~keepdims ~ddof x)

  let all ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"all" () in
    let bool_t = cmpne x (full_like x (Dtype.zero (dtype x))) in
    prod ?out ?axes ~keepdims bool_t

  let any ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"any" () in
    let bool_t = cmpne x (full_like x (Dtype.zero (dtype x))) in
    max ?out ?axes ~keepdims bool_t

  let array_equal x y =
    let@ _ = span ~op:"array_equal" () in
    let can_broadcast =
      try ignore (Shape.broadcast (shape x) (shape y)); true
      with _ -> false
    in
    if not can_broadcast then zeros (B.context x) Dtype.bool [||]
    else all (equal x y)

  (* ───── Shape Manipulation ───── *)

  let pad padding_config fill_value x =
    let@ _ = span ~op:"pad" () in
    Array.iter
      (fun (before, after) ->
        if before < 0 || after < 0 then
          invalid_arg "pad: padding values, negative values not allowed, use shrink or slice to remove elements")
      padding_config;
    B.pad x padding_config fill_value

  let shrink shrink_args x =
    let@ _ = span ~op:"shrink" () in
    B.shrink x shrink_args

  let flatten ?(start_dim = 0) ?(end_dim = -1) x =
    let@ _ = span ~op:"flatten" () in
    let sh = shape x in
    let r = Array.length sh in
    let s = if start_dim < 0 then start_dim + r else start_dim in
    let e = if end_dim < 0 then end_dim + r else end_dim in
    if not ((s >= 0 && s < r && e >= 0 && e < r)
            || (r = 0 && (s = 0 || start_dim = 0) && (e = -1 || end_dim = -1)))
    then
      err "flatten" "start_dim %d or end_dim %d, out of bounds for rank %d" start_dim end_dim r;
    if s > e then
      invalid_arg "flatten: dimensions, start_dim must be <= end_dim";
    if r = 0 then reshape [| 1 |] x
    else if s = 0 && e = r - 1 then reshape [| array_prod sh |] x
    else
      let pre = Array.to_list (Array.sub sh 0 s) in
      let mid = array_prod (Array.sub sh s (e - s + 1)) in
      let post = Array.to_list (Array.sub sh (e + 1) (r - (e + 1))) in
      reshape (Array.of_list (pre @ [ mid ] @ post)) x

  let unflatten dim sizes x =
    let@ _ = span ~op:"unflatten" () in
    let dim = resolve_single_axis x dim in
    let current_shape = shape x in
    let dim_size = current_shape.(dim) in
    let sizes = Array.copy sizes in
    let neg_one_count =
      Array.fold_left (fun acc s -> if s = -1 then acc + 1 else acc) 0 sizes
    in
    if neg_one_count > 1 then
      invalid_arg "unflatten: sizes, can only specify one unknown dimension (using -1)";
    if neg_one_count = 1 then begin
      let known_product =
        Array.fold_left (fun acc s -> if s = -1 then acc else acc * s) 1 sizes
      in
      if known_product = 0 || dim_size mod known_product <> 0 then
        err "unflatten" "cannot infer dimension from total size %d to known product %d, %d not divisible by %d, ensure total size is divisible by product of known dimensions"
          dim_size known_product dim_size known_product;
      let inferred = dim_size / known_product in
      Array.iteri (fun i s -> if s = -1 then sizes.(i) <- inferred) sizes
    end;
    let sizes_product = Array.fold_left ( * ) 1 sizes in
    if sizes_product <> dim_size then
      err "unflatten" "sizes, product %d does not match dimension size %d"
        sizes_product dim_size;
    reshape
      (Array.concat [
         Array.sub current_shape 0 dim;
         sizes;
         Array.sub current_shape (dim + 1)
           (Array.length current_shape - dim - 1);
       ])
      x

  let ravel x =
    let@ _ = span ~op:"ravel" () in
    flatten x

  let squeeze ?axes x =
    let@ _ = span ~op:"squeeze" () in
    let sh = shape x in
    let r = Array.length sh in
    let reshape_or_id new_sh =
      if Array.length new_sh = 0 && r > 0 then reshape [||] x
      else if Array.length new_sh = 0 then x
      else reshape new_sh x
    in
    match axes with
    | None ->
        reshape_or_id
          (Array.of_list (List.filter (( <> ) 1) (Array.to_list sh)))
    | Some axes_list ->
        if r = 0 then x
        else
          let normalized =
            List.map (fun ax -> if ax < 0 then ax + r else ax) axes_list
          in
          let seen = Array.make r false in
          List.iter
            (fun ax ->
              if ax < 0 || ax >= r then
                err "squeeze" "axis %d out of bounds for %dD tensor" ax r;
              if seen.(ax) then
                err "squeeze" "axis %d, duplicate axis" ax;
              seen.(ax) <- true)
            normalized;
          List.iter
            (fun ax ->
              if sh.(ax) <> 1 then
                err "squeeze" "cannot remove dimension at axis %d (size %d), size %d≠1" ax sh.(ax) sh.(ax))
            normalized;
          let axes_set =
            List.fold_left (fun s ax -> IntSet.add ax s) IntSet.empty normalized
          in
          reshape_or_id
            (Array.of_list
               (List.filteri (fun i _ -> not (IntSet.mem i axes_set))
                  (Array.to_list sh)))

  let unsqueeze ?axes x =
    let@ _ = span ~op:"unsqueeze" () in
    let sh = shape x in
    let r = Array.length sh in
    let axes_list = match axes with
      | None ->
          invalid_arg "unsqueeze: axes must be specified"
      | Some lst -> lst
    in
    if List.length axes_list = 0 then x
    else
      let output_rank = r + List.length axes_list in
      let normalized =
        List.map (fun ax -> if ax < 0 then ax + output_rank else ax) axes_list
      in
      let seen = Array.make output_rank false in
      List.iter
        (fun ax ->
          if ax < 0 || ax >= output_rank then
            err "unsqueeze" "axis %d, out of bounds for output rank %d, valid range is [%d, %d)"
              ax output_rank (-output_rank) output_rank;
          if seen.(ax) then
            err "unsqueeze" "axis %d, duplicate axis" ax;
          seen.(ax) <- true)
        normalized;
      let axes_set =
        List.fold_left (fun s ax -> IntSet.add ax s) IntSet.empty normalized
      in
      let new_shape = ref [] in
      let input_idx = ref 0 in
      for output_idx = 0 to output_rank - 1 do
        if IntSet.mem output_idx axes_set then
          new_shape := 1 :: !new_shape
        else if !input_idx < r then begin
          new_shape := sh.(!input_idx) :: !new_shape;
          incr input_idx
        end
      done;
      reshape (Array.of_list (List.rev !new_shape)) x

  let squeeze_axis axis x = squeeze ~axes:[ axis ] x
  let unsqueeze_axis axis x = unsqueeze ~axes:[ axis ] x
  let expand_dims axes x = unsqueeze ~axes x

  let transpose ?axes x =
    let@ _ = span ~op:"transpose" () in
    let r = ndim x in
    let resolved = match axes with
      | None -> Array.init r (fun i -> r - 1 - i)
      | Some ax_list ->
          if List.length ax_list <> r then
            err "transpose" "axes (length %d), expected rank %d, got %d, provide exactly one axis per dimension"
              (List.length ax_list) r (List.length ax_list);
          let seen = Array.make r false in
          List.iter
            (fun ax_val ->
              let ax = if ax_val < 0 then ax_val + r else ax_val in
              if ax < 0 || ax >= r then
                err "transpose" "axis %d out of bounds for %dD tensor" ax_val r;
              if seen.(ax) then
                err "transpose" "axis %d, repeated" ax_val;
              seen.(ax) <- true)
            ax_list;
          if not (Array.for_all Fun.id seen) then
            invalid_arg "transpose: axes do not form a permutation";
          Array.of_list
            (List.map (fun v -> if v < 0 then v + r else v) ax_list)
    in
    B.permute x resolved

  let flip ?axes x =
    let@ _ = span ~op:"flip" () in
    let r = ndim x in
    let flip_bools = Array.make r false in
    (match axes with
     | None -> Array.fill flip_bools 0 r true
     | Some ax_list ->
         List.iter
           (fun ax_val ->
             let ax = if ax_val < 0 then ax_val + r else ax_val in
             if ax < 0 || ax >= r then
               err "flip" "axis %d out of bounds for %dD tensor" ax_val r;
             flip_bools.(ax) <- true)
           ax_list);
    B.flip x flip_bools

  let moveaxis src dst x =
    let@ _ = span ~op:"moveaxis" () in
    let r = ndim x in
    let s = if src < 0 then src + r else src in
    let d = if dst < 0 then dst + r else dst in
    if s < 0 || s >= r || d < 0 || d >= r then
      err "moveaxis" "source %d or destination %d, out of bounds for shape %s"
        src dst (Shape.to_string (shape x));
    if s = d then x
    else
      let axes = Array.to_list (Array.init r Fun.id) in
      let without = List.filter (( <> ) s) axes in
      let rec insert_at idx item = function
        | [] -> [ item ]
        | hd :: tl ->
            if idx = 0 then item :: hd :: tl
            else hd :: insert_at (idx - 1) item tl
      in
      B.permute x (Array.of_list (insert_at d s without))

  let swapaxes axis1 axis2 x =
    let@ _ = span ~op:"swapaxes" () in
    let r = ndim x in
    let a1 = if axis1 < 0 then axis1 + r else axis1 in
    let a2 = if axis2 < 0 then axis2 + r else axis2 in
    if a1 < 0 || a1 >= r || a2 < 0 || a2 >= r then
      err "swapaxes" "axes (%d, %d), out of bounds for shape %s"
        axis1 axis2 (Shape.to_string (shape x));
    if a1 = a2 then x
    else
      let axes = Array.init r Fun.id in
      axes.(a1) <- a2;
      axes.(a2) <- a1;
      B.permute x axes

  let cat_tensors ~axis tensors =
    match tensors with
    | [] ->
        invalid_arg "concatenate: tensor list cannot be empty, provide at least one tensor"
    | first :: _ ->
        let first_shape = shape first in
        let ndim = Array.length first_shape in
        let axis = if axis < 0 then axis + ndim else axis in
        let out_shape = Array.copy first_shape in
        out_shape.(axis) <-
          List.fold_left (fun acc t -> acc + (shape t).(axis)) 0 tensors;
        let out = empty (B.context first) (B.dtype first) out_shape in
        B.cat ~out tensors ~axis;
        out

  let roll ?axis shift x =
    let@ _ = span ~op:"roll" () in
    let original_shape = shape x in
    let x, ax_idx = match axis with
      | None -> (flatten x, 0)
      | Some a ->
          let r = ndim x in
          let norm = if a < 0 then a + r else a in
          if norm < 0 || norm >= r then
            err "roll" "axis %d out of bounds for %dD tensor" a r;
          (x, norm)
    in
    let sh = shape x in
    let r = ndim x in
    if r = 0 then x
    else
      let dim_size = sh.(ax_idx) in
      if dim_size = 0 then x
      else
        let s = shift mod dim_size in
        let actual = if s < 0 then s + dim_size else s in
        if actual = 0 then
          if axis = None then reshape (shape x) x else x
        else
          let ranges_p1 =
            Array.mapi
              (fun i d ->
                if i = ax_idx then (dim_size - actual, d) else (0, d))
              sh
          in
          let ranges_p2 =
            Array.mapi
              (fun i d ->
                if i = ax_idx then (0, dim_size - actual) else (0, d))
              sh
          in
          let rolled =
            cat_tensors ~axis:ax_idx
              [ shrink ranges_p1 x; shrink ranges_p2 x ]
          in
          if axis = None then reshape original_shape rolled else rolled

  let tile reps x =
    let@ _ = span ~op:"tile" () in
    let t_shape = shape x in
    let t_ndim = ndim x in
    let reps_len = Array.length reps in
    if reps_len < t_ndim then
      invalid_arg "tile: reps length must be >= tensor rank";
    let x_promoted, promoted_shape =
      if reps_len > t_ndim then
        let new_shape = Array.make reps_len 1 in
        Array.blit t_shape 0 new_shape (reps_len - t_ndim) t_ndim;
        (reshape new_shape x, new_shape)
      else (x, t_shape)
    in
    Array.iteri
      (fun i r ->
        if r < 0 then
          err "tile" "reps[%d], negative (%d<0), use positive integers (or 0 for empty result)" i r)
      reps;
    if Array.for_all (( = ) 1) reps then B.copy x_promoted
    else if Array.exists (( = ) 0) reps
            || Array.exists (( = ) 0) promoted_shape
    then
      empty (B.context x) (dtype x)
        (Array.mapi (fun i s -> s * reps.(i)) promoted_shape)
    else
      let rec tile_axis curr axis =
        if axis >= reps_len then curr
        else if reps.(axis) = 1 then tile_axis curr (axis + 1)
        else
          tile_axis
            (cat_tensors ~axis (List.init reps.(axis) (fun _ -> curr)))
            (axis + 1)
      in
      tile_axis x_promoted 0

  let repeat ?axis count x =
    let@ _ = span ~op:"repeat" () in
    if count < 0 then
      err "repeat" "count must be >= 0, got %d" count;
    let x, ax_idx = match axis with
      | None -> (flatten x, 0)
      | Some a ->
          let r = ndim x in
          let norm = if a < 0 then a + r else a in
          if norm < 0 || norm >= r then
            err "repeat" "axis %d out of bounds for %dD tensor" a r;
          (x, norm)
    in
    let t_shape = shape x in
    let t_ndim = ndim x in
    if count = 0 then begin
      let s = Array.copy t_shape in
      if t_ndim > 0 then s.(ax_idx) <- 0;
      empty (B.context x) (dtype x) (if axis = None then [| 0 |] else s)
    end
    else if count = 1 then B.copy x
    else if t_ndim = 0 then
      let repeated = expand [| count |] (reshape [| 1 |] x) in
      if axis = None then repeated else reshape (shape x) repeated
    else
      let axis_size = t_shape.(ax_idx) in
      let slices = ref [] in
      for i = axis_size - 1 downto 0 do
        let slice =
          Array.init t_ndim (fun dim ->
              if dim = ax_idx then (i, i + 1) else (0, t_shape.(dim)))
        in
        let sv = B.shrink x slice in
        for _ = 1 to count do slices := sv :: !slices done
      done;
      cat_tensors ~axis:ax_idx !slices

  (* ───── Concatenation and Stacking ───── *)

  let check_dtypes_match ~op ts =
    let first_dtype = dtype (List.hd ts) in
    List.iter
      (fun x ->
        let d = dtype x in
        if not (Dtype.equal first_dtype d) then
          err op "expected dtype %s, got %s" (Dtype.to_string first_dtype) (Dtype.to_string d))
      (List.tl ts)

  let concatenate ?axis ts =
    let@ _ = span ~op:"concatenate" () in
    match ts with
    | [] ->
        invalid_arg "concatenate: tensor list cannot be empty, provide at least one tensor"
    | [ x ] -> copy x
    | _ ->
        check_dtypes_match ~op:"concatenate" ts;
        match axis with
        | None ->
            cat_tensors ~axis:0 (List.map flatten ts)
        | Some a ->
            let first = List.hd ts in
            let first_ndim = ndim first in
            let axis = resolve_single_axis ~ndim_opt:first_ndim first a in
            if not (List.for_all (fun x -> ndim x = first_ndim) ts) then
              invalid_arg "concatenate: arrays must have same number of dimensions";
            let first_shape = shape first in
            List.iter
              (fun x ->
                let s = shape x in
                Array.iteri
                  (fun i d ->
                    if i <> axis && d <> first_shape.(i) then
                      err "concatenate" "dimension %d, size %d≠%d" i d first_shape.(i))
                  s)
              (List.tl ts);
            cat_tensors ~axis ts

  let stack ?axis ts =
    let@ _ = span ~op:"stack" () in
    match ts with
    | [] -> invalid_arg "stack: tensor list cannot be empty"
    | _ ->
        let first_ndim = Array.length (shape (List.hd ts)) in
        let axis = match axis with
          | None -> 0
          | Some a ->
              let a = if a < 0 then a + first_ndim + 1 else a in
              if a < 0 || a > first_ndim then
                err "stack" "axis %d out of bounds for %dD tensor" a first_ndim;
              a
        in
        concatenate ~axis (List.map (fun x -> unsqueeze ~axes:[ axis ] x) ts)

  let ensure_ndim n x =
    let s = shape x in
    let nd = Array.length s in
    if nd >= n then x
    else
      let new_shape = Array.make n 1 in
      Array.blit s 0 new_shape 0 nd;
      reshape new_shape x

  let vstack ts =
    let@ _ = span ~op:"vstack" () in
    match ts with
    | [] -> invalid_arg "vstack: tensor list cannot be empty"
    | _ ->
        concatenate ~axis:0
          (List.map
             (fun x ->
               if ndim x = 0 then reshape [| 1; 1 |] x
               else if ndim x = 1 then reshape [| 1; numel x |] x
               else x)
             ts)

  let hstack ts =
    let@ _ = span ~op:"hstack" () in
    match ts with
    | [] -> invalid_arg "hstack: tensor list cannot be empty"
    | _ ->
        if List.for_all (fun x -> ndim x <= 1) ts then
          concatenate ~axis:0
            (List.map
               (fun x -> if ndim x = 0 then reshape [| 1 |] x else x)
               ts)
        else
          concatenate ~axis:1
            (List.map
               (fun x ->
                 if ndim x = 0 then reshape [| 1; 1 |] x
                 else if ndim x = 1 then reshape [| numel x; 1 |] x
                 else x)
               ts)

  let dstack ts =
    let@ _ = span ~op:"dstack" () in
    match ts with
    | [] -> invalid_arg "dstack: tensor list cannot be empty"
    | _ ->
        concatenate ~axis:2
          (List.map
             (fun x ->
               let s = shape x in
               let nd = Array.length s in
               if nd = 0 then reshape [| 1; 1; 1 |] x
               else if nd = 1 then reshape [| 1; s.(0); 1 |] x
               else if nd = 2 then reshape [| s.(0); s.(1); 1 |] x
               else x)
             ts)

  let broadcast_arrays ts =
    let@ _ = span ~op:"broadcast_arrays" () in
    match ts with
    | [] -> []
    | [ x ] -> [ x ]
    | _ ->
        let target =
          List.fold_left
            (fun acc x -> Shape.broadcast acc (shape x))
            (shape (List.hd ts)) (List.tl ts)
        in
        List.map (fun x -> broadcast_to target x) ts

  (* ───── Array Creation ───── *)

  let eye ctx ?m ?k dtype n =
    let@ _ = span ~op:"eye" () in
    let rows = match m with Some v -> v | None -> n in
    let cols = n in
    let k_val = match k with Some v -> v | None -> 0 in
    if rows <= 0 || cols <= 0 || k_val >= cols || k_val <= -rows then
      zeros ctx dtype [| rows; cols |]
    else
      let arr = Array.make (rows * cols) (Dtype.zero dtype) in
      let one = Dtype.one dtype in
      for i = 0 to Stdlib.min rows cols - 1 do
        let col = i + k_val in
        if col >= 0 && col < cols then arr.((i * cols) + col) <- one
      done;
      create ctx dtype [| rows; cols |] arr

  let identity ctx dtype n = eye ctx ~m:n ~k:0 dtype n

  let diag ?(k = 0) v =
    let v_shape = shape v in
    let v_ndim = Array.length v_shape in
    if v_ndim = 1 then
      let n = v_shape.(0) in
      let size = n + Int.abs k in
      let v_arr = to_array v in
      init (B.context v) (dtype v) [| size; size |] (fun indices ->
          let row = indices.(0) in
          let col = indices.(1) in
          let diag_idx =
            if k >= 0 then
              if col = row + k && row >= 0 && row < n then row else -1
            else if row = col - k && col >= 0 && col < n then col
            else -1
          in
          if diag_idx >= 0 && diag_idx < n then v_arr.(diag_idx)
          else Dtype.zero (dtype v))
    else if v_ndim >= 2 then
      let rows = v_shape.(0) in
      let cols = v_shape.(1) in
      let diag_len =
        Stdlib.max 0
          (if k >= 0 then Int.min rows (cols - k)
           else Int.min (rows + k) cols)
      in
      if diag_len = 0 then empty (B.context v) (dtype v) [| 0 |]
      else
        let v_arr = to_array v in
        init (B.context v) (dtype v) [| diag_len |] (fun indices ->
            let i = indices.(0) in
            let row = if k >= 0 then i else i - k in
            let col = if k >= 0 then i + k else i in
            v_arr.((row * cols) + col))
    else
      err "diag" "input, expected 1D or 2D array, got %dD" v_ndim

  let arange (type a b) ctx (dtype : (a, b) Dtype.t) start stop step =
    let@ _ = span ~op:"arange" () in
    if start >= stop && step > 0 then
      err "arange" "range [%d, %d), empty with step=%d, ensure start < stop for positive step, or start > stop for negative step"
        start stop step;
    if step = 0 then
      invalid_arg "arange: step cannot be zero";
    let num_elements =
      if step > 0 then
        if start >= stop then 0 else (stop - start + step - 1) / step
      else if start <= stop then 0
      else (start - stop + -step - 1) / -step
    in
    if num_elements <= 0 then empty ctx dtype [| 0 |]
    else
      let float_at i =
        float_of_int start +. (float_of_int i *. float_of_int step)
      in
      let int_at i = start + (i * step) in
      let f_init idx_arr : a =
        let i = idx_arr.(0) in
        match dtype with
        | Dtype.Float16 -> float_at i
        | Dtype.Float32 -> float_at i
        | Dtype.Float64 -> float_at i
        | Dtype.BFloat16 -> float_at i
        | Dtype.Float8_e4m3 -> float_at i
        | Dtype.Float8_e5m2 -> float_at i
        | Dtype.Int8 -> int_at i
        | Dtype.UInt8 -> int_at i
        | Dtype.Int16 -> int_at i
        | Dtype.UInt16 -> int_at i
        | Dtype.Int4 -> int_at i
        | Dtype.UInt4 -> int_at i
        | Dtype.Bool -> i <> 0
        | Dtype.Int32 ->
            Int32.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.UInt32 ->
            Int32.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.Int64 ->
            Int64.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.UInt64 ->
            Int64.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.Complex64 -> { Complex.re = float_at i; im = 0. }
        | Dtype.Complex128 -> { Complex.re = float_at i; im = 0. }
      in
      init ctx dtype [| num_elements |] f_init

  let arange_f ctx dtype start_f stop_f step_f =
    let@ _ = span ~op:"arange_f" () in
    if step_f = 0. then
      invalid_arg "arange_f: step cannot be zero";
    let num_exact_steps = (stop_f -. start_f) /. step_f in
    let eps = 1e-9 in
    let num_elements =
      if (step_f > 0. && stop_f <= start_f +. (eps *. Float.abs step_f))
         || (step_f < 0. && stop_f >= start_f +. (eps *. Float.abs step_f))
         || (Float.abs num_exact_steps < eps && num_exact_steps <= 0.)
      then 0
      else
        let corrected =
          num_exact_steps -. Float.copy_sign eps num_exact_steps
        in
        int_of_float (Float.floor corrected +. 1.)
    in
    let n = Stdlib.max 0 num_elements in
    if n <= 0 then empty ctx dtype [| 0 |]
    else
      init ctx dtype [| n |] (fun idx ->
          start_f +. (float_of_int idx.(0) *. step_f))

  let linspace ctx dtype ?(endpoint = true) start_f stop_f count =
    let@ _ = span ~op:"linspace" () in
    if count < 0 then
      err "linspace" "count %d, negative count, use count >= 0" count;
    if count = 0 then empty ctx dtype [| 0 |]
    else if count = 1 then full ctx dtype [| 1 |] (Dtype.of_float dtype start_f)
    else
      let div_factor = float_of_int (if endpoint then count - 1 else count) in
      let step = (stop_f -. start_f) /. div_factor in
      init ctx dtype [| count |] (fun idx ->
          Dtype.of_float dtype (start_f +. (float_of_int idx.(0) *. step)))

  let logspace ctx dtype ?(endpoint = true) ?(base = 10.0) start_exp stop_exp
      count =
    let@ _ = span ~op:"logspace" () in
    if count < 0 then
      err "logspace" "count must be >= 0, got %d" count;
    if count = 0 then empty ctx dtype [| 0 |]
    else
      let exponents = linspace ctx dtype ~endpoint start_exp stop_exp count in
      if base = Float.exp 1.0 then exp exponents
      else if base = 2.0 then exp2 exponents
      else
        let log2_base = Stdlib.log base /. Stdlib.log 2.0 in
        let log2_base_t =
          broadcast_to (shape exponents) (scalar ctx dtype log2_base)
        in
        exp2 (mul exponents log2_base_t)

  let geomspace ctx dtype ?(endpoint = true) start_f stop_f count =
    if start_f <= 0. || stop_f <= 0. then
      err "geomspace" "%s, must be positive (>0), geomspace requires positive values for logarithmic spacing"
        (if start_f <= 0. then Printf.sprintf "start %g" start_f
         else Printf.sprintf "stop %g" stop_f);
    if count < 0 then
      err "geomspace" "count must be >= 0, got %d" count;
    if count = 0 then empty ctx dtype [| 0 |]
    else if count = 1 then full ctx dtype [| 1 |] start_f
    else
      exp (linspace ctx dtype ~endpoint (Stdlib.log start_f) (Stdlib.log stop_f)
             count)

  let meshgrid ?(indexing = `xy) x y =
    let x_shape = shape x in
    let y_shape = shape y in
    if Array.length x_shape <> 1 then invalid_arg "meshgrid: x must be 1D";
    if Array.length y_shape <> 1 then invalid_arg "meshgrid: y must be 1D";
    let nx = x_shape.(0) in
    let ny = y_shape.(0) in
    match indexing with
    | `xy ->
        ( broadcast_to [| ny; nx |] (reshape [| 1; nx |] x),
          broadcast_to [| ny; nx |] (reshape [| ny; 1 |] y) )
    | `ij ->
        ( broadcast_to [| nx; ny |] (reshape [| nx; 1 |] x),
          broadcast_to [| nx; ny |] (reshape [| 1; ny |] y) )

  (* Triangular mask: tril uses (>=), triu uses (<=) *)
  let triangular_mask ~op ~cmp ?k x =
    let@ _ = span ~op () in
    let k_val = match k with Some v -> v | None -> 0 in
    let sh = shape x in
    let nd = Array.length sh in
    if nd < 2 then
      err op "input requires at least 2D tensor";
    let rows = sh.(nd - 2) in
    let cols = sh.(nd - 1) in
    let row_idx = reshape [| rows; 1 |] (arange (B.context x) int32 0 rows 1) in
    let col_idx = reshape [| 1; cols |] (arange (B.context x) int32 0 cols 1) in
    let k_offset =
      sub col_idx (scalar (B.context x) int32 (Int32.of_int k_val))
    in
    let mask = cmp row_idx k_offset in
    let mask =
      if nd > 2 then
        broadcast_to
          (Array.concat [ Array.sub sh 0 (nd - 2); [| rows; cols |] ])
          mask
      else mask
    in
    where mask x (zeros_like x)

  let tril ?k x = triangular_mask ~op:"tril" ~cmp:greater_equal ?k x
  let triu ?k x = triangular_mask ~op:"triu" ~cmp:less_equal ?k x

  (* ───── Take Operations ───── *)

  let apply_index_mode ~mode ~n ctx indices =
    match mode with
    | `raise -> indices
    | `wrap ->
        mod_ indices (scalar (B.context indices) Int32 (Int32.of_int n))
    | `clip ->
        let s = shape indices in
        minimum
          (maximum indices (zeros ctx Int32 s))
          (full ctx Int32 s (Int32.of_int (n - 1)))

  let take ?axis ?(mode = `raise) indices t =
    let@ _ = span ~op:"take" () in
    let ctx = B.context t in
    match axis with
    | None ->
        let t_flat = reshape [| numel t |] t in
        let idx = apply_index_mode ~mode ~n:(numel t) ctx indices in
        let out = empty ctx (dtype t_flat) (shape idx) in
        B.gather ~out t_flat idx ~axis:0;
        out
    | Some axis ->
        let t_shape = shape t in
        let axis = resolve_single_axis t axis in
        let idx = apply_index_mode ~mode ~n:t_shape.(axis) ctx indices in
        let n_idx = numel idx in
        (* Reshape indices for broadcasting: [1,...,1,n_idx,1,...,1] *)
        let expanded_shape =
          Array.init (Array.length t_shape) (fun i ->
              if i = axis then n_idx else 1)
        in
        let broadcast_shape = Array.copy t_shape in
        broadcast_shape.(axis) <- n_idx;
        let idx_broadcast =
          broadcast_to broadcast_shape (reshape expanded_shape idx)
        in
        let out = empty ctx (dtype t) (shape idx_broadcast) in
        B.gather ~out t idx_broadcast ~axis;
        let out_shape = Array.copy t_shape in
        out_shape.(axis) <- n_idx;
        reshape out_shape out

  let take_along_axis ~axis indices t =
    let@ _ = span ~op:"take_along_axis" () in
    let axis = resolve_single_axis t axis in
    let t_shape = shape t in
    let idx_shape = shape indices in
    if Array.length t_shape <> Array.length idx_shape then
      err "take_along_axis" "cannot reshape %s to %s" (Shape.to_string idx_shape) (Shape.to_string t_shape);
    Array.iteri
      (fun i dim ->
        if i <> axis && dim <> idx_shape.(i) then
          err "take_along_axis" "shape, dimension %d: indices has %d but tensor has %d"
            i idx_shape.(i) dim)
      t_shape;
    let out = empty (B.context t) (dtype t) idx_shape in
    B.gather ~out t indices ~axis;
    out

  (* ───── Indexing and Slicing ───── *)

  let normalize_index dim_size idx = if idx < 0 then dim_size + idx else idx

  let normalize_and_check_index ~op dim_size idx =
    let idx' = if idx < 0 then dim_size + idx else idx in
    if idx' < 0 || idx' >= dim_size then
      err op "index %d out of bounds [0, %d)" idx dim_size;
    idx'

  type dim_op =
    | View of { start : int; stop : int; step : int; dim_len : int }
    | Squeeze of { idx : int }
    | Gather of int array
    | New_axis

  let normalize_slice_spec dim_size = function
    | I idx ->
        Squeeze { idx = normalize_and_check_index ~op:"slice" dim_size idx }
    | A -> View { start = 0; stop = dim_size; step = 1; dim_len = dim_size }
    | R (start, stop) ->
        let s = Int.max 0 (Int.min (normalize_index dim_size start) dim_size) in
        let e = Int.max 0 (Int.min (normalize_index dim_size stop) dim_size) in
        View { start = s; stop = e; step = 1; dim_len = Int.max 0 (e - s) }
    | Rs (start, stop, step) ->
        if step = 0 then
          invalid_arg "slice: step cannot be zero, use positive step for forward slicing or negative for reverse";
        let s = normalize_index dim_size start in
        let e = normalize_index dim_size stop in
        let len, actual_stop =
          if step > 0 then
            let s = Int.max 0 (Int.min s dim_size) in
            let e = Int.max 0 (Int.min e dim_size) in
            (if s >= e then 0 else ((e - 1 - s) / step) + 1), e
          else
            let s = Int.min (dim_size - 1) (Int.max (-1) s) in
            let e = Int.min (dim_size - 1) (Int.max (-1) e) in
            (if s <= e then 0 else ((s - e - 1) / -step) + 1), e
        in
        View { start = s; stop = actual_stop; step; dim_len = len }
    | L indices ->
        Gather
          (Array.map
             (normalize_and_check_index ~op:"slice" dim_size)
             (Array.of_list indices))
    | N -> New_axis
    | M _ -> failwith "Mask slicing not supported in slice_internal"

  let slice_internal specs x =
    let@ _ = span ~op:"slice" () in
    let input_shape = shape x in
    let ndim_in = Array.length input_shape in
    (* Parse specs, then pad with A for unspecified trailing dimensions *)
    let ops, consumed =
      List.fold_left
        (fun (acc, dim) spec ->
          match spec with
          | N -> (New_axis :: acc, dim)
          | _ ->
              if dim >= ndim_in then
                invalid_arg "slice: too many indices";
              (normalize_slice_spec input_shape.(dim) spec :: acc, dim + 1))
        ([], 0) specs
    in
    let rec pad_trailing acc dim =
      if dim >= ndim_in then List.rev acc
      else pad_trailing (normalize_slice_spec input_shape.(dim) A :: acc) (dim+1)
    in
    let ops = pad_trailing ops consumed in
    let gather_axis axis indices t =
      let idx_t =
        init (B.context t) Dtype.int32 [| Array.length indices |]
          (fun i -> Int32.of_int indices.(i.(0)))
      in
      take ~axis idx_t t
    in
    let shrink_axis axis start stop t =
      if start < stop then
        B.shrink t
          (Array.mapi
             (fun i dim -> if i = axis then (start, stop) else (0, dim))
             (shape t))
      else
        take ~axis (empty (B.context t) Dtype.int32 [| 0 |]) t
    in
    let rec apply current axis sq_axes = function
      | [] -> (current, sq_axes)
      | New_axis :: rest ->
          apply (unsqueeze ~axes:[ axis ] current) (axis + 1) sq_axes rest
      | Squeeze { idx } :: rest ->
          apply (shrink_axis axis idx (idx + 1) current) (axis + 1)
            (axis :: sq_axes) rest
      | Gather indices :: rest ->
          apply (gather_axis axis indices current) (axis + 1) sq_axes rest
      | View { start; step; dim_len; _ } :: rest ->
          let current' =
            if step = 1 then shrink_axis axis start (start + dim_len) current
            else if step = -1 then
              if dim_len = 0 then shrink_axis axis 0 0 current
              else
                let sliced = shrink_axis axis (start - dim_len + 1) (start + 1) current in
                let fb = Array.make (ndim sliced) false in
                fb.(axis) <- true;
                B.flip sliced fb
            else
              gather_axis axis
                (Array.init dim_len (fun i -> start + (i * step)))
                current
          in
          apply current' (axis + 1) sq_axes rest
    in
    let result, sq_axes = apply x 0 [] ops in
    match List.sort_uniq compare sq_axes with
    | [] -> result
    | axes -> squeeze ~axes result

  let set_slice_internal specs x y =
    let@ _ = span ~op:"set_slice" () in
    let x_shape = shape x in
    let nd = Array.length x_shape in
    let full_specs =
      if List.length specs < nd then
        specs @ List.init (nd - List.length specs) (fun _ -> A)
      else specs
    in
    (* Fast path: contiguous view — just assign *)
    let is_view_compatible =
      List.for_all
        (function
          | L _ | M _ -> false | Rs (_, _, s) -> Int.abs s = 1 | _ -> true)
        full_specs
    in
    if is_view_compatible then
      let target = slice_internal full_specs x in
      B.assign target (broadcast_to (shape target) y)
    else begin
      (* Slow path: scatter for fancy indexing *)
      let strides = Array.make nd 1 in
      for i = nd - 2 downto 0 do
        strides.(i) <- strides.(i + 1) * x_shape.(i + 1)
      done;
      let ctx = B.context x in
      let dims_info =
        List.mapi
          (fun i spec ->
            match normalize_slice_spec x_shape.(i) spec with
            | Squeeze { idx } ->
                (true, scalar ctx Dtype.int32 (Int32.of_int idx))
            | View { start; stop; step; _ } ->
                (false, arange ctx Dtype.int32 start stop step)
            | Gather indices ->
                (false,
                 init ctx Dtype.int32 [| Array.length indices |]
                   (fun k -> Int32.of_int indices.(k.(0))))
            | New_axis ->
                failwith "New_axis not supported in set_slice")
          full_specs
      in
      let target_shape =
        Array.of_list
          (List.filter_map
             (fun (sq, t) -> if sq then None else Some (numel t))
             dims_info)
      in
      let target_rank = Array.length target_shape in
      let flat_idx = ref (scalar ctx Dtype.int32 0l) in
      let tdim = ref 0 in
      List.iteri
        (fun i (squeezed, idx_t) ->
          let stride = Int32.of_int strides.(i) in
          let weighted =
            if stride = 1l then idx_t
            else mul idx_t (scalar ctx Dtype.int32 stride)
          in
          if squeezed then flat_idx := add !flat_idx weighted
          else begin
            let rs = Array.make target_rank 1 in
            rs.(!tdim) <- numel idx_t;
            flat_idx := add !flat_idx (reshape rs weighted);
            incr tdim
          end)
        dims_info;
      let x_flat = reshape [| numel x |] x in
      let y_flat = reshape [| numel (broadcast_to target_shape y) |]
          (broadcast_to target_shape y) in
      let result =
        B.scatter ~mode:`Set ~unique_indices:false x_flat
          ~indices:(reshape [| numel !flat_idx |] !flat_idx)
          ~updates:y_flat ~axis:0
      in
      B.assign x (reshape x_shape result)
    end

  let get indices x =
    let@ _ = span ~op:"get" () in
    let x_shape = shape x in
    let checked =
      List.mapi
        (fun dim idx ->
          if dim >= Array.length x_shape then
            err "get" "indices, too many for shape %s" (Shape.to_string x_shape);
          let idx' = normalize_index x_shape.(dim) idx in
          if idx' < 0 || idx' >= x_shape.(dim) then
            err "get" "index [%s] out of bounds for shape %s, index %d at dim %d: %d not in [0, %d)"
              (String.concat "," (List.map string_of_int indices))
              (Shape.to_string x_shape)
              dim dim idx' x_shape.(dim);
          idx')
        indices
    in
    slice_internal (List.map (fun i -> I i) checked) x

  let set indices x value =
    let@ _ = span ~op:"set" () in
    let x_shape = shape x in
    let checked =
      List.mapi
        (fun dim idx ->
          if dim >= Array.length x_shape then
            err "set" "indices, too many for shape %s" (Shape.to_string x_shape);
          let idx' = normalize_index x_shape.(dim) idx in
          if idx' < 0 || idx' >= x_shape.(dim) then
            err "set" "index %d at dimension %d, out of bounds for shape %s, index %d at dim %d: %d not in [0, %d)"
              idx dim (Shape.to_string x_shape) dim dim idx' x_shape.(dim);
          idx')
        indices
    in
    set_slice_internal (List.map (fun i -> I i) checked) x value

  let unsafe_get indices x =
    let t = get indices x in
    let ba = data t in
    if numel t <> 1 then
      err "unsafe_get" "expected scalar result, got %d elements" (numel t);
    match View.strides_opt (B.view t) with
    | Some _ -> Nx_buffer.get ba (offset t)
    | None ->
        if Nx_buffer.length ba = 1 then Nx_buffer.get ba 0
        else
          invalid_arg "unsafe_get: cannot read from non-composable scalar view"

  let unsafe_set indices value x =
    set indices x (scalar (B.context x) (dtype x) value)

  let slice specs t = slice_internal specs t
  let set_slice specs t value = set_slice_internal specs t value

  let item indices t =
    let s = shape t in
    if List.length indices <> Array.length s then
      invalid_arg
        (Printf.sprintf "item: need %d indices for %d-d tensor, got %d"
           (Array.length s) (Array.length s) (List.length indices));
    unsafe_get [] (get indices t)

  let set_item indices value t =
    let s = shape t in
    if List.length indices <> Array.length s then
      invalid_arg
        (Printf.sprintf "set_item: need %d indices for %dD tensor, got %d"
           (Array.length s) (Array.length s) (List.length indices));
    unsafe_set indices value t

  let put ?axis ~indices ~values ?(mode = `raise) t =
    let@ _ = span ~op:"put" () in
    let indices =
      if dtype indices = Int32 then indices else astype Int32 indices
    in
    let ctx = B.context t in
    match axis with
    | None ->
        let orig_shape = shape t in
        let t_flat = reshape [| numel t |] t in
        let idx = apply_index_mode ~mode ~n:(numel t) ctx indices in
        let result =
          B.scatter ~mode:`Set ~unique_indices:false t_flat
            ~indices:(reshape [| numel indices |] idx)
            ~updates:(reshape [| numel values |] values) ~axis:0
        in
        blit (reshape orig_shape result) t
    | Some axis ->
        let axis = resolve_single_axis t axis in
        let idx = apply_index_mode ~mode ~n:(dim axis t) ctx indices in
        let result =
          B.scatter ~mode:`Set ~unique_indices:false t
            ~indices:idx ~updates:values ~axis
        in
        blit result t

  let index_put ~indices ~values ?(mode = `raise) t =
    let@ _ = span ~op:"index_put" () in
    let ctx = B.context t in
    let t_shape = shape t in
    let nd = Array.length t_shape in
    if nd = 0 then
      invalid_arg "index_put: tensor rank, cannot index into scalar tensor";
    if Array.length indices <> nd then
      err "index_put" "indices, expected %d index tensors, got %d"
        nd (Array.length indices);
    let indices_bc =
      Array.map (fun idx ->
          if dtype idx = Int32 then idx else astype Int32 idx)
        indices
      |> Array.to_list |> broadcast_arrays |> Array.of_list
    in
    let indices_processed =
      Array.mapi
        (fun axis idx ->
          let n = t_shape.(axis) in
          if n = 0 && numel idx <> 0 then
            err "index_put" "axis %d, cannot index into zero-sized dimension" axis;
          if numel idx = 0 then idx
          else
            match mode with
            | `raise -> idx
            | `wrap ->
                let m = broadcast_to (shape idx) (scalar ctx Int32 (Int32.of_int n)) in
                let wrapped = mod_ idx m in
                let z = zeros ctx Int32 (shape idx) in
                where (cmplt wrapped z) (add wrapped m) wrapped
            | `clip ->
                minimum
                  (maximum idx (zeros ctx Int32 (shape idx)))
                  (full ctx Int32 (shape idx) (Int32.of_int (n - 1))))
        indices_bc
    in
    let target_shape = shape indices_processed.(0) in
    if array_prod target_shape = 0 then ()
    else
      let values =
        if shape values = target_shape then values
        else broadcast_to target_shape values
      in
      let strides = Shape.c_contiguous_strides t_shape in
      let flat_indices =
        let acc = ref (zeros ctx Int32 target_shape) in
        for axis = 0 to nd - 1 do
          let idx = indices_processed.(axis) in
          let s = strides.(axis) in
          let contribution =
            if s = 0 || s = 1 then idx
            else mul idx (full ctx Int32 target_shape (Int32.of_int s))
          in
          acc := add !acc contribution
        done;
        !acc
      in
      put ~indices:flat_indices ~values ~mode:`raise t

  let put_along_axis ~axis ~indices ~values t =
    let@ _ = span ~op:"put_along_axis" () in
    let axis = resolve_single_axis t axis in
    let t_shape = shape t in
    let idx_shape = shape indices in
    if Array.length t_shape <> Array.length idx_shape then
      err "put_along_axis" "cannot reshape %s to %s" (Shape.to_string idx_shape) (Shape.to_string t_shape);
    let values =
      if shape values = idx_shape then values
      else broadcast_to idx_shape values
    in
    blit
      (B.scatter ~mode:`Set ~unique_indices:false t ~indices ~updates:values
         ~axis)
      t

  (* Data-dependent output shapes — not differentiable *)

  let nonzero_indices_only (condition : (bool, bool_elt) t) =
    let total = numel condition in
    let cond_flat = reshape [| total |] condition in
    let n =
      sum (astype Int32 cond_flat) |> squeeze |> unsafe_get [] |> Int32.to_int
    in
    if n = 0 then [| empty (B.context condition) Int32 [| 0 |] |]
    else
      let result =
        create (B.context condition) Int32 [| n |] (Array.make n 0l)
      in
      let idx = ref 0 in
      for i = 0 to total - 1 do
        if unsafe_get [ i ] cond_flat then begin
          set_item [ !idx ] (Int32.of_int i) result;
          incr idx
        end
      done;
      [| result |]

  let compress ?axis ~(condition : (bool, bool_elt) t) t =
    let@ _ = span ~op:"compress" () in
    match axis with
    | None ->
        let t_flat = flatten t in
        let cond_flat = flatten condition in
        let n =
          sum ~axes:[ 0 ] (astype Int32 cond_flat)
          |> squeeze |> unsafe_get [] |> Int32.to_int
        in
        if n = 0 then empty (B.context t) (dtype t) [| 0 |]
        else take (nonzero_indices_only cond_flat).(0) t_flat
    | Some axis ->
        let axis = resolve_single_axis t axis in
        let axis_size = dim axis t in
        if numel condition <> axis_size then
          invalid_arg
            (Printf.sprintf "compress: length %d doesn't match axis %d size %d"
               (numel condition) axis axis_size);
        let cond_1d = reshape [| axis_size |] condition in
        let true_idx = nonzero_indices_only cond_1d in
        if numel true_idx.(0) = 0 then begin
          let s = Array.copy (shape t) in
          s.(axis) <- 0;
          empty (B.context t) (dtype t) s
        end
        else take ~axis true_idx.(0) t

  let extract ~condition t =
    let@ _ = span ~op:"extract" () in
    if shape condition <> shape t then invalid_arg "extract: shape mismatch";
    compress ~condition (flatten t)

  let nonzero (type a b) (t : (a, b) t) =
    let@ _ = span ~op:"nonzero" () in
    let t_shape = shape t in
    let nd = Array.length t_shape in
    let mask =
      not_equal t (broadcast_to t_shape (zeros (B.context t) (dtype t) [| 1 |]))
    in
    let mask_flat = reshape [| numel mask |] mask in
    let n =
      sum (astype Int32 mask_flat) |> squeeze |> unsafe_get [] |> Int32.to_int
    in
    if n = 0 then
      Array.init nd (fun _ -> empty (B.context t) Int32 [| 0 |])
    else
      let coords =
        Array.init nd (fun _ ->
            create (B.context t) Int32 [| n |] (Array.make n 0l))
      in
      let idx = ref 0 in
      let pos = Array.make nd 0 in
      let rec walk dim =
        if dim = nd then begin
          let elem = get (Array.to_list pos) t in
          let z = zeros (B.context t) (dtype t) (shape elem) in
          if unsafe_get [] (not_equal elem z) <> false then begin
            for d = 0 to nd - 1 do
              set_item [ !idx ] (Int32.of_int pos.(d)) coords.(d)
            done;
            incr idx
          end
        end
        else
          for i = 0 to t_shape.(dim) - 1 do
            pos.(dim) <- i;
            walk (dim + 1)
          done
      in
      walk 0;
      Array.map (fun c -> slice [ Rs (0, !idx, 1) ] c) coords

  let argwhere t =
    let coords = nonzero t in
    if Array.length coords = 0 then empty (B.context t) Int32 [| 0; 0 |]
    else
      let n = dim 0 coords.(0) in
      let nd = Array.length coords in
      if n = 0 then empty (B.context t) Int32 [| 0; nd |]
      else
        let result = zeros (B.context t) Int32 [| n; nd |] in
        for i = 0 to nd - 1 do
          blit (flatten coords.(i)) (slice_internal [ A; I i ] result)
        done;
        result

  (* ───── Splitting ───── *)

  let array_split ~axis sections x =
    let nd = ndim x in
    let axis = resolve_single_axis x axis in
    let axis_size = dim axis x in
    let make_slice start stop =
      if start < stop then
        slice_internal
          (List.init nd (fun j -> if j = axis then R (start, stop) else A)) x
      else
        let s = Array.copy (shape x) in
        s.(axis) <- 0;
        empty (B.context x) (dtype x) s
    in
    match sections with
    | `Indices indices ->
        let idx = Array.of_list indices in
        let n = Array.length idx + 1 in
        let bounds = Array.make (n + 1) 0 in
        Array.iteri (fun i v -> bounds.(i + 1) <- v) idx;
        bounds.(n) <- axis_size;
        Array.to_list (Array.init n (fun i -> make_slice bounds.(i) bounds.(i+1)))
    | `Count n ->
        if n <= 0 then
          err "array_split" "sections must be >= 1, got %d" n;
        let base = axis_size / n in
        let rem = axis_size mod n in
        let splits = Array.make n x in
        let start = ref 0 in
        for i = 0 to n - 1 do
          let sz = base + (if i < rem then 1 else 0) in
          splits.(i) <- make_slice !start (!start + sz);
          start := !start + sz
        done;
        Array.to_list splits

  let split ~axis sections x =
    let axis = resolve_single_axis x axis in
    let axis_size = dim axis x in
    if axis_size mod sections <> 0 then
      err "split" "cannot divide evenly axis %d (size %d) to %d sections, %d %% %d = %d, use array_split for uneven division"
        axis axis_size sections axis_size sections (axis_size mod sections);
    array_split ~axis (`Count sections) x

  (* ───── Sorting and Searching ───── *)

  let sort (type a b) ?(descending = false) ?(axis = -1) (x : (a, b) t) =
    let@ _ = span ~op:"sort" () in
    if ndim x = 0 then (x, scalar (B.context x) Dtype.int32 0l)
    else
      let r = ndim x in
      let axis = if axis < 0 then axis + r else axis in
      if axis < 0 || axis >= r then
        err "sort" "axis %d out of bounds for %dD tensor" axis r;
      let out_sorted = empty (B.context x) (dtype x) (shape x) in
      let out_indices = empty (B.context x) Dtype.int32 (shape x) in
      B.sort ~out:out_sorted ~axis ~descending x;
      B.argsort ~out:out_indices ~axis ~descending x;
      (out_sorted, out_indices)

  let argsort ?(descending = false) ?(axis = -1) x =
    let@ _ = span ~op:"argsort" () in
    snd (sort ~descending ~axis x)

  let argmax ?axis ?(keepdims = false) x =
    let@ _ = span ~op:"argmax" () in
    let x', axis = match axis with
      | None -> (flatten x, 0)
      | Some a ->
          let r = ndim x in
          let a = resolve_single_axis ~ndim_opt:r x a in
          if a < 0 || a >= r then
            err "argmax" "axis %d out of bounds for %dD tensor" a r;
          (x, a)
    in
    let out =
      empty (B.context x) Dtype.int32
        (reduce_output_shape (shape x') [| axis |] keepdims)
    in
    B.argmax ~out ~axis ~keepdims x';
    out

  let argmin (type a b) ?axis ?(keepdims = false) (x : (a, b) t)
      : (int32, Dtype.int32_elt) t =
    let@ _ = span ~op:"argmin" () in
    let x', axis = match axis with
      | None -> (flatten x, 0)
      | Some a ->
          let r = ndim x in
          let a = resolve_single_axis ~ndim_opt:r x a in
          if a < 0 || a >= r then
            err "argmin" "axis %d out of bounds for %dD tensor" a r;
          (x, a)
    in
    let out =
      empty (B.context x) Dtype.int32
        (reduce_output_shape (shape x') [| axis |] keepdims)
    in
    B.argmin ~out ~axis ~keepdims x';
    out

  (* ───── Random Number Generation ───── *)

  let validate_random_float_params op dtype shape =
    if not (Dtype.is_float dtype) then
      err op "dtype %s, not a float type, rand/randn only support Float16, Float32, Float64"
        (Dtype.to_string dtype);
    if Array.exists (fun x -> x < 0) shape then
      err op "invalid shape %s, dimensions must be non-negative" (Shape.to_string shape)

  let rand ctx dtype shape =
    let@ _ = span ~op:"rand" () in
    validate_random_float_params "rand" dtype shape;
    let key = Rng.next_key () in
    let n = array_prod shape in
    if n = 0 then zeros ctx dtype shape
    else
      (* Threefry: each value needs 2 int32s for key and counter *)
      let key_t =
        create ctx Dtype.int32 [| n; 2 |]
          (Array.init (n * 2) (fun i -> Int32.of_int (Rng.fold_in key i)))
      in
      let counter =
        create ctx Dtype.int32 [| n; 2 |]
          (Array.init (n * 2) (fun i -> Int32.of_int i))
      in
      let bits = empty ctx Dtype.int32 [| n; 2 |] in
      B.threefry ~out:bits key_t counter;
      let bits_flat = flatten bits in
      let bits_needed =
        if n < size bits_flat then shrink [| (0, n) |] bits_flat
        else bits_flat
      in
      (* Signed int32 → [0, 1): add 2^31 then divide by 2^32 *)
      let f32 = cast Dtype.float32 bits_needed in
      let normalized =
        div (add f32 (scalar ctx Dtype.float32 2147483648.0))
          (scalar ctx Dtype.float32 4294967296.0)
      in
      reshape shape (cast dtype normalized)

  let randn ctx dtype shape =
    let@ _ = span ~op:"randn" () in
    validate_random_float_params "randn" dtype shape;
    if array_prod shape = 0 then zeros ctx dtype shape
    else
      (* Box-Muller: z = cos(2π u1) · sqrt(-2 ln(u2)) *)
      let u1 = rand ctx Dtype.float32 shape in
      let u2 = rand ctx Dtype.float32 shape in
      let angle = mul u1 (scalar ctx Dtype.float32 (2.0 *. Float.pi)) in
      let u2_safe = maximum (sub (ones_like u2) u2) (scalar ctx Dtype.float32 1e-7) in
      let result =
        mul (cos angle)
          (sqrt (mul (scalar ctx Dtype.float32 (-2.0)) (log u2_safe)))
      in
      cast dtype result

  let randint ctx dtype ?(high = 10) shape low =
    if low >= high then
      err "randint" "range, low=%d >= high=%d" low high;
    if not (Dtype.is_int dtype) then
      invalid_arg "randint: dtype, only integer dtypes supported";
    let u = rand ctx Dtype.float32 shape in
    astype dtype
      (add (mul u (scalar ctx Dtype.float32 (float_of_int (high - low))))
         (scalar ctx Dtype.float32 (float_of_int low)))

  let bernoulli ctx ~p shape =
    if p < 0.0 || p > 1.0 then
      invalid_arg "bernoulli: p must be in [0, 1]";
    if Array.exists (fun x -> x < 0) shape then
      err "bernoulli" "invalid shape %s, dimensions must be non-negative" (Shape.to_string shape);
    cmplt (rand ctx Dtype.float32 shape) (scalar ctx Dtype.float32 p)

  let permutation ctx n =
    if n <= 0 then
      invalid_arg "permutation: n must be positive";
    argsort (rand ctx Dtype.float32 [| n |]) ~axis:0 ~descending:false

  let shuffle ctx x =
    let s = shape x in
    if Array.length s = 0 then x
    else take ~axis:0 (permutation ctx s.(0)) x

  let categorical (type a b) ctx ?(axis = -1) ?shape:(batch_shape = [||])
      (logits : (a, b) t) =
    let logits_dtype = dtype logits in
    let logits_shape = shape logits in
    if not (Dtype.is_float logits_dtype) then
      invalid_arg "categorical: logits requires floating point dtype";
    let nd = Array.length logits_shape in
    let axis = if axis < 0 then nd + axis else axis in
    if axis < 0 || axis >= nd then
      err "categorical" "axis %d out of bounds for %dD tensor" axis nd;
    let full_shape = Array.append batch_shape logits_shape in
    (* Gumbel-max trick: argmax(logits + Gumbel noise) *)
    let run_float float_dtype eps =
      let u = clip (rand ctx float_dtype full_shape) ~min:eps ~max:(1. -. eps) in
      let neg_one = scalar ctx float_dtype (-1.0) in
      let gumbel =
        mul (log (mul (log u) neg_one)) neg_one |> astype logits_dtype
      in
      astype Dtype.int32
        (argmax (add logits gumbel)
           ~axis:(axis + Array.length batch_shape) ~keepdims:false)
    in
    match logits_dtype with
    | Float64 -> run_float Dtype.float64 1e-12
    | Float32 -> run_float Dtype.float32 1e-6
    | Float16 -> run_float Dtype.float32 1e-3
    | BFloat16 -> run_float Dtype.float32 1e-2
    | Float8_e4m3 | Float8_e5m2 ->
        invalid_arg "categorical: logits, float8 logits not supported"
    | _ ->
        invalid_arg "categorical: logits requires floating point dtype"

  let truncated_normal (type a b) ctx (dtype : (a, b) Dtype.t) ~lower ~upper
      shape =
    if lower >= upper then
      invalid_arg "truncated_normal: bounds, lower must be less than upper";
    (match dtype with
     | Float16 | Float32 | Float64 | BFloat16 -> ()
     | _ ->
         invalid_arg "truncated_normal: dtype must be floating point");
    let lo = scalar ctx Dtype.float64 lower |> astype dtype in
    let hi = scalar ctx Dtype.float64 upper |> astype dtype in
    let has_remaining mask =
      match to_array (any mask) with [| v |] -> v | _ -> false
    in
    let initial = randn ctx dtype shape in
    let accepted =
      logical_and (greater_equal initial lo) (less_equal initial hi)
    in
    let remaining = logical_not accepted in
    let rec fill acc remaining attempt =
      if not (has_remaining remaining) then acc
      else if attempt > 1000 then
        invalid_arg "truncated_normal: generation, failed to find samples within bounds after 1000 tries"
      else
        let c = randn ctx dtype shape in
        let within =
          logical_and (greater_equal c lo) (less_equal c hi)
        in
        let take_new = logical_and remaining within in
        fill (where take_new c acc)
          (logical_and remaining (logical_not within))
          (attempt + 1)
    in
    fill initial remaining 1

  (*---------------------------------------------------------------------------
     Linear Algebra
    ---------------------------------------------------------------------------*)

  let matmul_output_shape a b =
    let sa = shape_symbolic a in
    let sb = shape_symbolic b in
    let ra = Symbolic_shape.rank sa in
    let rb = Symbolic_shape.rank sb in
    let batch_out =
      broadcast_shapes (Array.sub sa 0 (ra - 2)) (Array.sub sb 0 (rb - 2))
    in
    Array.concat [ batch_out; [| sa.(ra - 2); sb.(rb - 1) |] ]

  let matmul_with_alloc ?out a b =
    let out =
      match out with
      | Some o -> o
      | None ->
          if ndim a = 2 && ndim b = 2 then
            empty (B.context a) (B.dtype a) [| dim 0 a; dim 1 b |]
          else
            let s = matmul_output_shape a b in
            let s =
              match Symbolic_shape.eval s with
              | Some s -> s
              | None ->
                  invalid_arg "matmul: cannot compute output shape with symbolic dimensions"
            in
            empty (B.context a) (B.dtype a) s
    in
    B.matmul ~out a b;
    out

  let dot ?out x w =
    let@ _ = span ~op:"dot" () in
    if not (ndim x > 0 && ndim w > 0) then
      invalid_arg "dot: tensors, both must be at least 1D";
    match ndim x, ndim w with
    | 1, 1 -> sum ?out (mul x w)
    | 1, _ ->
        let r = matmul_with_alloc (unsqueeze ~axes:[ 0 ] x) w in
        copy_to_out ?out (squeeze ~axes:[ ndim r - 2 ] r)
    | _, 1 ->
        let r = matmul_with_alloc x (unsqueeze ~axes:[ 1 ] w) in
        copy_to_out ?out (squeeze ~axes:[ ndim r - 1 ] r)
    | _ -> matmul_with_alloc ?out x w

  let matmul ?out a_orig b_orig =
    let@ _ = span ~op:"matmul" () in
    if ndim a_orig = 0 || ndim b_orig = 0 then
      invalid_arg "matmul: inputs cannot be 0-D (scalars)";
    if ndim a_orig >= 2 && ndim b_orig >= 2 then
      matmul_with_alloc ?out a_orig b_orig
    else
      let a, b =
        match ndim a_orig, ndim b_orig with
        | 1, 1 -> unsqueeze ~axes:[ 0 ] a_orig, unsqueeze ~axes:[ 1 ] b_orig
        | 1, _ -> unsqueeze ~axes:[ 0 ] a_orig, b_orig
        | _ -> a_orig, unsqueeze ~axes:[ 1 ] b_orig
      in
      let r = matmul_with_alloc a b in
      if ndim a_orig = 1 && ndim b_orig = 1 then squeeze r
      else if ndim a_orig = 1 then squeeze ~axes:[ ndim r - 2 ] r
      else squeeze ~axes:[ ndim r - 1 ] r

  let diagonal ?(offset = 0) ?axis1 ?axis2 x =
    let@ _ = span ~op:"diagonal" () in
    let nd = ndim x in
    let ax1 =
      let a = Option.value axis1 ~default:(nd - 2) in
      if a < 0 then nd + a else a
    in
    let ax2 =
      let a = Option.value axis2 ~default:(nd - 1) in
      if a < 0 then nd + a else a
    in
    if ax1 = ax2 then
      invalid_arg "diagonal: axes must be different";
    let perm =
      let others =
        List.filter (fun a -> a <> ax1 && a <> ax2) (List.init nd Fun.id)
      in
      others @ [ ax1; ax2 ]
    in
    let x_trans = transpose ~axes:perm x in
    let d1 = dim (nd - 2) x_trans in
    let d2 = dim (nd - 1) x_trans in
    let diag_len =
      if offset >= 0 then Stdlib.max 0 (Stdlib.min d1 (d2 - offset))
      else Stdlib.max 0 (Stdlib.min (d1 + offset) d2)
    in
    if diag_len = 0 then
      empty (B.context x) (dtype x)
        (Array.append (Array.sub (shape x_trans) 0 (nd - 2)) [| 0 |])
    else
      let prefix = Array.sub (shape x_trans) 0 (nd - 2) in
      let x_flat =
        reshape (Array.append prefix [| d1 * d2 |]) (contiguous x_trans)
      in
      (* Diagonal indices: start + i*(d2+1) for i in 0..diag_len-1 *)
      let start = if offset >= 0 then offset else -offset * d2 in
      let step = d2 + 1 in
      let ctx = B.context x in
      let idx =
        add
          (mul (arange ctx Dtype.int32 0 diag_len 1)
             (scalar ctx Dtype.int32 (Int32.of_int step)))
          (scalar ctx Dtype.int32 (Int32.of_int start))
      in
      take ~axis:(nd - 2) idx x_flat

  let matrix_transpose x =
    let nd = ndim x in
    if nd < 2 then x else swapaxes (nd - 2) (nd - 1) x

  (* ───── Complex ───── *)

  let extract_complex_part (type a b) ~op ~field (x : (a, b) t) =
    let extract (type c d e f)
        (x : (Complex.t, c) t) (out_dt : (d, e) Dtype.t) (get : Complex.t -> d)
        : (f, _) t =
      let s = shape x in
      let size = array_prod s in
      let data =
        Array.init size (fun i ->
            let idx = Shape.unravel_index i s |> Array.to_list in
            get (unsafe_get idx x))
      in
      Obj.magic (create (B.context x) out_dt s data)
    in
    match dtype x with
    | Complex64 ->
        extract (x : (Complex.t, complex32_elt) t) Dtype.float32
          (fun c -> field c)
    | Complex128 ->
        extract (x : (Complex.t, complex64_elt) t) Dtype.float64
          (fun c -> field c)
    | _ ->
        err op "dtype, input must be complex64 or complex128"

  let complex (type a b) ~(real : (a, b) t) ~(imag : (a, b) t) =
    let s = shape real in
    if s <> shape imag then
      err "complex" "cannot reshape %s to %s" (Shape.to_string (shape imag)) (Shape.to_string s);
    let size = array_prod s in
    match dtype real with
    | Float32 ->
        let real = (real : (float, float32_elt) t) in
        let imag = (imag : (float, float32_elt) t) in
        let data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i s |> Array.to_list in
              Complex.{ re = unsafe_get idx real; im = unsafe_get idx imag })
        in
        Obj.magic (create (B.context real) Dtype.complex64 s data)
    | Float64 ->
        let real = (real : (float, float64_elt) t) in
        let imag = (imag : (float, float64_elt) t) in
        let data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i s |> Array.to_list in
              Complex.{ re = unsafe_get idx real; im = unsafe_get idx imag })
        in
        Obj.magic (create (B.context real) Dtype.complex128 s data)
    | _ ->
        invalid_arg "complex: dtype, real and imag must be float32 or float64"

  let real (type a b) (x : (a, b) t) =
    extract_complex_part ~op:"real" ~field:(fun c -> c.Complex.re) x

  let imag (type a b) (x : (a, b) t) =
    extract_complex_part ~op:"imag" ~field:(fun c -> c.Complex.im) x

  let conjugate (type a b) (x : (a, b) t) =
    match dtype x with
    | Complex64 | Complex128 -> complex ~real:(real x) ~imag:(neg (imag x))
    | _ -> x

  (* ───── Dot Products and Tensor Contractions ───── *)

  let vdot (type a b) (a : (a, b) t) (b : (a, b) t) =
    let@ _ = span ~op:"vdot" () in
    let a', b' =
      try
        let bc = broadcast_arrays [ a; b ] in
        contiguous (List.nth bc 0), contiguous (List.nth bc 1)
      with _ -> a, b
    in
    let fa = flatten a' in
    let fb = flatten b' in
    if numel fa <> numel fb then invalid_arg "vdot: different number of elements";
    match dtype a with
    | (Complex64 | Complex128) when dtype a = dtype b ->
        sum (mul (conjugate fa) fb)
    | _ -> sum (mul fa fb)

  let vecdot ?axis x1 x2 =
    let@ _ = span ~op:"vecdot" () in
    let ax =
      match axis with
      | None -> ndim x1 - 1
      | Some a -> if a < 0 then ndim x1 + a else a
    in
    sum ~axes:[ ax ] ~keepdims:false (mul x1 x2)

  let inner a b =
    let@ _ = span ~op:"inner" () in
    if (shape a).(ndim a - 1) <> (shape b).(ndim b - 1) then
      invalid_arg "inner: last dimensions differ";
    vecdot ~axis:(-1) a b

  let outer ?out a b =
    let@ _ = span ~op:"outer" () in
    let fa = if ndim a = 0 then reshape [| 1 |] a else flatten a in
    let fb = if ndim b = 0 then reshape [| 1 |] b else flatten b in
    let r =
      matmul ?out (reshape [| numel fa; 1 |] fa) (reshape [| 1; numel fb |] fb)
    in
    let r = if ndim a = 0 then squeeze ~axes:[ 0 ] r else r in
    if ndim b = 0 then squeeze ~axes:[ (if ndim a = 0 then 0 else 1) ] r else r

  let tensordot ?axes a b =
    let@ _ = span ~op:"tensordot" () in
    match axes with
    | None -> matmul a b
    | Some (axes_a, axes_b) ->
        let n_axes = List.length axes_a in
        if n_axes <> List.length axes_b then
          invalid_arg "tensordot: axes lists must have same length";
        let ndim_a = ndim a in
        let ndim_b = ndim b in
        let axes_a =
          Array.of_list
            (List.map (fun ax -> if ax < 0 then ndim_a + ax else ax) axes_a)
        in
        let axes_b =
          Array.of_list
            (List.map (fun ax -> if ax < 0 then ndim_b + ax else ax) axes_b)
        in
        let sa = shape a in
        let sb = shape b in
        Array.iter2
          (fun ax_a ax_b ->
            if sa.(ax_a) <> sb.(ax_b) then
              invalid_arg "tensordot: axes have different sizes")
          axes_a axes_b;
        let axes_a_set =
          Array.fold_left (fun s x -> IntSet.add x s) IntSet.empty axes_a
        in
        let axes_b_set =
          Array.fold_left (fun s x -> IntSet.add x s) IntSet.empty axes_b
        in
        let free_a =
          Array.of_list
            (List.filter
               (fun i -> not (IntSet.mem i axes_a_set))
               (List.init ndim_a Fun.id))
        in
        let free_b =
          Array.of_list
            (List.filter
               (fun i -> not (IntSet.mem i axes_b_set))
               (List.init ndim_b Fun.id))
        in
        let perm_a = Array.append free_a axes_a in
        let perm_b = Array.append axes_b free_b in
        let do_transpose perm len t =
          if Array.length perm > 1 then
            contiguous (transpose ~axes:(Array.to_list perm) t)
          else t
        in
        let at = do_transpose perm_a ndim_a a in
        let bt = do_transpose perm_b ndim_b b in
        let sat = shape at in
        let sbt = shape bt in
        let nfa = Array.length free_a in
        let nfb = Array.length free_b in
        let prod arr = Array.fold_left ( * ) 1 arr in
        let free_size_a =
          if nfa = 0 then 1 else prod (Array.sub sat 0 nfa)
        in
        let free_size_b =
          if nfb = 0 then 1 else prod (Array.sub sbt n_axes (ndim_b - n_axes))
        in
        let contract_size = prod (Array.sub sat nfa n_axes) in
        let r =
          matmul
            (reshape [| free_size_a; contract_size |] at)
            (reshape [| contract_size; free_size_b |] bt)
        in
        let result_shape =
          Array.append
            (if nfa = 0 then [||] else Array.sub sat 0 nfa)
            (if nfb = 0 then [||]
             else Array.sub sbt n_axes (ndim_b - n_axes))
        in
        if Array.length result_shape = 0 then squeeze r
        else reshape result_shape r

  module Einsum = struct
    type token = Axis of char | Ellipsis

    let parse_operand str =
      let len = String.length str in
      if len = 0 then []
      else
        let rec loop idx acc ell =
          if idx >= len then List.rev acc
          else
            match str.[idx] with
            | '.' ->
                if idx + 2 >= len || str.[idx + 1] <> '.' || str.[idx + 2] <> '.'
                then invalid_arg "einsum: ellipsis must be '...'";
                if ell then invalid_arg "einsum: multiple ellipsis in operand";
                loop (idx + 3) (Ellipsis :: acc) true
            | c
              when (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                   || (c >= '0' && c <= '9') || c = '_' ->
                loop (idx + 1) (Axis c :: acc) ell
            | c ->
                invalid_arg
                  (Printf.sprintf "einsum: invalid character '%c'" c)
        in
        loop 0 [] false

    let parse_equation subscripts =
      let parts = String.split_on_char '-' subscripts in
      match parts with
      | [ lhs; rhs ] when String.length rhs > 0 && rhs.[0] = '>' ->
          let inputs =
            String.split_on_char ',' lhs
            |> List.map String.trim
            |> List.filter (( <> ) "")
          in
          let output =
            String.trim (String.sub rhs 1 (String.length rhs - 1))
          in
          Array.of_list (List.map parse_operand inputs),
          Some (parse_operand output)
      | [ lhs ] ->
          let inputs =
            String.split_on_char ',' lhs
            |> List.map String.trim
            |> List.filter (( <> ) "")
          in
          Array.of_list (List.map parse_operand inputs), None
      | _ -> invalid_arg "einsum: invalid format, expected inputs->output"

    let handle_repeated_indices tensor tokens =
      let rec find_dups acc idx = function
        | [] -> None
        | Axis c :: rest -> (
            match List.find_opt (fun (ch, _) -> ch = c) acc with
            | Some (_, prev) -> Some (prev, idx, c)
            | None -> find_dups ((c, idx) :: acc) (idx + 1) rest)
        | Ellipsis :: rest -> find_dups acc (idx + 1) rest
      in
      let rec process t toks =
        match find_dups [] 0 toks with
        | None -> t, toks
        | Some (ax1, ax2, c) ->
            let s = shape t in
            if s.(ax1) <> s.(ax2) then
              invalid_arg
                (Printf.sprintf
                   "einsum: index var '%c' must have consistent dimensions \
                    (%d vs %d)"
                   c s.(ax1) s.(ax2));
            let t' = diagonal ~axis1:ax1 ~axis2:ax2 t in
            let rec remove_at i = function
              | [] -> []
              | _ :: xs when i = 0 -> xs
              | x :: xs -> x :: remove_at (i - 1) xs
            in
            process t' (remove_at ax2 toks)
      in
      process tensor tokens

    type tensor_info = {
      id : int;
      shape : int array;
      axis_labels : char list;
    }

    type contraction_path =
      | Leaf of int
      | Node of contraction_path * contraction_path * tensor_info

    let estimate_cost (t1 : tensor_info) (t2 : tensor_info) common_chars =
      let dim_map = Hashtbl.create 16 in
      List.iteri (fun i c -> Hashtbl.replace dim_map c t1.shape.(i)) t1.axis_labels;
      List.iteri (fun i c -> Hashtbl.replace dim_map c t2.shape.(i)) t2.axis_labels;
      let all =
        List.sort_uniq Char.compare (t1.axis_labels @ t2.axis_labels)
      in
      let output_size =
        List.fold_left
          (fun acc c ->
            if List.mem c common_chars then acc
            else acc * Hashtbl.find dim_map c)
          1 all
      in
      let op_cost =
        List.fold_left
          (fun acc c -> acc * Hashtbl.find dim_map c)
          1 all
      in
      float_of_int op_cost, float_of_int output_size

    let optimize_path inputs output_chars =
      let workset = ref (List.mapi (fun i t -> Leaf i, t) inputs) in
      let contract_info (p1, t1) (p2, t2) =
        let common =
          List.filter (fun c -> List.mem c t2.axis_labels) t1.axis_labels
        in
        let new_labels =
          let all =
            List.sort_uniq Char.compare (t1.axis_labels @ t2.axis_labels)
          in
          List.filter
            (fun c -> (not (List.mem c common)) || List.mem c output_chars)
            all
        in
        let find_index x lst =
          let rec aux i = function
            | [] -> raise Not_found
            | h :: _ when h = x -> i
            | _ :: t -> aux (i + 1) t
          in
          aux 0 lst
        in
        let get_dim c =
          if List.mem c t1.axis_labels then
            t1.shape.(find_index c t1.axis_labels)
          else t2.shape.(find_index c t2.axis_labels)
        in
        let new_shape = Array.of_list (List.map get_dim new_labels) in
        let info = { id = -1; shape = new_shape; axis_labels = new_labels } in
        let cost, size =
          estimate_cost t1 t2
            (List.filter (fun c -> not (List.mem c new_labels)) common)
        in
        cost, size, Node (p1, p2, info), info
      in
      while List.length !workset > 1 do
        let items = !workset in
        let best = ref None in
        let min_cost = ref Float.infinity in
        let rec iter_pairs = function
          | [] -> ()
          | x :: rest ->
              List.iter
                (fun y ->
                  let cost, _, path, info = contract_info x y in
                  if cost < !min_cost then (
                    min_cost := cost;
                    best := Some (x, y, path, info)))
                rest;
              iter_pairs rest
        in
        iter_pairs items;
        match !best with
        | None -> failwith "einsum: could not find valid contraction"
        | Some (i1, i2, new_path, new_info) ->
            workset :=
              (new_path, new_info) ::
              List.filter (fun x -> x != i1 && x != i2) items
      done;
      match !workset with
      | [ (p, _) ] -> p
      | _ -> failwith "einsum: optimization failed"

    let contract_pair op_a str_a op_b str_b result_str =
      let sa = shape op_a in
      let sb = shape op_b in
      let chars_a = String.to_seq str_a |> List.of_seq in
      let chars_b = String.to_seq str_b |> List.of_seq in
      let chars_out = String.to_seq result_str |> List.of_seq in
      let batch_chars =
        List.filter
          (fun c -> List.mem c chars_b && List.mem c chars_out)
          chars_a
      in
      let contract_chars =
        List.filter
          (fun c -> List.mem c chars_b && not (List.mem c chars_out))
          chars_a
      in
      let a_free = List.filter (fun c -> not (List.mem c chars_b)) chars_a in
      let b_free = List.filter (fun c -> not (List.mem c chars_a)) chars_b in
      let get_axes source target =
        List.map
          (fun c ->
            let rec find i = function
              | [] -> failwith "char not found"
              | x :: _ when x = c -> i
              | _ :: xs -> find (i + 1) xs
            in
            find 0 source)
          target
      in
      let perm_a = get_axes chars_a (batch_chars @ a_free @ contract_chars) in
      let perm_b = get_axes chars_b (batch_chars @ contract_chars @ b_free) in
      let is_identity perm n =
        let rec check i = function
          | [] -> i = n
          | x :: xs -> x = i && check (i + 1) xs
        in
        check 0 perm
      in
      let at =
        if is_identity perm_a (String.length str_a) then op_a
        else contiguous (transpose ~axes:perm_a op_a)
      in
      let bt =
        if is_identity perm_b (String.length str_b) then op_b
        else contiguous (transpose ~axes:perm_b op_b)
      in
      let prod dims = Array.fold_left ( * ) 1 dims in
      let pa = Array.of_list perm_a in
      let pb = Array.of_list perm_b in
      let nb = List.length batch_chars in
      let naf = List.length a_free in
      let nc = List.length contract_chars in
      let nbf = List.length b_free in
      let batch_dims =
        Array.init nb (fun i ->
            let da = sa.(pa.(i)) in
            let db = sb.(pb.(i)) in
            if da = db then da
            else if da = 1 then db
            else if db = 1 then da
            else
              invalid_arg
                (Printf.sprintf
                   "einsum: incompatible broadcast dimensions (%d vs %d)" da db))
      in
      let a_free_dims = Array.init naf (fun i -> sa.(pa.(nb + i))) in
      let contract_dims = Array.init nc (fun i -> sa.(pa.(nb + naf + i))) in
      let b_free_dims = Array.init nbf (fun i -> sb.(pb.(nb + nc + i))) in
      let bs = prod batch_dims in
      let m = prod a_free_dims in
      let k = prod contract_dims in
      let n = prod b_free_dims in
      let broadcast_batch tensor parr src_shape =
        if nb = 0 then tensor
        else
          let needs = ref false in
          let target =
            Array.init (ndim tensor) (fun i ->
                if i < nb then (
                  let src = src_shape.(parr.(i)) in
                  let tgt = batch_dims.(i) in
                  if src <> tgt then needs := true;
                  tgt)
                else src_shape.(parr.(i)))
          in
          if !needs then broadcast_to target tensor else tensor
      in
      let at = broadcast_batch at pa sa in
      let bt = broadcast_batch bt pb sb in
      let r =
        matmul (reshape [| bs; m; k |] at) (reshape [| bs; k; n |] bt)
      in
      let intermediate =
        reshape (Array.concat [ batch_dims; a_free_dims; b_free_dims ]) r
      in
      let inter_chars = batch_chars @ a_free @ b_free in
      if inter_chars = chars_out then intermediate
      else transpose ~axes:(get_axes inter_chars chars_out) intermediate

    let calculate subscripts operands =
      let n_ops = Array.length operands in
      if n_ops = 0 then invalid_arg "einsum: no input operands";
      match subscripts, n_ops with
      | "i,i->", 2 -> sum (mul operands.(0) operands.(1))
      | "ij,jk->ik", 2 -> matmul operands.(0) operands.(1)
      | "ij->ji", 1 -> transpose operands.(0)
      | _ ->
          let input_tokens, output_opt = parse_equation subscripts in
          if Array.length input_tokens <> n_ops then
            invalid_arg
              "einsum: number of inputs must equal number of operands";
          let ell_rank =
            let max_rank = ref 0 in
            for i = 0 to n_ops - 1 do
              let n_named =
                List.length
                  (List.filter
                     (function Axis _ -> true | _ -> false)
                     input_tokens.(i))
              in
              let r = ndim operands.(i) - n_named in
              if r < 0 then
                invalid_arg "einsum: operand rank too small for subscripts";
              if r > !max_rank then max_rank := r
            done;
            !max_rank
          in
          let get_ell_char i = char_of_int (200 + i) in
          let normalized_inputs =
            Array.mapi
              (fun i tokens ->
                let op = operands.(i) in
                let n_named =
                  List.length
                    (List.filter
                       (function Axis _ -> true | _ -> false)
                       tokens)
                in
                let ell_dim = ndim op - n_named in
                let expanded =
                  List.concat_map
                    (function
                     | Axis c -> [ Axis c ]
                     | Ellipsis ->
                         List.init ell_dim (fun k ->
                             Axis (get_ell_char (ell_rank - ell_dim + k))))
                    tokens
                in
                let op_diag, final = handle_repeated_indices op expanded in
                let chars =
                  List.map
                    (function Axis c -> c | _ -> assert false)
                    final
                in
                { id = i; shape = shape op_diag; axis_labels = chars }, op_diag)
              input_tokens
          in
          let ops_info = Array.map fst normalized_inputs in
          let ops_tensors = Array.map snd normalized_inputs in
          (* Validate dimension consistency *)
          let char_dims = Hashtbl.create 16 in
          Array.iter
            (fun info ->
              List.iteri
                (fun idx c ->
                  let d = info.shape.(idx) in
                  match Hashtbl.find_opt char_dims c with
                  | None -> Hashtbl.add char_dims c d
                  | Some prev ->
                      if prev <> d && prev <> 1 && d <> 1 then
                        invalid_arg
                          (Printf.sprintf
                             "einsum: index var '%c' must have consistent \
                              dimensions (%d vs %d)"
                             c prev d)
                      else if d > prev then Hashtbl.replace char_dims c d)
                info.axis_labels)
            ops_info;
          let inputs_have_ell =
            Array.exists
              (fun toks -> List.exists (( = ) Ellipsis) toks)
              input_tokens
          in
          let target_chars =
            match output_opt with
            | Some tokens ->
                if List.exists (( = ) Ellipsis) tokens
                   && not inputs_have_ell then
                  invalid_arg
                    "einsum: output ellipsis requires ellipsis in inputs";
                List.concat_map
                  (function
                   | Axis c -> [ c ]
                   | Ellipsis ->
                       List.init ell_rank (fun k -> get_ell_char k))
                  tokens
            | None ->
                let all_chars =
                  List.concat
                    (Array.to_list
                       (Array.map
                          (fun toks ->
                            List.filter_map
                              (function
                               | Axis c -> Some c
                               | Ellipsis -> None)
                              toks)
                          input_tokens))
                in
                let counts = Hashtbl.create 16 in
                List.iter
                  (fun c ->
                    Hashtbl.replace counts c
                      (1
                      + (Hashtbl.find_opt counts c
                        |> Option.value ~default:0)))
                  all_chars;
                let ell_chars =
                  List.init ell_rank (fun k -> get_ell_char k)
                in
                let named =
                  List.filter (fun c -> int_of_char c < 200) all_chars
                  |> List.sort_uniq Char.compare
                  |> List.filter (fun c -> Hashtbl.find counts c = 1)
                in
                ell_chars @ named
          in
          let all_input_chars =
            Array.fold_left
              (fun acc info -> acc @ info.axis_labels)
              [] ops_info
          in
          List.iter
            (fun c ->
              if not (List.mem c all_input_chars) then
                invalid_arg
                  (Printf.sprintf
                     "einsum: output index '%c' not found in inputs" c))
            target_chars;
          (* Pre-reduce single-operand axes absent from output *)
          Array.iteri
            (fun i info ->
              let reduce_axes = ref [] in
              let new_labels = ref [] in
              let char_count = Hashtbl.create 16 in
              Array.iter
                (fun inf ->
                  List.iter
                    (fun c ->
                      Hashtbl.replace char_count c
                        (1
                        + (Hashtbl.find_opt char_count c
                          |> Option.value ~default:0)))
                    inf.axis_labels)
                ops_info;
              List.iteri
                (fun axis_idx c ->
                  if Hashtbl.find char_count c = 1
                     && not (List.mem c target_chars)
                  then reduce_axes := axis_idx :: !reduce_axes
                  else new_labels := c :: !new_labels)
                info.axis_labels;
              match !reduce_axes with
              | [] -> ()
              | axes ->
                  ops_tensors.(i) <- sum ~axes:(List.rev axes) ops_tensors.(i);
                  ops_info.(i) <-
                    { info with
                      shape = shape ops_tensors.(i);
                      axis_labels = List.rev !new_labels })
            ops_info;
          let finalize result current_chars =
            let reduce =
              List.filter_map
                (fun (i, c) ->
                  if not (List.mem c target_chars) then Some i else None)
                (List.mapi (fun i c -> i, c) current_chars)
            in
            let result =
              if reduce = [] then result else sum ~axes:reduce result
            in
            let final =
              List.filter (fun c -> List.mem c target_chars) current_chars
            in
            if final = target_chars then result
            else
              let perm =
                List.map
                  (fun c ->
                    let rec find i = function
                      | [] -> 0
                      | x :: xs -> if x = c then i else find (i + 1) xs
                    in
                    find 0 final)
                  target_chars
              in
              transpose ~axes:perm result
          in
          if n_ops = 1 then
            finalize ops_tensors.(0) ops_info.(0).axis_labels
          else if n_ops = 2 then
            let ia = ops_info.(0) in
            let ib = ops_info.(1) in
            let stra = ia.axis_labels |> List.to_seq |> String.of_seq in
            let strb = ib.axis_labels |> List.to_seq |> String.of_seq in
            let common =
              List.filter
                (fun c -> List.mem c ib.axis_labels)
                ia.axis_labels
            in
            let result_labels =
              List.sort_uniq Char.compare
                (ia.axis_labels @ ib.axis_labels)
              |> List.filter
                   (fun c ->
                     (not (List.mem c common)) || List.mem c target_chars)
            in
            let str_out =
              result_labels |> List.to_seq |> String.of_seq
            in
            finalize
              (contract_pair ops_tensors.(0) stra ops_tensors.(1) strb
                 str_out)
              result_labels
          else
            let plan =
              optimize_path (Array.to_list ops_info) target_chars
            in
            let rec execute = function
              | Leaf idx ->
                  ops_tensors.(idx),
                  ops_info.(idx).axis_labels |> List.to_seq |> String.of_seq
              | Node (left, right, info) ->
                  let ra, sa = execute left in
                  let rb, sb = execute right in
                  let so =
                    info.axis_labels |> List.to_seq |> String.of_seq
                  in
                  contract_pair ra sa rb sb so, so
            in
            let result, rstr = execute plan in
            finalize result (String.to_seq rstr |> List.of_seq)
  end

  let einsum subscripts operands =
    let@ _ = span ~op:"einsum" () in
    Einsum.calculate subscripts operands

  let kron a b =
    let@ _ = span ~op:"kron" () in
    let sa = shape a in
    let sb = shape b in
    let a2 = if ndim a = 1 then reshape [| sa.(0); 1 |] a else a in
    let b2 = if ndim b = 1 then reshape [| sb.(0); 1 |] b else b in
    let sa2 = shape a2 in
    let sb2 = shape b2 in
    let r =
      mul (reshape [| sa2.(0); 1; sa2.(1); 1 |] a2)
        (reshape [| 1; sb2.(0); 1; sb2.(1) |] b2)
    in
    let flat = reshape [| sa2.(0) * sb2.(0); sa2.(1) * sb2.(1) |] r in
    if ndim a = 1 && ndim b = 1 then flatten flat else flat

  let multi_dot arrays =
    let@ _ = span ~op:"multi_dot" () in
    match arrays with
    | [||] -> invalid_arg "multi_dot: empty array"
    | [| arr |] -> arr
    | _ ->
        let n = Array.length arrays in
        let dims = Array.make (n + 1) 0 in
        let matrix_dims idx =
          let t = arrays.(idx) in
          match ndim t with
          | 1 ->
              let len = (shape t).(0) in
              if idx = 0 then 1, len
              else if idx = n - 1 then len, 1
              else
                invalid_arg
                  "multi_dot: only first and last arguments may be 1D vectors"
          | 2 ->
              let s = shape t in
              s.(0), s.(1)
          | _ ->
              invalid_arg
                (Printf.sprintf
                   "multi_dot: argument %d must be 1D (endpoints) or 2D \
                    matrix" idx)
        in
        for i = 0 to n - 1 do
          let rows, cols = matrix_dims i in
          if i = 0 then dims.(0) <- rows
          else if dims.(i) <> rows then
            invalid_arg
              (Printf.sprintf
                 "multi_dot: shapes not aligned between arguments %d and %d \
                  (%d <> %d)"
                 (i - 1) i dims.(i) rows);
          dims.(i + 1) <- cols
        done;
        (* MCM dynamic programming *)
        let d64 = Array.map Int64.of_int dims in
        let cost = Array.make_matrix n n Int64.zero in
        let split = Array.make_matrix n n 0 in
        for len = 2 to n do
          for i = 0 to n - len do
            let j = i + len - 1 in
            let best_c = ref Int64.max_int in
            let best_s = ref i in
            for k = i to j - 1 do
              let c =
                Int64.(
                  add cost.(i).(k)
                    (add cost.(k + 1).(j)
                       (mul d64.(i) (mul d64.(k + 1) d64.(j + 1)))))
              in
              if c < !best_c then (best_c := c; best_s := k)
            done;
            cost.(i).(j) <- !best_c;
            split.(i).(j) <- !best_s
          done
        done;
        let memo = Array.init n (fun _ -> Array.make n None) in
        let rec compute i j =
          match memo.(i).(j) with
          | Some t -> t
          | None ->
              let r =
                if i = j then arrays.(i)
                else matmul (compute i split.(i).(j))
                       (compute (split.(i).(j) + 1) j)
              in
              memo.(i).(j) <- Some r;
              r
        in
        compute 0 (n - 1)

  let cross ?out ?axis a b =
    let@ _ = span ~op:"cross" () in
    let axis =
      let ax = Option.value axis ~default:(-1) in
      if ax < 0 then ndim a + ax else ax
    in
    if axis >= ndim a then
      invalid_arg "cross: axis out of bounds";
    if (shape a).(axis) <> 3 then invalid_arg "cross: axis dim not 3";
    if (shape b).(axis) <> 3 then invalid_arg "cross: axis dim not 3";
    let at i t =
      squeeze ~axes:[ axis ]
        (slice_internal
           (Array.to_list
              (Array.init (ndim t) (fun j ->
                   if j = axis then R (i, i + 1) else A)))
           t)
    in
    let c1 = sub (mul (at 1 a) (at 2 b)) (mul (at 2 a) (at 1 b)) in
    let c2 = sub (mul (at 2 a) (at 0 b)) (mul (at 0 a) (at 2 b)) in
    let c3 = sub (mul (at 0 a) (at 1 b)) (mul (at 1 a) (at 0 b)) in
    match out with
    | Some r ->
        let write_at i v =
          set_slice_internal
            (Array.to_list
               (Array.init (ndim r) (fun j ->
                    if j = axis then R (i, i + 1) else A)))
            r (expand_dims [ axis ] v)
        in
        write_at 0 c1; write_at 1 c2; write_at 2 c3; r
    | None -> stack ~axis [ c1; c2; c3 ]

  (*---------------------------------------------------------------------------
     Matrix Decompositions and Solving
    ---------------------------------------------------------------------------*)

  let check_square ~op a =
    let sh = shape a in
    let n = Array.length sh in
    if n < 2 then
      err op "input requires at least 2D array";
    if sh.(n - 1) <> sh.(n - 2) then
      invalid_arg (Printf.sprintf "%s: coefficient matrix must be square" op)

  let check_float_or_complex (type a b) ~op (a : (a, b) t) =
    match dtype a with
    | Float16 | Float32 | Float64 | Complex64 | Complex128 -> ()
    | _ -> err op "dtype must be float or complex"

  let check_real (type a b) ~op (a : (a, b) t) =
    match dtype a with
    | Float16 | Float32 | Float64 -> ()
    | _ -> err op "dtype must be real (float)"

  let cholesky ?upper a =
    let@ _ = span ~op:"cholesky" () in
    check_square ~op:"cholesky" a;
    check_float_or_complex ~op:"cholesky" a;
    B.cholesky ~upper:(Option.value upper ~default:false) a

  let qr ?mode a =
    let@ _ = span ~op:"qr" () in
    check_float_or_complex ~op:"qr" a;
    let reduced =
      match mode with Some `Reduced -> true | None | Some `Complete -> false
    in
    B.qr ~reduced a

  let svd ?full_matrices a =
    let@ _ = span ~op:"svd" () in
    check_float_or_complex ~op:"svd" a;
    B.svd ~full_matrices:(Option.value full_matrices ~default:false) a

  let svdvals a =
    let@ _ = span ~op:"svdvals" () in
    check_float_or_complex ~op:"svdvals" a;
    let _, s, _ = B.svd ~full_matrices:false a in
    s

  let eig a =
    let@ _ = span ~op:"eig" () in
    check_square ~op:"eig" a;
    check_float_or_complex ~op:"eig" a;
    match B.eig ~vectors:true a with
    | vals, Some vecs -> vals, vecs
    | _ ->
        invalid_arg "eig: result, expected eigenvectors"

  let eigh ?uplo a =
    let@ _ = span ~op:"eigh" () in
    check_square ~op:"eigh" a;
    check_real ~op:"eigh" a;
    let _ = uplo in
    match B.eigh ~vectors:true a with
    | vals, Some vecs -> vals, vecs
    | _ ->
        invalid_arg "eigh: result, expected eigenvectors"

  let eigvals a =
    let@ _ = span ~op:"eigvals" () in
    check_square ~op:"eigvals" a;
    check_float_or_complex ~op:"eigvals" a;
    fst (B.eig ~vectors:false a)

  let eigvalsh ?uplo a =
    let@ _ = span ~op:"eigvalsh" () in
    check_square ~op:"eigvalsh" a;
    check_real ~op:"eigvalsh" a;
    let _ = uplo in
    fst (B.eigh ~vectors:false a)

  let norm (type a b) ?ord ?axes ?keepdims (x : (a, b) t) =
    let@ _ = span ~op:"norm" () in
    let keepdims = Option.value keepdims ~default:false in
    match ord, axes with
    | None, None -> sqrt (sum (square (abs x)) ~keepdims)
    | None, Some _ | Some `Fro, _ ->
        sqrt (sum (square (abs x)) ?axes ~keepdims)
    | Some `One, None ->
        max (sum (abs x) ~axes:[ ndim x - 2 ] ~keepdims) ~keepdims
    | Some `NegOne, None ->
        if ndim x = 1 then min (abs x) ~keepdims
        else min (sum (abs x) ~axes:[ ndim x - 2 ]) ~keepdims
    | Some `Two, None ->
        max (svdvals x |> cast (dtype x)) ~keepdims
    | Some `NegTwo, None ->
        min (svdvals x |> cast (dtype x)) ~keepdims
    | Some `Inf, None ->
        if ndim x = 1 then max (abs x) ~keepdims
        else max (sum (abs x) ~axes:[ ndim x - 1 ] ~keepdims) ~keepdims
    | Some `NegInf, None ->
        if ndim x = 1 then min (abs x) ~keepdims
        else min (sum (abs x) ~axes:[ ndim x - 1 ]) ~keepdims
    | Some `Nuc, None ->
        if ndim x < 2 then
          invalid_arg "norm: input, nuclear norm defined for matrices";
        sum (svdvals x |> cast (dtype x)) ~keepdims
    | Some `NegOne, _ | Some `NegTwo, _ | Some `NegInf, _ | Some `Nuc, _ ->
        invalid_arg "norm: this combination of ord and axis not implemented"
    | Some (`P p), _ ->
        if p = 1.0 && axes = None && ndim x = 2 then
          max (sum (abs x) ~axes:[ ndim x - 2 ] ~keepdims) ~keepdims
        else
          let p_t = full (B.context x) (dtype x) [||] (Dtype.of_float (dtype x) p) in
          let inv_p =
            div
              (full (B.context x) (dtype x) [||] (Dtype.one (dtype x)))
              p_t
          in
          pow (sum (pow (abs x) p_t) ?axes ~keepdims) inv_p
    | _ ->
        invalid_arg "norm: this combination of ord and axis not implemented"

  let rec slogdet a =
    let@ _ = span ~op:"slogdet" () in
    check_square ~op:"slogdet" a;
    check_float_or_complex ~op:"slogdet" a;
    let dtype_a = dtype a in
    let is_complex =
      Dtype.equal dtype_a Dtype.complex64
      || Dtype.equal dtype_a Dtype.complex128
    in
    let sh = shape a in
    let rank = Array.length sh in
    if (not is_complex) && sh.(rank - 1) = 2 && sh.(rank - 2) = 2 then
      (* 2x2 fast path *)
      let prefix = List.init (Stdlib.max 0 (rank - 2)) (fun _ -> A) in
      let a11 = slice_internal (prefix @ [ I 0; I 0 ]) a in
      let a12 = slice_internal (prefix @ [ I 0; I 1 ]) a in
      let a21 = slice_internal (prefix @ [ I 1; I 0 ]) a in
      let a22 = slice_internal (prefix @ [ I 1; I 1 ]) a in
      let det64 = sub (mul a11 a22) (mul a12 a21) |> cast Dtype.float64 in
      let z = zeros (B.context det64) Dtype.float64 (shape det64) in
      let sign_float =
        sub
          (cast Dtype.float32 (cast Dtype.float64 (greater det64 z)))
          (cast Dtype.float32 (cast Dtype.float64 (less det64 z)))
      in
      let abs_det = abs det64 in
      let logdet =
        cast Dtype.float32
          (where (cmpeq abs_det z)
             (full (B.context det64) Dtype.float64 (shape det64)
                Float.neg_infinity)
             (log abs_det))
      in
      sign_float, logdet
    else
      let _q, r = B.qr ~reduced:false a in
      let r_diag = diagonal r in
      let sign_det =
        let signs = sign r_diag in
        if ndim signs > 1 then prod signs ~axes:[ -1 ] ~keepdims:false
        else prod signs
      in
      let sign_float = cast Dtype.float32 (cast Dtype.float64 sign_det) in
      let abs_f64 = cast Dtype.float64 (abs r_diag) in
      let z = zeros (B.context abs_f64) Dtype.float64 (shape abs_f64) in
      let log_abs =
        where (cmpeq abs_f64 z)
          (full (B.context abs_f64) Dtype.float64 (shape abs_f64)
             Float.neg_infinity)
          (log abs_f64)
      in
      let logdet64 =
        if ndim log_abs > 1 then sum log_abs ~axes:[ -1 ] ~keepdims:false
        else sum log_abs
      in
      sign_float, cast Dtype.float32 logdet64

  and det a =
    let@ _ = span ~op:"det" () in
    check_square ~op:"det" a;
    check_float_or_complex ~op:"det" a;
    let sign, logabs = slogdet a in
    mul (cast (dtype a) sign) (exp logabs |> cast (dtype a))

  let matrix_rank ?tol ?rtol ?hermitian a =
    let@ _ = span ~op:"matrix_rank" () in
    check_float_or_complex ~op:"matrix_rank" a;
    let s =
      match hermitian with
      | Some true -> abs (fst (B.eigh ~vectors:false a))
      | _ -> svdvals a
    in
    let max_s = max s |> unsafe_get [] in
    let sh = shape a in
    let m = sh.(Array.length sh - 2) in
    let n = sh.(Array.length sh - 1) in
    let eps =
      let dt = dtype a in
      if Dtype.equal dt Dtype.float32 || Dtype.equal dt Dtype.complex64
      then 1.2e-7
      else if Dtype.equal dt Dtype.float64 || Dtype.equal dt Dtype.complex128
      then 2.2e-16
      else 1e-15
    in
    let tol =
      match tol, rtol with
      | Some t, _ -> t
      | None, Some r -> r *. max_s
      | None, None -> float_of_int (Stdlib.max m n) *. eps *. max_s
    in
    let mask = greater s (scalar (B.context a) (dtype s) tol) in
    int_of_float (Float.round (sum (cast (dtype s) mask) |> unsafe_get []))

  let trace ?out ?offset a =
    let@ _ = span ~op:"trace" () in
    if ndim a < 2 then
      invalid_arg "trace: input requires at least 2D array";
    sum ?out (diagonal ~offset:(Option.value offset ~default:0) a)
      ~axes:[ -1 ] ~keepdims:false

  let solve a b =
    let@ _ = span ~op:"solve" () in
    check_square ~op:"solve" a;
    check_float_or_complex ~op:"solve" a;
    check_float_or_complex ~op:"solve" b;
    let b_expanded =
      if ndim a > 2 && ndim b = 2 then
        let sa = shape a in
        let sb = shape b in
        let batch = array_prod (Array.sub sa 0 (ndim a - 2)) in
        if sb.(0) = batch && sb.(1) = sa.(ndim a - 2) then
          expand_dims [ -1 ] b
        else b
      else b
    in
    let q, r = B.qr ~reduced:true a in
    let r_diag = diagonal r |> cast Dtype.float64 in
    let m = dim (-2) a in
    let eps = if Dtype.equal (dtype a) Dtype.float32 then 1e-6 else 1e-12 in
    let tol_t =
      full (B.context r_diag) Dtype.float64 (shape r_diag)
        (eps *. float_of_int m)
    in
    if sum (cast Dtype.float64 (less (abs r_diag) tol_t)) |> unsafe_get [] > 0.
    then invalid_arg "solve: matrix is singular";
    let y = matmul (matrix_transpose q) b_expanded in
    let result =
      B.triangular_solve ~upper:true ~transpose:false ~unit_diag:false r y
    in
    if b_expanded != b then squeeze ~axes:[ ndim result - 1 ] result
    else result

  let pinv (type a b) ?rtol ?hermitian (a : (a, b) t) =
    let@ _ = span ~op:"pinv" () in
    check_float_or_complex ~op:"pinv" a;
    let sh = shape a in
    let m = sh.(Array.length sh - 2) in
    let n = sh.(Array.length sh - 1) in
    let dtype_a = dtype a in
    let eps =
      if Dtype.equal dtype_a Dtype.float32
         || Dtype.equal dtype_a Dtype.complex64 then 1.2e-7
      else if Dtype.equal dtype_a Dtype.float64
              || Dtype.equal dtype_a Dtype.complex128 then 2.2e-16
      else 1e-15
    in
    let max_dim = float_of_int (Stdlib.max m n) in
    let cutoff ~max_s =
      match rtol with
      | Some r -> r *. max_s *. max_dim
      | None -> max_dim *. eps *. max_s
    in
    let pinv_from_factors u s vh =
      let max_s = max s |> unsafe_get [] in
      let cutoff = cutoff ~max_s in
      let ones_s = ones (B.context s) (dtype s) (shape s) in
      let threshold = scalar (B.context s) (dtype s) cutoff in
      let mask = greater s threshold in
      let s_inv =
        mul (div ones_s (where mask s ones_s)) (cast (dtype s) mask)
        |> cast dtype_a
      in
      let v = matrix_transpose vh in
      let vs = mul v (unsqueeze ~axes:[ 0 ] s_inv) in
      if Dtype.is_complex dtype_a then
        matmul vs (matrix_transpose (conjugate u))
      else matmul vs (matrix_transpose u)
    in
    let pinv_via_svd () =
      let u, s, vh = B.svd ~full_matrices:false a in
      pinv_from_factors u s vh
    in
    match hermitian with
    | Some true -> (
        match B.eigh ~vectors:true a with
        | vals, Some vecs ->
            let abs_vals = abs vals in
            let sign_vals = sign vals in
            let o = ones (B.context vals) (dtype vals) (shape vals) in
            let z = zeros (B.context vals) (dtype vals) (shape vals) in
            let sign_fixed = where (cmpeq sign_vals z) o sign_vals in
            let vh =
              mul (expand_dims [ -1 ] (cast dtype_a sign_fixed))
                (matrix_transpose vecs)
            in
            pinv_from_factors vecs abs_vals vh
        | _ -> pinv_via_svd ())
    | _ -> pinv_via_svd ()

  let lstsq ?rcond a b =
    let@ _ = span ~op:"lstsq" () in
    check_float_or_complex ~op:"lstsq" a;
    check_float_or_complex ~op:"lstsq" b;
    let sh = shape a in
    let m = sh.(Array.length sh - 2) in
    let n = sh.(Array.length sh - 1) in
    let rcond_value =
      match rcond with
      | Some v -> v
      | None ->
          let eps =
            if Dtype.equal (dtype a) Dtype.float32 then 1.2e-7
            else if Dtype.equal (dtype a) Dtype.float64 then 2.2e-16
            else 1e-15
          in
          float_of_int (Stdlib.max m n) *. eps
          *. (max (svdvals a) |> unsafe_get [])
    in
    let x =
      if m >= n then
        let q, r = B.qr ~reduced:true a in
        let y = matmul (matrix_transpose q) b in
        let r_sq =
          if ndim r = 2 then slice_internal [ R (0, n); R (0, n) ] r
          else slice_internal [ A; R (0, n); R (0, n) ] r
        in
        let y_top =
          if ndim y = 2 then slice_internal [ R (0, n); A ] y
          else if ndim y = 1 then slice_internal [ R (0, n) ] y
          else slice_internal [ A; R (0, n); A ] y
        in
        B.triangular_solve ~upper:true ~transpose:false ~unit_diag:false
          r_sq y_top
      else matmul (pinv a ~rtol:rcond_value) b
    in
    let residuals =
      if m > n then
        let res = sub b (matmul a x) in
        sum (square res) ~axes:[ ndim res - 2 ] ~keepdims:false
      else zeros (B.context a) (dtype b) [||]
    in
    x, residuals, matrix_rank a, svdvals a

  let inv a =
    let@ _ = span ~op:"inv" () in
    check_square ~op:"inv" a;
    check_float_or_complex ~op:"inv" a;
    let sh = shape a in
    let n = sh.(Array.length sh - 1) in
    let batch = Array.sub sh 0 (Array.length sh - 2) in
    let i =
      broadcast_to (Array.append batch [| n; n |])
        (eye (B.context a) (dtype a) n)
    in
    try solve a i
    with Invalid_argument msg when String.sub msg 0 5 = "solve" ->
      invalid_arg ("inv" ^ String.sub msg 5 (String.length msg - 5))

  let matrix_power a n =
    let@ _ = span ~op:"matrix_power" () in
    let sh = shape a in
    let rank = Array.length sh in
    if rank < 2 then
      invalid_arg "matrix_power: input requires at least 2D array";
    if sh.(rank - 2) <> sh.(rank - 1) then
      err "matrix_power" "matrix must be square, got %dx%d"
        sh.(rank - 2) sh.(rank - 1);
    let rec power acc base exp =
      if exp = 0 then acc
      else if exp mod 2 = 0 then power acc (matmul base base) (exp / 2)
      else power (matmul acc base) (matmul base base) (exp / 2)
    in
    if n = 0 then eye (B.context a) (dtype a) sh.(rank - 1)
    else if n > 0 then power a a (n - 1)
    else
      try
        let ia = inv a in
        if -n = 1 then ia else power ia ia (-n - 1)
      with Invalid_argument _ ->
        invalid_arg "matrix_power: singular for negative exponent"

  let cond ?p x =
    check_square ~op:"cond" x;
    check_float_or_complex ~op:"cond" x;
    match p with
    | None | Some `Two ->
        let s = svdvals x in
        let ds = dtype s in
        let mx = max s in
        let max_v = mx |> unsafe_get [] in
        let eps =
          if Dtype.equal ds Dtype.float32 then 1.2e-7
          else if Dtype.equal ds Dtype.float64 then 2.2e-16
          else 1e-15
        in
        let tol_t = scalar (B.context x) ds (eps *. max_v) in
        let safe_s = where (greater s tol_t) s tol_t in
        let mn =
          if ndim safe_s > 1 then min safe_s ~axes:[ -1 ] ~keepdims:false
          else min safe_s
        in
        cast (dtype x) (div mx mn)
    | Some `One ->
        mul (norm ~ord:`One x) (norm ~ord:`One (inv x))
    | Some `Inf ->
        mul (norm ~ord:`Inf x) (norm ~ord:`Inf (inv x))
    | _ -> invalid_arg "cond: unsupported norm"

  let tensorsolve ?axes a b =
    check_float_or_complex ~op:"tensorsolve" a;
    check_float_or_complex ~op:"tensorsolve" b;
    let sa = shape a in
    let sb = shape b in
    let ra = Array.length sa in
    let rb = Array.length sb in
    if rb = 0 then
      invalid_arg "tensorsolve: b must have at least one dimension";
    if ra < rb then
      invalid_arg "tensorsolve: a, rank must be >= rank of b";
    let axes_for_b =
      match axes with
      | None -> Array.init rb (fun i -> ra - rb + i)
      | Some axes ->
          if List.length axes <> rb then
            err "tensorsolve" "axes, expected %d entries, got %d" rb
              (List.length axes);
          let seen = Array.make ra false in
          Array.map
            (fun ax ->
              let axis = if ax < 0 then ax + ra else ax in
              if axis < 0 || axis >= ra then
                err "tensorsolve" "axis %d out of bounds for %dD tensor" ax ra;
              if seen.(axis) then
                err "tensorsolve" "axis %d, repeated" ax;
              seen.(axis) <- true;
              axis)
            (Array.of_list axes)
    in
    let selected = Array.make ra false in
    Array.iter (fun ax -> selected.(ax) <- true) axes_for_b;
    let free =
      Array.of_list
        (List.filter (fun ax -> not selected.(ax))
           (List.init ra Fun.id))
    in
    let perm = Array.append free axes_for_b in
    let a_perm =
      let rec is_id i =
        if i = ra then true
        else if perm.(i) <> i then false
        else is_id (i + 1)
      in
      if is_id 0 then a else transpose ~axes:(Array.to_list perm) a
    in
    let ps = shape a_perm in
    let nf = Array.length free in
    let free_shape = Array.sub ps 0 nf in
    let rhs_shape = Array.sub ps nf rb in
    if rhs_shape <> sb then
      err "tensorsolve" "cannot reshape %s to %s" (Shape.to_string rhs_shape) (Shape.to_string sb);
    let rows = array_prod free_shape in
    let cols = array_prod rhs_shape in
    if rows <> cols then
      invalid_arg "tensorsolve: a, leading dimensions must match trailing dimensions";
    let a_mat = reshape [| rows; cols |] a_perm in
    let b_vec = reshape [| rows |] b in
    let solution =
      try solve a_mat b_vec
      with Invalid_argument _ ->
        let x_col = matmul (pinv a_mat) (reshape [| rows; 1 |] b_vec) in
        reshape [| cols |] x_col
    in
    reshape free_shape solution

  let tensorinv ?ind a =
    check_float_or_complex ~op:"tensorinv" a;
    let sh = shape a in
    let rank = Array.length sh in
    if rank = 0 then
      invalid_arg "tensorinv: input must have at least one dimension";
    let ind = Option.value ind ~default:(rank / 2) in
    if ind <= 0 || ind >= rank then
      invalid_arg "tensorinv: ind must split dimensions into two non-empty groups";
    let left = Array.sub sh 0 ind in
    let right = Array.sub sh ind (rank - ind) in
    let ls = array_prod left in
    let rs = array_prod right in
    if ls <> rs then
      invalid_arg "tensorinv: input, leading and trailing dimensions must have equal product";
    let inv_mat =
      try inv (reshape [| ls; rs |] a)
      with Invalid_argument _ -> pinv (reshape [| ls; rs |] a)
    in
    reshape (Array.append right left) inv_mat

  (*---------------------------------------------------------------------------
     FFT
    ---------------------------------------------------------------------------*)

  type fft_norm = [ `Backward | `Forward | `Ortho ]

  let pad_or_truncate_for_fft x axes s =
    match s with
    | None -> x
    | Some sizes ->
        let s_arr = Array.of_list sizes in
        let acc = ref x in
        List.iteri
          (fun i ax ->
            let ax = if ax < 0 then ndim !acc + ax else ax in
            let cur = dim ax !acc in
            let target = s_arr.(i) in
            if target > cur then (
              let pad_config = Array.make (ndim !acc) (0, 0) in
              pad_config.(ax) <- (0, target - cur);
              acc := B.pad !acc pad_config (Dtype.zero (dtype !acc)))
            else if target < cur then
              acc :=
                B.shrink !acc
                  (Array.init (ndim !acc) (fun idx ->
                       if idx = ax then (0, target) else (0, dim idx !acc))))
          axes;
        !acc

  let fft_norm_scale norm axes_list x =
    match norm with
    | `Backward -> 1.0
    | `Forward ->
        let n =
          List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_list
        in
        1.0 /. float_of_int n
    | `Ortho ->
        let n =
          List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_list
        in
        1.0 /. Stdlib.sqrt (float_of_int n)

  let apply_fft_scale (type a) ?out scale (result : (Complex.t, a) t)
      : (Complex.t, a) t =
    if scale <> 1.0 then
      let sv =
        match B.dtype result with
        | Complex64 | Complex128 -> Complex.{ re = scale; im = 0.0 }
      in
      mul ?out result (scalar (B.context result) (B.dtype result) sv)
    else copy_to_out ?out result

  let fftn (type a) ?out ?axes ?s ?(norm = `Backward) (x : (Complex.t, a) t)
      : (Complex.t, a) t =
    let@ _ = span ~op:"fftn" () in
    let nd = ndim x in
    let axes_list =
      match axes with
      | None -> List.init nd Fun.id
      | Some a -> List.map (fun ax -> if ax < 0 then nd + ax else ax) a
    in
    (match s with
     | Some sizes when List.length sizes <> List.length axes_list ->
         invalid_arg "fft: s parameter must have same length as axes"
     | _ -> ());
    let xp = pad_or_truncate_for_fft x axes_list s in
    let scale = fft_norm_scale norm axes_list xp in
    let r = B.fft xp ~axes:(Array.of_list axes_list) in
    apply_fft_scale ?out scale r

  let ifftn (type a) ?out ?axes ?s ?(norm = `Backward) (x : (Complex.t, a) t)
      : (Complex.t, a) t =
    let@ _ = span ~op:"ifftn" () in
    let nd = ndim x in
    let axes_list =
      match axes with
      | None -> List.init nd Fun.id
      | Some a -> List.map (fun ax -> if ax < 0 then nd + ax else ax) a
    in
    (match s with
     | Some sizes when List.length sizes <> List.length axes_list ->
         invalid_arg "ifft: s parameter must have same length as axes"
     | _ -> ());
    let xi, norm_scale =
      match s with
      | None ->
          let scale =
            match norm with
            | `Backward ->
                let n =
                  List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_list
                in
                1.0 /. float_of_int n
            | `Forward -> 1.0
            | `Ortho ->
                let n =
                  List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_list
                in
                1.0 /. Stdlib.sqrt (float_of_int n)
          in
          x, scale
      | Some sizes ->
          let xp = pad_or_truncate_for_fft x axes_list s in
          let scale =
            match norm with
            | `Backward ->
                let n = List.fold_left ( * ) 1 sizes in
                1.0 /. float_of_int n
            | `Forward -> 1.0
            | `Ortho ->
                1.0 /. Stdlib.sqrt (float_of_int (List.fold_left ( * ) 1 sizes))
          in
          xp, scale
    in
    let r = B.ifft xi ~axes:(Array.of_list axes_list) in
    apply_fft_scale ?out norm_scale r

  let rfftn ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"rfftn" () in
    let nd = ndim x in
    let axes_list = match axes with None -> [ nd - 1 ] | Some ax -> ax in
    let xp = pad_or_truncate_for_fft x axes_list s in
    let scale = fft_norm_scale norm axes_list xp in
    let r = B.rfft xp ~dtype:Dtype.Complex128 ~axes:(Array.of_list axes_list) in
    if scale <> 1.0 then
      let sv = Complex.{ re = scale; im = 0.0 } in
      mul ?out r (scalar (B.context r) (B.dtype r) sv)
    else copy_to_out ?out r

  let irfftn ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"irfftn" () in
    let nd = ndim x in
    let axes_list = match axes with None -> [ nd - 1 ] | Some ax -> ax in
    let input_shape = shape x in
    let output_sizes =
      match s with
      | Some sizes -> sizes
      | None ->
          List.mapi
            (fun i axis ->
              let axis = if axis < 0 then nd + axis else axis in
              if i = List.length axes_list - 1 then
                (input_shape.(axis) - 1) * 2
              else input_shape.(axis))
            axes_list
    in
    let norm_sizes =
      List.mapi
        (fun i axis ->
          let axis = if axis < 0 then nd + axis else axis in
          if i = List.length axes_list - 1 then
            match s with
            | Some sizes -> List.nth sizes i
            | None -> (input_shape.(axis) - 1) * 2
          else
            match s with
            | Some sizes -> List.nth sizes i
            | None -> input_shape.(axis))
        axes_list
    in
    let norm_scale =
      match norm with
      | `Backward ->
          1.0 /. float_of_int (List.fold_left ( * ) 1 norm_sizes)
      | `Forward -> 1.0
      | `Ortho ->
          1.0 /. Stdlib.sqrt (float_of_int (List.fold_left ( * ) 1 norm_sizes))
    in
    let s_param =
      match s with None -> None | Some _ -> Some (Array.of_list output_sizes)
    in
    let r =
      B.irfft ?s:s_param x ~dtype:Dtype.Float64
        ~axes:(Array.of_list axes_list)
    in
    if norm_scale <> 1.0 then
      mul ?out r (scalar (B.context r) (B.dtype r) norm_scale)
    else copy_to_out ?out r

  (* 1D FFT convenience *)
  let fft ?out ?(axis = -1) ?n ?(norm = `Backward) x =
    let@ _ = span ~op:"fft" () in
    let s = match n with None -> None | Some sz -> Some [ sz ] in
    fftn ?out x ~axes:[ axis ] ?s ~norm

  let ifft ?out ?(axis = -1) ?n ?(norm = `Backward) x =
    let@ _ = span ~op:"ifft" () in
    let s = match n with None -> None | Some sz -> Some [ sz ] in
    ifftn ?out x ~axes:[ axis ] ?s ~norm

  let rfft ?out ?(axis = -1) ?n ?(norm = `Backward) x =
    let@ _ = span ~op:"rfft" () in
    let s = match n with None -> None | Some sz -> Some [ sz ] in
    rfftn ?out x ~axes:[ axis ] ?s ~norm

  let irfft ?out ?(axis = -1) ?n ?(norm = `Backward) x =
    let@ _ = span ~op:"irfft" () in
    let s = match n with None -> None | Some sz -> Some [ sz ] in
    irfftn ?out x ~axes:[ axis ] ?s ~norm

  (* 2D FFT *)

  let check_fft2 ~op x axes =
    let n = ndim x in
    if n < 2 then
      err op "input requires at least 2D array, got %dD" n;
    let axes_list =
      match axes with None -> [ n - 2; n - 1 ] | Some ax -> ax
    in
    if List.length axes_list <> 2 then
      err op "axes must specify exactly 2 axes";
    axes_list

  let fft2 ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"fft2" () in
    let axes_list = check_fft2 ~op:"fft2" x axes in
    fftn ?out x ~axes:axes_list ?s ~norm

  let ifft2 ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"ifft2" () in
    let axes_list = check_fft2 ~op:"ifft2" x axes in
    ifftn ?out x ~axes:axes_list ?s ~norm

  (* N-dimensional FFT public wrappers *)
  let fftn ?out ?axes ?s ?(norm = `Backward) x =
    fftn ?out x
      ~axes:(match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax)
      ?s ~norm

  let ifftn ?out ?axes ?s ?(norm = `Backward) x =
    ifftn ?out x
      ~axes:(match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax)
      ?s ~norm

  let rfft2 ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"rfft2" () in
    let axes_list = check_fft2 ~op:"rfft2" x axes in
    rfftn ?out x ~axes:axes_list ?s ~norm

  let irfft2 ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"irfft2" () in
    let axes_list = check_fft2 ~op:"irfft2" x axes in
    irfftn ?out x ~axes:axes_list ?s ~norm

  let rfftn ?out ?axes ?s ?(norm = `Backward) x =
    rfftn ?out x
      ~axes:(match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax)
      ?s ~norm

  let irfftn ?out ?axes ?s ?(norm = `Backward) x =
    irfftn ?out x
      ~axes:(match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax)
      ?s ~norm

  (* Hermitian FFT *)
  let hfft ?(axis = -1) ?n ?norm x =
    let n = match n with None -> 2 * (dim axis x - 1) | Some n -> n in
    let axis = resolve_single_axis x axis in
    irfftn x ~axes:[ axis ] ~s:[ n ] ?norm

  let ihfft ?(axis = -1) ?n ?norm x =
    let n = match n with None -> dim axis x | Some n -> n in
    let axis = resolve_single_axis x axis in
    rfftn x ~axes:[ axis ] ~s:[ n ] ?norm

  (* FFT helpers *)
  let fftfreq ctx ?(d = 1.0) n =
    let@ _ = span ~op:"fftfreq" () in
    let dt = Dtype.float64 in
    let v = 1.0 /. (float_of_int n *. d) in
    let freqs =
      if n mod 2 = 0 then
        concatenate ~axis:0
          [ cast dt (arange ctx Dtype.int32 0 (n / 2) 1);
            cast dt (arange ctx Dtype.int32 (-(n / 2)) 0 1) ]
      else
        concatenate ~axis:0
          [ cast dt (arange ctx Dtype.int32 0 ((n + 1) / 2) 1);
            cast dt (arange ctx Dtype.int32 (-((n - 1) / 2)) 0 1) ]
    in
    mul_s freqs v

  let rfftfreq ctx ?(d = 1.0) n =
    let@ _ = span ~op:"rfftfreq" () in
    let dt = Dtype.float64 in
    let v = 1.0 /. (float_of_int n *. d) in
    mul (cast dt (arange ctx Dtype.int32 0 ((n / 2) + 1) 1))
      (scalar ctx dt v)

  let fftshift ?axes x =
    let@ _ = span ~op:"fftshift" () in
    let sh = shape x in
    let axes_list =
      match axes with None -> List.init (Array.length sh) Fun.id | Some ax -> ax
    in
    List.fold_left
      (fun acc axis ->
        let axis = resolve_single_axis acc axis in
        roll (sh.(axis) / 2) acc ~axis)
      x axes_list

  let ifftshift ?axes x =
    let@ _ = span ~op:"ifftshift" () in
    let sh = shape x in
    let axes_list =
      match axes with None -> List.init (Array.length sh) Fun.id | Some ax -> ax
    in
    List.fold_left
      (fun acc axis ->
        let axis = resolve_single_axis acc axis in
        roll (-(sh.(axis) / 2)) acc ~axis)
      x axes_list

  (*---------------------------------------------------------------------------
     Neural Network Operations
    ---------------------------------------------------------------------------*)

  let softmax ?out ?(axes = [ -1 ]) ?(scale = 1.0) x =
    let@ _ = span ~op:"softmax" () in
    let nd = Array.length (shape x) in
    let axes_norm = List.map (fun ax -> if ax < 0 then nd + ax else ax) axes in
    let max_x = max x ~axes:axes_norm ~keepdims:true in
    let shifted =
      if scale = 1.0 then sub x max_x
      else mul (scalar_like x scale) (sub x max_x)
    in
    let e = exp shifted in
    div ?out e (sum e ~axes:axes_norm ~keepdims:true)

  let log_softmax ?out ?(axes = [ -1 ]) ?(scale = 1.0) x =
    let axes_norm = normalize_and_dedup_axes ~op:"log_softmax" (ndim x) axes in
    if axes_norm = [] then
      copy_to_out ?out (zeros_like x)
    else
      let max_x = max x ~axes:axes_norm ~keepdims:true in
      let shifted = sub x max_x in
      let scaled =
        if scale = 1.0 then shifted
        else mul (scalar_like shifted scale) shifted
      in
      let log_den = log (sum (exp scaled) ~axes:axes_norm ~keepdims:true) in
      sub ?out scaled log_den

  let logsumexp ?out ?axes ?(keepdims = false) x =
    let axes_norm =
      match axes with
      | None -> List.init (ndim x) Fun.id
      | Some lst -> normalize_and_dedup_axes ~op:"logsumexp" (ndim x) lst
    in
    if axes_norm = [] then copy_to_out ?out x
    else
      let max_x = max x ~axes:axes_norm ~keepdims:true in
      let log_sum = add (log (sum (exp (sub x max_x)) ~axes:axes_norm ~keepdims:true)) max_x in
      if keepdims then copy_to_out ?out log_sum
      else copy_to_out ?out (squeeze ~axes:(List.rev axes_norm) log_sum)

  let logmeanexp ?out ?axes ?(keepdims = false) x =
    let axes_norm =
      match axes with
      | None -> List.init (ndim x) Fun.id
      | Some lst -> normalize_and_dedup_axes ~op:"logmeanexp" (ndim x) lst
    in
    if axes_norm = [] then copy_to_out ?out x
    else
      let log_sum = logsumexp ~axes:axes_norm ~keepdims:true x in
      let count = List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_norm in
      let log_mean =
        sub log_sum (log (scalar_like log_sum (float_of_int count)))
      in
      if keepdims then copy_to_out ?out log_mean
      else copy_to_out ?out (squeeze ~axes:(List.rev axes_norm) log_mean)

  let standardize ?out ?axes ?mean:mean_param ?variance:variance_param
      ?(epsilon = 1e-5) x =
    let nd = ndim x in
    let axes_norm =
      match axes with
      | None -> List.init nd Fun.id
      | Some lst -> normalize_and_dedup_axes ~op:"standardize" nd lst
    in
    let x_shape = shape x in
    let keep_shape =
      Array.mapi
        (fun idx d -> if List.exists (( = ) idx) axes_norm then 1 else d)
        x_shape
    in
    let unaffected =
      List.filter
        (fun idx -> not (List.exists (( = ) idx) axes_norm))
        (List.init nd Fun.id)
    in
    let core_shape =
      Array.of_list (List.map (fun idx -> x_shape.(idx)) unaffected)
    in
    let broadcast_param name param =
      let ps = shape param in
      if ps = x_shape || ps = keep_shape then param
      else if ps = core_shape then reshape keep_shape param
      else
        err "standardize" "%s, shape must match normalized axes" name
    in
    let mean_tensor =
      match mean_param with
      | Some m -> broadcast_param "mean" m
      | None ->
          if axes_norm = [] then x else mean x ~axes:axes_norm ~keepdims:true
    in
    let variance_tensor =
      match variance_param with
      | Some v -> broadcast_param "variance" v
      | None ->
          if axes_norm = [] then zeros_like x
          else var x ~axes:axes_norm ~keepdims:true
    in
    div ?out (sub x mean_tensor)
      (sqrt (add variance_tensor (scalar_like x epsilon)))

  let erf ?out x = unaryop ~op_name:"erf" ?out B.erf x

  let extract_patches ~kernel_size ~stride ~dilation ~padding x =
    B.unfold x ~kernel_size ~stride ~dilation ~padding

  let combine_patches ~output_size ~kernel_size ~stride ~dilation ~padding x =
    B.fold x ~output_size ~kernel_size ~stride ~dilation ~padding

  (* Correlation and convolution *)

  let correlate_padding ~mode input_spatial k_shape =
    let k = Array.length k_shape in
    match mode with
    | `Valid -> Array.make k (0, 0)
    | `Full ->
        Array.init k (fun i ->
            let p = k_shape.(i) - 1 in
            (p, p))
    | `Same ->
        Array.init k (fun i ->
            let total = k_shape.(i) - 1 in
            let _ = input_spatial.(i) in
            (total / 2, total - (total / 2)))

  let correlate ?(padding = `Valid) x kernel =
    let kr = ndim kernel in
    let xr = ndim x in
    if xr < kr then
      err "correlate" "input rank %d < kernel rank %d" xr kr;
    let ks = shape kernel in
    let input_spatial = Array.sub (shape x) (xr - kr) kr in
    let pad_pairs = correlate_padding ~mode:padding input_spatial ks in
    let ones_arr = Array.make kr 1 in
    let x_unf = B.unfold x ~kernel_size:ks ~stride:ones_arr ~dilation:ones_arr ~padding:pad_pairs in
    let und = ndim x_unf in
    let kp = (shape x_unf).(und - 2) in
    let l = (shape x_unf).(und - 1) in
    let result =
      sum (mul x_unf (reshape [| kp; 1 |] kernel)) ~axes:[ und - 2 ]
    in
    let leading = Array.sub (shape x) 0 (xr - kr) in
    let out_spatial =
      Array.init kr (fun i ->
          input_spatial.(i) + fst pad_pairs.(i) + snd pad_pairs.(i)
          - ks.(i) + 1)
    in
    let _ = l in
    reshape (Array.concat [ leading; out_spatial ]) result

  let convolve ?(padding = `Valid) x kernel =
    correlate ~padding x (flip ~axes:(List.init (ndim kernel) Fun.id) kernel)

  (* Sliding window filters *)

  let sliding_filter ~reduce_fn ~kernel_size ?stride x =
    let kr = Array.length kernel_size in
    let stride = match stride with Some s -> s | None -> kernel_size in
    let ones_arr = Array.make kr 1 in
    let zeros_arr = Array.make kr (0, 0) in
    let x_unf =
      B.unfold x ~kernel_size ~stride ~dilation:ones_arr ~padding:zeros_arr
    in
    let und = ndim x_unf in
    let reduced = reduce_fn x_unf ~axes:[ und - 2 ] ~keepdims:false in
    let xr = ndim x in
    let leading = Array.sub (shape x) 0 (xr - kr) in
    let input_spatial = Array.sub (shape x) (xr - kr) kr in
    let out_spatial =
      Array.init kr (fun i ->
          ((input_spatial.(i) - kernel_size.(i)) / stride.(i)) + 1)
    in
    reshape (Array.concat [ leading; out_spatial ]) reduced

  let maximum_filter ~kernel_size ?stride x =
    sliding_filter
      ~reduce_fn:(fun x ~axes ~keepdims -> max x ~axes ~keepdims)
      ~kernel_size ?stride x

  let minimum_filter ~kernel_size ?stride x =
    sliding_filter
      ~reduce_fn:(fun x ~axes ~keepdims -> min x ~axes ~keepdims)
      ~kernel_size ?stride x

  let uniform_filter ~kernel_size ?stride x =
    sliding_filter
      ~reduce_fn:(fun x ~axes ~keepdims:_ -> mean x ~axes)
      ~kernel_size ?stride x

  let one_hot ~num_classes index_tensor =
    let dt = dtype index_tensor in
    if not (Dtype.is_int dt || Dtype.is_uint dt) then
      err "one_hot" "dtype %s, indices must be integer type" (Dtype.to_string dt);
    let idx_exp = unsqueeze index_tensor ~axes:[ ndim index_tensor ] in
    let nd_exp = ndim idx_exp in
    let s = Array.make nd_exp 1 in
    s.(nd_exp - 1) <- num_classes;
    let arange_b = reshape s (arange (B.context index_tensor) dt 0 num_classes 1) in
    cast Dtype.uint8 (cmpeq idx_exp arange_b)

  (* ───── Display and Formatting ───── *)

  let pp_data (type a b) fmt (x : (a, b) t) =
    let open Format in
    let view = B.view x in
    let buffer = B.to_host x in
    let dtype = dtype x in
    let shape =
      match Symbolic_shape.eval (View.shape view) with
      | Some arr -> arr
      | None ->
          invalid_arg "pp_data: cannot print tensor with symbolic shape"
    in
    let ndim = Array.length shape in
    let sz =
      match Symbolic_shape.eval_dim (View.numel view) with
      | Some n -> n
      | None ->
          invalid_arg "pp_data: cannot print tensor with symbolic size"
    in
    let pp_element fmt (elt : a) =
      match dtype with
      | Float16 -> fprintf fmt "%g" elt
      | Float32 -> fprintf fmt "%g" elt
      | Float64 -> fprintf fmt "%g" elt
      | BFloat16 -> fprintf fmt "%g" elt
      | Float8_e4m3 -> fprintf fmt "%g" elt
      | Float8_e5m2 -> fprintf fmt "%g" elt
      | Int8 -> fprintf fmt "%d" elt
      | Int16 -> fprintf fmt "%d" elt
      | Int32 -> fprintf fmt "%ld" elt
      | Int64 -> fprintf fmt "%Ld" elt
      | UInt8 -> fprintf fmt "%d" elt
      | UInt16 -> fprintf fmt "%d" elt
      | UInt32 -> fprintf fmt "%ld" elt
      | UInt64 -> fprintf fmt "%Ld" elt
      | Int4 -> fprintf fmt "%d" elt
      | UInt4 -> fprintf fmt "%d" elt
      | Bool -> fprintf fmt "%b" elt
      | Complex64 -> fprintf fmt "(%g+%gi)" elt.re elt.im
      | Complex128 -> fprintf fmt "(%g+%gi)" elt.re elt.im
    in
    if sz = 0 && ndim > 0 then fprintf fmt "[]"
    else if ndim = 0 then
      if sz > 0 then
        pp_element fmt (Nx_buffer.unsafe_get buffer (View.offset view))
      else fprintf fmt "<empty scalar>"
    else
      let rec pp_slice fmt indices =
        let depth = List.length indices in
        if depth = ndim then
          let md_index = Array.of_list indices in
          let strides =
            match View.strides_opt view with
            | Some s -> s
            | None ->
                invalid_arg "pp_data: cannot print non-contiguous symbolic tensor"
          in
          let offset = Shape.ravel_index md_index strides + View.offset view in
          if offset < 0 || offset >= Nx_buffer.length buffer then
            fprintf fmt "<OOB:%d/%d>" offset (Nx_buffer.length buffer)
          else pp_element fmt (Nx_buffer.unsafe_get buffer offset)
        else
          let axis = depth in
          let dim_size = shape.(axis) in
          fprintf fmt "[";
          if dim_size > 0 then (
            if axis < ndim - 1 then pp_open_vbox fmt 0 else pp_open_hbox fmt ();
            for i = 0 to dim_size - 1 do
              if i > 0 then (
                fprintf fmt ",";
                if axis = ndim - 1 then fprintf fmt " "
                else pp_print_cut fmt ());
              pp_slice fmt (indices @ [ i ])
            done;
            pp_close_box fmt ());
          fprintf fmt "]"
      in
      if sz > 0 then pp_slice fmt [] else fprintf fmt "[]"

  let format_to_string pp x =
    let buf = Stdlib.Buffer.create 1024 in
    let fmt = Format.formatter_of_buffer buf in
    pp fmt x;
    Format.pp_print_flush fmt ();
    Stdlib.Buffer.contents buf

  let print_with_formatter pp x =
    pp Format.std_formatter x;
    Format.pp_print_newline Format.std_formatter ();
    Format.pp_print_flush Format.std_formatter ()

  let data_to_string x = format_to_string pp_data x
  let print_data x = print_with_formatter pp_data x
  let pp_dtype fmt dtype = Format.fprintf fmt "%s" (Dtype.to_string dtype)
  let dtype_to_string dtype = Dtype.to_string dtype

  let shape_to_string shape =
    Printf.sprintf "[%s]"
      (Array.map string_of_int shape |> Array.to_list |> String.concat "x")

  let pp_shape fmt shape = Format.fprintf fmt "%s" (shape_to_string shape)

  let pp fmt x =
    let open Format in
    let view = B.view x in
    fprintf fmt "@[<v 0>";
    fprintf fmt "Nx Info:@,";
    fprintf fmt "  Shape: %s@," (Symbolic_shape.to_string (View.shape view));
    fprintf fmt "  Dtype: %a@," pp_dtype (dtype x);
    fprintf fmt "  Strides: %s@,"
      (match View.strides_opt view with
       | Some s ->
           "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s))
           ^ "]"
       | None -> "<symbolic>");
    fprintf fmt "  Offset: %d@," (View.offset view);
    fprintf fmt "  Size: %s@,"
      (match Symbolic_shape.eval_dim (View.numel view) with
       | Some n -> string_of_int n
       | None -> "<symbolic>");
    fprintf fmt "  Data: %a@," pp_data x

  let print x = print_with_formatter pp x
  let to_string x = format_to_string pp x

  (* ───── Higher-order Functions ───── *)

  let map_item f x =
    let dt = dtype x in
    let sh = shape x in
    let result = empty (B.context x) dt sh in
    let src = data (contiguous x) in
    let dst = data result in
    let sz = size x in
    for i = 0 to sz - 1 do
      Nx_buffer.unsafe_set dst i (f (Nx_buffer.unsafe_get src i))
    done;
    result

  let iter_item f x =
    let src = data (contiguous x) in
    let sz = size x in
    for i = 0 to sz - 1 do
      f (Nx_buffer.unsafe_get src i)
    done

  let fold_item f init x =
    let src = data (contiguous x) in
    let sz = size x in
    let acc = ref init in
    for i = 0 to sz - 1 do
      acc := f !acc (Nx_buffer.unsafe_get src i)
    done;
    !acc

  let map f x =
    let dt = dtype x in
    let sh = shape x in
    let result = empty (B.context x) dt sh in
    let total = size x in
    for i = 0 to total - 1 do
      let idx = Shape.unravel_index i sh |> Array.to_list in
      set idx result (f (get idx x))
    done;
    result

  let iter f x =
    let sh = shape x in
    let total = size x in
    for i = 0 to total - 1 do
      f (get (Shape.unravel_index i sh |> Array.to_list) x)
    done

  let fold f init x =
    let sh = shape x in
    let total = size x in
    let acc = ref init in
    for i = 0 to total - 1 do
      acc := f !acc (get (Shape.unravel_index i sh |> Array.to_list) x)
    done;
    !acc

  (* ───── Infix Operators ───── *)

  module Infix = struct
    let ( + ) a b = add a b
    let ( +$ ) a s = add_s a s
    let ( - ) a b = sub a b
    let ( -$ ) a s = sub_s a s
    let ( * ) a b = mul a b
    let ( *$ ) a s = mul_s a s
    let ( / ) a b = div a b
    let ( /$ ) a s = div_s a s
    let ( ** ) a b = pow a b
    let ( **$ ) a s = pow_s a s
    let ( % ) a b = mod_ a b
    let ( mod ) a b = mod_ a b
    let ( %$ ) a s = mod_s a s
    let ( lxor ) a b = bitwise_xor a b
    let ( lor ) a b = bitwise_or a b
    let ( land ) a b = bitwise_and a b
    let ( ^ ) a b = logical_xor a b
    let ( && ) a b = logical_and a b
    let ( || ) a b = logical_or a b
    let ( ~- ) x = logical_not x
    let ( < ) a b = less a b
    let ( <$ ) a b = less_s a b
    let ( <> ) a b = not_equal a b
    let ( <>$ ) a b = not_equal_s a b
    let ( = ) a b = equal a b
    let ( =$ ) a b = equal_s a b
    let ( > ) a b = greater a b
    let ( >$ ) a b = greater_s a b
    let ( <= ) a b = less_equal a b
    let ( <=$ ) a b = less_equal_s a b
    let ( >= ) a b = greater_equal a b
    let ( >=$ ) a b = greater_equal_s a b
    let ( @@ ) a b = matmul a b
    let ( /@ ) = solve
    let ( **@ ) = matrix_power
    let ( <.> ) a b = dot a b
    let ( @= ) a b = concatenate ~axis:0 [ a; b ]
    let ( @|| ) a b = concatenate ~axis:1 [ a; b ]
    let ( .%{} ) x indices = get indices x
    let ( .%{}<- ) x indices value = set indices x value
    let ( .${} ) x slice_def = slice slice_def x
    let ( .${}<- ) x slice_def value = set_slice slice_def x value
  end
end
