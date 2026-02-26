(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* High-level tensor operations built on backend [B]. *)

module Make (B : Backend_intf.S) = struct
  module B = B

  let span ?attrs ~op () =
    let hook = !Instrumentation.current_hook in
    if hook.enabled then hook.with_span ~op ?attrs else fun f -> f ()

  let ( let@ ) m f = m f

  module Core_rng = Rng

  (* ───── Core Types and Context ───── *)

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

  (* Index type for tensor slicing *)
  type index =
    | I of int (* Single index *)
    | L of int list (* List of indices *)
    | R of int * int (* Range [start, stop) *)
    | Rs of int * int * int (* Range with step *)
    | A (* All indices *)
    | M of (bool, bool_elt) t (* Boolean mask *)
    | N (* New axis *)

  (* ───── Basic Tensor Properties ───── *)

  let data x = B.to_host x

  let shape x =
    let view = B.view x in
    match Symbolic_shape.eval (View.shape view) with
    | Some arr -> arr
    | None ->
        Error.failed ~op:"shape"
          ~what:"cannot get shape with unbound symbolic dimensions" ()

  let shape_symbolic x =
    let view = B.view x in
    View.shape view

  let dtype x = B.dtype x
  let itemsize x = Dtype.itemsize (B.dtype x)

  let strides x =
    let view = B.view x in
    let itemsize = itemsize x in

    (* Use high-level API instead of accessing internals *)
    match View.strides_opt view with
    | None ->
        let reason =
          if not (View.is_materializable view) then
            "view has non-materializable layout"
          else if not (Symbolic_shape.is_static (View.shape view)) then
            "view has symbolic shape"
          else "view has complex striding pattern"
        in
        Error.failed ~op:"strides" ~what:reason
          ~hint:"call contiguous() to get a standard layout" ()
    | Some elem_strides -> Array.map (fun s -> s * itemsize) elem_strides

  let stride i x =
    let view = B.view x in
    let itemsize = itemsize x in

    (* Get strides if available *)
    match View.strides_opt view with
    | None ->
        Error.failed ~op:"stride"
          ~what:(Printf.sprintf "stride for dimension %d" i)
          ~reason:"tensor does not have defined strides"
          ~hint:"call contiguous() first or check has_strides()" ()
    | Some elem_strides ->
        let ndim = View.ndim view in
        let i = if i < 0 then i + ndim else i in
        if i < 0 || i >= ndim then
          Error.axis_out_of_bounds ~op:"stride" ~axis:i ~ndim ()
        else elem_strides.(i) * itemsize

  let dims x =
    let view = B.view x in
    let sym_shape = View.shape view in
    match Symbolic_shape.eval sym_shape with
    | Some arr -> arr
    | None ->
        Error.failed ~op:"dims"
          ~what:"cannot get dimensions with unbound symbolic values" ()

  let dim i x =
    let view = B.view x in
    let shape = View.shape view in
    let ndim = Symbolic_shape.rank shape in
    let i = if i < 0 then i + ndim else i in
    if i < 0 || i >= ndim then
      Error.axis_out_of_bounds ~op:"dim" ~axis:i ~ndim ()
    else
      match Symbolic_shape.eval_dim shape.(i) with
      | Some n -> n
      | None ->
          Error.failed ~op:"dim"
            ~what:"cannot get dimension with unbound symbolic value" ()

  let ndim x =
    let view = B.view x in
    View.ndim view

  let size x =
    let view = B.view x in
    match Symbolic_shape.eval_dim (View.numel view) with
    | Some n -> n
    | None ->
        Error.failed ~op:"size"
          ~what:"cannot get size of tensor with symbolic shape"
          ~hint:"bind symbolic dimensions first" ()

  let numel x = size x

  let numel_symbolic x =
    let view = B.view x in
    View.numel view

  let nbytes x =
    (* This might also need to handle symbolic case *)
    let itemsize = itemsize x in
    try numel x * itemsize
    with _ ->
      (* If numel fails due to symbolic shape, we might still compute symbolic
         nbytes *)
      Error.failed ~op:"nbytes" ~what:"cannot compute bytes for symbolic tensor"
        ()

  let offset x =
    let view = B.view x in
    View.offset view

  let is_c_contiguous x =
    let view = B.view x in
    View.is_c_contiguous view

  (* ───── Internal Utilities ───── *)

  (* Create a power of 2 for integer shift operations *)
  let power_of_two : type a b. (a, b) Dtype.t -> int -> a =
   fun dtype shift_val ->
    if shift_val < 0 then
      Error.check_bounds ~op:"power_of_two" ~name:"shift_val" ~value:shift_val
        ~min:0 ();
    match dtype with
    | Int8 | UInt8 | Int16 | UInt16 -> (
        let power = 1 lsl shift_val in
        match dtype with
        | Int8 -> power
        | UInt8 -> power land 0xFF
        | Int16 -> power
        | UInt16 -> power land 0xFFFF
        | _ -> Error.failed ~op:"power_of_two" ~what:"unreachable code path" ())
    | Int32 -> Int32.shift_left Int32.one shift_val
    | UInt32 -> Int32.shift_left Int32.one shift_val
    | Int64 -> Int64.shift_left Int64.one shift_val
    | UInt64 -> Int64.shift_left Int64.one shift_val
    | _ ->
        Error.invalid ~op:"power_of_two"
          ~what:(Printf.sprintf "dtype %s" (Dtype.to_string dtype))
          ~reason:"not an integer type"
          ~hint:
            "use Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, or UInt64"
          ()

  let array_prod arr = Array.fold_left ( * ) 1 arr
  let dims_equal a b = Symbolic_shape.equal [| a |] [| b |]

  let ensure_no_infer ~op shape =
    Array.iter
      (fun dim ->
        if Symbolic_shape.is_infer dim then
          Error.invalid ~op ~what:"target shape"
            ~reason:"cannot contain infer (-1) dimensions" ())
      shape

  (* Integer ceiling division: (a + b - 1) / b for integers a, b where b > 0. *)
  let ceildiv a b =
    Error.check_bounds ~op:"ceildiv" ~name:"divisor" ~value:b ~min:1 ();
    (a + b - 1) / b

  (* Type checking helpers *)
  let ensure_float_dtype fname x =
    if not (Dtype.is_float (dtype x)) then
      Error.invalid ~op:fname
        ~what:(Printf.sprintf "dtype %s" (Dtype.to_string (dtype x)))
        ~reason:"expected float type (Float16, Float32, or Float64)" ()

  let ensure_int_dtype fname x =
    if not (Dtype.is_int (dtype x)) then
      Error.invalid ~op:fname ~what:"dtype" ~reason:"must be an integer type" ()

  (* Helper to convert tuple to array *)
  let pair_to_array (a, b) = [| a; b |]

  let resolve_axis ?ndim_opt x (axis_opt : int option) =
    let ndim = match ndim_opt with Some n -> n | None -> ndim x in
    match axis_opt with
    | None -> Array.init ndim Fun.id (* all axes *)
    | Some a ->
        let resolved_a = if a < 0 then a + ndim else a in
        [| resolved_a |]

  let resolve_single_axis ?ndim_opt x axis : int =
    let ndim = match ndim_opt with Some n -> n | None -> ndim x in
    if axis < 0 then axis + ndim else axis

  let reshape_symbolic new_shape x =
    let current_shape = shape_symbolic x in
    let has_infer = Array.exists Symbolic_shape.is_infer new_shape in
    let resolved =
      if has_infer then
        match
          Symbolic_shape.resolve_reshape ~from_shape:current_shape
            ~to_shape:new_shape
        with
        | Some s -> s
        | None ->
            Error.failed ~op:"reshape"
              ~what:"cannot infer dimension for symbolic reshape"
              ~hint:"bind symbolic dimensions or avoid using -1" ()
      else new_shape
    in
    (match
       (Symbolic_shape.eval current_shape, Symbolic_shape.eval resolved)
     with
    | Some old_arr, Some new_arr ->
        let old_numel = array_prod old_arr in
        let new_numel = array_prod new_arr in
        if old_numel <> new_numel && old_numel <> 0 && new_numel <> 0 then
          Error.shape_mismatch ~op:"reshape" ~expected:new_arr ~actual:old_arr
            ()
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
            Error.invalid ~op:"reshape" ~what:"shape specification"
              ~reason:(Printf.sprintf "dimension %d < -1" dim)
              ()
          else Symbolic_shape.static dim)
        shape_spec
    in
    if !infer_count > 1 then
      Error.invalid ~op:"reshape" ~what:"shape specification"
        ~reason:"multiple -1 dimensions"
        ~hint:"can only specify one unknown dimension" ();
    reshape_symbolic target_shape x

  let broadcast_shapes shape_a shape_b =
    let rank_a = Symbolic_shape.rank shape_a in
    let rank_b = Symbolic_shape.rank shape_b in
    let rank_out = max rank_a rank_b in
    let static_one = Symbolic_shape.static 1 in
    let result = Array.make rank_out static_one in
    for i = 0 to rank_out - 1 do
      let idx_a = rank_a - rank_out + i in
      let idx_b = rank_b - rank_out + i in
      let dim_a = if idx_a >= 0 then shape_a.(idx_a) else static_one in
      let dim_b = if idx_b >= 0 then shape_b.(idx_b) else static_one in
      let chosen =
        if dims_equal dim_a dim_b then dim_a
        else
          let eval_a = Symbolic_shape.eval_dim dim_a in
          let eval_b = Symbolic_shape.eval_dim dim_b in
          match (eval_a, eval_b) with
          | Some a, Some b -> (
              if a = b then dim_a
              else if a = 1 then dim_b
              else if b = 1 then dim_a
              else
                match
                  (Symbolic_shape.eval shape_a, Symbolic_shape.eval shape_b)
                with
                | Some sa, Some sb ->
                    Error.broadcast_incompatible ~op:"broadcast" ~shape1:sa
                      ~shape2:sb ()
                | _ ->
                    Error.failed ~op:"broadcast"
                      ~what:
                        (Printf.sprintf "cannot broadcast dimensions %s and %s"
                           (Symbolic_shape.to_string [| dim_a |])
                           (Symbolic_shape.to_string [| dim_b |]))
                      ())
          | Some a, None ->
              if a = 1 then dim_b
              else
                Error.failed ~op:"broadcast"
                  ~what:
                    (Printf.sprintf "cannot broadcast dimension %s to %s"
                       (Symbolic_shape.to_string [| dim_a |])
                       (Symbolic_shape.to_string [| dim_b |]))
                  ~hint:"bind symbolic dimensions first" ()
          | None, Some b ->
              if b = 1 then dim_a
              else
                Error.failed ~op:"broadcast"
                  ~what:
                    (Printf.sprintf "cannot broadcast dimension %s to %s"
                       (Symbolic_shape.to_string [| dim_b |])
                       (Symbolic_shape.to_string [| dim_a |]))
                  ~hint:"bind symbolic dimensions first" ()
          | None, None ->
              Error.failed ~op:"broadcast"
                ~what:
                  (Printf.sprintf
                     "cannot broadcast symbolic dimensions %s and %s"
                     (Symbolic_shape.to_string [| dim_a |])
                     (Symbolic_shape.to_string [| dim_b |]))
                ~hint:"bind symbolic dimensions first" ()
      in
      result.(i) <- chosen
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
        Error.failed ~op:"broadcast_to"
          ~what:
            (Printf.sprintf
               "rank mismatch: source rank %d exceeds target rank %d (%s -> %s)"
               rank_current rank_target
               (Symbolic_shape.to_string current_shape)
               (Symbolic_shape.to_string target_shape))
          ~hint:"target shape must have at least as many dimensions as source"
          ()
      else
        let pad_count = rank_target - rank_current in
        let padded_shape =
          if pad_count <= 0 then current_shape
          else
            let arr = Array.make rank_target (Symbolic_shape.static 1) in
            Array.blit current_shape 0 arr pad_count rank_current;
            arr
        in
        for i = 0 to rank_target - 1 do
          let curr_dim = padded_shape.(i) in
          let target_dim = target_shape.(i) in
          if dims_equal curr_dim target_dim then ()
          else
            match Symbolic_shape.eval_dim curr_dim with
            | Some 1 -> ()
            | Some curr_val -> (
                match Symbolic_shape.eval_dim target_dim with
                | Some target_val when curr_val = target_val -> ()
                | Some _target_val ->
                    let shape_curr =
                      match Symbolic_shape.eval padded_shape with
                      | Some s -> s
                      | None ->
                          Array.init rank_target (fun j ->
                              match
                                Symbolic_shape.eval_dim padded_shape.(j)
                              with
                              | Some v -> v
                              | None -> -1)
                    in
                    let shape_target =
                      match Symbolic_shape.eval target_shape with
                      | Some s -> s
                      | None ->
                          Array.init rank_target (fun j ->
                              match
                                Symbolic_shape.eval_dim target_shape.(j)
                              with
                              | Some v -> v
                              | None -> -1)
                    in
                    Error.broadcast_incompatible ~op:"broadcast_to"
                      ~shape1:shape_curr ~shape2:shape_target ()
                | None ->
                    Error.failed ~op:"broadcast_to"
                      ~what:
                        (Printf.sprintf
                           "dimension %d is symbolic (%s) and cannot be \
                            broadcast to %s"
                           i
                           (Symbolic_shape.to_string [| curr_dim |])
                           (Symbolic_shape.to_string [| target_dim |]))
                      ())
            | None ->
                Error.failed ~op:"broadcast_to"
                  ~what:
                    (Printf.sprintf
                       "dimension %d is symbolic (%s) and cannot be broadcast \
                        to %s"
                       i
                       (Symbolic_shape.to_string [| curr_dim |])
                       (Symbolic_shape.to_string [| target_dim |]))
                  ~hint:"bind symbolic dimensions first" ()
        done;
        let x_aligned =
          if pad_count <= 0 then x else reshape_symbolic padded_shape x
        in
        if Symbolic_shape.equal (shape_symbolic x_aligned) target_shape then
          x_aligned
        else B.expand x_aligned target_shape

  (* reshape and expand [x] to [new_shape] following numpy-style rules *)
  let broadcast_to new_shape x =
    let@ _ = span ~op:"broadcast_to" () in
    let target_shape =
      Array.map
        (fun dim ->
          if dim < 0 then
            Error.invalid ~op:"broadcast_to" ~what:"target shape"
              ~reason:(Printf.sprintf "dimension %d < 0" dim)
              ()
          else Symbolic_shape.static dim)
        new_shape
    in
    broadcast_to_symbolic target_shape x

  (* return [x] and [y] broadcasted to a common shape *)
  let broadcasted ?(reverse = false) x y =
    let a, b = if reverse then (y, x) else (x, y) in
    let shape_a = shape_symbolic a in
    let shape_b = shape_symbolic b in
    let broadcast_shape = broadcast_shapes shape_a shape_b in
    let a_broad = broadcast_to_symbolic broadcast_shape a in
    let b_broad = broadcast_to_symbolic broadcast_shape b in
    (a_broad, b_broad)

  (* like [broadcast_to] but [-1] keeps the original dimension *)
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
            Error.invalid ~op:"expand"
              ~what:(Printf.sprintf "dimension %d" i)
              ~reason:(Printf.sprintf "negative size %d" spec_dim)
              ()
          else Symbolic_shape.static spec_dim)
    in
    broadcast_to_symbolic target_shape x

  let cast (type a b c d) (dt : (c, d) Dtype.t) (x : (a, b) t) : (c, d) t =
    match Dtype.equal_witness (dtype x) dt with
    | Some Equal ->
        (* Here the compiler now *knows* that [x] has type [(c,d) t], so this
           type-safe “no-op” copy type-checks. *)
        B.copy x
    | None ->
        let out = B.buffer (B.context x) dt (shape x) in
        B.cast ~out x;
        out

  let astype dt x = cast dt x

  (* ───── Tensor Creation ───── *)

  let contiguous x =
    let@ _ = span ~op:"contiguous" () in
    B.contiguous x

  let copy x =
    let@ _ = span ~op:"copy" () in
    B.copy x

  let blit src dst =
    let@ _ = span ~op:"blit" () in
    if shape src <> shape dst then
      Error.shape_mismatch ~op:"blit" ~expected:(shape dst) ~actual:(shape src)
        ~hint:"source and destination must have identical shapes" ();
    B.assign dst src

  let create ctx dtype shape arr =
    let@ _ = span ~op:"create" () in
    let n = Array.fold_left ( * ) 1 shape in
    if Array.length arr <> n then
      Error.invalid ~op:"create" ~what:"array size"
        ~reason:
          (Printf.sprintf "got %d elements, expected %d" (Array.length arr) n)
        ();

    (* Create bigarray buffer with proper dtype *)
    let kind = Dtype.to_buffer_kind dtype in
    let bigarray = Nx_buffer.create kind n in

    (* Copy data from OCaml array to buffer *)
    for i = 0 to n - 1 do
      Nx_buffer.unsafe_set bigarray i arr.(i)
    done;

    (* Create flat tensor and reshape if needed *)
    let tensor_1d = B.from_host ctx bigarray in
    if Array.length shape = 1 && shape.(0) = n then tensor_1d
    else B.reshape tensor_1d (Symbolic_shape.of_ints shape)

  let init ctx dtype shape f =
    let@ _ = span ~op:"init" () in
    let size = Array.fold_left ( * ) 1 shape in

    (* Helper to convert linear index to multi-dimensional indices *)
    let unravel_index idx shape =
      let ndim = Array.length shape in
      let indices = Array.make ndim 0 in
      let remaining = ref idx in
      for i = 0 to ndim - 1 do
        let stride =
          Array.fold_left ( * ) 1 (Array.sub shape (i + 1) (ndim - i - 1))
        in
        indices.(i) <- !remaining / stride;
        remaining := !remaining mod stride
      done;
      indices
    in

    (* Create OCaml array with values from f *)
    let arr = Array.init size (fun i -> f (unravel_index i shape)) in

    (* Use create to handle the conversion *)
    create ctx dtype shape arr

  let scalar ctx dt value =
    let@ _ = span ~op:"scalar" () in
    B.full ctx dt [||] value

  let scalar_like x_ref value = scalar (B.context x_ref) (B.dtype x_ref) value

  let ifill value x =
    let@ _ = span ~op:"ifill" () in
    let value_tensor = scalar_like x value in
    let value_broadcasted = broadcast_to (shape x) value_tensor in
    B.assign x value_broadcasted;
    x

  let fill value x =
    let@ _ = span ~op:"fill" () in
    let copied = B.copy x in
    ignore (ifill value copied);
    copied

  let empty ctx dtype shape_arr =
    let@ _ = span ~op:"empty" () in
    B.buffer ctx dtype shape_arr

  let zeros ctx dtype shape_arr =
    let@ _ = span ~op:"zeros" () in
    B.full ctx dtype shape_arr (Dtype.zero dtype)

  let ones ctx dtype shape_arr =
    let@ _ = span ~op:"ones" () in
    B.full ctx dtype shape_arr (Dtype.one dtype)

  let full ctx dt target_shape fill_value =
    let@ _ = span ~op:"full" () in
    B.full ctx dt target_shape fill_value

  (* Generic _like helper *)
  let create_like x_ref fill_fn =
    let dtype = B.dtype x_ref in
    let shape = shape x_ref in
    fill_fn (B.context x_ref) dtype shape

  let empty_like x_ref =
    let@ _ = span ~op:"empty_like" () in
    create_like x_ref empty

  let full_like x_ref fill_value =
    let@ _ = span ~op:"full_like" () in
    create_like x_ref (fun ctx dt sh -> full ctx dt sh fill_value)

  let zeros_like x =
    let@ _ = span ~op:"zeros_like" () in
    full_like x (Dtype.zero (B.dtype x))

  let ones_like x =
    let@ _ = span ~op:"ones_like" () in
    full_like x (Dtype.one (B.dtype x))

  (* ───── Tensor Conversion ───── *)

  let to_buffer x =
    let@ _ = span ~op:"to_buffer" () in
    let ensure_contiguous_size t =
      let t = if is_c_contiguous t && offset t = 0 then t else contiguous t in
      let buffer = data t in
      let buffer_elems = Nx_buffer.length buffer in
      if buffer_elems = numel t then t else copy t
    in
    data (ensure_contiguous_size x)

  let to_bigarray x =
    let@ _ = span ~op:"to_bigarray" () in
    let buf = to_buffer x in
    let _ = Dtype.to_bigarray_kind (B.dtype x) in
    let ga = Nx_buffer.to_genarray buf (shape x) in
    (Obj.magic ga : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t)

  let of_buffer ctx ~shape buf =
    let@ _ = span ~op:"of_buffer" () in
    let flat_tensor = B.from_host ctx buf in
    reshape shape flat_tensor

  let of_bigarray ctx ba =
    let@ _ = span ~op:"of_bigarray" () in
    let ga_ext : ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t =
      Obj.magic ba
    in
    let shape = Bigarray.Genarray.dims ga_ext in
    let buf = Nx_buffer.of_genarray ga_ext in
    of_buffer ctx ~shape buf

  let to_array x =
    let@ _ = span ~op:"to_array" () in
    let t_contiguous = contiguous x in
    let ba = data t_contiguous in
    let n = numel t_contiguous in
    Array.init n (fun i -> Nx_buffer.get ba i)

  (* ───── Element-wise Binary Operations ───── *)

  (* Binary operation with broadcasting and optional output buffer *)
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

  (* Comparison operation with broadcasting - returns bool tensor *)
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

  (* In-place binary operation *)
  let inplace_binop op target value =
    let value_broadcasted = broadcast_to (shape target) value in
    op ~out:target target value_broadcasted;
    target

  (* Addition *)
  let add ?out a b = binop ~op_name:"add" ?out B.add a b
  let add_s ?out t s = add ?out t (scalar_like t s)
  let radd_s ?out s t = add ?out (scalar_like t s) t
  let iadd target value = inplace_binop B.add target value
  let iadd_s t s = iadd t (scalar_like t s)

  (* Subtraction *)
  let sub ?out a b = binop ~op_name:"sub" ?out B.sub a b
  let sub_s ?out t s = sub ?out t (scalar_like t s)
  let rsub_s ?out s t = sub ?out (scalar_like t s) t
  let isub target value = inplace_binop B.sub target value
  let isub_s t s = isub t (scalar_like t s)

  (* Multiplication *)
  let mul ?out a b = binop ~op_name:"mul" ?out B.mul a b
  let mul_s ?out t s = mul ?out t (scalar_like t s)
  let rmul_s ?out s t = mul ?out (scalar_like t s) t
  let imul target value = inplace_binop B.mul target value
  let imul_s t s = imul t (scalar_like t s)

  (* Division: uses fdiv for float/complex, idiv for integers *)
  let div ?out a b =
    let@ _ = span ~op:"div" () in
    let dt = dtype a in
    if Dtype.is_float dt || Dtype.is_complex dt then binop ?out B.div a b
    else if Dtype.is_int dt || Dtype.is_uint dt then binop ?out B.div a b
    else failwith "Unsupported dtype for division"

  let div_s ?out t s = div ?out t (scalar_like t s)
  let rdiv_s ?out s t = div ?out (scalar_like t s) t

  let idiv target value =
    let dt = dtype target in
    if Dtype.is_float dt || Dtype.is_complex dt then
      inplace_binop B.div target value
    else if Dtype.is_int dt || Dtype.is_uint dt then
      inplace_binop B.div target value
    else
      Error.invalid ~op:"idiv"
        ~what:("dtype " ^ Dtype.to_string dt)
        ~reason:"not supported" ()

  let idiv_s t s = idiv t (scalar_like t s)

  (* Power *)
  let pow ?out a b = binop ~op_name:"pow" ?out B.pow a b
  let pow_s ?out t s = pow ?out t (scalar_like t s)
  let rpow_s ?out s t = pow ?out (scalar_like t s) t
  let ipow target value = inplace_binop B.pow target value
  let ipow_s t s = ipow t (scalar_like t s)

  (* Maximum *)
  let maximum ?out a b = binop ~op_name:"maximum" ?out B.max a b
  let maximum_s ?out t s = maximum ?out t (scalar_like t s)
  let rmaximum_s ?out s t = maximum ?out (scalar_like t s) t
  let imaximum target value = inplace_binop B.max target value
  let imaximum_s t s = imaximum t (scalar_like t s)

  (* Minimum *)
  let minimum ?out a b = binop ~op_name:"minimum" ?out B.min a b
  let minimum_s ?out t s = minimum ?out t (scalar_like t s)
  let rminimum_s ?out s t = minimum ?out (scalar_like t s) t
  let iminimum target value = inplace_binop B.min target value
  let iminimum_s t s = iminimum t (scalar_like t s)

  (* Modulo *)
  let mod_ ?out a b = binop ~op_name:"mod" ?out B.mod_ a b
  let mod_s ?out t s = mod_ ?out t (scalar_like t s)
  let rmod_s ?out s t = mod_ ?out (scalar_like t s) t
  let imod target value = inplace_binop B.mod_ target value
  let imod_s t s = imod t (scalar_like t s)

  (* Bitwise operations *)
  let bitwise_xor ?out a b = binop ~op_name:"bitwise_xor" ?out B.xor a b
  let bitwise_or ?out a b = binop ~op_name:"bitwise_or" ?out B.or_ a b
  let bitwise_and ?out a b = binop ~op_name:"bitwise_and" ?out B.and_ a b

  (* ───── Logical Operations ───── *)

  let logical_and ?out a b = binop ?out B.and_ a b
  let logical_or ?out a b = binop ?out B.or_ a b
  let logical_xor ?out a b = binop ?out B.xor a b

  let logical_not (type a b) ?out (x : (a, b) t) : (a, b) t =
    let dt = dtype x in
    let one = full (B.context x) dt (shape x) (Dtype.one dt) in
    match dt with
    | Dtype.UInt8 | Dtype.Bool | Dtype.UInt4 -> binop ?out B.xor x one
    | _ -> sub ?out one x

  (* ───── Comparison Operations ───── *)

  let cmpeq ?out a b = cmpop ~op_name:"equal" ?out B.cmpeq a b
  let cmpne ?out a b = cmpop ~op_name:"not_equal" ?out B.cmpne a b
  let cmplt ?out a b = cmpop ~op_name:"less" ?out B.cmplt a b
  let cmple ?out a b = cmpop ~op_name:"less_equal" ?out B.cmple a b
  let cmpgt ?out a b = cmplt ?out b a
  let cmpge ?out a b = cmple ?out b a

  (* Aliases *)
  let less = cmplt
  let less_equal = cmple
  let greater = cmpgt
  let greater_equal = cmpge
  let equal = cmpeq
  let not_equal = cmpne

  (* Scalar comparison operations *)
  let equal_s ?out a s = equal ?out a (scalar_like a s)
  let not_equal_s ?out a s = not_equal ?out a (scalar_like a s)
  let less_s ?out a s = less ?out a (scalar_like a s)
  let greater_s ?out a s = greater ?out a (scalar_like a s)
  let less_equal_s ?out a s = less_equal ?out a (scalar_like a s)
  let greater_equal_s ?out a s = greater_equal ?out a (scalar_like a s)

  (* ───── Element-wise Unary Operations ───── *)

  (* Unary operation with optional output buffer *)
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
    let minus_one_val = Dtype.minus_one dt in
    let minus_one_tensor = B.full (B.context x) dt [||] minus_one_val in
    let minus_one_b = broadcast_to (shape x) minus_one_tensor in
    binop ?out B.xor x minus_one_b

  let invert ?out x = bitwise_not ?out x

  (* Math functions - assume float inputs as per B.op signatures *)
  let sin ?out x = unaryop ~op_name:"sin" ?out B.sin x
  let cos ?out x = unaryop ~op_name:"cos" ?out B.cos x
  let sqrt ?out x = unaryop ~op_name:"sqrt" ?out B.sqrt x
  let recip ?out x = unaryop ~op_name:"recip" ?out B.recip x
  let log ?out x = unaryop ~op_name:"log" ?out B.log x
  let exp ?out x = unaryop ~op_name:"exp" ?out B.exp x
  let abs ?out x = unaryop ~op_name:"abs" ?out B.abs x

  (* log2(x) = log(x) / log(2) = log(x) * (1/log(2)) *)
  let log2 ?out x =
    let dt = dtype x in
    let inv_ln2_val = Dtype.of_float dt (1.0 /. Stdlib.log 2.0) in
    let inv_ln2 = B.full (B.context x) dt [||] inv_ln2_val in
    let inv_ln2_b = broadcast_to (shape x) inv_ln2 in
    mul ?out (log x) inv_ln2_b

  (* exp2(x) = exp(x * log(2)) *)
  let exp2 ?out x =
    let dt = dtype x in
    let ln2_val = Dtype.of_float dt (Stdlib.log 2.0) in
    let ln2 = B.full (B.context x) dt [||] ln2_val in
    let ln2_b = broadcast_to (shape x) ln2 in
    exp ?out (mul x ln2_b)

  let tan ?out x = unaryop ~op_name:"tan" ?out B.tan x
  let square ?out x = mul ?out x x
  let sign ?out x = unaryop ~op_name:"sign" ?out B.sign x

  (* Activations & related *)
  let relu ?out x = maximum ?out x (zeros_like x)
  (* equivalent to (x > 0).where(x, 0) *)

  let sigmoid ?out x =
    (* 1 / (1 + exp(-x)) = 1 / (1 + (exp2(-x / log(2)))) *)
    let dt = dtype x in
    let neg_one_over_log2 =
      B.full (B.context x) dt [||] (-1.0 /. Stdlib.log 2.0)
    in
    let one_x = ones_like x in
    let exp_term = exp2 (mul x neg_one_over_log2) in
    recip ?out (add one_x exp_term)

  let rsqrt ?out x = recip ?out (sqrt x)
  let asin ?out x = unaryop ~op_name:"asin" ?out B.asin x
  let acos ?out x = unaryop ~op_name:"acos" ?out B.acos x
  let atan ?out x = unaryop ~op_name:"atan" ?out B.atan x
  let sinh ?out x = unaryop ~op_name:"sinh" ?out B.sinh x
  let cosh ?out x = unaryop ~op_name:"cosh" ?out B.cosh x
  let tanh ?out x = unaryop ~op_name:"tanh" ?out B.tanh x

  let asinh ?out x =
    (* log(x + sqrt(x^2 + 1)) *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let x_squared = square x in
    let sqrt_term = sqrt (add x_squared one_x) in
    log ?out (add x sqrt_term)

  let acosh ?out x =
    (* log(x + sqrt(x^2 - 1)) *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let x_squared = square x in
    let sqrt_term = sqrt (sub x_squared one_x) in
    log ?out (add x sqrt_term)

  let atanh ?out x =
    (* log((1+x)/(1-x)) / 2 *)
    let dt = dtype x in
    let one_x = full (B.context x) dt (shape x) 1.0 in
    let two_x = full (B.context x) dt (shape x) 2.0 in
    let term_plus = add one_x x in
    let term_minus = sub one_x x in
    div ?out (log (div term_plus term_minus)) two_x

  (* Rounding, properties *)
  let trunc ?out x = unaryop ~op_name:"trunc" ?out B.trunc x
  let ceil ?out x = unaryop ~op_name:"ceil" ?out B.ceil x
  let floor ?out x = unaryop ~op_name:"floor" ?out B.floor x
  let round ?out x = unaryop ~op_name:"round" ?out B.round x

  let isinf ?out x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then
      let result = zeros (B.context x) Dtype.bool (shape x) in
      match out with
      | Some o ->
          B.add ~out:o result (zeros_like result);
          o
      | None -> result
    else
      let pos_inf_const = B.full (B.context x) dt [||] Float.infinity in
      let neg_inf_const = B.full (B.context x) dt [||] Float.neg_infinity in
      let is_pos_inf = cmpeq x (broadcast_to (shape x) pos_inf_const) in
      let is_neg_inf = cmpeq x (broadcast_to (shape x) neg_inf_const) in
      logical_or ?out is_pos_inf is_neg_inf

  let isnan ?out x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then
      let result = zeros (B.context x) Dtype.bool (shape x) in
      match out with
      | Some o ->
          B.add ~out:o result (zeros_like result);
          o
      | None -> result
    else cmpne ?out x x

  let isfinite ?out x =
    let dt = dtype x in
    if not (Dtype.is_float dt) then
      let result = ones (B.context x) Dtype.bool (shape x) in
      match out with
      | Some o ->
          B.add ~out:o result (zeros_like result);
          o
      | None -> result
    else logical_not ?out (logical_or (isinf x) (isnan x))

  let lerp ?out start_tensor end_tensor weight =
    let end_minus_start = sub end_tensor start_tensor in
    let weighted_diff = mul end_minus_start weight in
    add ?out start_tensor weighted_diff

  (* Scalar version of lerp weight *)
  let lerp_scalar_weight ?out start_tensor end_tensor weight_val =
    let dt = dtype start_tensor in
    let weight_tensor =
      full (B.context start_tensor) dt (shape start_tensor) weight_val
    in
    lerp ?out start_tensor end_tensor weight_tensor

  let lshift ?out x shift_val =
    let dt = dtype x in
    if not (Dtype.is_int dt) then
      Error.invalid ~op:"lshift"
        ~what:("dtype " ^ Dtype.to_string dt)
        ~reason:"expected integer type" ();

    if shift_val < 0 then
      Error.check_bounds ~op:"lshift" ~name:"shift_val" ~value:shift_val ~min:0
        ();

    if shift_val = 0 then
      match out with
      | Some o ->
          B.add ~out:o x (zeros_like x);
          o
      | None -> x
    else
      let factor_val = power_of_two dt shift_val in
      let factor_tensor = B.full (B.context x) dt [||] factor_val in
      let factor_b = broadcast_to (shape x) factor_tensor in
      mul ?out x factor_b

  let rshift ?out x shift_val =
    let dt = dtype x in
    if not (Dtype.is_int dt) then
      Error.invalid ~op:"rshift"
        ~what:("dtype " ^ Dtype.to_string dt)
        ~reason:"expected integer type" ();

    if shift_val < 0 then
      Error.check_bounds ~op:"rshift" ~name:"shift_val" ~value:shift_val ~min:0
        ();

    if shift_val = 0 then
      match out with
      | Some o ->
          B.add ~out:o x (zeros_like x);
          o
      | None -> x
    else
      let divisor_val = power_of_two dt shift_val in
      let divisor_tensor = B.full (B.context x) dt [||] divisor_val in
      let divisor_b = broadcast_to (shape x) divisor_tensor in
      binop ?out B.div x divisor_b

  let clamp ?out ?min ?max x =
    let x_clamped_min =
      match min with
      | None -> x
      | Some min_v ->
          let min_x = full_like x min_v in
          maximum x min_x
    in
    match max with
    | None -> (
        match out with
        | Some o ->
            B.add ~out:o x_clamped_min (zeros_like x_clamped_min);
            o
        | None -> x_clamped_min)
    | Some max_v ->
        let max_x = full_like x_clamped_min max_v in
        minimum ?out x_clamped_min max_x

  let clip = clamp

  (* ───── Ternary Operations ───── *)

  (* select between [if_true] and [if_false] based on [cond] *)
  let where ?out cond if_true if_false =
    let@ _ = span ~op:"where" () in
    let s_true = shape if_true in
    let s_false = shape if_false in
    let s_cond = shape cond in
    (* Broadcast all three to a common shape. Order matters for shape inference.
       First, find common shape for if_true and if_false. *)
    let target_data_shape = Shape.broadcast s_true s_false in
    (* Then, find common shape for that and cond. *)
    let final_target_shape = Shape.broadcast target_data_shape s_cond in

    let cond_b = broadcast_to final_target_shape cond in
    let if_true_b = broadcast_to final_target_shape if_true in
    let if_false_b = broadcast_to final_target_shape if_false in
    let out =
      match out with
      | Some o -> o
      | None ->
          empty (B.context if_true_b) (B.dtype if_true_b) final_target_shape
    in
    B.where ~out cond_b if_true_b if_false_b;
    out

  (* ───── Binary Mathematical Functions ───── *)

  (* Two-argument arctangent: atan2(y, x) returns angle in [-π, π] *)
  let atan2 ?out y x = binop ~op_name:"atan2" ?out B.atan2 y x

  (* Hypotenuse: sqrt(x² + y²) with overflow protection *)
  let hypot ?out x y =
    let x', y' = broadcasted x y in
    let x_abs = abs x' in
    let y_abs = abs y' in

    (* Use the numerically stable formula: max * sqrt(1 + (min/max)²) *)
    let max_val = maximum x_abs y_abs in
    let min_val = minimum x_abs y_abs in

    (* Handle the case where both are zero *)
    let both_zero =
      logical_and
        (cmpeq x_abs (zeros_like x_abs))
        (cmpeq y_abs (zeros_like y_abs))
    in

    (* Avoid division by zero *)
    let ratio = where both_zero (zeros_like min_val) (div min_val max_val) in
    let ratio_sq = square ratio in
    let one = ones_like ratio_sq in
    let sqrt_term = sqrt (add one ratio_sq) in

    let result = mul max_val sqrt_term in
    where ?out both_zero (zeros_like result) result

  (* ───── Reduction Operations ───── *)

  (* Compute output shape for reduction *)
  let reduce_output_shape input_shape axes_to_reduce keepdims =
    let rank = Array.length input_shape in
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
      let result = Array.of_list (List.rev !filtered) in
      if Array.length result = 0 && rank > 0 then [||] else result

  (* Generic reduction helper *)
  let reduce_op ?out backend_op ?axes ?(keepdims = false) x =
    let input_shape = shape x in
    let rank = Array.length input_shape in
    let axes_to_reduce =
      match axes with
      | None -> Array.init rank Fun.id
      | Some ax_list ->
          Array.of_list
            (List.map (fun ax -> if ax < 0 then ax + rank else ax) ax_list)
    in
    (* Validate axes are in bounds *)
    Array.iter
      (fun ax ->
        if ax < 0 || ax >= rank then
          Error.invalid ~op:"reduce" ~what:"axis"
            ~reason:
              (Printf.sprintf "axis %d out of bounds for tensor of rank %d" ax
                 rank)
            ())
      axes_to_reduce;
    let out =
      match out with
      | Some o -> o
      | None ->
          let out_shape =
            reduce_output_shape input_shape axes_to_reduce keepdims
          in
          empty (B.context x) (B.dtype x) out_shape
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
      let normalized_axis = if axis < 0 then axis + 1 else axis in
      if normalized_axis = 0 then x
      else
        Error.invalid ~op:"associative_scan" ~what:"axis"
          ~reason:
            (Printf.sprintf
               "axis %d out of bounds for rank 0 tensor (only axis 0 valid)"
               axis)
          ()
    else
      let normalized_axis = if axis < 0 then axis + rank else axis in
      if normalized_axis < 0 || normalized_axis >= rank then
        Error.invalid ~op:"associative_scan" ~what:"axis"
          ~reason:(Printf.sprintf "axis %d out of bounds for rank %d" axis rank)
          ()
      else
        let out = empty (B.context x) (B.dtype x) x_shape in
        B.associative_scan ~out ~axis:normalized_axis ~op x;
        out

  let cumulative_scan ?axis op x =
    let orig_shape = shape x in
    match axis with
    | Some axis -> associative_scan ~axis op x
    | None ->
        let numel = array_prod orig_shape in
        let flattened = reshape [| numel |] x in
        let scanned = associative_scan ~axis:0 op flattened in
        if Array.length orig_shape = 0 then
          (* Reshape to scalar shape *)
          reshape [||] scanned
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
    let x_dtype = B.dtype x in
    let num_for_sum = sum ?axes ~keepdims x in

    let s_orig = shape x in
    let r_orig = Array.length s_orig in
    let actual_axes_to_reduce =
      match axes with
      | None -> Array.init r_orig Fun.id
      | Some ax_list ->
          Array.of_list
            (List.map (fun ax -> if ax < 0 then ax + r_orig else ax) ax_list)
    in
    let num_elements_in_reduced_dims =
      if Array.length actual_axes_to_reduce = 0 then 1
      else
        array_prod
          (Array.map (fun ax_idx -> s_orig.(ax_idx)) actual_axes_to_reduce)
    in
    let num_elements_divisor_float =
      float_of_int
        (if num_elements_in_reduced_dims = 0 then 1
         else num_elements_in_reduced_dims)
    in

    let divisor_val_ocaml = Dtype.of_float x_dtype num_elements_divisor_float in
    let divisor_scalar = scalar (B.context x) x_dtype divisor_val_ocaml in
    let divisor_tensor = broadcast_to (shape num_for_sum) divisor_scalar in

    div ?out num_for_sum divisor_tensor

  let var ?out ?axes ?(keepdims = false) ?(ddof = 0) x =
    let@ _ = span ~op:"var" () in
    let x_dtype = B.dtype x in
    let mean_x_keepdim_true = mean ?axes ~keepdims:true x in

    let diff = sub x mean_x_keepdim_true in
    let diff_sq = square diff in
    let sum_diff_sq = sum ?axes ~keepdims diff_sq in

    let s_orig = shape x in
    let r_orig = Array.length s_orig in
    let actual_axes_to_reduce =
      match axes with
      | None -> Array.init r_orig Fun.id
      | Some ax_list ->
          Array.of_list
            (List.map (fun ax -> if ax < 0 then ax + r_orig else ax) ax_list)
    in
    let num_elements_in_reduced_dims =
      if Array.length actual_axes_to_reduce = 0 then 1
      else
        array_prod
          (Array.map (fun ax_idx -> s_orig.(ax_idx)) actual_axes_to_reduce)
    in

    let n_corrected_val = num_elements_in_reduced_dims - ddof in
    let n_corrected_float = float_of_int (Stdlib.max 0 n_corrected_val) in

    let divisor_val_ocaml = Dtype.of_float x_dtype n_corrected_float in
    let divisor_scalar = scalar (B.context x) x_dtype divisor_val_ocaml in
    let divisor_tensor = broadcast_to (shape sum_diff_sq) divisor_scalar in

    div ?out sum_diff_sq divisor_tensor

  let std ?out ?axes ?(keepdims = false) ?(ddof = 0) x =
    let@ _ = span ~op:"std" () in
    let variance = var ?axes ~keepdims ~ddof x in
    sqrt ?out variance

  (* Check if all elements are true (non-zero) *)
  let all ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"all" () in
    (* Convert to boolean first by comparing with zero *)
    let zero_val = Dtype.zero (dtype x) in
    let zero_tensor = full_like x zero_val in
    let bool_tensor = cmpne x zero_tensor in

    (* Now use prod on the boolean tensor *)
    prod ?out ?axes ~keepdims bool_tensor

  (* Check if any element is true (non-zero) *)
  let any ?out ?axes ?(keepdims = false) x =
    let@ _ = span ~op:"any" () in
    (* Convert to boolean first by comparing with zero *)
    let zero_val = Dtype.zero (dtype x) in
    let zero_tensor = full_like x zero_val in
    let bool_tensor = cmpne x zero_tensor in

    (* Now use max on the boolean tensor - any 1 will give 1 *)
    max ?out ?axes ~keepdims bool_tensor

  (* Check if two arrays are element-wise equal *)
  let array_equal x y =
    let@ _ = span ~op:"array_equal" () in
    (* First, check if we can broadcast the shapes *)
    let can_broadcast =
      try
        let _ = Shape.broadcast (shape x) (shape y) in
        true
      with _ -> false
    in

    if not can_broadcast then
      (* If shapes can't be broadcast, arrays are not equal Return a scalar
         False (0) *)
      zeros (B.context x) Dtype.bool [||]
    else
      (* Check element-wise equality and then check if all are true *)
      let eq_result = equal x y in
      all eq_result (* Reduce over all axes to get scalar result *)

  (* ───── Shape Manipulation ───── *)

  let pad padding_config fill_value x =
    let@ _ = span ~op:"pad" () in
    (* Validate padding values are non-negative *)
    Array.iter
      (fun (before, after) ->
        if before < 0 || after < 0 then
          Error.invalid ~op:"pad" ~what:"padding values"
            ~reason:"negative values not allowed"
            ~hint:"use shrink or slice to remove elements" ())
      padding_config;
    B.pad x padding_config fill_value

  let shrink shrink_args x =
    let@ _ = span ~op:"shrink" () in
    B.shrink x shrink_args

  (* collapse dimensions between [start_dim] and [end_dim] *)
  let flatten ?(start_dim = 0) ?(end_dim = -1) x =
    let@ _ = span ~op:"flatten" () in
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
      Error.invalid ~op:"flatten"
        ~what:(Printf.sprintf "start_dim %d or end_dim %d" start_dim end_dim)
        ~reason:(Printf.sprintf "out of bounds for rank %d" r)
        ();
    if s > e then
      Error.invalid ~op:"flatten" ~what:"dimensions"
        ~reason:"start_dim must be <= end_dim" ();

    let new_shape_list =
      if r = 0 then [ 1 ] (* Flatten scalar to shape [1] *)
      else if s = 0 && e = r - 1 then [ array_prod sh ] (* Flatten all to 1D *)
      else
        let pre = Array.to_list (Array.sub sh 0 s) in
        let mid_slice = Array.sub sh s (e - s + 1) in
        let mid_prod =
          if Array.length mid_slice = 0 then 1 else array_prod mid_slice
        in
        let post = Array.to_list (Array.sub sh (e + 1) (r - (e + 1))) in
        pre @ [ mid_prod ] @ post
    in
    reshape (Array.of_list new_shape_list) x

  let unflatten dim sizes x =
    let@ _ = span ~op:"unflatten" () in
    let dim = resolve_single_axis x dim in
    let current_shape = shape x in
    let dim_size = current_shape.(dim) in

    (* Handle -1 in sizes (infer dimension) *)
    let sizes = Array.copy sizes in
    let neg_one_count =
      Array.fold_left (fun acc s -> if s = -1 then acc + 1 else acc) 0 sizes
    in

    if neg_one_count > 1 then
      Error.invalid ~op:"unflatten" ~what:"sizes"
        ~reason:"can only specify one unknown dimension (using -1)" ();

    if neg_one_count = 1 then (
      let known_product =
        Array.fold_left (fun acc s -> if s = -1 then acc else acc * s) 1 sizes
      in
      if known_product = 0 || dim_size mod known_product <> 0 then
        Error.cannot ~op:"unflatten" ~what:"infer dimension"
          ~from:(Printf.sprintf "total size %d" dim_size)
          ~to_:(Printf.sprintf "known product %d" known_product)
          ~reason:
            (Printf.sprintf "%d not divisible by %d" dim_size known_product)
          ~hint:"ensure total size is divisible by product of known dimensions"
          ();
      let inferred_size = dim_size / known_product in
      Array.iteri (fun i s -> if s = -1 then sizes.(i) <- inferred_size) sizes);

    (* Verify that product of sizes equals original dimension *)
    let sizes_product = Array.fold_left ( * ) 1 sizes in
    if sizes_product <> dim_size then
      Error.invalid ~op:"unflatten" ~what:"sizes"
        ~reason:
          (Printf.sprintf "product %d does not match dimension size %d"
             sizes_product dim_size)
        ();

    (* Build new shape *)
    let new_shape =
      Array.concat
        [
          Array.sub current_shape 0 dim;
          sizes;
          Array.sub current_shape (dim + 1)
            (Array.length current_shape - dim - 1);
        ]
    in

    reshape new_shape x

  let ravel x =
    let@ _ = span ~op:"ravel" () in
    flatten x

  module IntSet = Set.Make (Int)

  (* drop axes of size 1; [axes] restricts which axes to squeeze *)
  let squeeze ?axes x =
    let@ _ = span ~op:"squeeze" () in
    let sh = shape x in
    let r = Array.length sh in

    match axes with
    | None ->
        (* Squeeze all dimensions of size 1 *)
        let new_shape_list = List.filter (( <> ) 1) (Array.to_list sh) in
        let new_shape = Array.of_list new_shape_list in
        if Array.length new_shape = 0 && Array.length sh > 0 then
          reshape [||] x (* Result is scalar *)
        else if Array.length new_shape = 0 && Array.length sh = 0 then x
          (* scalar to scalar *)
        else reshape new_shape x
    | Some axes_list ->
        if r = 0 then x (* Cannot squeeze a scalar *)
        else
          (* Normalize negative indices and validate *)
          let normalized_axes =
            List.map (fun ax -> if ax < 0 then ax + r else ax) axes_list
          in

          (* Check for duplicates *)
          let seen = Array.make r false in
          List.iter
            (fun ax ->
              if ax < 0 || ax >= r then
                Error.axis_out_of_bounds ~op:"squeeze" ~axis:ax ~ndim:r ();
              if seen.(ax) then
                Error.invalid ~op:"squeeze"
                  ~what:(Printf.sprintf "axis %d" ax)
                  ~reason:"duplicate axis" ();
              seen.(ax) <- true)
            normalized_axes;

          (* Check that all specified axes have size 1 *)
          List.iter
            (fun ax ->
              if sh.(ax) <> 1 then
                Error.cannot ~op:"squeeze" ~what:"remove dimension"
                  ~from:(Printf.sprintf "axis %d (size %d)" ax sh.(ax))
                  ~to_:"squeezed"
                  ~reason:(Printf.sprintf "size %d≠1" sh.(ax))
                  ())
            normalized_axes;

          (* Build new shape by filtering out squeezed dimensions *)
          let axes_set =
            List.fold_left
              (fun set ax -> IntSet.add ax set)
              IntSet.empty normalized_axes
          in

          let new_shape_list =
            List.filteri
              (fun i _ -> not (IntSet.mem i axes_set))
              (Array.to_list sh)
          in

          let new_shape = Array.of_list new_shape_list in

          if Array.length new_shape = 0 && Array.length sh > 0 then
            reshape [||] x (* Result is scalar *)
          else if Array.length new_shape = 0 && Array.length sh = 0 then x
            (* scalar to scalar *)
          else reshape new_shape x

  (* insert size-1 dimensions at specified axes *)
  let unsqueeze ?axes x =
    let@ _ = span ~op:"unsqueeze" () in
    let sh = shape x in
    let r = Array.length sh in

    let axes_list =
      match axes with
      | None ->
          Error.invalid ~op:"unsqueeze" ~what:"axes" ~reason:"must be specified"
            ()
      | Some lst -> lst
    in

    if List.length axes_list = 0 then x (* No dimensions to add *)
    else
      let output_rank = r + List.length axes_list in

      (* Normalize negative indices (relative to output shape) *)
      let normalized_axes =
        List.map (fun ax -> if ax < 0 then ax + output_rank else ax) axes_list
      in

      (* Validate axes *)
      let seen = Array.make output_rank false in
      List.iter
        (fun ax ->
          if ax < 0 || ax >= output_rank then
            Error.invalid ~op:"unsqueeze"
              ~what:(Printf.sprintf "axis %d" ax)
              ~reason:
                (Printf.sprintf "out of bounds for output rank %d" output_rank)
              ~hint:
                (Printf.sprintf "valid range is [%d, %d)" (-output_rank)
                   output_rank)
              ();
          if seen.(ax) then
            Error.invalid ~op:"unsqueeze"
              ~what:(Printf.sprintf "axis %d" ax)
              ~reason:"duplicate axis" ();
          seen.(ax) <- true)
        normalized_axes;

      (* Sort axes to process in order *)
      (* let sorted_axes = List.sort compare normalized_axes in *)

      (* Build mapping from output position to input position *)
      let axes_set =
        List.fold_left
          (fun set ax -> IntSet.add ax set)
          IntSet.empty normalized_axes
      in

      (* Create new shape *)
      let new_shape_list = ref [] in
      let input_idx = ref 0 in

      for output_idx = 0 to output_rank - 1 do
        if IntSet.mem output_idx axes_set then
          new_shape_list :=
            1 :: !new_shape_list (* Insert dimension of size 1 *)
        else if !input_idx < r then (
          new_shape_list := sh.(!input_idx) :: !new_shape_list;
          incr input_idx)
      done;

      let new_shape = Array.of_list (List.rev !new_shape_list) in
      reshape new_shape x

  (* For backward compatibility, you might want to add these helper
     functions: *)

  (* squeeze a single axis *)
  let squeeze_axis axis x = squeeze ~axes:[ axis ] x

  (* unsqueeze a single axis *)
  let unsqueeze_axis axis x = unsqueeze ~axes:[ axis ] x

  (* expand_dims is an alias for unsqueeze *)
  let expand_dims axes x = unsqueeze ~axes x

  let transpose ?axes x =
    let@ _ = span ~op:"transpose" () in
    let r = ndim x in
    let resolved_axes =
      match axes with
      | None -> Array.init r (fun i -> r - 1 - i) (* Reverse dimensions *)
      | Some ax_list ->
          if List.length ax_list <> r then
            Error.invalid ~op:"transpose"
              ~what:(Printf.sprintf "axes (length %d)" (List.length ax_list))
              ~reason:
                (Printf.sprintf "expected rank %d, got %d" r
                   (List.length ax_list))
              ~hint:"provide exactly one axis per dimension" ();
          let seen = Array.make r false in
          List.iter
            (fun ax_val ->
              let ax = if ax_val < 0 then ax_val + r else ax_val in
              if ax < 0 || ax >= r then
                Error.axis_out_of_bounds ~op:"transpose" ~axis:ax_val ~ndim:r ();
              if seen.(ax) then
                Error.invalid ~op:"transpose"
                  ~what:(Printf.sprintf "axis %d" ax_val)
                  ~reason:"repeated" ();
              seen.(ax) <- true)
            ax_list;
          if not (Array.for_all Fun.id seen) then
            Error.invalid ~op:"transpose" ~what:"axes"
              ~reason:"do not form a permutation" ();
          (* Normalize negative axes *)
          Array.of_list
            (List.map
               (fun ax_val -> if ax_val < 0 then ax_val + r else ax_val)
               ax_list)
    in
    let result = B.permute x resolved_axes in
    result

  let flip ?axes x =
    let@ _ = span ~op:"flip" () in
    let r = ndim x in
    let flip_bools = Array.make r false in
    (match axes with
    | None -> Array.fill flip_bools 0 r true (* Flip all axes *)
    | Some ax_list ->
        List.iter
          (fun ax_val ->
            let ax = if ax_val < 0 then ax_val + r else ax_val in
            if ax < 0 || ax >= r then
              Error.axis_out_of_bounds ~op:"flip" ~axis:ax_val ~ndim:r ();
            flip_bools.(ax) <- true)
          ax_list);
    B.flip x flip_bools

  let moveaxis src dst x =
    let@ _ = span ~op:"moveaxis" () in
    let r = ndim x in
    let norm_src = if src < 0 then src + r else src in
    let norm_dst = if dst < 0 then dst + r else dst in

    if norm_src < 0 || norm_src >= r || norm_dst < 0 || norm_dst >= r then
      Error.invalid ~op:"moveaxis"
        ~what:(Printf.sprintf "source %d or destination %d" src dst)
        ~reason:
          (Format.asprintf "out of bounds for shape %a" Shape.pp (shape x))
        ();

    if norm_src = norm_dst then x (* No change *)
    else
      let axes_list = Array.to_list (Array.init r Fun.id) in
      let item_to_move = List.nth axes_list norm_src in
      let list_without_item = List.filter (( <> ) item_to_move) axes_list in

      let rec insert_at idx item lst acc =
        match lst with
        | [] -> List.rev (item :: acc)
        | hd :: tl ->
            if idx = 0 then List.rev_append acc (item :: hd :: tl)
            else insert_at (idx - 1) item tl (hd :: acc)
      in
      let final_axes_list =
        insert_at norm_dst item_to_move list_without_item []
      in
      B.permute x (Array.of_list final_axes_list)

  let swapaxes axis1 axis2 x =
    let@ _ = span ~op:"swapaxes" () in
    let r = ndim x in
    let norm_axis1 = if axis1 < 0 then axis1 + r else axis1 in
    let norm_axis2 = if axis2 < 0 then axis2 + r else axis2 in

    if norm_axis1 < 0 || norm_axis1 >= r || norm_axis2 < 0 || norm_axis2 >= r
    then
      Error.invalid ~op:"swapaxes"
        ~what:(Printf.sprintf "axes (%d, %d)" axis1 axis2)
        ~reason:
          (Format.asprintf "out of bounds for shape %a" Shape.pp (shape x))
        ();

    if norm_axis1 = norm_axis2 then x (* No change *)
    else
      let axes = Array.init r Fun.id in
      let temp = axes.(norm_axis1) in
      axes.(norm_axis1) <- axes.(norm_axis2);
      axes.(norm_axis2) <- temp;
      B.permute x axes

  let cat_tensors ~axis tensors =
    match tensors with
    | [] ->
        Error.invalid ~op:"concatenate" ~what:"tensor list" ~reason:"empty"
          ~hint:"provide at least one tensor" ()
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
    let x, ax_idx =
      match axis with
      | None ->
          let flat_x = flatten x in
          (* flatten handles rank 0 correctly for its own purpose *)
          (flat_x, 0)
      | Some specified_axis ->
          let r = ndim x in
          let norm_axis =
            if specified_axis < 0 then specified_axis + r else specified_axis
          in
          if norm_axis < 0 || norm_axis >= r then
            Error.axis_out_of_bounds ~op:"roll" ~axis:specified_axis ~ndim:r ();
          (x, norm_axis)
    in
    let current_shape = shape x in
    let r = ndim x in

    if r = 0 then x (* Cannot roll a scalar *)
    else
      let dim_size = current_shape.(ax_idx) in
      if dim_size = 0 then x (* Cannot roll an empty dimension *)
      else
        let s = shift mod dim_size in
        let actual_shift = if s < 0 then s + dim_size else s in

        if actual_shift = 0 then
          if axis = None then reshape (shape x) x
          else x (* Reshape back if flattened and no-op roll *)
        else
          let ranges_part1 =
            Array.mapi
              (fun i cur_dim ->
                if i = ax_idx then (dim_size - actual_shift, cur_dim)
                else (0, cur_dim))
              current_shape
          in
          let ranges_part2 =
            Array.mapi
              (fun i cur_dim ->
                if i = ax_idx then (0, dim_size - actual_shift) else (0, cur_dim))
              current_shape
          in
          let part1 = shrink ranges_part1 x in
          let part2 = shrink ranges_part2 x in
          let rolled_x = cat_tensors ~axis:ax_idx [ part1; part2 ] in
          if axis = None then reshape original_shape rolled_x else rolled_x

  let tile reps x =
    let@ _ = span ~op:"tile" () in
    let t_shape = shape x in
    let t_ndim = ndim x in
    let reps_len = Array.length reps in

    if reps_len < t_ndim then
      Error.invalid ~op:"tile" ~what:"reps length"
        ~reason:"must be >= tensor rank" ();

    (* If reps has more dimensions than x, prepend 1s to x's shape *)
    let x_promoted, promoted_shape =
      if reps_len > t_ndim then (
        let new_shape = Array.make reps_len 1 in
        Array.blit t_shape 0 new_shape (reps_len - t_ndim) t_ndim;
        (reshape new_shape x, new_shape))
      else (x, t_shape)
    in

    Array.iteri
      (fun i r ->
        if r < 0 then
          Error.invalid ~op:"tile"
            ~what:(Printf.sprintf "reps[%d]" i)
            ~reason:(Printf.sprintf "negative (%d<0)" r)
            ~hint:"use positive integers (or 0 for empty result)" ())
      reps;

    if Array.for_all (( = ) 1) reps then
      B.copy x_promoted (* optimization: no tiling needed *)
    else if Array.exists (( = ) 0) reps || Array.exists (( = ) 0) promoted_shape
    then
      (* If any rep is 0, or original shape has a 0, the tiled dimension becomes
         0 *)
      let tiled_shape =
        Array.mapi (fun i s_i -> s_i * reps.(i)) promoted_shape
      in
      empty (B.context x) (dtype x) tiled_shape
    else
      (* Tile using concatenation along each axis *)
      let rec tile_axis curr_x axis =
        if axis >= reps_len then curr_x
        else if reps.(axis) = 1 then tile_axis curr_x (axis + 1)
        else
          (* Concatenate reps.(axis) copies along this axis *)
          let copies = List.init reps.(axis) (fun _ -> curr_x) in
          let concatenated = cat_tensors ~axis copies in
          tile_axis concatenated (axis + 1)
      in
      tile_axis x_promoted 0

  let repeat ?axis count x =
    let@ _ = span ~op:"repeat" () in
    if count < 0 then
      Error.check_bounds ~op:"repeat" ~name:"count" ~value:count ~min:0 ();

    let x, ax_idx_eff =
      match axis with
      | None ->
          let flat_x = flatten x in
          (flat_x, 0)
      | Some specified_axis ->
          let r = ndim x in
          let norm_axis =
            if specified_axis < 0 then specified_axis + r else specified_axis
          in
          if norm_axis < 0 || norm_axis >= r then
            Error.axis_out_of_bounds ~op:"repeat" ~axis:specified_axis ~ndim:r
              ();
          (x, norm_axis)
    in

    let t_shape = shape x in
    let t_ndim = ndim x in

    if count = 0 then (
      let new_s = Array.copy t_shape in
      if t_ndim > 0 then new_s.(ax_idx_eff) <- 0;
      let final_shape_if_flattened = if axis = None then [| 0 |] else new_s in
      empty (B.context x) (dtype x) final_shape_if_flattened)
    else if count = 1 then B.copy x
    else if t_ndim = 0 then
      let scalar_reshaped = reshape [| 1 |] x in
      let repeated = expand [| count |] scalar_reshaped in

      if axis = None then repeated else reshape (shape x) repeated
    else
      (* Repeat using concatenation of individual elements *)
      let axis_size = t_shape.(ax_idx_eff) in
      let slices = ref [] in

      (* Extract each element along the axis and repeat it *)
      for i = axis_size - 1 downto 0 do
        (* Get slice at position i *)
        let slice =
          Array.init t_ndim (fun dim ->
              if dim = ax_idx_eff then (i, i + 1) else (0, t_shape.(dim)))
        in
        let slice_view = B.shrink x slice in

        (* Repeat this slice count times *)
        for _ = 1 to count do
          slices := slice_view :: !slices
        done
      done;

      (* Concatenate all slices *)
      let result = cat_tensors ~axis:ax_idx_eff !slices in

      if axis = None then result else result

  let concatenate ?axis ts =
    let@ _ = span ~op:"concatenate" () in
    match ts with
    | [] ->
        Error.invalid ~op:"concatenate" ~what:"tensor list" ~reason:"empty"
          ~hint:"provide at least one tensor" ()
    | [ x ] -> copy x
    | _ ->
        let axis =
          match axis with
          | None ->
              (* Check all arrays have same dtype *)
              let first_dtype = dtype (List.hd ts) in
              List.iter
                (fun x ->
                  let x_dtype = dtype x in
                  if not (Dtype.equal first_dtype x_dtype) then
                    Error.dtype_mismatch ~op:"concatenate"
                      ~expected:(Dtype.to_string first_dtype)
                      ~actual:(Dtype.to_string x_dtype) ())
                (List.tl ts);

              (* Flatten all arrays first *)
              let flattened = List.map flatten ts in
              cat_tensors ~axis:0 flattened
          | Some a ->
              let first = List.hd ts in
              let first_ndim = ndim first in
              let axis = resolve_single_axis ~ndim_opt:first_ndim first a in

              (* Check all arrays have same dtype *)
              let first_dtype = dtype first in
              List.iter
                (fun x ->
                  let x_dtype = dtype x in
                  if not (Dtype.equal first_dtype x_dtype) then
                    Error.dtype_mismatch ~op:"concatenate"
                      ~expected:(Dtype.to_string first_dtype)
                      ~actual:(Dtype.to_string x_dtype) ())
                (List.tl ts);

              (* Check all arrays have same ndim *)
              if not (List.for_all (fun x -> ndim x = first_ndim) ts) then
                Error.invalid ~op:"concatenate" ~what:"arrays"
                  ~reason:"must have same number of dimensions" ();

              (* Check shapes match except on concatenation axis *)
              let first_shape = shape (List.hd ts) in
              List.iter
                (fun x ->
                  let t_shape = shape x in
                  Array.iteri
                    (fun i s ->
                      if i <> axis && s <> first_shape.(i) then
                        Error.invalid ~op:"concatenate"
                          ~what:(Printf.sprintf "dimension %d" i)
                          ~reason:
                            (Printf.sprintf "size %d≠%d" s first_shape.(i))
                          ())
                    t_shape)
                (List.tl ts);

              cat_tensors ~axis ts
        in
        axis

  let stack ?axis ts =
    let@ _ = span ~op:"stack" () in
    match ts with
    | [] -> Error.empty_input ~op:"stack" ~what:"tensor list"
    | _ ->
        let first_shape = shape (List.hd ts) in
        let first_ndim = Array.length first_shape in

        (* Determine stacking axis *)
        let axis =
          match axis with
          | None -> 0
          | Some a ->
              let a = if a < 0 then a + first_ndim + 1 else a in
              if a < 0 || a > first_ndim then
                Error.axis_out_of_bounds ~op:"stack" ~axis:a ~ndim:first_ndim ();
              a
        in

        (* Add new dimension to each array *)
        let expanded = List.map (fun x -> unsqueeze ~axes:[ axis ] x) ts in

        (* Concatenate along the new axis *)
        concatenate ~axis expanded

  (* Helper to ensure arrays have at least n dimensions *)
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
    | [] -> Error.empty_input ~op:"vstack" ~what:"tensor list"
    | _ ->
        (* Make all arrays at least 2D *)
        let arrays_2d =
          List.map
            (fun x ->
              let nd = ndim x in
              if nd = 0 then reshape [| 1; 1 |] x
              else if nd = 1 then reshape [| 1; numel x |] x
              else x)
            ts
        in
        (* Concatenate along first axis *)
        concatenate ~axis:0 arrays_2d

  let hstack ts =
    let@ _ = span ~op:"hstack" () in
    match ts with
    | [] -> Error.empty_input ~op:"hstack" ~what:"tensor list"
    | _ ->
        (* Handle different dimensions *)
        let all_1d = List.for_all (fun x -> ndim x <= 1) ts in

        if all_1d then
          (* For 1D arrays, concatenate along axis 0 *)
          let arrays_1d =
            List.map (fun x -> if ndim x = 0 then reshape [| 1 |] x else x) ts
          in
          concatenate ~axis:0 arrays_1d
        else
          (* Make all arrays at least 2D *)
          let arrays_2d =
            List.map
              (fun x ->
                let nd = ndim x in
                if nd = 0 then reshape [| 1; 1 |] x
                else if nd = 1 then reshape [| numel x; 1 |] x
                else x)
              ts
          in

          (* Concatenate along second axis *)
          concatenate ~axis:1 arrays_2d

  let dstack ts =
    let@ _ = span ~op:"dstack" () in
    match ts with
    | [] -> Error.empty_input ~op:"dstack" ~what:"tensor list"
    | _ ->
        (* Make all arrays at least 3D *)
        let arrays_3d =
          List.map
            (fun x ->
              let s = shape x in
              let nd = Array.length s in
              if nd = 0 then reshape [| 1; 1; 1 |] x
              else if nd = 1 then reshape [| 1; s.(0); 1 |] x
              else if nd = 2 then reshape [| s.(0); s.(1); 1 |] x
              else x)
            ts
        in

        (* Concatenate along third axis *)
        concatenate ~axis:2 arrays_3d

  let broadcast_arrays ts =
    let@ _ = span ~op:"broadcast_arrays" () in
    match ts with
    | [] -> []
    | [ x ] -> [ x ]
    | _ ->
        (* Find broadcast shape *)
        let broadcast_shape =
          List.fold_left
            (fun acc_shape x -> Shape.broadcast acc_shape (shape x))
            (shape (List.hd ts))
            (List.tl ts)
        in

        (* Broadcast all arrays to common shape *)
        List.map (fun x -> broadcast_to broadcast_shape x) ts

  let eye ctx ?m ?k dtype n =
    let@ _ = span ~op:"eye" () in
    let rows = match m with Some v -> v | None -> n in
    let cols = n in
    let k_val = match k with Some v -> v | None -> 0 in

    let final_shape = [| rows; cols |] in

    (* Early exit if k is out of bounds such that no ones can be placed *)
    if rows <= 0 || cols <= 0 || k_val >= cols || k_val <= -rows then
      zeros ctx dtype final_shape
    else
      (* Simple implementation: create array and set diagonal elements *)
      let arr = Array.make (rows * cols) (Dtype.zero dtype) in

      (* Set diagonal elements to one *)
      let one = Dtype.one dtype in
      for i = 0 to (if rows < cols then rows else cols) - 1 do
        let row = i in
        let col = i + k_val in
        if col >= 0 && col < cols then arr.((row * cols) + col) <- one
      done;

      create ctx dtype final_shape arr

  let identity ctx dtype n = eye ctx ~m:n ~k:0 dtype n

  let diag ?(k = 0) v =
    let v_shape = shape v in
    let v_ndim = Array.length v_shape in
    if v_ndim = 1 then
      (* Construct 2D array with v on the k-th diagonal *)
      let n = v_shape.(0) in
      let size = n + Int.abs k in
      let v_arr = to_array v in
      init (B.context v) (dtype v) [| size; size |] (fun indices ->
          let row = indices.(0) in
          let col = indices.(1) in
          let diag_idx =
            if k >= 0 then
              (* Diagonal above main: col = row + k, so row = col - k *)
              if col = row + k && row >= 0 && row < n then row else -1
            else if
              (* Diagonal below main: row = col - k, so row = col - k *)
              row = col - k && col >= 0 && col < n
            then col
            else -1
          in
          if diag_idx >= 0 && diag_idx < n then v_arr.(diag_idx)
          else Dtype.zero (dtype v))
    else if v_ndim >= 2 then
      (* Extract k-th diagonal from 2D array *)
      let rows = v_shape.(0) in
      let cols = v_shape.(1) in
      let diag_len =
        if k >= 0 then Int.min rows (cols - k) else Int.min (rows + k) cols
      in
      let diag_len = Int.max 0 diag_len in
      if diag_len = 0 then empty (B.context v) (dtype v) [| 0 |]
      else
        let v_arr = to_array v in
        init (B.context v) (dtype v) [| diag_len |] (fun indices ->
            let i = indices.(0) in
            let row = if k >= 0 then i else i - k in
            let col = if k >= 0 then i + k else i in
            (* Calculate linear index in row-major order *)
            let linear_idx = (row * cols) + col in
            v_arr.(linear_idx))
    else
      Error.invalid ~op:"diag" ~what:"input"
        ~reason:(Printf.sprintf "expected 1D or 2D array, got %dD" v_ndim)
        ()

  let arange (type a b) ctx (dtype : (a, b) Dtype.t) start stop step =
    let@ _ = span ~op:"arange" () in
    if start >= stop && step > 0 then
      Error.invalid ~op:"arange"
        ~what:(Printf.sprintf "range [%d, %d)" start stop)
        ~reason:(Printf.sprintf "empty with step=%d" step)
        ~hint:
          "ensure start < stop for positive step, or start > stop for negative \
           step"
        ();
    if step = 0 then
      Error.invalid ~op:"arange" ~what:"step" ~reason:"cannot be zero" ();
    let num_elements =
      if step > 0 then
        if start >= stop then 0
        else
          (stop - start + step - 1)
          / step (* Equivalent to ceil((stop-start)/step) for int math *)
      else if
        (* step < 0 *)
        start <= stop
      then 0
      else (start - stop + -step - 1) / -step
      (* Equivalent to ceil((start-stop)/(-step)) *)
    in
    if num_elements <= 0 then empty ctx dtype [| 0 |]
    else
      let f_init idx_arr : a =
        let i = idx_arr.(0) in
        match dtype with
        | Dtype.Float16 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.Float32 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.Float64 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.BFloat16 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.Float8_e4m3 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.Float8_e5m2 ->
            float_of_int start +. (float_of_int i *. float_of_int step)
        | Dtype.Int8 -> start + (i * step)
        | Dtype.UInt8 -> start + (i * step)
        | Dtype.Int16 -> start + (i * step)
        | Dtype.UInt16 -> start + (i * step)
        | Dtype.Int4 -> start + (i * step)
        | Dtype.UInt4 -> start + (i * step)
        | Dtype.Bool -> if i = 0 then false else true
        | Dtype.Int32 ->
            Int32.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.Int64 ->
            Int64.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.UInt32 ->
            Int32.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.UInt64 ->
            Int64.(add (of_int start) (mul (of_int i) (of_int step)))
        | Dtype.Complex64 ->
            {
              Complex.re =
                float_of_int start +. (float_of_int i *. float_of_int step);
              im = 0.;
            }
        | Dtype.Complex128 ->
            {
              Complex.re =
                float_of_int start +. (float_of_int i *. float_of_int step);
              im = 0.;
            }
      in
      init ctx dtype [| num_elements |] f_init

  let arange_f ctx dtype start_f stop_f step_f =
    let@ _ = span ~op:"arange_f" () in
    if step_f = 0. then
      Error.invalid ~op:"arange_f" ~what:"step" ~reason:"cannot be zero" ();
    let num_exact_steps = (stop_f -. start_f) /. step_f in
    let eps_factor = 1e-9 in
    (* Small factor to subtract before floor for robust exclusive bound *)
    let num_elements =
      (* Check if the range is non-positive or extremely small *)
      if
        (step_f > 0. && stop_f <= start_f +. (eps_factor *. Float.abs step_f))
        || (step_f < 0. && stop_f >= start_f +. (eps_factor *. Float.abs step_f))
        || (Float.abs num_exact_steps < eps_factor && num_exact_steps <= 0.)
      then 0
      else
        (* Apply epsilon correction for floor to ensure exclusive upper bound *)
        let corrected_num_steps =
          num_exact_steps -. Float.copy_sign eps_factor num_exact_steps
        in
        int_of_float (Float.floor corrected_num_steps +. 1.)
    in
    let num_elements = Stdlib.max 0 num_elements in
    (* Final guard, though prior logic should handle it *)

    if num_elements <= 0 then empty ctx dtype [| 0 |]
    else
      let f_init idx_arr =
        (* OCaml type 'a is float here *)
        start_f +. (float_of_int idx_arr.(0) *. step_f)
      in
      init ctx dtype [| num_elements |] f_init

  let linspace ctx dtype ?(endpoint = true) start_f stop_f count =
    let@ _ = span ~op:"linspace" () in
    if count < 0 then
      Error.invalid ~op:"linspace"
        ~what:(Printf.sprintf "count %d" count)
        ~reason:"negative count" ~hint:"use count >= 0" ();
    if count = 0 then empty ctx dtype [| 0 |]
    else if count = 1 then full ctx dtype [| 1 |] (Dtype.of_float dtype start_f)
    else
      let div_factor = float_of_int (if endpoint then count - 1 else count) in
      let step_f = (stop_f -. start_f) /. div_factor in
      let f_init idx_arr =
        let i_f = float_of_int idx_arr.(0) in
        Dtype.of_float dtype (start_f +. (i_f *. step_f))
      in
      init ctx dtype [| count |] f_init

  let logspace ctx dtype ?(endpoint = true) ?(base = 10.0) start_exp_f
      stop_exp_f count =
    let@ _ = span ~op:"logspace" () in
    if count < 0 then
      Error.check_bounds ~op:"logspace" ~name:"count" ~value:count ~min:0 ();
    if count = 0 then empty ctx dtype [| 0 |]
    else
      (* The exponents should be generated with the same float precision as the
         final tensor type. *)
      let exponents_tensor =
        linspace ctx dtype ~endpoint start_exp_f stop_exp_f count
      in
      if base = Float.exp 1.0 then (* base is e *)
        exp exponents_tensor
      else if base = 2.0 then exp2 exponents_tensor
      else
        (* General case: base ** exponents = exp2(exponents * log2(base)) *)
        let log2_base = Stdlib.log base /. Stdlib.log 2.0 in
        let log2_base_tensor = scalar ctx dtype log2_base in
        (* Ensure log2_base_tensor is broadcastable with exponents_tensor *)
        let broadcasted_log2_base =
          broadcast_to (shape exponents_tensor) log2_base_tensor
        in
        let scaled_exponents = mul exponents_tensor broadcasted_log2_base in
        exp2 scaled_exponents

  let geomspace ctx dtype ?(endpoint = true) start_val_f stop_val_f count =
    if start_val_f <= 0. || stop_val_f <= 0. then
      Error.invalid ~op:"geomspace"
        ~what:
          (if start_val_f <= 0. then Printf.sprintf "start %g" start_val_f
           else Printf.sprintf "stop %g" stop_val_f)
        ~reason:"must be positive (>0)"
        ~hint:"geomspace requires positive values for logarithmic spacing" ();
    if count < 0 then
      Error.check_bounds ~op:"geomspace" ~name:"count" ~value:count ~min:0 ();
    if count = 0 then empty ctx dtype [| 0 |]
    else if count = 1 then
      full ctx dtype [| 1 |] start_val_f (* OCaml type 'a is float here *)
    else
      let log_start_f = Stdlib.log start_val_f in
      let log_stop_f = Stdlib.log stop_val_f in
      (* The log-points should be generated with the same float precision as the
         final tensor type. *)
      let log_points_tensor =
        linspace ctx dtype ~endpoint log_start_f log_stop_f count
      in
      exp log_points_tensor

  let meshgrid ?(indexing = `xy) x y =
    let x_shape = shape x in
    let y_shape = shape y in

    (* Check inputs are 1D *)
    if Array.length x_shape <> 1 then invalid_arg "meshgrid: x must be 1D";
    if Array.length y_shape <> 1 then invalid_arg "meshgrid: y must be 1D";

    let nx = x_shape.(0) in
    let ny = y_shape.(0) in

    match indexing with
    | `xy ->
        (* Standard Cartesian indexing *)
        let x_grid = reshape [| 1; nx |] x in
        let x_grid = broadcast_to [| ny; nx |] x_grid in

        let y_grid = reshape [| ny; 1 |] y in
        let y_grid = broadcast_to [| ny; nx |] y_grid in

        (x_grid, y_grid)
    | `ij ->
        (* Matrix indexing *)
        let x_grid = reshape [| nx; 1 |] x in
        let x_grid = broadcast_to [| nx; ny |] x_grid in

        let y_grid = reshape [| 1; ny |] y in
        let y_grid = broadcast_to [| nx; ny |] y_grid in

        (x_grid, y_grid)

  let tril ?k x =
    let@ _ = span ~op:"tril" () in
    (* Lower triangular part of matrix *)
    let k_val = match k with Some v -> v | None -> 0 in
    let shape = shape x in
    let ndim = Array.length shape in

    if ndim < 2 then
      Error.invalid ~op:"tril" ~what:"input"
        ~reason:"requires at least 2D tensor" ()
    else
      let rows = shape.(ndim - 2) in
      let cols = shape.(ndim - 1) in

      (* Create mask for lower triangular part *)
      let row_idx = arange (B.context x) int32 0 rows 1 in
      let col_idx = arange (B.context x) int32 0 cols 1 in

      (* Reshape for broadcasting: rows -> [rows, 1], cols -> [1, cols] *)
      let row_idx = reshape [| rows; 1 |] row_idx in
      let col_idx = reshape [| 1; cols |] col_idx in

      (* Create mask: row_idx >= col_idx - k *)
      let mask =
        greater_equal row_idx
          (sub col_idx (scalar (B.context x) int32 (Int32.of_int k_val)))
      in

      (* Broadcast mask to match input shape if needed *)
      let mask =
        if ndim > 2 then
          let batch_shape = Array.sub shape 0 (ndim - 2) in
          let full_shape = Array.concat [ batch_shape; [| rows; cols |] ] in
          broadcast_to full_shape mask
        else mask
      in

      (* Apply mask using where operation *)
      where mask x (zeros_like x)

  let triu ?k x =
    let@ _ = span ~op:"triu" () in
    (* Upper triangular part of matrix *)
    let k_val = match k with Some v -> v | None -> 0 in
    let shape = shape x in
    let ndim = Array.length shape in

    if ndim < 2 then
      Error.invalid ~op:"triu" ~what:"input"
        ~reason:"requires at least 2D tensor" ()
    else
      let rows = shape.(ndim - 2) in
      let cols = shape.(ndim - 1) in

      (* Create mask for upper triangular part *)
      let row_idx = arange (B.context x) int32 0 rows 1 in
      let col_idx = arange (B.context x) int32 0 cols 1 in

      (* Reshape for broadcasting: rows -> [rows, 1], cols -> [1, cols] *)
      let row_idx = reshape [| rows; 1 |] row_idx in
      let col_idx = reshape [| 1; cols |] col_idx in

      (* Create mask: row_idx <= col_idx - k *)
      let mask =
        less_equal row_idx
          (sub col_idx (scalar (B.context x) int32 (Int32.of_int k_val)))
      in

      (* Broadcast mask to match input shape if needed *)
      let mask =
        if ndim > 2 then
          let batch_shape = Array.sub shape 0 (ndim - 2) in
          let full_shape = Array.concat [ batch_shape; [| rows; cols |] ] in
          broadcast_to full_shape mask
        else mask
      in

      (* Apply mask using where operation *)
      where mask x (zeros_like x)

  (* ───── Take Operations ───── *)

  let take ?axis ?(mode = `raise) indices t =
    let@ _ = span ~op:"take" () in
    let t_shape = shape t in
    let context = B.context t in

    match axis with
    | None ->
        (* Flatten t and take from flat array *)
        let t_flat = reshape [| numel t |] t in
        (* Handle out-of-bounds based on mode *)
        let indices_processed =
          match mode with
          | `raise -> indices
          | `wrap ->
              let n = numel t in
              mod_ indices (scalar (B.context indices) Int32 (Int32.of_int n))
          | `clip ->
              let max_idx = numel t - 1 in
              minimum
                (maximum indices (zeros context Int32 (shape indices)))
                (full context Int32 (shape indices) (Int32.of_int max_idx))
        in
        let out = empty context (dtype t_flat) (shape indices_processed) in
        B.gather ~out t_flat indices_processed ~axis:0;
        out
    | Some axis ->
        let axis = resolve_single_axis t axis in
        let axis_size = t_shape.(axis) in

        (* Handle out-of-bounds based on mode *)
        let indices_processed =
          match mode with
          | `raise -> indices
          | `wrap ->
              let n = axis_size in
              mod_ indices (scalar (B.context indices) Int32 (Int32.of_int n))
          | `clip ->
              let max_idx = axis_size - 1 in
              minimum
                (maximum indices (zeros context Int32 (shape indices)))
                (full context Int32 (shape indices) (Int32.of_int max_idx))
        in

        (* Expand indices to match data rank for op_gather *)
        let n_indices = numel indices_processed in
        let out_shape = Array.copy t_shape in
        out_shape.(axis) <- n_indices;

        let expanded_indices_shape = Array.copy t_shape in
        expanded_indices_shape.(axis) <- n_indices;
        for i = 0 to Array.length t_shape - 1 do
          if i <> axis then expanded_indices_shape.(i) <- 1
        done;

        let indices_expanded =
          reshape expanded_indices_shape indices_processed
        in
        let broadcast_shape = Array.copy t_shape in
        broadcast_shape.(axis) <- n_indices;
        let indices_broadcast = broadcast_to broadcast_shape indices_expanded in

        let result = empty context (dtype t) (shape indices_broadcast) in
        B.gather ~out:result t indices_broadcast ~axis;
        reshape out_shape result

  let take_along_axis ~axis indices t =
    let@ _ = span ~op:"take_along_axis" () in
    let axis = resolve_single_axis t axis in
    let t_shape = shape t in
    let idx_shape = shape indices in

    if Array.length t_shape <> Array.length idx_shape then
      Error.shape_mismatch ~op:"take_along_axis" ~expected:t_shape
        ~actual:idx_shape ();

    Array.iteri
      (fun i dim ->
        if i <> axis && dim <> idx_shape.(i) then
          Error.invalid ~op:"take_along_axis" ~what:"shape"
            ~reason:
              (Printf.sprintf "dimension %d: indices has %d but tensor has %d" i
                 idx_shape.(i) dim)
            ())
      t_shape;

    let out = empty (B.context t) (dtype t) idx_shape in
    B.gather ~out t indices ~axis;
    out

  (* ───── Indexing and Slicing ───── *)

  (* Helper to normalize negative indices *)
  let normalize_index dim_size idx = if idx < 0 then dim_size + idx else idx

  (* Convert index specification to list of indices *)
  let indices_of_spec dim_size = function
    | I idx ->
        let idx' = normalize_index dim_size idx in
        if idx' < 0 || idx' >= dim_size then
          Error.invalid ~op:"slice"
            ~what:(Printf.sprintf "index %d" idx)
            ~reason:
              (Printf.sprintf "out of bounds [%d, %d)"
                 (if idx < 0 then -dim_size else 0)
                 dim_size)
            ()
        else [ idx' ]
    | L indices ->
        List.map
          (fun idx ->
            let idx' = normalize_index dim_size idx in
            if idx' < 0 || idx' >= dim_size then
              Error.invalid ~op:"slice"
                ~what:(Printf.sprintf "index %d" idx)
                ~reason:
                  (Printf.sprintf "out of bounds [%d, %d)"
                     (if idx < 0 then -dim_size else 0)
                     dim_size)
                ()
            else idx')
          indices
    | A ->
        (* All indices *)
        List.init dim_size (fun i -> i)
    | R (start_idx, stop_idx) ->
        let start = if start_idx < 0 then dim_size + start_idx else start_idx in
        let stop = if stop_idx < 0 then dim_size + stop_idx else stop_idx in
        let stop = stop - 1 in
        (* Make exclusive end inclusive *)
        let step = 1 in
        let rec collect acc i =
          if step > 0 then
            if i > stop then List.rev acc
            else if i >= dim_size then List.rev acc (* Out of bounds, stop *)
            else collect (i :: acc) (i + step)
          else if i < stop then List.rev acc
          else if i < 0 then List.rev acc (* Out of bounds, stop *)
          else collect (i :: acc) (i + step)
        in
        collect [] start
    | Rs (start_idx, stop_idx, step_val) ->
        if step_val = 0 then
          Error.invalid ~op:"slice" ~what:"step" ~reason:"cannot be zero"
            ~hint:
              "use positive step for forward slicing or negative for reverse"
            ();
        let start = if start_idx < 0 then dim_size + start_idx else start_idx in
        let stop = if stop_idx < 0 then dim_size + stop_idx else stop_idx in
        let stop =
          if step_val > 0 then
            stop - 1 (* Make exclusive end inclusive for positive step *)
          else stop + 1 (* Make exclusive end inclusive for negative step *)
        in
        let step = step_val in
        let rec collect acc i =
          if step > 0 then
            if i > stop then List.rev acc
            else if i >= dim_size then List.rev acc (* Out of bounds, stop *)
            else if i < 0 then collect acc (i + step)
              (* Skip negative, may become valid *)
            else collect (i :: acc) (i + step)
          else if
            (* step < 0 *)
            i < stop
          then List.rev acc
          else if i < 0 then List.rev acc (* Out of bounds, stop *)
          else if i >= dim_size then collect acc (i + step)
            (* Skip too large, may become valid *)
          else collect (i :: acc) (i + step)
        in
        collect [] start
    | N ->
        Error.invalid ~op:"indices_of_spec" ~what:"spec"
          ~reason:"new axis not supported in this context" ()
    | M _ ->
        Error.invalid ~op:"indices_of_spec" ~what:"spec"
          ~reason:"mask indexing not supported in this context" ()

  (* Normalized slice operation per dimension *)
  type dim_op =
    | View of { start : int; stop : int; step : int; dim_len : int }
    | Squeeze of { idx : int }
    | Gather of int array
    | New_axis

  (* Normalize index with bounds checking *)
  let normalize_and_check_index ~op dim_size idx =
    let idx' = if idx < 0 then dim_size + idx else idx in
    if idx' < 0 || idx' >= dim_size then
      Error.invalid ~op
        ~what:(Printf.sprintf "index %d" idx)
        ~reason:(Printf.sprintf "out of bounds [0, %d)" dim_size)
        ();
    idx'

  let normalize_slice_spec dim_size = function
    | I idx ->
        Squeeze { idx = normalize_and_check_index ~op:"slice" dim_size idx }
    | A -> View { start = 0; stop = dim_size; step = 1; dim_len = dim_size }
    | R (start, stop) ->
        let s = if start < 0 then dim_size + start else start in
        let e = if stop < 0 then dim_size + stop else stop in
        let s = Int.max 0 (Int.min s dim_size) in
        let e = Int.max 0 (Int.min e dim_size) in
        let len = Int.max 0 (e - s) in
        View { start = s; stop = e; step = 1; dim_len = len }
    | Rs (start, stop, step) ->
        if step = 0 then
          Error.invalid ~op:"slice" ~what:"step" ~reason:"cannot be zero"
            ~hint:
              "use positive step for forward slicing or negative for reverse"
            ();
        let s = if start < 0 then dim_size + start else start in
        let e = if stop < 0 then dim_size + stop else stop in
        (* Python/Numpy style range logic for steps *)
        let len, actual_stop =
          if step > 0 then
            let s = Int.max 0 (Int.min s dim_size) in
            let e = Int.max 0 (Int.min e dim_size) in
            let len = if s >= e then 0 else ((e - 1 - s) / step) + 1 in
            (len, e)
          else
            let s = Int.min (dim_size - 1) (Int.max (-1) s) in
            let e = Int.min (dim_size - 1) (Int.max (-1) e) in
            let len = if s <= e then 0 else ((s - e - 1) / -step) + 1 in
            (len, e)
        in
        View { start = s; stop = actual_stop; step; dim_len = len }
    | L indices ->
        let arr = Array.of_list indices in
        let arr_norm =
          Array.map (normalize_and_check_index ~op:"slice" dim_size) arr
        in
        Gather arr_norm
    | N -> New_axis
    | M _ -> failwith "Mask slicing not supported in slice_internal"

  let slice_internal specs x =
    let@ _ = span ~op:"slice" () in
    let input_shape = shape x in
    let ndim_in = Array.length input_shape in

    (* Parse and normalize specs *)
    let ops, consumed_dims =
      List.fold_left
        (fun (acc, dim_idx) spec ->
          match spec with
          | N -> (New_axis :: acc, dim_idx)
          | _ ->
              if dim_idx >= ndim_in then
                Error.invalid ~op:"slice" ~what:"too many indices" ~reason:"" ();
              let op = normalize_slice_spec input_shape.(dim_idx) spec in
              (op :: acc, dim_idx + 1))
        ([], 0) specs
    in
    (* Append implicit A for remaining dimensions *)
    let ops =
      let rec add_trailing acc dim_idx =
        if dim_idx >= ndim_in then List.rev acc
        else
          let op = normalize_slice_spec input_shape.(dim_idx) A in
          add_trailing (op :: acc) (dim_idx + 1)
      in
      add_trailing ops consumed_dims
    in

    let gather_along_axis axis indices t =
      let idx_tensor =
        init (B.context t) Dtype.int32
          [| Array.length indices |]
          (fun i -> Int32.of_int indices.(i.(0)))
      in
      take ~axis idx_tensor t
    in

    let slice_along_axis axis start stop t =
      let t_shape = shape t in
      if start < stop then
        let ranges =
          Array.mapi
            (fun i dim -> if i = axis then (start, stop) else (0, dim))
            t_shape
        in
        B.shrink t ranges
      else
        let idx = empty (B.context t) Dtype.int32 [| 0 |] in
        take ~axis idx t
    in

    let rec apply_ops current axis squeeze_axes = function
      | [] -> (current, squeeze_axes)
      | op :: rest -> (
          match op with
          | New_axis ->
              let current' = unsqueeze ~axes:[ axis ] current in
              apply_ops current' (axis + 1) squeeze_axes rest
          | Squeeze { idx } ->
              let current' = slice_along_axis axis idx (idx + 1) current in
              apply_ops current' (axis + 1) (axis :: squeeze_axes) rest
          | Gather indices ->
              let current' = gather_along_axis axis indices current in
              apply_ops current' (axis + 1) squeeze_axes rest
          | View { start; step; dim_len; _ } ->
              if step = 1 then
                let current' =
                  slice_along_axis axis start (start + dim_len) current
                in
                apply_ops current' (axis + 1) squeeze_axes rest
              else if step = -1 then
                let current' =
                  if dim_len = 0 then slice_along_axis axis 0 0 current
                  else
                    let lo = start - dim_len + 1 in
                    let hi = start + 1 in
                    let sliced = slice_along_axis axis lo hi current in
                    let flip_axes = Array.make (ndim sliced) false in
                    flip_axes.(axis) <- true;
                    B.flip sliced flip_axes
                in
                apply_ops current' (axis + 1) squeeze_axes rest
              else
                let indices =
                  Array.init dim_len (fun i -> start + (i * step))
                in
                let current' = gather_along_axis axis indices current in
                apply_ops current' (axis + 1) squeeze_axes rest)
    in

    let result, squeeze_axes = apply_ops x 0 [] ops in
    match List.sort_uniq compare squeeze_axes with
    | [] -> result
    | axes -> squeeze ~axes result

  (* Optimized set_slice_internal using vectorized index calculation *)
  let set_slice_internal specs x y =
    let@ _ = span ~op:"set_slice" () in
    let x_shape = shape x in
    let ndim = Array.length x_shape in

    (* Pad specs with A *)
    let full_specs =
      if List.length specs < ndim then
        specs @ List.init (ndim - List.length specs) (fun _ -> A)
      else specs
    in

    (* Check if this is a simple view-compatible slice (no lists/gather) *)
    let is_simple_view =
      List.for_all
        (function
          | L _ | M _ -> false | Rs (_, _, step) -> Int.abs step = 1 | _ -> true)
        full_specs
    in

    if is_simple_view then
      (* FAST PATH: View-based assignment *)
      let target_view = slice_internal full_specs x in
      let y_b = broadcast_to (shape target_view) y in
      B.assign target_view y_b
    else
      (* SLOW PATH: Scatter-based assignment for fancy indexing *)
      (* Calculate logical strides for x (row-major) *)
      let logical_strides = Array.make ndim 1 in
      for i = ndim - 2 downto 0 do
        logical_strides.(i) <- logical_strides.(i + 1) * x_shape.(i + 1)
      done;

      let ctx = B.context x in

      (* Parse specs into (is_squeezed, tensor_indices) *)
      let dims_info =
        List.mapi
          (fun i spec ->
            match normalize_slice_spec x_shape.(i) spec with
            | Squeeze { idx } ->
                (true, scalar ctx Dtype.int32 (Int32.of_int idx))
            | View { start; stop; step; _ } ->
                let idx_tensor = arange ctx Dtype.int32 start stop step in
                (false, idx_tensor)
            | Gather indices ->
                let idx_tensor =
                  init ctx Dtype.int32
                    [| Array.length indices |]
                    (fun k -> Int32.of_int indices.(k.(0)))
                in
                (false, idx_tensor)
            | New_axis ->
                failwith
                  "New_axis not supported in set_slice (use expand on Y first)")
          full_specs
      in

      (* Calculate target shape (orthogonal indexing) *)
      let target_shape_list =
        List.filter_map
          (fun (is_squeezed, t) -> if is_squeezed then None else Some (numel t))
          dims_info
      in
      let target_shape = Array.of_list target_shape_list in
      let target_rank = Array.length target_shape in

      (* Construct flat indices via broadcasting *)
      let flat_indices_acc = ref (scalar ctx Dtype.int32 0l) in
      let current_target_dim = ref 0 in

      List.iteri
        (fun i (is_squeezed, idx_tensor) ->
          let stride_val = Int32.of_int logical_strides.(i) in
          let weighted_idx =
            if stride_val = 1l then idx_tensor
            else mul idx_tensor (scalar ctx Dtype.int32 stride_val)
          in

          if is_squeezed then
            flat_indices_acc := add !flat_indices_acc weighted_idx
          else
            let reshape_spec = Array.make target_rank 1 in
            reshape_spec.(!current_target_dim) <- numel idx_tensor;
            let reshaped_weighted_idx = reshape reshape_spec weighted_idx in
            flat_indices_acc := add !flat_indices_acc reshaped_weighted_idx;
            incr current_target_dim)
        dims_info;

      (* Perform scatter *)
      let x_flat = reshape [| numel x |] x in
      let indices_flat =
        reshape [| numel !flat_indices_acc |] !flat_indices_acc
      in

      let y_b = broadcast_to target_shape y in
      let y_flat = reshape [| numel y_b |] y_b in

      let result_flat =
        B.scatter ~mode:`Set ~unique_indices:false x_flat ~indices:indices_flat
          ~updates:y_flat ~axis:0
      in
      let result_reshaped = reshape x_shape result_flat in
      B.assign x result_reshaped

  (* Get a single element or sub-tensor *)
  let get indices x =
    let@ _ = span ~op:"get" () in
    let x_shape = shape x in
    (* Normalize negative indices and check bounds *)
    let normalized_indices =
      List.mapi
        (fun dim idx ->
          if dim >= Array.length x_shape then
            Error.invalid ~op:"get" ~what:"indices"
              ~reason:(Format.asprintf "too many for shape %a" Shape.pp x_shape)
              ()
          else
            let normalized_idx = normalize_index x_shape.(dim) idx in
            if normalized_idx < 0 || normalized_idx >= x_shape.(dim) then
              Error.invalid ~op:"get"
                ~what:
                  (Printf.sprintf "index [%s]"
                     (String.concat "," (List.map string_of_int indices)))
                ~reason:
                  (Printf.sprintf "out of bounds for shape %s"
                     (Shape.to_string x_shape))
                ~hint:
                  (Printf.sprintf "index %d at dim %d: %d ∉ [0, %d)" dim dim
                     normalized_idx x_shape.(dim))
                ()
            else normalized_idx)
        indices
    in
    slice_internal (List.map (fun i -> I i) normalized_indices) x

  (* Set a single element or sub-tensor *)
  let set indices x value =
    let@ _ = span ~op:"set" () in
    let x_shape = shape x in
    (* Normalize negative indices and check bounds *)
    let normalized_indices =
      List.mapi
        (fun dim idx ->
          if dim >= Array.length x_shape then
            Error.invalid ~op:"set" ~what:"indices"
              ~reason:(Format.asprintf "too many for shape %a" Shape.pp x_shape)
              ()
          else
            let normalized_idx = normalize_index x_shape.(dim) idx in
            if normalized_idx < 0 || normalized_idx >= x_shape.(dim) then
              Error.invalid ~op:"set"
                ~what:(Printf.sprintf "index %d at dimension %d" idx dim)
                ~reason:
                  (Format.asprintf "out of bounds for shape %a" Shape.pp x_shape)
                ~hint:
                  (Printf.sprintf "index %d at dim %d: %d ∉ [0, %d)" dim dim
                     normalized_idx x_shape.(dim))
                ()
            else normalized_idx)
        indices
    in
    set_slice_internal (List.map (fun i -> I i) normalized_indices) x value

  let unsafe_get indices x =
    (* Get the element at the specified indices *)
    let scalar_tensor = get indices x in
    (* For a scalar tensor, we need to read the single element *)
    let ba = data scalar_tensor in
    (* The scalar tensor should be 0-dimensional or have been squeezed to
       scalar *)
    if numel scalar_tensor <> 1 then
      Error.failed ~op:"unsafe_get" ~what:"expected scalar result"
        ~reason:(Printf.sprintf "got %d elements" (numel scalar_tensor))
        ();

    (* For scalar tensors, there are two cases: *)
    match View.strides_opt (B.view scalar_tensor) with
    | Some _ ->
        (* Has valid strides - use the offset *)
        let view_offset = offset scalar_tensor in
        Nx_buffer.get ba view_offset
    | None ->
        (* Non-composable views - the scalar should have been materialized by get *)
        (* If it's truly a scalar with 1 element, it should be at index 0 *)
        if Nx_buffer.length ba = 1 then Nx_buffer.get ba 0
        else
          Error.failed ~op:"unsafe_get"
            ~what:"cannot read from non-composable scalar view"
            ~hint:"this is likely a bug in get/slice implementation" ()

  let unsafe_set indices value x =
    let scalar_tensor = scalar (B.context x) (dtype x) value in
    set indices x scalar_tensor

  (* Public slicing API using the new index type *)
  let slice specs t = slice_internal specs t
  let set_slice specs t value = set_slice_internal specs t value

  let item indices t =
    (* item returns a scalar value, not a tensor *)
    let t_shape = shape t in
    if List.length indices <> Array.length t_shape then
      invalid_arg
        (Printf.sprintf "item: need %d indices for %d-d tensor, got %d"
           (Array.length t_shape) (Array.length t_shape) (List.length indices));

    (* Get the scalar tensor *)
    let scalar_t = get indices t in

    (* Extract the value *)
    unsafe_get [] scalar_t

  let set_item indices value t =
    (* set_item sets a scalar value *)
    let t_shape = shape t in
    if List.length indices <> Array.length t_shape then
      invalid_arg
        (Printf.sprintf "set_item: need %d indices for %d-d tensor, got %d"
           (Array.length t_shape) (Array.length t_shape) (List.length indices));

    unsafe_set indices value t

  let put ?axis ~indices ~values ?(mode = `raise) t =
    let@ _ = span ~op:"put" () in
    (* Convert indices to int32 if needed *)
    let indices =
      if dtype indices = Int32 then indices else astype Int32 indices
    in
    let context = B.context t in

    match axis with
    | None ->
        (* Flatten t and scatter *)
        let t_shape_orig = shape t in
        let t_flat = reshape [| numel t |] t in

        (* Process indices based on mode *)
        let indices_processed =
          match mode with
          | `raise -> indices
          | `wrap ->
              let n = numel t in
              mod_ indices (scalar (B.context indices) Int32 (Int32.of_int n))
          | `clip ->
              let max_idx = numel t - 1 in
              minimum
                (maximum indices (zeros context Int32 (shape indices)))
                (full context Int32 (shape indices) (Int32.of_int max_idx))
        in

        (* Flatten indices and values *)
        let indices_flat = reshape [| numel indices |] indices_processed in
        let values_flat = reshape [| numel values |] values in

        (* Scatter and reshape back *)
        let result =
          B.scatter ~mode:`Set ~unique_indices:false t_flat
            ~indices:indices_flat ~updates:values_flat ~axis:0
        in
        blit (reshape t_shape_orig result) t
    | Some axis ->
        let axis = resolve_single_axis t axis in

        (* Process indices based on mode *)
        let indices_processed =
          match mode with
          | `raise -> indices
          | `wrap ->
              let n = dim axis t in
              mod_ indices (scalar (B.context indices) Int32 (Int32.of_int n))
          | `clip ->
              let max_idx = dim axis t - 1 in
              minimum
                (maximum indices (zeros context Int32 (shape indices)))
                (full context Int32 (shape indices) (Int32.of_int max_idx))
        in

        (* Use scatter along axis *)
        let result =
          B.scatter ~mode:`Set ~unique_indices:false t
            ~indices:indices_processed ~updates:values ~axis
        in
        blit result t

  let index_put ~indices ~values ?(mode = `raise) t =
    let@ _ = span ~op:"index_put" () in
    let context = B.context t in
    let t_shape = shape t in
    let ndim = Array.length t_shape in

    if ndim = 0 then
      Error.invalid ~op:"index_put" ~what:"tensor rank"
        ~reason:"cannot index into scalar tensor" ();

    let indices_count = Array.length indices in
    if indices_count <> ndim then
      Error.invalid ~op:"index_put" ~what:"indices"
        ~reason:
          (Printf.sprintf "expected %d index tensors, got %d" ndim indices_count)
        ();

    (* Ensure all indices are int32 and broadcastable to a common shape *)
    let indices_int32 =
      Array.map
        (fun idx -> if dtype idx = Int32 then idx else astype Int32 idx)
        indices
    in
    let indices_broadcasted =
      indices_int32 |> Array.to_list |> broadcast_arrays |> Array.of_list
    in

    (* Apply bounds handling per-axis and validate zero-sized axes *)
    let indices_processed =
      Array.mapi
        (fun axis idx ->
          let axis_size = t_shape.(axis) in
          let has_updates = numel idx <> 0 in
          if axis_size = 0 && has_updates then
            Error.invalid ~op:"index_put"
              ~what:(Printf.sprintf "axis %d" axis)
              ~reason:"cannot index into zero-sized dimension" ();

          if not has_updates then idx
          else
            match mode with
            | `raise -> idx
            | `wrap ->
                if axis_size = 0 then idx
                else
                  let modulus_scalar =
                    scalar (B.context idx) Int32 (Int32.of_int axis_size)
                  in
                  let modulus = broadcast_to (shape idx) modulus_scalar in
                  let wrapped = mod_ idx modulus in
                  let zeros_idx = zeros (B.context idx) Int32 (shape idx) in
                  let needs_fix = cmplt wrapped zeros_idx in
                  let wrapped_plus_mod = add wrapped modulus in
                  where needs_fix wrapped_plus_mod wrapped
            | `clip ->
                if axis_size = 0 then idx
                else
                  let zeros_idx = zeros (B.context idx) Int32 (shape idx) in
                  let max_idx =
                    full (B.context idx) Int32 (shape idx)
                      (Int32.of_int (axis_size - 1))
                  in
                  minimum (maximum idx zeros_idx) max_idx)
        indices_broadcasted
    in

    let target_shape = shape indices_processed.(0) in
    let num_updates = array_prod target_shape in

    if num_updates = 0 then ()
    else
      let values =
        if shape values = target_shape then values
        else broadcast_to target_shape values
      in

      let strides = Shape.c_contiguous_strides t_shape in
      let flat_indices =
        let acc = ref (zeros context Int32 target_shape) in
        for axis = 0 to ndim - 1 do
          let idx = indices_processed.(axis) in
          let stride = strides.(axis) in
          let contribution =
            if stride = 0 || stride = 1 then idx
            else
              let stride_tensor =
                full context Int32 target_shape (Int32.of_int stride)
              in
              mul idx stride_tensor
          in
          acc := add !acc contribution
        done;
        !acc
      in

      (* Flatten scatter into the original tensor *)
      put ~indices:flat_indices ~values ~mode:`raise t

  let put_along_axis ~axis ~indices ~values t =
    let@ _ = span ~op:"put_along_axis" () in
    let axis = resolve_single_axis t axis in

    (* Check shape compatibility *)
    let t_shape = shape t in
    let idx_shape = shape indices in
    let val_shape = shape values in

    if Array.length t_shape <> Array.length idx_shape then
      Error.shape_mismatch ~op:"put_along_axis" ~expected:t_shape
        ~actual:idx_shape ();

    (* Broadcast values if needed *)
    let values =
      if val_shape = idx_shape then values else broadcast_to idx_shape values
    in

    (* Use scatter *)
    let result =
      B.scatter ~mode:`Set ~unique_indices:false t ~indices ~updates:values
        ~axis
    in
    blit result t

  (* Note: compress, nonzero, and argwhere have data-dependent output shapes and
     are NOT differentiable. They use unsafe_get to materialize values. *)

  (* Forward declaration for mutual recursion *)
  let nonzero_indices_only (condition : (bool, bool_elt) t) =
    (* Special version for compress that only returns indices for boolean
       masks *)
    let total = numel condition in
    let cond_flat = reshape [| total |] condition in

    (* Count non-zeros *)
    let n_nonzero =
      let sum_result = sum (astype Int32 cond_flat) in
      let scalar_val = squeeze sum_result |> unsafe_get [] in
      Int32.to_int scalar_val
    in

    if n_nonzero = 0 then [| empty (B.context condition) Int32 [| 0 |] |]
    else
      (* Build indices of non-zero positions *)
      let indices =
        create (B.context condition) Int32 [| n_nonzero |]
          (Array.make n_nonzero 0l)
      in
      let idx = ref 0 in
      for i = 0 to total - 1 do
        let elem_val = unsafe_get [ i ] cond_flat in
        if elem_val then (
          set_item [ !idx ] (Int32.of_int i) indices;
          incr idx)
      done;
      [| indices |]

  let compress ?axis ~(condition : (bool, bool_elt) t) t =
    let@ _ = span ~op:"compress" () in
    match axis with
    | None ->
        (* Flatten and compress *)
        let t_flat = flatten t in
        let cond_flat = flatten condition in

        (* Count true values *)
        let n_true =
          sum ~axes:[ 0 ] (astype Int32 cond_flat)
          |> squeeze |> unsafe_get [] |> Int32.to_int
        in

        if n_true = 0 then empty (B.context t) (dtype t) [| 0 |]
        else
          (* Get indices where condition is true *)
          let indices = nonzero_indices_only cond_flat in
          take indices.(0) t_flat
    | Some axis ->
        let axis = resolve_single_axis t axis in
        let axis_size = dim axis t in

        (* Check condition length *)
        if numel condition <> axis_size then
          invalid_arg
            (Printf.sprintf "compress: length %d doesn't match axis %d size %d"
               (numel condition) axis axis_size);

        (* Get indices where condition is true *)
        let cond_1d = reshape [| axis_size |] condition in
        let true_indices = nonzero_indices_only cond_1d in

        if Array.length true_indices = 0 || numel true_indices.(0) = 0 then (
          (* No true values - return empty tensor *)
          let new_shape = Array.copy (shape t) in
          new_shape.(axis) <- 0;
          empty (B.context t) (dtype t) new_shape)
        else take ~axis true_indices.(0) t

  let extract ~condition t =
    let@ _ = span ~op:"extract" () in
    (* Check shape compatibility *)
    if shape condition <> shape t then invalid_arg "extract: shape mismatch";

    compress ~condition (flatten t)

  let nonzero (type a b) (t : (a, b) t) =
    let@ _ = span ~op:"nonzero" () in
    (* Returns array of coordinate tensors *)
    let t_shape = shape t in
    let ndim = Array.length t_shape in

    (* Create a mask of non-zero elements *)
    let zero = zeros (B.context t) (dtype t) [| 1 |] in
    let mask = not_equal t (broadcast_to t_shape zero) in

    (* Find all non-zero positions *)
    let total = numel mask in
    let mask_flat = reshape [| total |] mask in

    (* Count non-zeros - mask is boolean (true or false) *)
    let n_nonzero =
      let sum_result = sum (astype Int32 mask_flat) in
      let scalar_val = squeeze sum_result |> unsafe_get [] in
      Int32.to_int scalar_val
    in

    if n_nonzero = 0 then
      (* No non-zero elements *)
      Array.init ndim (fun _ -> empty (B.context t) Int32 [| 0 |])
    else
      (* Generate coordinate arrays *)
      let coords =
        Array.init ndim (fun _ ->
            create (B.context t) Int32 [| n_nonzero |] (Array.make n_nonzero 0l))
      in

      (* Create indices for each dimension *)
      let idx = ref 0 in
      let rec process_indices pos dim_idx =
        if dim_idx = ndim then (
          (* Check if this position is non-zero *)
          let indices = Array.to_list pos in
          let elem = get indices t in

          (* Check if element is non-zero - elem is already a scalar *)
          (* For differentiability, we should compare tensors, not extract values *)
          let zero_scalar = zeros (B.context t) (dtype t) (shape elem) in
          let is_nonzero_tensor = not_equal elem zero_scalar in
          (* We need to materialize this to iterate - this breaks
             differentiability *)
          let is_nonzero = unsafe_get [] is_nonzero_tensor <> false in

          if is_nonzero then (
            (* Store coordinates *)
            for i = 0 to ndim - 1 do
              let coord_arr = coords.(i) in
              set_item [ !idx ] (Int32.of_int pos.(i)) coord_arr
            done;
            incr idx))
        else
          for i = 0 to t_shape.(dim_idx) - 1 do
            pos.(dim_idx) <- i;
            process_indices pos (dim_idx + 1)
          done
      in

      let pos = Array.make ndim 0 in
      process_indices pos 0;

      (* Resize arrays to actual number of non-zeros found *)
      Array.map (fun coord -> slice [ Rs (0, !idx, 1) ] coord) coords

  let argwhere t =
    (* Returns 2D tensor of coordinates *)
    let coords = nonzero t in
    if Array.length coords = 0 then empty (B.context t) Int32 [| 0; 0 |]
    else
      let n_nonzero = dim 0 coords.(0) in
      let ndim = Array.length coords in

      if n_nonzero = 0 then empty (B.context t) Int32 [| 0; ndim |]
      else
        (* Stack coordinates into 2D array *)
        let result = zeros (B.context t) Int32 [| n_nonzero; ndim |] in
        for i = 0 to ndim - 1 do
          (* Select column i and set values *)
          let col_slice = slice_internal [ A; I i ] result in
          let coord_values = flatten coords.(i) in
          blit coord_values col_slice
        done;
        result

  let array_split ~axis sections x =
    let ndim = ndim x in
    let axis = resolve_single_axis x axis in
    let axis_size = dim axis x in

    match sections with
    | `Indices indices ->
        (* Split at specific indices *)
        let indices = Array.of_list indices in
        let n_sections = Array.length indices + 1 in
        let splits = Array.make n_sections x in

        (* Add boundaries *)
        let boundaries = Array.make (n_sections + 1) 0 in
        boundaries.(0) <- 0;
        Array.iteri (fun i idx -> boundaries.(i + 1) <- idx) indices;
        boundaries.(n_sections) <- axis_size;

        (* Create slices *)
        for i = 0 to n_sections - 1 do
          let start = boundaries.(i) in
          let stop = boundaries.(i + 1) in

          if start < stop then
            let slice_spec =
              List.init ndim (fun j -> if j = axis then R (start, stop) else A)
            in
            splits.(i) <- slice_internal slice_spec x
          else
            (* Empty slice *)
            let empty_shape = Array.copy (shape x) in
            empty_shape.(axis) <- 0;
            splits.(i) <- empty (B.context x) (dtype x) empty_shape
        done;
        Array.to_list splits
    | `Count n ->
        (* Split into n sections *)
        if n <= 0 then
          Error.check_bounds ~op:"array_split" ~name:"sections" ~value:n ~min:1
            ();

        let base_size = axis_size / n in
        let remainder = axis_size mod n in

        (* Calculate section sizes *)
        let sizes = Array.make n base_size in
        for i = 0 to remainder - 1 do
          sizes.(i) <- sizes.(i) + 1
        done;

        (* Create slices *)
        let splits = Array.make n x in
        let start = ref 0 in

        for i = 0 to n - 1 do
          let size = sizes.(i) in
          let stop = !start + size in

          let slice_spec =
            List.init ndim (fun j -> if j = axis then R (!start, stop) else A)
          in
          splits.(i) <- slice_internal slice_spec x;
          start := stop
        done;

        Array.to_list splits

  let split ~axis sections x =
    let axis = resolve_single_axis x axis in
    let axis_size = dim axis x in

    if axis_size mod sections <> 0 then
      Error.cannot ~op:"split" ~what:"divide evenly"
        ~from:(Printf.sprintf "axis %d (size %d)" axis axis_size)
        ~to_:(Printf.sprintf "%d sections" sections)
        ~reason:
          (Printf.sprintf "%d %% %d = %d" axis_size sections
             (axis_size mod sections))
        ~hint:"use array_split for uneven division" ();

    array_split ~axis (`Count sections) x

  (* ───── Sorting and Searching ───── *)

  let sort (type a b) ?(descending = false) ?(axis = -1) (x : (a, b) t) =
    let@ _ = span ~op:"sort" () in
    if ndim x = 0 then (x, scalar (B.context x) Dtype.int32 0l)
    else
      let r = ndim x in
      let axis = if axis < 0 then axis + r else axis in
      if axis < 0 || axis >= r then
        Error.axis_out_of_bounds ~op:"sort" ~axis ~ndim:r ();
      let out_sorted = empty (B.context x) (dtype x) (shape x) in
      let out_indices = empty (B.context x) Dtype.int32 (shape x) in
      B.sort ~out:out_sorted ~axis ~descending x;
      B.argsort ~out:out_indices ~axis ~descending x;
      (out_sorted, out_indices)

  let argsort ?(descending = false) ?(axis = -1) x =
    let@ _ = span ~op:"argsort" () in
    let _, indices = sort ~descending ~axis x in
    indices

  let argmax ?axis ?(keepdims = false) x =
    let@ _ = span ~op:"argmax" () in
    let x', axis =
      match axis with
      | None -> (flatten x, 0)
      | Some axis ->
          let rank = ndim x in
          let axis = resolve_single_axis ~ndim_opt:rank x axis in
          if axis < 0 || axis >= rank then
            Error.axis_out_of_bounds ~op:"argmax" ~axis ~ndim:rank ();
          (x, axis)
    in
    let out_shape = reduce_output_shape (shape x') [| axis |] keepdims in
    let out = empty (B.context x) Dtype.int32 out_shape in
    B.argmax ~out ~axis ~keepdims x';
    out

  let argmin (type a b) ?axis ?(keepdims = false) (x : (a, b) t) :
      (int32, Dtype.int32_elt) t =
    let@ _ = span ~op:"argmin" () in
    let x', axis =
      match axis with
      | None -> (flatten x, 0)
      | Some axis ->
          let rank = ndim x in
          let axis = resolve_single_axis ~ndim_opt:rank x axis in
          if axis < 0 || axis >= rank then
            Error.axis_out_of_bounds ~op:"argmin" ~axis ~ndim:rank ();
          (x, axis)
    in
    let out_shape = reduce_output_shape (shape x') [| axis |] keepdims in
    let out = empty (B.context x) Dtype.int32 out_shape in
    B.argmin ~out ~axis ~keepdims x';
    out

  (* ───── Random Number Generation ───── *)

  module Rng = struct
    include Core_rng

    let tensor_shape = shape

    (* Validate parameters for random functions *)
    let validate_random_params fname dtype shape =
      if not (Dtype.is_float dtype) then
        Error.invalid ~op:fname
          ~what:(Printf.sprintf "dtype %s" (Dtype.to_string dtype))
          ~reason:"not a float type"
          ~hint:"uniform/normal only support Float16, Float32, Float64" ();
      if Array.exists (fun x -> x < 0) shape then
        Error.invalid_shape ~op:fname ~shape
          ~reason:"dimensions must be non-negative" ()

    let uniform ctx ~key dtype shape =
      let@ _ = span ~op:"uniform" () in
      validate_random_params "uniform" dtype shape;

      (* If shape has 0, return zeros *)
      let numel = array_prod shape in
      if numel = 0 then zeros ctx dtype shape
      else
        (* Generate random int32 values using threefry *)
        (* Threefry2x32 requires inputs with shape [..., 2] *)
        let num_values = numel in

        (* Create key and counter tensors for threefry *)
        (* Each vector needs 2 int32 values, so we create arrays with shape
           [num_values, 2] *)
        let key_vals =
          Array.init (num_values * 2) (fun i ->
              Int32.of_int (Core_rng.fold_in key i))
        in
        let key = create ctx Dtype.int32 [| num_values; 2 |] key_vals in

        let ctr_vals = Array.init (num_values * 2) (fun i -> Int32.of_int i) in
        let counter = create ctx Dtype.int32 [| num_values; 2 |] ctr_vals in

        (* Generate random bits using threefry *)
        let random_bits = empty ctx Dtype.int32 [| num_values; 2 |] in
        B.threefry ~out:random_bits key counter;

        (* Flatten and take only what we need *)
        let bits_flat = flatten random_bits in
        let bits_needed =
          if numel < size bits_flat then shrink [| (0, numel) |] bits_flat
          else bits_flat
        in

        (* Convert to float32 - Metal doesn't support float64 on most devices *)
        let bits_float32 = cast Dtype.float32 bits_needed in

        (* Add 2^31 to shift from signed [-2^31, 2^31-1] to unsigned [0, 2^32-1]
           range *)
        let offset = scalar ctx Dtype.float32 2147483648.0 in
        (* 2^31 *)
        let shifted = add bits_float32 offset in

        (* Normalize to [0, 1) by dividing by 2^32 *)
        let normalizer = scalar ctx Dtype.float32 4294967296.0 in
        (* 2^32 *)
        let normalized = div shifted normalizer in

        (* Cast to target dtype *)
        let result = cast dtype normalized in

        (* Reshape to final shape *)
        reshape shape result

    let normal ctx ~key dtype shape =
      let@ _ = span ~op:"normal" () in
      validate_random_params "normal" dtype shape;

      (* If shape has 0, return zeros *)
      let numel = array_prod shape in
      if numel = 0 then zeros ctx dtype shape
      else
        let base_key = key in
        let key_pair =
          match Core_rng.split ~n:2 base_key with
          | [| k1; k2 |] -> (k1, k2)
          | _ -> assert false
        in
        let key_u1, key_u2 = key_pair in
        (* Box-Muller transform: generate pairs of uniform random values *)
        (* Generate two sets of uniform random values *)
        let u1 = uniform ctx ~key:key_u1 Dtype.float32 shape in
        let u2 = uniform ctx ~key:key_u2 Dtype.float32 shape in

        (* Box-Muller transform: z0 = cos(2π * u1) * sqrt(-2 * ln(u2)) We use u2
           for the log to avoid log(0) *)

        (* Compute 2π * u1 *)
        let two_pi = scalar ctx Dtype.float32 (2.0 *. Float.pi) in
        let angle = mul u1 two_pi in

        (* Compute cos(2π * u1) *)
        let cos_part = cos angle in

        (* Compute sqrt(-2 * ln(u2)) *)
        (* First ensure u2 is not exactly 0 by using 1 - original_uniform *)
        let one = ones_like u2 in
        let u2_safe = sub one u2 in
        (* Now in [0, 1) *)

        (* Add small epsilon to avoid log(0) *)
        let eps = scalar ctx Dtype.float32 1e-7 in
        let u2_nonzero = maximum u2_safe eps in

        let log_u2 = log u2_nonzero in
        let neg_two = scalar ctx Dtype.float32 (-2.0) in
        let sqrt_arg = mul neg_two log_u2 in
        let sqrt_part = sqrt sqrt_arg in

        (* Combine: z0 = cos_part * sqrt_part *)
        let result_f32 = mul cos_part sqrt_part in

        (* Cast to target dtype *)
        cast dtype result_f32

    let randint ctx dtype ~key ?(high = 10) shape low =
      if low >= high then
        Error.invalid ~op:"randint" ~what:"range"
          ~reason:(Printf.sprintf "low=%d ≥ high=%d" low high)
          ();
      if not (Dtype.is_int dtype) then
        Error.invalid ~op:"randint" ~what:"dtype"
          ~reason:"only integer dtypes supported" ();
      let range = high - low in
      let uniform_vals = uniform ctx ~key Dtype.float32 shape in
      let scaled =
        mul uniform_vals (scalar ctx Dtype.float32 (float_of_int range))
      in
      let shifted = add scaled (scalar ctx Dtype.float32 (float_of_int low)) in
      astype dtype shifted

    let bernoulli ctx ~key ~p shape =
      if p < 0.0 || p > 1.0 then
        Error.invalid ~op:"bernoulli" ~what:"p" ~reason:"must be in [0, 1]" ();
      if Array.exists (fun x -> x < 0) shape then
        Error.invalid_shape ~op:"bernoulli" ~shape
          ~reason:"dimensions must be non-negative" ();
      let u = uniform ctx ~key Dtype.float32 shape in
      let threshold = scalar ctx Dtype.float32 p in
      cmplt u threshold

    let permutation ctx ~key n =
      if n <= 0 then
        Error.invalid ~op:"permutation" ~what:"n" ~reason:"must be positive" ();
      let random_vals = uniform ctx ~key Dtype.float32 [| n |] in
      argsort random_vals ~axis:0 ~descending:false

    let shuffle ctx ~key x =
      let shape_x = tensor_shape x in
      if Array.length shape_x = 0 then x
      else
        let n = shape_x.(0) in
        let perm = permutation ctx ~key n in
        take ~axis:0 perm x

    let categorical (type a b) ctx ~key ?(axis = -1) ?(shape : int array = [||])
        (logits : (a, b) t) =
      let logits_dtype = dtype logits in
      let logits_shape = tensor_shape logits in
      if not (Dtype.is_float logits_dtype) then
        Error.invalid ~op:"categorical" ~what:"logits"
          ~reason:"requires floating point dtype" ();
      let ndim = Array.length logits_shape in
      let axis = if axis < 0 then ndim + axis else axis in
      if axis < 0 || axis >= ndim then
        Error.axis_out_of_bounds ~op:"categorical" ~axis ~ndim ();
      let full_shape = Array.append shape logits_shape in

      let run_float float_dtype eps =
        let u = uniform ctx ~key float_dtype full_shape in
        let u_clamped = clip u ~min:eps ~max:(1. -. eps) in
        let neg_one = scalar ctx float_dtype (-1.0) in
        let log_u = log u_clamped in
        let neg_log_u = mul log_u neg_one in
        let log_neg_log_u = log neg_log_u in
        let gumbel = mul log_neg_log_u neg_one |> astype logits_dtype in
        let noisy = add logits gumbel in
        let prefix_len = Array.length shape in
        let argmax_axis = axis + prefix_len in
        let inds = argmax noisy ~axis:argmax_axis ~keepdims:false in
        astype Dtype.int32 inds
      in

      match logits_dtype with
      | Float64 -> run_float Dtype.float64 1e-12
      | Float32 -> run_float Dtype.float32 1e-6
      | Float16 -> run_float Dtype.float32 1e-3
      | BFloat16 -> run_float Dtype.float32 1e-2
      | Float8_e4m3 | Float8_e5m2 ->
          Error.invalid ~op:"categorical" ~what:"logits"
            ~reason:"float8 logits not supported" ()
      | _ ->
          Error.invalid ~op:"categorical" ~what:"logits"
            ~reason:"requires floating point dtype" ()

    let truncated_normal (type a b) ctx ~key (dtype : (a, b) Dtype.t) ~lower
        ~upper shape =
      if lower >= upper then
        Error.invalid ~op:"truncated_normal" ~what:"bounds"
          ~reason:"lower must be less than upper" ();
      let supported =
        match dtype with
        | Float16 | Float32 | Float64 | BFloat16 -> true
        | _ -> false
      in
      if not supported then
        Error.invalid ~op:"truncated_normal" ~what:"dtype"
          ~reason:"must be floating point" ();

      let scalar_lower = scalar ctx Dtype.float64 lower |> astype dtype in
      let scalar_upper = scalar ctx Dtype.float64 upper |> astype dtype in

      let split2 k =
        match Core_rng.split ~n:2 k with
        | [| a; b |] -> (a, b)
        | _ -> assert false
      in

      let has_remaining mask =
        let any_mask = any mask in
        let arr = to_array any_mask in
        match arr with [| v |] -> v | _ -> false
      in

      let max_attempts = 1000 in
      let sample_key, next_key = split2 key in
      let initial = normal ctx ~key:sample_key dtype shape in
      let within_lower = greater_equal initial scalar_lower in
      let within_upper = less_equal initial scalar_upper in
      let accepted = logical_and within_lower within_upper in
      let remaining = logical_not accepted in

      let rec fill k acc remaining attempt =
        if not (has_remaining remaining) then acc
        else if attempt > max_attempts then
          Error.invalid ~op:"truncated_normal" ~what:"generation"
            ~reason:
              (Printf.sprintf
                 "failed to find samples within bounds after %d tries"
                 max_attempts)
            ()
        else
          let resample_key, next_k = split2 k in
          let candidates = normal ctx ~key:resample_key dtype shape in
          let within_lower = greater_equal candidates scalar_lower in
          let within_upper = less_equal candidates scalar_upper in
          let within = logical_and within_lower within_upper in
          let take_new = logical_and remaining within in
          let acc = where take_new candidates acc in
          let still_remaining = logical_and remaining (logical_not within) in
          fill next_k acc still_remaining (attempt + 1)
      in
      fill next_key initial remaining 1
  end

  let rand ctx dtype ~key shape = Rng.uniform ctx ~key dtype shape
  let randn ctx dtype ~key shape = Rng.normal ctx ~key dtype shape

  let randint ctx dtype ~key ?high shape low =
    Rng.randint ctx dtype ~key ?high shape low

  (* ───── Linear Algebra ───── *)

  (* Compute matmul output shape: for a[..., m, k] @ b[..., k, n] -> [..., m, n]
     where batch dims are broadcast together. Both a and b must be at least
     2D. *)
  let matmul_output_shape a b =
    let shape_a = shape_symbolic a in
    let shape_b = shape_symbolic b in
    let rank_a = Symbolic_shape.rank shape_a in
    let rank_b = Symbolic_shape.rank shape_b in
    let batch_a = Array.sub shape_a 0 (rank_a - 2) in
    let batch_b = Array.sub shape_b 0 (rank_b - 2) in
    let batch_out = broadcast_shapes batch_a batch_b in
    let m = shape_a.(rank_a - 2) in
    let n = shape_b.(rank_b - 1) in
    Array.concat [ batch_out; [| m; n |] ]

  (* Helper that calls op_matmul, allocating output if not provided. Inputs must
     be at least 2D. *)
  let matmul_with_alloc ?out a b =
    let out =
      match out with
      | Some o -> o
      | None ->
          let ndim_a = ndim a in
          let ndim_b = ndim b in
          if ndim_a = 2 && ndim_b = 2 then
            (* Fast path for 2D×2D: skip Symbolic_shape/broadcast_shapes *)
            let m = dim 0 a in
            let n = dim 1 b in
            empty (B.context a) (B.dtype a) [| m; n |]
          else
            let out_shape = matmul_output_shape a b in
            let out_shape_concrete =
              match Symbolic_shape.eval out_shape with
              | Some s -> s
              | None ->
                  Error.failed ~op:"matmul"
                    ~what:"cannot compute output shape with symbolic dimensions"
                    ()
            in
            empty (B.context a) (B.dtype a) out_shape_concrete
    in
    B.matmul ~out a b;
    out

  let dot ?out x_tensor w_tensor =
    let@ _ = span ~op:"dot" () in
    let ndim_x = ndim x_tensor in
    let ndim_w = ndim w_tensor in

    if not (ndim_x > 0 && ndim_w > 0) then
      Error.invalid ~op:"dot" ~what:"tensors" ~reason:"both must be at least 1D"
        ();

    (* Handle special cases for 1D tensors *)
    match (ndim_x, ndim_w) with
    | 1, 1 ->
        (* 1D x 1D -> scalar (sum of element-wise product) *)
        let product = mul x_tensor w_tensor in
        sum ?out product
    | 1, _ -> (
        (* 1D x ND -> contract on first axis of w *)
        (* Reshape x to (1, k) and use matmul *)
        let x_2d = unsqueeze ~axes:[ 0 ] x_tensor in
        let result = matmul_with_alloc x_2d w_tensor in
        (* Result has shape (..., 1, n) -> squeeze the 1 *)
        let squeezed = squeeze ~axes:[ ndim result - 2 ] result in
        match out with
        | Some o ->
            B.add ~out:o squeezed (zeros_like squeezed);
            o
        | None -> squeezed)
    | _, 1 -> (
        (* ND x 1D -> contract on last axis of x *)
        (* Reshape w to (k, 1) and use matmul *)
        let w_2d = unsqueeze ~axes:[ 1 ] w_tensor in
        let result = matmul_with_alloc x_tensor w_2d in
        (* Result has shape (..., m, 1) -> squeeze the 1 *)
        let squeezed = squeeze ~axes:[ ndim result - 1 ] result in
        match out with
        | Some o ->
            B.add ~out:o squeezed (zeros_like squeezed);
            o
        | None -> squeezed)
    | _ ->
        (* ND x ND -> use matmul directly *)
        matmul_with_alloc ?out x_tensor w_tensor

  let matmul ?out a_orig b_orig =
    let@ _ = span ~op:"matmul" () in
    let ndim_a_orig = ndim a_orig in
    let ndim_b_orig = ndim b_orig in

    if ndim_a_orig = 0 || ndim_b_orig = 0 then
      Error.invalid ~op:"matmul" ~what:"inputs"
        ~reason:"cannot be 0-D (scalars)" ();

    (* When both inputs are >= 2D, we can use the provided out buffer
       directly *)
    if ndim_a_orig >= 2 && ndim_b_orig >= 2 then
      matmul_with_alloc ?out a_orig b_orig
    else
      (* 1D inputs require unsqueeze/squeeze, so out buffer shape wouldn't
         match *)
      let a, b =
        if ndim_a_orig = 1 && ndim_b_orig = 1 then
          (* (k), (k) -> a becomes (1,k), b becomes (k,1) *)
          (unsqueeze ~axes:[ 0 ] a_orig, unsqueeze ~axes:[ 1 ] b_orig)
        else if ndim_a_orig = 1 then
          (* (k), (...,k,n) -> a becomes (1,k) *)
          (unsqueeze ~axes:[ 0 ] a_orig, b_orig)
        else
          (* (...,m,k), (k) -> b becomes (k,1) *)
          (a_orig, unsqueeze ~axes:[ 1 ] b_orig)
      in

      let result_intermediate = matmul_with_alloc a b in

      (* Squeeze the result if original inputs were 1D to match matmul
         semantics *)
      if ndim_a_orig = 1 && ndim_b_orig = 1 then
        (* Original (k) @ (k) -> result (1,1) from backend -> squeeze to scalar
           () *)
        squeeze result_intermediate
      else if ndim_a_orig = 1 then
        (* Original (k) @ (...,k,n) -> result (...,1,n) from backend -> squeeze
           first matrix dim *)
        squeeze ~axes:[ ndim result_intermediate - 2 ] result_intermediate
      else
        (* Original (...,m,k) @ (k) -> result (...,m,1) from backend -> squeeze
           last matrix dim *)
        squeeze ~axes:[ ndim result_intermediate - 1 ] result_intermediate

  (* ───── Additional Linear Algebra Operations ───── *)

  let diagonal ?(offset = 0) ?axis1 ?axis2 x =
    let@ _ = span ~op:"diagonal" () in
    let nd = ndim x in
    let ax1 = Option.value axis1 ~default:(nd - 2) in
    let ax2 = Option.value axis2 ~default:(nd - 1) in
    let ax1 = if ax1 < 0 then nd + ax1 else ax1 in
    let ax2 = if ax2 < 0 then nd + ax2 else ax2 in

    if ax1 = ax2 then
      Error.invalid ~op:"diagonal" ~what:"axes" ~reason:"axes must be different"
        ();

    (* 1. Permute axes so the diagonal dimensions are at the end *)
    let perm =
      let axes_list = List.init nd Fun.id in
      let others = List.filter (fun a -> a <> ax1 && a <> ax2) axes_list in
      others @ [ ax1; ax2 ]
    in
    let x_trans = transpose ~axes:perm x in

    (* Get dimensions of the 2D plane *)
    let d1 = dim (nd - 2) x_trans in
    let d2 = dim (nd - 1) x_trans in

    (* 2. Calculate diagonal length *)
    let diag_len =
      if offset >= 0 then Stdlib.max 0 (Stdlib.min d1 (d2 - offset))
      else Stdlib.max 0 (Stdlib.min (d1 + offset) d2)
    in

    if diag_len = 0 then
      (* Return empty tensor with correct shape (batch_dims @ [0]) *)
      let prefix_shape = Array.sub (shape x_trans) 0 (nd - 2) in
      let final_shape = Array.append prefix_shape [| 0 |] in
      empty (B.context x) (dtype x) final_shape
    else
      (* 3. Flatten the last two dimensions to 1D *)
      let prefix_shape = Array.sub (shape x_trans) 0 (nd - 2) in
      let flat_last_dim = d1 * d2 in
      let flat_shape = Array.append prefix_shape [| flat_last_dim |] in
      (* Make contiguous before reshape to handle strided views *)
      let x_flat = reshape flat_shape (contiguous x_trans) in

      (* 4. Construct indices for gather *)
      (*
         In a flattened (d1 * d2) array:
         - Moving down 1 row adds d2 to index
         - Moving right 1 col adds 1 to index
         - Diagonal move (+1 row, +1 col) adds (d2 + 1) to index
         
         Start index:
         - If offset >= 0 (upper): start at (0, offset) -> index = offset
         - If offset < 0  (lower): start at (-offset, 0) -> index = -offset * d2
      *)
      let start_linear_idx = if offset >= 0 then offset else -offset * d2 in
      let step = d2 + 1 in

      (* Generate indices: [start, start+step, ..., start+(len-1)*step] *)
      let ctx = B.context x in
      (* Use int32 for indexing *)
      let indices_base = arange ctx Dtype.int32 0 diag_len 1 in
      let step_tensor = scalar ctx Dtype.int32 (Int32.of_int step) in
      let start_tensor =
        scalar ctx Dtype.int32 (Int32.of_int start_linear_idx)
      in

      let gather_indices = add (mul indices_base step_tensor) start_tensor in

      (* 5. Gather along the flattened axis (index nd-2, since the last two
         dimensions were merged into one) *)
      take ~axis:(nd - 2) gather_indices x_flat

  let matrix_transpose x =
    let nd = ndim x in
    if nd < 2 then x else swapaxes (nd - 2) (nd - 1) x

  (* ───── Complex Helpers ───── *)

  let complex (type a b) ~(real : (a, b) t) ~(imag : (a, b) t) =
    let real_shape = shape real in
    let imag_shape = shape imag in
    if real_shape <> imag_shape then
      Error.shape_mismatch ~op:"complex" ~expected:real_shape ~actual:imag_shape
        ();
    let size = Array.fold_left ( * ) 1 real_shape in
    match dtype real with
    | Float32 ->
        let real = (real : (float, float32_elt) t) in
        let imag = (imag : (float, float32_elt) t) in
        let complex_data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i real_shape |> Array.to_list in
              let re = unsafe_get idx real in
              let im = unsafe_get idx imag in
              Complex.{ re; im })
        in
        Obj.magic (create (B.context real) complex64 real_shape complex_data)
    | Float64 ->
        let real = (real : (float, float64_elt) t) in
        let imag = (imag : (float, float64_elt) t) in
        let complex_data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i real_shape |> Array.to_list in
              let re = unsafe_get idx real in
              let im = unsafe_get idx imag in
              Complex.{ re; im })
        in
        Obj.magic (create (B.context real) complex128 real_shape complex_data)
    | _ ->
        Error.invalid ~op:"complex" ~what:"dtype"
          ~reason:"real and imag must be float32 or float64" ()

  let real (type a b) (x : (a, b) t) =
    match dtype x with
    | Complex64 ->
        let x = (x : (Complex.t, complex32_elt) t) in
        let shape_x = shape x in
        let size = Array.fold_left ( * ) 1 shape_x in
        let real_data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i shape_x |> Array.to_list in
              let c = unsafe_get idx x in
              c.Complex.re)
        in
        Obj.magic (create (B.context x) float32 shape_x real_data)
    | Complex128 ->
        let x = (x : (Complex.t, complex64_elt) t) in
        let shape_x = shape x in
        let size = Array.fold_left ( * ) 1 shape_x in
        let real_data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i shape_x |> Array.to_list in
              let c = unsafe_get idx x in
              c.Complex.re)
        in
        Obj.magic (create (B.context x) float64 shape_x real_data)
    | _ ->
        Error.invalid ~op:"real" ~what:"dtype"
          ~reason:"input must be complex64 or complex128" ()

  let imag (type a b) (x : (a, b) t) =
    match dtype x with
    | Complex64 ->
        let x = (x : (Complex.t, complex32_elt) t) in
        let shape_x = shape x in
        let size = Array.fold_left ( * ) 1 shape_x in
        let imag_data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i shape_x |> Array.to_list in
              let c = unsafe_get idx x in
              c.Complex.im)
        in
        Obj.magic (create (B.context x) float32 shape_x imag_data)
    | Complex128 ->
        let x = (x : (Complex.t, complex64_elt) t) in
        let shape_x = shape x in
        let size = Array.fold_left ( * ) 1 shape_x in
        let imag_data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i shape_x |> Array.to_list in
              let c = unsafe_get idx x in
              c.Complex.im)
        in
        Obj.magic (create (B.context x) float64 shape_x imag_data)
    | _ ->
        Error.invalid ~op:"imag" ~what:"dtype"
          ~reason:"input must be complex64 or complex128" ()

  let conjugate (type a b) (x : (a, b) t) =
    match dtype x with
    | Complex64 ->
        let real_part = real x in
        let imag_part = imag x |> neg in
        complex ~real:real_part ~imag:imag_part
    | Complex128 ->
        let real_part = real x in
        let imag_part = imag x |> neg in
        complex ~real:real_part ~imag:imag_part
    | _ -> x

  let vdot (type a b) (a : (a, b) t) (b : (a, b) t) =
    let@ _ = span ~op:"vdot" () in
    (* Try to broadcast inputs to compatible shapes first *)
    let a', b' =
      try
        let broadcasted = broadcast_arrays [ a; b ] in
        let a_bc = List.nth broadcasted 0 in
        let b_bc = List.nth broadcasted 1 in
        (* Make contiguous to allow flattening *)
        (contiguous a_bc, contiguous b_bc)
      with _ ->
        (* If broadcasting fails, just use original tensors *)
        (a, b)
    in
    let flat_a = flatten a' in
    let flat_b = flatten b' in
    if numel flat_a <> numel flat_b then
      invalid_arg "vdot: different number of elements";

    (* For complex types, conjugate first vector *)
    match dtype a with
    | (Complex64 | Complex128) when dtype a = dtype b ->
        sum (mul (conjugate flat_a) flat_b)
    | _ -> sum (mul flat_a flat_b)

  let vecdot ?axis x1 x2 =
    let@ _ = span ~op:"vecdot" () in
    match axis with
    | None ->
        (* Default to last axis *)
        let ax = ndim x1 - 1 in
        let prod = mul x1 x2 in
        sum ~axes:[ ax ] ~keepdims:false prod
    | Some ax ->
        let ax = if ax < 0 then ndim x1 + ax else ax in
        let prod = mul x1 x2 in
        sum ~axes:[ ax ] ~keepdims:false prod

  let inner a b =
    let@ _ = span ~op:"inner" () in
    let shape_a = shape a in
    let shape_b = shape b in
    let last_a = shape_a.(ndim a - 1) in
    let last_b = shape_b.(ndim b - 1) in
    if last_a <> last_b then invalid_arg "inner: last dimensions differ";
    vecdot ~axis:(-1) a b

  let outer ?out a b =
    let@ _ = span ~op:"outer" () in
    let flat_a = if ndim a = 0 then reshape [| 1 |] a else flatten a in
    let flat_b = if ndim b = 0 then reshape [| 1 |] b else flatten b in
    let a_col = reshape [| numel flat_a; 1 |] flat_a in
    let b_row = reshape [| 1; numel flat_b |] flat_b in
    let result = matmul ?out a_col b_row in
    (* For scalar inputs, squeeze the resulting dimensions *)
    let result = if ndim a = 0 then squeeze ~axes:[ 0 ] result else result in
    let result =
      if ndim b = 0 then squeeze ~axes:[ (if ndim a = 0 then 0 else 1) ] result
      else result
    in
    result

  let tensordot ?axes a b =
    let@ _ = span ~op:"tensordot" () in
    match axes with
    | None ->
        (* Default: contract last axis of a with first axis of b *)
        matmul a b
    | Some (axes_a, axes_b) ->
        (* Validate axes have same length *)
        let n_axes = List.length axes_a in
        if n_axes <> List.length axes_b then
          invalid_arg "tensordot: axes lists must have same length";

        (* Normalize negative axes *)
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

        (* Check axes dimensions match *)
        let shape_a = shape a in
        let shape_b = shape b in
        Array.iter2
          (fun ax_a ax_b ->
            if shape_a.(ax_a) <> shape_b.(ax_b) then
              invalid_arg "tensordot: axes have different sizes")
          axes_a axes_b;

        (* Compute the permutation for moving contracted axes to the
           end/beginning *)
        let axes_a_set =
          Array.fold_left (fun s x -> IntSet.add x s) IntSet.empty axes_a
        in
        let axes_b_set =
          Array.fold_left (fun s x -> IntSet.add x s) IntSet.empty axes_b
        in

        (* Get free (non-contracted) axes *)
        let free_axes_a =
          Array.init ndim_a (fun i -> i)
          |> Array.to_list
          |> List.filter (fun i -> not (IntSet.mem i axes_a_set))
          |> Array.of_list
        in
        let free_axes_b =
          Array.init ndim_b (fun i -> i)
          |> Array.to_list
          |> List.filter (fun i -> not (IntSet.mem i axes_b_set))
          |> Array.of_list
        in

        (* Move axes: free axes first, then contracted axes *)
        let perm_a = Array.append free_axes_a axes_a in
        let perm_b = Array.append axes_b free_axes_b in

        (* Transpose tensors *)
        let a_transposed =
          if Array.length perm_a > 1 then
            transpose ~axes:(Array.to_list perm_a) a
          else a
        in
        let b_transposed =
          if Array.length perm_b > 1 then
            transpose ~axes:(Array.to_list perm_b) b
          else b
        in
        let a_transposed = contiguous a_transposed in
        let b_transposed = contiguous b_transposed in

        (* Compute new shapes for matrix multiplication *)
        let shape_a_t = shape a_transposed in
        let shape_b_t = shape b_transposed in

        let n_free_a = Array.length free_axes_a in
        let n_free_b = Array.length free_axes_b in

        (* Reshape for matmul: - a: (prod(free_dims_a), prod(contracted_dims)) -
           b: (prod(contracted_dims), prod(free_dims_b)) *)
        let free_size_a =
          if n_free_a = 0 then 1
          else Array.fold_left ( * ) 1 (Array.sub shape_a_t 0 n_free_a)
        in
        let free_size_b =
          if n_free_b = 0 then 1
          else
            Array.fold_left ( * ) 1
              (Array.sub shape_b_t n_axes (ndim_b - n_axes))
        in
        let contracted_size =
          Array.fold_left ( * ) 1 (Array.sub shape_a_t n_free_a n_axes)
        in

        let a_mat = reshape [| free_size_a; contracted_size |] a_transposed in
        let b_mat = reshape [| contracted_size; free_size_b |] b_transposed in

        (* Perform matrix multiplication *)
        let result_mat = matmul a_mat b_mat in

        (* Reshape result to final shape *)
        let result_shape =
          Array.append
            (if n_free_a = 0 then [||] else Array.sub shape_a_t 0 n_free_a)
            (if n_free_b = 0 then [||]
             else Array.sub shape_b_t n_axes (ndim_b - n_axes))
        in

        if Array.length result_shape = 0 then
          (* Scalar result *)
          squeeze result_mat
        else reshape result_shape result_mat

  module Einsum = struct
    (* Token type for einsum subscript parsing. *)
    type token = Axis of char | Ellipsis

    let parse_operand str =
      let len = String.length str in
      if len = 0 then []
      else
        let rec loop idx acc ellipsis_seen =
          if idx >= len then List.rev acc
          else
            match str.[idx] with
            | '.' ->
                if
                  idx + 2 >= len || str.[idx + 1] <> '.' || str.[idx + 2] <> '.'
                then invalid_arg "einsum: ellipsis must be '...'";
                if ellipsis_seen then
                  invalid_arg "einsum: multiple ellipsis in operand";
                loop (idx + 3) (Ellipsis :: acc) true
            | c
              when (c >= 'a' && c <= 'z')
                   || (c >= 'A' && c <= 'Z')
                   || (c >= '0' && c <= '9')
                   || c = '_' ->
                loop (idx + 1) (Axis c :: acc) ellipsis_seen
            | c ->
                invalid_arg (Printf.sprintf "einsum: invalid character '%c'" c)
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
          let output = String.trim (String.sub rhs 1 (String.length rhs - 1)) in
          ( Array.of_list (List.map parse_operand inputs),
            Some (parse_operand output) )
      | [ lhs ] ->
          (* Implicit mode *)
          let inputs =
            String.split_on_char ',' lhs
            |> List.map String.trim
            |> List.filter (( <> ) "")
          in
          (Array.of_list (List.map parse_operand inputs), None)
      | _ -> invalid_arg "einsum: invalid format, expected inputs->output"

    (* Extracts diagonal for repeated indices (e.g., "ii->i"). Applies diagonal
       iteratively until no repeated axis labels remain. *)
    let handle_repeated_indices tensor tokens =
      let rec find_dups acc idx = function
        | [] -> None
        | Axis c :: rest -> (
            match List.find_opt (fun (char, _) -> char = c) acc with
            | Some (_, prev_idx) -> Some (prev_idx, idx, c)
            | None -> find_dups ((c, idx) :: acc) (idx + 1) rest)
        | Ellipsis :: rest ->
            (* Ellipsis is expanded before this function is called *)
            find_dups acc (idx + 1) rest
      in
      let rec process t current_tokens =
        match find_dups [] 0 current_tokens with
        | None -> (t, current_tokens)
        | Some (ax1, ax2, c) ->
            (* Validate dimensions match before diagonalization *)
            let s = shape t in
            let dim1 = s.(ax1) in
            let dim2 = s.(ax2) in
            if dim1 <> dim2 then
              invalid_arg
                (Printf.sprintf
                   "einsum: index var '%c' must have consistent dimensions (%d \
                    vs %d)"
                   c dim1 dim2);
            let t' = diagonal ~axis1:ax1 ~axis2:ax2 t in

            (* Remove the second occurrence from tokens *)
            let rec remove_at i lst =
              match (i, lst) with
              | 0, _ :: xs -> xs
              | n, x :: xs -> x :: remove_at (n - 1) xs
              | _, [] -> []
            in
            process t' (remove_at ax2 current_tokens)
      in
      process tensor tokens

    (* Metadata for path optimization: tracks shape and axis labels for each
       intermediate tensor in the contraction tree. *)
    type tensor_info = {
      id : int;
      shape : int array;
      axis_labels : char list; (* mapped to chars for internal processing *)
    }

    type contraction_path =
      | Leaf of int (* index in operands array *)
      | Node of
          contraction_path
          * contraction_path
          * tensor_info (* left, right, result *)

    let estimate_cost (t1 : tensor_info) (t2 : tensor_info)
        (common_chars : char list) =
      (* Returns (operation_cost, output_size) where operation_cost is the
         product of all dimensions involved (standard M*N*K approximation). *)
      let get_dim t char =
        let rec find_dim dims labels i =
          match labels with
          | [] -> 1 (* Should not happen if char exists *)
          | c :: rest ->
              if c = char then dims.(i) else find_dim dims rest (i + 1)
        in
        find_dim t.shape t.axis_labels 0
      in

      (* Identify dimensions *)
      let dim_map = Hashtbl.create 16 in
      List.iter
        (fun c -> Hashtbl.replace dim_map c (get_dim t1 c))
        t1.axis_labels;
      List.iter
        (fun c -> Hashtbl.replace dim_map c (get_dim t2 c))
        t2.axis_labels;

      let all_chars =
        List.sort_uniq Char.compare (t1.axis_labels @ t2.axis_labels)
      in

      let output_size =
        List.fold_left
          (fun acc c ->
            if List.mem c common_chars then acc (* Contracted *)
            else acc * Hashtbl.find dim_map c)
          1 all_chars
      in

      let operation_cost =
        List.fold_left (fun acc c -> acc * Hashtbl.find dim_map c) 1 all_chars
      in

      (float_of_int operation_cost, float_of_int output_size)

    (* Greedy contraction path optimizer. Repeatedly contracts the pair with
       lowest estimated cost (product of all involved dimensions) until a single
       result remains. Returns a binary tree of contractions. *)
    let optimize_path inputs output_chars =
      let workset = ref (List.mapi (fun i t -> (Leaf i, t)) inputs) in

      (* Helper to compute result info for a pair *)
      let contract_info (path1, t1) (path2, t2) =
        let common =
          List.filter (fun c -> List.mem c t2.axis_labels) t1.axis_labels
        in
        (* Indices kept: unique to t1, unique to t2, or in output_chars *)
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
            | h :: t -> if h = x then i else aux (i + 1) t
          in
          aux 0 lst
        in
        let get_dim c =
          if List.mem c t1.axis_labels then
            let idx = find_index c t1.axis_labels in
            t1.shape.(idx)
          else
            let idx = find_index c t2.axis_labels in
            t2.shape.(idx)
        in
        let new_shape = Array.of_list (List.map get_dim new_labels) in

        let info = { id = -1; shape = new_shape; axis_labels = new_labels } in
        let cost, size =
          estimate_cost t1 t2
            (List.filter (fun c -> not (List.mem c new_labels)) common)
        in
        (cost, size, Node (path1, path2, info), info)
      in

      while List.length !workset > 1 do
        let current_items = !workset in
        let best_pair = ref None in
        let min_cost = ref Float.infinity in

        (* Iterate all pairs *)
        let rec iter_pairs = function
          | [] -> ()
          | x :: rest ->
              List.iter
                (fun y ->
                  let cost, _size, path, info = contract_info x y in
                  if cost < !min_cost then (
                    min_cost := cost;
                    best_pair := Some (x, y, path, info)))
                rest;
              iter_pairs rest
        in
        iter_pairs current_items;

        match !best_pair with
        | None -> failwith "einsum: could not find valid contraction"
        | Some (item1, item2, new_path, new_info) ->
            let remaining =
              List.filter (fun x -> x != item1 && x != item2) current_items
            in
            workset := (new_path, new_info) :: remaining
      done;

      match !workset with
      | [ (final_path, _) ] -> final_path
      | _ -> failwith "einsum: optimization failed"

    (* Contracts two tensors by: (1) transposing to [batch, free, contract]
       layout, (2) broadcasting batch dims, (3) flattening to 3D for batched
       matmul, (4) unflattening and transposing to match the target axis
       order. *)
    let contract_pair op_a str_a op_b str_b result_str =
      let sa = shape op_a in
      let sb = shape op_b in

      (* Map chars to indices *)
      let chars_a = String.to_seq str_a |> List.of_seq in
      let chars_b = String.to_seq str_b |> List.of_seq in
      let chars_out = String.to_seq result_str |> List.of_seq in

      (* Identify axes *)
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
      let a_free_chars =
        List.filter (fun c -> not (List.mem c chars_b)) chars_a
      in
      let b_free_chars =
        List.filter (fun c -> not (List.mem c chars_a)) chars_b
      in

      (* Build Permutations: [Batch, Free, Contract] *)
      let get_axes source_chars target_order =
        List.map
          (fun c ->
            let rec find_idx i = function
              | [] -> failwith "char not found"
              | x :: xs -> if x = c then i else find_idx (i + 1) xs
            in
            find_idx 0 source_chars)
          target_order
      in

      let perm_a =
        get_axes chars_a (batch_chars @ a_free_chars @ contract_chars)
      in
      let perm_b =
        get_axes chars_b (batch_chars @ contract_chars @ b_free_chars)
      in

      let is_identity_perm perm n =
        let rec check i = function
          | [] -> i = n
          | x :: xs -> x = i && check (i + 1) xs
        in
        check 0 perm
      in
      let a_t =
        if is_identity_perm perm_a (String.length str_a) then op_a
        else contiguous (transpose ~axes:perm_a op_a)
      in
      let b_t =
        if is_identity_perm perm_b (String.length str_b) then op_b
        else contiguous (transpose ~axes:perm_b op_b)
      in

      (* Flatten Dimensions for Matmul *)
      let prod dims = Array.fold_left ( * ) 1 dims in

      let perm_a_arr = Array.of_list perm_a in
      let perm_b_arr = Array.of_list perm_b in
      let n_batch = List.length batch_chars in
      let n_a_free = List.length a_free_chars in
      let n_contract = List.length contract_chars in
      let n_b_free = List.length b_free_chars in

      (* Broadcast batch dimensions *)
      let batch_dims =
        Array.init n_batch (fun i ->
            let dim_a = sa.(perm_a_arr.(i)) in
            let dim_b = sb.(perm_b_arr.(i)) in
            if dim_a = dim_b then dim_a
            else if dim_a = 1 then dim_b
            else if dim_b = 1 then dim_a
            else
              invalid_arg
                (Printf.sprintf
                   "einsum: incompatible broadcast dimensions (%d vs %d)" dim_a
                   dim_b))
      in
      let a_free_dims =
        Array.init n_a_free (fun i -> sa.(perm_a_arr.(n_batch + i)))
      in
      let contract_dims =
        Array.init n_contract (fun i ->
            sa.(perm_a_arr.(n_batch + n_a_free + i)))
      in
      let b_free_dims =
        Array.init n_b_free (fun i ->
            sb.(perm_b_arr.(n_batch + n_contract + i)))
      in

      let b_size = prod batch_dims in
      let m = prod a_free_dims in
      let k = prod contract_dims in
      let n = prod b_free_dims in

      (* Broadcast tensors to match batch dims if needed *)
      let broadcast_batch tensor perm_arr src_shape =
        if n_batch = 0 then tensor
        else
          let needs_broadcast = ref false in
          let target_shape =
            Array.init (ndim tensor) (fun i ->
                if i < n_batch then (
                  let src_dim = src_shape.(perm_arr.(i)) in
                  let target_dim = batch_dims.(i) in
                  if src_dim <> target_dim then needs_broadcast := true;
                  target_dim)
                else
                  let src_dim = src_shape.(perm_arr.(i)) in
                  src_dim)
          in
          if !needs_broadcast then broadcast_to target_shape tensor else tensor
      in

      let a_t_broadcast = broadcast_batch a_t perm_a_arr sa in
      let b_t_broadcast = broadcast_batch b_t perm_b_arr sb in

      (* Reshape to 3D for batched matmul: [batch, rows, cols] *)
      let a_mat = reshape [| b_size; m; k |] a_t_broadcast in
      let b_mat = reshape [| b_size; k; n |] b_t_broadcast in

      let result_mat = matmul a_mat b_mat in

      (* Restore original dimension structure *)
      let result_intermediate_shape =
        Array.concat [ batch_dims; a_free_dims; b_free_dims ]
      in
      let result_unflat = reshape result_intermediate_shape result_mat in

      let intermediate_chars = batch_chars @ a_free_chars @ b_free_chars in
      if intermediate_chars = chars_out then result_unflat
      else
        let final_perm = get_axes intermediate_chars chars_out in
        transpose ~axes:final_perm result_unflat

    let calculate subscripts operands =
      let n_ops = Array.length operands in
      if n_ops = 0 then invalid_arg "einsum: no input operands";

      (* Fast paths for common operations *)
      match (subscripts, n_ops) with
      | "i,i->", 2 -> sum (mul operands.(0) operands.(1))
      | "ij,jk->ik", 2 -> matmul operands.(0) operands.(1)
      | "ij->ji", 1 -> transpose operands.(0)
      | _ ->
          let input_tokens, output_token_opt = parse_equation subscripts in

          if Array.length input_tokens <> n_ops then
            invalid_arg "einsum: number of inputs must equal number of operands";

          (* Compute ellipsis rank from max difference between operand rank and
             named axes. This determines how many dimensions "..." expands
             to. *)
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

          (* Maps ellipsis dimension index to a reserved char (200+i) to avoid
             collision with user-specified axis labels. *)
          let get_ell_char i = char_of_int (200 + i) in

          let normalized_inputs =
            Array.mapi
              (fun i tokens ->
                let op = operands.(i) in
                (* Expand Ellipsis tokens *)
                let expanded_tokens =
                  let n_named =
                    List.length
                      (List.filter
                         (function Axis _ -> true | _ -> false)
                         tokens)
                  in
                  let ell_dim = ndim op - n_named in
                  List.flatten
                    (List.map
                       (function
                         | Axis c -> [ Axis c ]
                         | Ellipsis ->
                             List.init ell_dim (fun k ->
                                 Axis (get_ell_char (ell_rank - ell_dim + k))))
                       tokens)
                in
                (* Diagonalize *)
                let op_diag, final_tokens =
                  handle_repeated_indices op expanded_tokens
                in

                (* Convert tokens to char list *)
                let chars =
                  List.map
                    (function Axis c -> c | _ -> assert false)
                    final_tokens
                in
                ({ id = i; shape = shape op_diag; axis_labels = chars }, op_diag))
              input_tokens
          in

          let ops_info = Array.map fst normalized_inputs in
          let ops_tensors = Array.map snd normalized_inputs in

          (* Validate dimension consistency across operands. Dimensions must
             either match exactly or be 1 (broadcastable). *)
          let char_dims = Hashtbl.create 16 in
          Array.iter
            (fun info ->
              List.iteri
                (fun idx c ->
                  let dim = info.shape.(idx) in
                  match Hashtbl.find_opt char_dims c with
                  | None -> Hashtbl.add char_dims c dim
                  | Some prev_dim ->
                      (* Allow broadcasting: 1 is compatible with any dim *)
                      if prev_dim <> dim && prev_dim <> 1 && dim <> 1 then
                        invalid_arg
                          (Printf.sprintf
                             "einsum: index var '%c' must have consistent \
                              dimensions (%d vs %d)"
                             c prev_dim dim)
                      else if dim > prev_dim then
                        (* Keep the larger dimension for broadcasting *)
                        Hashtbl.replace char_dims c dim)
                info.axis_labels)
            ops_info;

          (* Check if any input has ellipsis *)
          let inputs_have_ellipsis =
            Array.exists
              (fun tokens -> List.exists (( = ) Ellipsis) tokens)
              input_tokens
          in

          (* Determine Output Chars *)
          let target_chars =
            match output_token_opt with
            | Some tokens ->
                let output_has_ellipsis = List.exists (( = ) Ellipsis) tokens in
                if output_has_ellipsis && not inputs_have_ellipsis then
                  invalid_arg
                    "einsum: output ellipsis requires ellipsis in inputs";
                List.flatten
                  (List.map
                     (function
                       | Axis c -> [ c ]
                       | Ellipsis ->
                           List.init ell_rank (fun k -> get_ell_char k))
                     tokens)
            | None ->
                (* Implicit mode: chars appearing exactly once in the original
                   input are kept in output, sorted alphabetically. We count in
                   the original tokens (before diagonalization) per NumPy
                   spec. *)
                let all_original_chars =
                  List.concat
                    (Array.to_list
                       (Array.map
                          (fun tokens ->
                            List.filter_map
                              (function Axis c -> Some c | Ellipsis -> None)
                              tokens)
                          input_tokens))
                in
                let counts = Hashtbl.create 16 in
                List.iter
                  (fun c ->
                    Hashtbl.replace counts c
                      ((Hashtbl.find_opt counts c |> Option.value ~default:0)
                      + 1))
                  all_original_chars;
                let ell_chars = List.init ell_rank (fun k -> get_ell_char k) in
                let named =
                  List.filter (fun c -> int_of_char c < 200) all_original_chars
                  |> List.sort_uniq Char.compare
                  |> List.filter (fun c -> Hashtbl.find counts c = 1)
                in
                ell_chars @ named
          in

          (* Validate that all output indices exist in inputs *)
          let all_input_chars =
            Array.fold_left (fun acc info -> acc @ info.axis_labels) [] ops_info
          in
          List.iter
            (fun c ->
              if not (List.mem c all_input_chars) then
                invalid_arg
                  (Printf.sprintf
                     "einsum: output index '%c' not found in inputs" c))
            target_chars;

          (* Pre-reduce axes that appear in exactly one operand and are absent
             from the output. This avoids materialising huge intermediates for
             patterns like "ab,cd->" where independent sums can be done
             first. *)
          let () =
            let char_count = Hashtbl.create 16 in
            Array.iter
              (fun info ->
                List.iter
                  (fun c ->
                    let n =
                      match Hashtbl.find_opt char_count c with
                      | None -> 0
                      | Some n -> n
                    in
                    Hashtbl.replace char_count c (n + 1))
                  info.axis_labels)
              ops_info;
            Array.iteri
              (fun i info ->
                let axes_to_reduce = ref [] in
                let new_labels = ref [] in
                List.iteri
                  (fun axis_idx c ->
                    if
                      Hashtbl.find char_count c = 1
                      && not (List.mem c target_chars)
                    then axes_to_reduce := axis_idx :: !axes_to_reduce
                    else new_labels := c :: !new_labels)
                  info.axis_labels;
                match !axes_to_reduce with
                | [] -> ()
                | axes ->
                    let axes = List.rev axes in
                    ops_tensors.(i) <- sum ~axes ops_tensors.(i);
                    ops_info.(i) <-
                      {
                        info with
                        shape = shape ops_tensors.(i);
                        axis_labels = List.rev !new_labels;
                      })
              ops_info
          in

          (* Reduce remaining axes and transpose to match target order *)
          let finalize result current_chars =
            let axes_to_reduce =
              List.mapi
                (fun i c ->
                  if not (List.mem c target_chars) then Some i else None)
                current_chars
              |> List.filter_map Fun.id
            in
            let result =
              if axes_to_reduce = [] then result
              else sum ~axes:axes_to_reduce result
            in
            let final_chars =
              List.filter (fun c -> List.mem c target_chars) current_chars
            in
            if final_chars = target_chars then result
            else
              let perm =
                List.map
                  (fun c ->
                    let rec find i = function
                      | [] -> 0
                      | x :: xs -> if x = c then i else find (i + 1) xs
                    in
                    find 0 final_chars)
                  target_chars
              in
              transpose ~axes:perm result
          in

          if n_ops = 1 then
            (* Single operand: reduce + permute, no contraction needed *)
            finalize ops_tensors.(0) ops_info.(0).axis_labels
          else if n_ops = 2 then
            (* Two operands: contract directly, skip optimizer *)
            let info_a = ops_info.(0) in
            let info_b = ops_info.(1) in
            let str_a = info_a.axis_labels |> List.to_seq |> String.of_seq in
            let str_b = info_b.axis_labels |> List.to_seq |> String.of_seq in
            (* Compute result labels: chars not contracted or in target *)
            let common =
              List.filter
                (fun c -> List.mem c info_b.axis_labels)
                info_a.axis_labels
            in
            let result_labels =
              let all =
                List.sort_uniq Char.compare
                  (info_a.axis_labels @ info_b.axis_labels)
              in
              List.filter
                (fun c -> (not (List.mem c common)) || List.mem c target_chars)
                all
            in
            let str_out = result_labels |> List.to_seq |> String.of_seq in
            let result =
              contract_pair ops_tensors.(0) str_a ops_tensors.(1) str_b str_out
            in
            finalize result result_labels
          else
            (* 3+ operands: use greedy path optimizer *)
            let plan = optimize_path (Array.to_list ops_info) target_chars in
            let rec execute = function
              | Leaf idx ->
                  ( ops_tensors.(idx),
                    ops_info.(idx).axis_labels |> List.to_seq |> String.of_seq
                  )
              | Node (left, right, info) ->
                  let res_a, str_a = execute left in
                  let res_b, str_b = execute right in
                  let str_out =
                    info.axis_labels |> List.to_seq |> String.of_seq
                  in
                  let res = contract_pair res_a str_a res_b str_b str_out in
                  (res, str_out)
            in
            let result, result_str = execute plan in
            let current_chars = String.to_seq result_str |> List.of_seq in
            finalize result current_chars
  end

  let einsum subscripts operands =
    let@ _ = span ~op:"einsum" () in
    Einsum.calculate subscripts operands

  let kron a b =
    let@ _ = span ~op:"kron" () in
    (* Kronecker product implementation that creates proper block structure *)
    let shape_a = shape a in
    let shape_b = shape b in

    (* Flatten to 2D if needed *)
    let a_2d = if ndim a = 1 then reshape [| shape_a.(0); 1 |] a else a in
    let b_2d = if ndim b = 1 then reshape [| shape_b.(0); 1 |] b else b in

    let a_shape = shape a_2d in
    let b_shape = shape b_2d in
    let m_a = a_shape.(0) in
    let n_a = if Array.length a_shape > 1 then a_shape.(1) else 1 in
    let m_b = b_shape.(0) in
    let n_b = if Array.length b_shape > 1 then b_shape.(1) else 1 in

    (* Expand dimensions for broadcasting *)
    let a_expanded = reshape [| m_a; 1; n_a; 1 |] a_2d in
    let b_expanded = reshape [| 1; m_b; 1; n_b |] b_2d in

    (* Multiply with broadcasting *)
    let result = mul a_expanded b_expanded in

    (* Reshape directly to final shape - no transpose needed *)
    let final_shape = [| m_a * m_b; n_a * n_b |] in
    let result_flat = reshape final_shape result in

    (* Return to original dimensionality if input was 1D *)
    if ndim a = 1 && ndim b = 1 then flatten result_flat else result_flat

  let multi_dot arrays =
    let@ _ = span ~op:"multi_dot" () in
    match arrays with
    | [||] -> invalid_arg "multi_dot: empty array"
    | [| arr |] -> arr
    | _ ->
        let n = Array.length arrays in
        let dims = Array.make (n + 1) 0 in
        let matrix_dims idx =
          let tensor = arrays.(idx) in
          let nd = ndim tensor in
          match nd with
          | 1 ->
              let shape = shape tensor in
              let len =
                if Array.length shape = 0 then
                  invalid_arg "multi_dot: 1D tensor must have non-empty shape"
                else shape.(0)
              in
              if idx = 0 then (1, len)
              else if idx = n - 1 then (len, 1)
              else
                invalid_arg
                  "multi_dot: only first and last arguments may be 1D vectors"
          | 2 ->
              let shape = shape tensor in
              (shape.(0), shape.(1))
          | _ ->
              invalid_arg
                (Printf.sprintf
                   "multi_dot: argument %d must be 1D (endpoints) or 2D matrix"
                   idx)
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
        let dims64 = Array.map Int64.of_int dims in
        let cost = Array.make_matrix n n Int64.zero in
        let split = Array.make_matrix n n 0 in
        for len = 2 to n do
          for i = 0 to n - len do
            let j = i + len - 1 in
            let best_cost = ref Int64.max_int in
            let best_split = ref i in
            for k = i to j - 1 do
              let candidate =
                Int64.(
                  add
                    cost.(i).(k)
                    (add
                       cost.(k + 1).(j)
                       (mul dims64.(i) (mul dims64.(k + 1) dims64.(j + 1)))))
              in
              if candidate < !best_cost then (
                best_cost := candidate;
                best_split := k)
            done;
            cost.(i).(j) <- !best_cost;
            split.(i).(j) <- !best_split
          done
        done;
        let memo = Array.init n (fun _ -> Array.make n None) in
        let rec compute i j =
          match memo.(i).(j) with
          | Some t -> t
          | None ->
              let result =
                if i = j then arrays.(i)
                else
                  let k = split.(i).(j) in
                  let left = compute i k in
                  let right = compute (k + 1) j in
                  matmul left right
              in
              memo.(i).(j) <- Some result;
              result
        in
        compute 0 (n - 1)

  let cross ?out ?axis a b =
    let@ _ = span ~op:"cross" () in
    let axis = Option.value axis ~default:(-1) in
    let axis = if axis < 0 then ndim a + axis else axis in

    let shape_a = shape a in
    let shape_b = shape b in

    if axis >= ndim a then
      Error.invalid ~op:"cross" ~what:"axis" ~reason:"out of bounds" ();

    if shape_a.(axis) <> 3 then invalid_arg "cross: axis dim not 3";

    if shape_b.(axis) <> 3 then invalid_arg "cross: axis dim not 3";

    (* Get indices for the three components *)
    let slice_at_index tensor ax idx =
      let slices =
        Array.init (ndim tensor) (fun i ->
            if i = ax then R (idx, idx + 1) else A)
      in
      squeeze ~axes:[ ax ] (slice_internal (Array.to_list slices) tensor)
    in

    (* Extract components *)
    let a1 = slice_at_index a axis 0 in
    let a2 = slice_at_index a axis 1 in
    let a3 = slice_at_index a axis 2 in

    let b1 = slice_at_index b axis 0 in
    let b2 = slice_at_index b axis 1 in
    let b3 = slice_at_index b axis 2 in

    (* Compute cross product components *)
    let c1 = sub (mul a2 b3) (mul a3 b2) in
    let c2 = sub (mul a3 b1) (mul a1 b3) in
    let c3 = sub (mul a1 b2) (mul a2 b1) in

    (* Write to output *)
    match out with
    | Some result ->
        (* Write each component into the pre-allocated output *)
        let write_at_index tensor ax idx value =
          let slices =
            Array.init (ndim tensor) (fun i ->
                if i = ax then R (idx, idx + 1) else A)
          in
          set_slice_internal (Array.to_list slices) tensor
            (expand_dims [ ax ] value)
        in
        write_at_index result axis 0 c1;
        write_at_index result axis 1 c2;
        write_at_index result axis 2 c3;
        result
    | None ->
        (* Stack along the axis *)
        stack ~axis [ c1; c2; c3 ]

  (* Matrix Decompositions *)

  let check_square ~op a =
    let sh = shape a in
    let n = Array.length sh in
    if n < 2 then
      Error.invalid ~op ~what:"input" ~reason:"requires at least 2D array" ();
    if sh.(n - 1) <> sh.(n - 2) then
      invalid_arg (Printf.sprintf "%s: coefficient matrix must be square" op)

  let check_float_or_complex (type a b) ~op (a : (a, b) t) =
    match dtype a with
    | Float16 -> ()
    | Float32 -> ()
    | Float64 -> ()
    | Complex64 -> ()
    | Complex128 -> ()
    | _ -> Error.invalid ~op ~what:"dtype" ~reason:"must be float or complex" ()

  let check_real (type a b) ~op (a : (a, b) t) =
    match dtype a with
    | Float16 -> ()
    | Float32 -> ()
    | Float64 -> ()
    | _ -> Error.invalid ~op ~what:"dtype" ~reason:"must be real (float)" ()

  let cholesky ?upper a =
    let@ _ = span ~op:"cholesky" () in
    check_square ~op:"cholesky" a;
    check_float_or_complex ~op:"cholesky" a;
    let upper = Option.value upper ~default:false in
    B.cholesky ~upper a

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
    let full_matrices = Option.value full_matrices ~default:false in
    B.svd ~full_matrices a

  let svdvals a =
    let@ _ = span ~op:"svdvals" () in
    check_float_or_complex ~op:"svdvals" a;
    let _, s, _ = B.svd ~full_matrices:false a in
    s

  (* Eigenvalues and Eigenvectors *)

  let eig a =
    let@ _ = span ~op:"eig" () in
    check_square ~op:"eig" a;
    check_float_or_complex ~op:"eig" a;
    match B.eig ~vectors:true a with
    | vals, Some vecs -> (vals, vecs)
    | _vals, None ->
        Error.invalid ~op:"eig" ~what:"result" ~reason:"expected eigenvectors"
          ()

  let eigh ?uplo a =
    let@ _ = span ~op:"eigh" () in
    check_square ~op:"eigh" a;
    check_real ~op:"eigh" a;
    let _ = uplo in
    (* uplo handled by backend if needed *)
    match B.eigh ~vectors:true a with
    | vals, Some vecs -> (vals, vecs)
    | _vals, None ->
        Error.invalid ~op:"eigh" ~what:"result" ~reason:"expected eigenvectors"
          ()

  let eigvals a =
    let@ _ = span ~op:"eigvals" () in
    check_square ~op:"eigvals" a;
    check_float_or_complex ~op:"eigvals" a;
    let vals, _ = B.eig ~vectors:false a in
    vals

  let eigvalsh ?uplo a =
    let@ _ = span ~op:"eigvalsh" () in
    check_square ~op:"eigvalsh" a;
    check_real ~op:"eigvalsh" a;
    let _ = uplo in
    (* uplo handled by backend if needed *)
    let vals, _ = B.eigh ~vectors:false a in
    vals

  (* Norms and Condition Numbers *)

  let norm (type a b) ?ord ?axes ?keepdims (x : (a, b) t) =
    let@ _ = span ~op:"norm" () in
    let keepdims = Option.value keepdims ~default:false in
    match (ord, axes) with
    | None, None ->
        (* Frobenius norm for matrices, 2-norm for vectors *)
        sqrt (sum (square (abs x)) ~keepdims)
    | None, Some _ ->
        (* 2-norm along specified axes when ord not specified *)
        sqrt (sum (square (abs x)) ?axes ~keepdims)
    | Some `Fro, _ -> sqrt (sum (square (abs x)) ?axes ~keepdims)
    | Some `One, None ->
        max (sum (abs x) ~axes:[ ndim x - 2 ] ~keepdims) ~keepdims
    | Some `NegOne, None ->
        if ndim x = 1 then min (abs x) ~keepdims
        else
          let column_sums = sum (abs x) ~axes:[ ndim x - 2 ] in
          min column_sums ~keepdims
    | Some `Two, None ->
        let s = svdvals x |> cast (dtype x) in
        max s ~keepdims
    | Some `NegTwo, None ->
        let s = svdvals x |> cast (dtype x) in
        min s ~keepdims
    | Some `Inf, None ->
        (* For vectors, just max of absolute values *)
        if ndim x = 1 then max (abs x) ~keepdims
        else max (sum (abs x) ~axes:[ ndim x - 1 ] ~keepdims) ~keepdims
    | Some `NegInf, None ->
        if ndim x = 1 then min (abs x) ~keepdims
        else
          let row_sums = sum (abs x) ~axes:[ ndim x - 1 ] in
          min row_sums ~keepdims
    | Some `Nuc, None ->
        if ndim x < 2 then
          Error.invalid ~op:"norm" ~what:"input"
            ~reason:"nuclear norm defined for matrices" ()
        else
          let s = svdvals x |> cast (dtype x) in
          sum s ~keepdims
    | Some `NegOne, _ | Some `NegTwo, _ | Some `NegInf, _ | Some `Nuc, _ ->
        Error.failed ~op:"norm"
          ~what:"this combination of ord and axis not implemented" ()
    | Some (`P p), _ ->
        (* General p-norm *)
        (* Special case for matrix 1-norm when axes is None *)
        if p = 1.0 && axes = None && ndim x = 2 then
          max (sum (abs x) ~axes:[ ndim x - 2 ] ~keepdims) ~keepdims
        else
          let abs_x = abs x in
          let p_val = Dtype.of_float (dtype x) p in
          let p_tensor = full (B.context x) (dtype x) [||] p_val in
          let pow_x = pow abs_x p_tensor in
          let sum_pow = sum pow_x ?axes ~keepdims in
          (* Compute 1/p *)
          let one = Dtype.one (dtype x) in
          let one_tensor = full (B.context x) (dtype x) [||] one in
          let inv_p_tensor = div one_tensor p_tensor in
          pow sum_pow inv_p_tensor
    | _ ->
        Error.failed ~op:"norm"
          ~what:"this combination of ord and axis not implemented" ()

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
      let prefix = List.init (Stdlib.max 0 (rank - 2)) (fun _ -> A) in
      let a11 = slice_internal (prefix @ [ I 0; I 0 ]) a in
      let a12 = slice_internal (prefix @ [ I 0; I 1 ]) a in
      let a21 = slice_internal (prefix @ [ I 1; I 0 ]) a in
      let a22 = slice_internal (prefix @ [ I 1; I 1 ]) a in
      let det64 = sub (mul a11 a22) (mul a12 a21) |> cast Dtype.float64 in
      let zero = zeros (B.context det64) Dtype.float64 (shape det64) in
      let sign_pos = greater det64 zero in
      let sign_neg = less det64 zero in
      let sign_pos_f = cast Dtype.float32 (cast Dtype.float64 sign_pos) in
      let sign_neg_f = cast Dtype.float32 (cast Dtype.float64 sign_neg) in
      let sign_float = sub sign_pos_f sign_neg_f in
      let abs_det = abs det64 in
      let logdet64 =
        let is_zero = cmpeq abs_det zero in
        let neg_inf =
          full (B.context det64) Dtype.float64 (shape det64) Float.neg_infinity
        in
        where is_zero neg_inf (log abs_det)
      in
      let logdet = cast Dtype.float32 logdet64 in
      (sign_float, logdet)
    else
      let _q, r = B.qr ~reduced:false a in
      let r_diag = diagonal r in
      let signs = sign r_diag in
      let sign_det =
        if ndim signs > 1 then prod signs ~axes:[ -1 ] ~keepdims:false
        else prod signs
      in
      let sign_float = cast Dtype.float32 (cast Dtype.float64 sign_det) in
      let abs_diag = abs r_diag in
      let abs_float64 = cast Dtype.float64 abs_diag in
      let zero =
        zeros (B.context abs_float64) Dtype.float64 (shape abs_float64)
      in
      let log_abs_diag =
        let is_zero = cmpeq abs_float64 zero in
        let neg_inf =
          full (B.context abs_float64) Dtype.float64 (shape abs_float64)
            Float.neg_infinity
        in
        where is_zero neg_inf (log abs_float64)
      in
      let logdet64 =
        if ndim log_abs_diag > 1 then
          sum log_abs_diag ~axes:[ -1 ] ~keepdims:false
        else sum log_abs_diag
      in
      let logdet = cast Dtype.float32 logdet64 in
      (sign_float, logdet)

  and det a =
    let@ _ = span ~op:"det" () in
    check_square ~op:"det" a;
    check_float_or_complex ~op:"det" a;
    let sign, logabs = slogdet a in
    let dtype_a = dtype a in
    let abs_det = exp logabs |> cast dtype_a in
    let sign_cast = cast dtype_a sign in
    mul sign_cast abs_det

  let matrix_rank ?tol ?rtol ?hermitian a =
    let@ _ = span ~op:"matrix_rank" () in
    check_float_or_complex ~op:"matrix_rank" a;
    let s =
      match hermitian with
      | Some true ->
          (* Use eigenvalue decomposition for hermitian matrices *)
          let vals, _ = B.eigh ~vectors:false a in
          (* Use absolute values to match SVD behavior for tolerance
             computation *)
          abs vals
      | _ ->
          (* Use SVD for general matrices *)
          svdvals a
    in
    let max_s = max s |> unsafe_get [] in
    let m, n =
      shape a |> fun sh -> (sh.(Array.length sh - 2), sh.(Array.length sh - 1))
    in
    (* Use appropriate epsilon for the dtype *)
    let dtype_a = dtype a in
    let eps =
      if
        Dtype.equal dtype_a Dtype.float32 || Dtype.equal dtype_a Dtype.complex64
      then 1.2e-7
      else if
        Dtype.equal dtype_a Dtype.float64
        || Dtype.equal dtype_a Dtype.complex128
      then 2.2e-16
      else 1e-15 (* Default for other types *)
    in
    let tol =
      match (tol, rtol) with
      | Some t, _ -> t
      | None, Some r -> r *. max_s
      | None, None -> float_of_int (Stdlib.max m n) *. eps *. max_s
    in
    let mask = greater s (scalar (B.context a) (dtype s) tol) in
    let mask = cast (dtype s) mask in
    let count = sum mask |> unsafe_get [] in
    int_of_float (Float.round count)

  let trace ?out ?offset a =
    let@ _ = span ~op:"trace" () in
    let offset = Option.value offset ~default:0 in
    let sh = shape a in
    let n = Array.length sh in
    if n < 2 then
      Error.invalid ~op:"trace" ~what:"input"
        ~reason:"requires at least 2D array" ();

    (* Extract diagonal and sum it *)
    let diag = diagonal ~offset a in
    sum ?out diag ~axes:[ -1 ] ~keepdims:false

  (* Solving Linear Systems *)

  let solve a b =
    let@ _ = span ~op:"solve" () in
    check_square ~op:"solve" a;
    check_float_or_complex ~op:"solve" a;
    check_float_or_complex ~op:"solve" b;

    (* Handle batch dimension compatibility *)
    let a_ndim = ndim a in
    let b_ndim = ndim b in
    let b_expanded =
      if a_ndim > 2 && b_ndim = 2 then
        (* Check if b could be batch of vectors matching a's batch size *)
        let a_shape = shape a in
        let b_shape = shape b in
        let a_batch_size =
          Array.fold_left ( * ) 1 (Array.sub a_shape 0 (a_ndim - 2))
        in
        if b_shape.(0) = a_batch_size && b_shape.(1) = a_shape.(a_ndim - 2) then
          (* Expand b from [batch, n] to [batch, n, 1] *)
          expand_dims [ -1 ] b
        else b
      else b
    in

    (* Use QR decomposition *)
    let q, r = B.qr ~reduced:true a in
    let r_diag = diagonal r |> cast Dtype.float64 in
    let m = dim (-2) a in
    let eps = if Dtype.equal (dtype a) Dtype.float32 then 1e-6 else 1e-12 in
    let tol = eps *. float_of_int m in
    let tol_tensor = full (B.context r_diag) Dtype.float64 (shape r_diag) tol in
    let zero_mask = less (abs r_diag) tol_tensor in
    let zero_count = sum (cast Dtype.float64 zero_mask) |> unsafe_get [] in
    if zero_count > 0. then invalid_arg "solve: matrix is singular";
    let y = matmul (matrix_transpose q) b_expanded in
    let result =
      B.triangular_solve ~upper:true ~transpose:false ~unit_diag:false r y
    in

    (* Squeeze result if we expanded b *)
    if b_expanded != b then squeeze ~axes:[ ndim result - 1 ] result else result

  (* Complex helpers placed before pinv to allow conjugation support *)

  let complex (type a b) ~(real : (a, b) t) ~(imag : (a, b) t) =
    let@ _ = span ~op:"complex" () in
    (* Check shapes match *)
    let real_shape = shape real in
    let imag_shape = shape imag in
    if real_shape <> imag_shape then
      Error.shape_mismatch ~op:"complex" ~expected:real_shape ~actual:imag_shape
        ();

    (* Create complex tensor based on the input dtype *)
    let size = Array.fold_left ( * ) 1 real_shape in
    match dtype real with
    | Float32 ->
        let real = (real : (float, float32_elt) t) in
        let imag = (imag : (float, float32_elt) t) in
        let complex_data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i real_shape |> Array.to_list in
              let re = unsafe_get idx real in
              let im = unsafe_get idx imag in
              Complex.{ re; im })
        in
        Obj.magic (create (B.context real) complex64 real_shape complex_data)
    | Float64 ->
        let real = (real : (float, float64_elt) t) in
        let imag = (imag : (float, float64_elt) t) in
        let complex_data =
          Array.init size (fun i ->
              let idx = Shape.unravel_index i real_shape |> Array.to_list in
              let re = unsafe_get idx real in
              let im = unsafe_get idx imag in
              Complex.{ re; im })
        in
        Obj.magic (create (B.context real) complex64 real_shape complex_data)
    | _ ->
        Error.invalid ~op:"complex" ~what:"dtype"
          ~reason:"real and imag must be float32 or float64" ()

  let pinv (type a b) ?rtol ?hermitian (a : (a, b) t) =
    let@ _ = span ~op:"pinv" () in
    check_float_or_complex ~op:"pinv" a;
    let m, n =
      shape a |> fun sh -> (sh.(Array.length sh - 2), sh.(Array.length sh - 1))
    in

    let dtype_a = dtype a in

    let eps_for_dtype =
      if
        Dtype.equal dtype_a Dtype.float32 || Dtype.equal dtype_a Dtype.complex64
      then 1.2e-7
      else if
        Dtype.equal dtype_a Dtype.float64
        || Dtype.equal dtype_a Dtype.complex128
      then 2.2e-16
      else 1e-15
    in

    let max_dim = float_of_int (Stdlib.max m n) in

    let cutoff ~max_s =
      match rtol with
      | Some rtol_value -> rtol_value *. max_s *. max_dim
      | None -> max_dim *. eps_for_dtype *. max_s
    in

    let pinv_from_factors u s vh =
      let max_s = max s |> unsafe_get [] in
      let cutoff = cutoff ~max_s in
      let ones_s = ones (B.context s) (dtype s) (shape s) in
      let threshold = scalar (B.context s) (dtype s) cutoff in
      let mask = greater s threshold in
      let safe_s = where mask s ones_s in
      let s_inv = div ones_s safe_s in
      let mask_float = cast (dtype s) mask in
      let s_inv = mul s_inv mask_float |> cast dtype_a in
      let s_inv_expanded = unsqueeze ~axes:[ 0 ] s_inv in
      let v = matrix_transpose vh in
      let vs = mul v s_inv_expanded in
      if Dtype.is_complex dtype_a then
        let u_adj = matrix_transpose (conjugate u) in
        matmul vs u_adj
      else matmul vs (matrix_transpose u)
    in

    let pinv_via_svd () =
      let u, s, vh = B.svd ~full_matrices:false a in
      pinv_from_factors u s vh
    in

    match hermitian with
    | Some true -> (
        let vals, vecs_opt = B.eigh ~vectors:true a in
        match vecs_opt with
        | None -> pinv_via_svd ()
        | Some vecs ->
            let abs_vals = abs vals in
            let sign_vals = sign vals in
            let ones_vals = ones (B.context vals) (dtype vals) (shape vals) in
            let zeros_vals = zeros (B.context vals) (dtype vals) (shape vals) in
            let sign_vals =
              where (cmpeq sign_vals zeros_vals) ones_vals sign_vals
            in
            let sign_cast = cast dtype_a sign_vals in
            let sign_expanded = expand_dims [ -1 ] sign_cast in
            let vh = mul sign_expanded (matrix_transpose vecs) in
            pinv_from_factors vecs abs_vals vh)
    | _ -> pinv_via_svd ()

  let lstsq ?rcond a b =
    let@ _ = span ~op:"lstsq" () in
    check_float_or_complex ~op:"lstsq" a;
    check_float_or_complex ~op:"lstsq" b;
    let m, n =
      shape a |> fun sh -> (sh.(Array.length sh - 2), sh.(Array.length sh - 1))
    in
    let rcond_value =
      match rcond with
      | Some v -> v
      | None ->
          let eps =
            if Dtype.equal (dtype a) Dtype.float32 then 1.2e-7
            else if Dtype.equal (dtype a) Dtype.float64 then 2.2e-16
            else 1e-15
          in
          let max_s = max (svdvals a) |> unsafe_get [] in
          float_of_int (Stdlib.max m n) *. eps *. max_s
    in
    let x =
      if m >= n then
        let q, r = B.qr ~reduced:true a in
        let y = matmul (matrix_transpose q) b in
        let r_square =
          if ndim r = 2 then slice_internal [ R (0, n); R (0, n) ] r
          else slice_internal [ A; R (0, n); R (0, n) ] r
        in
        let y_top =
          if ndim y = 2 then slice_internal [ R (0, n); A ] y
          else if ndim y = 1 then slice_internal [ R (0, n) ] y
          else slice_internal [ A; R (0, n); A ] y
        in
        B.triangular_solve ~upper:true ~transpose:false ~unit_diag:false
          r_square y_top
      else
        let a_pseudo = pinv a ~rtol:rcond_value in
        matmul a_pseudo b
    in
    let residuals =
      if m > n then
        let res = sub b (matmul a x) in
        sum (square res) ~axes:[ ndim res - 2 ] ~keepdims:false
      else zeros (B.context a) (dtype b) [||]
    in
    let rank = matrix_rank a in
    let s = svdvals a in
    (x, residuals, rank, s)

  let inv a =
    let@ _ = span ~op:"inv" () in
    check_square ~op:"inv" a;
    check_float_or_complex ~op:"inv" a;

    let sh = shape a in
    let n = sh.(Array.length sh - 1) in
    let batch_shape = Array.sub sh 0 (Array.length sh - 2) in
    let eye_shape = Array.append batch_shape [| n; n |] in
    let i = eye (B.context a) (dtype a) n in
    let i = broadcast_to eye_shape i in
    try solve a i
    with Invalid_argument msg when String.sub msg 0 5 = "solve" ->
      invalid_arg ("inv" ^ String.sub msg 5 (String.length msg - 5))

  let matrix_power a n =
    let@ _ = span ~op:"matrix_power" () in
    let shape_a = shape a in
    let ndim_a = Array.length shape_a in
    if ndim_a < 2 then
      Error.invalid ~op:"matrix_power" ~what:"input"
        ~reason:"requires at least 2D array" ();

    let m = shape_a.(ndim_a - 2) in
    let k = shape_a.(ndim_a - 1) in
    if m <> k then
      Error.invalid ~op:"matrix_power" ~what:"matrix"
        ~reason:(Printf.sprintf "must be square, got %dx%d" m k)
        ();

    if n = 0 then eye (B.context a) (dtype a) m
    else if n = 1 then copy a
    else if n > 0 then
      let rec power acc base exp =
        if exp = 0 then acc
        else if exp mod 2 = 0 then power acc (matmul base base) (exp / 2)
        else power (matmul acc base) (matmul base base) (exp / 2)
      in
      power a a (n - 1)
    else
      try
        let inv_a = inv a in
        let pos_n = -n in
        if pos_n = 1 then inv_a
        else
          let rec power acc base exp =
            if exp = 0 then acc
            else if exp mod 2 = 0 then power acc (matmul base base) (exp / 2)
            else power (matmul acc base) (matmul base base) (exp / 2)
          in
          power inv_a inv_a (pos_n - 1)
      with Invalid_argument _ ->
        invalid_arg "matrix_power: singular for negative exponent"

  let cond ?p x =
    check_square ~op:"cond" x;
    check_float_or_complex ~op:"cond" x;
    match p with
    | None | Some `Two ->
        let s = svdvals x in
        let dtype_s = dtype s in
        let max_s_tensor = max s in
        let max_s = max_s_tensor |> unsafe_get [] in
        let eps =
          if Dtype.equal dtype_s Dtype.float32 then 1.2e-7
          else if Dtype.equal dtype_s Dtype.float64 then 2.2e-16
          else 1e-15
        in
        let tol = eps *. max_s in
        let tol_tensor = scalar (B.context x) dtype_s tol in
        let safe_s = where (greater s tol_tensor) s tol_tensor in
        let min_s_tensor =
          if ndim safe_s > 1 then min safe_s ~axes:[ -1 ] ~keepdims:false
          else min safe_s
        in
        let ratio = div max_s_tensor min_s_tensor in
        cast (dtype x) ratio
    | Some `One ->
        let inv_x = inv x in
        let norm_x = norm ~ord:`One x in
        let norm_inv = norm ~ord:`One inv_x in
        mul norm_x norm_inv
    | Some `Inf ->
        let inv_x = inv x in
        let norm_x = norm ~ord:`Inf x in
        let norm_inv = norm ~ord:`Inf inv_x in
        mul norm_x norm_inv
    | _ -> Error.failed ~op:"cond" ~what:"unsupported norm" ()

  let tensorsolve ?axes a b =
    check_float_or_complex ~op:"tensorsolve" a;
    check_float_or_complex ~op:"tensorsolve" b;
    let a_shape = shape a in
    let b_shape = shape b in
    let a_rank = Array.length a_shape in
    let b_rank = Array.length b_shape in
    if b_rank = 0 then
      Error.invalid ~op:"tensorsolve" ~what:"b"
        ~reason:"must have at least one dimension" ();
    if a_rank < b_rank then
      Error.invalid ~op:"tensorsolve" ~what:"a"
        ~reason:"rank must be >= rank of b" ();

    let axes_for_b =
      match axes with
      | None -> Array.init b_rank (fun i -> a_rank - b_rank + i)
      | Some axes ->
          if List.length axes <> b_rank then
            Error.invalid ~op:"tensorsolve" ~what:"axes"
              ~reason:
                (Printf.sprintf "expected %d entries, got %d" b_rank
                   (List.length axes))
              ();
          let ax_arr = Array.of_list axes in
          let seen = Array.make a_rank false in
          Array.map
            (fun ax ->
              let axis = if ax < 0 then ax + a_rank else ax in
              if axis < 0 || axis >= a_rank then
                Error.axis_out_of_bounds ~op:"tensorsolve" ~axis:ax ~ndim:a_rank
                  ();
              if seen.(axis) then
                Error.invalid ~op:"tensorsolve"
                  ~what:(Printf.sprintf "axis %d" ax)
                  ~reason:"repeated" ();
              seen.(axis) <- true;
              axis)
            ax_arr
    in

    let selected = Array.make a_rank false in
    Array.iter (fun ax -> selected.(ax) <- true) axes_for_b;
    let free_axes =
      Array.init a_rank Fun.id |> Array.to_list
      |> List.filter (fun ax -> not selected.(ax))
      |> Array.of_list
    in
    let permutation = Array.append free_axes axes_for_b in
    let a_perm =
      let rec is_identity idx =
        if idx = a_rank then true
        else if permutation.(idx) <> idx then false
        else is_identity (idx + 1)
      in
      if is_identity 0 then a else transpose ~axes:(Array.to_list permutation) a
    in
    let perm_shape = shape a_perm in
    let free_rank = Array.length free_axes in
    let free_shape = Array.sub perm_shape 0 free_rank in
    let rhs_shape = Array.sub perm_shape free_rank b_rank in
    if rhs_shape <> b_shape then
      Error.shape_mismatch ~op:"tensorsolve" ~expected:b_shape ~actual:rhs_shape
        ();

    let rows = array_prod free_shape in
    let cols = array_prod rhs_shape in
    if rows <> cols then
      Error.invalid ~op:"tensorsolve" ~what:"a"
        ~reason:"leading dimensions must match trailing dimensions" ();

    let a_mat = reshape [| rows; cols |] a_perm in
    let b_vec = reshape [| rows |] b in
    let solution =
      try solve a_mat b_vec
      with Invalid_argument _ ->
        let pinv_a = pinv a_mat in
        let b_col = reshape [| rows; 1 |] b_vec in
        let x_col = matmul pinv_a b_col in
        reshape [| cols |] x_col
    in
    reshape free_shape solution

  let tensorinv ?ind a =
    check_float_or_complex ~op:"tensorinv" a;
    let shape_a = shape a in
    let rank = Array.length shape_a in
    if rank = 0 then
      Error.invalid ~op:"tensorinv" ~what:"input"
        ~reason:"must have at least one dimension" ();
    let ind = Option.value ind ~default:(rank / 2) in
    if ind <= 0 || ind >= rank then
      Error.invalid ~op:"tensorinv" ~what:"ind"
        ~reason:"must split dimensions into two non-empty groups" ();
    let left_dims = Array.sub shape_a 0 ind in
    let right_dims = Array.sub shape_a ind (rank - ind) in
    let left_size = array_prod left_dims in
    let right_size = array_prod right_dims in
    if left_size <> right_size then
      Error.invalid ~op:"tensorinv" ~what:"input"
        ~reason:"leading and trailing dimensions must have equal product" ();
    let a_mat = reshape [| left_size; right_size |] a in
    let inv_mat = try inv a_mat with Invalid_argument _ -> pinv a_mat in
    let out_shape = Array.append right_dims left_dims in
    reshape out_shape inv_mat

  (* ───── Complex Operations and FFT ───── *)

  (* FFT operations *)

  type fft_norm = [ `Backward | `Forward | `Ortho ]

  (* Helper to pad or truncate along axes *)
  let pad_or_truncate_for_fft x axes s =
    if s = None then x
    else
      let s_arr = Array.of_list (Option.get s) in
      let x_padded = ref x in
      List.iteri
        (fun i ax ->
          let ax = if ax < 0 then ndim !x_padded + ax else ax in
          let cur_size = dim ax !x_padded in
          let target = s_arr.(i) in
          if target <> cur_size then
            if target > cur_size then (
              (* Zero-pad at the end for FFT *)
              let pad_config = Array.make (ndim !x_padded) (0, 0) in
              let pad_amount = target - cur_size in
              pad_config.(ax) <- (0, pad_amount);
              x_padded :=
                B.pad !x_padded pad_config (Dtype.zero (dtype !x_padded)))
            else
              (* Truncate from the end for FFT - keep low frequencies *)
              let shrink_config =
                Array.init (ndim !x_padded) (fun idx ->
                    if idx = ax then (0, target) else (0, dim idx !x_padded))
              in
              x_padded := B.shrink !x_padded shrink_config)
        axes;
      !x_padded

  let fftn (type a) ?out ?axes ?s ?(norm = `Backward) (x : (Complex.t, a) t) :
      (Complex.t, a) t =
    let@ _ = span ~op:"fftn" () in
    let ndim_x = ndim x in
    let axes_list =
      match axes with
      | None -> List.init ndim_x Fun.id
      | Some a -> List.map (fun ax -> if ax < 0 then ndim_x + ax else ax) a
    in

    (* Validate s parameter *)
    (match s with
    | Some sizes when List.length sizes <> List.length axes_list ->
        Error.invalid ~op:"fft" ~what:"s parameter"
          ~reason:"must have same length as axes" ()
    | _ -> ());

    (* Pad or truncate if needed *)
    let x_padded = pad_or_truncate_for_fft x axes_list s in

    (* Compute normalization scale *)
    let norm_scale =
      match norm with
      | `Backward -> 1.0 (* No scaling on forward *)
      | `Forward ->
          let n =
            List.fold_left (fun acc ax -> acc * dim ax x_padded) 1 axes_list
          in
          1.0 /. float_of_int n
      | `Ortho ->
          let n =
            List.fold_left (fun acc ax -> acc * dim ax x_padded) 1 axes_list
          in
          1.0 /. Stdlib.sqrt (float_of_int n)
    in

    (* Apply normalization if needed *)
    if norm_scale <> 1.0 then
      let result = B.fft x_padded ~axes:(Array.of_list axes_list) in
      let scale_value =
        match B.dtype result with
        | Complex64 | Complex128 -> Complex.{ re = norm_scale; im = 0.0 }
      in
      let scale_tensor =
        scalar (B.context result) (B.dtype result) scale_value
      in
      mul ?out result scale_tensor
    else B.fft ?out x_padded ~axes:(Array.of_list axes_list)

  let ifftn (type a) ?out ?axes ?s ?(norm = `Backward) (x : (Complex.t, a) t) :
      (Complex.t, a) t =
    let@ _ = span ~op:"ifftn" () in
    let ndim_x = ndim x in
    let axes_list =
      match axes with
      | None -> List.init ndim_x Fun.id
      | Some a -> List.map (fun ax -> if ax < 0 then ndim_x + ax else ax) a
    in

    (* Validate s parameter *)
    (match s with
    | Some sizes when List.length sizes <> List.length axes_list ->
        Error.invalid ~op:"ifft" ~what:"s parameter"
          ~reason:"must have same length as axes" ()
    | _ -> ());

    (* For IFFT, we need special handling of the size parameter *)
    let x_input, norm_scale =
      match s with
      | None ->
          (* No size specified, standard IFFT *)
          let norm_scale =
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
          (x, norm_scale)
      | Some sizes ->
          (* Size specified - we need to handle this carefully *)
          (* First pad/truncate the input in frequency domain, then do IFFT *)
          let x_padded = pad_or_truncate_for_fft x axes_list s in
          let norm_scale =
            match norm with
            | `Backward ->
                (* Use the OUTPUT size for normalization (after truncation) *)
                let n = ref 1 in
                List.iter (fun size -> n := !n * size) sizes;
                1.0 /. float_of_int !n
            | `Forward -> 1.0
            | `Ortho ->
                let n = ref 1 in
                List.iter (fun size -> n := !n * size) sizes;
                1.0 /. Stdlib.sqrt (float_of_int !n)
          in
          (x_padded, norm_scale)
    in

    (* Apply normalization if needed *)
    if norm_scale <> 1.0 then
      let result = B.ifft x_input ~axes:(Array.of_list axes_list) in
      let scale_value =
        match B.dtype result with
        | Complex64 | Complex128 -> Complex.{ re = norm_scale; im = 0.0 }
      in
      let scale_tensor =
        scalar (B.context result) (B.dtype result) scale_value
      in
      mul ?out result scale_tensor
    else B.ifft ?out x_input ~axes:(Array.of_list axes_list)

  let rfftn ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"rfftn" () in
    let ndim_x = ndim x in
    let axes_list = match axes with None -> [ ndim_x - 1 ] | Some ax -> ax in

    (* Pad or truncate if needed *)
    let x_padded = pad_or_truncate_for_fft x axes_list s in

    (* Compute normalization scale *)
    let norm_scale =
      match norm with
      | `Backward -> 1.0
      | `Forward ->
          let n =
            List.fold_left (fun acc ax -> acc * dim ax x_padded) 1 axes_list
          in
          1.0 /. float_of_int n
      | `Ortho ->
          let n =
            List.fold_left (fun acc ax -> acc * dim ax x_padded) 1 axes_list
          in
          1.0 /. Stdlib.sqrt (float_of_int n)
    in

    (* Use Complex64 as default - matches NumPy behavior *)
    if norm_scale <> 1.0 then
      let result =
        B.rfft x_padded ~dtype:Dtype.Complex128 ~axes:(Array.of_list axes_list)
      in
      let scale_value = Complex.{ re = norm_scale; im = 0.0 } in
      let scale_tensor =
        scalar (B.context result) (B.dtype result) scale_value
      in
      mul ?out result scale_tensor
    else
      B.rfft ?out x_padded ~dtype:Dtype.Complex128
        ~axes:(Array.of_list axes_list)

  let irfftn ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"irfftn" () in
    let ndim_x = ndim x in
    let axes_list = match axes with None -> [ ndim_x - 1 ] | Some ax -> ax in

    (* Determine output sizes *)
    let output_sizes =
      match s with
      | Some sizes -> sizes
      | None ->
          (* Infer sizes from input shape *)
          let input_shape = shape x in
          List.mapi
            (fun i axis ->
              let axis = if axis < 0 then ndim_x + axis else axis in
              if i = List.length axes_list - 1 then
                (* Last axis: reconstruct full size from hermitian input *)
                (input_shape.(axis) - 1) * 2
              else input_shape.(axis))
            axes_list
    in

    (* For normalization: when output size is specified and smaller than what
       the frequency domain suggests, use the output size for normalization.
       This handles the case of truncation in frequency domain. *)
    let norm_sizes =
      let input_shape = shape x in
      List.mapi
        (fun i axis ->
          let axis = if axis < 0 then ndim_x + axis else axis in
          if i = List.length axes_list - 1 then
            (* Last axis: use specified size if provided, else infer *)
            match s with
            | Some sizes ->
                (* Use the specified output size for normalization *)
                List.nth sizes i
            | None ->
                (* No size specified: use inferred size *)
                let inferred_size = (input_shape.(axis) - 1) * 2 in
                inferred_size
          else
            (* Other axes: use output size if specified, else input size *)
            match s with
            | Some sizes -> List.nth sizes i
            | None -> input_shape.(axis))
        axes_list
    in

    (* Compute normalization scale based on the actual FFT size *)
    let norm_scale =
      match norm with
      | `Backward ->
          let n = List.fold_left (fun acc size -> acc * size) 1 norm_sizes in
          1.0 /. float_of_int n
      | `Forward -> 1.0
      | `Ortho ->
          let n = List.fold_left (fun acc size -> acc * size) 1 norm_sizes in
          1.0 /. Stdlib.sqrt (float_of_int n)
    in

    (* Use Float64 as default - matches NumPy behavior *)
    let s_param =
      match s with None -> None | Some _ -> Some (Array.of_list output_sizes)
    in

    if norm_scale <> 1.0 then
      let result =
        B.irfft ?s:s_param x ~dtype:Dtype.Float64
          ~axes:(Array.of_list axes_list)
      in
      let scale_tensor =
        scalar (B.context result) (B.dtype result) norm_scale
      in
      mul ?out result scale_tensor
    else
      B.irfft ?out ?s:s_param x ~dtype:Dtype.Float64
        ~axes:(Array.of_list axes_list)

  (* 1D FFT operations - convenience functions *)
  let fft ?out ?(axis = -1) ?n ?(norm = `Backward) x =
    let@ _ = span ~op:"fft" () in
    let n_param = match n with None -> None | Some size -> Some [ size ] in
    fftn ?out x ~axes:[ axis ] ?s:n_param ~norm

  let ifft ?out ?(axis = -1) ?n ?(norm = `Backward) x =
    let@ _ = span ~op:"ifft" () in
    let n_param = match n with None -> None | Some size -> Some [ size ] in
    ifftn ?out x ~axes:[ axis ] ?s:n_param ~norm

  let rfft ?out ?(axis = -1) ?n ?(norm = `Backward) x =
    let@ _ = span ~op:"rfft" () in
    let n_param = match n with None -> None | Some size -> Some [ size ] in
    rfftn ?out x ~axes:[ axis ] ?s:n_param ~norm

  let irfft ?out ?(axis = -1) ?n ?(norm = `Backward) x =
    let@ _ = span ~op:"irfft" () in
    let n_param = match n with None -> None | Some size -> Some [ size ] in
    irfftn ?out x ~axes:[ axis ] ?s:n_param ~norm

  (* 2D FFT operations *)
  let fft2 ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"fft2" () in
    let n = ndim x in
    if n < 2 then
      Error.invalid ~op:"fft2" ~what:"input"
        ~reason:(Printf.sprintf "requires at least 2D array, got %dD" n)
        ();
    let axes_list =
      match axes with None -> [ n - 2; n - 1 ] | Some ax -> ax
    in
    if List.length axes_list <> 2 then
      Error.invalid ~op:"fft2" ~what:"axes"
        ~reason:"must specify exactly 2 axes" ();
    fftn ?out x ~axes:axes_list ?s ~norm

  let ifft2 ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"ifft2" () in
    let n = ndim x in
    if n < 2 then
      Error.invalid ~op:"ifft2" ~what:"input"
        ~reason:(Printf.sprintf "requires at least 2D array, got %dD" n)
        ();
    let axes_list =
      match axes with None -> [ n - 2; n - 1 ] | Some ax -> ax
    in
    if List.length axes_list <> 2 then
      Error.invalid ~op:"ifft2" ~what:"axes"
        ~reason:"must specify exactly 2 axes" ();
    ifftn ?out x ~axes:axes_list ?s ~norm

  (* N-dimensional FFT operations *)
  let fftn ?out ?axes ?s ?(norm = `Backward) x =
    let axes_list =
      match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax
    in
    fftn ?out x ~axes:axes_list ?s ~norm

  let ifftn ?out ?axes ?s ?(norm = `Backward) x =
    let axes_list =
      match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax
    in
    ifftn ?out x ~axes:axes_list ?s ~norm

  (* 2D Real FFT operations *)
  let rfft2 ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"rfft2" () in
    let n = ndim x in
    if n < 2 then
      Error.invalid ~op:"rfft2" ~what:"input"
        ~reason:(Printf.sprintf "requires at least 2D array, got %dD" n)
        ();
    let axes_list =
      match axes with None -> [ n - 2; n - 1 ] | Some ax -> ax
    in
    if List.length axes_list <> 2 then
      Error.invalid ~op:"rfft2" ~what:"axes"
        ~reason:"must specify exactly 2 axes" ();
    rfftn ?out x ~axes:axes_list ?s ~norm

  let irfft2 ?out ?axes ?s ?(norm = `Backward) x =
    let@ _ = span ~op:"irfft2" () in
    let n = ndim x in
    if n < 2 then
      Error.invalid ~op:"irfft2" ~what:"input"
        ~reason:(Printf.sprintf "requires at least 2D array, got %dD" n)
        ();
    let axes_list =
      match axes with None -> [ n - 2; n - 1 ] | Some ax -> ax
    in
    if List.length axes_list <> 2 then
      Error.invalid ~op:"irfft2" ~what:"axes"
        ~reason:"must specify exactly 2 axes" ();
    irfftn ?out x ~axes:axes_list ?s ~norm

  (* N-dimensional Real FFT operations *)
  let rfftn ?out ?axes ?s ?(norm = `Backward) x =
    let axes_list =
      match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax
    in
    rfftn ?out x ~axes:axes_list ?s ~norm

  let irfftn ?out ?axes ?s ?(norm = `Backward) x =
    let axes_list =
      match axes with None -> List.init (ndim x) Fun.id | Some ax -> ax
    in
    irfftn ?out x ~axes:axes_list ?s ~norm

  (* Hermitian FFT operations *)
  let hfft ?(axis = -1) ?n ?norm x =
    let n = match n with None -> 2 * (dim axis x - 1) | Some n -> n in
    let axis = resolve_single_axis x axis in
    irfftn x ~axes:[ axis ] ~s:[ n ] ?norm

  let ihfft ?(axis = -1) ?n ?norm x =
    let n = match n with None -> dim axis x | Some n -> n in
    let axis = resolve_single_axis x axis in
    rfftn x ~axes:[ axis ] ~s:[ n ] ?norm

  (* FFT helper functions *)
  let fftfreq ctx ?(d = 1.0) n =
    let@ _ = span ~op:"fftfreq" () in
    (* Return the Discrete Fourier Transform sample frequencies *)
    let dtype = Dtype.float64 in
    let val_ = 1.0 /. (float_of_int n *. d) in
    let results =
      if n mod 2 = 0 then
        (* Even case *)
        let p1 = arange ctx Dtype.int32 0 (n / 2) 1 in
        let p2 = arange ctx Dtype.int32 (-(n / 2)) 0 1 in
        concatenate ~axis:0 [ cast dtype p1; cast dtype p2 ]
      else
        (* Odd case *)
        let p1 = arange ctx Dtype.int32 0 ((n + 1) / 2) 1 in
        let p2 = arange ctx Dtype.int32 (-((n - 1) / 2)) 0 1 in
        concatenate ~axis:0 [ cast dtype p1; cast dtype p2 ]
    in
    mul_s results val_

  let rfftfreq ctx ?(d = 1.0) n =
    let@ _ = span ~op:"rfftfreq" () in
    (* Return the Discrete Fourier Transform sample frequencies for rfft *)
    let dtype = Dtype.float64 in
    let val_ = 1.0 /. (float_of_int n *. d) in
    let results = arange ctx Dtype.int32 0 ((n / 2) + 1) 1 in
    let scale_tensor = scalar ctx dtype val_ in
    mul (cast dtype results) scale_tensor

  let fftshift ?axes x =
    let@ _ = span ~op:"fftshift" () in
    (* Shift the zero-frequency component to the center of the spectrum *)
    let shape_x = shape x in
    let ndim_x = Array.length shape_x in
    let axes_list =
      match axes with None -> List.init ndim_x Fun.id | Some ax -> ax
    in
    (* For each axis, roll by shape[axis] // 2 *)
    List.fold_left
      (fun acc axis ->
        let axis = resolve_single_axis acc axis in
        let n = shape_x.(axis) in
        let shift = n / 2 in
        roll shift acc ~axis)
      x axes_list

  let ifftshift ?axes x =
    let@ _ = span ~op:"ifftshift" () in
    (* The inverse of fftshift *)
    let shape_x = shape x in
    let ndim_x = Array.length shape_x in
    let axes_list =
      match axes with None -> List.init ndim_x Fun.id | Some ax -> ax
    in
    (* For each axis, roll by -(shape[axis] // 2) *)
    List.fold_left
      (fun acc axis ->
        let axis = resolve_single_axis acc axis in
        let n = shape_x.(axis) in
        let shift = -(n / 2) in
        roll shift acc ~axis)
      x axes_list

  (* ───── Neural Network Operations ───── *)

  (* Softmax: exp(scale * (x - max(x))) / sum(exp(scale * (x - max(x)))) along
     specified axes *)
  let softmax ?out ?(axes = [ -1 ]) ?(scale = 1.0) x =
    let@ _ = span ~op:"softmax" () in
    let ndim = Array.length (shape x) in
    let axes_normalized =
      List.map (fun ax -> if ax < 0 then ndim + ax else ax) axes
    in
    let max_x = max x ~axes:axes_normalized ~keepdims:true in
    let x_shifted =
      if scale = 1.0 then sub x max_x
      else
        let scaled = mul (scalar_like x scale) (sub x max_x) in
        scaled
    in
    let exp_x = exp x_shifted in
    let sum_exp = sum exp_x ~axes:axes_normalized ~keepdims:true in
    div ?out exp_x sum_exp

  let log_softmax ?out ?(axes = [ -1 ]) ?(scale = 1.0) x =
    let ndim = ndim x in
    let axes_sorted =
      List.map
        (fun ax ->
          let axis = if ax < 0 then ndim + ax else ax in
          if axis < 0 || axis >= ndim then
            Error.axis_out_of_bounds ~op:"log_softmax" ~axis:ax ~ndim ()
          else axis)
        axes
      |> List.sort compare
    in
    let rec dedup prev acc = function
      | [] -> List.rev acc
      | h :: t when Some h = prev -> dedup prev acc t
      | h :: t -> dedup (Some h) (h :: acc) t
    in
    let axes_norm = dedup None [] axes_sorted in
    if axes_norm = [] then
      match out with
      | Some o ->
          B.add ~out:o (zeros_like x) (zeros_like x);
          o
      | None -> zeros_like x
    else
      let max_x = max x ~axes:axes_norm ~keepdims:true in
      let shifted = sub x max_x in
      let scaled_shifted =
        if scale = 1.0 then shifted else mul (scalar_like shifted scale) shifted
      in
      let log_den =
        let sum_exp = sum (exp scaled_shifted) ~axes:axes_norm ~keepdims:true in
        log sum_exp
      in
      sub ?out scaled_shifted log_den

  let logsumexp ?out ?axes ?(keepdims = false) x =
    let ndim = ndim x in
    let axes_list =
      match axes with
      | None -> List.init ndim Fun.id
      | Some lst ->
          List.map
            (fun ax ->
              let axis = if ax < 0 then ndim + ax else ax in
              if axis < 0 || axis >= ndim then
                Error.axis_out_of_bounds ~op:"logsumexp" ~axis:ax ~ndim ()
              else axis)
            lst
    in
    let axes_sorted = List.sort compare axes_list in
    let rec dedup prev acc = function
      | [] -> List.rev acc
      | h :: t when Some h = prev -> dedup prev acc t
      | h :: t -> dedup (Some h) (h :: acc) t
    in
    let axes_norm = dedup None [] axes_sorted in
    if axes_norm = [] then
      match out with
      | Some o ->
          B.add ~out:o x (zeros_like x);
          o
      | None -> x
    else
      let max_x = max x ~axes:axes_norm ~keepdims:true in
      let shifted = sub x max_x in
      let sum_exp = sum (exp shifted) ~axes:axes_norm ~keepdims:true in
      let log_sum = add (log sum_exp) max_x in
      if keepdims then
        match out with
        | Some o ->
            B.add ~out:o log_sum (zeros_like log_sum);
            o
        | None -> log_sum
      else
        let axes_desc = List.rev axes_norm in
        let result = squeeze ~axes:axes_desc log_sum in
        match out with
        | Some o ->
            B.add ~out:o result (zeros_like result);
            o
        | None -> result

  let logmeanexp ?out ?axes ?(keepdims = false) x =
    let ndim = ndim x in
    let axes_list =
      match axes with
      | None -> List.init ndim Fun.id
      | Some lst ->
          List.map
            (fun ax ->
              let axis = if ax < 0 then ndim + ax else ax in
              if axis < 0 || axis >= ndim then
                Error.axis_out_of_bounds ~op:"logmeanexp" ~axis:ax ~ndim ()
              else axis)
            lst
    in
    let axes_sorted = List.sort compare axes_list in
    let rec dedup prev acc = function
      | [] -> List.rev acc
      | h :: t when Some h = prev -> dedup prev acc t
      | h :: t -> dedup (Some h) (h :: acc) t
    in
    let axes_norm = dedup None [] axes_sorted in
    if axes_norm = [] then
      match out with
      | Some o ->
          B.add ~out:o x (zeros_like x);
          o
      | None -> x
    else
      let log_sum = logsumexp ~axes:axes_norm ~keepdims:true x in
      let count = List.fold_left (fun acc ax -> acc * dim ax x) 1 axes_norm in
      let count_tensor = scalar_like log_sum (float_of_int count) in
      let log_mean = sub log_sum (log count_tensor) in
      if keepdims then
        match out with
        | Some o ->
            B.add ~out:o log_mean (zeros_like log_mean);
            o
        | None -> log_mean
      else
        let axes_desc = List.rev axes_norm in
        let result = squeeze ~axes:axes_desc log_mean in
        match out with
        | Some o ->
            B.add ~out:o result (zeros_like result);
            o
        | None -> result

  let standardize ?out ?axes ?mean:mean_param ?variance:variance_param
      ?(epsilon = 1e-5) x =
    let ndim = ndim x in
    let axes_list =
      match axes with
      | None -> List.init ndim Fun.id
      | Some lst ->
          List.map
            (fun ax ->
              let axis = if ax < 0 then ndim + ax else ax in
              if axis < 0 || axis >= ndim then
                Error.axis_out_of_bounds ~op:"standardize" ~axis:ax ~ndim ()
              else axis)
            lst
    in
    let axes_sorted = List.sort compare axes_list in
    let rec dedup prev acc = function
      | [] -> List.rev acc
      | h :: t when Some h = prev -> dedup prev acc t
      | h :: t -> dedup (Some h) (h :: acc) t
    in
    let axes_norm = dedup None [] axes_sorted in
    let x_shape = shape x in
    let keep_shape =
      Array.mapi
        (fun idx dim -> if List.exists (( = ) idx) axes_norm then 1 else dim)
        x_shape
    in
    let unaffected_axes =
      List.filter
        (fun idx -> not (List.exists (( = ) idx) axes_norm))
        (List.init ndim Fun.id)
    in
    let core_shape =
      Array.of_list (List.map (fun idx -> x_shape.(idx)) unaffected_axes)
    in
    let broadcast_param name param =
      let param_shape = shape param in
      if param_shape = x_shape then param
      else if param_shape = keep_shape then param
      else if param_shape = core_shape then reshape keep_shape param
      else
        Error.invalid ~op:"standardize" ~what:name
          ~reason:"shape must match normalized axes" ()
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
    let eps = scalar_like x epsilon in
    let denom = sqrt (add variance_tensor eps) in
    let centered = sub x mean_tensor in
    div ?out centered denom

  (* Error function. *)
  let erf ?out x = unaryop ~op_name:"erf" ?out B.erf x

  let im2col ~kernel_size ~stride ~dilation ~padding x =
    B.unfold x ~kernel_size ~stride ~dilation ~padding

  let col2im ~output_size ~kernel_size ~stride ~dilation ~padding x =
    B.fold x ~output_size ~kernel_size ~stride ~dilation ~padding

  (* --- scipy-style correlate / convolve --- *)

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
    let k_rank = ndim kernel in
    let x_rank = ndim x in
    if x_rank < k_rank then
      Error.invalid ~op:"correlate" ~what:"input"
        ~reason:(Printf.sprintf "rank %d < kernel rank %d" x_rank k_rank)
        ();
    let k_shape = shape kernel in
    let input_spatial = Array.sub (shape x) (x_rank - k_rank) k_rank in
    let pad_pairs = correlate_padding ~mode:padding input_spatial k_shape in
    let ones = Array.make k_rank 1 in
    (* unfold: (leading..., spatial...) -> (leading..., kernel_prod, L) *)
    let x_unf =
      B.unfold x ~kernel_size:k_shape ~stride:ones ~dilation:ones
        ~padding:pad_pairs
    in
    (* x_unf: (leading..., kernel_prod, L) *)
    let x_unf_ndim = ndim x_unf in
    let kernel_prod = (shape x_unf).(x_unf_ndim - 2) in
    let l = (shape x_unf).(x_unf_ndim - 1) in
    (* flatten kernel to (kernel_prod, 1) for broadcasting *)
    let k_flat = reshape [| kernel_prod; 1 |] kernel in
    (* multiply and reduce over kernel_prod axis *)
    let prod = mul x_unf k_flat in
    let result = sum prod ~axes:[ x_unf_ndim - 2 ] in
    (* reshape L back to spatial output dims *)
    let leading_shape = Array.sub (shape x) 0 (x_rank - k_rank) in
    let out_spatial =
      Array.init k_rank (fun i ->
          let padded =
            input_spatial.(i) + fst pad_pairs.(i) + snd pad_pairs.(i)
          in
          padded - k_shape.(i) + 1)
    in
    let _ = l in
    reshape (Array.concat [ leading_shape; out_spatial ]) result

  let convolve ?(padding = `Valid) x kernel =
    let k_rank = ndim kernel in
    let all_axes = List.init k_rank Fun.id in
    let flipped = flip ~axes:all_axes kernel in
    correlate ~padding x flipped

  (* --- sliding window filters --- *)

  let maximum_filter ~kernel_size ?stride x =
    let k_rank = Array.length kernel_size in
    let stride = match stride with Some s -> s | None -> kernel_size in
    let ones = Array.make k_rank 1 in
    let zeros = Array.make k_rank (0, 0) in
    let x_unf = B.unfold x ~kernel_size ~stride ~dilation:ones ~padding:zeros in
    let x_unf_ndim = ndim x_unf in
    let reduced = max x_unf ~axes:[ x_unf_ndim - 2 ] ~keepdims:false in
    let x_rank = ndim x in
    let leading_shape = Array.sub (shape x) 0 (x_rank - k_rank) in
    let input_spatial = Array.sub (shape x) (x_rank - k_rank) k_rank in
    let out_spatial =
      Array.init k_rank (fun i ->
          ((input_spatial.(i) - kernel_size.(i)) / stride.(i)) + 1)
    in
    reshape (Array.concat [ leading_shape; out_spatial ]) reduced

  let minimum_filter ~kernel_size ?stride x =
    let k_rank = Array.length kernel_size in
    let stride = match stride with Some s -> s | None -> kernel_size in
    let ones = Array.make k_rank 1 in
    let zeros = Array.make k_rank (0, 0) in
    let x_unf = B.unfold x ~kernel_size ~stride ~dilation:ones ~padding:zeros in
    let x_unf_ndim = ndim x_unf in
    let reduced = min x_unf ~axes:[ x_unf_ndim - 2 ] ~keepdims:false in
    let x_rank = ndim x in
    let leading_shape = Array.sub (shape x) 0 (x_rank - k_rank) in
    let input_spatial = Array.sub (shape x) (x_rank - k_rank) k_rank in
    let out_spatial =
      Array.init k_rank (fun i ->
          ((input_spatial.(i) - kernel_size.(i)) / stride.(i)) + 1)
    in
    reshape (Array.concat [ leading_shape; out_spatial ]) reduced

  let uniform_filter ~kernel_size ?stride x =
    let k_rank = Array.length kernel_size in
    let stride = match stride with Some s -> s | None -> kernel_size in
    let ones = Array.make k_rank 1 in
    let zeros = Array.make k_rank (0, 0) in
    let x_unf = B.unfold x ~kernel_size ~stride ~dilation:ones ~padding:zeros in
    let x_unf_ndim = ndim x_unf in
    let reduced = mean x_unf ~axes:[ x_unf_ndim - 2 ] in
    let x_rank = ndim x in
    let leading_shape = Array.sub (shape x) 0 (x_rank - k_rank) in
    let input_spatial = Array.sub (shape x) (x_rank - k_rank) k_rank in
    let out_spatial =
      Array.init k_rank (fun i ->
          ((input_spatial.(i) - kernel_size.(i)) / stride.(i)) + 1)
    in
    reshape (Array.concat [ leading_shape; out_spatial ]) reduced

  let calculate_padding_for_mode input_spatial_shape ~k_s ~s_s ~d_s
      ~(mode : [< `Full | `Valid | `Same ])
      ~(op_type : [ `Convolution | `Correlation ]) =
    let num_spatial = Array.length input_spatial_shape in
    if
      not
        (Array.length k_s = num_spatial
        && Array.length s_s = num_spatial
        && Array.length d_s = num_spatial)
    then
      Error.invalid ~op:"calculate_padding_for_mode" ~what:"array lengths"
        ~reason:"shape/kernel/stride/dilation must have same length" ();

    match mode with
    | `Valid -> Array.make num_spatial (0, 0)
    | `Full ->
        Array.init num_spatial (fun i ->
            let pad_each_side = d_s.(i) * (k_s.(i) - 1) in
            (pad_each_side, pad_each_side))
    | `Same ->
        Array.init num_spatial (fun i ->
            let is_d, ss_d, ks_d, ds_d =
              (input_spatial_shape.(i), s_s.(i), k_s.(i), d_s.(i))
            in
            let os_d = ceildiv is_d ss_d in
            let eff_ks_d = (ds_d * (ks_d - 1)) + 1 in
            let total_pad_d =
              Stdlib.max 0 (((os_d - 1) * ss_d) + eff_ks_d - is_d)
            in
            (* For even kernels with odd total padding, convolution and
               correlation pad differently to match NumPy/SciPy behavior *)
            let pad_before, pad_after =
              if
                ks_d mod 2 = 0
                && total_pad_d mod 2 = 1
                && op_type = `Convolution
              then
                (* Convolution: pad more on top/left (before) *)
                ((total_pad_d / 2) + 1, total_pad_d / 2)
              else
                (* Correlation: pad more on bottom/right (after) - default
                   behavior *)
                (total_pad_d / 2, total_pad_d - (total_pad_d / 2))
            in
            (pad_before, pad_after))

  let correlate_nd_general ~groups stride_s_arr ~padding_mode dilation_s_arr
      ?fillvalue:_ num_spatial_dims ?bias ~op_type x w =
    if ndim w <> num_spatial_dims + 2 then
      Error.invalid ~op:"correlate_nd" ~what:"weight tensor"
        ~reason:(Printf.sprintf "must be %dD" (num_spatial_dims + 2))
        ();
    if ndim x <> num_spatial_dims + 2 then
      Error.invalid ~op:"correlate_nd" ~what:"input tensor"
        ~reason:(Printf.sprintf "must be %dD" (num_spatial_dims + 2))
        ();
    if Array.length stride_s_arr <> num_spatial_dims then
      Error.invalid ~op:"correlate_nd" ~what:"stride_s_arr length"
        ~reason:"mismatch with num_spatial_dims" ();
    if Array.length dilation_s_arr <> num_spatial_dims then
      Error.invalid ~op:"correlate_nd" ~what:"dilation_s_arr length"
        ~reason:"mismatch with num_spatial_dims" ();

    let bs = dim 0 x in
    let cin_total = dim 1 x in
    let input_spatial_shape_arr =
      Array.init num_spatial_dims (fun i -> dim (i + 2) x)
    in

    let cout = dim 0 w in
    let cin_per_group = dim 1 w in
    let kernel_spatial_shape_arr =
      Array.init num_spatial_dims (fun i -> dim (i + 2) w)
    in

    (* Handle empty input - if any spatial dimension is 0, return empty
       output *)
    if Array.exists (fun d -> d = 0) input_spatial_shape_arr then
      let empty_spatial = Array.make num_spatial_dims 0 in
      let empty_shape = Array.concat [ [| bs; cout |]; empty_spatial ] in
      empty (B.context x) (dtype x) empty_shape
    else (
      if cin_total <> groups * cin_per_group then
        Error.invalid ~op:"correlate_nd"
          ~what:(Printf.sprintf "channel configuration")
          ~reason:(Printf.sprintf "%d ≠ %d×%d" cin_total groups cin_per_group)
          ~hint:
            (Printf.sprintf
               "expected %d channels for %d groups with %d channels each"
               (groups * cin_per_group) groups cin_per_group)
          ();
      let rcout = cout / groups in
      if groups * rcout <> cout then
        Error.invalid ~op:"correlate_nd"
          ~what:(Printf.sprintf "cout %d" cout)
          ~reason:(Printf.sprintf "%d %% %d ≠ 0" cout groups)
          ~hint:
            (Printf.sprintf
               "expected %d channels for %d groups with %d channels each" cout
               groups rcout)
          ();

      let padding_config_pairs_arr =
        calculate_padding_for_mode input_spatial_shape_arr
          ~k_s:kernel_spatial_shape_arr ~s_s:stride_s_arr ~d_s:dilation_s_arr
          ~mode:padding_mode ~op_type
      in

      (* Use unfold (im2col) for convolution *)
      let kernel_elements = Array.fold_left ( * ) 1 kernel_spatial_shape_arr in
      let x_unf =
        B.unfold x ~kernel_size:kernel_spatial_shape_arr ~stride:stride_s_arr
          ~dilation:dilation_s_arr ~padding:padding_config_pairs_arr
      in
      (* x_unf shape: [bs, cin_total, kernel_elements, num_blocks] *)
      let x_unf_shape = shape x_unf in
      let num_blocks = x_unf_shape.(Array.length x_unf_shape - 1) in
      (* Merge channels and kernel: [bs, cin_total * kernel_elements,
         num_blocks] *)
      let x_col =
        reshape [| bs; cin_total * kernel_elements; num_blocks |] x_unf
      in

      (* Calculate output spatial shape based on actual num_blocks *)
      let output_spatial_shape_arr =
        if num_spatial_dims = 1 then [| num_blocks |]
        else if num_spatial_dims = 2 then
          (* For 2D, we need to figure out height and width from num_blocks *)
          let padded_h =
            input_spatial_shape_arr.(0)
            + fst padding_config_pairs_arr.(0)
            + snd padding_config_pairs_arr.(0)
          in
          let padded_w =
            input_spatial_shape_arr.(1)
            + fst padding_config_pairs_arr.(1)
            + snd padding_config_pairs_arr.(1)
          in
          let effective_kh =
            ((kernel_spatial_shape_arr.(0) - 1) * dilation_s_arr.(0)) + 1
          in
          let effective_kw =
            ((kernel_spatial_shape_arr.(1) - 1) * dilation_s_arr.(1)) + 1
          in
          let out_h = ((padded_h - effective_kh) / stride_s_arr.(0)) + 1 in
          let out_w = ((padded_w - effective_kw) / stride_s_arr.(1)) + 1 in
          [| out_h; out_w |]
        else
          (* For higher dimensions, calculate normally *)
          let padded_spatial =
            Array.init num_spatial_dims (fun i ->
                input_spatial_shape_arr.(i)
                + fst padding_config_pairs_arr.(i)
                + snd padding_config_pairs_arr.(i))
          in
          Array.init num_spatial_dims (fun i ->
              let effective_kernel =
                ((kernel_spatial_shape_arr.(i) - 1) * dilation_s_arr.(i)) + 1
              in
              ((padded_spatial.(i) - effective_kernel) / stride_s_arr.(i)) + 1)
      in

      let result =
        if groups = 1 then
          (* Standard convolution: use direct matmul *)
          (* x_col: [bs, cin * kernel_elements, num_blocks] *)
          (* w: [cout, cin, *kernel] *)

          (* Reshape weights to [cout, cin * kernel_elements] *)
          let w_cont = B.contiguous w in
          let w_reshaped =
            reshape [| cout; cin_total * kernel_elements |] w_cont
          in

          (* Use matmul: w_reshaped @ x_col = [cout, cin*k] @ [bs, cin*k, num_blocks] -> [bs, cout, num_blocks] *)
          (* matmul handles broadcasting automatically for 2D × 3D case *)
          matmul w_reshaped x_col
        else
          (* Grouped convolution *)
          (* x_col: [bs, cin_total * kernel_elements, num_blocks] *)

          (* Reshape input to separate groups *)
          let x_col_grouped =
            reshape
              [| bs; groups; cin_per_group * kernel_elements; num_blocks |]
              x_col
          in

          (* Reshape weights to [groups, rcout, cin_per_group *
             kernel_elements] *)
          let w_cont = B.contiguous w in
          let w_grouped =
            reshape [| groups; rcout; cin_per_group * kernel_elements |] w_cont
          in

          (* Process groups using batched matmul *)
          (* Combine batch and group dimensions *)
          let x_col_batched =
            reshape
              [| bs * groups; cin_per_group * kernel_elements; num_blocks |]
              x_col_grouped
          in
          let w_batched =
            reshape
              [| groups; rcout; cin_per_group * kernel_elements |]
              w_grouped
          in

          (* Expand weights to match batch*groups dimension *)
          let w_expanded = unsqueeze ~axes:[ 0 ] w_batched in
          let w_expanded =
            expand
              [| bs; groups; rcout; cin_per_group * kernel_elements |]
              w_expanded
          in
          let w_expanded =
            reshape
              [| bs * groups; rcout; cin_per_group * kernel_elements |]
              w_expanded
          in

          (* Matmul: [bs*groups, rcout, cin_per_group*k] @ [bs*groups,
             cin_per_group*k, num_blocks] *)
          let result_batched = matmul w_expanded x_col_batched in

          (* Reshape back to separate batch and groups *)
          let result_grouped =
            reshape [| bs; groups; rcout; num_blocks |] result_batched
          in

          (* Merge groups and rcout to get final output channels *)
          reshape [| bs; cout; num_blocks |] result_grouped
      in

      (* Reshape result from [bs, cout, num_blocks] to [bs, cout,
         output_spatial] *)
      let final_shape =
        Array.concat [ [| bs; cout |]; output_spatial_shape_arr ]
      in
      let result_reshaped = reshape final_shape result in

      (* The reshape already produces the correct layout *)
      let result_corrected = result_reshaped in

      match bias with
      | None -> result_corrected
      | Some b ->
          let bias_shape =
            Array.concat [ [| 1; cout |]; Array.make num_spatial_dims 1 ]
          in
          let bias_reshaped = reshape bias_shape b in
          add result_corrected bias_reshaped)
  (* Close the begin block for non-empty input case *)

  let correlate_nd ?(groups = 1) stride_s_arr
      ?(padding_mode : [ `Full | `Valid | `Same ] = `Valid) dilation_s_arr
      ?fillvalue num_spatial_dims ?bias x w =
    correlate_nd_general ~groups stride_s_arr ~padding_mode dilation_s_arr
      ?fillvalue num_spatial_dims ?bias ~op_type:`Correlation x w

  let correlate1d ?groups ?(stride = 1) ?padding_mode ?(dilation = 1) ?fillvalue
      ?bias x w =
    correlate_nd ?groups [| stride |] ?padding_mode [| dilation |] ?fillvalue 1
      ?bias x w

  let correlate2d ?groups ?(stride = (1, 1)) ?padding_mode ?(dilation = (1, 1))
      ?fillvalue ?bias x w =
    correlate_nd ?groups (pair_to_array stride) ?padding_mode
      (pair_to_array dilation) ?fillvalue 2 ?bias x w

  (* Flips the kernel (weights) along all its spatial dimensions then calls
     correlate_nd. *)
  let convolve_nd ?groups stride_s_arr ?padding_mode dilation_s_arr ?fillvalue
      num_spatial_dims ?bias x w =
    let w_ndim = ndim w in
    if w_ndim < num_spatial_dims + 2 then
      Error.invalid ~op:"convolve_nd" ~what:"weight tensor"
        ~reason:
          (Printf.sprintf "needs at least %d dims for spatial flipping"
             (num_spatial_dims + 2))
        ();

    (* Flip all spatial dimensions of w: dims from 2 up to (2 + num_spatial_dims
       - 1) *)
    let flip_axes_bools = Array.make w_ndim false in
    for i = 0 to num_spatial_dims - 1 do
      flip_axes_bools.(2 + i) <- true
    done;

    let w_flipped = B.flip w flip_axes_bools in
    (* Call correlate_nd_general directly with Convolution op_type *)
    let groups = Option.value groups ~default:1 in
    let padding_mode = Option.value padding_mode ~default:`Valid in
    correlate_nd_general ~groups stride_s_arr ~padding_mode dilation_s_arr
      ?fillvalue num_spatial_dims ?bias ~op_type:`Convolution x w_flipped

  let convolve1d ?groups ?(stride = 1) ?padding_mode ?(dilation = 1) ?fillvalue
      ?bias x w =
    convolve_nd ?groups [| stride |] ?padding_mode [| dilation |] ?fillvalue 1
      ?bias x w

  let convolve2d ?groups ?(stride = (1, 1)) ?padding_mode ?(dilation = (1, 1))
      ?fillvalue ?bias x w =
    convolve_nd ?groups (pair_to_array stride) ?padding_mode
      (pair_to_array dilation) ?fillvalue 2 ?bias x w

  let resolve_padding_for_ops padding_spec ~input_spatial_shape ~k_s ~s_s ~d_s
      ~op_type =
    match padding_spec with
    | `Same | `Valid | `Full ->
        calculate_padding_for_mode input_spatial_shape ~k_s ~s_s ~d_s
          ~mode:padding_spec ~op_type

  (* Adjust padding for ceil_mode=true. *)
  let apply_ceil_mode ~current_pads_pairs ~input_spatial_shape ~k_s ~s_s ~d_s =
    let num_spatial_dims = Array.length k_s in
    let pads_adj = Array.copy current_pads_pairs in
    let o_s =
      Array.init num_spatial_dims (fun i ->
          let i_d = input_spatial_shape.(i) in
          let d_d = d_s.(i) in
          let k_d = k_s.(i) in
          let s_d = s_s.(i) in
          let p_b, p_a = current_pads_pairs.(i) in
          ceildiv (i_d + p_b + p_a - ((d_d * (k_d - 1)) + 1)) s_d + 1)
    in
    for i = 0 to num_spatial_dims - 1 do
      let o_d, i_d, s_d, k_d, d_d =
        (o_s.(i), input_spatial_shape.(i), s_s.(i), k_s.(i), d_s.(i))
      in
      let p_b, p_a = current_pads_pairs.(i) in
      let pad_needed_for_last_window_start =
        (s_d * (o_d - 1)) + ((d_d * (k_d - 1)) + 1) - (i_d + p_b + p_a)
      in
      let effective_pad_before_input_start =
        Stdlib.max 0 ((s_d * (o_d - 1)) - (p_b + i_d - 1))
      in
      (* Adjust pad_after (pads_adj.(i) |> snd) *)
      pads_adj.(i) <-
        ( fst pads_adj.(i),
          snd pads_adj.(i)
          + pad_needed_for_last_window_start - effective_pad_before_input_start
        )
    done;
    pads_adj

  let pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation ~padding_spec
      ~ceil_mode x =
    let x_ndim = ndim x in
    let input_spatial_shape =
      Array.sub (shape x) (x_ndim - num_spatial_dims) num_spatial_dims
    in
    let s_s = Option.value stride ~default:kernel_size in
    let d_s = Option.value dilation ~default:(Array.make num_spatial_dims 1) in

    let reg_pads =
      resolve_padding_for_ops padding_spec ~input_spatial_shape ~k_s:kernel_size
        ~s_s ~d_s ~op_type:`Convolution
    in
    let pads =
      if ceil_mode then
        apply_ceil_mode ~current_pads_pairs:reg_pads ~input_spatial_shape
          ~k_s:kernel_size ~s_s ~d_s
      else reg_pads
    in
    let full_pad_config =
      Array.concat [ Array.make (x_ndim - num_spatial_dims) (0, 0); pads ]
    in

    (input_spatial_shape, s_s, d_s, pads, reg_pads, full_pad_config)

  let avg_pool_nd ~kernel_size ?stride ?dilation ~padding_spec ~ceil_mode
      ~count_include_pad ~num_spatial_dims x =
    let x_ndim = ndim x in

    (* Use pool_setup helper *)
    let ( _input_spatial_shape,
          s_s,
          d_s,
          _current_pads_pairs,
          _reg_pads_pairs,
          full_pad_config ) =
      pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation ~padding_spec
        ~ceil_mode x
    in

    (* Use unfold for pooling *)
    let padding_pairs =
      Array.sub full_pad_config (x_ndim - num_spatial_dims) num_spatial_dims
    in
    let x_unfolded =
      B.unfold x ~kernel_size ~stride:s_s ~dilation:d_s ~padding:padding_pairs
    in
    (* x_unfolded shape: [*leading, kernel_elements, num_blocks] *)

    (* Calculate the output shape *)
    let prefix_shape = Array.sub (shape x) 0 (x_ndim - num_spatial_dims) in
    let padded_spatial =
      Array.init num_spatial_dims (fun i ->
          (shape x).(x_ndim - num_spatial_dims + i)
          + fst padding_pairs.(i)
          + snd padding_pairs.(i))
    in
    let output_spatial =
      Array.init num_spatial_dims (fun i ->
          let effective_kernel = ((kernel_size.(i) - 1) * d_s.(i)) + 1 in
          ((padded_spatial.(i) - effective_kernel) / s_s.(i)) + 1)
    in

    let kernel_elements = array_prod kernel_size in

    (* Sum over kernel elements axis (second-to-last) *)
    let x_unf_ndim = Array.length (shape x_unfolded) in
    let sum_pooled = sum x_unfolded ~axes:[ x_unf_ndim - 2 ] in

    (* Reshape back to original layout *)
    let result_shape = Array.concat [ prefix_shape; output_spatial ] in
    let sum_corrected = reshape result_shape sum_pooled in

    (* Compute divisor based on mode *)
    if count_include_pad && not ceil_mode then
      (* Simple case: divide by kernel size *)
      let kernel_numel = float_of_int kernel_elements in
      div_s sum_corrected kernel_numel
    else
      (* Need to count valid elements - use same unfold approach on ones *)
      let ones = ones_like x in
      let ones_unfolded =
        B.unfold ones ~kernel_size ~stride:s_s ~dilation:d_s
          ~padding:padding_pairs
      in
      (* ones_unfolded: [*leading, kernel_elements, num_blocks] *)
      let count =
        sum ones_unfolded ~axes:[ Array.length (shape ones_unfolded) - 2 ]
      in
      let count_reshaped = reshape result_shape count in
      let count_corrected =
        if num_spatial_dims = 2 then
          transpose
            ~axes:
              (Array.to_list
                 (Array.init x_ndim (fun i ->
                      if i = x_ndim - 2 then x_ndim - 1
                      else if i = x_ndim - 1 then x_ndim - 2
                      else i)))
            count_reshaped
        else count_reshaped
      in
      div sum_corrected count_corrected

  let max_pool_nd ~kernel_size ?stride ?dilation ~padding_spec ~ceil_mode
      ~return_indices ~num_spatial_dims x =
    let x_ndim = ndim x in

    (* Check for empty spatial dimensions *)
    let input_spatial_shape =
      Array.sub (shape x) (x_ndim - num_spatial_dims) num_spatial_dims
    in
    if Array.exists (fun d -> d = 0) input_spatial_shape then
      (* Return empty output with proper shape *)
      let empty_output = empty (B.context x) (dtype x) (shape x) in
      if return_indices then
        let empty_indices = empty (B.context x) Dtype.int32 (shape x) in
        (empty_output, Some empty_indices)
      else (empty_output, None)
    else
      (* Use pool_setup helper *)
      let ( _input_spatial_shape,
            s_s,
            d_s,
            _current_pads_pairs,
            _,
            full_pad_config ) =
        pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation
          ~padding_spec ~ceil_mode x
      in

      (* Use unfold for pooling *)
      let padding_pairs =
        Array.sub full_pad_config (x_ndim - num_spatial_dims) num_spatial_dims
      in
      let x_unfolded =
        B.unfold x ~kernel_size ~stride:s_s ~dilation:d_s ~padding:padding_pairs
      in
      (* x_unfolded shape: [*leading, kernel_elements, num_blocks] *)

      (* Calculate the output shape *)
      let prefix_shape = Array.sub (shape x) 0 (x_ndim - num_spatial_dims) in
      let padded_spatial =
        Array.init num_spatial_dims (fun i ->
            (shape x).(x_ndim - num_spatial_dims + i)
            + fst padding_pairs.(i)
            + snd padding_pairs.(i))
      in
      let output_spatial =
        Array.init num_spatial_dims (fun i ->
            let effective_kernel = ((kernel_size.(i) - 1) * d_s.(i)) + 1 in
            ((padded_spatial.(i) - effective_kernel) / s_s.(i)) + 1)
      in

      let kernel_elements = array_prod kernel_size in

      (* Compute max over kernel elements (second-to-last axis) *)
      let x_unf_ndim = Array.length (shape x_unfolded) in
      let max_pooled =
        max x_unfolded ~axes:[ x_unf_ndim - 2 ] ~keepdims:false
      in

      (* Reshape back to original layout *)
      let result_shape = Array.concat [ prefix_shape; output_spatial ] in
      let max_values = reshape result_shape max_pooled in

      if not return_indices then (max_values, None)
      else
        (* For indices, we need to track which element in the kernel was the max *)
        (* Create indices for each kernel position *)
        let kernel_indices =
          arange (B.context x) Dtype.int32 0 kernel_elements 1
        in
        (* Broadcast kernel indices to match x_unfolded shape *)
        let ki_shape = Array.make x_unf_ndim 1 in
        ki_shape.(x_unf_ndim - 2) <- kernel_elements;
        let kernel_indices_reshaped = reshape ki_shape kernel_indices in
        let kernel_indices_broadcast =
          broadcast_to (shape x_unfolded) kernel_indices_reshaped
        in

        (* Find which kernel position has the max value *)
        let max_expanded = unsqueeze ~axes:[ x_unf_ndim - 2 ] max_pooled in
        let max_broadcast = broadcast_to (shape x_unfolded) max_expanded in
        let is_max = equal x_unfolded max_broadcast in

        (* Use where to select indices, with a large value for non-max
           positions *)
        let large_val =
          scalar (B.context x) Dtype.int32 (Int32.of_int kernel_elements)
        in
        let masked_indices =
          where is_max kernel_indices_broadcast
            (broadcast_to (shape kernel_indices_broadcast) large_val)
        in

        (* Get the minimum index (first occurrence of max) *)
        let kernel_idx =
          min masked_indices
            ~axes:[ Array.length (shape masked_indices) - 2 ]
            ~keepdims:false
        in

        let final_indices = reshape result_shape kernel_idx in
        (max_values, Some final_indices)

  let avg_pool1d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(count_include_pad = true) x =
    avg_pool_nd x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ~ceil_mode ~count_include_pad ~num_spatial_dims:1

  let avg_pool2d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(count_include_pad = true) x =
    avg_pool_nd x
      ~kernel_size:(pair_to_array kernel_size)
      ?stride:(Option.map pair_to_array stride)
      ?dilation:(Option.map pair_to_array dilation)
      ~padding_spec ~ceil_mode ~count_include_pad ~num_spatial_dims:2

  let max_pool1d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(return_indices = false) x =
    max_pool_nd x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ~ceil_mode ~return_indices ~num_spatial_dims:1

  let max_pool2d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(return_indices = false) x =
    max_pool_nd x
      ~kernel_size:(pair_to_array kernel_size)
      ?stride:(Option.map pair_to_array stride)
      ?dilation:(Option.map pair_to_array dilation)
      ~padding_spec ~ceil_mode ~return_indices ~num_spatial_dims:2

  let min_pool_nd ~kernel_size ?stride ?dilation ~padding_spec ~ceil_mode
      ~return_indices ~num_spatial_dims x =
    let x_ndim = ndim x in

    (* Check for empty spatial dimensions *)
    let input_spatial_shape =
      Array.sub (shape x) (x_ndim - num_spatial_dims) num_spatial_dims
    in
    if Array.exists (fun d -> d = 0) input_spatial_shape then
      (* Return empty output with proper shape *)
      let empty_output = empty (B.context x) (dtype x) (shape x) in
      if return_indices then
        let empty_indices = empty (B.context x) Dtype.int32 (shape x) in
        (empty_output, Some empty_indices)
      else (empty_output, None)
    else
      (* Use pool_setup helper *)
      let ( _input_spatial_shape,
            s_s,
            d_s,
            _current_pads_pairs,
            _,
            full_pad_config ) =
        pool_setup ~num_spatial_dims ~kernel_size ?stride ?dilation
          ~padding_spec ~ceil_mode x
      in

      (* Use unfold for pooling *)
      let padding_pairs =
        Array.sub full_pad_config (x_ndim - num_spatial_dims) num_spatial_dims
      in
      let x_unfolded =
        B.unfold x ~kernel_size ~stride:s_s ~dilation:d_s ~padding:padding_pairs
      in
      (* x_unfolded shape: [*leading, kernel_elements, num_blocks] *)

      (* Calculate the output shape *)
      let prefix_shape = Array.sub (shape x) 0 (x_ndim - num_spatial_dims) in
      let padded_spatial =
        Array.init num_spatial_dims (fun i ->
            (shape x).(x_ndim - num_spatial_dims + i)
            + fst padding_pairs.(i)
            + snd padding_pairs.(i))
      in
      let output_spatial =
        Array.init num_spatial_dims (fun i ->
            let effective_kernel = ((kernel_size.(i) - 1) * d_s.(i)) + 1 in
            ((padded_spatial.(i) - effective_kernel) / s_s.(i)) + 1)
      in

      (* Compute min over kernel elements (second-to-last axis) *)
      let x_unf_ndim = Array.length (shape x_unfolded) in
      let min_pooled =
        min x_unfolded ~axes:[ x_unf_ndim - 2 ] ~keepdims:false
      in

      (* Reshape back to original layout *)
      let result_shape = Array.concat [ prefix_shape; output_spatial ] in
      let min_values_corrected = reshape result_shape min_pooled in

      if not return_indices then (min_values_corrected, None)
      else (min_values_corrected, None)
  (* Index tracking not implemented for min pool *)

  let min_pool1d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(return_indices = false) x =
    min_pool_nd x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ~ceil_mode ~return_indices ~num_spatial_dims:1

  let min_pool2d ~kernel_size ?stride ?dilation ?(padding_spec = `Valid)
      ?(ceil_mode = false) ?(return_indices = false) x =
    min_pool_nd x
      ~kernel_size:(pair_to_array kernel_size)
      ?stride:(Option.map pair_to_array stride)
      ?dilation:(Option.map pair_to_array dilation)
      ~padding_spec ~ceil_mode ~return_indices ~num_spatial_dims:2

  let one_hot ~num_classes index_tensor =
    let index_dt = dtype index_tensor in
    if not (Dtype.is_int index_dt || Dtype.is_uint index_dt) then
      Error.invalid ~op:"one_hot"
        ~what:(Printf.sprintf "dtype %s" (Dtype.to_string index_dt))
        ~reason:"indices must be integer type" ();

    let index_expanded = unsqueeze index_tensor ~axes:[ ndim index_tensor ] in
    (* Add new last dim *)

    let arange_x = arange (B.context index_tensor) index_dt 0 num_classes 1 in
    (* Classes 0 to num_classes-1 *)

    (* Reshape arange to be (1, ..., 1, num_classes) to align with new last dim
       of index_expanded *)
    let ndim_expanded = ndim index_expanded in
    let shape_for_arange = Array.make ndim_expanded 1 in
    shape_for_arange.(ndim_expanded - 1) <- num_classes;
    let arange_b = reshape shape_for_arange arange_x in

    (* Broadcasts to one-hot mask *)
    let bool_to_uint (x : (bool, bool_elt) t) : (int, uint8_elt) t =
      cast Dtype.uint8 x
    in
    let bool_tensor = cmpeq index_expanded arange_b in
    bool_to_uint bool_tensor

  let max_unpool_nd ~kernel_size ?stride ?dilation ~padding_spec
      ?output_size_opt ~num_spatial_dims input_x indices_x =
    let bs = dim 0 input_x in
    let c = dim 1 input_x in
    let pooled_spatial_shape = Array.sub (shape input_x) 2 num_spatial_dims in

    let output_spatial_shape =
      match output_size_opt with
      | Some os_arr -> os_arr
      | None ->
          let s_s = Option.value stride ~default:kernel_size in
          let d_s =
            Option.value dilation ~default:(Array.make num_spatial_dims 1)
          in
          let pads_pairs =
            resolve_padding_for_ops padding_spec
              ~input_spatial_shape:pooled_spatial_shape
                (* Placeholder, see note in thought process *)
              ~k_s:kernel_size ~s_s ~d_s ~op_type:`Convolution
          in
          Array.init num_spatial_dims (fun i ->
              let pooled_dim_size = pooled_spatial_shape.(i) in
              let k = kernel_size.(i) in
              let s = s_s.(i) in
              let d = d_s.(i) in
              let pb, pa = pads_pairs.(i) in
              ((pooled_dim_size - 1) * s) - pb - pa + ((d * (k - 1)) + 1))
    in
    let prod_output_spatial_size = array_prod output_spatial_shape in

    let one_hot_mask_for_indices =
      one_hot indices_x ~num_classes:prod_output_spatial_size
    in

    let input_expanded = unsqueeze input_x ~axes:[ ndim input_x ] in

    let multiplied = mul one_hot_mask_for_indices input_expanded in

    let sum_axes = Array.init num_spatial_dims (fun i -> 2 + i) in
    let result_flat_spatial =
      sum multiplied ~axes:(Array.to_list sum_axes) ~keepdims:false
    in

    let final_shape = Array.concat [ [| bs; c |]; output_spatial_shape ] in
    reshape final_shape result_flat_spatial

  let max_unpool1d input_x indices_x ~kernel_size ?stride ?dilation
      ?(padding_spec = `Valid) ?output_size_opt () =
    max_unpool_nd input_x indices_x ~kernel_size:[| kernel_size |]
      ?stride:(Option.map (fun s -> [| s |]) stride)
      ?dilation:(Option.map (fun d -> [| d |]) dilation)
      ~padding_spec ?output_size_opt ~num_spatial_dims:1

  let max_unpool2d input_x indices_x ~kernel_size ?stride ?dilation
      ?(padding_spec = `Valid) ?output_size_opt () =
    max_unpool_nd input_x indices_x
      ~kernel_size:(pair_to_array kernel_size)
      ?stride:(Option.map pair_to_array stride)
      ?dilation:(Option.map pair_to_array dilation)
      ~padding_spec ?output_size_opt ~num_spatial_dims:2

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
          Error.failed ~op:"pp_data"
            ~what:"cannot print tensor with symbolic shape" ()
    in
    let ndim = Array.length shape in
    let sz =
      match Symbolic_shape.eval_dim (View.numel view) with
      | Some n -> n
      | None ->
          Error.failed ~op:"pp_data"
            ~what:"cannot print tensor with symbolic size" ()
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
        let value = Nx_buffer.unsafe_get buffer (View.offset view) in
        pp_element fmt value
      else fprintf fmt "<empty scalar>"
    else
      let rec pp_slice fmt current_indices =
        let current_ndim = List.length current_indices in
        if current_ndim = ndim then
          let md_index = Array.of_list current_indices in
          let linear_offset =
            let strides =
              match View.strides_opt view with
              | Some s -> s
              | None ->
                  Error.failed ~op:"pp_data"
                    ~what:"cannot print non-contiguous symbolic tensor" ()
            in
            let offset = View.offset view in
            Shape.ravel_index md_index strides + offset
          in
          if linear_offset < 0 || linear_offset >= Nx_buffer.length buffer then
            fprintf fmt "<OOB:%d/%d>" linear_offset (Nx_buffer.length buffer)
          else
            let value = Nx_buffer.unsafe_get buffer linear_offset in
            pp_element fmt value
        else
          let axis = current_ndim in
          let dim_size = shape.(axis) in
          fprintf fmt "[";
          if dim_size > 0 then (
            if axis < ndim - 1 then pp_open_vbox fmt 0 else pp_open_hbox fmt ();
            for i = 0 to dim_size - 1 do
              if i > 0 then (
                fprintf fmt ",";
                if axis = ndim - 1 then fprintf fmt " " else pp_print_cut fmt ());
              pp_slice fmt (current_indices @ [ i ])
            done;
            pp_close_box fmt ());
          fprintf fmt "]"
      in
      if sz > 0 then pp_slice fmt [] else fprintf fmt "[]"

  (* Helper for formatter-based string conversion *)
  let format_to_string pp x =
    let buf = Stdlib.Buffer.create 1024 in
    let fmt = Format.formatter_of_buffer buf in
    pp fmt x;
    Format.pp_print_flush fmt ();
    Stdlib.Buffer.contents buf

  (* Helper for printing to stdout *)
  let print_with_formatter pp x =
    pp Format.std_formatter x;
    Format.pp_print_newline Format.std_formatter ();
    Format.pp_print_flush Format.std_formatter ()

  let data_to_string x = format_to_string pp_data x
  let print_data x = print_with_formatter pp_data x
  let pp_dtype fmt dtype = Format.fprintf fmt "%s" (Dtype.to_string dtype)
  let dtype_to_string dtype = Dtype.to_string dtype

  let shape_to_string shape =
    let shape_str =
      Array.map string_of_int shape |> Array.to_list |> String.concat "x"
    in
    Printf.sprintf "[%s]" shape_str

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
          "["
          ^ String.concat "; " (Array.to_list (Array.map string_of_int s))
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

  (* ───── Higher-order functions ───── *)

  (* Map a function over all elements of a tensor *)
  let map_item f x =
    let dt = dtype x in
    let sh = shape x in
    let result = empty (B.context x) dt sh in
    let data_src = data (contiguous x) in
    let data_dst = data result in
    let sz = size x in
    for i = 0 to sz - 1 do
      let v = Nx_buffer.unsafe_get data_src i in
      let v' = f v in
      Nx_buffer.unsafe_set data_dst i v'
    done;
    result

  (* Iterate a function over all elements of a tensor for side effects *)
  let iter_item f x =
    let data_src = data (contiguous x) in
    let sz = size x in
    for i = 0 to sz - 1 do
      let v = Nx_buffer.unsafe_get data_src i in
      f v
    done

  (* Fold a function over all elements of a tensor *)
  let fold_item f init x =
    let data_src = data (contiguous x) in
    let sz = size x in
    let acc = ref init in
    for i = 0 to sz - 1 do
      let v = Nx_buffer.unsafe_get data_src i in
      acc := f !acc v
    done;
    !acc

  (* Safe versions using backend operations - JAX semantics *)

  let map f x =
    let dt = dtype x in
    let sh = shape x in
    let result = empty (B.context x) dt sh in

    (* Process each element *)
    let total_size = size x in
    for i = 0 to total_size - 1 do
      let idx = Shape.unravel_index i sh |> Array.to_list in
      let v = get idx x in
      let v' = f v in
      set idx result v'
    done;
    result

  let iter f x =
    let sh = shape x in

    (* Process each element *)
    let total_size = size x in
    for i = 0 to total_size - 1 do
      let idx = Shape.unravel_index i sh |> Array.to_list in
      let v = get idx x in
      f v
    done

  let fold f init x =
    let sh = shape x in

    (* Process each element *)
    let total_size = size x in
    let acc = ref init in
    for i = 0 to total_size - 1 do
      let idx = Shape.unravel_index i sh |> Array.to_list in
      let v = get idx x in
      acc := f !acc v
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
