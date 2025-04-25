open Internal

(* device *)

let device (type dev) (x : ('a, 'b, dev) t) : dev device =
  match x.data with Cpu_data (_, ctx) -> Cpu ctx

(* properties *)

let shape x = Dispatch.shape x.data
let dim i x = Dispatch.dim i x.data
let size x = Dispatch.size x.data
let dtype x = Dispatch.dtype x.data
let itemsize x = Dispatch.itemsize x.data
let nbytes x = Dispatch.nbytes x.data
let strides x = Dispatch.strides x.data
let stride i x = Dispatch.stride i x.data
let offset x = Dispatch.offset x.data
let ndim x = Dispatch.ndim x.data
let dims x = Dispatch.dims x.data

(* device *)

let empty_on_device (type a b c) (device : c device) (dtype : (a, b) dtype)
    shape : (a, b, c) t =
  match device with
  | Cpu ctx -> const (Cpu_data (Backend_cpu.empty ctx dtype shape, ctx))

let create_on_device (type a b c) (device : c device) (dtype : (a, b) dtype)
    shape x : (a, b, c) t =
  match device with
  | Cpu ctx -> const (Cpu_data (Backend_cpu.create ctx dtype shape x, ctx))

let move device x =
  try Effect.perform (Move (device, x))
  with Effect.Unhandled _ -> create_internal (Dispatch.move device x.data)

(* creation *)

let ndarray (type a b) (x : (a, b) Ndarray.t) =
  let context = Backend_cpu.create_context () in
  let buffer = Ndarray.data x in
  let core_dtype : (a, b) dtype =
    match Ndarray.dtype x with
    | Ndarray.Float16 -> Ndarray_core.Float16
    | Ndarray.Float32 -> Ndarray_core.Float32
    | Ndarray.Float64 -> Ndarray_core.Float64
    | Ndarray.Int8 -> Ndarray_core.Int8
    | Ndarray.Int16 -> Ndarray_core.Int16
    | Ndarray.Int32 -> Ndarray_core.Int32
    | Ndarray.Int64 -> Ndarray_core.Int64
    | Ndarray.UInt8 -> Ndarray_core.UInt8
    | Ndarray.UInt16 -> Ndarray_core.UInt16
    | Ndarray.Complex32 -> Ndarray_core.Complex32
    | Ndarray.Complex64 -> Ndarray_core.Complex64
  in
  let core_layout =
    match Ndarray.layout x with
    | Ndarray.C_contiguous -> Ndarray_core.C_contiguous
    | Ndarray.Strided -> Ndarray_core.Strided
  in
  let descriptor =
    Ndarray_core.
      {
        dtype = core_dtype;
        shape = Ndarray.shape x;
        layout = core_layout;
        strides = Ndarray.strides x;
        offset = Ndarray.offset x;
      }
  in
  let data = Backend_cpu.from_buffer context descriptor buffer in
  const (Cpu_data (data, context))

let create dtype shape x =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.create context dtype shape x, context))

let empty dtype shape =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.empty context dtype shape, context))

let empty_like x = const (Dispatch.empty_like x.data)

let zeros dtype shape =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.zeros context dtype shape, context))

let zeros_like x = const (Dispatch.zeros_like x.data)

let ones dtype shape =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.ones context dtype shape, context))

let ones_like x = const (Dispatch.ones_like x.data)

let rand dtype shape =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.rand context dtype shape, context))

let randn dtype shape =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.randn context dtype shape, context))

let scalar dtype x =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.scalar context dtype x, context))

let scalar_like (type a b dev) (x : (a, b, dev) t) (v : float) : (a, b, dev) t =
  let dtype = Dispatch.dtype x.data in
  let v_casted = cast_float dtype v in
  match x.data with
  | Cpu_data (_, ctx) ->
      const (Cpu_data (Backend_cpu.scalar ctx dtype v_casted, ctx))

(* conversion *)

let astype dtype x =
  try Effect.perform (Cast (dtype, x))
  with Effect.Unhandled _ -> create_internal (Dispatch.astype dtype x.data)

(* access *)

let get indices x = Dispatch.get_item indices x.data
let set indices value x = Dispatch.set_item indices value x.data

(* ops *)

let neg x =
  try Effect.perform (Neg x)
  with Effect.Unhandled _ -> create_internal (Dispatch.neg x.data)

let sin x =
  try Effect.perform (Sin x)
  with Effect.Unhandled _ -> create_internal (Dispatch.sin x.data)

let cos x =
  try Effect.perform (Cos x)
  with Effect.Unhandled _ -> create_internal (Dispatch.cos x.data)

let exp x =
  try Effect.perform (Exp x)
  with Effect.Unhandled _ -> create_internal (Dispatch.exp x.data)

let log x =
  try Effect.perform (Log x)
  with Effect.Unhandled _ -> create_internal (Dispatch.log x.data)

let abs x =
  try Effect.perform (Abs x)
  with Effect.Unhandled _ -> create_internal (Dispatch.abs x.data)

let add x y =
  try Effect.perform (Add (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.add x.data y.data)

let add_inplace x y =
  try Effect.perform (Add (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.add_inplace x.data y.data)

let sub x y =
  try Effect.perform (Sub (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.sub x.data y.data)

let sub_inplace x y =
  try Effect.perform (Sub (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.sub_inplace x.data y.data)

let mul x y =
  try Effect.perform (Mul (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.mul x.data y.data)

let mul_inplace x y =
  try Effect.perform (Mul (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.mul_inplace x.data y.data)

let div x y =
  try Effect.perform (Div (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.div x.data y.data)

let div_inplace x y =
  try Effect.perform (Div (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.div_inplace x.data y.data)

let maximum x y =
  try Effect.perform (Maximum (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.maximum x.data y.data)

let maximum_inplace x y =
  try Effect.perform (Maximum (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.maximum_inplace x.data y.data)

let minimum x y =
  try Effect.perform (Minimum (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.minimum x.data y.data)

let minimum_inplace x y =
  try Effect.perform (Minimum (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.minimum_inplace x.data y.data)

let sum ?axes ?keepdims x =
  try Effect.perform (Sum (x, axes, keepdims))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.sum ?axes ?keepdims x.data)

let mean ?axes ?keepdims x =
  try Effect.perform (Mean (x, axes, keepdims))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.mean ?axes ?keepdims x.data)

let max ?axes ?keepdims x =
  try Effect.perform (Max (x, axes, keepdims))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.max ?axes ?keepdims x.data)

let min ?axes ?keepdims x =
  try Effect.perform (Min (x, axes, keepdims))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.min ?axes ?keepdims x.data)

let matmul x y =
  try Effect.perform (Matmul (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.matmul x.data y.data)

let transpose x =
  try Effect.perform (Transpose x)
  with Effect.Unhandled _ -> create_internal (Dispatch.transpose x.data)

let reshape shape x =
  try Effect.perform (Reshape (x, shape))
  with Effect.Unhandled _ -> create_internal (Dispatch.reshape shape x.data)

let slice ?steps start stop x =
  try Effect.perform (Slice (x, start, stop, steps))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.slice ?steps start stop x.data)

(* *)

let select_along_axis tensor axis index =
  let ndim = Dispatch.ndim tensor.data in
  let axis = if axis < 0 then axis + ndim else axis in
  let starts = Array.make ndim 0 in
  let stops = Array.map (fun d -> d) (Dispatch.shape tensor.data) in
  starts.(axis) <- index;
  stops.(axis) <- index + 1;
  let sliced = Dispatch.slice starts stops tensor.data in
  Dispatch.squeeze ~axes:[| axis |] sliced

let stack ~axis tensors =
  let data_list = List.map (fun t -> t.data) tensors in
  create_internal (Dispatch.stack ~axis data_list)

let vmap fun_ ?in_axes ?out_axes inputs =
  let n = List.length inputs in
  let in_axes =
    match in_axes with
    | None -> List.init n (fun _ -> Some 0)
    | Some axes ->
        if List.length axes <> n then
          failwith "in_axes length mismatch with inputs";
        axes
  in
  let out_axes = match out_axes with None -> 0 | Some ax -> ax in
  (* Determine batch_size from inputs with mapped axes *)
  let batch_size_opts =
    List.mapi
      (fun i ax ->
        match ax with
        | None -> None
        | Some axis -> Some (Dispatch.dim axis (List.nth inputs i).data))
      in_axes
  in
  let batch_size =
    let opts = List.filter_map (fun x -> x) batch_size_opts in
    if opts = [] then failwith "No mapped axis to determine batch_size";
    let bs = List.hd opts in
    if List.for_all (fun x -> x = bs) opts then bs
    else failwith "Inconsistent batch sizes across mapped inputs"
  in
  (* Apply fun_ for each index along the batch axis *)
  let outputs =
    List.init batch_size (fun b ->
        let args =
          List.mapi
            (fun i input ->
              match List.nth in_axes i with
              | None -> input
              | Some axis -> create_internal (select_along_axis input axis b))
            inputs
        in
        fun_ args)
  in
  (* Stack results along out_axes *)
  stack ~axis:out_axes outputs

(* logic *)

let array_equal x y = Dispatch.array_equal x.data y.data

(* utils *)

let pp fmt x = Dispatch.pp fmt x.data
let to_string x = Dispatch.to_string x.data
let print x = Dispatch.print x.data
