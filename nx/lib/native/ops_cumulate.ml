open Bigarray
open Nx_core
open Internal

let get_type_op (type a b) (dtype : (a, b) Dtype.t)
    (op : [ `Sum | `Prod | `Min | `Max ]) : a -> a -> a =
  match (op, dtype) with
  | `Sum, _ -> Dtype.add dtype
  | `Prod, _ -> Dtype.mul dtype
  | `Min, (Complex16 | Complex32 | Complex64) ->
      failwith "cummin: datatype not supported"
  | `Min, Float16 -> Float.min
  | `Min, Float32 -> Float.min
  | `Min, Float64 -> Float.min
  | `Min, Int8 -> Int.min
  | `Min, Int16 -> Int.min
  | `Min, Int32 -> Int32.min
  | `Min, Int64 -> Int64.min
  | `Min, UInt8 -> Int.min
  | `Min, UInt16 -> Int.min
  | `Min, Int -> Int.min
  | `Min, NativeInt -> Nativeint.min
  | `Min, BFloat16 -> Float.min
  | `Min, Bool -> ( || )
  | `Min, Int4 -> Int.min
  | `Min, UInt4 -> Int.min
  | `Min, Float8_e4m3 -> Float.min
  | `Min, Float8_e5m2 -> Float.min
  | `Min, QInt8 -> Int.min
  | `Min, QUInt8 -> Int.min
  | `Max, (Complex16 | Complex32 | Complex64) ->
      failwith "cummax: datatype not supported"
  | `Max, Float16 -> Float.max
  | `Max, Float32 -> Float.max
  | `Max, Float64 -> Float.max
  | `Max, Int8 -> Int.max
  | `Max, Int16 -> Int.max
  | `Max, Int32 -> Int32.max
  | `Max, Int64 -> Int64.max
  | `Max, UInt8 -> Int.max
  | `Max, UInt16 -> Int.max
  | `Max, Int -> Int.max
  | `Max, NativeInt -> Nativeint.max
  | `Max, BFloat16 -> Float.max
  | `Max, Bool -> ( || )
  | `Max, Int4 -> Int.max
  | `Max, UInt4 -> Int.max
  | `Max, Float8_e4m3 -> Float.max
  | `Max, Float8_e5m2 -> Float.max
  | `Max, QInt8 -> Int.max
  | `Max, QUInt8 -> Int.max

let cumulate (type a b) ~axis ~op (input : (a, b) t) (out : (a, b) t) =
  let size = size input in
  let dtype = dtype input in
  let shape = shape input in
  let strides = strides input in
  let view = view input in
  let dims_n = Array.length shape in
  let initial_value =
    match op with
    | `Sum -> Dtype.zero dtype
    | `Prod -> Dtype.one dtype
    | `Min -> Dtype.max_value dtype
    | `Max -> Dtype.min_value dtype
  in
  let view_off =
    match Lazy_view.offset view with Const v -> v | _ -> assert false
  in

  let in_buffer = buffer input in
  let out_buffer = buffer out in

  let md_index = Array.make dims_n 0 in
  let axis_size = shape.(axis) in
  let axis_stride = strides.(axis) in

  let operation = get_type_op dtype op in

  for i = 0 to (size / axis_size) - 1 do
    let temp_iter = ref i in
    for d = Array.length shape - 1 downto 0 do
      if d <> axis then (
        md_index.(d) <- !temp_iter mod shape.(d);
        temp_iter := !temp_iter / shape.(d))
      else md_index.(d) <- 0
    done;
    let base_off =
      view_off
      + Array.fold_left ( + ) 0
          (Array.mapi (fun d idx -> idx * strides.(d)) md_index)
    in

    let acc = ref initial_value in
    for j = 0 to axis_size - 1 do
      let off = base_off + (j * axis_stride) in
      let in_val = Array1.unsafe_get in_buffer off in
      let out_val = operation !acc in_val in
      acc := out_val;
      Array1.unsafe_set out_buffer off out_val
    done
  done;

  out
