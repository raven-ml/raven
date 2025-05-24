let astype : type a b c d.
    (c, d) Descriptor.dtype ->
    (a, b) Descriptor.descriptor ->
    (a, b) Buffer.buffer ->
    (c, d) Buffer.buffer =
 fun new_dtype desc buffer ->
  let size = Descriptor.size desc in
  let new_buffer = Buffer.create_buffer new_dtype size in
  let new_buffer_idx = ref 0 in

  Descriptor.iter_multi_indices desc.shape (fun md_index ->
      if !new_buffer_idx < size then (
        let input_linear_offset =
          Descriptor.md_to_linear md_index desc.strides + desc.offset
        in
        let input_value =
          Bigarray.Array1.unsafe_get buffer input_linear_offset
        in
        let output_value : c =
          match (desc.dtype, new_dtype) with
          (* Float16 Source *)
          | Float16, Float16 -> input_value
          | Float16, Float32 -> input_value
          | Float16, Float64 -> input_value
          | Float16, Int8 -> int_of_float input_value
          | Float16, UInt8 -> int_of_float input_value
          | Float16, Int16 -> int_of_float input_value
          | Float16, UInt16 -> int_of_float input_value
          | Float16, Int32 -> Int32.of_float input_value
          | Float16, Int64 -> Int64.of_float input_value
          | Float16, Complex32 -> { Complex.re = input_value; im = 0.0 }
          | Float16, Complex64 -> { Complex.re = input_value; im = 0.0 }
          (* Float32 Source *)
          | Float32, Float16 -> input_value
          | Float32, Float32 -> input_value
          | Float32, Float64 -> input_value
          | Float32, Int8 -> int_of_float input_value
          | Float32, UInt8 -> int_of_float input_value
          | Float32, Int16 -> int_of_float input_value
          | Float32, UInt16 -> int_of_float input_value
          | Float32, Int32 -> Int32.of_float input_value
          | Float32, Int64 -> Int64.of_float input_value
          | Float32, Complex32 -> { Complex.re = input_value; im = 0.0 }
          | Float32, Complex64 -> { Complex.re = input_value; im = 0.0 }
          (* Float64 Source *)
          | Float64, Float16 -> input_value
          | Float64, Float32 -> input_value
          | Float64, Float64 -> input_value
          | Float64, Int8 -> int_of_float input_value
          | Float64, UInt8 -> int_of_float input_value
          | Float64, Int16 -> int_of_float input_value
          | Float64, UInt16 -> int_of_float input_value
          | Float64, Int32 -> Int32.of_float input_value
          | Float64, Int64 -> Int64.of_float input_value
          | Float64, Complex32 -> { Complex.re = input_value; im = 0.0 }
          | Float64, Complex64 -> { Complex.re = input_value; im = 0.0 }
          (* Int8 Source *)
          | Int8, Float16 -> float_of_int input_value
          | Int8, Float32 -> float_of_int input_value
          | Int8, Float64 -> float_of_int input_value
          | Int8, Int8 -> input_value
          | Int8, UInt8 -> input_value
          | Int8, Int16 -> input_value
          | Int8, UInt16 -> input_value
          | Int8, Int32 -> Int32.of_int input_value
          | Int8, Int64 -> Int64.of_int input_value
          | Int8, Complex32 ->
              { Complex.re = float_of_int input_value; im = 0.0 }
          | Int8, Complex64 ->
              { Complex.re = float_of_int input_value; im = 0.0 }
          (* UInt8 Source *)
          | UInt8, Float16 -> float_of_int input_value
          | UInt8, Float32 -> float_of_int input_value
          | UInt8, Float64 -> float_of_int input_value
          | UInt8, Int8 -> input_value
          | UInt8, UInt8 -> input_value
          | UInt8, Int16 -> input_value
          | UInt8, UInt16 -> input_value
          | UInt8, Int32 -> Int32.of_int input_value
          | UInt8, Int64 -> Int64.of_int input_value
          | UInt8, Complex32 ->
              { Complex.re = float_of_int input_value; im = 0.0 }
          | UInt8, Complex64 ->
              { Complex.re = float_of_int input_value; im = 0.0 }
          (* Int16 Source *)
          | Int16, Float16 -> float_of_int input_value
          | Int16, Float32 -> float_of_int input_value
          | Int16, Float64 -> float_of_int input_value
          | Int16, Int8 -> input_value
          | Int16, UInt8 -> input_value
          | Int16, Int16 -> input_value
          | Int16, UInt16 -> input_value
          | Int16, Int32 -> Int32.of_int input_value
          | Int16, Int64 -> Int64.of_int input_value
          | Int16, Complex32 ->
              { Complex.re = float_of_int input_value; im = 0.0 }
          | Int16, Complex64 ->
              { Complex.re = float_of_int input_value; im = 0.0 }
          (* UInt16 Source *)
          | UInt16, Float16 -> float_of_int input_value
          | UInt16, Float32 -> float_of_int input_value
          | UInt16, Float64 -> float_of_int input_value
          | UInt16, Int8 -> input_value
          | UInt16, UInt8 -> input_value
          | UInt16, Int16 -> input_value
          | UInt16, UInt16 -> input_value
          | UInt16, Int32 -> Int32.of_int input_value
          | UInt16, Int64 -> Int64.of_int input_value
          | UInt16, Complex32 ->
              { Complex.re = float_of_int input_value; im = 0.0 }
          | UInt16, Complex64 ->
              { Complex.re = float_of_int input_value; im = 0.0 }
          (* Int32 Source *)
          | Int32, Float16 -> Int32.to_float input_value
          | Int32, Float32 -> Int32.to_float input_value
          | Int32, Float64 -> Int32.to_float input_value
          | Int32, Int8 -> Int32.to_int input_value
          | Int32, UInt8 -> Int32.to_int input_value
          | Int32, Int16 -> Int32.to_int input_value
          | Int32, UInt16 -> Int32.to_int input_value
          | Int32, Int32 -> input_value
          | Int32, Int64 -> Int64.of_int32 input_value
          | Int32, Complex32 ->
              { Complex.re = Int32.to_float input_value; im = 0.0 }
          | Int32, Complex64 ->
              { Complex.re = Int32.to_float input_value; im = 0.0 }
          (* Int64 Source *)
          | Int64, Float16 -> Int64.to_float input_value
          | Int64, Float32 -> Int64.to_float input_value
          | Int64, Float64 -> Int64.to_float input_value
          | Int64, Int8 -> Int64.to_int input_value
          | Int64, UInt8 -> Int64.to_int input_value
          | Int64, Int16 -> Int64.to_int input_value
          | Int64, UInt16 -> Int64.to_int input_value
          | Int64, Int32 -> Int64.to_int32 input_value
          | Int64, Int64 -> input_value
          | Int64, Complex32 ->
              { Complex.re = Int64.to_float input_value; im = 0.0 }
          | Int64, Complex64 ->
              { Complex.re = Int64.to_float input_value; im = 0.0 }
          (* Complex32 Source *)
          | Complex32, Float16 -> input_value.Complex.re
          | Complex32, Float32 -> input_value.Complex.re
          | Complex32, Float64 -> input_value.Complex.re
          | Complex32, Int8 -> int_of_float input_value.Complex.re
          | Complex32, UInt8 -> int_of_float input_value.Complex.re
          | Complex32, Int16 -> int_of_float input_value.Complex.re
          | Complex32, UInt16 -> int_of_float input_value.Complex.re
          | Complex32, Int32 -> Int32.of_float input_value.Complex.re
          | Complex32, Int64 -> Int64.of_float input_value.Complex.re
          | Complex32, Complex32 -> input_value
          | Complex32, Complex64 -> input_value
          (* Complex64 Source *)
          | Complex64, Float16 -> input_value.Complex.re
          | Complex64, Float32 -> input_value.Complex.re
          | Complex64, Float64 -> input_value.Complex.re
          | Complex64, Int8 -> int_of_float input_value.Complex.re
          | Complex64, UInt8 -> int_of_float input_value.Complex.re
          | Complex64, Int16 -> int_of_float input_value.Complex.re
          | Complex64, UInt16 -> int_of_float input_value.Complex.re
          | Complex64, Int32 -> Int32.of_float input_value.Complex.re
          | Complex64, Int64 -> Int64.of_float input_value.Complex.re
          | Complex64, Complex32 -> input_value
          | Complex64, Complex64 -> input_value
        in

        Bigarray.Array1.unsafe_set new_buffer !new_buffer_idx output_value;
        incr new_buffer_idx));
  if !new_buffer_idx <> size then
    Printf.eprintf "Warning: astype filled %d elements, expected %d\n"
      !new_buffer_idx size;

  new_buffer
