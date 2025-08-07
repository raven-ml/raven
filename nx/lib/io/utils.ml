open Bigarray_ext

type packed_nx = P : ('a, 'b) Nx.t -> packed_nx

let fail_msg fmt =
  Printf.ksprintf
    (fun s ->
      let msg = Format.asprintf "Error: %s\n%!" s in
      failwith msg)
    fmt

let kind_to_string : type a b. (a, b) Bigarray_ext.kind -> string = function
  | Bigarray_ext.Int8_unsigned -> "uint8"
  | Bigarray_ext.Int8_signed -> "int8"
  | Bigarray_ext.Int16_unsigned -> "uint16"
  | Bigarray_ext.Int16_signed -> "int16"
  | Bigarray_ext.Int32 -> "int32"
  | Bigarray_ext.Int64 -> "int64"
  | Bigarray_ext.Float16 -> "float16"
  | Bigarray_ext.Float32 -> "float32"
  | Bigarray_ext.Float64 -> "float64"
  | Bigarray_ext.Complex32 -> "complex32"
  | Bigarray_ext.Complex64 -> "complex64"
  | Bigarray_ext.Int -> "int"
  | Bigarray_ext.Nativeint -> "nativeint"
  | Bigarray_ext.Char -> "char"
  | Bigarray_ext.Bfloat16 -> "bfloat16"
  | Bigarray_ext.Bool -> "bool"
  | Bigarray_ext.Int4_signed -> "int4"
  | Bigarray_ext.Int4_unsigned -> "uint4"
  | Bigarray_ext.Float8_e4m3 -> "float8_e4m3"
  | Bigarray_ext.Float8_e5m2 -> "float8_e5m2"
  | Bigarray_ext.Complex16 -> "complex16"
  | Bigarray_ext.Qint8 -> "qint8"
  | Bigarray_ext.Quint8 -> "quint8"

let dtype_of_ba_kind : type a b. (a, b) Bigarray_ext.kind -> (a, b) Nx.dtype =
 fun kind ->
  match kind with
  | Bigarray_ext.Int8_unsigned -> Nx.UInt8
  | Bigarray_ext.Int8_signed -> Nx.Int8
  | Bigarray_ext.Int16_unsigned -> Nx.UInt16
  | Bigarray_ext.Int16_signed -> Nx.Int16
  | Bigarray_ext.Int32 -> Nx.Int32
  | Bigarray_ext.Int64 -> Nx.Int64
  | Bigarray_ext.Float16 -> Nx.Float16
  | Bigarray_ext.Float32 -> Nx.Float32
  | Bigarray_ext.Float64 -> Nx.Float64
  | Bigarray_ext.Complex32 -> Nx.Complex32
  | Bigarray_ext.Complex64 -> Nx.Complex64
  | Bigarray_ext.Bfloat16 -> Nx.BFloat16
  | Bigarray_ext.Bool -> Nx.Bool
  | Bigarray_ext.Int4_signed -> Nx.Int4
  | Bigarray_ext.Int4_unsigned -> Nx.UInt4
  | Bigarray_ext.Float8_e4m3 -> Nx.Float8_e4m3
  | Bigarray_ext.Float8_e5m2 -> Nx.Float8_e5m2
  | Bigarray_ext.Complex16 -> Nx.Complex16
  | Bigarray_ext.Qint8 -> Nx.QInt8
  | Bigarray_ext.Quint8 -> Nx.QUInt8
  | _ ->
      let kind_str = kind_to_string kind in
      fail_msg "Unsupported NPY dtype for saving: %s" kind_str

let ba_kind_of_dtype : type a b. (a, b) Nx.dtype -> (a, b) Bigarray_ext.kind =
 fun dtype ->
  match dtype with
  | Nx.UInt8 -> Bigarray_ext.Int8_unsigned
  | Nx.Int8 -> Bigarray_ext.Int8_signed
  | Nx.UInt16 -> Bigarray_ext.Int16_unsigned
  | Nx.Int16 -> Bigarray_ext.Int16_signed
  | Nx.Int32 -> Bigarray_ext.Int32
  | Nx.Int64 -> Bigarray_ext.Int64
  | Nx.Float16 -> Bigarray_ext.Float16
  | Nx.Float32 -> Bigarray_ext.Float32
  | Nx.Float64 -> Bigarray_ext.Float64
  | Nx.Complex32 -> Bigarray_ext.Complex32
  | Nx.Complex64 -> Bigarray_ext.Complex64
  | Nx.Int -> Bigarray_ext.Int
  | Nx.NativeInt -> Bigarray_ext.Nativeint
  | Nx.BFloat16 -> Bigarray_ext.Bfloat16
  | Nx.Bool -> Bigarray_ext.Bool
  | Nx.Int4 -> Bigarray_ext.Int4_signed
  | Nx.UInt4 -> Bigarray_ext.Int4_unsigned
  | Nx.Float8_e4m3 -> Bigarray_ext.Float8_e4m3
  | Nx.Float8_e5m2 -> Bigarray_ext.Float8_e5m2
  | Nx.Complex16 -> Bigarray_ext.Complex16
  | Nx.QInt8 -> Bigarray_ext.Qint8
  | Nx.QUInt8 -> Bigarray_ext.Quint8

let eq_dtype : type a b c d.
    (a, b) Nx.dtype ->
    (c, d) Nx.dtype ->
    ((a, b) kind, (c, d) kind) Npy.Eq.t option =
 fun k1 k2 -> Npy.Eq.Kind.( === ) (ba_kind_of_dtype k1) (ba_kind_of_dtype k2)

let convert : type a b. string -> (a, b) Nx.dtype -> packed_nx -> (a, b) Nx.t =
 fun name target_kind packed ->
  match packed with
  | P nx -> (
      let source_kind = Nx.dtype nx in
      match eq_dtype source_kind target_kind with
      | Some Npy.Eq.W -> nx
      | None ->
          fail_msg "Type mismatch in %s: Expected %s but got %s" name
            (Nx.dtype_to_string target_kind)
            (Nx.dtype_to_string source_kind))
