open Bigarray

type packed_nx = P : ('a, 'b) Nx.t -> packed_nx

let fail_msg fmt =
  Printf.ksprintf
    (fun s ->
      let msg = Format.asprintf "Error: %s\n%!" s in
      failwith msg)
    fmt

let kind_to_string : type a b. (a, b) Bigarray.kind -> string = function
  | Bigarray.Int8_unsigned -> "uint8"
  | Bigarray.Int8_signed -> "int8"
  | Bigarray.Int16_unsigned -> "uint16"
  | Bigarray.Int16_signed -> "int16"
  | Bigarray.Int32 -> "int32"
  | Bigarray.Int64 -> "int64"
  | Bigarray.Float16 -> "float16"
  | Bigarray.Float32 -> "float32"
  | Bigarray.Float64 -> "float64"
  | Bigarray.Complex32 -> "complex32"
  | Bigarray.Complex64 -> "complex64"
  | Bigarray.Int -> "int"
  | Bigarray.Nativeint -> "nativeint"
  | Bigarray.Char -> "char"

let dtype_of_ba_kind : type a b. (a, b) Bigarray.kind -> (a, b) Nx.dtype =
 fun kind ->
  match kind with
  | Bigarray.Int8_unsigned -> Nx.UInt8
  | Bigarray.Int8_signed -> Nx.Int8
  | Bigarray.Int16_unsigned -> Nx.UInt16
  | Bigarray.Int16_signed -> Nx.Int16
  | Bigarray.Int32 -> Nx.Int32
  | Bigarray.Int64 -> Nx.Int64
  | Bigarray.Float16 -> Nx.Float16
  | Bigarray.Float32 -> Nx.Float32
  | Bigarray.Float64 -> Nx.Float64
  | Bigarray.Complex32 -> Nx.Complex32
  | Bigarray.Complex64 -> Nx.Complex64
  | _ ->
      let kind_str = kind_to_string kind in
      fail_msg "Unsupported NPY dtype for saving: %s" kind_str

let ba_kind_of_dtype : type a b. (a, b) Nx.dtype -> (a, b) Bigarray.kind =
 fun dtype ->
  match dtype with
  | Nx.UInt8 -> Bigarray.Int8_unsigned
  | Nx.Int8 -> Bigarray.Int8_signed
  | Nx.UInt16 -> Bigarray.Int16_unsigned
  | Nx.Int16 -> Bigarray.Int16_signed
  | Nx.Int32 -> Bigarray.Int32
  | Nx.Int64 -> Bigarray.Int64
  | Nx.Float16 -> Bigarray.Float16
  | Nx.Float32 -> Bigarray.Float32
  | Nx.Float64 -> Bigarray.Float64
  | Nx.Complex32 -> Bigarray.Complex32
  | Nx.Complex64 -> Bigarray.Complex64
  | Nx.Int -> Bigarray.Int
  | Nx.NativeInt -> Bigarray.Nativeint

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
