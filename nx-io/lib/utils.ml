open Bigarray
module Nd = Nx

type packed_nx = P : ('a, 'b) Nd.t -> packed_nx

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

let dtype_of_ba_kind : type a b. (a, b) Bigarray.kind -> (a, b) Nd.dtype =
 fun kind ->
  match kind with
  | Bigarray.Int8_unsigned -> Nd.UInt8
  | Bigarray.Int8_signed -> Nd.Int8
  | Bigarray.Int16_unsigned -> Nd.UInt16
  | Bigarray.Int16_signed -> Nd.Int16
  | Bigarray.Int32 -> Nd.Int32
  | Bigarray.Int64 -> Nd.Int64
  | Bigarray.Float16 -> Nd.Float16
  | Bigarray.Float32 -> Nd.Float32
  | Bigarray.Float64 -> Nd.Float64
  | Bigarray.Complex32 -> Nd.Complex32
  | Bigarray.Complex64 -> Nd.Complex64
  | _ ->
      let kind_str = kind_to_string kind in
      fail_msg "Unsupported NPY dtype for saving: %s" kind_str

let ba_kind_of_dtype : type a b. (a, b) Nd.dtype -> (a, b) Bigarray.kind =
 fun dtype ->
  match dtype with
  | Nd.UInt8 -> Bigarray.Int8_unsigned
  | Nd.Int8 -> Bigarray.Int8_signed
  | Nd.UInt16 -> Bigarray.Int16_unsigned
  | Nd.Int16 -> Bigarray.Int16_signed
  | Nd.Int32 -> Bigarray.Int32
  | Nd.Int64 -> Bigarray.Int64
  | Nd.Float16 -> Bigarray.Float16
  | Nd.Float32 -> Bigarray.Float32
  | Nd.Float64 -> Bigarray.Float64
  | Nd.Complex32 -> Bigarray.Complex32
  | Nd.Complex64 -> Bigarray.Complex64

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
            (Nd.dtype_to_string target_kind)
            (Nd.dtype_to_string source_kind))
