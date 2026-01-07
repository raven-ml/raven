type t = P : ('a, 'b) Nx.t -> t
type archive = (string, t) Hashtbl.t

let convert_result_with_error : type a b.
    (a, b) Nx.dtype -> t -> ((a, b) Nx.t, Error.t) result =
 fun target_dtype packed ->
  match packed with
  | P nx -> (
      let source_dtype = Nx.dtype nx in
      let source_ba_kind = Nx_core.Dtype.to_bigarray_ext_kind source_dtype in
      let target_ba_kind = Nx_core.Dtype.to_bigarray_ext_kind target_dtype in
      (* Try Npy.Eq.Kind first for standard types *)
      match Npy.Eq.Kind.( === ) source_ba_kind target_ba_kind with
      | Some Npy.Eq.W -> Ok nx
      | None -> (
          match Nx_core.Dtype.equal_witness source_dtype target_dtype with
          | Some Type.Equal -> Ok (nx : (a, b) Nx.t)
          | None -> Error Unsupported_dtype))

let as_float16 packed = convert_result_with_error Nx.float16 packed
let as_bfloat16 packed = convert_result_with_error Nx.bfloat16 packed
let as_float32 packed = convert_result_with_error Nx.float32 packed
let as_float64 packed = convert_result_with_error Nx.float64 packed
let as_int8 packed = convert_result_with_error Nx.int8 packed
let as_int16 packed = convert_result_with_error Nx.int16 packed
let as_int32 packed = convert_result_with_error Nx.int32 packed
let as_int64 packed = convert_result_with_error Nx.int64 packed
let as_uint8 packed = convert_result_with_error Nx.uint8 packed
let as_uint16 packed = convert_result_with_error Nx.uint16 packed
let as_bool packed = convert_result_with_error Nx.bool packed
let as_complex32 packed = convert_result_with_error Nx.complex64 packed
let as_complex64 packed = convert_result_with_error Nx.complex128 packed
