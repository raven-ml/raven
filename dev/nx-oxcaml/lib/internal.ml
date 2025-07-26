open Nx_core

(* Unboxed buffer types using OxCaml's unboxed arrays *)
module Unboxed = struct
  (* For now, we'll use regular bigarrays but in the future we can switch to unboxed arrays
     when they become available for all types we need *)
  type ('a, 'b) buffer = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
end

type ('a, 'b) buffer = ('a, 'b) Unboxed.buffer
type context = { pool : Parallel.pool option }

type ('a, 'b) t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t;
}

(* Basic Accessors *)
let dtype { dtype; _ } = dtype
let buffer { buffer; _ } = buffer
let view { view; _ } = view
let shape { view; _ } = View.shape view
let strides { view; _ } = View.strides view
let stride axis { view; _ } = View.stride axis view
let offset { view; _ } = View.offset view
let size { view; _ } = View.numel view
let numel { view; _ } = View.numel view
let dims { view; _ } = View.shape view
let dim axis { view; _ } = View.dim axis view
let ndim { view; _ } = View.ndim view
let is_c_contiguous { view; _ } = View.is_c_contiguous view

(* Low-level helper to create a buffer *)
let create_buffer_unsafe (type a b) (dt : (a, b) Dtype.t)
    (size_in_elements : int) : (a, b) buffer =
  Bigarray.Array1.create
    (Dtype.to_bigarray_kind dt)
    Bigarray.c_layout size_in_elements

(* Unboxed arithmetic operations using float# and other unboxed types *)
module Unboxed_ops = struct
  (* These will use unboxed types for performance *)
  external add_float : float# -> float# -> float# = "%addfloat"
  external sub_float : float# -> float# -> float# = "%subfloat"
  external mul_float : float# -> float# -> float# = "%mulfloat"
  external div_float : float# -> float# -> float# = "%divfloat"
  external neg_float : float# -> float# = "%negfloat"
  external sqrt_float : float# -> float# = "sqrt" [@@unboxed] [@@noalloc]
  external sin_float : float# -> float# = "sin" [@@unboxed] [@@noalloc]
  external cos_float : float# -> float# = "cos" [@@unboxed] [@@noalloc]
  external exp_float : float# -> float# = "exp" [@@unboxed] [@@noalloc]
  external log_float : float# -> float# = "log" [@@unboxed] [@@noalloc]
  
  external add_int32 : int32# -> int32# -> int32# = "%int32_add"
  external sub_int32 : int32# -> int32# -> int32# = "%int32_sub"
  external mul_int32 : int32# -> int32# -> int32# = "%int32_mul"
  external div_int32 : int32# -> int32# -> int32# = "%int32_div"
  external neg_int32 : int32# -> int32# = "%int32_neg"
  
  external add_int64 : int64# -> int64# -> int64# = "%int64_add"
  external sub_int64 : int64# -> int64# -> int64# = "%int64_sub"
  external mul_int64 : int64# -> int64# -> int64# = "%int64_mul"
  external div_int64 : int64# -> int64# -> int64# = "%int64_div"
  external neg_int64 : int64# -> int64# = "%int64_neg"
  
  (* Conversions between boxed and unboxed *)
  external box_float : float# -> float = "%box_float"
  external unbox_float : float -> float# = "%unbox_float"
  external box_int32 : int32# -> int32 = "%box_int32"
  external unbox_int32 : int32 -> int32# = "%unbox_int32"
  external box_int64 : int64# -> int64 = "%box_int64"
  external unbox_int64 : int64 -> int64# = "%unbox_int64"
end

(* Helper to get pool from context, creating one if needed *)
let get_pool ctx =
  match ctx.pool with
  | Some pool -> pool
  | None -> Parallel.get_or_setup_pool ()