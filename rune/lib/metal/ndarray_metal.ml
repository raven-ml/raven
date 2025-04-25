open Ndarray_core

module B : Backend_intf = struct
  type ('a, 'b) b_t = ('a, 'b) Internal.t
  type context = Internal.context

  let create_context () = Metal_context.create ()
  let descriptor t = Internal.descriptor t
  let buffer t = Internal.host_buffer t

  let from_buffer _ctx descriptor host_buffer =
    let buffer = Internal.metal_buffer_from_host _ctx host_buffer in
    { Internal.buffer; host_buffer; descriptor }

  let view descriptor t = { t with Internal.descriptor }

  (* Creation *)

  let empty ctx dtype shape =
    let t = Internal.empty ctx dtype shape in
    t

  let copy ctx t = Metal_ops.copy ctx t
  let blit ctx src dst = Metal_ops.blit ctx src dst
  let fill ctx value t = Metal_ops.fill ctx value t

  (* Element-wise Binary Operations *)

  let add context a b out = Metal_ops.binary_op_impl context Add a b out
  let sub context a b out = Metal_ops.binary_op_impl context Sub a b out
  let mul context a b out = Metal_ops.binary_op_impl context Mul a b out
  let div context a b out = Metal_ops.binary_op_impl context Div a b out
  let pow context a b out = Metal_ops.binary_op_impl context Pow a b out
  let rem context a b out = Metal_ops.binary_op_impl context Remainder a b out

  (* Fused multiply‑add *)

  let fma _ _ _ _ _ = failwith "todo"

  (* Comparison Operations *)

  let equal context a b out = Metal_ops.comparison_op_impl context Equal a b out

  let greater context a b out =
    Metal_ops.comparison_op_impl context Greater a b out

  let greater_equal context a b out =
    Metal_ops.comparison_op_impl context GreaterEqual a b out

  let less context a b out = Metal_ops.comparison_op_impl context Less a b out

  let less_equal context a b out =
    Metal_ops.comparison_op_impl context LessEqual a b out

  (* Element-wise Unary Operations *)

  let neg context a out = Metal_ops.unary_op_impl context Neg a out
  let abs context a out = Metal_ops.unary_op_impl context Abs a out
  let sign context a out = Metal_ops.unary_op_impl context Sign a out
  let sqrt context a out = Metal_ops.unary_op_impl context Sqrt a out
  let exp context a out = Metal_ops.unary_op_impl context Exp a out
  let log context a out = Metal_ops.unary_op_impl context Log a out
  let sin context a out = Metal_ops.unary_op_impl context Sin a out
  let cos context a out = Metal_ops.unary_op_impl context Cos a out
  let tan context a out = Metal_ops.unary_op_impl context Tan a out
  let asin context a out = Metal_ops.unary_op_impl context Asin a out
  let acos context a out = Metal_ops.unary_op_impl context Acos a out
  let atan context a out = Metal_ops.unary_op_impl context Atan a out
  let sinh context a out = Metal_ops.unary_op_impl context Sinh a out
  let cosh context a out = Metal_ops.unary_op_impl context Cosh a out
  let tanh context a out = Metal_ops.unary_op_impl context Tanh a out
  let asinh context a out = Metal_ops.unary_op_impl context Asinh a out
  let acosh context a out = Metal_ops.unary_op_impl context Acosh a out
  let atanh context a out = Metal_ops.unary_op_impl context Atanh a out

  (* Rounding & checks *)

  let floor _ = failwith "todo"
  let ceil _ = failwith "todo"
  let round _ = failwith "todo"
  let isnan _ = failwith "todo"
  let isinf _ = failwith "todo"
  let isfinite _ = failwith "todo"

  (* Bitwise Ops *)

  let bit_and _ = failwith "todo"
  let bit_or _ = failwith "todo"
  let bit_xor _ = failwith "todo"
  let bit_not _ = failwith "todo"

  (* Indexing *)

  let where _ _ _ _ = failwith "todo"

  (* Sorting & selection *)

  let sort _ = failwith "todo"
  let argsort _ = failwith "todo"
  let argmax _ = failwith "todo"
  let argmin _ = failwith "todo"

  (* Reductions *)

  let sum ctx ~axes ~keepdims t out = Metal_ops.sum ctx ~axes ~keepdims t out
  let prod _ = failwith "todo"
  let max _ = failwith "todo"
  let min _ = failwith "todo"

  (* Linear Algebra *)

  let matmul ctx a b out = Metal_ops.matmul ctx a b out
end

include B
