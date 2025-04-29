open Ndarray_core

module B = struct
  type ('a, 'b) b_t = ('a, 'b) Internal.t
  type context = Internal.context

  let create_context () = { Internal.pool = Parallel.get_or_setup_pool () }
  let descriptor t = Internal.descriptor t
  let buffer t = Internal.buffer t
  let from_buffer _ctx descriptor buffer = { Internal.buffer; descriptor }
  let view descriptor t = { t with Internal.descriptor }

  (* Creation *)

  let empty _ctx dtype shape = Internal.empty dtype shape
  let copy _ctx t = Internal.copy t
  let blit _ctx src dst = Internal.blit src dst
  let fill _ctx value t = Internal.fill value t

  (* Element-wise Binary Operations *)

  let add context t1 t2 out = Binary_ops.add context t1 t2 out
  let sub context t1 t2 out = Binary_ops.sub context t1 t2 out
  let mul context t1 t2 out = Binary_ops.mul context t1 t2 out
  let div context t1 t2 out = Binary_ops.div context t1 t2 out
  let pow context t1 t2 out = Binary_ops.pow context t1 t2 out
  let rem context t1 t2 out = Binary_ops.rem context t1 t2 out

  (* Fused multiply‑add *)

  let fma context a b c out = Ternary_ops.fma context a b c out

  (* Bitwise Ops *)

  let bit_and context a b out = Binary_ops.bit_and context a b out
  let bit_or context a b out = Binary_ops.bit_or context a b out
  let bit_xor context a b out = Binary_ops.bit_xor context a b out
  let bit_not context t out = Unary_ops.bit_not context t out

  (* Comparison Operations *)

  let equal context t1 t2 out = Binary_ops.equal context t1 t2 out
  let greater context t1 t2 out = Binary_ops.greater context t1 t2 out
  let less context t1 t2 out = Binary_ops.less context t1 t2 out

  let greater_equal context t1 t2 out =
    Binary_ops.greater_equal context t1 t2 out

  let less_equal context t1 t2 out = Binary_ops.less_equal context t1 t2 out

  (* Element-wise Unary Operations *)

  let neg context t out = Unary_ops.neg context t out
  let abs context t out = Unary_ops.abs context t out
  let sign context t out = Unary_ops.sign context t out
  let sqrt context t out = Unary_ops.sqrt context t out
  let exp context t out = Unary_ops.exp context t out
  let log context t out = Unary_ops.log context t out
  let sin context t out = Unary_ops.sin context t out
  let cos context t out = Unary_ops.cos context t out
  let tan context t out = Unary_ops.tan context t out
  let asin context t out = Unary_ops.asin context t out
  let acos context t out = Unary_ops.acos context t out
  let atan context t out = Unary_ops.atan context t out
  let sinh context t out = Unary_ops.sinh context t out
  let cosh context t out = Unary_ops.cosh context t out
  let tanh context t out = Unary_ops.tanh context t out
  let asinh context t out = Unary_ops.asinh context t out
  let acosh context t out = Unary_ops.acosh context t out
  let atanh context t out = Unary_ops.atanh context t out

  (* Rounding & checks *)

  let floor context t out = Unary_ops.floor context t out
  let ceil context t out = Unary_ops.ceil context t out
  let round context t out = Unary_ops.round context t out
  let isnan context t out = Unary_ops.isnan context t out
  let isinf context t out = Unary_ops.isinf context t out
  let isfinite context t out = Unary_ops.isfinite context t out

  (* Indexing *)

  let where context cond t1 t2 out = Index_ops.where context cond t1 t2 out

  (* Sorting & selection *)

  let sort context ~axis t out = Sort_ops.sort context ~axis t out
  let argsort context ~axis t out = Sort_ops.argsort context ~axis t out
  let argmax context ~axis t out = Sort_ops.argmax context ~axis t out
  let argmin context ~axis t out = Sort_ops.argmin context ~axis t out

  (* Reductions *)

  let sum context ~axes ~keepdims arr out =
    Reduce_ops.sum context ~axes ~keepdims arr out

  let prod context ~axes ~keepdims arr out =
    Reduce_ops.prod context ~axes ~keepdims arr out

  let max context ~axes ~keepdims arr out =
    Reduce_ops.max context ~axes ~keepdims arr out

  let min context ~axes ~keepdims arr out =
    Reduce_ops.min context ~axes ~keepdims arr out

  (* Core linear‑algebra primitives *)

  let matmul context a b out = Linalg_ops.matmul context a b out
end

include B
