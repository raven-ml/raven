open Nx_core
open Metal

let sum ctx ~axes ~keepdims t =
  Dispatch.dispatch_reduce_op ctx "reduce_sum" ~axes ~keepdims t

let max ctx ~axes ~keepdims t =
  Dispatch.dispatch_reduce_op ctx "reduce_max" ~axes ~keepdims t

let prod ctx ~axes ~keepdims t =
  Dispatch.dispatch_reduce_op ctx "reduce_prod" ~axes ~keepdims t
