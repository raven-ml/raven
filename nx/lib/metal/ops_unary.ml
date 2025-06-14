open Nx_core
open Metal

let neg ctx t = Dispatch.dispatch_unary_op ctx "neg" t
let log2 ctx t = Dispatch.dispatch_unary_op ctx "log2" t
let exp2 ctx t = Dispatch.dispatch_unary_op ctx "exp2" t
let sin ctx t = Dispatch.dispatch_unary_op ctx "sin" t
let sqrt ctx t = Dispatch.dispatch_unary_op ctx "sqrt" t
let recip ctx t = Dispatch.dispatch_unary_op ctx "recip" t
