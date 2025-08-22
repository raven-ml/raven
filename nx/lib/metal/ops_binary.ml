open Nx_core
open Metal

let add ctx a b = Dispatch.dispatch_binary_op ctx "add" a b
let mul ctx a b = Dispatch.dispatch_binary_op ctx "mul" a b
let idiv ctx a b = Dispatch.dispatch_binary_op ctx "idiv" a b
let fdiv ctx a b = Dispatch.dispatch_binary_op ctx "fdiv" a b
let max ctx a b = Dispatch.dispatch_binary_op ctx "max" a b
let mod_ ctx a b = Dispatch.dispatch_binary_op ctx "mod" a b
let pow ctx a b = Dispatch.dispatch_binary_op ctx "pow" a b
let cmplt ctx a b = Dispatch.dispatch_comparison_op ctx "cmplt" a b
let cmpne ctx a b = Dispatch.dispatch_comparison_op ctx "cmpne" a b
let cmpeq ctx a b = Dispatch.dispatch_comparison_op ctx "cmpeq" a b

let is_integer_dtype : type a b. (a, b) Dtype.t -> bool = function
  | Dtype.Int32 | Dtype.Int64 | Dtype.UInt8 | Dtype.UInt16 | Dtype.Int8
  | Dtype.Int16 | Dtype.Int | Dtype.NativeInt | Dtype.Bool ->
      true
  | _ -> false

let dispatch_bitwise ctx op_name a b =
  (* Bitwise operations only work on integer types *)
  if is_integer_dtype a.Internal.dtype then
    Dispatch.dispatch_binary_op ctx op_name a b
  else
    failwith
      (Printf.sprintf "Bitwise operation %s not supported for dtype" op_name)

let xor ctx a b = dispatch_bitwise ctx "xor" a b
let or_ ctx a b = dispatch_bitwise ctx "or" a b
let and_ ctx a b = dispatch_bitwise ctx "and" a b
