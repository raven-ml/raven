module B = Nx_core.Make_frontend (Nx_native)

type ('a, 'b) t = ('a, 'b) B.t

let context = B.create_context ()

(* Accessors *)

let shape t = B.shape t
let size t = B.size t
let numel t = B.numel t
let dtype t = B.dtype t
let view t = B.view t
let ndim t = B.ndim t
let broadcast_to x target_shape = B.broadcast_to context x target_shape
let add a b = B.add context a b
let mul a b = B.mul context a b
let sum ?axes ?(keepdims = false) x = B.sum context ?axes ~keepdims x

(* New functions *)
let full ~dtype shape fill_value = B.full context ~dtype shape fill_value
let zeros ~dtype shape = B.zeros context ~dtype shape
let ones ~dtype shape = B.ones context ~dtype shape
let full_like self fill_value = B.full_like context self fill_value
let zeros_like self = B.zeros_like context self
let ones_like self = B.ones_like context self

(* Potentially other functions to expose like empty, reshape, etc. *)
let empty ?dtype shape = B.empty context ?dtype shape
let reshape t new_shape = B.reshape context t new_shape
let expand t new_shape = B.expand context t new_shape
