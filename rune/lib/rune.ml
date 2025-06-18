include Tensor

(* ───── Devices ───── *)

type 'a device = Nx_rune.context

let native : [ `cpu ] device = Nx_rune.create_context ~device:Native ()
let metal () : [ `metal ] device = Nx_rune.create_context ~device:Metal ()
let cblas : [ `cblas ] device = Nx_rune.create_context ~device:Cblas ()
let device t = Nx_rune.context t
let jit = Jit.jit
let grad = Autodiff.grad
let grads = Autodiff.grads
let value_and_grad = Autodiff.value_and_grad
let value_and_grads = Autodiff.value_and_grads

(* Export modules for testing *)
module Nx_rune = Nx_rune
module Jit = Jit
