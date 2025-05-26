include Tensor

(* ───── Devices ───── *)

type 'a device = Nx_rune.context

let cpu : [ `cpu ] device = Nx_rune.create_context ~device:Cpu ()
let metal () : [ `metal ] device = Nx_rune.create_context ~device:Metal ()
let device t = Nx_rune.context t
let jit = Jit.jit
let grad = Autodiff.grad
