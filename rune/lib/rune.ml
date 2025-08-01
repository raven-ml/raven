include Tensor
include Tensor_with_debug

type ('a, 'b, 'dev) t = ('a, 'b) Tensor.t
type 'dev float16_t = (float, float16_elt, 'dev) t
type 'dev float32_t = (float, float32_elt, 'dev) t
type 'dev float64_t = (float, float64_elt, 'dev) t
type 'dev int8_t = (int, int8_elt, 'dev) t
type 'dev uint8_t = (int, uint8_elt, 'dev) t
type 'dev int16_t = (int, int16_elt, 'dev) t
type 'dev uint16_t = (int, uint16_elt, 'dev) t
type 'dev int32_t = (int32, int32_elt, 'dev) t
type 'dev int64_t = (int64, int64_elt, 'dev) t
type 'dev std_int_t = (int, int_elt, 'dev) t
type 'dev std_nativeint_t = (nativeint, nativeint_elt, 'dev) t
type 'dev complex32_t = (Complex.t, complex32_elt, 'dev) t
type 'dev complex64_t = (Complex.t, complex64_elt, 'dev) t

(* ───── Devices ───── *)

type 'a device = Nx_rune.context

let ocaml : [ `ocaml ] device = Nx_rune.create_context ~device:Ocaml ()
let c : [ `c ] device = Nx_rune.create_context ~device:C ()
let metal () : [ `metal ] device = Nx_rune.create_context ~device:Metal ()
let device t = Nx_rune.context t

let is_device_available = function
  | `ocaml -> Nx_rune.is_device_available Ocaml
  | `c -> Nx_rune.is_device_available C
  | `metal -> Nx_rune.is_device_available Metal

(* ───── JIT ───── *)

let jit = Jit.jit
let xla = Jit_xla.jit

(* ───── Autodiff ───── *)

let grad = Autodiff.grad
let grads = Autodiff.grads
let value_and_grad = Autodiff.value_and_grad
let value_and_grads = Autodiff.value_and_grads
let jvp = Autodiff.jvp
let jvp_aux = Autodiff.jvp_aux
let jvps = Autodiff.jvps

(* ───── Vmap ───── *)

type axis_spec = Vmap.axis_spec = Map of int | NoMap

type 'a in_axes_spec = 'a Vmap.in_axes_spec =
  | Single of axis_spec
  | Container of 'a

type 'a out_axes_spec = 'a Vmap.out_axes_spec =
  | OutSingle of int option
  | OutContainer of 'a

let vmap = Vmap.vmap
let vmaps = Vmap.vmaps

(* ───── Debugging ───── *)

let debug = Debug.debug
let debug_with_context = Debug.with_context
let debug_push_context = Debug.push_context
let debug_pop_context = Debug.pop_context
