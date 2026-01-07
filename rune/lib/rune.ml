include Tensor
module Rng = Tensor.Rng

type ('a, 'b) t = ('a, 'b) Tensor.t
type float16_t = (float, float16_elt) t
type float32_t = (float, float32_elt) t
type float64_t = (float, float64_elt) t
type int8_t = (int, int8_elt) t
type uint8_t = (int, uint8_elt) t
type int16_t = (int, int16_elt) t
type uint16_t = (int, uint16_elt) t
type int32_t = (int32, int32_elt) t
type int64_t = (int64, int64_elt) t
type uint32_t = (int32, uint32_elt) t
type uint64_t = (int64, uint64_elt) t
type complex64_t = (Complex.t, complex32_elt) t
type complex128_t = (Complex.t, complex64_elt) t

(* Re-export extended type aliases *)
type bfloat16_t = (float, Bigarray_ext.bfloat16_elt) t
type bool_t = (bool, Bigarray_ext.bool_elt) t
type int4_t = (int, Bigarray_ext.int4_signed_elt) t
type uint4_t = (int, Bigarray_ext.int4_unsigned_elt) t
type float8_e4m3_t = (float, Bigarray_ext.float8_e4m3_elt) t
type float8_e5m2_t = (float, Bigarray_ext.float8_e5m2_elt) t

(* Re-export extended dtype value constructors *)
let bfloat16 = Nx_core.Dtype.bfloat16
let bool = Nx_core.Dtype.bool
let int4 = Nx_core.Dtype.int4
let uint4 = Nx_core.Dtype.uint4
let float8_e4m3 = Nx_core.Dtype.float8_e4m3
let float8_e5m2 = Nx_core.Dtype.float8_e5m2

(* ───── Instrumentation Helpers ───── *)

let debug_hook =
  {
    Nx_core.Instrumentation.enabled = true;
    with_span =
      (fun ~op ?attrs:_ f ->
        (* Attributes ignored for now; Debug logs tensor stats/shapes. *)
        Debug.with_context op f);
    emit = (fun _ -> ());
  }

let () = Nx_core.Instrumentation.set_hook (Some debug_hook)
let enable_debug () = Nx_core.Instrumentation.set_hook (Some debug_hook)
let disable_debug () = Nx_core.Instrumentation.set_hook None
let with_debug f = Nx_core.Instrumentation.with_hook (Some debug_hook) f

(* ───── JIT ───── *)

type jit_device = [ `metal | `llvm ]

let is_jit_device_available = function
  | `llvm -> true
  | `metal -> (
      try
        let _ = Rune_jit_metal_or_missing.Device_info.get_default () in
        true
      with _ -> false)

let jit = Jit.jit

(* ───── Autodiff ───── *)

let vjp = Autodiff.vjp
let vjps = Autodiff.vjps
let grad = Autodiff.grad
let grads = Autodiff.grads
let value_and_grad = Autodiff.value_and_grad
let value_and_grads = Autodiff.value_and_grads
let jvp = Autodiff.jvp
let jvp_aux = Autodiff.jvp_aux
let jvps = Autodiff.jvps
let no_grad = Autodiff.no_grad
let detach = Autodiff.detach

(* ───── Gradient Checking ───── *)

module Finite_diff = Finite_diff
module Gradcheck = Gradcheck

type method_ = Finite_diff.method_

type gradient_check_result = Gradcheck.gradient_check_result = {
  max_abs_error : float;
  max_rel_error : float;
  mean_abs_error : float;
  mean_rel_error : float;
  failed_indices : (int array * float * float * float) list;
  passed : bool;
  num_checked : int;
  num_failed : int;
}

let finite_diff = Finite_diff.finite_diff
let finite_diff_jacobian = Finite_diff.finite_diff_jacobian
let check_gradient = Gradcheck.check_gradient
let check_gradients = Gradcheck.check_gradients

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

(* ───── Nx Interop ───── *)

let of_nx nx_tensor = of_bigarray (Nx.to_bigarray nx_tensor)
let to_nx t = Nx.of_bigarray (to_bigarray t)
