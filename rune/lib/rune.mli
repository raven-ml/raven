include module type of Nx

(** {2 Nx Interop} *)

val to_nx : ('a, 'b) t -> ('a, 'b) Nx.t
val of_nx : ('a, 'b) Nx.t -> ('a, 'b) t

(** {2 Automatic Differentiation}

    Functions for automatic differentiation and gradient computation. *)

val grad : (('a, 'b) t -> ('c, 'd) t) -> ('a, 'b) t -> ('a, 'b) t
(** [grad f t] computes the gradient of [f] with respect to [t].

    Returns a tensor of the same shape as [t] containing the gradient values.

    {@ocaml[
      # let x = create float32 [| 2 |] [| 3.; 4. |] in
        let f t = sum (mul_s t 2.) in
        grad f x |> item []
      - : float = 2.
    ]} *)

val grads :
  (('a, 'b) t list -> ('c, 'd) t) -> ('a, 'b) t list -> ('a, 'b) t list
(** [grads f ts] computes gradients of [f] with respect to each tensor in [ts].

    Returns a list of gradients, one for each input tensor.

    {@ocaml[
      # let xs = [create float32 [| 2 |] [| 3. |]; create float32 [| 2 |] [| 4. |]] in
        let f ts = sum (mul_s (List.hd ts) 2.) +. sum (mul_s (List.nth ts 1) 3.) in
        grads f xs |> List.map (fun t -> item t [])
      - : float list = [6.; 12.]
    ]} *)

val value_and_grad :
  (('a, 'b) t -> ('c, 'd) t) -> ('a, 'b) t -> ('c, 'd) t * ('a, 'b) t
(** [value_and_grad f t] computes both the value of [f] and the gradient with
    respect to [t].

    Returns a tuple of the function value and the gradient tensor.

    {@ocaml[
      # let x = create float32 [| 2 |] [| 3. |] in
        let f t = sum (mul_s t 2.) in
        value_and_grad f x |> (fun (v, g) -> (item v [], item g []))
      - : float * float = (6., 2.)
    ]} *)

val value_and_grads :
  (('a, 'b) t list -> ('c, 'd) t) ->
  ('a, 'b) t list ->
  ('c, 'd) t * ('a, 'b) t list
(** [value_and_grads f ts] computes both the value of [f] and the gradients with
    respect to each tensor in [ts].

    Returns a tuple of the function value and a list of gradient tensors.

    {@ocaml[
      # let xs = [create float32 [| 2 |] [| 3. |]; create float32 [| 2 |] [| 4. |]] in
        let f ts = sum (mul_s (List.hd ts) 2.) +. sum (mul_s (List.nth ts 1) 3.) in
        value_and_grads f xs |> (fun (v, gs) -> (item v [], List.map (fun g -> item g []) gs))
      - : float * float list = (18., [6.; 12.])
    ]} *)

val jvp :
  (('a, 'b) t -> ('c, 'd) t) ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('c, 'd) t * ('c, 'd) t
(** [jvp f primals tangents] computes a Jacobian-vector product (forward-mode
    AD).

    Returns a tuple of (primal_output, tangent_output) where:
    - primal_output = f(primals)
    - tangent_output = Jf(primals) · tangents

    {@ocaml[
      # let x = scalar float32 2. in
        let v = scalar float32 1. in
        let f x = mul x x in
        jvp f x v |> (fun (p, t) -> (item p [], item t []))
      - : float * float = (4., 4.)
    ]} *)

val jvp_aux :
  (('a, 'b) t -> ('c, 'd) t * 'e) ->
  ('a, 'b) t ->
  ('a, 'b) t ->
  ('c, 'd) t * ('c, 'd) t * 'e
(** [jvp_aux f primals tangents] like [jvp] but for functions with auxiliary
    output.

    Returns (primal_output, tangent_output, aux) where aux is the auxiliary
    data.

    {@ocaml[
      # let x = scalar float32 2. in
        let v = scalar float32 1. in
        let f x = (mul x x, shape x) in
        jvp_aux f x v |> (fun (p, t, aux) -> (item p [], item t [], aux))
      - : float * float * int array = (4., 4., [||])
    ]} *)

val jvps :
  (('a, 'b) t list -> ('c, 'd) t) ->
  ('a, 'b) t list ->
  ('a, 'b) t list ->
  ('c, 'd) t * ('c, 'd) t
(** [jvps f primals tangents] computes JVP for functions with multiple inputs.

    Returns (primal_output, tangent_output) for the list of inputs.

    {@ocaml[
      # let xs = [scalar float32 3.; scalar float32 2.] in
        let vs = [scalar float32 1.; scalar float32 0.5] in
        let f inputs = mul (List.hd inputs) (List.nth inputs 1) in
        jvps f xs vs |> (fun (p, t) -> (item p [], item t []))
      - : float * float = (6., 3.5)
    ]} *)

val vjp :
  (('a, 'b) t -> ('c, 'd) t) ->
  ('a, 'b) t ->
  ('c, 'd) t ->
  ('c, 'd) t * ('a, 'b) t
(** [vjp f primal cotangent] computes a vector-Jacobian product (reverse-mode
    AD).

    Returns a tuple of (primal_output, gradient) where:
    - primal_output = f(primal)
    - gradient = cotangent · Jf(primal)

    This is like [grad] but allows specifying a custom cotangent instead of the
    implicit all-ones vector.

    {@ocaml[
      # let x = scalar float32 2. in
        let cot = scalar float32 1. in
        let f x = mul x x in
        vjp f x cot |> (fun (p, g) -> (item p [], item g []))
      - : float * float = (4., 4.)
    ]} *)

val vjps :
  (('a, 'b) t list -> ('c, 'd) t) ->
  ('a, 'b) t list ->
  ('c, 'd) t ->
  ('c, 'd) t * ('a, 'b) t list
(** [vjps f primals cotangent] computes VJP for functions with multiple inputs.

    Returns (primal_output, gradients) for the list of inputs.

    {@ocaml[
      # let xs = [scalar float32 3.; scalar float32 2.] in
        let cot = scalar float32 1. in
        let f inputs = mul (List.hd inputs) (List.nth inputs 1) in
        vjps f xs cot |> (fun (p, gs) -> (item p [], List.map (fun g -> item g []) gs))
      - : float * float list = (6., [2.; 3.])
    ]} *)

val no_grad : (unit -> 'a) -> 'a
(** [no_grad f] evaluates [f ()] without recording operations for automatic
    differentiation. This mirrors JAX's [lax.stop_gradient] semantics when
    applied to a computation block: all tensors produced within [f] are treated
    as constants for subsequent gradient calculations. *)

val detach : ('a, 'b) t -> ('a, 'b) t
(** [detach t] returns a tensor with the same value as [t] but which is treated
    as a constant with respect to automatic differentiation. Equivalent to JAX's
    [lax.stop_gradient] on a single tensor. *)

(** {2 Gradient Checking} *)

type method_ = [ `Central | `Forward | `Backward ]
(** Finite difference method to use:
    - [`Central]: (f(x+h) - f(x-h)) / 2h (most accurate)
    - [`Forward]: (f(x+h) - f(x)) / h
    - [`Backward]: (f(x) - f(x-h)) / h *)

val finite_diff :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b) t -> ('c, 'd) t) ->
  ('a, 'b) t ->
  ('a, 'b) t
(** [finite_diff ?eps ?method_ f x] computes the gradient of scalar-valued
    function [f] with respect to input [x] using finite differences. The
    function [f] must return a scalar tensor.

    @param eps Step size for finite differences (default: 1e-5)
    @param method_ Finite difference method (default: `Central)
    @param f Function to differentiate (must return scalar)
    @param x Input tensor at which to compute gradient (must be float type)
    @return Gradient tensor with same shape as [x] *)

val finite_diff_jacobian :
  ?eps:float ->
  ?method_:method_ ->
  (('a, 'b) t -> ('c, 'd) t) ->
  ('a, 'b) t ->
  ('c, 'd) t
(** [finite_diff_jacobian ?eps ?method_ f x] computes the Jacobian matrix of
    function [f] with respect to input [x] using finite differences.

    @param eps Step size for finite differences (default: 1e-5)
    @param method_ Finite difference method (default: `Central)
    @param f Function to differentiate
    @param x Input tensor at which to compute Jacobian (must be float type)
    @return
      Jacobian matrix of shape [output_size × input_size] if f returns
      non-scalar, or gradient vector with same shape as [x] if f returns scalar
*)

type gradient_check_result = {
  max_abs_error : float;
      (** Maximum absolute error between autodiff and finite difference
          gradients *)
  max_rel_error : float;
      (** Maximum relative error between autodiff and finite difference
          gradients *)
  mean_abs_error : float;
      (** Mean absolute error across all checked elements *)
  mean_rel_error : float;
      (** Mean relative error across all checked elements *)
  failed_indices : (int array * float * float * float) list;
      (** List of (index, autodiff_value, finite_diff_value, absolute_error) for
          failed elements *)
  passed : bool;  (** Whether all checked elements passed the tolerance tests *)
  num_checked : int;  (** Total number of elements checked *)
  num_failed : int;  (** Number of elements that failed the tolerance tests *)
}

val check_gradient :
  ?eps:float ->
  ?rtol:float ->
  ?atol:float ->
  ?verbose:bool ->
  ?check_indices:int list option ->
  ?method_:[ `Central | `Forward | `Backward ] ->
  ((float, 'a) t -> ('b, 'c) t) ->
  (float, 'a) t ->
  [ `Pass of gradient_check_result | `Fail of gradient_check_result ]
(** [check_gradient ?eps ?rtol ?atol ?verbose ?check_indices ?method_ f x]
    compares the gradient of [f] at [x] computed via automatic differentiation
    against finite differences.

    @param eps Step size for finite differences (default: 1e-5)
    @param rtol Relative tolerance for comparison (default: 1e-3)
    @param atol Absolute tolerance for comparison (default: 1e-5)
    @param verbose Whether to print detailed error information (default: false)
    @param check_indices Optional list of indices to check (default: all)
    @param method_ Finite difference method (default: `Central)
    @param f Function to check (must return scalar)
    @param x Input tensor at which to check gradient
    @return
      [`Pass result] if all gradients match within tolerance, [`Fail result]
      otherwise

    The check passes if for each element either:
    - absolute_error <= atol, or
    - relative_error <= rtol *)

val check_gradients :
  ?eps:float ->
  ?rtol:float ->
  ?atol:float ->
  ?verbose:bool ->
  ?method_:[ `Central | `Forward | `Backward ] ->
  ((float, 'a) t list -> ('b, 'c) t) ->
  (float, 'a) t list ->
  [ `Pass of gradient_check_result list | `Fail of gradient_check_result list ]
(** [check_gradients ?eps ?rtol ?atol ?verbose ?method_ f xs] compares the
    gradients of [f] with respect to each input in [xs] computed via automatic
    differentiation against finite differences.

    @param eps Step size for finite differences (default: 1e-5)
    @param rtol Relative tolerance for comparison (default: 1e-3)
    @param atol Absolute tolerance for comparison (default: 1e-5)
    @param verbose Whether to print detailed error information (default: false)
    @param method_ Finite difference method (default: `Central)
    @param f Function to check (must return scalar)
    @param xs List of input tensors at which to check gradients
    @return
      [`Pass results] if all gradients match within tolerance, [`Fail results]
      otherwise

    Returns a list of results, one for each input tensor. *)

(** {2 Vectorizing Map (vmap)}

    Functions for mapping computations over batch dimensions. *)

type axis_spec = Vmap.axis_spec =
  | Map of int  (** Map over this axis index *)
  | NoMap  (** Don't map this axis *)

type 'a in_axes_spec = 'a Vmap.in_axes_spec =
  | Single of axis_spec
  | Container of 'a

type 'a out_axes_spec = 'a Vmap.out_axes_spec =
  | OutSingle of int option
  | OutContainer of 'a

val vmap :
  ?in_axes:'a in_axes_spec ->
  ?out_axes:'b out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) t -> ('e, 'f) t) ->
  ('c, 'd) t ->
  ('e, 'f) t
(** [vmap ?in_axes ?out_axes ?axis_name ?axis_size f] creates a vectorized
    version of function [f].

    @param in_axes
      Specifies which input array axes to map over. Default: Single (Map 0) -
      maps over the first axis.
    @param out_axes
      Specifies where the mapped axis should appear in output. Default:
      OutSingle (Some 0) - mapped axis at position 0. Use None to not include
      mapped axis in output.
    @param axis_name
      Optional name for the mapped axis (for collective operations).
    @param axis_size
      Optional size of the mapped axis. Required when in_axes is NoMap.
    @param f The function to be mapped.

    {@ocaml[
      # let batch_x = create float32 [| 10; 3; 3 |] (Array.init 90 float_of_int) in
        let w = create float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let batched_matmul = vmap (fun x -> matmul x w) in
        batched_matmul batch_x |> shape
      - : int array = [| 10; 3; 2 |]
    ]} *)

val vmaps :
  ?in_axes:Vmap.axis_spec list ->
  ?out_axes:'b Vmap.out_axes_spec ->
  ?axis_name:string ->
  ?axis_size:int ->
  (('c, 'd) t list -> ('e, 'f) t) ->
  ('c, 'd) t list ->
  ('e, 'f) t
(** [vmaps ?in_axes ?out_axes ?axis_name ?axis_size f] creates a vectorized
    version of function [f] that takes multiple tensor arguments.

    Similar to {!vmap} but for functions taking multiple arguments.

    Examples:
    {[
      let x = create float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
      let y = create float32 [| 3; 2 |] [| 10.; 20.; 30.; 40.; 50.; 60. |] in
      let batched_add = vmaps (fun [x; y] -> add x y) in
      batched_add [x; y] |> to_float1
      - : float array = [| 11.; 22.; 33.; 44.; 55.; 66. |]
    ]} *)

(** {2 Random Number Generation}

    JAX-style splittable PRNG for reproducible random number generation. *)

module Rng : sig
  type key
  (** PRNG key type representing the random state *)

  val key : int -> key
  (** Create a PRNG key from a seed value.

      The seed is used to initialize the random state. Same seed produces same
      sequence of random numbers. *)

  val split : ?n:int -> key -> key array
  (** Split a PRNG key into multiple new keys.

      [split key n] returns an array of [n] new independent keys derived from
      the input key. The original key should not be reused after splitting to
      maintain statistical guarantees.

      @param key The key to split
      @param n Number of new keys to generate (default: 2)
      @return Array of new independent keys *)

  val fold_in : key -> int -> key
  (** Fold data into a key to derive a new key.

      [fold_in key data] combines a key with integer data to produce a new key.
      Useful for deriving keys based on iteration indices or other structured
      data.

      @param key The base key
      @param data Integer data to fold into the key
      @return New derived key *)

  val to_int : key -> int
  (** Convert key to integer representation for debugging.

      [to_int key] returns the internal integer representation of the key. This
      is mainly useful for debugging and should not be used to recreate keys. *)

  val uniform : key -> ('a, 'b) dtype -> int array -> ('a, 'b) t
  (** Generate uniform random values in \[0, 1).

      [uniform key dtype shape] generates a tensor of the given shape with
      values uniformly distributed in the half-open interval \[0, 1).

      @param key PRNG key for random generation
      @param dtype Data type of the output tensor
      @param shape Shape of the output tensor
      @return Tensor with uniform random values *)

  val normal : key -> ('a, 'b) dtype -> int array -> ('a, 'b) t
  (** Generate standard normal random values.

      [normal key dtype shape] generates a tensor of the given shape with values
      sampled from a standard normal distribution (mean=0, std=1).

      @param key PRNG key for random generation
      @param dtype Data type of the output tensor
      @param shape Shape of the output tensor
      @return Tensor with normal random values *)

  val randint : key -> min:int -> max:int -> int array -> int32_t
  (** Generate random integers in a range.

      [randint key ~min ~max shape] generates a tensor of integers uniformly
      distributed in the half-open interval \[min, max).

      @param key PRNG key for random generation
      @param min Minimum value (inclusive)
      @param max Maximum value (exclusive)
      @param shape Shape of the output tensor
      @return Tensor with random integer values *)

  val bernoulli : key -> p:float -> int array -> bool_t
  (** Generate Bernoulli random values.

      [bernoulli key ~p shape] generates a tensor of boolean values where each
      element is true with probability [p].

      @param key PRNG key for random generation
      @param p Probability of true (must be in \[0, 1\])
      @param shape Shape of the output tensor
      @return Tensor with boolean random values *)

  val permutation : key -> int -> int32_t
  (** Generate random permutation.

      [permutation key n] generates a random permutation of integers from 0 to
      n-1.

      @param key PRNG key for random generation
      @param n Number of elements to permute
      @return 1-D tensor containing a random permutation *)

  val shuffle : key -> ('a, 'b) t -> ('a, 'b) t
  (** Randomly shuffle the first dimension of a tensor.

      [shuffle key x] returns a copy of tensor [x] with its first dimension
      randomly shuffled.

      @param key PRNG key for random generation
      @param x Tensor to shuffle
      @return Shuffled tensor *)

  val categorical :
    key -> ?axis:int -> ?shape:int array -> ('a, 'b) t -> int32_t
  (** Sample from a categorical distribution.

      [categorical key logits ?axis] samples indices from a categorical
      distribution defined by logits along the specified axis.

      @param key PRNG key for random generation
      @param logits Unnormalized log probabilities
      @param axis Axis along which to sample (default: -1)
      @param shape Shape of the output tensor (default: scalar)
      @return Tensor of sampled indices *)

  val truncated_normal :
    key -> ('a, 'b) dtype -> lower:'a -> upper:'a -> int array -> ('a, 'b) t
  (** Generate random values from a truncated normal distribution.

      [truncated_normal key dtype ~lower ~upper shape] generates values from a
      normal distribution truncated to the range [lower, upper].

      @param key PRNG key for random generation
      @param dtype Data type of the output tensor
      @param lower Lower truncation bound
      @param upper Upper truncation bound
      @param shape Shape of the output tensor
      @return Tensor with truncated normal random values *)
end

(** {2 Debugging}

    Functions for debugging, JIT compilation, and gradient computation. *)

val debug : ('a -> 'b) -> 'a -> 'b
(** [debug f x] applies [f] to [x] and prints debug information.

    Useful for inspecting intermediate values during development. *)

val debug_with_context : string -> (unit -> 'a) -> 'a
(** [debug_with_context context f] runs [f] with a debug context.

    Prints the context name before executing [f]. Useful for tracing specific
    computation paths. *)

val debug_push_context : string -> unit
(** [debug_push_context context] pushes a new debug context.

    Use this to mark the start of a specific computation section. The context
    will be printed in debug messages. *)

val debug_pop_context : unit -> unit
(** [debug_pop_context ()] pops the last debug context.

    Use this to mark the end of a specific computation section. The context will
    be removed from the debug stack. *)

(** {2 Just-In-Time Compilation}

    Functions for JIT compilation of tensor operations. *)

type jit_device = [ `metal | `llvm ]
(** [jit_device] represents devices supported in JIT compilation.

    - [`llvm]: CPU device using LLVM for JIT-compiled operations.
    - [`metal]: GPU device using Metal for JIT-compiled operations on Apple
      devices. *)

val is_jit_device_available : jit_device -> bool
(** [is_jit_device_available dev] checks if the specified device is available.

    Returns true if the device can be used for tensor operations. *)

val jit :
  ?device:jit_device -> (('a, 'b) t -> ('c, 'd) t) -> ('a, 'b) t -> ('c, 'd) t
(** [jit f t] compiles the function [f] for efficient execution on [t].

    Returns a compiled version of [f] that can be called with tensors of the
    same shape and type as [t]. This can significantly speed up repeated calls.

    {@ocaml[
      # let x = create float32 [| 2 |] [| 3. |] in
        let f t = sum (mul_s t 2.) in
        let compiled_f = jit f x in
        compiled_f x |> item []
      - : float = 6.
    ]} *)
