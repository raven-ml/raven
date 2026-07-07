(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Functional transformations over structured tensor collections.

    Rune differentiates functions over any user-defined parameter structure. A
    structure declares how to traverse its tensor leaves by implementing
    {!Nx.Ptree.S}; every transformation then works on it directly, preserving
    its type. Leaves may have different dtypes: a single forward and backward
    pass produces gradients for all of them.

    {[
    type params = { w : Nx.float32_t; b : Nx.float32_t }

    module Params = struct
      type t = params

      let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b } =
        { w = f w; b = f b }

      let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q
          =
        { w = f p.w q.w; b = f p.b q.b }

      let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b } =
        f w;
        f b
    end

    let grads = Rune.grad (module Params) loss params
    ]}

    Use {!Ptree} when the parameter structure is only known at runtime. *)

(** {1:differentiable Ptree.S structures} *)

module Ptree = Nx.Ptree
(** Parameter trees: {!Nx.Ptree.S} is the traversal interface every
    transformation takes, and {!Nx.Ptree.t} the stock dynamic instance. *)

(** {1:reverse Reverse-mode differentiation} *)

val grad : (module P : Ptree.S) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> P.t
(** [grad (module P) f params] is the gradient of [f] at [params], with the same
    structure and leaf types as [params]. Leaves of [params] that do not
    contribute to the result have all-zero gradients.

    Raises [Invalid_argument] if a parameter leaf has an integer or boolean
    dtype (hold non-differentiable data in the closure or the auxiliary output),
    or if [f params] is not a scalar (a tensor with exactly one element); use
    {!vjp} to differentiate non-scalar outputs against an explicit cotangent. *)

val value_and_grad :
  (module P : Ptree.S) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t * P.t
(** [value_and_grad (module P) f params] is
    [(f params, grad (module P) f params)], computed in a single forward and
    backward pass. *)

val value_and_grad_aux :
  (module P : Ptree.S) ->
  (P.t -> ('c, 'd) Nx.t * 'aux) -> P.t -> ('c, 'd) Nx.t * P.t * 'aux
(** [value_and_grad_aux (module P) f params] is like {!value_and_grad} for an
    objective returning auxiliary data alongside its result. The auxiliary value
    is returned as-is and does not contribute to the gradient. *)

val vjp :
  (module P : Ptree.S) ->
  (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t -> ('c, 'd) Nx.t * P.t
(** [vjp (module P) f params cotangent] is [(f params, grads)] where [grads] is
    the vector-Jacobian product of [f] at [params] against [cotangent], with the
    same structure and leaf types as [params]. [cotangent] must have
    [f params]'s shape and dtype. *)

val vjp2 :
  (module P : Ptree.S) ->
  (module Q : Ptree.S) -> (P.t -> Q.t) -> P.t -> Q.t -> Q.t * P.t
(** [vjp2 (module P) (module Q) f params cotangents] is like {!vjp} for an
    objective returning a structured output: [cotangents] provides one cotangent
    per output leaf, each with its output leaf's shape and dtype, and the
    pulled-back cotangents have [params]' structure.

    Raises [Invalid_argument] if a cotangent leaf's shape does not match its
    output leaf's shape. *)

val vjp_fun :
  (module P : Ptree.S) ->
  (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t * (('c, 'd) Nx.t -> P.t)
(** [vjp_fun (module P) f params] is [(f params, pullback)]. [pullback ct] is
    the vector-Jacobian product of [f] at [params] against [ct]; it may be
    called any number of times with different cotangents, each call running one
    backward pass over the recorded computation without re-running [f]. Calling
    the pullback under another transformation (for example {!val-vmap})
    transforms the backward pass. Pullbacks are not thread-safe.

    Raises [Invalid_argument] if [ct]'s shape does not match the output's. *)

val vjp_fun' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * (('c, 'd) Nx.t -> ('a, 'b) Nx.t)
(** [vjp_fun' f x] is like {!vjp_fun} for a function of a single tensor. *)

(** {1:forward Forward-mode differentiation} *)

val jvp :
  (module P : Ptree.S) ->
  (P.t -> ('c, 'd) Nx.t) -> P.t -> P.t -> ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp (module P) f params tangents] is [(f params, df)] where [df] is the
    Jacobian-vector product of [f] at [params] against [tangents], computed in a
    single forward pass. [tangents] must be structurally equal to [params]; each
    tangent leaf must have its parameter leaf's shape. The output may have any
    shape.

    Raises [Invalid_argument] if a tangent leaf's shape does not match its
    parameter leaf's shape. *)

val jvp_aux :
  (module P : Ptree.S) ->
  (P.t -> ('c, 'd) Nx.t * 'aux) ->
  P.t ->
  P.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t * 'aux
(** [jvp_aux (module P) f params tangents] is like {!jvp} for an objective
    returning auxiliary data alongside its result. The auxiliary value is
    returned as-is and does not contribute to the tangent. *)

val jvp2 :
  (module P : Ptree.S) ->
  (module Q : Ptree.S) -> (P.t -> Q.t) -> P.t -> P.t -> Q.t * Q.t
(** [jvp2 (module P) (module Q) f params tangents] is like {!jvp} for an
    objective returning a structured output: the result tangent has the output's
    structure, one tangent per output leaf. *)

(** {1:vmap Vectorizing maps} *)

val vmap :
  ?in_axes:int option list ->
  ?out_axis:int ->
  (module P : Ptree.S) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t
(** [vmap ?in_axes ?out_axis (module P) f params] maps [f] over the tensor
    leaves of [params]. [f] is written for unbatched values: it observes each
    mapped leaf without its mapped axis, and its result gains a batch axis at
    [out_axis] (default [0]). Values [f] closes over are constants of the map,
    and a result that does not depend on the mapped inputs is broadcast along
    the batch axis.

    [in_axes] gives the mapped axis of each leaf, paired with leaves in the
    structure's traversal order: [Some i] maps axis [i] (negative from the end),
    [None] passes the leaf whole as a constant. It defaults to mapping axis [0]
    of every leaf. Mapped axes must agree on their size.

    Composes with the other transformations: [vmap] of {!grad} computes
    per-example gradients, and {!grad} of [vmap] differentiates through the map.

    {b Note.} Implicit random number generation ([Nx.rand] and friends) inside
    the mapped function draws {e identical} values for every lane: the RNG key
    is a constant of the map. Thread distinct randomness in as mapped inputs
    instead. Reading a batched tensor's value inside the mapped function raises.

    Raises [Invalid_argument] if [in_axes] does not have one entry per leaf,
    maps no leaf, names an axis out of bounds, or if the mapped axis sizes
    disagree. *)

val vmap2 :
  ?in_axes:int option list ->
  ?out_axis:int ->
  (module P : Ptree.S) -> (module Q : Ptree.S) -> (P.t -> Q.t) -> P.t -> Q.t
(** [vmap2 ?in_axes ?out_axis (module P) (module Q) f params] is like
    {!val-vmap} for a mapped function returning a structured output: every
    output leaf gains a batch axis at [out_axis], and output leaves that do not
    depend on the mapped inputs are broadcast. *)

val vmap' :
  ?in_axis:int ->
  ?out_axis:int ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [vmap' ?in_axis ?out_axis f x] is like {!val-vmap} for a function of a
    single tensor, mapping over [x]'s axis [in_axis] and placing the batch axis
    of the result at [out_axis]. Both default to [0]. *)

(** {1:custom Custom differentiation rules} *)

val custom_vjp :
  (module P : Ptree.S) ->
  fwd:(P.t -> ('c, 'd) Nx.t * 'res) ->
  bwd:('res -> ('c, 'd) Nx.t -> P.t) ->
  P.t ->
  ('c, 'd) Nx.t
(** [custom_vjp (module P) ~fwd ~bwd params] is [fst (fwd params)], with a
    user-defined reverse rule. Under the innermost reverse-mode transformation,
    [fwd]'s internal operations are not differentiated; instead
    [bwd residual cotangent] provides the parameter gradients, with each leaf
    matching its parameter leaf's shape and dtype. [residual] is whatever [fwd]
    returned alongside its result. Enclosing transformations (an outer {!grad},
    {!val-vmap}) see the forward computation itself.

    Raises [Invalid_argument] if the call is differentiated in forward mode;
    define a {!custom_jvp} rule for that. *)

val custom_jvp :
  (module P : Ptree.S) ->
  f:(P.t -> ('c, 'd) Nx.t) ->
  jvp:(P.t -> P.t -> ('c, 'd) Nx.t * ('c, 'd) Nx.t) ->
  P.t ->
  ('c, 'd) Nx.t
(** [custom_jvp (module P) ~f ~jvp params] is [f params], with a user-defined
    forward rule. Under the innermost forward-mode transformation,
    [jvp params tangents] provides both the result and its tangent, replacing
    [f]'s internal operations.

    Raises [Invalid_argument] if the call is differentiated in reverse mode;
    define a {!custom_vjp} rule for that. *)

(** {1:tensor Single-tensor variants} *)

val grad' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [grad' f x] is like {!grad} for a function of a single tensor. *)

val value_and_grad' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [value_and_grad' f x] is like {!value_and_grad} for a function of a single
    tensor. *)

val vjp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [vjp' f x cotangent] is like {!vjp} for a function of a single tensor. *)

val jvp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp' f x tangent] is like {!jvp} for a function of a single tensor. *)

(** {1:remat Gradient checkpointing} *)

val remat :
  (module P : Ptree.S) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t
(** [remat (module P) f params] is [f params], recomputed during the backward
    pass instead of having its intermediate results retained by the tape:
    reverse-mode differentiation of a [remat]ed function trades compute for
    memory. Gradients are unchanged.

    Raises [Invalid_argument] if differentiated in forward mode (it is a
    {!custom_vjp} rule underneath). *)

(** {1:jacobians Jacobians} *)

val jacfwd' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('c, 'd) Nx.t
(** [jacfwd' f x] is the Jacobian of [f] at [x], with shape
    [shape (f x) @ shape x], computed column by column in forward mode (one
    vectorized pass). Prefer it when the input is smaller than the output. *)

val jacrev' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [jacrev' f x] is the Jacobian of [f] at [x], with shape
    [shape (f x) @ shape x], computed row by row in reverse mode (one forward
    pass, one vectorized backward pass). Prefer it when the output is smaller
    than the input. *)

val hessian' :
  (('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [hessian' f x] is the Hessian of the scalar objective [f] at [x], with shape
    [shape x @ shape x] (forward over reverse). *)

val hvp : (module P : Ptree.S) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> P.t -> P.t
(** [hvp (module P) f params v] is the Hessian-vector product of the scalar
    objective [f] at [params] against [v], with [params]' structure, computed
    without materializing the Hessian (forward over reverse). *)

val hvp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [hvp' f x v] is like {!hvp} for a function of a single tensor. *)

(** {1:checks Gradient checking} *)

val check_grads :
  ?eps:float ->
  ?tol:float ->
  (module P : Ptree.S) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> (unit, string) result
(** [check_grads (module P) f params] compares the reverse-mode gradient of the
    scalar objective [f] at [params] against central-difference directional
    derivatives along deterministic directions. [Ok ()] means they agree within
    [tol] (relative, default [1e-2]); [Error msg] describes the disagreement.
    [eps] is the finite-difference step (default [1e-4]).

    The check is directional, not per-element: it validates gradients cheaply
    rather than exhaustively. Use float64 parameters for reliable results;
    float32 may need a looser [tol]. *)

(** {1:jit Just-in-time compilation} *)

exception Jit_error of string
(** Raised when a function cannot be compiled: it read the value of a traced
    tensor (for example [Nx.item] on a value that depends on the inputs, or a
    data-dependent branch), or it used an operation the compiler does not
    support (FFT, linear algebra, random number generation, complex, int4 and
    uint4 tensors, assigning into a view). *)

val jit :
  ?device:string ->
  (module P : Ptree.S) -> (P.t -> ('c, 'd) Nx.t) -> P.t -> ('c, 'd) Nx.t
(** [jit (module P) f] is [f] compiled. The first application traces [f],
    compiles the traced computation into fused kernels, and runs them; later
    applications with the same leaf signature — dtypes and shapes, in traversal
    order — replay the compiled program on the new leaf values. A new signature
    triggers a fresh trace and compilation.

    [device] selects where the kernels compile and run: ["CPU"] (the default),
    ["CUDA"] (NVIDIA GPUs), or ["METAL"] (macOS only). On the CPU device,
    contiguous inputs and captured tensors are read in place and outputs are
    computed directly into the returned tensors' storage; on other devices, and
    for non-contiguous tensors, data is copied to and from the device on every
    call.

    The compilation cache lives in the partial application [jit (module P) f]:
    apply [jit] once and reuse the returned function. Tensors [f] closes over
    are inputs of the compiled program too, read afresh on every call, so
    mutations between calls behave as they do eagerly.

    Under an enclosing transformation ({!grad}, {!val-vmap}, {!with_debug}, an
    outer [jit]), the wrapped function runs directly so the transformation
    observes its operations: [jit] never changes results, only speed. Compose
    the other way — differentiate {e inside} the jitted function — to compile
    the forward and backward passes together:

    {[
    let train_step =
      Rune.jit2
        (module Params)
        (module Params)
        (fun p ->
          let g = Rune.grad (module Params) loss p in
          Params.map2 (fun w g -> Nx.sub w (Nx.mul_s g lr)) p g)
    ]}

    Whole-tensor in-place updates ([Nx.assign] on a leaf or a captured tensor)
    are replayed by writing the computed value back into the destination.
    Structured values read during tracing must not depend on traced tensors: a
    data-dependent {!cond} or {!while_loop} predicate raises {!Jit_error}.
    Compiled functions are not thread-safe.

    Raises {!Jit_error} when tracing fails ({!exception-Jit_error}), and
    [Invalid_argument] for an unknown or unavailable [device]. *)

val jit2 :
  ?device:string ->
  (module P : Ptree.S) -> (module Q : Ptree.S) -> (P.t -> Q.t) -> P.t -> Q.t
(** [jit2 (module P) (module Q) f] is like {!val-jit} for a function returning a
    structured output. *)

val jit' :
  ?device:string ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [jit' f] is like {!val-jit} for a function of a single tensor. *)

(** {1:flow Control flow}

    Eager combinators with staging-ready signatures: code written with them
    differentiates and vectorizes today, and a future staging [jit] can trace
    them as structured control flow instead of unrolled traces. Today,
    {!val-jit} unrolls {!scan} and rejects data-dependent {!cond} and
    {!while_loop} predicates. *)

val scan :
  (module C : Ptree.S) ->
  f:(C.t -> ('a, 'b) Nx.t -> C.t * ('c, 'd) Nx.t) ->
  init:C.t ->
  ('a, 'b) Nx.t ->
  C.t * ('c, 'd) Nx.t
(** [scan (module C) ~f ~init xs] folds [f] over slices of [xs] along axis 0:
    [f carry x] returns the next carry and a per-step output. The result is the
    final carry and the outputs stacked along a new axis 0. Differentiating
    traces every step.

    Raises [Invalid_argument] if [xs] is a scalar or empty along axis 0. *)

val cond :
  (bool, Nx.bool_elt) Nx.t -> then_:(unit -> 'r) -> else_:(unit -> 'r) -> 'r
(** [cond pred ~then_ ~else_] runs one branch according to the scalar [pred].
    Reading [pred] concretizes it: inside {!val-vmap}, a predicate that depends
    on the mapped inputs raises, since the lanes could diverge. *)

val while_loop :
  (module C : Ptree.S) ->
  cond:(C.t -> (bool, Nx.bool_elt) Nx.t) -> body:(C.t -> C.t) -> C.t -> C.t
(** [while_loop (module C) ~cond ~body init] iterates [body] on the carry while
    [cond] holds. Reading the predicate concretizes it, with the same
    {!val-vmap} caveat as {!cond}. Differentiating traces every iteration
    actually taken. *)

(** {1:debug Debugging} *)

val with_debug : ?ppf:Format.formatter -> (unit -> 'a) -> 'a
(** [with_debug f] runs [f] and logs each tensor operation it performs — the
    operation name and output shape — to [ppf] (defaults to
    [Format.err_formatter]). Composes with the other transformations: run it
    outermost to also observe the operations they emit. Uncommon operations may
    execute unlogged. *)

(** {1:control Autodiff control} *)

val detach : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [detach t] is a copy of [t] through which gradients do not flow. Use it to
    hold a value constant inside a differentiated function, including as input
    to an operation whose gradient is not implemented. *)

val no_grad : (unit -> 'a) -> 'a
(** [no_grad f] runs [f] with gradient tracking disabled: tensors it produces
    are constants of the surrounding differentiation. *)
