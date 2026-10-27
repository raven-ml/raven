(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transformations of functions over nx values: derivatives, loops, totals and
    compilation.

    A transformation takes the structure ({!Nx.Ptree.t}) of each value whose
    arrays it enumerates; everything else the function uses is captured and is a
    constant of the transformation.

    {b Functions of a construct are traced.} The functions that {!scan},
    {!iterate} and {!jit} receive are {e traced} at the call: their OCaml code
    runs over stand-ins of their arguments' arrays, values with a dtype, a shape
    and a placement but no elements, and what it computes from them is recorded
    as a program, which the construct then evaluates. A loop's step is traced
    once, and again where it places a carry array it received of every set
    ({!scan}). So a step's code runs once or a few times whatever the number of
    steps, as do its side effects, and inside it {!Nx.item} of a stand-in, or of
    a value computed from one, raises
    ([Nx.item: the value is traced by Rune.scan]); a branch on a value is
    {!Nx.where} or a nested {!iterate}. An operation that reads no stand-in
    computes at the call, once. A draw without [~key] inside a traced function
    raises, naming the construct, so a step draws from a key in its carry or its
    rows. Outside every transformation a construct is its program evaluated: a
    {!scan} is the fold of its step, bit for bit but for a NaN result's sign and
    payload, which nx leaves unspecified.

    {b Extents.} A transformation, a map or a {!Total.collect} is around the
    code its function runs, on whatever domain that code resumes, and the fibers
    a scheduler installed in the function runs on the domain the function runs
    on. It is not around a systhread or a domain the function starts, nor a
    fiber such a scheduler resumes on another domain: there {!lanes} and
    {!lane_index} raise, and an addition to a total is dropped.

    {b Errors.} Every message starts with the entry point the caller applied, as
    ["Rune.scan"] for {!scan}. *)

(** {1:derivatives Derivatives}

    A differentiation tracks the float and complex arrays of its arguments;
    integer and boolean arrays have no derivative. Every operation nx computes
    has a derivative but a block scale of [Nx.contract] that depends on the
    parameters and the complete factors that {b Factorisations} below names,
    whose tangents raise [Invalid_argument].

    {b Tangents.} A {e tangent} is a displacement: move the parameters by [h]
    times the tangent and the result moves by [h] times its tangent, to first
    order. A complex array is two real parts per element, and its tangent the
    displacement of both, packed as [dre + i dim]: the tangent of a
    complex-differentiable [f] is [f'(z) t], and that of any other is the
    derivative of the real function of the parts, so the tangent of
    [Nx.conjugate] is the conjugate of the tangent.

    {b Cotangents.} A {e cotangent} pairs with a tangent of its array by
    [<w, t> = Re Σ w t], element by element, unconjugated on complex arrays. A
    {e pullback} maps a cotangent [w] of the result to the cotangent [u] of the
    arguments such that [<u, t> = <w, dy>] for every tangent [t] of the
    arguments and its result's tangent [dy]: the transpose of the tangent map.
    So the pullback of a complex-differentiable [f] applies [f'(z)] to [w], and
    the gradient of a real [f] of [z = x + iy] is [∂f/∂x - i ∂f/∂y], against
    whose conjugate a step descends.

    {b Sums.} A cotangent is the sum of the contributions its array receives,
    element by element, each added to those before with no zero before the
    first. An element that receives one contribution through elementwise
    functions, reductions, scans, movements, pads and windows is that
    contribution, sign of zero included: [grad' (fun x -> Nx.sum (Nx.mul x x))]
    at [-0.] is [-0.]. The contributions through a gather, as [Nx.take]'s, and
    the copies of a broadcast are summed from [+0.], as [Nx.sum] sums. An
    element that receives none is [+0.].

    {b Placement.} A cotangent lies where its array does. That of an array of
    every set lies on the set of the first placed argument or result, and of
    every set where every value it meets is.

    {b Kinks.} Where an operation has no derivative: [Nx.abs] passes the tangent
    at [0.] and [-0.]; where the operands of [Nx.maximum] or [Nx.minimum] tie,
    each takes half of each tangent, and where one is a NaN the tangent is zero;
    the tied elements along the axes of [Nx.max] and [Nx.min] share their
    tangents equally, as a target of [Nx.scatter ~combine:Max] or [Min] and the
    updates that tie with it do; a running extreme, as [Nx.cummax]'s, takes the
    tangent of the first element that reaches it; where a log-sum-exp, as
    [Nx.logsumexp]'s or [Nx.logcumsumexp]'s, is an infinity, the terms equal to
    it share its tangent equally; a real [Nx.sign], [Nx.floor], [Nx.ceil],
    [Nx.round] and [Nx.trunc] have a zero tangent, and [Nx.pow x y] has a zero
    tangent along [x] where [y] is zero and along [y] where [x] is zero. A
    function nx composes of others, as [Nx.softplus], takes these rules at its
    parts' kinks.

    {b Factorisations.} The tangents of [Nx.cholesky], [Nx.lu], [Nx.qr],
    [Nx.svd], [Nx.eigh] and [Nx.eig] are the derivatives of the factors nx
    computes, where those are differentiable. [Nx.qr ~factors:Complete] of a
    tall matrix and [Nx.svd ~factors:Complete] of a non-square one have none:
    their tangents raise [Invalid_argument]. The tangents of singular vectors
    and eigenvectors are not finite where two singular values or eigenvalues are
    equal. A vector is defined up to its sign, or a complex one's phase, which
    nx leaves open and the tangent fixes: the tangent of each eigenvector of
    [Nx.eigh] and [Nx.eig] is orthogonal to it, so a vector of [Nx.eig] keeps
    its unit norm, and the tangent of each right singular vector of [Nx.svd] is
    orthogonal to it, the left one carrying the change of phase that keeps
    [a = u diag(s) vh]. A factorisation's values alone, as
    [Nx.svd ~vectors:false]'s, take their tangents from the vectors the full
    routine computes.

    {b Reads.} Outside constructs, {!Nx.item} of a value a differentiation
    tracks reads its primal, so an OCaml [if] on it differentiates the path
    taken. Inside a construct it raises.

    {b Nesting.} Each differentiation tracks the values of its own function
    alone: in
    [jvp' (fun x -> Nx.mul x (snd (jvp' (fun y -> Nx.add x y) one one))) one
     one], the inner tangent is a constant to the outer differentiation, and the
    result's tangent is [1].

    {b Derivatives are fresh.} A tangent is a value of its own, in C order from
    the start of its memory, never a view of an argument or of the movement its
    computation ends with. Under another transformation it is that
    transformation's value. *)

val jvp :
  ('p, 'd) Nx.Ptree.t ->
  ('q, 'd) Nx.Ptree.t ->
  ('p -> 'q) ->
  'p ->
  'p ->
  'q * 'q
(** [jvp p q f params tangents] is [(f params, dy)], [dy] the tangent of
    [f params] where [params] moves along [tangents]: the Jacobian-vector
    product of [f] at [params], computed in one forward pass. [tangents] has
    [params]' structure, dtypes and shapes; [dy] has the result's structure [q],
    one tangent per array, zeros of its dtype and shape for an integer or
    boolean array and for one that does not depend on [params]. The tangent of
    an integer or boolean parameter is ignored.

    Raises [Invalid_argument] if [params] holds no float or complex array
    (["Rune.jvp: the parameters hold no real or complex tensor"]), if [tangents]
    and [params] differ in their {!Nx.Ptree.val-visits}, naming the first path
    where they differ, or if a tangent differs from its parameter in dtype or
    shape (["Rune.jvp: 1: shape [2] in the parameters, [3] in the tangents"]).
*)

(** {1:reverse Reverse mode}

    Reverse mode runs the function once in forward mode, recording the
    operations on tangents, then transposes them from the result's cotangents
    back to the arguments'. The record keeps the values those operations read:
    of an elementwise function computed in one loop, its operands, from which
    the backward pass computes its intermediates again. *)

val grad : ('p, 'd) Nx.Ptree.t -> ('p -> ('v, 's, 'd) Nx.t) -> 'p -> 'p
(** [grad p f params] is the gradient of the real scalar [f params] at [params]:
    the pullback of [1], a value of [params]' structure, dtypes and shapes. Its
    pairing with a tangent of [params] is [f]'s derivative along that tangent,
    as {!jvp} computes it. The gradient of an integer or boolean array, and of
    one [f] does not depend on, is zero.

    Raises [Invalid_argument] as {!jvp} does for [params], and if [f params] is
    not one float element, of any shape, naming its dtype and shape
    (["Rune.grad: the objective must return a real scalar, got complex128 []"]);
    {!vjp} differentiates any other result. *)

val value_and_grad :
  ('p, 'd) Nx.Ptree.t ->
  ('p -> ('v, 's, 'd) Nx.t) ->
  'p ->
  ('v, 's, 'd) Nx.t * 'p
(** [value_and_grad p f params] is [(f params, grad p f params)], from one
    forward and one backward pass. It raises as {!grad} does. *)

val value_and_grad_aux :
  ('p, 'd) Nx.Ptree.t ->
  ('x, 'd) Nx.Ptree.t ->
  ('p -> ('v, 's, 'd) Nx.t * 'x) ->
  'p ->
  ('v, 's, 'd) Nx.t * 'p * 'x
(** [value_and_grad_aux p x f params] is [(y, g, aux)], where
    [(y, aux) = f params] and [g] is the gradient of [y] at [params]. [aux], of
    structure [x], is a value of [f] that contributes nothing to [g]. It raises
    as {!grad} does. *)

val vjp :
  ('p, 'd) Nx.Ptree.t ->
  ('q, 'd) Nx.Ptree.t ->
  ('p -> 'q) ->
  'p ->
  'q * ('q -> 'p)
(** [vjp p q f params] is [(f params, pullback)], [pullback] mapping the
    result's cotangents to [params]' ({!section-derivatives}). Its argument has
    the result's structure [q], one cotangent per array, of that array's dtype
    and shape, and the cotangent of an integer or boolean array is ignored.
    [pullback] runs no part of [f] again. When [vjp] runs outside every
    transformation, [pullback] may be applied any number of times, from any
    domain.

    Raises [Invalid_argument] as {!jvp} does for [params]. [pullback] raises
    [Invalid_argument] if its argument and the result differ in their
    {!Nx.Ptree.val-visits}, dtypes or shapes, naming the first path where they
    differ (["Rune.vjp: 0: shape [2] in the result, [3] in the cotangents"]). *)

val check_grads :
  ?eps:float ->
  ?tol:float ->
  ('p, 'd) Nx.Ptree.t ->
  ('p -> ('v, 's, 'd) Nx.t) ->
  'p ->
  (unit, string) result
(** [check_grads p f params] compares the gradient [g] of the real scalar [f] at
    [params] with central differences of step [eps] (default [1e-4]) along two
    fixed directions [v], ones and [sin params + 1.1] on the float and complex
    arrays: it is [Ok ()] if [Re Σ g v], the derivative [g] predicts, is within
    [tol] (default [1e-2]) of each difference, relative to its magnitude where
    above [1], and [Error msg] naming the direction and both derivatives
    otherwise. It reads [f]'s result, so [f] must not be traced. float64
    parameters give reliable differences.

    Raises [Invalid_argument] as {!grad} does. *)

(** {1:control Controlling differentiation} *)

val detach : ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [detach x] is [x] with a zero derivative under every differentiation around
    the call: its tangent is zero. Outside every transformation it is [x]
    itself. *)

(** {1:maps Maps}

    A map's lanes are named by an axis, so that code inside nested maps reads
    the lanes of the map it names. *)

type axis
(** The type for names of maps. *)

val axis : unit -> axis
(** [axis ()] is a fresh name, distinct from every other. *)

val lanes : axis -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [lanes a x], inside the map named [a] of [n] lanes, is every lane's [x]
    stacked on a new leading axis of length [n].

    Raises [Invalid_argument] with no map named [a] around the call
    (["Rune.lanes: no map named by the axis is around the call"]). *)

val lane_index : axis -> unit -> 'd Nx.int32_t
(** [lane_index a ()], inside the map named [a], is the calling lane's index in
    it, a 0-d [int32] from [0] to the map's number of lanes minus one.

    Raises [Invalid_argument] with no map named [a] around the call. *)

(** {1:totals Totals} *)

(** Write-only sums.

    A total is a sum that code anywhere inside a function adds to and that the
    caller reads when the function returns: {!Total.collect}[ t ~zero f] is
    [f ()] with [zero] plus the additions to [t] that reach it. Nothing reads a
    total before its [collect] returns, so an addition never changes a value the
    function computes, and with no [collect] of [t] open it does nothing.

    An addition reaches the innermost open [collect] of its total around the
    call, once per execution of the code that makes it. The additions a traced
    function makes reach it when the construct that traced the function returns,
    so a construct that raises adds none: a {!jit} adds its function's additions
    once per call, as the call returns. A step of {!scan} or {!iterate}, and an
    [until] once per test, makes its additions once per step: a loop chains each
    total's additions in the order they are made, from an identity that changes
    no sum (a negative zero), and that chain reaches the [collect] as one
    addition, as the loop returns. So the total is the one the steps' additions
    give made in turn where the [zero] is [0.] or [-0.] and the loop's are the
    first additions the [collect] receives. A loop that takes no step adds that
    identity. *)
module Total : sig
  type ('v, 's, 'd) t
  (** The type for totals of [('v, 's, 'd) Nx.t] values. *)

  val make : unit -> ('v, 's, 'd) t
  (** [make ()] is a fresh total, distinct from every other. *)

  val add : ('v, 's, 'd) t -> ('v, 's, 'd) Nx.t -> unit
  (** [add t v] adds [v] to the innermost open {!collect} of [t], and does
      nothing if none is open.

      Raises [Invalid_argument] if [v]'s shape is not the [zero]'s of the
      [collect] it reaches. *)

  val collect :
    ('v, 's, 'd) t ->
    zero:('v, 's, 'd) Nx.t ->
    (unit -> 'r) ->
    'r * ('v, 's, 'd) Nx.t
  (** [collect t ~zero f] is [(f (), total)], [total] being [zero] plus each
      addition to [t] that reaches it, in the order they reach it. If [f]
      raises, [collect] raises the same exception. *)
end

(** {1:loops Loops} *)

val scan :
  ('c, 'd) Nx.Ptree.t ->
  ('x, 'd) Nx.Ptree.t ->
  ('y, 'd) Nx.Ptree.t ->
  f:('c -> 'x -> 'c * 'y) ->
  init:'c ->
  'x ->
  'c * 'y
(** [scan c x y ~f ~init xs] folds the step [f] over the rows of [xs], a value
    of structure [x]: every array of [xs] has the same leading length [n], and
    step [i] passes [f] row [i] of every array. [f carry row] returns the next
    carry, of structure [c], and the step's outputs, of structure [y]; the
    result is the final carry and the outputs, each array stacked along a new
    axis 0. With [n = 0] the result is [init] and outputs of leading length [0].

    [f] is traced at the call over stand-ins of [init]'s arrays and of [xs]'s
    arrays without axis 0. It is traced again while it returns placed an array
    of the carry that it received of every set, such as a constant [init] it
    adds a placed value to, each array at most once.

    Raises [Invalid_argument] if [xs] has no array
    (["Rune.scan: xs has no leaf"]), a 0-d array, or arrays of different leading
    lengths, naming their paths (["Rune.scan: xs: 1: 3 rows, 0: 2"]), and, at
    the trace, if the carry [f] returns differs from the one it received in its
    {!Nx.Ptree.val-visits}, dtypes or shapes, naming the first path where they
    differ, as in
    ["Rune.scan: 0: shape [3] in the carry the step returned, [2] in the carry
     it received"]. *)

val iterate :
  ('c, 'd) Nx.Ptree.t ->
  max:int ->
  until:('c -> 'd Nx.bool_t) ->
  f:('c -> 'c) ->
  'c ->
  'c
(** [iterate c ~max ~until ~f init] applies the step [f] to a carry of structure
    [c], from [init], until [until carry] holds, and returns that carry. [until]
    returns one boolean and is tested before each step, so an [init] that
    satisfies it is returned unchanged. [f] and [until] are traced at the call,
    [f] as {!scan}'s step is.

    Raises [Invalid_argument] if [max < 0]
    (["Rune.iterate: max = -1 is negative"]), if [until] returns other than one
    element, at the trace if [f]'s carry differs from the one it received as
    {!scan} says, and if [until] is still false after [max] steps
    (["Rune.iterate: until is still false after max = 50 steps"]). *)

(** {1:jit Compilation} *)

val jit : ('a -> 'b, 'd) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [jit s f] is [f] traced into a program, [s] its signature, one structure per
    argument and one for the result. Each call traces [f] over stand-ins of its
    arguments' arrays and evaluates the program. Every operation [f] applies is
    staged, those over values [f] captures included.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument, and at
    a call if [f] captures a value that another transformation traces. *)

(** {1:one Functions of one array} *)

val jvp' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t * ('w, 'r, 'd) Nx.t
(** [jvp' f x t] is {!jvp} of one array. It raises as {!jvp} does. *)

val jacfwd' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t
(** [jacfwd' f x] is the Jacobian of [f] at [x], of shape
    [shape (f x) @ shape x] and [f x]'s dtype: its column [k] is the tangent of
    [f] along element [k] of [x] in C order, computed in one forward pass per
    element.

    Raises [Invalid_argument] as {!jvp'} does. *)

val grad' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [grad' f x] is {!grad} of one array. It raises as {!grad} does. *)

val value_and_grad' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t * ('v, 's, 'd) Nx.t
(** [value_and_grad' f x] is {!value_and_grad} of one array. *)

val vjp' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t * (('w, 'r, 'd) Nx.t -> ('v, 's, 'd) Nx.t)
(** [vjp' f x] is {!vjp} of one array. *)

val jacrev' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [jacrev' f x] is the Jacobian of [f] at [x], of shape
    [shape (f x) @ shape x] and [x]'s dtype: its row [k] is the pullback of
    element [k] of the result in C order, computed in one backward pass per
    element. For a complex-differentiable [f] it is the complex derivative, as
    {!jacfwd'} computes it.

    Raises [Invalid_argument] as {!vjp'} does, and if [x] is real and [f x]
    complex, whose Jacobian [x]'s dtype cannot hold. *)

val scan' :
  f:
    (('a, 'b, 'd) Nx.t ->
    ('c, 'e, 'd) Nx.t ->
    ('a, 'b, 'd) Nx.t * ('g, 'h, 'd) Nx.t) ->
  init:('a, 'b, 'd) Nx.t ->
  ('c, 'e, 'd) Nx.t ->
  ('a, 'b, 'd) Nx.t * ('g, 'h, 'd) Nx.t
(** [scan' ~f ~init xs] is {!scan} of one array each for the carry, the rows and
    the outputs. *)

val iterate' :
  max:int ->
  until:(('v, 's, 'd) Nx.t -> 'd Nx.bool_t) ->
  f:(('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [iterate' ~max ~until ~f x] is {!iterate} of one array. *)

val jit' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t
(** [jit' f] is {!jit} of a function of one array. *)
