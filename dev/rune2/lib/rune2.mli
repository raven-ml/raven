(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Derivatives, maps, loops and compilation of functions over nx values.

    rune2 transforms OCaml functions whose arguments and results hold nx arrays.
    A {e structure} ({!Nx.Ptree.t}) enumerates the arrays of a value, so each
    transformation and construct takes one for each value whose arrays it reads,
    and a {e signature} ({!Nx.Ptree.fn}) is one structure per argument of a
    function and one for its result. A value a function uses without receiving
    it is {e captured}. An array is {e fresh} when it has memory of its own in C
    order, from the start of its buffer, and is no view of another.

    A {e transformation} ({!jvp}, {!grad}, {!vjp}, {!linearize}, {!vmap} and the
    functions built on them, {!check_grads} aside) runs its function once per
    call, over values that carry more than their elements. Under {!jvp} each
    value carries a tangent ({!section-tangents}), and under {!vmap} one row per
    lane of a map ({!section-maps}). Code is {e around} a call when the call
    runs inside it, as the function of a transformation runs inside the
    transformation. Transformations nest ({!section-derivatives}).

    A {e construct} ({!scan}, {!iterate}, {!switch}, {!root}, {!remat} and
    {!jit}) {e traces} its functions. Their OCaml code runs over {e stand-ins},
    values with a dtype, a shape and a placement but no elements. What they
    compute from them is recorded as a {e program}, which the construct then
    runs ({!section-tracing}). {!linear_transpose}, {!program} and {!eval_shape}
    trace their function too. A transformation of a loop, a {!switch}, a
    {!remat} or a {!jit} is that construct over the transformation of its
    program: the derivative of a {!scan} is a scan of its step's derivative. A
    custom rule and a {!root} have the derivatives their sections state.

    Two arrays agree {e bit for bit} when their elements have the same bits,
    except for a NaN's sign and payload, which nx leaves unspecified. Outside
    every transformation a construct other than {!jit} agrees bit for bit with
    its functions applied eagerly: a {!scan} is the fold of its step. A {!jit}
    computes to the accuracy {!section-numerics} states. A value rune2 makes
    lies where nx places it ({!Nx.section-routing}): a zero beside an array lies
    at that array's placement, and an operation's result where nx routes it.

    The life cycle of a compiled function goes like this:
    + [let g = jit s f] makes a compiled function with no program. [g] must be
      bound once: [jit s f x] written at each call makes a new [g] and traces
      every call.
    + The first call of [g] with a new key ({!section-keys}), the dtypes, shapes
      and placements of its arguments among others, traces [f] into a program.
    + The first call that runs the program compiles it for the call's variant
      ({!section-compiled}), the layout of its arguments in memory among others.
    + Later calls with that key and variant run the compiled program.
    + A transformation around a call derives programs from its key's, once per
      derivation ({!section-derived}), and compiles them as it does the key's.

    Unless a function's doc says otherwise:
    - any domain may call it, concurrently with other domains;
    - misuse raises [Invalid_argument] with a message that starts with the entry
      point, at a moment {!section-errors} names;
    - an exception the function it transforms raises passes through it
      unchanged;
    - a tangent, a gradient and a pullback's result are fresh, and under another
      transformation they are that transformation's values;
    - integer and boolean arrays have no derivative, so a derivative is zero
      there.

    Here [loss] is traced once for the key of [w], and {!grad'} derives and
    compiles its reverse programs from that trace:

    {[
    let loss = Rune2.jit' (fun w -> Nx.sum Nx.Infix.(w * w))
    let g = Rune2.grad' loss (Nx.create Nx.float64 [| 3 |] [| 1.; 2.; 3. |])
    ]} *)

(** {1:tracing Tracing}

    A construct other than {!jit} traces its functions each time it is called:
    at each call outside a {!jit}, and inside one at each trace of the {!jit}. A
    custom rule's functions run on values where {!section-custom} says so, and
    are traced elsewhere. A {!jit} itself traces once per key. A loop's step is
    traced once per trace of the loop, whatever its number of steps, so its code
    and its side effects run once per trace too. Inside a traced function
    {!Nx.item} of a stand-in, or of a value computed from one, raises, naming
    the construct, as [Nx.item: the value is traced by Rune.scan] in a {!scan}'s
    step, so a branch on a value is a {!switch}, {!Nx.where} or a nested
    {!iterate}. A draw without [~key] raises, naming the construct, so a step
    draws from a key in its carry or its rows.

    An operation that reads no stand-in computes once, at the call, except in
    {!switch}, {!remat}, a {!root}'s [solve] and {!jit}. These {e stage} every
    operation their functions apply: they record it in the program, operations
    on captured values included. A traced function reads what it captures when
    it is traced, so it sees a [ref]'s contents at the trace: under {!jit}, at
    the first call of each key.

    {b Extents.} A transformation, a {!Total.collect} or a {!Probe.collect} is
    around all code its function runs, on whichever domain that code resumes,
    including the fibers of a scheduler the function installs while they run on
    its domain. It is not around a systhread or a domain the function starts,
    nor around a fiber such a scheduler resumes on another domain: there
    {!lanes} and {!lane_index} raise, and an addition to a
    {{!section-totals}total} and an emission to a {{!section-probes}probe} are
    dropped. *)

(** {1:errors Errors}

    Every message starts with the entry point the caller applied, under the
    module name [Rune], as ["Rune.scan"] for {!scan} and ["Rune.scan'"] for
    {!scan'}. Misuse of an entry point raises [Invalid_argument] at one of four
    moments:
    - when a signature is applied, as [jit s], [vmap s] or [remat s] with an [s]
      of no argument;
    - at the call, for arguments that do not fit the entry point;
    - at the trace, for a traced function's misuse, as a step's carry that
      changes shape ({!section-tracing} says when each construct traces);
    - when a function rune2 hands out is applied after its call, as {!vjp}'s
      pullback, {!linearize}'s tangent map or the [op] a {!root} passes to its
      [linear_solve].

    A failure that depends on the values, as an {!iterate} still running after
    [max] steps or a {!switch} index out of range, raises at the call, and under
    a {!jit} as the call returns, as a check does ({!section-checks}). A later
    operation on an argument a {!jit} consumed raises in nx, naming the entry
    point that consumed it ({!section-donation}). A {!jit} whose trace raises
    keeps no program for the key. *)

(** {1:derivatives Derivatives}

    A {e differentiation} ({!jvp}, {!grad}, {!vjp}, {!linearize} and the
    functions built on them) {e tracks} the float and complex arrays of its
    arguments and the values computed from them. A tracked value's {e primal} is
    its array without the derivative it carries. Every operation nx computes has
    a derivative, except a block scale of {!Nx.contract} that depends on the
    parameters and the complete factors that {!section-kinks} names, whose
    tangents raise [Invalid_argument].

    {b Reads.} {!Nx.item} of a value a differentiation tracks reads its primal,
    so an OCaml [if] on it differentiates the path taken. A traced function that
    captures the value reads it so too, and takes that path wherever its program
    runs: a loop's step at every step. A custom rule may read so a value its
    differentiation tracks: a read is not one of the uses {!section-custom}
    refuses. A pullback that runs in the backward pass may not read such a value
    ({!section-custom}). Where a construct that stages ({!section-tracing}) lies
    between the read and the differentiation, it stages the read, which raises.

    {b Nesting.} Each differentiation tracks the values of its own function
    alone: in
    [jvp' (fun x -> Nx.mul x (snd (jvp' (fun y -> Nx.add x y) one one))) one
     one], the inner tangent is a constant to the outer differentiation, and the
    result's tangent is [1]. *)

(** {2:tangents Tangents and cotangents}

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
    element that receives none is [+0.]. A {!scan}'s or an {!iterate}'s carry
    holds a derivative at every step: where the call or a step gives a carry
    array none, the next step takes a zero from it, so the derivative may
    receive zeros of either sign, and NaN where such a zero meets an infinite
    partial derivative, where the steps written out one after another give
    none.

    {b Selections.} Under reverse mode the cotangent of the operand {!Nx.where}
    does not select is [+0.] there; where that operand's derivative is infinite
    or NaN, as {!Nx.log}'s at [0.], the gradient is NaN, [0 × ∞]. Compute such
    an operand on its domain, as [Nx.log (Nx.where c x one)], or as
    [Nx.where c x (detach x)] on its input, or branch with {!cond}, per element
    under {!val-vmap}.

    {b Placement.} A cotangent lies where the transposed operations put it, as
    nx routes them ({!Nx.section-routing}): at its array's placement where every
    value it meets lies there, and otherwise at the union of their cuts. A
    {!Nx.place}'s cotangent is moved back to its operand's placement. So
    [grad' (fun x -> Nx.sum (Nx.mul x w)) x] lies cut along axis 0 where [x] is
    whole on each device of a set and [w] is cut along axis 0 across them. *)

(** {2:kinks Kinks and factorisations}

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
    routine computes. *)

(** {2:forward Forward mode} *)

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

(** {1:special Derivatives of nx's special functions}

    rune2 differentiates {!Nx.erfinv}, {!Nx.erfc}, {!Nx.ndtr}, {!Nx.log_ndtr},
    {!Nx.ndtri}, {!Nx.lgamma}, {!Nx.digamma}, {!Nx.lbeta}, {!Nx.i0e}, {!Nx.i1e},
    the incomplete gamma family ({!Nx.gammainc}, {!Nx.gammaincc}, their
    logarithms and their inverses) and the incomplete beta family
    ({!Nx.betainc}, {!Nx.betaincc} and their logarithms) through the operations
    nx computes them with, in every argument and at every order. Each derivative
    is within the budget below of the exact derivative, at [float32] and
    [float64], eagerly and compiled, where the function and its derivative are
    finite, which leaves out such points as {!Nx.erfinv} at [±1] and {!Nx.ndtri}
    at [0] and [1]. At [±0], {!Nx.i0e}'s derivative is its right-hand one, [-1];
    {!Nx.i1e}'s is [1/2]. [ε] is the distance from 1 to the next float of the
    dtype. The budgets of the first derivatives are:

    - {!Nx.erfinv}: 64 ulps.
    - {!Nx.erfc}: 128 ulps.
    - {!Nx.ndtr}: 256 ulps.
    - {!Nx.log_ndtr}: 512 ulps.
    - {!Nx.ndtri}: 64 ulps.
    - {!Nx.lgamma}: for [x > 0], 256 ulps, or [256ε] absolutely where below 1;
      below 0, 256 ulps or [256ε (1 + |ψ(1 - x)| + |π cot (πx)|)] absolutely.
    - {!Nx.digamma}: for [x > 0], 256 ulps; below 0, 256 ulps or
      [256ε (1 + ψ'(1 - x) + π² / sin² (πx))] absolutely.
    - {!Nx.lbeta}, in either argument: 4096 ulps, or [8192ε] absolutely where
      below 1.
    - {!Nx.i0e}: 128 ulps.
    - {!Nx.i1e}: 128 ulps, or [128ε] absolutely for [1 ≤ |x| ≤ 2], about its
      zero at [±1.545].
    - {!Nx.gammainc}, {!Nx.gammaincc}, in [x]: [512 (1 + |log f|) + 64] ulps.
    - {!Nx.log_gammainc}, {!Nx.log_gammaincc}, in [x]: 256 ulps, or [256ε]
      absolutely where below 1.
    - The four, in [a]: [2^-40 (1 + |log f|)] relative at [float64],
      [2^-16 (1 + |log f|)] at [float32].
    - {!Nx.gammaincinv}, {!Nx.gammainccinv}, in [p] or [q]:
      [64 + 16κ (32 (1 + |log p|) + 4)] ulps, [q] in place of [p] for
      {!Nx.gammainccinv}.
    - The two, in [a]: [2^-40] relative at [float64], [2^-16] at [float32].
    - {!Nx.betainc}, {!Nx.betaincc}, in [x]: [1024 (1 + |log f|) + 64] ulps.
    - {!Nx.log_betainc}, {!Nx.log_betaincc}, in [x]: 512 ulps, or [512ε]
      absolutely where below 1.
    - The four, in [a] or [b]: [2^-40 (1 + |log f|)] relative at [float64],
      [2^-16 (1 + |log f|)] at [float32].

    [f] is the smaller of [P] and [Q], or of [I_x(a, b)] and its complement: a
    tail and its complement share one derivative up to sign, computed from the
    smaller tail. [κ] is the inverse's condition number. The incomplete gamma's
    budgets hold for [0 < a ≤ 2^20]: those of {!Nx.gammainc} and {!Nx.gammaincc}
    where [f] is a normal number, below which the derivative underflows, and an
    inverse's where its quantile is normal. The incomplete beta's hold for
    [0 < a, b ≤ 2^20] where the function is a normal number. At [x = 0],
    [∂P/∂a = 0] and [∂P/∂x] is its one-sided limit [x^{a-1} e^{-x} / Γ(a)]: [1]
    at [a = 1], [0] above and infinite below. At [a = 0], where
    [P = 1 - a E₁(x) + O(a²)] for [x > 0], [∂P/∂a] is its one-sided limit
    [-E₁(x)], [E₁(x) = ∫ₓ^∞ e^{-t}/t dt], and [∂P/∂x = 0]; there [log Q] is
    [-inf]. At an end of [x] the incomplete beta's derivatives in [a] and [b]
    are [0], and in [x] its one-sided limit [x^{a-1} (1 - x)^{b-1} / B(a, b)]:
    [b] at [x = 0] for [a = 1], [0] above and infinite below; at [x = 1]
    likewise with [a] and [b] exchanged. At [a = 0] its upper tail vanishes as
    [a g], [g = ∫ₓ¹ (1 - t)^{b-1} / t dt], and at [b = 0] its lower tail as
    [b h], [h = ∫₀ˣ t^{a-1} / (1 - t) dt]: for [x] inside (0, 1) its derivatives
    there are those of these forms, within [2^-40] relative at [float64] and
    [2^-16] at [float32], except the vanishing tail's logarithm in [a] or [b],
    which is infinite. An inverse's derivative in [a] is the implicit one,
    [-(∂P/∂a) / (∂P/∂x)] at the quantile, for either tail; at [a = 0], where the
    quantile is [0] away from its upper end, both its derivatives are [0].

    Second derivatives of {!Nx.lgamma} and {!Nx.lbeta} are within [2^-40]
    relative at [float64] and [2^-16] at [float32]. Those of {!Nx.lbeta} hold
    where both arguments are at least the cube root of the largest float's
    reciprocal ([1.4e-13] at [float32], [1.8e-103] at [float64]); below, the
    terms they subtract pass the largest float and the result may be NaN. *)

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
    [pullback] runs no part of [f] again, except the pullbacks of the
    {!custom_vjp} rules [f] calls on values ({!section-custom}), each at every
    application, on the domain applying it. When [vjp] runs outside every
    transformation, [pullback] may be applied any number of times, from any
    domain.

    Raises [Invalid_argument] as {!jvp} does for [params]. [pullback] raises
    [Invalid_argument] if its argument and the result differ in their
    {!Nx.Ptree.val-visits}, dtypes or shapes, naming the first path where they
    differ (["Rune.vjp: 0: shape [2] in the result, [3] in the cotangents"]). *)

val linearize :
  ('p, 'd) Nx.Ptree.t ->
  ('q, 'd) Nx.Ptree.t ->
  ('p -> 'q) ->
  'p ->
  'q * ('p -> 'q)
(** [linearize p q f params] is [(f params, jvp_at)], [jvp_at] mapping a tangent
    of [params] to the result's tangent, as {!jvp} computes it at [params]: [f]
    runs once, in forward mode, recording the operations on tangents, and
    [jvp_at] evaluates that record, as often as it is applied, without running
    [f] again. Its argument has [params]' structure, dtypes and shapes; the
    tangent of an integer or boolean parameter is ignored. When [linearize] runs
    outside every transformation, [jvp_at] may be applied any number of times,
    from any domain.

    Raises [Invalid_argument] as {!jvp} does for [params], and [jvp_at] as
    {!jvp} does for its tangents. *)

val linear_transpose :
  ('p, 'd) Nx.Ptree.t -> ('q, 'd) Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'q -> 'p
(** [linear_transpose p q f params] is the transpose of [f], a function linear
    in its argument: the function that maps a cotangent [w] of the result to [u]
    such that [<u, v> = <w, f v>] for every [v] ({!section-derivatives}).
    [params] gives the argument's structure, dtypes, shapes and placements; no
    element of it is read. [f] runs once, at the call, over stand-ins of its
    arrays, so a value [f] uses as a constant is one it captures. A term [f]
    adds that does not depend on its argument, as an affine function has, is
    dropped. The cotangent of an integer or boolean array is zero.

    An {!iterate} over the argument is the steps it takes, transposed, where its
    stop reads no value computed from the argument, so that the number of steps
    does not depend on it.

    Raises [Invalid_argument] at the call for an {!iterate} whose stop reads a
    value computed from the argument
    (["Rune.linear_transpose: an iterate's stop reads the function's argument"]),
    and as {!iterate} raises at the call. When the transpose is applied, it
    raises as {!vjp}'s pullback does for its argument, and naming an operation
    of [f] that is not linear in its argument, as in
    ["Rune.linear_transpose: Sin is not linear in the function's argument"]. *)

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

(** {1:custom Custom differentiation rules}

    A custom rule gives a function the derivative a differentiation uses for it,
    in the form of that differentiation's answer: a tangent map for forward
    mode, a pullback for reverse mode, or a linear function's transpose.

    A value {e belongs} to a differentiation that tracks it, to a map whose
    lanes it holds, and, as a stand-in or a value computed from one, to the
    function traced over it.

    {b Values.} A {!custom_jvp} or {!custom_vjp} runs its rule {e on values},
    its arguments' or their primals', in two cases, so the rule may read them
    with {!Nx.item}, as an iteration tests its tolerance. Where no argument
    belongs to any code and no construct stages the call ({!jit}, {!remat},
    {!switch}, a {!root}'s [solve]), the call is [fst (rule args)] and the
    derivative never runs. Under one differentiation that tracks an argument,
    where the arguments' primals belong to no other code and no transformation
    or traced function lies between the differentiation and the call, the rule
    runs on the primals and the derivative runs where the differentiation
    applies it. The rule's code runs once per call. Without a differentiation
    the derivative's closure never runs, so work only the derivative needs goes
    in it.

    {b Derivatives on values.} A tangent map runs at the call, once per
    differentiation that applies it, on the tangents: values under {!jvp}, so it
    may read them too. Under {!grad} reverse mode traces it to transpose it, so
    its tangents are stand-ins and a read of one raises. A pullback runs in each
    backward pass that applies it, on that pass's cotangents, so a {!vjp}
    pullback applied twice runs it twice. The cotangents are values where the
    backward pass runs outside every transformation, so the pullback may read
    them. Where a differentiation tracks them, as a {!grad} of a {!vjp}
    pullback, a read gives their primals. Where a construct stages the read, or
    the cotangents belong to a map or a traced function, as in {!jacrev'}'s map
    of the pullback, a read raises. The backward pass runs after the
    differentiation has returned, so such a pullback must neither use nor read a
    value that differentiation tracks. The rule may read such a value, and the
    pullback may use what it read.

    {b Traced rules.} Elsewhere the rule is traced at the call with its
    derivative. That is the case inside a construct that stages the call, and
    where an argument belongs to a map, a traced function or a second
    differentiation. It is the case too where a transformation or a traced
    function lies between the call and the differentiation that tracks an
    argument: an inner {!grad} over other values, or a map or a loop's step that
    captures the argument. There a read of an argument raises, naming the entry
    point, as [Nx.item: the value is traced by Rune.custom_jvp]. The
    derivative's code runs once per trace, whether a differentiation applies it
    or not, so a tangent map that calls {!lanes} of a map it does not run in
    raises at every trace.

    {b Errors.} A draw without [~key] in the rule or its derivative raises
    wherever it runs. An error the derivative raises reaches the code that runs
    it: the call for a tangent map and a traced pullback, the backward pass for
    a pullback on values.

    {b Arguments.} A differentiation applies the rule where it tracks one of the
    rule's arguments. The rule receives them without their derivatives, and must
    not use another value that differentiation tracks: pass such a value as an
    argument. A read of one with {!Nx.item} is not a use: it reads the primal,
    as {!section-derivatives} states under Reads. A differentiation that tracks
    only values the rule captures differentiates the rule's code, as it does any
    other code.

    {b Orders.} A differentiation around another differentiates the derivative's
    code, with the arrays it reads, which the rule computes: the second
    derivative of a {!custom_jvp} is the derivative of its tangent map along the
    arguments, and that of a {!custom_vjp} the derivative of its rule and
    pullback. A {!custom_jvp} or {!custom_vjp} called in a tangent map on its
    tangents is, under reverse mode, its function: reverse mode records the
    function's operations, and a differentiation around reverse mode
    differentiates their code. Around forward mode, a differentiation applies
    that rule.

    {b Reverse mode} transposes a tangent map's operations on tangents, so a
    tangent map must be linear in its tangents: one applying [Nx.exp] to a
    tangent raises, naming the entry point, when the derivative is transposed
    (in {!grad}, or as a {!vjp} pullback is applied), as in
    ["Rune.grad: a custom_jvp tangent map applies Exp to a tangent; a tangent
     map must be linear in its tangents"]. An {!iterate} in a tangent map whose
    stop reads a tangent takes steps that depend on it, and raises at the call
    of {!grad}, {!vjp} or {!linearize}, as in
    ["Rune.grad: a custom_jvp tangent map applies an iterate's stop to a
     tangent; a tangent map must be linear in its tangents"]. Reverse mode
    transposes the linear part: a value a tangent map adds to a tangent, or
    selects beside one with {!Nx.where}, counts as zero. A pullback or a
    transpose reads a result's cotangent at the result's set.

    {b Maps.} {!vmap} batches a rule over an argument that belongs to the map:
    the rule and its derivative are traced and evaluated over the lanes, and a
    differentiation around the map applies the rule to every lane. A rule whose
    arguments belong to no code runs on them inside the map, which batches the
    operations it applies to the lanes it captures. One over an argument that a
    differentiation outside the map tracks is traced, as {b Traced rules} says.

    {b Totals.} The additions a rule makes reach the innermost open
    {!Total.collect} of their total around the call, once per call. A tangent
    map's are made once per differentiation that applies it, none without one; a
    pullback's once per backward pass that applies it, to the {!Total.collect}
    around that pass. Under reverse mode a tangent map that adds a tangent to a
    total raises at the call. *)

val custom_jvp :
  ('p, 'd) Nx.Ptree.t ->
  ('q, 'd) Nx.Ptree.t ->
  ('p -> 'q * ('p -> 'q)) ->
  'p ->
  'q
(** [custom_jvp p q rule args] is [fst (rule args)], a value of structure [q],
    whose derivative is the tangent map [map = snd (rule args)]: [map dargs] is
    the result's tangent for the arguments' tangents [dargs], of structure [p],
    zeros for an array the differentiation does not track. Forward mode applies
    [map] and reverse mode transposes it, at every order: the first derivative
    of [softplus] below is [Nx.sigmoid x] whatever [stable] computes.

    {[
    let stable x = Nx.Infix.(Nx.maximum_s x 0. + Nx.log1p (Nx.exp (-Nx.abs x)))

    let softplus =
      Rune2.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          (stable x, fun dx -> Nx.Infix.(Nx.sigmoid x * dx)))
    ]}

    Where the result has no float or complex array, reverse mode does not apply
    [map], so such a rule observes the tangents under forward mode alone.

    Raises [Invalid_argument] at the call where [map] runs or is traced
    ({!section-custom}) if its result differs from the result in its
    {!Nx.Ptree.val-visits}, dtypes or shapes, naming the first path where they
    differ, as in
    ["Rune.custom_jvp: the root: shape [] in the tangent map's result, [3] in
     the result"], and if [rule] uses a value that a differentiation applying it
    tracks
    (["Rune.custom_jvp: the rule uses a value its own differentiation tracks;
      pass it as an argument"]). *)

val custom_vjp :
  ('p, 'd) Nx.Ptree.t ->
  ('q, 'd) Nx.Ptree.t ->
  ('p -> 'q * ('q -> 'p)) ->
  'p ->
  'q
(** [custom_vjp p q rule args] is [fst (rule args)], a value of structure [q],
    whose reverse-mode derivative is the pullback [pb = snd (rule args)]:
    [pb cts] is the arguments' cotangents, of structure [p], for the result's
    cotangents [cts], as {!vjp}'s pullback computes them
    ({!section-derivatives}). [cts] holds zeros for an array of the result that
    nothing the differentiation tracks depends on.

    {[
    let clip_grad c =
      Rune2.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          (x, fun g -> Nx.clamp ~min:(-.c) ~max:c g))
    ]}

    Every reverse mode tracking one of [args] applies [pb]. Forward mode has no
    rule for the result: around a reverse mode that applies [pb], as in a
    Hessian-vector product, it differentiates the code of [rule] and [pb].

    Raises [Invalid_argument] at the call where [pb] is traced
    ({!section-custom}) if its result differs from [args] in its
    {!Nx.Ptree.val-visits}, dtypes or shapes, naming the first path where they
    differ; if [rule] uses a value that a differentiation applying it tracks, as
    {!custom_jvp} does; and under forward mode around no such reverse mode, if
    the result has a float or complex array
    (["Rune.jvp: a custom_vjp rule has no forward derivative; give the function
      a custom_jvp rule"]). Where [pb] runs on values, the check of its result
    raises in the backward pass, as does a [pb] that uses or reads a value its
    differentiation tracks
    (["Nx.mul: a value of Rune.grad', used after Rune.grad' returned"]). *)

val custom_transpose :
  ('p, 'd) Nx.Ptree.t ->
  ('q, 'd) Nx.Ptree.t ->
  ('p -> 'q) ->
  transpose:('q -> 'p) ->
  'p ->
  'q
(** [custom_transpose p q f ~transpose args] is [f args], for an [f] linear in
    [args] whose transpose is [transpose]: [<w, f v> = <transpose w, v>] for
    every [v] and [w] ({!section-derivatives}). Both may read other values, the
    {e residuals}, in which they need not be linear. That they are linear and
    transposes is the caller's promise, which {!check_transpose} tests. It gives
    a tangent map whose transpose has a better form than its operations'
    transposes:

    {[
    let solve =
      Rune2.custom_jvp
        Nx.Ptree.(pair tensor tensor)
        Nx.Ptree.tensor
        (fun (a, b) ->
          let x = Nx.solve a b in
          let along v =
            Rune2.custom_transpose Nx.Ptree.tensor Nx.Ptree.tensor
              (fun v -> Nx.solve a v)
              ~transpose:(fun w -> Nx.solve (Nx.matrix_transpose a) w)
              v
          in
          (x, fun (da, db) -> along Nx.Infix.(db - Nx.matmul da x)))
    ]}

    Where no argument belongs to any code ({!section-custom}) and no construct
    stages the call, [f] runs on the arguments' values, so it may read them with
    {!Nx.item}, and [transpose] never runs. That is the case eagerly, and in a
    {!custom_jvp} tangent map that {!jvp} runs on its tangents. Elsewhere, as in
    a tangent map that {!grad} runs on its stand-ins, [f] and [transpose] are
    traced at the call, apart, and a read of an argument raises. A
    differentiation applies [f] to the tangents, and differentiates [f]'s code
    along the values [f] captures. Reverse mode transposes the application to
    the tangents by applying [transpose], never by transposing [f]'s operations,
    and a transpose of [transpose] applies [f]. A draw without [~key] in either
    raises.

    Raises [Invalid_argument] where [transpose] is traced, if its result differs
    from [args] in its {!Nx.Ptree.val-visits}, dtypes, shapes or placements,
    naming the first path where they differ, as in
    ["Rune.custom_transpose: the root: shape [2] in the transpose's result, [3]
     in the arguments"]. Raises it too if [f] or [transpose] adds to a total, at
    the addition where [f] runs on values and at the call where they are traced:
    each also runs as the other's transpose, away from the call, so its
    additions would have no single place. *)

val check_transpose :
  ?tol:float ->
  ('p, 'd) Nx.Ptree.t ->
  ('q, 'd) Nx.Ptree.t ->
  ('p -> 'q) ->
  transpose:('q -> 'p) ->
  'p ->
  (unit, string) result
(** [check_transpose p q f ~transpose args] tests that [transpose] is [f]'s
    transpose on three pairs of standard normal [v], of [args]' structure,
    dtypes, shapes and placements, and [w], of [f v]'s, drawn from fixed keys
    (both parts of a complex array, zeros in an integer or boolean one): it is
    [Ok ()] if [<w, f v> = Re Σ w (f v)] is within [tol] (default [1e-5]) of
    [<transpose w, v>], relative to the larger sum of the products' magnitudes
    where it is above [1], and [Error msg] naming the pair and both pairings
    otherwise. It reads the results, so [f] and [transpose] must not be traced.

    Raises [Invalid_argument] if [f v] and [w], or [transpose w] and [args],
    differ in their {!Nx.Ptree.val-visits}, dtypes or shapes. *)

(** {1:maps Maps}

    A {e map} applies a function once per {e lane}. Its arguments' arrays share
    a leading length [n], and lane [k] receives row [k] of each. The function
    runs once, over values that stand for all [n] lanes, and the lanes' results
    are stacked along a new axis 0. A value it captures is the same in every
    lane, and so is a draw without [~key]. A lane's values have no elements to
    read: {!Nx.item} of one raises
    ([Nx.item: the value is traced by Rune.vmap]), so an OCaml [if] cannot take
    one branch per lane: {!switch} does, and {!Nx.where} selects per lane.

    {b Held lanes.} Where an {!iterate}'s lanes stop apart, or a {!switch}'s
    lanes take different branches, a lane may be {e held}: it computes at its
    {e donor}, the first lane that runs, on the donor's values detached, its own
    discarded, so that no code runs at a point no lane reached. A held lane's
    {!Nx.check}s pass, its {!lane_index} is its donor's, its additions to a
    {!Total} are dropped, and {!lanes} of the map raises. A captured array every
    lane shares receives, under reverse mode, a held lane's zero cotangent times
    its donor's partial derivative, which turns an infinite gradient into NaN.

    A map's lanes are named by an axis, so that code inside nested maps reads
    the lanes of the map it names: {!lanes} gathers every lane's value and
    {!lane_index} gives the calling lane's position. An inner map of the same
    name hides an outer one. A map applied without [~axis] has a fresh name, so
    no code reads its lanes. *)

type axis
(** The type for names of maps. *)

val axis : unit -> axis
(** [axis ()] is a fresh name, distinct from every other. *)

val vmap : ?axis:axis -> ('a -> 'b, 'd) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [vmap ?axis s f] is [f] mapped over axis 0 of its arguments' arrays, [s]
    being its signature, one structure per argument and one for the result. Each
    array of the result is the lanes' arrays at its path stacked along a new
    axis 0, as {!Nx.stack} stacks [f] applied to each lane's rows; one that
    depends on no lane is broadcast along that axis. [axis] names the map.

    Each lane computes what [f] computes on its rows, the elements of its arrays
    bit for bit, but for a NaN's sign and payload and for the order in which a
    float reduction or contraction associates its terms, which nx lets depend on
    the shape of the arrays it reads. A lane lies where its rows lie: an
    argument cut along axis 0 across devices computes each lane on the devices
    that hold its row, and [Nx.place] of a lane at its own placement moves
    nothing. The lanes of an {!iterate} stop apart where its [until] depends on
    them.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument, and at
    a call if the arguments hold no array
    (["Rune.vmap: the arguments hold no array"]), a 0-d one, or arrays of
    different leading lengths, naming their paths
    (["Rune.vmap: 1: 3 rows along axis 0, 0: 2"]). *)

val lanes : axis -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [lanes a x], inside the map named [a] of [n] lanes, is every lane's [x]
    stacked on a new leading axis of length [n].

    [lanes a] is linear. The tangent of [lanes a x] is [lanes a] of [x]'s
    tangent, and under reverse mode the cotangent of [x] is the calling lane's
    row of the sum over the lanes of their cotangents of [lanes a x], so a
    lane's gradient collects every lane's use of its [x].

    Raises [Invalid_argument] with no map named [a] around the call
    (["Rune.lanes: no map named by the axis is around the call"]). *)

val lane_index : axis -> ('v, 's, 'd) Nx.t -> 'd Nx.int32_t
(** [lane_index a x], inside the map named [a], is the calling lane's index in
    it, a 0-d [int32] from [0] to the map's number of lanes minus one, whole on
    each of [x]'s devices.

    Raises [Invalid_argument] with no map named [a] around the call. *)

(** {1:totals Totals}

    A {e total} is a write-only sum, which code anywhere inside a function adds
    to and the caller reads when the function returns:
    {!Total.collect}[ t ~zero f] is [f ()] with [zero] plus the additions to [t]
    that reach it. Nothing reads a total before its [collect] returns, so an
    addition never changes a value the function computes, and with no [collect]
    of [t] open it does nothing.

    An addition reaches the innermost open [collect] of its total around the
    call, once per execution of the code that makes it. A [collect]'s
    {e running value} is its [zero] plus the additions that have reached it so
    far. The additions a traced function makes reach it when the construct that
    traced the function returns, so a construct that raises adds none: a {!jit}
    adds its function's additions once per call, as the call returns. An
    addition of another shape than its [collect]'s zero raises where it is made,
    inside a traced function as the function is traced. A step of {!scan} or
    {!iterate}, and an [until] once per test, makes its additions once per step.
    A construct adds its additions to the running value in the order they are
    made, at any depth: a loop inside another loop's step, a {!remat} or a
    custom rule in a step, and a {!jit}, which computes the sums in its program.
    So the total equals the sum of the same additions made one after another,
    under every differentiation too. Under a differentiation an addition adds
    its value's primal. A construct that makes no addition leaves a running
    value as it is. Inside a {!vmap}, each lane makes the additions of the code
    it runs: a [collect] inside the map receives its lane's, and one outside it
    receives, as one addition, their sum over the lanes as {!Nx.sum} computes it
    along the stacked axis, or nothing from a map of no lane; a construct inside
    the map sums its additions there starting from an identity that changes no
    sum (a negative zero). *)

(** Write-only sums. *)
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
      raises, [collect] raises the same exception, with the backtrace of its
      raise. *)
end

(** {1:probes Probes} *)

(** Values observed inside functions.

    A probe receives values that code anywhere inside a function emits, and the
    caller reads them when the function returns or raises:
    {!Probe.collect}[ p f] is [f ()]'s outcome beside the {e offers} that
    reached it. Nothing reads an offer before [collect] returns, so an emission
    never changes a value the function computes.

    {b Sites and offers.} An emission's {e site} is an [emit] call. Outside
    every traced function each call is a site run once, its offer the value with
    each array given a leading axis of one row. Inside a traced function each
    [emit] made as the function is traced is a site, and each call of the
    outermost construct or map inside the [collect] offers once per site that
    ran, one row per execution along a new axis 0. Enclosing rows, steps and
    lanes merge into that axis, the outer one major: a {!scan} of [n] rows whose
    step emits a [[3]] value offers [[n; 3]]; a {!vmap} of [4] lanes around it
    offers [[4n; 3]], lane [0]'s rows first. A step or branch not taken and a
    held lane ({{!section-maps}held lanes}) offer no row, and a site with no row
    no offer.

    {b Differentiation.} Emissions offer primals, once per execution: under
    reverse mode those of the forward pass. A backward pass, a transpose, a
    {!remat}'s computing again, a {!root}'s derivative, and a custom rule's
    function derived again or applied to tangents offer nothing. A {!custom_jvp}
    tangent map's emissions are offered under forward mode. *)
module Probe : sig
  type 'a t
  (** The type for probes of ['a] values. *)

  val make : ('a, 'd) Nx.Ptree.t -> 'a t
  (** [make s] is a fresh probe of values of [s]'s structure, distinct from
      every other. *)

  val emit : 'a t -> 'a -> unit
  (** [emit p v] offers [v] to the innermost open {!collect} of [p]. With none
      open it does nothing, at the cost of one effect, walking nothing; one that
      reaches a [collect] reshapes [v]'s arrays.

      Raises [Invalid_argument] while a [collect] of [p] is open: naming
      [Rune.Probe.emit] for a tangent under reverse mode, and naming the entry
      point in an {!iterate}'s [until], whose tests transformations make apart.
  *)

  val collect :
    'a t -> (unit -> 'r) -> ('r, exn * Printexc.raw_backtrace) result * 'a list
  (** [collect p f] is [f ()]'s result, or the exception it raised with its
      backtrace, and the offers to [p] that reached it. Outside every
      transformation they reach it in the order their emits and constructs
      return, a construct's in the order its sites were made; a transformation
      around a map may deliver a site's offer before another's. When [f] raises,
      the offers are those of the emits that ran before the raise: an
      {!Nx.check} failing in row [k] of a scan's step offers the rows up to [k],
      row [k]'s emits before the check included.

      Offers are arrays left unread on the sets their values lie on, except
      where lanes were held: [collect] then reads on the host which rows ran,
      once per offer, and gathers them.

      Raises [Invalid_argument] inside a traced function, a map or a
      differentiation, where a construct could be answered outside it. *)
end

(** {1:loops Loops}

    A loop applies a {e step} [f] to a {e carry}, a value of structure [c], from
    [init]. A {!scan} passes each step a row of its [xs] as well and stacks the
    outputs the steps return, and an {!iterate} runs until a test holds. The
    step is traced once, at the call, over stand-ins of [init]'s arrays and of a
    row's ({!section-tracing}). The carry the step returns must have the
    received carry's {!Nx.Ptree.val-visits}, dtypes and shapes, and lies where
    the step returns it.

    A transformation of a loop is a loop of its step's transformation, whose
    carry gains a tangent, under {!jvp}, or lanes, under {!vmap}, only for the
    carry arrays that have one at the call or that a step gives one. Reverse
    mode runs the loop once forward, keeping of each step the values its
    tangent's operations read (of an elementwise function computed in one loop,
    its operands) and none of the rows' or captures' own, then runs the step's
    transpose in reverse. A captured array's cotangent is the sum over the steps
    of each step's, added in reverse order of the steps.

    {b Iterates.} The derivative of an {!iterate} is that of the steps taken;
    its test has none. Where the loop runs at the call and reads no value of
    another transformation, reverse mode keeps the steps taken. Inside a traced
    function ({!jit}, a {!scan}'s step, a {!remat}), a map, or forward mode
    around it, it keeps room for [max] steps and transposes only the steps
    taken, but a value the step captures receives a zero cotangent where the
    loop takes no step, NaN where that zero meets an infinite partial
    derivative.

    Under {!vmap}, where the test depends on the lanes, each lane stops on its
    own test, and the loop runs until the last lane stops. A lane that has
    stopped keeps its carry bit for bit and is held
    ({{!section-maps}held lanes}) on the step of its donor, the first lane still
    running, and its row of every array the step captures with lanes. Where the
    test does not depend on the lanes, they stop together. *)

val scan :
  ('c, 'd) Nx.Ptree.t ->
  ('x, 'd) Nx.Ptree.t ->
  ('y, 'd) Nx.Ptree.t ->
  f:('c -> 'x -> 'c * 'y) ->
  init:'c ->
  'x ->
  'c * 'y
(** [scan c x y ~f ~init xs] folds the step [f] over the rows of [xs], a value
    of structure [x] whose arrays share a leading length [n]: step [i] passes
    [f] row [i] of every array. [f carry row] returns the next carry and the
    step's outputs, of structure [y]. The result is the final carry and the
    outputs, each array stacked along a new axis 0 and lying where [f] returns
    it. With [n = 0] it is [init] and outputs of leading length [0]. Under
    reverse mode a row's cotangent is stacked as the outputs are.

    Raises [Invalid_argument] if [xs] has no array
    (["Rune.scan: xs has no leaf"]), a 0-d array, or arrays of different leading
    lengths, naming their paths (["Rune.scan: xs: 1: 3 rows, 0: 2"]), and, at
    the trace, if the carry [f] returns differs from the one it received, naming
    the first path where they differ, as in
    ["Rune.scan: 0: shape [3] in the carry the step returned, [2] in the carry
     it received"]. *)

val iterate :
  ('c, 'd) Nx.Ptree.t ->
  max:int ->
  until:('c -> 'd Nx.bool_t) ->
  f:('c -> 'c) ->
  'c ->
  'c
(** [iterate c ~max ~until ~f init] applies the step [f] to the carry, from
    [init], until [until carry] holds, and returns that carry. [until] returns
    one boolean and is tested before each step, so an [init] that satisfies it
    is returned unchanged. [until] is traced as [f] is.

    Raises [Invalid_argument] if [max < 0]
    (["Rune.iterate: max = -1 is negative"]), if [until] returns other than one
    element, at the trace if [f]'s carry differs from the one it received as
    {!scan} says, and if [until] is still false after [max] steps
    (["Rune.iterate: until is still false after max = 50 steps"]), naming under
    a map the first lane still running
    (["Rune.iterate: until is still false after max = 50 steps, in lane 3"]). *)

(** {1:branches Branches} *)

val switch : ('b, 'd) Nx.Ptree.t -> 'd Nx.int32_t -> (unit -> 'b) list -> 'b
(** [switch b i branches] is [f ()], [f] the branch at position [i] of
    [branches], a 0-d [int32]: that branch alone computes. Every branch is
    traced at the call, each operation it applies staged, those on values it
    captures included, so a branch not taken computes nothing: no work, no
    check, no addition to a {!Total}, no NaN reaching a value or a derivative.
    Each branch's code runs once per call; {!Nx.item} inside one raises. Outside
    every construct [switch] reads [i], a wait on its device set
    ({!Nx.section-devices}).

    {[
    let safe_log x =
      Rune2.cond' (Nx.greater_s x 0.) (fun () -> Nx.log x) (fun () -> x)
    ]}

    The derivative is the taken branch's, [i] having none, at a boundary too:
    the comparison that computes [i] chooses the side. Under {!val-vmap} with
    [i] the same in every lane, the branch runs over the lanes. With [i] per
    lane, the lanes' indices are read once where they are values, and otherwise
    each branch reads whether some lane takes it; each branch some lane takes
    runs over every lane, a lane that did not take it held
    ({{!section-maps}held lanes}) on the first lane that did, and each lane
    receives its own branch's results and derivatives. A branch no lane takes
    runs nothing.

    Raises [Invalid_argument] if [branches] is empty, if [i] is not 0-d, at the
    call if two branches' results differ in their visits, dtypes, shapes or
    device sets, naming the first path where they differ, as in
    ["Rune.switch: 0: shape [3] in branch 1, [2] in branch 0"], and if [i] is
    outside [0 .. n-1], as in ["Rune.switch: index 3 outside 0 .. 2"], naming
    the lane under a map. *)

val cond :
  ('b, 'd) Nx.Ptree.t -> 'd Nx.bool_t -> (unit -> 'b) -> (unit -> 'b) -> 'b
(** [cond b p t f] is [t ()] where the 0-d [p] holds and [f ()] where it does
    not, as {!switch} on [p]. A NaN compares false:
    [cond b (Nx.greater_s x 0.) t f] takes [f] at NaN and at [0.]. *)

(** {1:roots Roots}

    A {!root} states that the value its [solve] returns is a zero of a
    [residual], and differentiates that statement. Under differentiation the
    result's tangent is the [u] with [J u + r = 0], [J] the derivative of
    [residual] at the result and [r] the tangent of [residual] there with the
    result held: every tracked value [residual] captures contributes to [r]. The
    derivative is taken at the returned point, whether or not [residual]
    vanishes there. [solve] has no derivative: a tracked value it reads is its
    value alone.

    [solve], [residual] and [linear_solve] are traced at the call: {!Nx.item} of
    a value [residual] or [linear_solve] computes from its argument raises.
    [solve] stages every operation it applies ({!section-tracing}), over the
    values it captures detached, so {!Nx.item} inside it of a value a
    transformation around tracks raises.

    {b Linear solves.} [linear_solve op b] returns a [v] with [op v = b] for the
    linear [op]. It is traced once, against an abstract [op], which a derivative
    fills in with [J], and reverse mode with [J]'s transpose: the solve is
    linear in [b], and its transpose solves the transposed system. It may apply
    [op] under {!jvp}, {!grad} or {!val-vmap} opened inside it, as in
    [Nx.solve (Rune2.jacfwd' op b) b], but not under a {!jit}, whose function
    may not capture [op]'s stand-ins. Without it, a derivative builds [J]'s
    matrix, its columns [J] of the unit vectors in one map, and solves it with
    {!Nx.solve}, which suits small systems; the arrays of [x] must then share a
    dtype. A complex [J] need only be real-linear: its matrix is then the real
    system's over the real and imaginary parts. Under {!val-vmap}, each lane
    solves its own system: [op] gives each lane its own product, whatever maps
    [linear_solve] opens.

    The derivative of the solve itself, along [J]'s dependence on the solution
    and the values [residual] captures, is the solve of the system's derivative:
    of [v = S b], [S (db - dJ v)]. A derivative never differentiates
    [linear_solve]'s code. So a second derivative solves twice more, whatever
    [linear_solve] iterates; a tracked value [linear_solve] captures has no
    derivative. With the default linear solve, a singular [J] at the result, as
    {!Nx.solve} defines it, gives a NaN derivative.

    Additions to a {!Total} that [solve] makes count once per call; those of
    [residual] and [linear_solve], which only derivatives evaluate, are dropped.
*)

val root :
  ?linear_solve:(('x -> 'x) -> 'x -> 'x) ->
  ('x, 'd) Nx.Ptree.t ->
  residual:('x -> 'x) ->
  (unit -> 'x) ->
  'x
(** [root x ~residual solve] is [solve ()], a value of structure [x] stated to
    be a zero of [residual]. The system must be square: [residual]'s result has
    [x]'s structure, dtypes and shapes.

    {[
    (* The square root of [a] by Newton's method, whose derivative [1 / (2 sqrt
       a)] comes from the residual *)
    let sqrt_newton a =
      let residual x = Nx.Infix.((x * x) - a) in
      Rune2.root Nx.Ptree.tensor ~residual (fun () ->
          Rune2.iterate' ~max:50
            ~until:(fun x -> Nx.all (Nx.less_s (Nx.abs (residual x)) 1e-12))
            ~f:(fun x -> Nx.Infix.((x + (a / x)) /$ 2.))
            a)
    ]}

    Raises [Invalid_argument] at the call if [residual]'s result differs from
    the solution, naming the first path where they differ, as in
    ["Rune.root: 0: shape [3] in the residual's result, [4] in the solution"];
    without [linear_solve], if [x]'s arrays differ in dtype; if [residual] reads
    the {!lanes} of a map, which joins the lanes' systems; if [linear_solve]
    applies [op] after it returned; and, naming the {!jit}'s entry point, if it
    applies [op] inside a {!jit}. *)

(** {1:remat Rematerialization} *)

val remat : ('a -> 'b, 'd) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [remat s f] is [f], [s] its signature as for {!jit}, whose reverse mode
    keeps [f]'s arguments and the values it captures, and computes [f]'s
    intermediate values again in the backward pass: it trades computation for
    memory. [f] is traced at each call over stand-ins of its arguments' arrays,
    every operation it applies staged, those over values it captures included,
    and its program evaluated, so outside every transformation [remat s f x] is
    [f x], bit for bit but for a NaN's sign and payload. Every transformation
    sees [remat s f] as it sees [f]: its derivatives in either mode, and its
    map, are [f]'s. An addition to a {!Total} that [f] makes counts once,
    however often the backward pass computes [f] again. Inside a {!jit}, the
    backward pass computes again each of [f]'s intermediate values that reads a
    value the program computes, among [f]'s arguments and captures, once the
    cotangents of [f]'s results are computed, so the program keeps none of them
    from one pass to the other. A value [f] computes from the program's
    arguments and constants alone is the forward pass's, kept for the backward
    pass, and so is one [f] computes in a {!scan}'s or an {!iterate}'s step from
    the step's captures alone, which the loop computes once.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument. *)

(** {1:jit Compilation}

    A {e compiled function} [jit s f] runs [f] as a program compiled through
    tolk, raven's compiler. A call traces [f] over stand-ins of its arguments'
    arrays, staging every operation [f] applies ({!section-tracing}), and runs
    the program compiled. The compiled function keeps the programs of its calls,
    one per key, and compiles each once per variant, so a call that finds its
    key and variant runs the compiled program without tracing [f] or compiling.
*)

val jit :
  ?beam:int ->
  ?parallel:int ->
  ('a -> 'b, 'd) Nx.Ptree.fn ->
  ('a -> 'b) ->
  'a ->
  'b
(** [jit ~beam ~parallel s f] is [f] compiled, [s] its signature, its kernels
    searched as {!section-search} states. Its results are fresh, at the
    placement eager computation gives them ({!section-compiled}), except a
    result that takes a donated argument's memory ({!section-donation}).

    Raises [Invalid_argument] when applied to [s] if [s] has no argument, and at
    a trace if [f] captures a value that another transformation tracks
    (["Rune.jit: the function captures a value another transformation traces;
      pass it as an argument"]). *)

(** {2:keys Keys}

    [jit s f] keeps a program per {e key}, the abstract value of a call's
    arguments:
    - their {!Nx.Ptree.val-visits}, the integers and cases their walk reports
      included;
    - each array's dtype, shape and placement;
    - the number of lanes of each map whose name the program reads through
      {!lanes} or {!lane_index};
    - the placement of the running value of each {!Total} [f] adds to, where a
      call finds it elsewhere than the key's first trace did;
    - whether a {!Probe.collect} of each probe [f] emitted to is open.

    Whether an argument is donated is not part of the key. A call retraces
    exactly when its key is new to [jit s f], over stand-ins in C order, so
    [f]'s code and its side effects run once per key. Domains share the
    programs. A call whose key is being traced waits for that trace, except on
    the tracing domain, where it traces again.

    With the variable [RUNE_JIT_DEBUG] holding a nonzero integer in the
    environment when rune2 is initialised, each retrace prints on standard error
    the first difference between its key and that of the latest call that
    returned, on any domain, and the first call to find its program or a variant
    of it declined ({!section-evaluated}) prints why. *)

(** {2:compiled Compiled programs}

    A program compiles the first time a call runs it, then once per {e variant}:
    for each argument, its strides and its offset counted from the last 16-byte
    boundary of its buffer at or below its lowest element, the {e phase} of that
    boundary (the number of bytes by which it lies past an address that is a
    multiple of 16), and whether the caller donated it; together with tolk's
    settings that shape what it compiles, the search widths [BEAM] and [JITBEAM]
    among them where no [beam] is given, and whether tolk's [DEBUG] is 2 or
    more. A view thus reads in place, without a copy, from a program compiled
    for its layout. Each result that takes no donated argument's memory is
    fresh, at the placement eager computation gives it. A value [f] captures on
    another device, which [f] places where the program computes, is copied there
    once per program, which holds the copy. *)

(** {2:search Search}

    A [beam] above [0] searches each kernel of a program, and of the programs
    derived from it, for the optimisations that run it fastest, keeping the
    [beam] fastest candidates of each round and timing them on its device, which
    makes the first call of a variant much longer; [0] searches none. [parallel]
    compiles the candidates on at most that many domains, and on one where it is
    below [2]. An explicit [beam] stands for the [BEAM] and [JITBEAM] settings,
    and [parallel] for the [PARALLEL] setting, which decide otherwise: a change
    of [BEAM] or [JITBEAM] around a call of a function given [beam] compiles
    nothing, and functions given different widths share no program. A call
    inlined in an enclosing [jit] compiles as that one's program does. *)

(** {2:donation Donation}

    A call consumes the donated handles among its arguments
    ({!Nx.section-moving}), as one operation consumes its operands: the values
    they give up die once the call has checked its arguments, and a later use of
    one raises [Invalid_argument] naming the entry point. A result may take a
    donated argument's memory instead of fresh memory where the two have one
    dtype, element count and placement, no check reads the argument, and the
    result either reads it only at each element's own index (through elementwise
    functions, casts and bitcasts that keep the width, reshapes in C order,
    copies, placements, a scatter's target or a contraction's [init]), is the
    argument itself, or reads no donated argument while nothing reads that one.
    A result that takes an argument's memory is stored over it: each other value
    of the call that reads the result reads it there, after the store, and each
    that reads the argument as it was reads it before. Where no order of the
    stores holds both, as for two results that each read the other's argument,
    the latest result of the cycle that takes memory gives up that argument, and
    the arguments are lent again without the pair. A donated argument whose
    memory another value shares, or that is a view off 16 bytes or out of C
    order, lends nothing. Each donated argument lends to one result at most, the
    results of a scatter first. A result's values are the same either way.

    A traced function reads a donated value it captures as its arrays and
    consumes nothing: the value is lent ({!Nx.section-moving}), its memory
    shared from then on, so its program reads that value at every evaluation. *)

(** {2:checks Checks}

    A check the function makes ({!Nx.check}) on the host is computed with the
    program, and the first that fails, in the order the function made them,
    raises its exception when the call returns, with the data at its first
    failing index, as eagerly; the call's results are then lost. A program with
    a check on a device other than the host, which raises at a later wait there,
    is evaluated ({!section-evaluated}). *)

(** {2:numerics Numerics}

    A compiled call computes each nx function to the accuracy below, so a float
    result may differ from eager's within it. The {e unit roundoff} [u] of a
    float dtype is [2{^-24}] for [float32] and [2{^-53}] for [float64], that of
    its parts for a complex one. An {e ulp} is one rank in the order of a
    dtype's bit patterns, [-0.] and [+0.] one rank apart, and an error of [n]
    ulps is a distance of [n] ranks from the correctly rounded result. [γ(k)] is
    [k u / (1 - k u)]. A function is {e exact} when it gives eager's bits, a NaN
    result's sign and payload excepted.

    Moves of elements, such as {!Nx.take}, {!Nx.concatenate}, {!Nx.sort} and a
    {!Nx.scatter} by [Set], keep a NaN's bits too. Casts, bitcasts, {!Nx.where},
    comparisons, extremes ({!Nx.max}, {!Nx.argmax}) and integer and boolean
    arithmetic are exact, as are {!Nx.neg}, {!Nx.abs}, {!Nx.sign}, {!Nx.floor},
    {!Nx.ceil}, {!Nx.round}, {!Nx.trunc}, {!Nx.mod_}, {!Nx.maximum},
    {!Nx.minimum} and IEEE 754's correctly rounded {!Nx.add}, {!Nx.sub},
    {!Nx.mul} and {!Nx.fma}. {!Nx.div}, {!Nx.recip} and {!Nx.sqrt} are exact on
    a device that rounds division and the square root correctly. Of the complex
    functions, {!Nx.neg}, {!Nx.add}, {!Nx.sub}, the extremes and the comparisons
    are exact.

    A float narrower than [float32] computes at [float32] and rounds once, so
    each real function of the table below is within one ulp of the correctly
    rounded result there, {!Nx.pow} where its result is finite in that dtype. On
    a device that flushes subnormals, as Metal does at [float32], a function may
    be inexact or break its bound where an operand, a result or an intermediate
    is below the least normal float.

    {b Real functions.} Where the device computes [exp2], [log2] and [sin]
    within 4 ulps and rounds division and the square root correctly, at the
    default of the [TRANSCENDENTAL] setting, each function is within these ulps
    of the correctly rounded result, to first order in [u]:

    {v
      function                      float32  float64
      Nx.log                           1        1
      Nx.exp2, Nx.log2, Nx.sin         4        4
      Nx.erf                           2        3
      Nx.atan                          2        7
      Nx.asin, Nx.acos                 5        9
      Nx.log1p                         5        5
      Nx.atan2                         6        8
      Nx.exp, Nx.cos                   9        9
      Nx.cosh                         10       10
      Nx.tanh                         18       18
      Nx.tan                          19       19
      Nx.expm1, Nx.sinh               23       23
      Nx.pow x y                9 + 6 |y log2 |x||
    v}

    {!Nx.log} needs only correctly rounded division. On the host CPU, {!Nx.exp},
    {!Nx.sin} and {!Nx.cos} are within 2 ulps at [float32]. Where every operand
    is [±0.], [±inf] or a NaN, a function gives the value of C99 Annex F
    (ISO/IEC 9899, F.9), a zero's sign included: [sin (-0.)] is [-0.] and
    [log 0.] is [-inf]. {!Nx.pow} is [1.] where [y] is [±0.] or [x] is [1.], NaN
    included, and where [x] is [-1.] and [y] is infinite. A function nx composes
    from these, such as {!Nx.asinh} or {!Nx.softmax}, computes as nx composes
    it.

    {b Complex functions.} A result [z'] of exact value [z] holds a {e normwise}
    bound [c] in the sense that [|z' - z| / |z| <= c u], to first order, for the
    operand [x + iy] and the exponent [w]:

    {v
      function                                      c
      Nx.mul                                        2
      Nx.sign, Nx.sqrt                              3
      Nx.div, Nx.recip                              4
      Nx.log                                       18
      Nx.log2                                      20
      Nx.atan, Nx.atanh, Nx.acos, Nx.acosh         21
      Nx.asin, Nx.asinh                            23
      Nx.exp                                       31
      Nx.cos, Nx.cosh                              57
      Nx.sin, Nx.sinh                              67
      Nx.tan, Nx.tanh                              73
      Nx.exp2                            30 + 1.4 |y|
      Nx.pow z w                    31 + 20 |w log z|
    v}

    A bound holds where the operands' parts are finite, the result's modulus is
    normal and no value computed for it, a part of it included, is subnormal.
    {!Nx.div} and {!Nx.recip} also need the parts below half the greatest float
    and the divisor's larger part normal, and {!Nx.pow} needs [|w| |log z|]
    below the greatest float. A NaN part and no infinite one give a NaN part,
    except [z{^w}] at [w = 0], which is [1]. At an infinite part, each part of a
    transcendental function's result is a NaN, an infinity, a zero or finite as
    Annex G (G.6) gives it, an infinite part with Annex G's sign. There, where
    no part is a NaN, {!Nx.sign} gives the direction of the infinite parts, one
    of [±1], [±i] and [(±1 ± i) / √2], and {!Nx.recip} the sorts of
    [(1 + 0i) / z] by Smith's division, which divides by the divisor's larger
    part. Zeros' signs are unspecified, and so is an infinity's sign that comes
    from a NaN part or that Annex G leaves open. {!Nx.mul}, {!Nx.div} and
    {!Nx.pow} are unspecified at an infinite part. {!Nx.exp2} at [x = +inf] may
    give an infinite part the wrong sign where [y ln 2] lies within a relative
    [2u] of a non-zero multiple of [π/2].

    {b Reductions.} A float reduction, scan, contraction or scatter by [Add]
    computes in its {e accumulator}, [float32] for a narrower float, the dtype
    otherwise, or a contraction's [acc] ({!Nx.contract}), and [u] is the
    accumulator's. It combines terms in an order the compiler chooses and rounds
    once into its dtype, which adds its own error. Barring overflow and
    underflow, a result of [n] terms is within this bound of the exact one:

    {v
      function             bound
      Nx.sum, Nx.cumsum    γ(n - 1) Σ|x|
      Nx.prod, Nx.cumprod  relative γ(n - 1)
      Nx.matmul, ...       γ(n + 1 + s) (|init| + Σ|a b|)
        complex real       γ(n + 2) (|init| + Σ(|ar br| + |ai bi|))
        complex imaginary  γ(n + 2) (|init| + Σ(|ar bi| + |ai br|))
      Nx.scatter by Add    γ(n) (|x| + Σ|ui|)
      Nx.combine_patches   γ(n - 1) Σ|x|
      Nx.logsumexp r       (1.37 (n - 1) + 2E + 1 + (2L + 1) ln n) u + u|r|
      Nx.logcumsumexp r    ((2E + 3) D + (2L + 2) ln n) u + u|r|
      Nx.var V             (n + 4) u V + 2 n u B √V + 3 (n + 2) u B²
    v}

    A cumulative output's [n] counts the terms up to it, a contraction's its
    products ({!Nx.matmul}, {!Nx.dot}, {!Nx.einsum}, {!Nx.contract}), and [s]
    its operands with a block scale. A scatter adds [n] updates [ui] to [x]. [E]
    and [L] are the ulp bounds of {!Nx.exp} and {!Nx.log},
    [D = max 0 (2⌈log₂ n⌉ - 1)], [B = (n + 1) u M] for [M] the mean of the
    terms' magnitudes. {!Nx.logcumsumexp}'s bound needs [n] exact in the
    accumulator, and {!Nx.var}'s [V] precedes its [ddof] scale, which rounds
    once more. A complex sum adds each part as a real one. A sum, product or
    extreme with a NaN term is NaN. Terms all [-0.] sum to [+0.], or to [-0.] in
    a scatter and {!Nx.combine_patches}. Special values are eager's, and a
    variance is never negative.

    {b Matrices.} A routine computes each matrix of a [float32] or [float64]
    operand, of [m] rows, [n] columns and [k = min m n], to first order in [u]
    barring overflow and underflow. [|·|] is elementwise and [‖·‖_F] the
    Frobenius norm. [a] is the matrix read, symmetric from its triangle for
    {!Nx.cholesky} and {!Nx.eigh}, and for {!Nx.solve_triangular} the triangle,
    transposed under [adjoint], unit on its diagonal under [unit_diagonal]. [P]
    is {!Nx.lu}'s permutation, its rule applied to the computed columns. For [b]
    the dtype's bits of precision, [R] is [S] times [k - 1] for an even [k] or
    [k] for an odd one, [0] for [k < 2], where [S = ⌈b ⌈log₂ k⌉ / 10⌉] for
    {!Nx.eigh} and [S = ⌈b (⌈log₂ k⌉ + 1) / 10⌉] for {!Nx.svd}:

    {v
      Nx.cholesky          L Lᵀ = a + Δ, |Δ| <= γ(n + 1) |L| |Lᵀ|
      Nx.lu                L U = P a + Δ, |Δ| <= γ(k) |L| |U|
      Nx.solve_triangular  |a x - b| <= γ(n) |a| |x| for each column
      Nx.qr                ‖a_j - q r_j‖₂ <= g ‖a_j‖₂ for each column,
                           each element of qᵀ q - I at most g,
                           g = γ(k (2m + 10))
      Nx.eigh              each eigenvalue within γ(16 (R + n)) ‖a‖_F,
                           ‖a - v diag w vᵀ‖_F <= g ‖a‖_F,
                           ‖vᵀ v - I‖_F <= g,
                           g = γ(16 (R + n) (1 + √n))
      Nx.svd               ‖a - u diag s vh‖_F <= g ‖a‖_F,
                           ‖uᵀ u - I‖_F <= g, ‖vh vhᵀ - I‖_F <= g,
                           each singular value within 3 g ‖a‖_F,
                           g = γ(8 k (p (p + 6) + 8 R)), p = max m n
    v}

    {!Nx.svd}'s bound is over [u]'s first [k] columns and [vh]'s first [k] rows,
    and is vacuous from [k = p = 68] at [float32] and [k = p = 82313] at
    [float64], and at a smaller [k] where [p > k]. {!Nx.cholesky}'s [Upper]
    factor is the transpose of [Lower]'s, bit for bit, and {!Nx.qr}'s [r] has a
    diagonal of either sign. A matrix that holds NaN or an infinity, or whose
    iteration does not converge, makes only its own results NaN.

    {b Fourier transforms.} A transform [ŷ] of [N] points in all, before
    [norm]'s scaling, which rounds once more, has [‖ŷ - y‖₂ <= ε u √N ‖x‖₂] for
    the exact transform [y] and the operand's norm [‖x‖₂], [√2] times that for
    {!Nx.irfft} and {!Nx.hfft}. [ε] is the sum over [axes] of [e(n)] for an axis
    of [n] points:

    {v
      e(n) = Σ c(p) + 15 (r - 1)   over n's r prime factors p, repeated
      c(2) = 1
      c(p) = (2p + 15) √p          for 2 < p <= 250
      c(p) = (2 e(m) + 18) m / √p + (e(m) + 12) √m + 15   for p > 250
    v}

    where [m] is the least power of two at least [2p - 1], so
    [e(m) = 16 log₂ m - 15]. A NaN or infinite point makes the results it enters
    NaN. *)

(** {2:derived Transformations of compiled calls}

    Under {!jvp}, {!vmap} and reverse mode, a call is a call of a program
    derived from its key's program. Under {!jvp} it is that program's
    derivative, and under {!vmap} its map. Under reverse mode it is the part
    that reads no tangent, which computes at the call and returns the values the
    rest reads; the backward pass transposes the rest, linear in the tangents,
    into one more program. The rest reads an argument without a tangent, or a
    value [f] captures, as it is, uncopied, unless the call has a donated
    argument. The part computed at the call consumes it and then returns each
    value the rest reads. Each derived program compiles as the key's does. A
    {e derivation} is a value: its kind, which arguments have a tangent or
    lanes, which are donated under reverse mode, their dtypes, shapes and
    placements, the number of lanes of each map the program reads, and the name
    of the map it is derived under where it reads that name. A call keeps one
    program per derivation of its key's, so two calls share a derived program
    only when their derivations are equal, and [f] runs once per key, whatever
    is derived from it.

    Inside an enclosing [jit], a call is inlined: its program's operations, or
    its derived program's, are staged there. Inside a {!scan}'s or an
    {!iterate}'s step, a {!remat}'s function or a custom rule, the call stays a
    compiled call of its own, which computes the bits it computes outside the
    construct. *)

(** {2:rules Loops and rules}

    A {!scan} or an {!iterate} compiles as one loop of the program, its step
    compiled once whatever its number of steps: each step reads its row, writes
    its outputs' row, and updates the carry in its own storage. Where the rows
    of an array do not lie a multiple of 16 bytes apart, the step reads them
    from a copy whose rows do. A check in a step raises, once the call returns,
    the first failure of the first step that fails, as the steps written out
    would, and an {!iterate} stops at the first failure. A map's {!iterate}
    whose lanes stop apart runs while any lane runs on, a lane that has stopped
    {{!section-maps}held}. An {!iterate}'s derivative or map is evaluated where
    it runs a count of rows or stacks its steps. A {!switch} compiles as one
    loop per branch of at most one trip, which runs where the index names its
    branch, so a branch not taken computes nothing. A {!remat}, a {!custom_jvp},
    a {!custom_vjp} and a {!custom_transpose} compile as their function's
    operations, the program's own. *)

(** {2:emissions Emissions}

    A call offers its function's emissions ({!Probe}) once it returns. A
    compiled program returns them as more results: those its function makes, and
    those of a {!scan}'s step, a call, a {!remat} and a custom rule inside it, a
    scan's stacked over its rows. A program whose emissions lie in another
    construct, or that both emits and checks, is evaluated, so that a failing
    check keeps the offers made before it. *)

(** {2:evaluated Evaluated programs}

    An {e evaluated} program runs as its operations evaluated by nx, giving
    eager's values. A program's {e device} is the one its first argument, or
    else its first result, lies whole on. The compiler declines, and so
    evaluates, a program that holds:
    - a map's {!lanes} or {!lane_index}, or a {!root};
    - an emission {!section-emissions} names as evaluated;
    - the tangents of a {!custom_vjp}'s results, which a {!jvp} of it makes and
      which then raise as eagerly;
    - a loop {!section-rules} names as evaluated, or an operation the compiler
      does not lower;
    - a check on a device other than the host;
    - no value, an argument or a result with an element on a device other than
      the program's, or a value over several devices. A result the function
      returns without computing it, a constant with an element that lies whole
      on another device, its elements in C order from a byte boundary, is no
      such result: each call copies it into fresh memory of its own device.

    It also declines a program whose device computes nothing ({!Rig.computes}),
    and one tolk cannot compile. An evaluated call's results are fresh, as a
    compiled call's are, and a call with donated arguments consumes them and
    lends nothing. *)

(** {2:inspect Inspecting a trace} *)

type program
(** The type for traced programs. *)

val program :
  ('a, 'd) Nx.Ptree.t -> ('b, 'd) Nx.Ptree.t -> ('a -> 'b) -> 'a -> program
(** [program a b f x] is the program that {!jit} traces of [f], a function of
    structure [a] to structure [b], at a call over [x], without compiling or
    computing it: [f] runs once, over stand-ins of [x]'s arrays in C order. The
    program holds what [f] stages, from {!jvp}, a {!scan}'s step or another
    construct, every construct with the programs it carries. [program] is for
    reading what a function computes.

    Raises what {!jit} raises at a call that traces. *)

val pp_program : Format.formatter -> program -> unit
(** [pp_program] formats a program: the dtype, shape and placement of each
    constant it reads, its parameters, each equation with its results' dtypes,
    shapes and placements and its operands, a construct with the programs it
    carries indented, and its outputs. The top program's variables are numbered
    as traced, and each carried program's follow its parent's, so a function
    prints alike at every trace. *)

val eval_shape :
  ('a, 'd) Nx.Ptree.t -> ('b, 'd) Nx.Ptree.t -> ('a -> 'b) -> 'a -> 'b
(** [eval_shape a b f x] is [f x]'s structure with each array a stand-in of its
    dtype, shape and placement, as {!jit} traces it, computing nothing:
    {!Nx.shape}, {!Nx.dtype} and {!Nx.placement} work on it, and an operation on
    it or a read raises [Invalid_argument]. An array [f] captured and returns
    unchanged is that array.

    Raises what {!jit} raises at a call that traces. *)

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
    [f] along element [k] of [x] in C order. Every column is computed in one
    forward pass, a {!vmap} of {!jvp'} over the elements' unit tangents.

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
    element [k] of the result in C order. Every row is computed in one backward
    pass, a {!vmap} of {!vjp'}'s pullback over the elements' unit cotangents.
    For a complex-differentiable [f] it is the complex derivative, as {!jacfwd'}
    computes it.

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

val cond' :
  'd Nx.bool_t ->
  (unit -> ('v, 's, 'd) Nx.t) ->
  (unit -> ('v, 's, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t
(** [cond' p t f] is {!cond} over one array. *)

val iterate' :
  max:int ->
  until:(('v, 's, 'd) Nx.t -> 'd Nx.bool_t) ->
  f:(('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [iterate' ~max ~until ~f x] is {!iterate} of one array. *)

val vmap' :
  ?axis:axis ->
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t
(** [vmap' ?axis f x] is {!vmap} of a function of one array. *)

val jit' :
  ?beam:int ->
  ?parallel:int ->
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t
(** [jit' ~beam ~parallel f] is {!jit} of a function of one array. *)

(**/**)

(** The lowering to tolk, for its suites. *)

module Lower = Lower
module Lower_arith = Lower_arith

val derivations : unit -> int
(** [derivations ()] is the number of programs that transformations of compiled
    calls derived on the calling domain, for the suites. *)

val evaluations : unit -> int
(** [evaluations ()] is the number of compiled calls on the calling domain that
    ran their program evaluated by nx, the compiler having declined it, for the
    suites. *)

val variants : unit -> int
(** [variants ()] is the number of compiled variants that calls on the calling
    domain lowered and linked, or that the compiler declined, for the suites. A
    compiled program makes one variant per layout, phase and donation of its
    arguments and value of tolk's settings and its [DEBUG]. *)

(**/**)
