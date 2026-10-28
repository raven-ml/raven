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
    {!iterate}, {!remat}, {!jit}, {!custom_jvp}, {!custom_vjp} and
    {!custom_transpose} receive are {e traced} at the call: their OCaml code
    runs over stand-ins of their arguments' arrays, values with a dtype, a shape
    and a placement but no elements, and what it computes from them is recorded
    as a program, which the construct then evaluates. A loop's step is traced
    once, so its code runs once whatever the number of steps, as do its side
    effects, and inside it {!Nx.item} of a stand-in, or of a value computed from
    one, raises ([Nx.item: the value is traced by Rune.scan]); a branch on a
    value is {!Nx.where} or a nested {!iterate}. An operation that reads no
    stand-in computes at the call, once, but in {!remat} and {!jit}, which stage
    every operation their function applies. A draw without [~key] inside a
    traced
    function raises, naming the construct, so a step draws from a key in its
    carry or its rows. Outside every transformation a construct is its program
    evaluated: a {!scan} is the fold of its step, bit for bit but for a NaN
    result's sign and payload, which nx leaves unspecified.

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
    element that receives none is [+0.]. A {!scan}'s carry holds a derivative at
    every step: where a step gives a carry array none, the next takes a zero
    from it, so the derivative may receive zeros of either sign, and NaN where
    such a zero meets an infinite partial derivative, where the steps written
    out one after another give none.

    {b Placement.} A cotangent lies where the transposed operations put it,
    as {!Nx} routes them: at its array's placement where every value it meets
    lies there, and otherwise at the union of their cuts. A {!Nx.place}'s
    cotangent is moved back to its operand's placement. So
    [grad' (fun x -> Nx.sum (Nx.mul x w)) x] lies cut along axis 0 where [x]
    is whole on each device of a set and [w] is cut along axis 0 across them.

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

    {b Derivatives are fresh.} A tangent, a gradient and a pullback's result are
    values of their own, in C order from the start of their memory, never a view
    of an argument or of the movement their computation ends with. Under another
    transformation each is that transformation's value. *)

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

    Rune differentiates {!Nx.erfinv}, {!Nx.erfc}, {!Nx.ndtr}, {!Nx.log_ndtr},
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
    dtype.

    {t
      | Function                                             | First derivative                                                                                                                  |
      |------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
      | {!Nx.erfinv}                                         | 64 ulps                                                                                                                           |
      | {!Nx.erfc}                                           | 128 ulps                                                                                                                          |
      | {!Nx.ndtr}                                           | 256 ulps                                                                                                                          |
      | {!Nx.log_ndtr}                                       | 512 ulps                                                                                                                          |
      | {!Nx.ndtri}                                          | 64 ulps                                                                                                                           |
      | {!Nx.lgamma}                                         | for [x > 0], 256 ulps, or [256ε] absolutely where below 1; below 0, 256 ulps or [256ε (1 + |ψ(1 - x)| + |π cot (πx)|)] absolutely |
      | {!Nx.digamma}                                        | for [x > 0], 256 ulps; below 0, 256 ulps or [256ε (1 + ψ'(1 - x) + π² / sin² (πx))] absolutely                                    |
      | {!Nx.lbeta}, in either argument                      | 4096 ulps, or [8192ε] absolutely where below 1                                                                                    |
      | {!Nx.i0e}                                            | 128 ulps                                                                                                                          |
      | {!Nx.i1e}                                            | 128 ulps, or [128ε] absolutely for [1 ≤ |x| ≤ 2], about its zero at [±1.545]                                                      |
      | {!Nx.gammainc}, {!Nx.gammaincc}, in [x]              | [512 (1 + |log f|) + 64] ulps                                                                                                     |
      | {!Nx.log_gammainc}, {!Nx.log_gammaincc}, in [x]      | 256 ulps, or [256ε] absolutely where below 1                                                                                      |
      | the four, in [a]                                     | [2^-40 (1 + |log f|)] relative at [float64], [2^-16 (1 + |log f|)] at [float32]                                                   |
      | {!Nx.gammaincinv}, {!Nx.gammainccinv}, in [p] or [q] | [64 + 16κ (32 (1 + |log p|) + 4)] ulps, [q] in place of [p] for {!Nx.gammainccinv}                                                |
      | the two, in [a]                                      | [2^-40] relative at [float64], [2^-16] at [float32]                                                                               |
      | {!Nx.betainc}, {!Nx.betaincc}, in [x]                | [1024 (1 + |log f|) + 64] ulps                                                                                                    |
      | {!Nx.log_betainc}, {!Nx.log_betaincc}, in [x]        | 512 ulps, or [512ε] absolutely where below 1                                                                                      |
      | the four, in [a] or [b]                              | [2^-40 (1 + |log f|)] relative at [float64], [2^-16 (1 + |log f|)] at [float32]                                                   |
    }

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
    [pullback] runs no part of [f] again. When [vjp] runs outside every
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

    Raises [Invalid_argument] as {!vjp}'s pullback does for its argument, and,
    when the transpose is applied, naming an operation of [f] that is not linear
    in its argument, as in
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
    mode, a pullback for reverse mode, or a linear function's transpose. The
    rule is traced at the call, and the derivative it returns is traced inside
    it, so the derivative's code runs once per call, whatever differentiation
    applies it, and an error it raises reaches the call.

    {b Arguments.} A differentiation applies the rule where it tracks one of the
    rule's arguments. The rule receives them without their derivatives, and must
    not use another value that differentiation tracks: pass such a value as an
    argument. A differentiation that tracks only values the rule captures
    differentiates the rule's code, as it does any other code.

    {b Orders.} A differentiation around another differentiates the derivative's
    code, with the arrays it reads, which the rule computes: the second
    derivative of a {!custom_jvp} is the derivative of its tangent map along the
    arguments, and that of a {!custom_vjp} the derivative of its rule and
    pullback. A custom rule that a tangent map applies to its tangents runs its
    function under reverse mode, which records the function's operations, so a
    differentiation around reverse mode differentiates that function's code,
    where one around forward mode applies the rule.

    {b Reverse mode} transposes a tangent map's operations on tangents, so a
    tangent map must be linear in its tangents: one applying [Nx.exp] to a
    tangent raises, naming the entry point, when the derivative is transposed
    (in {!grad}, or as a {!vjp} pullback is applied), as in
    ["Rune.grad: a custom_jvp tangent map applies Exp to a tangent; a tangent
     map must be linear in its tangents"]. Reverse mode transposes the linear
    part: a value a tangent map adds to a tangent, or selects beside one with
    {!Nx.where}, counts as zero. A pullback or a transpose reads a result's
    cotangent at the result's set.

    {b Maps.} {!vmap} batches a custom rule: the rule and its derivative are
    evaluated over the lanes, and a differentiation around the map applies the
    rule to every lane.

    {b Totals.} The additions a rule makes reach the scopes around the call once
    per call. A tangent map's are made once per differentiation that applies it,
    none without one; a pullback's once per backward pass that applies it, to
    the scopes around that pass. Under reverse mode a tangent map that adds a
    tangent raises at the call. *)

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
    let softplus =
      Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          (stable x, fun dx -> Nx.mul (Nx.sigmoid x) dx))
    ]}

    Where the result has no float or complex array, reverse mode does not apply
    [map], so such a rule observes the tangents under forward mode alone.

    Raises [Invalid_argument] at the call if [map]'s result differs from the
    result in its {!Nx.Ptree.val-visits}, dtypes or shapes, naming the first
    path where they differ, as in
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
      Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          (x, fun g -> Nx.clamp ~min:(-.c) ~max:c g))
    ]}

    Every reverse mode tracking one of [args] applies [pb]. Forward mode has no
    rule for the result: around a reverse mode that applies it, as in a
    Hessian-vector product, it differentiates [rule]'s code.

    Raises [Invalid_argument] at the call if [pb]'s result differs from [args]
    in its {!Nx.Ptree.val-visits}, dtypes or shapes, naming the first path where
    they differ; if [rule] uses a value that a differentiation applying it
    tracks, as {!custom_jvp} does; and under forward mode around no such reverse
    mode, if the result has a float or complex array
    (["Rune.jvp: a custom_vjp rule has no forward derivative; give the function
      a custom_jvp rule"]). *)

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
      Rune.custom_jvp
        Nx.Ptree.(pair tensor tensor)
        Nx.Ptree.tensor
        (fun (a, b) ->
          let x = Nx.solve a b in
          let along v =
            Rune.custom_transpose Nx.Ptree.tensor Nx.Ptree.tensor
              (fun v -> Nx.solve a v)
              ~transpose:(fun w -> Nx.solve (Nx.matrix_transpose a) w)
              v
          in
          (x, fun (da, db) -> along (Nx.sub db (Nx.matmul da x))))
    ]}

    [f] and [transpose] are traced at the call, apart. A differentiation applies
    [f] to the tangents and differentiates [f]'s code along the residuals.
    Reverse mode transposes the first by applying [transpose], never [f]'s
    operations, and transposing that again applies [f].

    Raises [Invalid_argument] at the call if [transpose]'s result differs from
    [args] in its {!Nx.Ptree.val-visits}, dtypes, shapes or placements, naming
    the first path where they differ, as in
    ["Rune.custom_transpose: the root: shape [2] in the transpose's result, [3]
     in the arguments"], and if [f] or [transpose] adds to a total: each also
    runs as the other's transpose, where its additions would have no single
    place. *)

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
    one branch per lane, and {!Nx.where} selects per lane.

    A map's lanes are named by an axis, so that code inside nested maps reads
    the lanes of the map it names: {!lanes} gathers every lane's value and
    {!lane_index} gives the calling lane's position. An inner map of the same
    name hides an outer one, and a map without a name is read by neither. *)

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
    nothing. An {!iterate} whose [until] depends on the lanes raises
    [Invalid_argument].

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
    total's additions in the order they are made, starting from the running
    value of the [collect] they reach, which the chain's end replaces as the
    loop returns. So the total is the one the steps' additions give made in
    turn, under every differentiation too. Where they reach the [collect]
    through a traced function or a map, as from a loop inside another loop's
    step, a {!jit} or a {!vmap}, the chain starts from an identity that changes
    no sum (a negative zero) and reaches the [collect] as one addition. A loop
    that takes no step leaves a running value as it is, and otherwise adds that
    identity. Inside a {!vmap}, each lane makes the additions of the code it
    runs: a [collect] inside the map receives its lane's, and one outside it
    receives, as one addition, their sum over the lanes as {!Nx.sum} computes it
    along the stacked axis, or nothing from a map of no lane. *)
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
    axis 0, where [f] returns them. The final carry lies where [f] returns it.
    With [n = 0] the result is [init] and outputs of leading length [0].

    [f] is traced once, at the call, over stand-ins of [init]'s arrays and of
    [xs]'s arrays without axis 0.

    A transformation of a scan is a scan of its step's transformation, whose
    carry gains a tangent, under {!jvp}, or lanes, under {!vmap}, only for the
    carry arrays that have one at the call or that a step gives one. Reverse
    mode runs the scan once forward, keeping of each step the values its
    tangent's operations read (of an elementwise function computed in one loop,
    its operands) and none of the rows' or captures' own, then runs the step's
    transpose over the rows in reverse. A row's cotangent is stacked like the
    outputs, and a captured array's is the sum over the steps of each step's,
    added in reverse order of the steps. Where the step runs an {!iterate}, a
    transformation runs the scan's steps one by one where it is called, and
    reverse mode keeps each step's values as it would the steps written out.

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
    [c], from [init], until [until carry] holds, and returns that carry, where
    the last step returns it. [until] returns one boolean and is tested before
    each step, so an [init] that satisfies it is returned unchanged. [f] and
    [until] are traced at the call, [f] as {!scan}'s step is.

    Raises [Invalid_argument] if [max < 0]
    (["Rune.iterate: max = -1 is negative"]), if [until] returns other than one
    element, at the trace if [f]'s carry differs from the one it received as
    {!scan} says, and if [until] is still false after [max] steps
    (["Rune.iterate: until is still false after max = 50 steps"]). *)

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
    however often the backward pass computes [f] again. Where [f] runs an
    {!iterate}, a transformation runs [f]'s program where it is called, so
    reverse mode keeps its intermediate values. Inside a {!jit}, [f]'s
    operations compile with the program's, and the compiler may keep their
    values for the backward pass instead of computing them again.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument. *)

(** {1:jit Compilation} *)

val jit : ('a -> 'b, 'd) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [jit s f] is [f] compiled, [s] its signature, one structure per argument and
    one for the result. A call traces [f] over stand-ins of its arguments'
    arrays into a program and runs it compiled. Every operation [f] applies is
    staged, those over values [f] captures included, which the program reads as
    they are when it is traced.

    {b Keys.} [jit s f] keeps a program per {e key}, the abstract value of a
    call's arguments: their {!Nx.Ptree.val-visits} and reports, the static data
    of a call, each array's dtype, shape and placement, and the number of lanes
    of each map whose name the program reads through {!lanes} or
    {!lane_index}. A call retraces exactly when its key is new to [jit s f],
    over stand-ins in C order: [f]'s code, and its side effects, run once per
    key. A call that raises while it
    traces keeps no program for its key. Calls on several domains share the
    programs; a key being traced makes its other callers wait, but on the domain
    that traces it, where a call traces again for itself. With
    [RUNE_JIT_DEBUG=1] in the environment, each retrace prints on standard error
    the first difference between its key and the previous call's.

    {b Compiled programs.} A program compiles the first time a call runs it,
    then once per {e variant}: each argument's strides and offset from where its
    16 bytes of memory start, and tolk's settings that shape what it compiles,
    the search widths [BEAM] and [JITBEAM] among them. A view thus reads in
    place, without a copy, from a program compiled for its layout. Each result
    array is fresh: memory of its own in C order, at the placement eager
    computation gives it.

    {b Donation.} A call consumes the donated handles among its arguments
    ({!Nx.donate}), as one operation consumes its operands: their donors die
    when the call claims them, and a later use of one raises [Invalid_argument]
    naming [Rune.jit]. A result may take a donated argument's memory instead of
    fresh memory: where it has the argument's dtype, element count and
    placement, reads it only at each element's own index (through elementwise
    functions, casts and bitcasts that keep the width, reshapes in C order,
    copies, placements, a scatter's target or a contraction's [init]) or is the
    argument itself, and no other result or check reads it; or where it reads no
    donated argument and nothing reads that one. A donor whose memory another
    value reads, or that is a view off 16 bytes or out of C order, lends
    nothing, and its result is fresh. Each donated argument lends to one result
    at most, the results of a scatter first. Its values are the same either way.

    {b Checks.} A check the function makes ({!Nx.check}) on the host is computed
    with the program, and the first that fails, in the order the function made
    them, raises its exception when the call returns, with the data at its first
    failing index, as eagerly; the call's results are then lost. A program with
    a check on another set, which raises at a later wait there, is evaluated.

    {b Numerics.} A compiled call gives eager's bits, but for a NaN's sign and
    payload, where both compute exactly: movements, selections, comparisons,
    casts, extremes, integer and boolean arithmetic, and the correctly rounded
    float operations of IEEE 754 ([Nx.add], [Nx.sub], [Nx.mul], [Nx.div],
    [Nx.sqrt]). Another float function is within a bound in ulps of the
    correctly rounded result, from 2 to 21 by function at [float32] and
    [float64] (exp 9, sin 4, tanh 18, expm1 21), and a power's bound grows with
    its exponent; special values are exact. A float sum, product or contraction
    adds its terms in an order the compiler chooses, a sum of [n] terms within
    [γ(n - 1) Σ|x|] of the exact one, [γ(k) = k u / (1 - k u)]. Narrow floats
    compute at [float32] and round once.

    {b Transformations.} Under {!jvp}, {!vmap} and reverse mode, a call is a
    call of a program derived from its key's: under {!jvp} the program's
    derivative, under {!vmap} its map, and under reverse mode the part that
    reads no tangent, which computes at the call and returns the values the
    rest reads, and the rest, linear in the tangents, which the backward pass
    transposes into one more program.
    Each derived program compiles as the key's does. A {e derivation} is a
    value: its kind, which arguments have a tangent or lanes, their dtypes,
    shapes and placements, the number of lanes of each map the program reads,
    and the name of the map it is derived under where it reads that name. A
    call keeps one program per derivation of its key's, so two calls share a
    derived program only when their derivations are equal, and [f] runs once
    per key, whatever is derived from it. Inside an enclosing [jit], a
    {!scan}'s step or another traced function, a call over the values it
    traces is inlined: its program's operations, or its derived program's, are
    staged there. A program that holds an {!iterate} is evaluated under a
    transformation, which receives its operations.

    {b Loops and rules.} A {!scan} or an {!iterate} compiles as one loop of the
    program, its step compiled once whatever its number of steps; a step reads
    its rows and writes its outputs in place, and the carry is updated in place
    each step. A check in a step raises, once the call returns, the first
    failure of the first step that fails, as the steps written out would. A
    {!remat}, a {!custom_jvp}, a {!custom_vjp} and a
    {!custom_transpose} compile as their function's operations, the program's
    own.

    {b Evaluated programs.} A program that holds a map's {!lanes} or
    {!lane_index}, a check in a construct inside a loop's step, an operation
    the compiler does not lower, values on several devices, or one tolk cannot
    compile, runs as its operations evaluated by nx, giving eager's values; a
    result that is an argument is a copy of it, and a call with donated
    arguments claims them first, consumes them and lends nothing.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument, and at
    a call if [f] captures a value that another transformation traces. *)

(** {2:inspect Inspecting a trace} *)

type program
(** The type for traced programs. *)

val program :
  ('a, 'd) Nx.Ptree.t -> ('b, 'd) Nx.Ptree.t -> ('a -> 'b) -> 'a -> program
(** [program a b f x] is the program that {!jit} traces of [f], a function of
    structure [a] to structure [b], at a call over [x], without compiling or
    computing it: [f] runs once, over stand-ins of [x]'s arrays in C order. What
    [f] stages from {!jvp}, a {!scan}'s step or another construct is in it, a
    construct with the programs it carries, for reading what a function
    computes.

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
    dtype, shape and placement, as {!jit} traces it, computing nothing: its
    shape, dtype and placement answer ({!Nx.shape}, {!Nx.dtype},
    {!Nx.placement}), and an operation on it or a read raises
    [Invalid_argument]. An array [f] captured and returns unchanged is that
    array.

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
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t
(** [jit' f] is {!jit} of a function of one array. *)

(**/**)

(** The lowering to tolk, for its suites. *)

module Lower = Lower
module Lower_arith = Lower_arith

val derivations : unit -> int
(** [derivations ()] is the number of programs that transformations of
    compiled calls derived on the calling domain, for the suites. *)

(**/**)
