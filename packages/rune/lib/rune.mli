(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Functional transformations of tensor functions: differentiation,
    vectorization and compilation.

    A transformation enumerates the tensors of some of the values it handles,
    and takes the structure of each such value, an {!Nx.Ptree.t}; everything
    else the function uses is captured and is a constant of the transformation.
    - {!grad}, {!vjp}, {!jvp} and the forms they extend take the structure of
      the value they differentiate and of the result they rebuild.
    - {!val-vmap}, {!remat} and {!val-jit} take the signature
      ({!Nx.Ptree.type-fn}) of the function they transform and return a function
      of the same type.
    - {!scan} takes the structures of its carry, rows and outputs.
    - A function of one tensor has its own form of the transformations that take
      one structure or signature: {!grad'}, {!vjp'}, {!jvp'}, {!vmap'},
      {!scan'}, {!jit'}, ...

    Tensors of a structure may have different dtypes: one forward and backward
    pass produces gradients for all of them.

    {[
    type 'a linear = { w : 'a; b : 'a }

    module Linear = struct
      type 'a t = 'a linear

      let walk c { w; b } =
        let open Nx.Ptree.Walk in
        let w = field c "w" leaf w in
        let b = field c "b" leaf b in
        { w; b }
    end

    let linear = Nx.Ptree.instantiate (module Linear)
    let grads = Rune.grad linear loss params
    ]}

    {b Arguments are positions.} A transformation tracks each tensor of its
    arguments at its position. A tensor behind two positions is two arguments,
    each with its own derivative, and a tensor the function captures is a
    constant even when it is also an argument: [grad' (fun x -> Nx.mul x w) w]
    is [w]. Tie weights by structure, one position used twice by the function.

    {b Values stay inside.} A value a transformation computes inside its
    function leaves it through the function's result. Kept in a reference, used
    on another fiber, thread or domain, or held by a closure that runs after the
    transformation returns, it has no bytes: using it raises
    [Invalid_argument "a traced tensor has no bytes; it was used outside the
     trace that made it"]. Inside the function {!Nx.item} and {!Nx.print} read
    it, so an OCaml [if] on a value differentiates, except inside {!val-vmap} on
    a value that depends on the lanes and inside {!val-jit} on one that depends
    on the arguments.

    {b Derivatives are fresh.} A gradient, a pullback's result and a tangent are
    values of their own, in C order from the start of their storage, never a
    view such as the transpose or the broadcast their computation ends with:
    [Nx.reshape [| -1 |]] takes each as it is. Under another transformation they
    are its values, and their layout is its own.

    {b Kinks.} At a point where an operation has no derivative, the
    transformations use these: [Nx.relu] and [Nx.abs] have derivative [0] at
    [0], and where the operands of [Nx.maximum] or [Nx.minimum] tie, each takes
    half of the derivative, as the tied elements along the axes of [Nx.max] and
    [Nx.min] share it equally.

    {b Nesting.} Transformations nest in any order, and each differentiates,
    maps or compiles only the values of its own function: in
    [grad' (fun x -> Nx.mul x (grad' (fun y -> Nx.add x y) one)) one], the inner
    gradient is a constant to the outer {!grad'}, and the result is [1]. Code
    outside the function a transformation receives is never transformed.

    {b Errors.} Every message starts with the entry point the caller applied,
    such as ["Rune.grad'"] for {!grad'} and ["Rune.jacrev'"] for {!jacrev'}. A
    scalar is a tensor with exactly one element, of any shape: [[||]], [[|1|]]
    and [[|1; 1|]] all are. *)

(** {1:reverse Reverse-mode differentiation} *)

val grad : 'p Nx.Ptree.t -> ('p -> ('c, 'd) Nx.t) -> 'p -> 'p
(** [grad p f params] is the gradient of [f] at [params], a value of structure
    [p] with [params]' dtypes and shapes. Tensors of [params] that do not
    contribute to the result have all-zero gradients.

    To differentiate with respect to several values, pass them as one:
    [grad Nx.Ptree.(pair p q) (fun (a, b) -> loss a b x) (a0, b0)] is the pair
    of their gradients.

    Gradients are defined for real and complex tensors; {!section-complex} says
    what one is on a complex tensor. A structure may hold others (an {!Nx.Rng.t}
    threaded through a compiled step, a counter, a batch of indices), and they
    are {e carried}: nothing accumulates into them and their gradient is zero.
    One structure then serves both [grad] and {!val-jit}, which needs such
    values as arguments.

    A gradient is the sum of the contributions its tensor receives, with no zero
    added: a single [-0.] contribution keeps its sign, so
    [grad' (fun x -> Nx.sum (Nx.mul x x))] at [-0.] is [-0.]. A tensor that
    receives none has a gradient of [+0.].

    Raises [Invalid_argument] if [params] holds no real or complex tensor
    (["Rune.grad: the parameters hold no real or complex tensor"]), or if
    [f params] is not a real or complex scalar, naming its dtype and shape
    (["Rune.grad: the objective must return a real or complex scalar, got
      float64 [2]"], ["... got int32 []"]); use {!vjp} to differentiate a result
    that is not a scalar. *)

val value_and_grad :
  'p Nx.Ptree.t -> ('p -> ('c, 'd) Nx.t) -> 'p -> ('c, 'd) Nx.t * 'p
(** [value_and_grad p f params] is [(f params, grad p f params)], computed in
    one forward and one backward pass. It raises as {!grad} does. *)

val value_and_grad_aux :
  'p Nx.Ptree.t ->
  'x Nx.Ptree.t ->
  ('p -> ('c, 'd) Nx.t * 'x) ->
  'p ->
  ('c, 'd) Nx.t * 'p * 'x
(** [value_and_grad_aux p x f params] is [(y, grad, aux)], where
    [(y, aux) = f params] and [grad] is the gradient of [y] at [params]. [aux],
    of structure [x], leaves the differentiation as its values: it contributes
    nothing to the gradient. It raises as {!grad} does. *)

val vjp : 'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'q * ('q -> 'p)
(** [vjp p q f params] is [(f params, pullback)]. [pullback cts] is the
    vector-Jacobian product of [f] at [params] against [cts], a value of
    structure [p], the adjoint of {!jvp} (see {!section-complex}). [cts] has the
    result's structure [q]: one cotangent per tensor of the result, of that
    tensor's dtype and shape.

    The cotangent of an integer or boolean tensor of the result is checked like
    the others and ignored.

    [pullback] runs no part of [f] again, except the functions [f] passes to
    {!remat}. When [vjp] runs outside every transformation, [pullback] may be
    applied any number of times, from any domain, several at once. Under another
    transformation it is transformed: under {!val-vmap} the backward pass is
    batched, under {!jvp} differentiated.

    Raises [Invalid_argument] if [params] holds no real or complex tensor.
    [pullback] raises [Invalid_argument] if [cts] and the result differ in their
    visits ({!Nx.Ptree.visits}), naming the first path where they differ and
    what each holds there, as in
    ["Rune.vjp: the root: length 2 in the result, length 1 in the cotangents"],
    or if a cotangent differs from its result tensor in dtype or shape. *)

(** {1:forward Forward-mode differentiation} *)

val jvp : 'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'p -> 'q * 'q
(** [jvp p q f params tangents] is [(f params, dy)], where [dy] is the
    Jacobian-vector product of [f] at [params] against [tangents], the
    directional derivative (see {!section-complex}), computed in one forward
    pass. [tangents] has [params]' structure, dtypes and shapes; [dy] has the
    result's structure [q], one tangent per tensor of the result: zeros of its
    dtype and shape for an integer or boolean tensor and for one that does not
    depend on [params]. The tangent of an integer or boolean parameter is
    ignored.

    A tangent dies with the value it belongs to, so a fold that [f] runs holds
    the tangents of one step at a time.

    Raises [Invalid_argument] if [params] holds no real or complex tensor, if
    [tangents] and [params] differ in their visits
    (["Rune.jvp: b: None in the parameters, Some in the tangents"]), or if a
    tangent differs from its parameter in dtype or shape. *)

(** {1:complex Complex tensors}

    A complex tensor is two real components per element, so a function of one is
    a function of twice as many real numbers, and its derivative is a real
    linear map on them. Rune packs a pair of real components [(re, im)] as the
    complex number [re + i*im], and measures these vectors with the real inner
    product [Re (sum (conj u * v))].

    A {e tangent} is a displacement. {!jvp} takes [dre + i*dim] and returns the
    directional derivative: move the input by [h] times the tangent and the
    components of the result move by [h] times what it returns.

    A {e gradient} is the vector of partial derivatives in the same packing. For
    a real-valued objective [l], {!grad} returns [dl/dre + i*dl/dim], the
    direction in which [l] grows fastest, so [z - lr * g] descends. The gradient
    of [|z|] is [z / |z|], that of [|z|^2] is [2 z], and that of [Re (c * z)] is
    [conj c].

    {!vjp}'s pullback is the adjoint of {!jvp} under that inner product: if
    {!jvp} maps a tangent [v] to [dy] and the pullback maps a cotangent [w] to
    [g], then [Re (sum (conj w * dy)) = Re (sum (conj g * v))]. Equivalently,
    [g] is the gradient of the real objective [Re (sum (conj w * f params))]. A
    complex-differentiable [f] with derivative [f'] pulls [w] back to
    [conj (f' z) * w]. {!grad} seeds the result with [1], so a complex-valued
    objective is differentiated through its real part. A {!custom_vjp} rule's
    pullback receives cotangents and returns gradients in this sense.

    On real tensors the imaginary parts are zero and none of this changes the
    derivatives. *)

(** {1:factorisations Derivatives of factorisations}

    The tangents of {!Nx.cholesky}, {!Nx.qr}, {!Nx.lu}, {!Nx.svd}, {!Nx.eigh}
    and {!Nx.eig} are the derivatives of the factors nx computes, where those
    are differentiable:
    - [Nx.qr ~mode:`Complete] of a tall matrix and [Nx.svd ~full_matrices:true]
      of a non-square one have none: their tangents raise [Invalid_argument].
    - The tangents of singular vectors and eigenvectors are non-finite where two
      singular values or eigenvalues are equal; the tangents of the values are
      finite there and depend on the vectors nx chose.
    - A vector is defined up to its sign, or on complex values its phase, which
      nx does not fix; the tangent does. The tangent of each eigenvector that
      {!Nx.eigh} and {!Nx.eig} give is orthogonal to it ([Qᴴ dQ] has a zero
      diagonal), so an eigenvector of [eig] keeps its unit norm. The tangent of
      each right singular vector that {!Nx.svd} gives is orthogonal to it, and
      the left one carries the change of phase that keeps [a = u diag(s) vh].

    {!Nx.mod_}'s tangent is [da - trunc (a / b) db], one-sided at a multiple of
    [b]. *)

(** {1:jacobians Jacobians} *)

val jacfwd' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('c, 'd) Nx.t
(** [jacfwd' f x] is the Jacobian of [f] at [x], with shape
    [shape (f x) @ shape x], computed column by column in forward mode (one
    vectorized pass). Its dtype is the dtype of [f x]. Prefer it when the input
    is smaller than the output.

    A Hessian is [jacfwd' (grad' f) x], and a Hessian-vector product
    [snd (jvp p p (grad p f) params v)].

    Raises [Invalid_argument] as {!jvp'} does. *)

val jacrev' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [jacrev' f x] is the Jacobian of [f] at [x], with shape
    [shape (f x) @ shape x], computed row by row in reverse mode (one forward
    pass, one vectorized backward pass). Its dtype is the dtype of [x]. Prefer
    it when the output is smaller than the input. For a complex-differentiable
    [f] it is the complex derivative, as {!jacfwd'} computes it: row [k] is the
    conjugate of the gradient of [Re y_k].

    Raises [Invalid_argument] as {!vjp'} does. *)

(** {1:control Controlling differentiation} *)

val detach : ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [detach x] is [x] with a zero derivative under every differentiation around
    the call. It copies nothing: outside every transformation, [detach x] is [x]
    itself. Use it to hold a value constant inside a differentiated function,
    such as the running statistics of a batch normalization or the input of an
    operation whose derivative has no definition. *)

val check_grads :
  ?eps:float ->
  ?tol:float ->
  'p Nx.Ptree.t ->
  ('p -> ('c, 'd) Nx.t) ->
  'p ->
  (unit, string) result
(** [check_grads p f params] compares the reverse-mode gradient of the scalar
    objective [f] at [params] against central-difference directional derivatives
    along two deterministic directions. It is [Ok ()] if they agree within [tol]
    (relative, default [1e-2]) and [Error msg] otherwise, where [msg] names the
    direction and both derivatives. [eps] is the finite-difference step (default
    [1e-4]).

    The check is directional, not per element: it validates a gradient cheaply
    rather than exhaustively. Use float64 parameters for reliable results;
    float32 may need a looser [tol].

    Raises [Invalid_argument] as {!grad} does. *)

(** {1:custom Custom differentiation rules}

    A custom rule gives a function the derivative a transformation would compute
    for it, in the form of that transformation's answer: a tangent map for
    forward mode, a pullback for reverse mode. Each rule receives the arguments'
    values, with no derivative attached, and must not use a value its own
    differentiation tracks other than through them: pass such a value as an
    argument.

    An exception of a rule, of its tangent map or of its pullback is raised at
    the call or application that ran it. *)

val custom_jvp :
  'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q * ('p -> 'q)) -> 'p -> 'q
(** [custom_jvp p q rule args] is [fst (rule args)], a value of structure [q],
    whose derivative is the tangent map [snd (rule args)]: [map dargs] is the
    result's tangent for the arguments' tangents [dargs], of structure [p], and
    must be linear in them. [map] receives zeros for an argument the
    differentiation does not track, and its result is checked against [q].

    {[
    let softplus =
      Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          (stable x, fun dx -> Nx.mul (Nx.sigmoid x) dx))
    ]}

    The rule serves both modes at every order. Forward mode applies [map];
    reverse mode transposes the operations [map] issues; and every
    differentiation around the call that tracks one of [args] applies the rule
    too, differentiating [map]'s code for the second-order terms. A Hessian
    through [softplus] is the rule's derivative. Under reverse mode, when the
    result holds no tensor, [map] is not applied: such a rule observes the
    arguments' tangents in forward mode and is inert under {!grad}. {!val-vmap}
    batches the rule.

    Under reverse mode, a loop in [map] ({!scan}) runs written out, even under
    {!val-jit}, and a value that [map] selects with {!Nx.where}, concatenates,
    scatters or writes beside a tangent is taken as zero: [map] must give such a
    value only as a tangent's zero fill.

    Raises [Invalid_argument] if [map]'s result differs from the result in its
    visits, or a tangent from its result tensor in dtype or shape; and if [rule]
    uses a value its own differentiation tracks
    (["Rune.custom_jvp: the rule uses a value its own differentiation tracks;
      pass it as an argument"]). Under reverse mode it raises at the operation,
    naming the entry point that differentiates, if [map] applies an operation
    that is not linear in the tangents
    (["Rune.grad: a custom_jvp tangent map applies exp to a tangent; a tangent
      map must be linear in its tangents"]), adds a value to a tangent or pads
    one with a nonzero fill, which are affine, adds a tangent to a {!Total}, or
    reads a tangent's value
    (["Rune.grad: a custom_jvp tangent map reads a tangent's value with Nx.item;
      under reverse mode a tangent has none"]). *)

val custom_vjp :
  'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q * ('q -> 'p)) -> 'p -> 'q
(** [custom_vjp p q rule args] is [fst (rule args)], a value of structure [q],
    whose reverse-mode derivative is the pullback [snd (rule args)]: [pb cts] is
    the gradient for the result's cotangents [cts], as {!vjp}'s pullback
    computes one (see {!section-complex}). [cts] holds zeros for a tensor of the
    result that nothing used, and [pb]'s result is checked against [p].

    {[
    let clip_grad c =
      Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          (x, fun g -> Nx.clamp ~min:(-.c) ~max:c g))
    ]}

    Differentiations around the call differentiate the rule's code and its
    pullback's. {!val-vmap} batches the rule, and the gradient of an argument
    that is not batched is summed over the lanes.

    Raises [Invalid_argument] if [pb]'s result differs from [args] in its visits
    or a gradient from its argument in dtype or shape; under forward mode, if
    the result holds a tensor
    (["Rune.jvp: a custom_vjp rule has no forward derivative; give the function
      a custom_jvp rule"]); and if [rule] uses a value its own differentiation
    tracks, as {!custom_jvp} does. *)

(** {1:remat Gradient checkpointing} *)

val remat : ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [remat s f] is [f], recomputed during the backward pass instead of having
    its intermediate results retained: reverse-mode differentiation of
    [remat s f] keeps [f]'s arguments and runs [f] again at them when the
    backward pass reaches it, trading compute for memory. [s] is [f]'s
    signature, as for {!val-vmap}. Every transformation sees [remat s f] as it
    sees [f]: its derivatives in either mode, including those with respect to
    tensors [f] captures, and its batched form under {!val-vmap} are [f]'s. An
    addition to a {!Total} that [f] makes counts once, however often [f] runs.

    Under {!val-jit}, the backward pass reads the arguments again only once the
    cotangents of [f]'s result exist, so [f]'s intermediates are live for one
    run at a time. A remat whose arguments are all arguments or constants of the
    compiled function reads them directly, and so does one inside the body of a
    compiled {!scan}, whose backward loop recomputes each step already.

    Raises [Invalid_argument] when applied to [s] if [s] consumes an argument
    ({!Nx.Ptree.consumes}). *)

(** {1:vmap Vectorizing maps} *)

type axis
(** The type for names of maps. *)

val axis : unit -> axis
(** [axis ()] is a fresh name, distinct from every other. *)

val vmap : ?axis:axis -> ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [vmap ?axis s f] is [f] mapped over axis 0 of every tensor of its arguments:
    a function of [f]'s type whose result is the loop of [f] over the rows,
    stacked. [axis] names the map for {!lanes} and {!lane_index}; a map without
    one is anonymous. [s] is [f]'s signature, one structure per argument and one
    for the result:

    {[
    let per_example params =
      Rune.vmap
        Nx.Ptree.(tensor @-> tensor @-> returns mlp)
        (fun x y -> Rune.grad mlp (fun p -> Loss.mse (Mlp.apply p x) y) params)
    ]}

    [f] is written for unbatched values: it sees each argument tensor without
    its axis 0, its {e lane}, and each tensor of its result gains a batch axis
    0. A value [f] captures is a constant of the map, and a result tensor that
    does not depend on the arguments is broadcast along the batch axis. To map
    another axis, move it to the front with {!Nx.moveaxis}, a view; to keep a
    value whole, capture it.

    Randomness a lane captures (an {!Nx.Rng.t}, or [Nx.rand] under a scope the
    map captures) draws {e identical} values for every lane: it is a constant of
    the map. Decorrelate them either by folding the lane index into one key,
    [Nx.Rng.fold_in_tensor k (Rune.lane_index ())], or by mapping over a batch
    of keys from {!Nx.Rng.split_batch}, walked with {!Nx.Rng.ptree}: each lane
    sees one key.

    Reading a lane's value inside [f] raises [Invalid_argument] with a message
    that starts with the name of the function that read, as in
    ["Nx.item: cannot read the value of a batched tensor inside vmap; return it
     from the mapped function instead"]: an OCaml [if] on a value that depends
    on the lanes raises, and {!Nx.where} selects per lane.

    Raises [Invalid_argument] when applied to [s] if [s] consumes an argument;
    and when applied to its arguments if they have no tensor, if a tensor is a
    scalar, or if two tensors differ in the length of their axis 0, naming each
    tensor by its path ({!val-jit}):
    ["Rune.vmap: 1: 3 rows along axis 0, 0: 2"]. *)

val lanes : axis -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [lanes a x], inside the map named [a] of [n] lanes, is every lane's [x]
    stacked on a new leading axis of length [n]: the same value in every lane, a
    constant of that map. [x] of shape [s] gives shape [n :: s]; an [x] every
    lane shares gives [n] copies of it.

    Maps between the call and the map named [a] keep their own lanes: under an
    anonymous map of [m] lanes inside the map named [a], each of the [m] lanes
    gathers its own [x] across [a]. With no map named [a] around the call there
    is one lane, and [lanes a x] is [Nx.unsqueeze ~axes:[0] x].

    [lanes a] is linear, and both modes differentiate it as such: the tangent of
    [lanes a x] is [lanes a dx], and under reverse mode inside the map named [a]
    the cotangent of [x] is the calling lane's row of the sum over the lanes of
    their cotangents, so a lane's gradient collects every lane's use of its [x].
    [Nx.sum ~axes:[0] (lanes a x)] is the sum of [x] over the lanes of [a]. *)

val lane_index : ?axis:axis -> unit -> (int32, Nx.int32_elt) Nx.t
(** [lane_index ?axis ()] is the calling lane's index in the map named [axis],
    or in the innermost anonymous map when [axis] is absent: an [int32] scalar,
    from [0] to the map's number of lanes minus one. With no such map around the
    call there is one lane, and it is [0].

    [Nx.Rng.fold_in_tensor k (lane_index ())] gives each lane its own key from a
    key [k] the map captures. *)

(** {1:totals Totals} *)

(** Write-only sums.

    A total is a sum that code anywhere inside a function adds to and that the
    caller reads when the function returns: {!Total.collect}[ t ~zero f] is
    [f ()] with [zero] plus everything [f] added to [t]. Nothing reads a total
    before its [collect] returns, so an addition never changes a value the
    function computes, and with no [collect] open it does nothing. Totals let
    code deep inside a function report to its caller across the transformations
    between them without threading a value through every function in between.

    {[
    let saturated = Rune.Total.make ()

    let cell params h x =
      let h = Nx.tanh (Nx.add (Nx.matmul h params.w) (Nx.matmul x params.u)) in
      Rune.Total.add saturated
        (Nx.mean (Nx.cast Nx.float32 (Nx.greater (Nx.abs h) threshold)));
      h

    let train_step =
      Rune.jit sig_ (fun params batch ->
          let (l, g), sat =
            Rune.Total.collect saturated ~zero:(Nx.zeros Nx.float32 [||])
              (fun () ->
                Rune.value_and_grad params_ptree (fun p -> loss p batch) params)
          in
          (update params g, l, sat))
    ]}

    An addition counts once per execution of the code that makes it, whatever
    the transformations between it and the scope:
    - {!jvp} passes on its value; it has no tangent.
    - {!val-vmap} passes on the sum of its lanes' additions: a value batched
      across the lanes summed over them, one every lane shares times their
      number. A transformation built on a map counts per lane: {!jacfwd'} counts
      an addition once per column, and {!jacrev'} once.
    - {!grad} and the other reverse-mode transformations pass it on when they
      first run the code that makes it, and drop it when they run that code
      again: the backward pass of a compiled {!scan} and a {!remat}
      recomputation.
    - A {!scan} that a compiled function stages, and a {!remat}, carry the sum
      of their additions out as a value, so a staged loop stays one loop and a
      replay computes the total again.

    A scope inside a transformation is ordinary arithmetic to it: the collected
    total is differentiated under {!jvp}, under {!grad}, computed per lane under
    {!val-vmap} (the map returns the totals stacked) and returned by a compiled
    function like any value. *)
module Total : sig
  type ('a, 'b) t
  (** The type for totals of [('a, 'b) Nx.t] values. *)

  val make : unit -> ('a, 'b) t
  (** [make ()] is a fresh total, distinct from every other. *)

  val add : ('a, 'b) t -> ('a, 'b) Nx.t -> unit
  (** [add t v] adds [v] to the innermost open {!collect} of [t], and does
      nothing if none is open.

      Raises [Invalid_argument] if [v]'s shape, as the scope sees it, is not its
      [zero]'s, and if [v] is a tangent under reverse mode, which only a
      {!custom_jvp} tangent map holds
      (["Rune.Total.add: a custom_jvp tangent map adds a tangent under reverse
        mode; a total takes values"]). *)

  val collect :
    ('a, 'b) t -> zero:('a, 'b) Nx.t -> (unit -> 'r) -> 'r * ('a, 'b) Nx.t
  (** [collect t ~zero f] is [(f (), total)], where [total] is [zero] plus every
      addition [f] made to [t]. [zero] gives the total's dtype, shape and
      placement.

      A {!scan} inside [f] that no compiled function stages folds inside the
      scope. A {!val-jit} inside [f] runs its function eagerly: to compile, open
      the scope inside the function {!val-jit} compiles and return the total.

      The scope checks each addition's shape against [zero]'s when it receives
      it, where every map inside the scope has summed its lanes.

      An addition a custom rule's function makes ({!custom_jvp}'s and
      {!custom_vjp}'s [rule], a tangent map, a pullback) reaches the scopes
      around the transformation that runs it, and skips a scope opened between
      the call and that transformation.

      If [f] raises, [collect] raises the same exception; an addition made
      before an exception [f] itself catches counts. *)
end

(** {1:flow Loops and branches}

    A branch on a value is OCaml's [if] on {!Nx.item}, and a loop whose length
    depends on a value is recursion: both run under {!grad} and {!jvp}, which
    differentiate the path taken. A predicate that depends on a map's lanes
    raises, one that depends on a compiled function's arguments raises
    {!Jit_error}, and {!Nx.where} selects everywhere. *)

val scan :
  'c Nx.Ptree.t ->
  'x Nx.Ptree.t ->
  'y Nx.Ptree.t ->
  f:('c -> 'x -> 'c * 'y) ->
  init:'c ->
  'x ->
  'c * 'y
(** [scan c x y ~f ~init xs] folds [f] over the rows of [xs], a value of
    structure [x]: every tensor of [xs] has the same leading length [n], and
    step [i] passes [f] row [i] of every tensor. [f carry row] returns the next
    carry, of structure [c], and the step's outputs, of structure [y]; the
    result is the final carry and the outputs, every tensor stacked along a new
    axis 0. A fold with nothing to emit passes {!Nx.Ptree.unit} for [y] and
    returns [()].

    Every carry the body returns has the visits ({!Nx.Ptree.visits}) of the one
    it received, and every step's outputs have the first step's: a list keeps
    its length, an option its presence, a case and an integer their value.

    Under {!val-jit} the body compiles once and runs as a loop in the compiled
    program, and differentiating compiles a reversed loop that runs each step
    again at its carry. {!jvp}, {!val-vmap} and {!grad} of a scan compile as one
    loop too, whose carry gains a tangent or a lane only for the carry tensors
    that have one. The loop reads row [i] of each tensor of [xs] in place, so
    data that differs per step, such as the weights of stacked layers, belongs
    in [xs]; the cotangent of [xs] is stacked like the outputs, while a captured
    tensor's cotangent is the sum over the steps. A carry tensor the step
    updates with {!Nx.set}, or reads only at the index it writes, is updated in
    place. A compiled function writes the loop out instead, step by step, each
    step's carry stored before the next step reads it, when the carry changes
    its shapes across steps, when the body runs on a device with command queues
    and on the host, or on devices of two kinds, inside the body of a loop it
    compiles (an inner scan runs step by step within each step of the outer
    loop), and inside a {!custom_jvp} tangent map under reverse mode.

    Everywhere else the scan is its loop, run where it is written, inside every
    transformation, {!Total.collect} and {!Nx.Rng.with_key} around it.

    Raises [Invalid_argument], before any step, if [xs] has no tensor
    (["Rune.scan: xs has no leaf"]), a scalar tensor
    (["Rune.scan: an xs leaf is a scalar"]), tensors of different leading
    lengths (["Rune.scan: the xs leaves differ in their leading length"]), or if
    [n] is [0] (["Rune.scan: xs is empty along the scan axis"]); and, at the
    step, if the body returns a carry whose visits or dtypes differ from the
    carry it received, or outputs whose visits, dtypes or shapes differ from the
    first step's, naming the first path where they differ and what each holds
    there, as in
    ["Rune.scan: 1: length 3 in the carry the body returned, length 2 in the
     carry it received"]. *)

(** {1:jit Compilation} *)

exception Jit_error of string
(** Raised when a function cannot be compiled, while it is traced: it reads the
    value of a traced tensor ({!Nx.item} on a value that depends on the
    arguments, or a branch on one), draws random values that do not depend on
    the arguments (from a captured {!Nx.Rng.t}, or {!Nx.Rng.with_key} on a
    constant key, at counters that do not either: the draw would be one constant
    replayed on every call), or uses an operation or dtype the target of its
    devices cannot compute. Nothing is consumed. *)

val jit :
  ?beam:int -> ?parallel:int -> ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [jit ~beam ~parallel s f] is [f] compiled, a function of [f]'s type whose
    arguments and result have the structures of the signature [s]:

    {[
    let step =
      Rune.jit
        Nx.Ptree.(
          tensor @-> tensor @-> consumes state @@ returns (pair tensor state))
        (fun ids targets (params, opt) ->
          let loss, grads =
            Rune.value_and_grad mlp (objective ids targets) params
          in
          let params, opt = Vega.adamw_step mlp ~lr opt ~params ~grads in
          (loss, (params, opt)))
    ]}

    An argument built with {!Nx.Ptree.( @-> )} is read; one built with
    {!Nx.Ptree.consumes} is given up by each call, which may write the result
    over its storage. Tensors [f] closes over are constants of the compiled
    function.

    The first application traces [f], compiles the traced computation into
    kernels and runs them. Later applications replay the program of their key.
    Apply [jit s f] once and reuse the result: its programs live in it.

    {b Paths.} A leaf of the arguments is named by its path: the argument's
    position counted from 0, then the leaf's path inside that argument. The
    window of the second argument is [1.window], and a first argument that is
    one tensor is [0]. Keys, errors and [RUNE_JIT_DEBUG] reports use these
    paths.

    {b Keys.} Two calls share a program when their arguments have the same
    leaves at the same paths, each of the same dtype, shape and placement, and
    of the same layout (its strides when its view is not C order over its whole
    storage, and where the run of storage it reaches starts within 16 bytes of
    memory), and make the same reports (an integer, a case, an option's
    presence, a list's length) at the same paths. An integer that changes on
    every call compiles a program per value; a value that varies belongs in a
    tensor. [RUNE_JIT_DEBUG=1] reports each retrace with the first difference
    from the previous call's key, such as
    ["rune.jit: retrace: 1.window: int 3 here, int 2 in the previous key"].

    {b Search.} With [beam], each kernel of the compiled function is searched
    for the optimisations that run it fastest, keeping the [beam] fastest
    candidates of each round and timing them on its device, which makes the
    first call of a key much longer; [parallel] compiles the candidates on that
    many domains. An explicit [beam], [0] (no search) included, overrides the
    [BEAM] and [JITBEAM] settings, and [parallel] the [PARALLEL] setting, which
    decide otherwise. The width is part of a call's key, so functions searched
    at different widths never share a program.

    {b Placement.} A call compiles for the memories of its placed arguments and
    captures, and for the host's when there are none: the backends their devices
    carry take no part. Every value the function computes lives where nx would
    place it, decided as it traces: a misplaced operand raises
    [Invalid_argument] with nx's message, and nothing moves between devices
    unless the function places it with {!Nx.place}. A host argument is uploaded
    on each call, as an eager operation would move it. A view is read in place,
    its strides expressed in the program.

    {b Results.} Every result leaf is a value with storage of its own. A result
    that returns a read argument or a capture unchanged is a copy, and a value
    [f] returns at two leaves comes back as two values, the second a copy of the
    first.

    {b Checks.} {!Nx.check} of a value [f] computes reads nothing as [f] traces:
    the program also returns the index of each check's first false element, and
    once it has run, the call raises [Invalid_argument] with the message of the
    first check traced that failed, its consumed arguments consumed. A check in
    a staged {!scan} reports the first step that fails. The message is made
    then, from the values [f] captured as it traced.

    {b Consumption.} Before its first kernel, a call marks every storage that a
    leaf of a consumed argument reaches as consumed; nothing unmarks it. From
    then on a read of any value over that storage, or its use as an operand or
    an argument, raises [Invalid_argument] naming the argument and the leaf's
    path; its shape and dtype stay readable. A consumed leaf must cover its
    whole storage, and no other leaf of the call nor a capture of its program
    may reach that storage: the call raises [Invalid_argument] before anything
    runs, naming both paths, as in
    ["Rune.jit: 0 is consumed and 1 reaches its storage"], and consumes nothing.
    A call that raises before its first kernel consumes nothing; once execution
    begins, consumed arguments stay consumed even if the call raises.

    {b Lending.} A result may take the storage of a consumed leaf, so a loop
    that consumes its state holds one generation of it. It does when their
    dtypes, sizes and placements are equal, the leaf's storage starts on 16
    bytes of memory, as fresh storage does, and writing the result there cannot
    change what the program still reads: the result reads the leaf only where it
    derives from it at its own index (elementwise operations, equal-width casts
    and reshapes: an optimizer update, a window written into a cache), or does
    not read it at all. Partners are chosen once per program: first the results
    of an indexed write, then the other results that derive from a consumed leaf
    at their own index, then the results that do not read one, in the order the
    function computes them. Each storage lends at most once. On a call, storage
    that cannot lend (memory the value borrows, or storage a compiled function
    binds) is copied first, and the argument is still consumed.
    [RUNE_JIT_DEBUG=1] reports each consumed leaf:
    ["rune.jit: 2.0.keys -> result 1.0.keys reused"], [copied] when its storage
    was copied.

    {b Captures.} A capture is bound once per compiled function, at the trace
    that meets it: one element becomes a constant of the program; a value placed
    where the operation computes is bound in place, with no copy; any other
    value is placed there once. The compiled function keeps what it binds
    reachable. A closure whose capture was consumed raises at its next trace.
    Mutating a captured tensor between calls has unspecified visibility: pass
    values that change between calls as arguments.

    {b Numerics.} A sum over an axis ({!Nx.sum}, {!Nx.mean}, the contraction of
    {!Nx.matmul}) is the sum of its terms in an unspecified association, so
    results differ from eager's in rounding, and at overflow in whether a term
    overflows. A maximum over an axis is exact. Compiled float results can also
    differ from eager's in the last bits where the compiler fuses a multiply and
    an add, turns a division by a constant into a multiplication, or turns a
    multiplication by a reciprocal ({!Nx.recip}) into a division, which also
    differs where the reciprocal overflows: [x * recip y] is NaN at [x = 0] and
    a [y] whose reciprocal is infinite, where [x / y] is [0]. They also differ
    in transcendental functions, which are approximations within 4 units in the
    last place of the result's dtype; Metal flushes float32 subnormals to zero.
    A product over an axis ({!Nx.prod}) multiplies in an unspecified association
    too. A failed factorisation gives non-finite values where eager raises
    {!Nx_backend.Linalg_error}.

    {b Domains.} A compiled function may be called from any domain, several at
    once, and from inside its own function. A key being compiled makes the other
    calls with that key wait, blocking their domain, so [f] must not suspend its
    fiber. A call returns once its work is queued, and a read of a result waits
    for it.

    {b Transformations.} A transformation of a compiled function compiles:
    [grad (jit s f)], [jvp], [vmap] of it and a {!Total.collect} around it each
    run programs compiled for the function the transformation derives from [f],
    kept in the compiled function by key as its own are, and consume nothing.
    Under {!grad} the forward pass is [f]'s program, and the backward pass a
    program that runs [f] again before its pullback, so [f]'s forward work runs
    twice: [jit (grad f)], differentiating {e inside} the compiled function as
    [step] above does, compiles the two passes together. Inside an outer [jit],
    [jit s f] is [f], traced into the outer program. A compiled function that
    reads, through its closure, a value a transformation around it tracks raises
    [Invalid_argument]: pass the value as an argument.

    Raises {!Jit_error} when tracing fails, and [Invalid_argument] if [s] has no
    argument, for a misplaced leaf or capture, for a name met with two devices,
    and as consumption above says. *)

(** {1:tensor Functions of one tensor}

    Each is its structured form at {!Nx.Ptree.tensor}. *)

val grad' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [grad' f x] is [grad Nx.Ptree.tensor f x].

    Raises [Invalid_argument] if [x] is neither real nor complex, or if [f x] is
    not a scalar. *)

val value_and_grad' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [value_and_grad' f x] is [value_and_grad Nx.Ptree.tensor f x]. It raises as
    {!grad'} does. *)

val vjp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * (('c, 'd) Nx.t -> ('a, 'b) Nx.t)
(** [vjp' f x] is [vjp Nx.Ptree.tensor Nx.Ptree.tensor f x]. *)

val jvp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp' f x tangent] is [jvp Nx.Ptree.tensor Nx.Ptree.tensor f x tangent]. *)

val vmap' :
  ?axis:axis ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [vmap' ?axis f x] is [vmap ?axis Nx.Ptree.(tensor @-> returns tensor) f x]:
    [f] mapped over axis 0 of [x], its result stacked along a new axis 0. *)

val scan' :
  f:(('a, 'b) Nx.t -> ('c, 'd) Nx.t -> ('a, 'b) Nx.t * ('e, 'f) Nx.t) ->
  init:('a, 'b) Nx.t ->
  ('c, 'd) Nx.t ->
  ('a, 'b) Nx.t * ('e, 'f) Nx.t
(** [scan' ~f ~init xs] is
    [scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor ~f ~init xs]. *)

val jit' :
  ?beam:int ->
  ?parallel:int ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [jit' ~beam ~parallel f] is
    [jit ~beam ~parallel Nx.Ptree.(tensor @-> returns tensor) f]. *)
