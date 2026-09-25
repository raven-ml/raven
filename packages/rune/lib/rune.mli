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
      the value they differentiate and, where they rebuild one, of the result.
    - {!val-vmap} and {!remat} take the signature ({!Nx.Ptree.type-fn}) of the
      function they transform and return a function of the same type.
    - {!scan} takes the structures of its carry, rows and outputs.
    - {!val-jit} takes the signature of the function it compiles, whose
      arguments are read or consumed ({!Nx.Ptree.consumes}).
    - A function of one tensor has its own form of most of them: {!grad'},
      {!vmap'}, {!jit'}, {!scan'}, ...

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

    {b Arguments are positions.} A transformation replaces each tensor of its
    arguments by a fresh alias, a new value over the same storage with no copy,
    before it differentiates, maps or compiles it. A tensor behind two positions
    is two arguments, each with its own gradient, and a tensor the function
    captures is a constant even when it is also an argument:
    [grad' (fun x -> Nx.mul x w) w] is [w]. Tie weights by structure, one
    position used twice by the function. *)

(** {1:reverse Reverse-mode differentiation} *)

val grad : 'p Nx.Ptree.t -> ('p -> ('c, 'd) Nx.t) -> 'p -> 'p
(** [grad p f params] is the gradient of [f] at [params], a value of structure
    [p] with [params]' dtypes. Tensors of [params] that do not contribute to the
    result have all-zero gradients.

    To differentiate with respect to several values, pass them as one:
    [grad Nx.Ptree.(pair p q) (fun (a, b) -> loss a b x) (a0, b0)] is the pair
    of their gradients.

    Gradients are defined for real and complex tensors. A structure may hold
    others (an {!Nx.Rng.t} threaded through a compiled step, a counter, a batch
    of indices), and they are {e carried}: nothing accumulates into them and
    their gradient is zero. One structure then serves both [grad] and
    {!val-jit}, which needs such values as inputs, and Vega's optimizers leave
    them alone in turn.

    Raises [Invalid_argument] if [f params] is not a scalar (a tensor with
    exactly one element); use {!vjp} to differentiate non-scalar results against
    an explicit cotangent. *)

val value_and_grad :
  'p Nx.Ptree.t -> ('p -> ('c, 'd) Nx.t) -> 'p -> ('c, 'd) Nx.t * 'p
(** [value_and_grad p f params] is [(f params, grad p f params)], computed in
    one forward and one backward pass. *)

val value_and_grad_aux :
  'p Nx.Ptree.t ->
  ('p -> ('c, 'd) Nx.t * 'aux) ->
  'p ->
  ('c, 'd) Nx.t * 'p * 'aux
(** [value_and_grad_aux p f params] is like {!value_and_grad} for an objective
    that returns auxiliary data beside its result. The auxiliary value is
    returned as it is and does not contribute to the gradient. *)

val vjp : 'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'q -> 'q * 'p
(** [vjp p q f params cts] is [(f params, g)], where [g], of structure [p], is
    the vector-Jacobian product of [f] at [params] against [cts]. [cts] has the
    result's structure [q]: one cotangent per tensor of the result, of that
    tensor's dtype and shape.

    Raises [Invalid_argument] if [cts] and the result differ in their visits
    ({!Nx.Ptree.visits}), naming the first path where they differ and what each
    holds there, as in
    ["Rune.vjp: the root: length 2 in the result, length 1 in the cotangents"];
    or if a cotangent differs from its result tensor in dtype or shape. *)

val vjp_fun :
  'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'q * ('q -> 'p)
(** [vjp_fun p q f params] is [(f params, pullback)]. [pullback cts] is
    [snd (vjp p q f params cts)], and may be called any number of times: each
    call runs one backward pass over the recorded computation without running
    [f] again. Calling the pullback under another transformation (for example
    {!val-vmap}) transforms the backward pass. Pullbacks are not thread-safe.

    [pullback] raises [Invalid_argument] as {!vjp} does for its cotangents. *)

(** {1:forward Forward-mode differentiation} *)

val jvp : 'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'p -> 'q * 'q
(** [jvp p q f params tangents] is [(f params, dy)], where [dy] is the
    Jacobian-vector product of [f] at [params] against [tangents], computed in
    one forward pass. [tangents] has [params]' structure, dtypes and shapes;
    [dy] has the result's structure [q], one tangent per tensor of the result.

    Raises [Invalid_argument] if [tangents] and [params] differ in their visits
    (["Rune.jvp: b: None in the parameters, Some in the tangents"]), or if a
    tangent differs from its parameter in dtype or shape. *)

val jvp_aux :
  'p Nx.Ptree.t ->
  'q Nx.Ptree.t ->
  ('p -> 'q * 'aux) ->
  'p ->
  'p ->
  'q * 'q * 'aux
(** [jvp_aux p q f params tangents] is like {!jvp} for a function that returns
    auxiliary data beside its result. The auxiliary value is returned as it is
    and has no tangent. *)

(** {1:complex Complex tensors}

    A complex tensor is two real components per element, so a function of one is
    a function of twice as many real numbers, and its derivative is a real
    linear map on them. Rune packs the two directions of that map into complex
    tensors differently.

    A {e tangent} carries the perturbation itself. {!jvp} takes and returns
    [dre + i*dim], and its result is the directional derivative: move the input
    by [h] times the tangent and both components of the output move by [h] times
    the result.

    A {e cotangent} carries the conjugate of the sensitivity. For a real-valued
    objective [l], {!grad} and {!vjp} return [dl/dre - i*dl/dim]. That sign is
    what makes the plain chain rule correct: with it, a rule that multiplies the
    cotangent by a derivative and conjugates nothing is right for every
    operation that has a complex derivative — [mul] pulls back as
    [cotangent * b], [exp] as [cotangent * exp z], [matmul] through an ordinary
    transpose. The real-valued formulas carry over unchanged. Using the result
    as a direction has to undo the conjugation: [z - lr * conj g] descends,
    while [z - lr * g] moves the imaginary component the wrong way. An objective
    that is complex-valued rather than real is seeded with a cotangent of [1],
    so for a complex-differentiable objective {!grad} is its complex derivative.

    Operations with no complex derivative carry the conjugation explicitly.
    [abs] is the modulus: real-valued, and its differential mixes the two
    components rather than scaling by one complex number. It pulls back through
    [conj (sign z)] — through [sign z] the imaginary contribution would come
    back negated — and keeps only the real part of the cotangent, since a
    real-valued output cannot move in the imaginary direction; in forward mode
    it produces a real tangent. Both are the identity on real dtypes, so real
    gradients are unaffected. *)

(** {1:vmap Vectorizing maps} *)

val vmap : ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [vmap s f] is [f] mapped over axis 0 of every tensor of its arguments. [s]
    is [f]'s signature, one structure per argument and one for the result:

    {[
    let per_example =
      Rune.vmap
        Nx.Ptree.(tensor @-> tensor @-> returns linear)
        (fun x y -> Rune.grad linear (loss x y) params)
    ]}

    [f] is written for unbatched values: it sees each argument tensor without
    its axis 0, and each tensor of its result gains a batch axis 0. A value [f]
    captures is a constant of the map, and a result tensor that does not depend
    on the arguments is broadcast along the batch axis. To map another axis,
    move it to the front with {!Nx.moveaxis}, a view; to keep a value whole,
    capture it.

    Composes with the other transformations: [vmap] of {!grad} computes
    per-example gradients, and {!grad} of [vmap] differentiates through the map.

    {b Note.} Randomness a lane captures (an {!Nx.Rng.t}, or [Nx.rand] under a
    scope the map captures) draws {e identical} values for every lane: it is a
    constant of the map. Decorrelate them either by folding the lane index into
    one key with {!Nx.Rng.fold_in_axis}, or by mapping over a batch of keys from
    {!Nx.Rng.split_batch}, walked with {!Nx.Rng.ptree}: each lane sees one key.
    Reading a batched tensor's value inside the mapped function raises.

    Raises [Invalid_argument] when applied to [s] if [s] consumes an argument
    ({!Nx.Ptree.consumes}); and when applied to its arguments if they have no
    tensor, if a tensor is a scalar, or if two tensors differ in the length of
    their axis 0, naming each tensor by its path, as {!val-jit}'s messages do:
    ["Rune.vmap: 1: 3 rows along axis 0, 0: 2"]. *)

val vmap' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('c, 'd) Nx.t
(** [vmap' f x] is [vmap Nx.Ptree.(tensor @-> returns tensor) f x]: [f] mapped
    over axis 0 of [x], its result stacked along a new axis 0.

    Raises [Invalid_argument] if [x] is a scalar. *)

(** {1:custom Custom differentiation rules} *)

val custom_vjp :
  'p Nx.Ptree.t ->
  'q Nx.Ptree.t ->
  fwd:('p -> 'q * 'res) ->
  bwd:('res -> 'q -> 'p) ->
  'p ->
  'q
(** [custom_vjp p q ~fwd ~bwd params] is [fst (fwd params)], a value of
    structure [q], with a user-defined reverse rule. Under the innermost
    reverse-mode transformation, [fwd]'s operations are not differentiated;
    [bwd residual cts] gives the gradients instead. [cts] holds the result's
    cotangents, of structure [q], zero for a tensor of the result that nothing
    used; the gradients have structure [p], and each tensor its parameter's
    dtype and shape. [residual] is what [fwd] returned beside its result.
    Enclosing transformations (an outer {!grad}, {!val-vmap}) see the forward
    computation itself.

    A tensor of the result that is one of [params] is a new value there: its
    cotangent is the result's alone.

    Raises [Invalid_argument] if the call is differentiated in forward mode
    (define a {!custom_jvp} rule for that), or if [bwd]'s gradients differ from
    [params] in their visits or in a tensor's dtype. *)

val custom_jvp :
  'p Nx.Ptree.t ->
  'q Nx.Ptree.t ->
  f:('p -> 'q) ->
  jvp:('p -> 'p -> 'q * 'q) ->
  'p ->
  'q
(** [custom_jvp p q ~f ~jvp params] is [f params], a value of structure [q],
    with a user-defined forward rule. Under the innermost forward-mode
    transformation, [jvp params tangents] gives both the result and its
    tangents, of structure [q], in place of [f]'s operations. A tensor of the
    result that is one of [params] is a new value there: the parameter keeps its
    own tangent.

    Raises [Invalid_argument] if the call is differentiated in reverse mode
    (define a {!custom_vjp} rule for that), or if [jvp]'s tangents differ from
    its result in their visits, or a tangent from its result tensor in dtype or
    shape. *)

(** {1:tensor Single-tensor variants} *)

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
  ('c, 'd) Nx.t ->
  ('c, 'd) Nx.t * ('a, 'b) Nx.t
(** [vjp' f x ct] is [vjp Nx.Ptree.tensor Nx.Ptree.tensor f x ct]. *)

val vjp_fun' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * (('c, 'd) Nx.t -> ('a, 'b) Nx.t)
(** [vjp_fun' f x] is [vjp_fun Nx.Ptree.tensor Nx.Ptree.tensor f x]. *)

val jvp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t * ('c, 'd) Nx.t
(** [jvp' f x tangent] is [jvp Nx.Ptree.tensor Nx.Ptree.tensor f x tangent]. *)

(** {1:remat Gradient checkpointing} *)

val remat : ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [remat s f] is [f], recomputed during the backward pass instead of having
    its intermediate results retained: reverse-mode differentiation of
    [remat s f] keeps [f]'s arguments and runs [f] again when the backward pass
    reaches it, trading compute for memory. [s] is [f]'s signature, as for
    {!val-vmap}. Every transformation sees [remat s f] as it sees [f]: its
    derivatives in either mode, including those with respect to tensors [f]
    captures, and its batched form under {!val-vmap} are [f]'s.

    Under {!val-jit}, reverse mode materialises the arguments and reads them
    again only once the cotangents of [f]'s result exist, so [f] runs again in
    the backward pass and its intermediates are live for one run at a time. A
    remat whose arguments are all inputs or constants of the compiled function
    is not recomputed, nor is one whose cotangents are, as for a {!val-vjp}
    given its cotangents as arguments: nothing is saved. A derivative that
    combines both modes, such as a Hessian-vector product, saves less under jit
    today: the compiled program keeps the forward-mode values of every layer.
    Inside the body of a compiled {!scan}, remat changes nothing: the backward
    loop recomputes each step already.

    Raises [Invalid_argument] when applied to [s] if [s] consumes an argument.
*)

(** {1:jacobians Jacobians} *)

val jacfwd' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('c, 'd) Nx.t
(** [jacfwd' f x] is the Jacobian of [f] at [x], with shape
    [shape (f x) @ shape x], computed column by column in forward mode (one
    vectorized pass). Its dtype is the dtype of [f x]. Prefer it when the input
    is smaller than the output. *)

val jacrev' : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [jacrev' f x] is the Jacobian of [f] at [x], with shape
    [shape (f x) @ shape x], computed row by row in reverse mode (one forward
    pass, one vectorized backward pass). Its dtype is the dtype of [x]. Prefer
    it when the output is smaller than the input. *)

val hessian' :
  (('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [hessian' f x] is the Hessian of the scalar objective [f] at [x], with shape
    [shape x @ shape x] (forward over reverse). *)

val hvp : 'p Nx.Ptree.t -> ('p -> ('c, 'd) Nx.t) -> 'p -> 'p -> 'p
(** [hvp p f params v] is the Hessian-vector product of the scalar objective [f]
    at [params] against [v], a value of structure [p], computed without
    materializing the Hessian (forward over reverse).

    Raises [Invalid_argument] as {!jvp} does for its tangents, naming
    [Rune.hvp]. *)

val hvp' :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [hvp' f x v] is [hvp Nx.Ptree.tensor f x v]. *)

(** {1:checks Gradient checking} *)

val check_grads :
  ?eps:float ->
  ?tol:float ->
  'p Nx.Ptree.t ->
  ('p -> ('c, 'd) Nx.t) ->
  'p ->
  (unit, string) result
(** [check_grads p f params] compares the reverse-mode gradient of the scalar
    objective [f] at [params] against central-difference directional derivatives
    along deterministic directions. [Ok ()] means they agree within [tol]
    (relative, default [1e-2]); [Error msg] describes the disagreement. [eps] is
    the finite-difference step (default [1e-4]).

    The check is directional, not per-element: it validates gradients cheaply
    rather than exhaustively. Use float64 parameters for reliable results;
    float32 may need a looser [tol]. *)

(** {1:rng Random number generation}

    Random number generation lives entirely in {!Nx.Rng}: keys, the keyed
    samplers ({!Nx.Rng.uniform}, {!Nx.Rng.normal}, …) and the scope
    ({!Nx.Rng.with_key}). A key is an [[|2|]] int32 tensor that only {!Nx.Rng}
    builds, walked with {!Nx.Rng.ptree}, so it traces, batches and shards like
    any tensor — thread it as an input of a jitted function and derive per-call
    keys with {!Nx.Rng.split} or {!Nx.Rng.fold_in}. The transforms answer the
    generator's effects but add no RNG vocabulary of their own. A sampler's
    distribution parameters are tensors too ({!Nx.Rng.bernoulli}'s probability,
    {!Nx.Rng.poisson}'s rate), so a parameter that is a jitted function's input
    or a mapped axis traces or batches the draw with it, where a host float
    would have been frozen into the program.

    Under a transform, what matters is where the key comes from, not which
    front-end draws from it. A traced or mapped key works either way: passed to
    a keyed sampler, or as the root of a {!Nx.Rng.with_key} scope that the
    keyless [Nx.rand] draws from. A key the transform closes over is a constant
    of that transform, whichever front-end reads it. *)

(** {1:devices Devices}

    A device is a {!Nx.Device.t}: rune opens it by name, and it carries the
    engine that holds values on it ({!Nx.place}). A device has one name:
    ["METAL"], ["CUDA:3"], never an index [0] (["CUDA:0"] is ["CUDA"]). ["CPU"]
    is the host, {!Nx.Device.host}; ["CPU:1"], ["CPU:2"], ... are devices with
    storage of their own, for testing placement without a GPU.

    {!Nx.place} on a device copies the value's bytes 64 MiB at a time into one
    device buffer, and the result is resident like an output of a compiled call
    (see {!val-jit}): metadata reads are free, a read copies the elements it
    reads and leaves the buffer, a compiled function that takes it as an input
    leaf uses the buffer with no transfer, and a call that consumes the argument
    it is a leaf of ends it ({!val-jit}). Use it to put a model's weights on the
    device once, as they are imported, instead of once per compiled function at
    its first call. A buffer uploaded from a mapped file is returned to the
    system when the value is released, not kept for reuse. *)

val device : string -> Nx.Device.t
(** [device name] is the device [name] names, opened at the first call. Every
    call with the same name returns the same value, and the backend part of the
    name is case-insensitive.

    Raises [Invalid_argument] if the backend is unknown or the device cannot be
    opened. *)

val devices : string -> Nx.Device.t list
(** [devices backend] is every device of [backend] in index order: ["CUDA"],
    ["CUDA:1"], ... for as long as they open. [devices "CPU"] is
    [[Nx.Device.host]], and [devices "METAL"] is the one Metal device.

    Raises [Invalid_argument] if [backend] names one device (["CUDA:1"]), is
    unknown, or its first device cannot be opened. *)

val default_device : unit -> Nx.Device.t
(** [default_device ()] is the device that compiled functions run on unless they
    are told otherwise: the backend that the [DEV] environment variable names,
    or else the first of METAL, AMD, NV and CUDA that opens, or else the host.
    It is resolved once per process, at the first call. *)

(** {1:jit Just-in-time compilation} *)

exception Jit_error of string
(** Raised when a function cannot be compiled: it read the value of a traced
    tensor (for example [Nx.item] on a value that depends on the inputs, or a
    data-dependent branch), it drew random values from a key that does not
    depend on the inputs (a captured {!Nx.Rng.t}, or a scope opened with
    [Nx.Rng.with_key] on a constant key — the draw would be a compile-time
    constant replayed on every call; pass the key as an input instead), or it
    used an operation the compiler does not support (FFT, the SVD and
    eigensolvers, complex, int4 and uint4 tensors, a bitcast to or from float8).
    QR, triangular solves, Cholesky, [solve], and [inv] do compile: they unroll
    at trace time into the fixed number of steps their shapes imply. *)

val jit :
  ?devices:Nx.Device.t list ->
  ?beam:int ->
  ?beam_parallel:int ->
  ('a -> 'b) Nx.Ptree.fn ->
  ('a -> 'b) ->
  'a ->
  'b
(** [jit s f] is [f] compiled, a function of [f]'s type whose arguments and
    result have the structures of the signature [s]:

    {[
    let step =
      Rune.jit
        Nx.Ptree.(
          tensor @-> Cache_index.ptree @-> consumes caches
          @@ returns (pair tensor caches))
        (fun tokens index caches -> decode params tokens index caches)
    ]}

    An argument built with {!Nx.Ptree.( @-> )} is read; one built with
    {!Nx.Ptree.consumes} is given up by each call, which may write the result
    over its storage. Tensors [f] closes over ([params] above) are constants of
    the compiled function.

    The first application traces [f], compiles the traced computation into fused
    kernels, and runs them. Later applications replay a compiled program on the
    new tensors when their key equals the program's.

    {b Paths.} A leaf of the arguments is named by its path: the argument's
    position counted from 0, then the leaf's path inside that argument. The
    window of the second argument is [1.window], and a first argument that is
    one tensor is [0]. Keys, errors and [RUNE_JIT_DEBUG] reports use these
    paths.

    {b Keys.} A key is the devices, every tensor's path, dtype, shape, placement
    and layout, a host tensor counting as a copy on each device, and every
    report of the arguments' walks (an integer, a case, an option's presence, a
    list's length), compared by path segments. An argument with another window,
    or a list that gained an element with no tensor, traces and compiles its own
    program. An integer that changes on every call compiles a program per value;
    a value that varies belongs in a tensor. [RUNE_JIT_DEBUG=1] reports each
    retrace with the first difference from the previous call's key, such as
    ["rune.jit: retrace: 1.window: int 3 here, int 2 in the previous key"].

    {b Numerics.} A sum over an axis ({!Nx.sum}, {!Nx.mean}, the contraction
    of {!Nx.matmul}) is the sum of its terms in an unspecified association: the
    compiled program may add them in another order than eager and move factors
    that do not vary along the summed axis out of the sum. Results then differ
    from eager's in rounding, and at overflow in whether a term overflows. A
    maximum over an axis is exact, except which zero it returns when -0 and +0
    tie. Beyond that, compiled float results can differ from eager's in the
    last bits where the kernel compiler fuses a multiply and an add, where a
    division by a constant becomes a multiplication by its rounded reciprocal,
    and in transcendental functions, which are approximations within a few
    units in the last place ({!Nx.pow} about 70); Metal flushes float32
    subnormals to zero, a [float16] program on the CPU is not rounded after
    each operation, and signed integer overflow is undefined in the generated
    C.

    {b Results.} Every result leaf is a value with storage of its own. A result
    that returns a read argument or a capture unchanged is a copy, and a value
    [f] returns at two leaves comes back as two values, the second a copy of the
    first. So a result can be read, or consumed by a later call, whatever
    happens to the arguments and to the other results.

    {b Consumption.} Before its first kernel, a call marks every storage that a
    leaf of a consumed argument reaches as consumed; nothing unmarks it. From
    then on a read of any value over that storage, or its use as an operand or
    an argument, raises [Invalid_argument]
    ["this value was consumed at 2.0.keys in a compiled call's arguments; use
     the value the call returned"]; its shape and dtype stay readable. A
    consumed leaf must hold its storage alone: the call raises
    [Invalid_argument], before anything runs and without consuming anything, if
    a consumed leaf views part of its storage (a slice, a transpose, a
    broadcast: pass [Nx.copy] of it), or if another leaf of the call or a
    capture of the function, bound or copied, reaches that storage, naming both
    paths. A host leaf has no storage to consume: it is uploaded, stays usable,
    and lends nothing. A call that raises before its first kernel consumes
    nothing; one that fails after it has consumed its consumed arguments and
    returns nothing.

    {b Lending.} A result may take the storage of a consumed leaf, so a loop
    that consumes its state holds one generation of it on the device. It does
    when their dtypes, sizes and devices are equal, the storage is bound by no
    compiled function, and writing the result there cannot change it: no kernel
    reads the leaf after the first kernel that writes the result, and that
    kernel reads it only when the result derives from it at its own index
    (elementwise operations, equal-width casts and reshapes: an optimizer
    update, a window write into a cache). Partners are chosen once per program:
    first the results of an indexed write into a consumed leaf, then the results
    that derive from one at their own index, a consumed leaf returned unchanged
    included, then the rest, in the order the program writes them. Each storage
    lends at most once; a result without a partner gets fresh storage, and the
    consumed storage goes back to the device's allocator. [RUNE_JIT_DEBUG=1]
    reports each consumed leaf:
    ["rune.jit: 2.0.keys -> result 1.0.keys reused"], or what became of its
    storage. On the host, results are host tensors and nothing is lent.

    {b Devices.} A call runs on the devices where its placed input leaves and
    captures live ({!Nx.placement}), and on {!default_device} when none is
    placed. Captures are found by tracing: when the inputs are on the host, the
    first trace that meets a placed capture runs again on its devices, and later
    calls run there. [devices] names the devices instead ({!val-device},
    {!val-devices}): at least one, distinct, of one backend. Host values join
    the devices a call runs on, a full copy on each. A placed input leaf on
    other devices raises [Invalid_argument] naming its path and both placements,
    before anything runs, and so does a placed capture, at the trace that meets
    it: move it with {!Nx.place} first. So does a dtype the device cannot hold,
    such as [float64] on Metal, in an input leaf; in a value the function
    computes or captures it raises {!Jit_error}. On the host, contiguous inputs
    and captured tensors are read in place and outputs are computed directly
    into the returned tensors' storage; non-contiguous tensors are copied.

    On other devices, results are bit-identical. Host inputs are copied to the
    device on every call; a placed input, an output of an earlier call included,
    seeds the program with no transfer. Outputs are values on the device, and a
    read copies the elements it reads. A view of part of a storage is read in
    place; only views whose windows overlap ({!Nx.sliding_window}) are copied.
    Device memory backing an output is held until the output is
    garbage-collected or consumed; an allocation that fails, after a major
    collection, raises {!Nx.Device.Out_of_memory} before the call consumes
    anything, and a transfer failure raises at the first read of the affected
    output. doc/05-compilation.md describes the memory budget and the scratch
    memory compiled functions share.

    {b Several devices.} Over several devices the function sees global shapes,
    and each leaf stays where it lives: a split leaf is one slice on each device
    ({!Nx.Placement.sharded}), a copy or a host leaf the whole value on each.
    Every value the function computes lives where nx's rules put it, decided as
    it traces ({!Nx.placement} answers), and results come back there: an
    elementwise operation keeps its operands' split, and a reduction over a
    split axis is an allreduce whose result is a copy on each device, so the
    gradient of a loss over a batch split across devices is summed across them.
    Operands split differently (a row-split matrix times a column-split one
    included), an operation along a split axis, and a movement that would move
    elements between devices raise [Invalid_argument] as the function traces,
    with nx's message: nothing moves between devices unless the function places
    it with {!Nx.place}, which gathers a split value to a copy on each device or
    splits a copy, over the program's devices. A cut of one whole slice of a
    split axis is copied to every device, and one strictly inside a slice
    raises. A consumed leaf's placement goes to the result paired with it, the
    first in walk order that derives from it at its own index, each leaf to one
    result (as for storage, under Lending): where nx's rules put that result
    elsewhere, the program reshards it at its end, which [RUNE_JIT_DEBUG=1]
    reports, so a carry keeps its placement from call to call and one that
    starts on the host stays a copy on each device. Only a split value orders
    the devices, which decides the slice each holds: the first split leaf, else
    a split capture, while copies list them as a set, and [devices] fixes the
    order; a split leaf or capture in another order raises. A call returns once
    its work is queued on every device, and a read waits for it. Storage reuse,
    staged scans and in-place indexed writes apply on one device only, for now:
    a consumed carry keeps two generations.

    {b Captures.} The compilation cache lives in the partial application
    [jit s f]: apply [jit] once and reuse the returned function. Tensors [f]
    closes over are compile-time constants, bound once when the trace first
    compiles: on the host contiguous captures are read in place, and every other
    capture is copied to the device once per closure, and signatures share the
    copy. A capture placed on the program's devices, a split one or a view of it
    included, is bound instead: the program uses its buffers as the constant
    from its first compilation on, no bytes move, and every compiled function
    that captures the value shares them. A compiled function keeps the values it
    binds reachable, and their buffers stay while it is reachable: a call that
    consumes a bound storage ends it for its values, and the programs that bind
    it keep replaying with it. A closure whose capture was consumed raises
    [Invalid_argument] at its next trace. Mutating a captured tensor between
    calls is not supported and has unspecified visibility (the host may observe
    the mutation through its in-place binding; other devices never do): pass
    values that change between calls as arguments rather than capturing them.

    {b Tuning.} [beam] searches kernel schedules with a beam of that width,
    compiling and timing candidates on the device; compilation is much slower
    and the kernels usually faster. When [beam] is omitted, the [BEAM] context
    gives the width, initially set by the environment variable. Explicit [0]
    disables search for kernels without their own positive beam width.
    [beam_parallel] compiles a round's candidates on that many domains
    without changing the result, and is not part of any key; it defaults to
    [BEAM_PARALLEL] (sequential).

    {b Persistence.} Compiled programs are also written to a disk cache and
    loaded by later processes that compile the same trace; [JITCACHE=0] disables
    it, and results are identical either way. Programs over several devices are
    never persisted. doc/05-compilation.md gives the cache's location and when
    entries are invalidated.

    {b Transformations.} Under an enclosing transformation ({!grad},
    {!val-vmap}, {!with_debug}, an outer [jit]), the wrapped function runs
    directly so the transformation observes its operations, and it checks and
    consumes nothing: [jit] never changes results, only speed. Compose the other
    way, differentiating {e inside} the compiled function, to compile the
    forward and backward passes together:

    {[
    let state = Nx.Ptree.pair linear (Vega.adam_ptree linear)

    let step =
      Rune.jit
        Nx.Ptree.(
          tensor @-> tensor @-> consumes state @@ returns (pair tensor state))
        (fun inputs targets (params, opt) ->
          let loss, grads =
            Rune.value_and_grad linear (objective inputs targets) params
          in
          let params, opt = Vega.adamw_step linear ~lr opt ~params ~grads in
          (loss, (params, opt)))
    ]}

    Tensors are values, so state threads through the arguments: the function
    returns its updated parameters, optimizer state or cache, and the caller
    feeds them to the next call. Structured values read during tracing must not
    depend on traced tensors: a data-dependent {!cond} or {!while_loop}
    predicate raises {!Jit_error}. Overlapping or reentrant calls to one
    compiled function raise [Invalid_argument] before accessing its compiled
    state. Sequential calls may run on different domains. Calls under an
    enclosing transformation execute [f] directly and do not claim that state.

    Randomness inside a compiled function comes from a {!Nx.Rng} key passed as
    an argument: samplers are pure functions of their key, so the compiled
    program recomputes each draw from the current key on every call; feed a
    fresh key ({!Nx.Rng.split}, {!Nx.Rng.fold_in}) for fresh values. Either
    front-end works. Pass the key to each sampler ({!Nx.Rng.uniform} and
    friends), or wrap the body in {!Nx.Rng.with_key} on that key and keep
    writing the keyless [Nx.rand]: the scope derives every draw from its root,
    so a traced root makes the whole scope traced. What raises {!Jit_error} is a
    root that does not depend on the arguments (a captured key, or
    [Nx.Rng.with_key] on a constant key), since the draw would be a compile-time
    constant replayed on every call.

    Raises {!Jit_error} when tracing fails ({!exception-Jit_error}), and
    [Invalid_argument] if [s] has no argument, if [devices] is empty, repeats a
    device or mixes backends, for a leaf or capture placed on other devices, and
    as consumption above says. *)

val jit' :
  ?devices:Nx.Device.t list ->
  ?beam:int ->
  ?beam_parallel:int ->
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  ('a, 'b) Nx.t ->
  ('c, 'd) Nx.t
(** [jit' f] is [jit Nx.Ptree.(tensor @-> returns tensor) f]: {!val-jit} for a
    function of one tensor that reads it. *)

type jit_stats = {
  bytes_to_device : int;  (** Cumulative bytes copied host to device. *)
  bytes_from_device : int;  (** Cumulative bytes copied device to host. *)
  resident_bytes : int;
      (** Device bytes held by outputs and placed values that are still
          reachable. *)
  reused_bytes : int;
      (** Cumulative bytes of consumed inputs whose storage an output took
          instead of a fresh buffer. *)
}
(** Transfer accounting for compiled functions. The zero-copy CPU path moves no
    bytes and counts nothing. *)

val jit_stats : unit -> jit_stats
(** [jit_stats ()] is the current transfer counters, cumulative over the whole
    program. An output dropped releases its device buffers once it is collected,
    at the next read, placement or compiled call, or at this query, whichever
    comes first. Set the [RUNE_JIT_DEBUG] environment variable to [1] to also
    log a per-call summary to stderr. *)

val reset_jit_stats : unit -> unit
(** [reset_jit_stats ()] zeroes the cumulative transfer counters.
    [resident_bytes] tracks live state and is not reset. *)

(** {1:flow Control flow}

    Eager combinators with staging-ready signatures: code written with them
    differentiates and vectorizes today, and a staging [jit] traces them as
    structured control flow instead of unrolled traces. Today, {!val-jit}
    compiles {!scan} as a loop, forward and reverse, and rejects data-dependent
    {!cond} and {!while_loop} predicates. *)

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
    step [i] passes [f] the value of row [i] of every tensor. [f carry row]
    returns the next carry, of structure [c], and the step's outputs, of
    structure [y]; the result is the final carry and the outputs, every tensor
    stacked along a new axis 0. A fold with nothing to emit passes
    {!Nx.Ptree.unit} for [y] and returns [()].

    Every carry the body returns has the visits ({!Nx.Ptree.visits}) of the one
    it received, and every step's outputs have the first step's: a list keeps
    its length, an option its presence, a case and an integer their value.

    Under {!val-jit} the fold step compiles once and runs as a loop in the
    compiled program, and differentiating compiles a reversed loop over the
    step's pullback. The loop reads row [i] of each tensor of [xs] in place, so
    data that differs per step, such as the weights of stacked layers, belongs
    in [xs]: reading it from a tensor [f] captures, for instance with
    {!Nx.index.D} at a step counter, is a gather. The cotangent of [xs] is
    stacked like the outputs, row [i] coming from step [i], while a captured
    tensor's cotangent is the sum over the steps, accumulated on every one. A
    carry tensor the step updates with {!Nx.set}, or reads only at the index it
    writes, is updated in place: a step that writes one row of a cache in the
    carry moves that row, not the cache. Staging needs the carry to keep its
    shapes across steps; a fold that changes them, one reached through
    {!val-vmap}, or one in a program over several devices unrolls into the
    compiled program instead. Everywhere else the scan folds eagerly, tracing
    every step.

    Raises [Invalid_argument] if [xs] has no tensor, a scalar tensor or tensors
    of different leading lengths, or if [n] is [0]; and, eagerly and under
    {!val-jit}, if the body returns a carry whose visits or dtypes differ from
    the carry it received, or outputs whose visits or dtypes differ from the
    first step's, naming the first path where they differ and what each holds
    there, as in
    ["Rune.scan: 1: length 3 in the carry the body returned, length 2 in the
     carry it received"]. *)

val scan' :
  f:(('a, 'b) Nx.t -> ('c, 'd) Nx.t -> ('a, 'b) Nx.t * ('e, 'f) Nx.t) ->
  init:('a, 'b) Nx.t ->
  ('c, 'd) Nx.t ->
  ('a, 'b) Nx.t * ('e, 'f) Nx.t
(** [scan' ~f ~init xs] is {!scan} for a carry, rows and outputs that are single
    tensors: it folds [f] over the slices of [xs] along axis 0 and returns the
    final carry and the outputs stacked along a new axis 0. *)

val cond :
  (bool, Nx.bool_elt) Nx.t -> then_:(unit -> 'r) -> else_:(unit -> 'r) -> 'r
(** [cond pred ~then_ ~else_] runs one branch according to the scalar [pred].
    Reading [pred] concretizes it: inside {!val-vmap}, a predicate that depends
    on the mapped inputs raises, since the lanes could diverge. *)

val while_loop :
  cond:('p -> (bool, Nx.bool_elt) Nx.t) -> body:('p -> 'p) -> 'p -> 'p
(** [while_loop ~cond ~body init] iterates [body] on the carry while [cond]
    holds. Reading the predicate concretizes it, with the same {!val-vmap}
    caveat as {!cond}. Differentiating traces every iteration actually taken. *)

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
