# RFC 0006: Structures and compiled signatures

- Status: discussion
- Date: 2026-09-24
- Revision: 2. Replaces the first revision in place, which described a
  structure by a value of a public GADT. Rationale says what it chose, what it
  would have guaranteed, and why it lost.
- Packages: nx (`Nx.Ptree`, `Nx.packed`, a cell for host storage), nx.quant
  (`Nx_quant`'s traversal), nx.io (`Nx_io.P` merges into `Nx.packed`), rune
  (every transformation's signature; `jit`'s key, consumption and lending),
  vega, kaun (layers, `Checkpoint`, `Cache_index`, the examples), ppx_ptree (one derived `walk`),
  munin, contrib (fehu, sowilo, norn)

## Summary

A structure of tensors is a type `'a t` with one function, `walk`, that walks
its parts with a cursor: the positions of its type parameter, the tensors of a
fixed type, and the data a compiled program depends on (a window, a list's
length, a variant's case). nx derives everything else from that one function:
rune's flatten, rebuild and cache key, vega's updates, checkpoint names, dtype
casts, and maps over parameter-shaped metadata such as a mask, a sharding
plan or symo's per-leaf symmetries. Paths are typed, so a mask or a plan is a
pattern match. A compiled function keys its programs on each leaf's path and
on the data the structure reports, so a changed window or block shape compiles
a new program. `jit`, `vmap` and `remat` take a signature with the shape of the
function they compile, `tensor @-> index @-> consumes caches @@ returns (pair
tensor caches)`; `consumes` marks the arguments a compiled call gives up, every
other argument is read, and every result is a fresh value, so a result can be
read at any later time. A consumed value raises on use on every device, the
host included once host cells pass their measurement. The single-tensor
shorthands (`grad'`, `jit'`, `vmap'`, ...) stay; the structured variants
(`jit2`, `jit_step`, `vjp2`, `jvp2`, `vmap2`, `pmap2`), `?donate`, `Uniform`,
`Make` and the dynamic tree go.

## Motivation

A structure today is a first-class module, `Nx.Ptree.S = { type t; map; map2;
iter }` with rank-2 leaf functions (`ptree.mli:24-42`), plus a payload-generic
`Traverse` and a path-carrying `Uniform` for checkpoints. It has three costs,
each visible in the tree.

**Every structure is written three to six times by hand.** The tree holds
about 140 hand-written structures: 119 rank-2 `map2` traversals, each beside a
`map` and an `iter`, and kaun's ten layers write six functions each.
`ptree.mli` states that every traversal visits the leaves in one order, and
nothing enforces it. rune cannot rely on the order of `map`'s callbacks, so it
recovers positions with marker tensors (`jit.ml:2893-2910`, `scan.ml:31-58`),
pairs `vmap`'s axes by physical identity (`rune.ml:261-280`), rebuilds a
compiled call's outputs by an identity search that is quadratic in the leaf
count (`jit.ml:3675`), and uses `Obj.magic` at seven sites in `scan.ml`,
`reverse.ml` and `jit.ml`, six of which explicit flatten and rebuild remove.

**Every role is a module.** A transformation takes one module per structure it
touches, so 432 call sites pass `(module ...)` (Rune 280, Vega 105, Checkpoint
47). A structure that exists only to satisfy a transformation is a module of
its own: gpt-oss's layer loop writes `Layer` and `Stream`, 30 lines of rank-2
traversals, to call `Rune.jit_step` (`layer_loop.ml:14-44`); GPT-2's trainer
writes `Step_out`, `Scaled`, `Key` and `Step_in`, and a `no_loss` dummy to
seed its state (`04-gpt2/train.ml:160-296`). Structured outputs need a second
variant of each transformation (`jit2`, `vjp2`, `jvp2`, `vmap2`, `pmap2`).

**`jit_step` cannot serve an engine.** RFC 0005 gave compiled steps a read
argument and a consumed one, `jit_step (module R) (module S) : 'r -> 's -> 's`
(landed at fbc48b93). Every output that is not state (sampled ids, a loss)
must then be a field of `'s`, which the next call consumes. An engine that
queues step `n + 1` and then reads step `n`'s ids on the host, so that host
work overlaps the device, reads a consumed value and fails.

Two correctness gaps sit in the same code. jit's cache key holds each leaf's
dtype and shape and nothing else (`jit.ml:3786`): a `Cache_index` whose window
changed, or two FP8 weights that differ only in block shape (RFC 0004), replay
the old program. And consumption does nothing on the CPU device, where host
values carry no state (`jit.ml:3435-3441`), so a test suite on the CPU never
sees a stale holder that a device would reject. Each gap, and the quadratic
rebuild, has a fix of under 50 lines in today's code; this RFC fixes them
because its compiled call replaces the code they live in.

Structures are also used outside raven, with payloads that are not tensors.
symo (hennequin-lab/symo-raven) describes a model's per-leaf dimensions and
symmetries as values of the model's own type, `int list P.t` and
`Symmetry.spec list P.t`, and its optimiser maps and zips them into `int P.t`,
closures and lists of tensors (`lib/orbit.ml:118-330`). Whatever replaces
`Uniform` must keep that.

## Guide

### A structure is one function

```ocaml
(* linear.ml *)
type 'a t = { w : 'a; b : 'a option }

let walk c { w; b } =
  let open Nx.Ptree.Walk in
  { w = field c "w" leaf w; b = field c "b" (option leaf) b }
```

`walk` visits each part once. `field c "w" walk x` descends into the field named
`"w"` and walks it; `leaf` is a position of the type's parameter, and the
compiler checks that none is forgotten, because `leaf` changes its type.
`option` and `list` walk containers and report their shape. A tensor of a fixed
type is `tensor`, an integer the computation depends on is `int`, and a
sub-structure is walked by its own `walk`:

```ocaml
(* mlp.ml *)
type 'a t = { l1 : 'a Linear.t; l2 : 'a Linear.t; steps : Nx.int32_t; window : int option }

let walk c m =
  let open Nx.Ptree.Walk in
  { l1 = field c "l1" Linear.walk m.l1;
    l2 = field c "l2" Linear.walk m.l2;
    steps = field c "steps" tensor m.steps;
    window = field c "window" (option int) m.window }
```

A variant is a `match` that names its case first:

```ocaml
(* gpt-oss's weights *)
type 'a weight = Float of 'a | Mxfp4 of { blocks : Nx.uint8_t; scales : Nx.uint8_t }

let walk_weight c =
  let open Nx.Ptree.Walk in
  function
  | Float w -> case c "float"; Float (leaf c w)
  | Mxfp4 { blocks; scales } ->
      case c "mxfp4";
      Mxfp4 { blocks = field c "blocks" tensor blocks; scales = field c "scales" tensor scales }
```


### Using one

A transformation takes a structure at one type, which `instantiate` gives:

```ocaml
let mlp = Nx.Ptree.instantiate (module Mlp)          (* : Nx.float32_t Mlp.t Nx.Ptree.t, by use *)

let g = Rune.grad mlp loss params
Nx.Ptree.map mlp (fun _path t -> Nx.place p t) params          (* every tensor, its own dtype *)

let frozen path = match Nx.Ptree.Path.segments path with
  | Nx.Ptree.Path.Field "l1" :: _ -> true
  | _ -> false
let masked = Nx.Ptree.map mlp (fun path g -> if frozen path then Nx.zeros_like g else g) grads
```

The one function also changes the payload's type, which is how a model is
cast and how metadata shaped like the model is used:

```ocaml
let bf16 = Nx.Ptree.cast (module Mlp) Nx.bfloat16 master     (* steps stays int32 *)
let sizes = Nx.Ptree.Payload.map (module Linear) (fun _ dims -> List.fold_left ( * ) 1 dims) dims
(* dims : int list Linear.t, sizes : int Linear.t *)
```

A single tensor keeps its shorthands: `Rune.grad' f x`, `Rune.jit' f`,
`Rune.vmap' f x`.

### Compiling

```ocaml
let caches = Nx.Ptree.list (Nx.Ptree.instantiate (module Attention.Cache))

let step =
  Rune.jit Nx.Ptree.(tensor @-> Cache_index.ptree @-> consumes caches @@ returns (pair tensor caches))
    (fun tokens index caches ->
      let h, caches = Llama.cached cfg params caches index tokens in
      (Nx.argmax ~axis:1 (Llama.logits cfg params (Nx.slice [ A; I (-1) ] h)), caches))
```

The signature has the function's own shape, so `step` has the type of the
function it compiles. `params` is captured: bound once, read by every program,
never consumed. The caches are given up by each call and come back written in
place. The sampled ids are a fresh result that nothing will consume:

```ocaml
let rec generate n tokens index caches pending =
  if n = steps then Option.iter emit pending
  else
    let ids, caches = step tokens index caches in       (* queued, returns at once *)
    Option.iter emit pending;                           (* step n - 1's ids, read now *)
    generate (n + 1) ids (Cache_index.advance index) caches (Some ids)
```

The old caches are dead after the call that consumed them, on every device:

```ocaml
Nx.to_array (List.hd caches).keys
(* Invalid_argument "this value was consumed at 2.0.keys in a compiled call's
   arguments; use the value the call returned" *)
```

A model's own curried function compiles as it is; the gpt-oss layer loop
compiles one program per layer kind:

```ocaml
let block = Nx.Ptree.instantiate (module Gpt_oss.Block)
and cache = Nx.Ptree.instantiate (module Attention.Cache)

let compile kind =
  Rune.jit
    Nx.Ptree.(block @-> consumes cache @@ Cache_index.ptree @-> consumes tensor
              @@ returns (pair tensor cache))
    (Gpt_oss.block cfg kind)
```

A training step consumes its state and returns the loss beside it:

```ocaml
let state = Nx.Ptree.pair mlp (Vega.adam_ptree mlp)

let step =
  Rune.jit Nx.Ptree.(tensor @-> tensor @-> consumes state @@ returns (pair tensor state))
    (fun ids targets (master, opt) ->
      let loss, grads = Rune.value_and_grad mlp (objective ids targets) master in
      let master, opt = Vega.adamw_step mlp ~lr opt ~params:master ~grads in
      (loss, (master, opt)))
```

A result that is a record needs no `walk` of its own: `iso` adapts a structure
to it. A record whose names matter, such as a training state saved as one
checkpoint, is a module whose `walk` visits each part with `Walk.structure`: its
paths are `opt.mu.l1.w` where `iso` over pairs gives `1.0.mu.l1.w`.

```ocaml
type out = { loss : Nx.float32_t; params : Nx.float32_t Mlp.t }
let out = Nx.Ptree.(iso (fun (loss, params) -> { loss; params }) (fun o -> (o.loss, o.params))
                      (pair tensor mlp))
```

`vmap` maps axis 0 of every leaf of its arguments; a value that is not mapped
is captured:

```ocaml
Rune.vmap Nx.Ptree.(tensor @-> tensor @-> returns mlp)
  (fun x y -> Rune.grad mlp (fun p -> Loss.mse (Mlp.apply p x) y) params)
  xs ys                                          (* per-example gradients *)
```

### Saving

```ocaml
Checkpoint.of_value ~prefix:"model" mlp params
Checkpoint.to_value ~prefix:"model" mlp ~like:params ckpt
Checkpoint.of_value ~prefix:"optim" (Vega.adam_ptree mlp) opt
```

## Reference

### `Nx.Ptree`

```ocaml
module Ptree : sig
  module Path : sig
    type seg = Field of string | Index of int
    type t
    val segments : t -> seg list
    val equal : t -> t -> bool
    val to_string : t -> string           (* "blocks.3.fc.b"; the root is "" *)
    val pp : Format.formatter -> t -> unit
  end

  (* a structure at one type: what transformations, vega and Checkpoint take *)
  type 's t

  module Walk : sig
    type ('a, 'b) cursor
    val field : ('a, 'b) cursor -> string -> (('a, 'b) cursor -> 's -> 't) -> 's -> 't
    val index : ('a, 'b) cursor -> int -> (('a, 'b) cursor -> 's -> 't) -> 's -> 't
    val leaf : ('a, 'b) cursor -> 'a -> 'b
    val tensor : ('a, 'b) cursor -> ('x, 'y) Nx.t -> ('x, 'y) Nx.t
    val int : ('a, 'b) cursor -> int -> int
    val case : ('a, 'b) cursor -> string -> unit
    val option : (('a, 'b) cursor -> 's -> 't) -> ('a, 'b) cursor -> 's option -> 't option
    val list : (('a, 'b) cursor -> 's -> 't) -> ('a, 'b) cursor -> 's list -> 't list
    val structure : 's t -> ('a, 'b) cursor -> 's -> 's
  end

  module type S = sig
    type 'a t
    val walk : ('a, 'b) Walk.cursor -> 'a t -> 'b t
  end

  val instantiate : (module U : S) -> ('a, 'b) Nx.t U.t t
  val nest : (module U : S) -> 's t -> 's U.t t
  val tensor : ('a, 'b) Nx.t t
  val unit : unit t
  val pair : 'a t -> 'b t -> ('a * 'b) t
  val option : 'a t -> 'a option t
  val list : 'a t -> 'a list t
  val iso : ('a -> 'b) -> ('b -> 'a) -> 'a t -> 'b t

  val map : 's t -> ('a 'b. Path.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> 's -> 's
  val map2 : 's t -> ('a 'b. Path.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> 's -> 's -> 's
  val fold : 's t -> ('a 'b. Path.t -> ('a, 'b) Nx.t -> 'acc -> 'acc) -> 's -> 'acc -> 'acc

  (* what a walk visits: what visit tests and RUNE_JIT_DEBUG read *)
  type report = Int of int | Case of string | Present of bool | Length of int
  type visit = Leaf of Path.t | Report of Path.t * report
  val visits : 's t -> 's -> visit list
  val pp_visit : Format.formatter -> visit -> unit

  (* flattening, for compiled calls and optimisers *)
  module Skeleton : sig
    type t
    val equal : t -> t -> bool
    val hash : t -> int
    val diff : this:string -> t -> that:string -> t -> string option
  end
  val flatten : 's t -> 's -> Nx.packed list * Skeleton.t
  val rebuild : 's t -> like:'s -> Nx.packed list -> 's

  (* maps that change the payload's type *)
  val cast : (module U : S) -> ('c, 'd) Nx.dtype -> ('a, 'b) Nx.t U.t -> ('c, 'd) Nx.t U.t
  module Payload : sig
    val map : (module U : S) -> (Path.t -> 'a -> 'b) -> 'a U.t -> 'b U.t
    val map2 : (module U : S) -> (Path.t -> 'a -> 'b -> 'c) -> 'a U.t -> 'b U.t -> 'c U.t
    val fold : (module U : S) -> (Path.t -> 'a -> 'acc -> 'acc) -> 'a U.t -> 'acc -> 'acc
  end

  (* signatures of compiled functions *)
  type 'f fn
  val ( @-> ) : 'a t -> 'b fn -> ('a -> 'b) fn
  val consumes : 'a t -> 'b fn -> ('a -> 'b) fn
  val returns : 'a t -> 'a fn
end
```

- **One module type.** A structure is `type 'a t` with `walk`. A monomorphic
  structure is `type _ t = affine`, whose `walk` visits every tensor with
  `tensor`. Today's `S` (a `type t` with three rank-2 functions), `Uniform`,
  `Make`, `leaf` and the dynamic `Tree` go; the new `S` is today's `Traverse`
  reduced to one `walk` that takes a cursor.
- **The contract of `walk`.** It walks every part once: each position of the
  parameter with `leaf`, each tensor of a fixed type with `tensor`, each
  sub-structure with its own `walk`, each part that is a value of a structure
  at one type with `structure s` (its tensors are fixed, as `tensor`'s are;
  `tensor c x` is `structure Nx.Ptree.tensor c x`), each integer that changes what a program
  computes (a window, a block shape, a layer kind) with `int`, and each variant's case with `case`, before any other part of the case and
  with a tag no other case of the type uses. `option` and `list` (and `Nx.Ptree.option` and `Nx.Ptree.list`) report an option's presence and a list's length as they walk, so a part with no leaves is still in the key. A dictionary reports its size
  with `int` and each key with `case` before walking that key's value with
  `field`, so its keys are in the key too. A bool is an `int` of 0 or 1; other
  scalar data a program depends on belongs in a tensor. A record literal's
  fields are evaluated in an order OCaml leaves unspecified; every operation
  runs the same `walk`, so they all see one order, and `fold` lists paths in
  it. Write the fields as a sequence of `let`s where declaration order must
  show.
- **Paths.** A field contributes `Field name`; a list or array element, a tuple's
  components and a pair's sides contribute `Index i` (`index`); `Some` and a case add nothing; the
  root is empty. `to_string` joins them with `.`, which is today's checkpoint
  naming. Paths compare by segments, so `"a.b"` as one field and `a` then `b`
  are different paths that print alike.
- **What the compiler checks, and what it does not.** A forgotten field of the
  parameter's type does not compile, since `leaf` changes its type. A
  forgotten fixed tensor, integer or case compiles, and the leaf is frozen into
  the first program, or the program replays for other data. Law 1 says how
  each is held.
- **`'s t`** is a structure at one type. `instantiate (module U)` is `nest
  (module U) tensor`; `nest` builds a structure whose payload is itself a
  structure, which is how vega's states and any record of models are built.
  Both are functions, so a binding serves one dtype, and needs an annotation
  only when nothing in its unit fixes the dtype; a monomorphic structure never
  needs one. The combinators are values. `iso f g s` adapts `s` to an isomorphic type, such as a record returned by
  a compiled function; it keeps `s`'s paths (a record over `pair` walks as
  `0` and `1`), and `f` and `g` must be inverse. A record whose names matter
  is a module.
- **Derived.** `map`, `map2` and `fold` pass every tensor its path (rank-2,
  each tensor at its own dtype). `cast` applies `Nx.cast` at the parameter's
  positions and keeps every fixed tensor. `Payload.map`, `Payload.map2` and
  `Payload.fold` change or read the payload at any type: this is what symo's
  per-leaf dimensions and symmetries, a mask or a sharding plan use. They
  keep the first value's fixed tensors, and take the structure's module, since
  only a module names the type constructor `U.t`.
- **`visits`** lists what `walk` visits, leaves and reports, in walk order.
  It is what each structure's visit test compares, and what
  `RUNE_JIT_DEBUG` prints when a key differs.
- **`Skeleton.diff ~this a ~that b`** is [None] when the skeletons are
  equal and otherwise the one message that names the first differing
  visit, its path and what each side holds there. `map2`, vega's steps,
  `scan` and `RUNE_JIT_DEBUG`'s retrace report all use it.
- **Errors name the path and both sides.** Every mismatch between two values
  of one structure (`map2`, `Payload.map2`, `Checkpoint.to_value` against its
  template, a `scan` carry against the one it received, a retrace against the
  previous key) raises or reports with the first path where they differ and
  what each side holds there: `blocks.3.moe.gate_up: case "mxfp4" here,
  "float" in the template`.
- **Signatures** are built with `@->`, `consumes ... @@` and `returns`. Each
  slot has its own type, so `tensor @-> tensor @-> ...` takes two tensors of
  unrelated dtypes. `consumes` builds an argument of a signature, so it cannot
  appear inside a structure. A signature with no argument is a type error.

`flatten` returns a value's tensors in walk order and its `Skeleton.t` (its
visits without the tensors, comparable and hashable); `rebuild` puts
tensors back into a template in the same order, checking each dtype. They
are what rune's `jit` keys and replays with and what vega's steps use, and
users get them for the same purposes (JAX's `tree_flatten` and `treedef`).
They live in `Nx.Ptree` with the rest; `nx.effect` holds no structure code. `Ptree.tensor`
becomes `Nx.packed`, and `Nx_io.P`, a second packed type, merges into it.

### Transformations

**The rule.** A transformation takes the structure of each value whose leaves
it enumerates, and nothing else; what it does not enumerate is closed over and
is a constant of the transformation. `jit`, `vmap` and `remat` (and `pmap`
until RFC 0005 removes it) enumerate every argument and the result of the
function they return at its own type, so they take that function's signature.
`grad`, `vjp`, `jvp` and their kin enumerate the value they differentiate and
the result they rebuild, so they take those structures; `scan` takes its
carry, rows and outputs. A role the transformation does not define raises
`Invalid_argument` when the transformation is applied to its signature (`jit`
and `pmap` define `consumes`).

Every transformation replaces each argument leaf by a fresh alias (a new value
over the same storage, no copy) before it tracks, marks or seeds it, so a
capture that is the same value as an argument is a constant:
`Rune.grad' (fun x -> Nx.mul x w) w` is `w`. `grad`'s `untie` becomes this
aliasing for every leaf.

```ocaml
val grad : 'p Nx.Ptree.t -> ('p -> ('c, 'd) Nx.t) -> 'p -> 'p
val vjp : 'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'q -> 'q * 'p
val jvp : 'p Nx.Ptree.t -> 'q Nx.Ptree.t -> ('p -> 'q) -> 'p -> 'p -> 'q * 'q
(* value_and_grad, value_and_grad_aux, vjp_fun, jvp_aux, custom_vjp, custom_jvp,
   hvp and check_grads take the structures of the form they extend *)

val jit : ?devices:Nx.Device.t list -> ?beam:int -> ?beam_parallel:int ->
  ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
val vmap : ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
val remat : ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
val pmap : devices:Nx.Device.t list -> ?in_axes:int option list ->
  ('a -> 'b) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
val scan : 'c Nx.Ptree.t -> 'x Nx.Ptree.t -> 'y Nx.Ptree.t ->
  f:('c -> 'x -> 'c * 'y) -> init:'c -> 'x -> 'c * 'y
val while_loop : cond:('p -> (bool, Nx.bool_elt) Nx.t) -> body:('p -> 'p) -> 'p -> 'p
```

- **Single-tensor shorthands stay.** `grad'`, `value_and_grad'`, `vjp'`,
  `vjp_fun'`, `jvp'`, `vmap'`, `jit'`, `scan'`, `hvp'`, `jacfwd'`, `jacrev'`
  and `hessian'` keep their meaning: a transformation of a function of one
  tensor. They follow their structured forms: `vmap'` maps axis 0 and loses
  `?in_axis` and `?out_axis` (three test callers), and `jit'` takes
  `?devices`. OCaml cannot give the structure argument a default that fixes its
  type, so the one-tensor case keeps its own name.
- **Structured variants go.** `jit2`, `jit_step`, `vjp2`, `jvp2`, `vmap2`,
  `pmap2`, `?donate` and rune's `module Ptree` go: a structure argument
  covers the result as well as the input.
- **`vmap`** maps axis 0 of every leaf of every argument; a value that is not
  mapped is captured, and another axis is `Nx.moveaxis`, a view. `?in_axes`
  and `?out_axis` go: they paired leaves by identity, and their non-zero axes
  had two callers, both tests.
- **`pmap`**, until RFC 0005's multi-device stage removes it, takes one
  `in_axes` entry per argument of its signature. A consumed argument's
  per-device buffers are released after the call, a leaf read through the
  host because its placement differs included; `pmap` lends nothing.
- **`scan`** with nothing to emit passes `Nx.Ptree.unit`. It raises, eagerly
  and staged, when the body returns a carry whose paths or reports differ from
  the ones it received, or a row whose differ from the first row's.
- **`while_loop`** loses the module it never used (`rune.ml:490-494`).

### A compiled call

1. **Key.** Two calls share a program exactly when their arguments visit the
   same leaves at the same paths with equal signatures (dtype, shape, and RFC
   0005's placement and view clauses), and make the same reports (list
   lengths, option presence, cases and integers) at the same paths. The key
   compares paths and reports in visit order, by equality of their segments;
   a hash only finds the candidate. A result's paths and reports are the ones
   its trace built. `RUNE_JIT_DEBUG=1` reports every retrace with the first
   path at which its key differs from the closure's previous one. Every
   message of a compiled call names a leaf by one path in its arguments,
   whose first segment is the argument's position from 0: the window of the
   second argument is `1.window`. An integer
   that changes on every call compiles a program per value; a value that
   varies belongs in a tensor.
2. **Fresh results.** Every result leaf is a new value with storage of its
   own. A result that returns a read argument or a capture unchanged is a
   copy. Two result leaves that the function returns as one tensor, or that
   the compiler resolves to one buffer, are two values: the first in walk
   order takes the storage and each other is a copy. Only consumed storage
   can back a result, and each consumed storage backs at most one.
3. **Consumption.** Before it queues its first kernel, the call marks every
   storage its consumed arguments' leaves reach as consumed; nothing unmarks
   it. A read of any value over that storage, or its use as an operand or an
   argument, raises `Invalid_argument` naming the argument and the leaf's
   path; shape and dtype stay readable. Consumption changes no value; it ends
   the caller's right to use one.
4. **Exclusive consumption.** A consumed leaf must cover its whole storage,
   and no other leaf of the call, nor a capture of its program, may reach that
   storage. Otherwise the call raises before it runs, naming both paths.
   `zeros`, `ones` and `full` outside a trace return a value whose storage
   covers its view on every placement; a consumed leaf that is a broadcast or
   a partial view raises, naming the leaf and `Nx.copy`. Each device-context
   scalar has a cell of its own.
5. **Lending.** A result may take the storage of a consumed leaf when all of
   these hold: equal dtype, byte size and placement; the storage is a buffer
   that no program binds and that seeds no other leaf of the call; no kernel
   reads the input after the first kernel that writes the result, and that
   kernel reads it only when the result derives from it at its own index
   (every path from input to result is elementwise, an equal-width cast, a
   reshape or a contiguous marker: RFC 0001's conditions). Partners are chosen
   once per program, in three passes: first the results of an indexed write
   that starts from a consumed input, which may take only that input; then the
   results that derive from a consumed input at their own index, a consumed
   input returned unchanged included; then the rest, in increasing order of
   their first write, which lends as much storage as any pairing can. Walk
   order breaks ties, each storage lends at most once, and a result without a
   partner gets fresh storage. Its analyses run in time linear in the
   program's schedule and its leaves; the third pass scans the free inputs
   for each of its results. A result that derives from a consumed
   input keeps its placement (RFC 0005). `RUNE_JIT_DEBUG=1` reports each
   pairing: `arg 2.0.keys -> result 1.0.keys reused`.
6. **Under an enclosing transformation** a compiled function runs its source,
   as it does today (`jit.ml:3783`), checks nothing and consumes nothing.
7. **Failure.** A call that raises before its first kernel is queued consumes
   nothing; every allocation of a call precedes its first kernel. A call that
   fails after that has consumed its consumed arguments and returns nothing.
   An error the device reports later poisons every result of that submission
   and of each later submission that reads one.

A bound storage that is consumed dies for its handles; the programs that bind
it keep its buffer and replay with it, and it is released when the last of
them is collected. A closure whose capture was consumed raises at its next
trace, naming the capture.

### A cell for host storage

RFC 0005 gives a placed value a cell, shared by every view of its storage,
that records whether the storage was donated. The host has none, so a host
value cannot be consumed. This RFC gives the host a cell of one field beside
an immutable id:

```ocaml
type ('a, 'b) t =
  | Host : ('a, 'b) Nx_backend.t * host_cell -> ('a, 'b) t
  | Placed : ('a, 'b) resident -> ('a, 'b) t               (* r_cell : cell *)
  | Traced : ('a, 'b) traced -> ('a, 'b) t
and host_cell = { h_id : int; mutable consumed : consumption option }  (* None: live *)
and state = Live of storage | Consumed of consumption      (* RFC 0005's cell *)
and consumption = { arg : int; path : string }
```

A cell mutates, so no table hashes a value's structure: rune's `Tensor_map`
and jit's capture and constant tables hash a value by an immutable id (`h_id`,
`r_id`, `t_id`) and compare it by `==`. A host result shares its operand's
cell exactly when the backend returned its operand's buffer; every other
result, `of_bigarray`, `of_buffer` and each checkpoint entry get a fresh cell.
The state is checked in `unwrap`, the one path from a host value to its bytes.
A consumed host storage lends nothing on the CPU, where programs write fresh
buffers as today. A fresh host result allocates four words more than without a
cell, and a view one word. `RUNE_JIT_FORCE_COPY=1` stays, as the way to
exercise lending on the CPU.

### vega, kaun, nx.quant, ppx_ptree

- **vega.** Every function that takes a module takes an `'s Nx.Ptree.t`. Each
  state record gets a module (`type 'p t = 'p adam_state` with its `walk`), and
  `adam_ptree p = Nx.Ptree.nest (module Adam_state) p`; likewise `sgd_ptree`,
  `lbfgs_ptree` and `Loss_scale.ptree`. The functors `Sgd_state`,
  `Adam_state`, `Lbfgs_state` and `Loss_scale`'s traversals go. Leaf paths are
  `mu.…`, `nu.…`, `step`.
- **kaun's layers** keep their type and one function, `walk`; their other five
  traversal functions and `Kaun.ptree` go. `Attention.Cache.List` is
  `Nx.Ptree.list` and goes.
- **`Cache_index`** exports `ptree : Cache_index.t Nx.Ptree.t`. Its `walk` over
  the private representation reports the tokens' case, `every`, each block's
  size, the window and the presence of `row` and `columns`, so the key sees
  them.
- **`Checkpoint`** takes structures: `of_value : ?prefix:string -> 's
  Nx.Ptree.t -> 's -> t` and `to_value : ?prefix:string -> 's Nx.Ptree.t ->
  like:'s -> t -> 's` replace `of_params`, `of_packed`, `to_params` and
  `to_packed`. It raises when two leaves of a value have one path, when an entry the
  template names is missing, and when an entry's dtype or shape differs from
  its template leaf's; entries the template does not name are ignored, as
  today. A payload leaf's
  path equals its name today, so files written today load, except for types
  with a fixed tensor, which today's `Uniform` does not name and which now has
  an entry. A checkpoint stores no reports: a value loads into its template's
  shape, and cases whose leaves have equal paths, dtypes and shapes load into
  one another.
- **`Nx_quant`** (RFC 0004) implements `S` with `type _ t = Nx_quant.t`; its
  codes and scales are fixed tensors, and its format's case, its scale's case
  and an Fp8 block's two sides are reported.
- **ppx_ptree derives one `walk`.** `[@@deriving ptree]` writes the function
  a hand-written structure has, from the type: each field under its name, the
  parameter with `leaf`, a tensor type with `tensor`, `'a M.t` with `M.walk`,
  `M.t` with `structure M.ptree`, options, lists, arrays and tuples, and a
  variant's case under its constructor's name. A type without a parameter also
  gets `ptree`, its one structure. An `int` or `bool` is marked `[@ptree.int]`
  (reported) or `[@ptree.skip]` (left out), and anything else it cannot walk
  is a located error, so for the types it derives the gap Law 1 leaves to
  visit tests closes. `[@ptree.walk f]` walks any other part with `f`. The six
  traversals, `Uniform` and `names` it generated go. raven's packages keep
  their hand-written walks and do not depend on it; it serves users such as
  symo and sofo.

### Cost on jit's hot path

Per call: collecting the curried arguments, one walk of each argument that
flattens the leaves and writes the key (each leaf's path and signature, and
the reports), the consumption checks, the replay, and one rebuild of the
result through the output template's `walk`. A one-function walk without paths costs 1.04 to 1.09 times today's module up
to 1,000 leaves. Paths roughly double the words it allocates (4,882 against
2,776 per flatten of gpt-oss-20b's 459 leaves, for a visitor), and the
cursor, a record per field, nearly doubles them again (9,285 against 5,238
words on a synthetic 384-leaf walk). That is microseconds against a 125 ms
decode step, and the `jit-run` acceptance rows hold it. The quadratic output rebuild goes.

### Amendments

- **RFC 0001.** Law 3 reads: reuse only through consumption; a result takes
  storage only from a consumed input that no other leaf of the call reaches,
  and a consumed value raises on use on every device, the host included.
- **RFC 0002.** An engine is parameterised by a step and the structure of its
  state. Law 6's "No two leaves hold one tensor" is checked by rule 4.
- **RFC 0003.** `Checkpoint.to_params` and `to_packed` are
  `Checkpoint.to_value`, which keeps `~like`.
- **RFC 0004.** `Nx_quant.t` implements `Nx.Ptree.S` in its new form; its cases
  and block shape are reported.
- **RFC 0005.** §Compiled functions' `jit_step` and its positional pairing
  are replaced by this RFC's compiled call; `?devices` lands on the one `jit`.
  Its cell state `Donated` becomes `Consumed` with a location, and host storage
  gets a cell. A cell reached from two leaves, or from a leaf and a capture,
  raises (rule 4) where it was read. Its key gains paths and reports. Until
  `pmap` is removed, its `in_axes` has one entry per argument of its signature.

### Order of work

One sweep on a branch, landing whole, with no bridge: every consumer of
`Nx.Ptree` moves with it, munin, contrib (fehu's two examples, and six
structured calls in sowilo and norn), ppx_ptree (rebuilt to derive one
`walk`), docs and `CHANGES.md` included. The build breaks when nx's `Ptree` changes and is
restored package by package, one commit each. About 510 call sites change
(432 module-taking, 30 `Kaun.ptree`, 8 placeholders, 34 `in_axes` and
`out_axis` arguments, and the structured variants' callers), about 140
hand-written structures shrink to one function each, and about 190 examples
in user-facing Markdown change. `CHANGES.md` maps each deleted value to its
replacement (`jit2 p q f` to `jit Nx.Ptree.(p @-> returns q) f`, `of_params
(module P)` to `of_value p`).

The sweep starts after RFC 0005's stage 1 lands whole and after RFC 0004's
`Nx_quant` with gpt-oss's importer, so that `jit.ml`, `nx_effect.ml` and
`Nx_quant`'s traversal move once. It carries RFC 0005's `?device` to
`?devices`. Until it starts, the landed `jit_step` stays. Host cells land on
their own, after RFC 0005's representation, once their measurement passes.

Budget: 65 engineer-days (revision 1's 68, less the deletion of the prime forms, which revision 2
keeps; the `Walk`, `nest` and `iso` work is within the re-derivation), re-derived before the sweep starts. It stops at 80
engineer-days; or if any cache leaf of gpt-oss's layer loop loses storage
reuse; or if gpt-oss's decode median is more than 2% above the baseline after
one round of optimisation.

Acceptance: every package's tests at their current tolerances; gpt-oss-20b's
228 checks and decode median within 2% of the baseline, with storage reuse
for every cache leaf of its layer loop; the GPT-2 trainer's loss curve bit for
bit on the CPU; rune's `Jit/jit-run-chain` and `Jit/jit-run-mlp` within 5%; a
visit test per kaun layer, example model and vega state (its leaf paths and
its reports); a CPU test that a stale holder raises (with host cells, or under
`RUNE_JIT_FORCE_COPY=1`); a test that a changed `Cache_index` window, and a
list gaining a leafless element, each compile a second program; symo's orbit
machinery ported to `Payload` as an external check.

## Laws

1. **Every part, once, in one order.** A structure's `walk` is the only walk:
   flatten, rebuild, names, casts and payload maps all run it, so they agree
   on the order. That it visits every part is held by the type checker for the
   parameter's positions, and by each structure's visit test (its leaf paths
   and reports) for fixed tensors, integers and cases. Prevents rune's marker
   pairing and a checkpoint whose names disagree with a compiled call's
   leaves.
2. **Round trip.** Rebuilding a value from its own leaves gives the value.
   Held for a `walk` that rebuilds what it visits and an `iso` whose functions
   are inverse; a test that `Nx.Ptree.map` with the
   identity returns an equal value checks it.
3. **A program is determined by its key:** leaf paths, leaf signatures and
   reports, compared by segments. Prevents replaying a program traced for
   other data.
4. **A leaf is a position,** in every transformation, and a capture is never a
   leaf, even when it is the same value as one.
5. **A role belongs to a whole argument.** `consumes` builds a signature's
   argument and nothing else.
6. **A consumed value is dead on every device, the host included.** If host
   cells fail their measurement, this law narrows as Unresolved questions
   states.
7. **Results are fresh.** No result of a call that runs its program shares
   storage with a read argument, a capture or another result.
8. **Names are stable.** Paths follow today's rules, so checkpoints written
   today load, except for the entries of fixed tensors, which today's names
   omit.

## Drawbacks

- **A forgotten fixed tensor, integer or case, or a reused case tag,
  compiles.** The type checker
  guards only the parameter's positions. A step counter that `walk` returns
  without visiting is frozen into the first program; a window it does not
  report replays a program traced for another window. Each structure's visit
  test catches both, but only where written. This is JAX's `aux_data`
  discipline, and its cost.
- **`instantiate` gives one dtype per binding,** annotated when nothing in its
  unit fixes the dtype; the payload operations take the structure's module.
- **Two currencies stay:** a structure's module (for payload maps and casts)
  and its instantiated `'s Nx.Ptree.t` (for transformations).
- **Two names for the one-tensor case** of each transformation, since OCaml has
  no default that fixes an argument's type.
- **A consumed argument needs a second operator** (`consumes caches @@ ...`).
- **Paths and cursors on the hot path** allocate about three times the words
  of a walk without them.
- **Every host value carries a cell,** four words per fresh host result and
  one per view. It lands only if its measurement passes.
- **One large sweep,** about 65 engineer-days, host cells not included.

## Rationale and alternatives

### A structure as data: revision 1 of this RFC

The strongest alternative, and this RFC's first revision. It is recorded here
so that it can be taken up again without redoing the campaign.

**The design.** A structure is described once by a value of a public GADT,
and every walk interprets that value. The lenses reviewed a one-parameter form
with a dtype-free `Tensor` leaf and a per-type `cast`; after a check against
symo, the final form took a payload parameter, type-checked in a prototype but
not reviewed by the lenses:

```ocaml
type ('p, 's, 'role) node =
  | Leaf : ('p, 'p, read) node                          (* a position of the type's parameter *)
  | Tensor : ('a, 'b) Nx.dtype -> ('p, ('a, 'b) Nx.t, read) node   (* a tensor of fixed type *)
  | Int : ('p, int, read) node                          (* data the key sees *)
  | Option : ('p, 's) t -> ('p, 's option, read) node
  | List : ('p, 's) t -> ('p, 's list, read) node
  | Pair : ('p, 'a) t * ('p, 'b) t -> ('p, 'a * 'b, read) node
  | Record : 'f * ('p, 'r, 'f) fields -> ('p, 'r, read) node
  | Variant : ('p, 's) case list -> ('p, 's, read) node
  | Consumed : ('p, 's) t -> ('p, 's, consumed) node   (* a signature's argument only *)
and ('p, 's) t = ('p, 's, read) node
and ('p, 'r, 'f) fields =                               (* rebinds [] and :: *)
  | [] : ('p, 'r, 'r) fields
  | ( :: ) : (string * ('r -> 'a) * ('p, 'a) t) * ('p, 'r, 'f) fields -> ('p, 'r, 'a -> 'f) fields
and ('p, 's) case =
  | Case : { tag : string; desc : ('p, 'a) t; inject : 'a -> 's; project : 's -> 'a option }
      -> ('p, 's) case
```

A description transcribes its type: `'a` is `Leaf`, `Nx.int32_t` is `Tensor
Nx.int32`, `'a Linear.t` is `Linear.ptree`, `int` is `Int`.

```ocaml
let ptree = Nx.Ptree.(Record ((fun w b -> { w; b }),
                              [ ("w", (fun l -> l.w), Leaf); ("b", (fun l -> l.b), Option Leaf) ]))
let bf16 = Nx.Ptree.map Mlp.ptree Mlp.ptree (Nx.cast Nx.bfloat16) master
Rune.jit Nx.Ptree.(Leaf @-> Cache_index.ptree @-> Consumed caches
                   @-> returning (Pair (Tensor Nx.int32, caches))) greedy
```

**What it established, and what carries over.** A description built only from
constructors, tuples and `fun` is a value OCaml generalises, so one binding
serves every payload; any combinator hiding the representation makes it
weakly polymorphic (a jsont-style prototype: every use a function call, 375
ns and 1,770 words for gpt-oss). A type-changing map over two instances
type-checks and carries fixed tensors by `Dtype.equal_witness`, with no `Obj`.
A generic cast over a leaf not tied to the payload does not type-check: a walk
reaching a target leaf has nothing of that leaf's type to return. Everything
in §A compiled call, §A cell for host storage and the transformation rule came
from that campaign and stands in this revision. A revisit re-applies its
rulings on descriptions: a variant case keyed by its position, pair segments
`0` and `1`, field lists that rebind `[]` and `::` only inside `fields`, role
types defined with constructors so that interpreters need no `Consumed` case,
`fn` private, and Law 2's premise that getters, `make`, `inject` and
`project` are inverse.

**What it guaranteed that this revision does not.** Every field is visited,
since `make` demands each one and every operation reads one list: a forgotten
fixed tensor or integer is a type error. Static data, variant cases, list
lengths and option presence are in the key by construction, even where they
hold no leaf, with no author discipline. A structure can be read without
running it, so an interpreter can check or compile a structure before any
value exists.

**What it cost.**
- A public GADT that can never be made abstract.
- Two type parameters in every description and in about 45 public signatures.
- A rule for when a field is `Leaf` and when it is `Tensor dt`; getting it
  wrong ties the description to one dtype, and the compiler reports it at a
  distant second use.
- `[]` and `::` rebound for field lists, so an ordinary list of unknown type
  inside `Nx.Ptree.( ... )` means something else.
- Variants that write each constructor twice, as `inject` and `project`:
  `Cache_index` in 37 lines against 15 as a `match`.
- A description passed once per payload type (`map d d f`, `map2 d d d f`),
  since a value cannot name a type constructor.
- Flatten at 1.2 to 1.4 times today's module and rebuild at 1.4 to 1.8 (12.6
  µs against 7.9 at gpt-oss's 459 leaves); the one-function module without
  paths costs 1.04 to 1.09 up to 1,000 leaves and more than the description
  at 10,000. All of it is under 0.1% of a decode step, and cost decided
  nothing.

**The campaign's vote.** Three of its four designs and its shape lens chose
this form, for the key and for completeness; its cost lens found the designs
within about 7 engineer-days and 500 lines of each other. Revision 2 answers
the key with reports and completeness with visit tests, and chose on the
grounds below.

**Why it lost.** Its guarantees concern few type definitions (vega's states,
`Loss_scale`, `Cache_index`, RFC 0004's formats, the gpt-oss block), each of
which every compiled call reaches through a training or decode state, and each
coverable by `walk`'s contract and a visit test. What almost every coming use
of a structure needs is a payload map: casts, masks, per-leaf learning rates,
RFC 0005's sharding plans, symo's dimensions and symmetries. OCaml abstracts
over a type constructor with a module, so one `walk` in a module serves all of
them natively, where descriptions reproduce it with a second type parameter
and repeated arguments. It also keeps structures as ordinary OCaml, costs
external users like symo the least, and gains from the language's direction:
modular explicits landed in 5.5, and modular implicits would remove
`instantiate` and the module arguments altogether, where a public GADT is
fixed for good.

**Revisit it if** raven needs to read structures without running them (a
schema, a compile-time plan over a whole model, checks at definition); if
forgotten fixed fields and reports become a recurring class of bug that visit
tests do not stop; or if `nest` and the module arguments of payload maps prove
a recurring cost that modular implicits do not remove. The campaign's audit,
probe, four designs, four lenses and fold checks are kept outside the
repository, with the maintainer.

### Other alternatives

**Dropping the single-tensor shorthands.** One name per transformation, and
`Rune.grad Nx.Ptree.tensor f x` for the one-tensor case. It lengthens the
numpy-like entry point that tests, notebooks and tutorials use most (408
calls), and `jit'` and `vmap'` would become signatures. The ceremony this RFC
removes was in the structured paths.

**Removing ppx_ptree.** A `walk` is one `field` line per field, and each
structure's visit test guards a hand-written one, so a deriver saves little;
removing it drops a package and a ppxlib dependency. It keeps a second way to
write a structure, but the deriver reports every case and length and refuses
an unmarked integer, which closes, for the types it derives, the one gap this
revision leaves to discipline.

**A state noun** (`Nx.Var`: a cell owning storage that programs write in
place). It removes stale holders by construction. It makes a compiled
function's type silent about what it writes, makes calls order-dependent,
forces `remat` and staged `scan` bodies to refuse writes, turns the layer loop
into one compiled function per layer (about 490 traces at DeepSeek start-up
against 16), and reverses RFC 0001's Law 1. Host cells give the part of its
safety that values can have.

**`jit_step`'s model with a third module for outputs.** Fixes the late read,
and keeps a module per role and a second function shape beside `jit`. Its
positional pairing is what made a loss a field of the state.

**Static data compared structurally** (a copy of the value with leaves erased,
compared with polymorphic equality). No author discipline, but it raises on a
closure and compares by content any tensor the `walk` forgot, turning a frozen
leaf into a silent recompile or none.

**A visitor record in `walk`** (`v.leaf (Path.(p / "w")) w`, this revision's
first draft). Every structure built paths by hand and read callbacks as record
fields, which needs a type annotation for disambiguation; the cursor carries
the path and the walk functions name what each part is.

**Roles at any depth, and a `mapped` annotation for `vmap`.** A role inside a
structure must then be ignored or rejected by `Checkpoint`, vega and `grad`,
and no call site in view needs one that currying does not give.

**Host values that cannot be consumed** (today's CPU). One cell cheaper per
host tensor, and it leaves RFC 0001's Law 3 false on the device every test
suite runs on. It is this RFC's fallback if host cells fail their measurement.

**n-ary signatures for `grad`** (a `wrt` role marking the arguments to
differentiate). A design pass built it: one GADT serves `grad`,
`value_and_grad`, `vjp`, `jvp`, `hvp` and `custom_vjp`, a signature with no
mark is a type error, and 25 numerical checks pass. It needs six type
indices, which every misuse error prints in full. A let-bound signature
serves only one transformation, because one index is weak. rune would need
its own `@->` beside `Nx.Ptree`'s. Of 202 differentiation call sites, 33, all
in rune's tests, differentiate several values, and every training step reads
the same or worse. A closure over `pair` expresses the same selection. `scan`
keeps three structures: a signature would state the carry twice, and its
result pair is opaque to `scan`.

**Consumption marked at the call site** (`step (Rune.consume s)`) repeats what
the signature must know anyway.

Precedent: JAX registers a structure as a pair of functions, `flatten`
returning the leaves and hashable `aux_data` that enters the program key, and
`unflatten`; Haskell's `Traversable` and the lens `Traversal` are one
traversal from which the rest derives; ctypes describes a C function by `@->`
and `returning`; jsont types its paths, adapts a mapping to another type by
its two conversions, and reports every error with its path.

## Non-goals

- Typing linearity: consumption is checked at the first use, on every device.
- A registry of structure types (JAX's pytree registration): hidden global
  state.
- Per-leaf or non-zero `vmap` axes.

## Unresolved questions

Before host cells land:
- **Their cost.** Measured on RFC 0005's stage-1 build, with and without host
  cells, in alternating paired runs, with overhead rows added to nx's and
  rune's benchmarks. Host cells stay if every overhead row is within 5% or 10
  ns, a rune eager training row within 2%, every other nx row within 3%, and
  every alloc row grows by exactly four words per fresh host result and one
  per view. Otherwise host values carry no cell, CPU tests of consumption set
  `RUNE_JIT_FORCE_COPY=1`, and Law 6 reads "on every device, and on the CPU
  under `RUNE_JIT_FORCE_COPY=1`".

## Future possibilities

- A walker over a walker: `Walk.nest outer inner` would walk a container's
  positions with another structure's `walk` (Adam's state over a model's
  parameters), so a whole training state could be cast or payload-mapped. It
  needs every part to export its walker beside its `'s t`; `Walk.structure`
  covers naming and masks without it.
- Policies for absent and unknown checkpoint entries (`?absent`, `?unknown` on
  `Checkpoint.to_value`), once a load needs them: a base checkpoint without adapter weights, a strict
  load that refuses extra entries.
- Queries and updates by path (`Nx.Ptree.find`, `Nx.Ptree.update`) for
  surgical edits: injecting adapters, freezing a sub-tree, loading only an
  encoder.
- Modular implicits, if OCaml gains them, would resolve a structure's module
  from its type and remove `instantiate` and the payload operations' module
  arguments.
- A description of a structure could be recorded by running its `walk` once on
  a value, if a need to read structures without running them appears.

Nothing listed here is a reason to accept this or a later RFC.
