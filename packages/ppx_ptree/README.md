# Ppx_ptree

`ppx_ptree` derives the rank-2 tensor traversals required by `Nx.Ptree.S`:

```ocaml
type t = {
  weight : Nx.float32_t;
  bias : Nx.float32_t option;
  name : string [@ptree.ignore];
}
[@@deriving ptree]
```

The declaration generates `map`, `map2`, and `iter`. A type with any other
name generates suffixed functions, such as `map_state`, `map2_state`, and
`iter_state`. This lets one declaration group contain helper structures while
reserving the unsuffixed names for one primary `t` or `params` type.

## Supported shapes

Tensor leaves may use `('a, 'b) Nx.t`, `Nx_effect.t`, or any of Nx's concrete
tensor aliases. Records, tuples, `option`, `list`, and `array` compose
recursively. Qualified `M.t` and `M.params` types delegate to `M.map`,
`M.map2`, and `M.iter`.

Use attributes when syntax alone cannot express the intended role:

- `[@ptree.leaf]` treats the annotated type as a tensor leaf. OCaml still
  checks that it is an `Nx.t`.
- `[@ptree.ignore]` copies metadata in `map`, takes the left value in `map2`,
  and skips it in `iter`.
- `[@ptree.using M]` delegates a subtree to module `M`.

Attributes on record labels apply to the whole field. Put an attribute on a
core type to annotate a nested component, for example
`(Nx.Rng.key [@ptree.leaf]) option`.

Variants and dynamic tree representations are intentionally out of scope.
Container constructors and lengths, as well as ignored values, must remain
stable for the lifetime of a Rune JIT closure.

## Build setup

Add `ppx_ptree` as a PPX and depend directly on `nx`, since generated code uses
the public `Nx.t` type:

```lisp
(library
 (libraries nx)
 (preprocess (pps ppx_ptree)))
```

The PPX adds no runtime dependency.

## Deriving uniform gtree traversals

`[@@deriving gtree]` generates rank-1, type-changing traversals for
*higher-kinded data* — structures whose payload positions all share a single
type parameter:

```ocaml
type 'a t = { w : 'a; b : 'a; name : string }
[@@deriving gtree]
```

This produces `map`, `map2`, `iter`, `fold`, `fold2`, and `names` directly on
`'a t`. The `fold` and `fold2` operations receive a dotted path string at
each leaf, built from field names and container indices:

```ocaml
val fold  : (string -> 'a -> 'acc -> 'acc) -> 'a t -> 'acc -> 'acc
val fold2 : (string -> 'a -> 'b -> 'acc -> 'acc) -> 'a t -> 'b t -> 'acc -> 'acc
```

Path convention (stable for JIT caching):

- Root starts at `""`.
- Record field `w` appends `".w"`.
- List/array item at index `i` appends `".i"`.
- Tuple positions use `".0"`, `".1"`, etc.

Payload positions are fields (or nested positions) where the type parameter
`'a` occurs directly — not inside an `Nx.t` leaf. Fields without `'a` are
static metadata, following the same semantics as `[@ptree.ignore]`: preserved
by `map`, left-biased by `map2`, skipped by `iter`/`fold`/`fold2`.

### Nesting

A field `'a Sub.t` delegates to `Sub.map`, `Sub.fold`, etc. — `Sub` must also
derive `[@@deriving gtree]`.

### Names

`val names : string t` is a constant tree of dotted paths, derived when the
structure is statically known (no payload `option`/`list`/`array`). Omitted
otherwise.

### Bridge to Ptree.S

A gtree instantiated with `Nx.Ptree.tensor` as its payload satisfies
`Nx.Ptree.S` via the `Nx.Ptree.Tensor_tree` functor:

```ocaml
type 'a params = { w : 'a; b : 'a }
[@@deriving gtree]

type t = Nx.Ptree.tensor params

module T = Nx.Ptree.Tensor_tree (struct
  type 'a t = 'a params
  let map = map
  let map2 = map2
  let iter = iter
end)

let loss (p : T.t) = ...
let g = Rune.grad (module T) loss params
```

(The `[@@deriving ptree, gtree]` combination on a static record automates the
packing — see mirror mode below.)

### Mirror mode (static records)

When applied to a concrete record without a type parameter,
`[@@deriving ptree, gtree]` generates:

- `module Gtree` — the uniform mirror type `'a t` + all six traversals
- `val to_gtree : t -> Nx.Ptree.tensor Gtree.t` — packs tensor leaves
- `val of_gtree : Nx.Ptree.tensor Gtree.t -> t` — unpacks with dtype checks
  (only when all leaf dtypes are concrete, e.g. `Nx.float32_t`)

```ocaml
type t = { w : Nx.float32_t; b : Nx.float16_t }
[@@deriving ptree, gtree]

(* Gtree traversals on the uniform view: *)
let symmetrize (params : t) (syms : Symmetry.t Params.Gtree.t) : t =
  let u = to_gtree params in
  let zipped = Params.Gtree.map2 (fun (Nx.Ptree.P x) sym ->
    Nx.Ptree.P (Symmetry.project sym x)) u syms
  in
  of_gtree zipped
```

Also works for nested models where each sub-module derives `[@@deriving ptree, gtree]`:
a field `linear : Linear.t` maps to `'a Linear.Gtree.t` in the mirror.

## Example

The [Rune linear-regression example](examples/01-rune-linear-regression/)
derives a parameter module and passes it directly to `Rune.grad` and
`Rune.jit2`:

```sh
dune exec packages/ppx_ptree/examples/01-rune-linear-regression/main.exe
```

## License

ISC License. See [LICENSE](../../LICENSE) for details.
