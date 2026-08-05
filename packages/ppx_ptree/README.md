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

The same attribute on a payload-generic type — one type parameter, occurring
outside tensor leaves — derives the rank-1 uniform traversals instead; see
[Uniform types](#uniform-payload-generic-types) below.

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

## Uniform (payload-generic) types

When a type has exactly one type parameter and that parameter occurs outside
tensor leaves, `[@@deriving ptree]` derives rank-1, type-changing traversals
instead — the `Nx.Ptree.Uniform` shape:

```ocaml
type 'a t = { w : 'a; b : 'a option; name : string }
[@@deriving ptree]
```

This produces `map`, `map2`, `iter`, `fold`, `fold2`, and `names`:

```ocaml
val map   : ('a -> 'b) -> 'a t -> 'b t
val map2  : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
val iter  : ('a -> unit) -> 'a t -> unit
val fold  : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
val fold2 : (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
val names : 'a t -> string t
```

`fold` and `fold2` pass each payload leaf's path to the callback, and `names`
is the tree of those paths, computed from a value so optional and variable-size
containers resolve. A path is the dot-joined sequence of record field names and
container indices from the root — `"w"`, `"layers.0"`, `"pair.1"` — matching
the checkpoint naming convention; `Some` contributes no segment, and a payload
at the root has the empty path.

Payload positions are the positions where `'a` occurs directly — not inside an
`Nx.t` leaf. Fields without `'a` are static metadata: preserved by `map`,
left-biased by `map2`, skipped by `iter`/`fold`/`fold2`, copied by `names`.
The `[@ptree.*]` attributes do not apply to payload positions of uniform
types; a field whose bare `'a` carries an attribute keeps its rank-2 meaning
for mode selection, so records like `{ tag : 'tag [@ptree.ignore]; ... }`
continue to derive the classic traversals.

### Nesting

A field `'a sub` (a sibling type in the same declaration group) or `'a Sub.t`
delegates to the sibling's or `Sub`'s uniform traversals. Delegated types must
be applied to the payload parameter alone.

### Bridge to Ptree.S

A uniform type instantiated at `Nx.Ptree.tensor` satisfies `Nx.Ptree.S` via
the `Nx.Ptree.Make` functor:

```ocaml
type 'a params = { w : 'a; b : 'a }
[@@deriving ptree]

module P = Nx.Ptree.Make (struct
  type 'a t = 'a params
  let map = map
  let map2 = map2
  let iter = iter
end)

let g = Rune.grad (module P) loss params
```

### Mirror mode (concrete records)

`[@@deriving ptree ~mirror]` on a concrete type additionally generates a
uniform mirror alongside the rank-2 traversals:

- `module Uniform` — the payload-generic mirror `'m t` with all six uniform
  traversals; static fields keep their original types.
- `val to_uniform : t -> Nx.Ptree.tensor Uniform.t` — packs tensor leaves.
- `val of_uniform : Nx.Ptree.tensor Uniform.t -> t` — unpacks with dtype
  checks (generated only when every leaf dtype is statically known, e.g.
  `Nx.float32_t`; errors carry the leaf's path).

```ocaml
type t = { w : Nx.float32_t; b : Nx.float16_t }
[@@deriving ptree ~mirror]

(* uniform traversals on the mirror view: *)
let symmetrize (params : t) (syms : Symmetry.t Uniform.t) : t =
  let u = to_uniform params in
  let zipped =
    Uniform.map2
      (fun (Nx.Ptree.P x) sym -> Nx.Ptree.P (Symmetry.project sym x))
      u syms
  in
  of_uniform zipped
```

A field `linear : Linear.t` maps to `'m Linear.Uniform.t` in the mirror;
`Linear` must itself derive `[@@deriving ptree ~mirror]`. Mirror mode applies
to a single declaration whose definition is visible, without locally declared
or recursive sub-structures.

## Example

The [Rune linear-regression example](examples/01-rune-linear-regression/)
derives a parameter module and passes it directly to `Rune.grad` and
`Rune.jit2`:

```sh
dune exec packages/ppx_ptree/examples/01-rune-linear-regression/main.exe
```

## License

ISC License. See [LICENSE](../../LICENSE) for details.
