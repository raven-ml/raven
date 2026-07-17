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

## Example

The [Rune linear-regression example](examples/01-rune-linear-regression/)
derives a parameter module and passes it directly to `Rune.grad` and
`Rune.jit2`:

```sh
dune exec packages/ppx_ptree/examples/01-rune-linear-regression/main.exe
```

## License

ISC License. See [LICENSE](../../LICENSE) for details.
