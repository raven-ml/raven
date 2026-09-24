# Ppx_jit

`ppx_jit` compiles a function with `Rune.jit2` without the boilerplate of
hand-written input and output `Ptree.S` modules. Annotate the function:

```ocaml
let[@jit] f (a : A.t) (b : B.t) (c : C.t) : D.t * E.t =
  let d, e = ... in
  (d, e)
```

and `f` is the compiled function, with the same type as the eager original —
signature files do not change. The attribute expands to an input module
packing the arguments into a record, an output module packing the result into
a one-field record, and the `Rune.jit2` call wrapped to recover the original
calling convention; see
[test/expansion_cases/simple.expected](test/expansion_cases/simple.expected)
for the full expansion. The traversals of the generated modules are derived
with [ppx_ptree](../ppx_ptree/).

## Arguments and results

Every argument must carry a type annotation: the annotation selects the
structure used to traverse the argument, exactly as a field type does for
`[@@deriving ptree]`. A qualified type such as `A.t` delegates to `A`'s
traversals, a concrete tensor type such as `Nx.float32_t` is a leaf, and
tuples, `option`, `list` and `array` compose. Arguments with no traversable
content (hyperparameters, configuration) are not supported yet; capture them
or recompile per value.

Labelled and optional arguments keep their syntax and defaults:

```ocaml
let[@jit] train ~(params : Params.t) ?(mask : Mask.t = Mask.none) batch : Params.t = ...
```

The result type annotation is required and may be any type `ppx_ptree` can
traverse — a single `D.t`, a tuple `D.t * E.t`, or a composite such as
`(D.t * E.t) option`.

## Compilation options

The optional payload is a record whose fields are spliced verbatim into the
`Rune.jit2` call:

```ocaml
let device = if on_gpu then "NV" else "CPU"

let[@jit { device; beam = 8 }] f (a : A.t) : B.t = ...
```

Fields may be arbitrary expressions, but they are evaluated once, when the
enclosing module is initialised: the compilation cache lives in the partial
application of `Rune.jit2`, so `device`, `beam` and `beam_parallel` are
properties of the compiled function, not of a call. When the payload is
omitted, the usual `DEV`/`BEAM` environment defaults apply.

## Semantics and limitations

The compiled function is created once, at module initialisation; tensors the
body captures are compile-time constants, as with a manual `Rune.jit2`
wrapper. Under an enclosing transformation (`Rune.grad`, `Rune.vmap`, an
outer `jit`) the wrapped function runs directly, so the attribute composes
with differentiation exactly as the hand-written form does.

The attribute applies to non-recursive, single `let` bindings at module
level. Arguments must be annotated variables — patterns such as `(a, b)` are
rejected — and every argument is traced. Violations are compile-time errors
pointing at the offending form.

## Build setup

Add `ppx_jit` as a PPX and depend on `rune`, since generated code calls
`Rune.jit2`:

```lisp
(library
 (libraries rune nx)
 (preprocess (pps ppx_jit)))
```

The PPX adds no runtime dependency. `ppx_ptree` does not need to be listed
in `pps` unless the file also uses `[@@deriving ptree]`.

## License

ISC License. See [LICENSE](../../LICENSE) for details.
