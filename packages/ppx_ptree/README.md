# Ppx_ptree

`ppx_ptree` derives the `walk` of a structure from its type. A structure
(`Nx.Ptree.S`) is a type `'a t` with one function, `walk`, that visits the
parts of a value with an `Nx.Ptree.Walk` cursor; every transformation,
optimizer and checkpoint works from it. The deriver writes the function a
hand-written structure would:

```ocaml
module Mlp = struct
  type 'a t = {
    l1 : 'a Kaun.Linear.t;
    l2 : 'a Kaun.Linear.t;
    window : int option; [@ptree.int]
    name : string; [@ptree.skip]
  }
  [@@deriving ptree]
end

let mlp = Nx.Ptree.instantiate (module Mlp)
```

The derived `walk` visits `l1` and `l2` with `Kaun.Linear.walk` under their
field names, reports the window's presence and value, and copies the name.
It is equivalent to:

```ocaml
let walk c x =
  let l1 = Nx.Ptree.Walk.field c "l1" Kaun.Linear.walk x.l1 in
  let l2 = Nx.Ptree.Walk.field c "l2" Kaun.Linear.walk x.l2 in
  let window = Nx.Ptree.Walk.field c "window" Nx.Ptree.Walk.(option int) x.window in
  { l1; l2; window; name = x.name }
```

A derived walk and a hand-written one are interchangeable; raven's own
packages write theirs by hand.

## Setup

```lisp
(library
 (libraries nx)
 (preprocess (pps ppx_ptree)))
```

The generated code uses `Nx.Ptree`, so the library depends on `nx`. The
deriver adds no runtime dependency.

## Generated values

- `walk` for a type named `t`, and `walk_name` for a type `name`.
- For a type without a parameter, also `ptree : t Nx.Ptree.t` (or
  `ptree_name`), its structure at its one type:

  ```ocaml
  type state = { scale : Nx.float32_t; steps : Nx.int32_t } [@@deriving ptree]
  (* walk_state, and ptree_state : state Nx.Ptree.t *)
  ```

  A type with a parameter has one structure per payload type, so it gets no
  `ptree`; `Nx.Ptree.instantiate (module M)` builds one where the dtype is
  known.
- In an interface, `[@@deriving ptree]` declares the same values.

## How a part is walked

Records walk their fields in declaration order under the fields' names.
Variants report the constructor's name as their case, then walk their
arguments: one argument at the constructor's path, several at indices `0`,
`1`, ..., an inline record's fields by name.

| Part's type | Walk |
|---|---|
| the parameter `'a` | `Walk.leaf` |
| `('x, 'y) Nx.t`, `Nx.float32_t` and the other aliases | `Walk.tensor` |
| `ty option`, `ty list` | `Walk.option`, `Walk.list` |
| `ty array` | length reported with `Walk.int`, then each element at its index |
| a tuple | each component with `Walk.index` |
| `'a M.t`, `'a M.name` | `M.walk`, `M.walk_name` |
| `'a name` of the declaration group or defined before it | `walk_name` |
| `M.t`, `M.name` | `Walk.structure M.ptree`, `Walk.structure M.ptree_name` |
| `name` of the declaration group | `walk_name` |
| `name` defined before it | `Walk.structure ptree_name` |
| `Nx.float32_t M.t`, `state M.t`: a fixed instance of a module's `t` | `Walk.structure (Nx.Ptree.nest (module M) s)`, with `s` the payload's structure: `Nx.Ptree.tensor`, `ptree_state`, ... |
| `int`, `bool` under `[@ptree.int]` | `Walk.int` |

A qualified type without arguments is taken to be a structure at one type
named by the `ptree` convention, as `Kaun.Cache_index.ptree` and
`Nx_quant.ptree` are. A type that follows neither convention fails to compile
at the part's type, for example with `Unbound value M.walk`. A `ptree` of
another type, or an `[@ptree.walk f]` whose `f` does not walk the part's type,
is reported at the part too.

## Attributes

An attribute goes after a record field, after a constructor's single
argument (`Frozen of Nx.float32_t [@ptree.skip]`), or on a type to annotate a
nested part: `(int [@ptree.int]) list`. A part takes at most one. On a
constructor with no argument or several, or on the type declaration, it is an
error.

- `[@ptree.int]` reports the part's integers and bools. A compiled program is
  cached per reported value: mark an integer that changes what a program
  computes, such as a window or a block size.
- `[@ptree.skip]` leaves the part out of the walk; the rebuilt value keeps it.
  It is for data no compiled program depends on, such as a name, and its type
  must not mention the parameter.
- `[@ptree.walk f]` walks the part with `f`. A part that has a structure at one
  type and no module, such as an optimizer state, is
  `[@ptree.walk Nx.Ptree.Walk.structure (Vega.adam_ptree mlp)]`.

An `int` or `bool` without an attribute is an error. Reporting it would compile
a program per value for data no program reads, and leaving it out would freeze
data a program reads into its first trace; the attribute says which it is.

## Errors

The deriver reports each unsupported part at its source location: scalar
data without an attribute, functions, objects, polymorphic variants,
first-class modules, GADT constructors, abstract, extensible and private
types, types with more than one parameter, and the parameter inside a tensor
type or a skipped part. For a field `count : int`:

```
Error: ppx_ptree: field [count] is an int; report it with [@ptree.int] if a
compiled program depends on it, or leave it out with [@ptree.skip]
```

## Example

The [linear-regression example](examples/01-rune-linear-regression/) derives
a parameter record and trains it with `Rune.grad` under `Rune.jit`:

```sh
dune exec packages/ppx_ptree/examples/01-rune-linear-regression/main.exe
```

## License

ISC License. See [LICENSE](../../LICENSE) for details.
