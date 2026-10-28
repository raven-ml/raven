# Ppx_ptree

`ppx_ptree` derives the `walk` of a structure from its type. A structure
(`Nx.Ptree.S`) is a type `('a, 'd) t` with one function, `walk`, that visits
the parts of a value with an `Nx.Ptree.Walk` cursor: the positions of its
payload `'a` and its fixed arrays, on the device set `'d`. Every
transformation, optimizer and checkpoint works from it. The deriver writes
the function a hand-written structure would:

```ocaml
module Mlp = struct
  type ('a, 'd) t = {
    l1 : ('a, 'd) Kaun.Linear.t;
    l2 : ('a, 'd) Kaun.Linear.t;
    steps : 'd Nx.int32_t;
    window : int option; [@ptree.int]
    name : string; [@ptree.skip]
  }
  [@@deriving ptree]
end

let mlp = Nx.Ptree.instantiate (module Mlp)
```

The derived `walk` visits `l1` and `l2` with `Kaun.Linear.walk` under their
field names, walks `steps` as a fixed array, reports the window's presence
and value, and copies the name. It is equivalent to:

```ocaml
let walk c x =
  let open Nx.Ptree.Walk in
  let l1 = field c "l1" Kaun.Linear.walk x.l1 in
  let l2 = field c "l2" Kaun.Linear.walk x.l2 in
  let steps = field c "steps" tensor x.steps in
  let window = field c "window" (option int) x.window in
  { l1; l2; steps; window; name = x.name }
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

## Declarations

A derived type has at most two parameters, read by position:

| Declaration | Holds | Walk |
|---|---|---|
| `('a, 'd) t` | payload `'a`, fixed arrays on `'d` | `('a, 'b, 'd, 'e) cursor -> ('a, 'd) t -> ('b, 'e) t` |
| `'d t` | fixed arrays only, as `Nx_quant.t` and `Kaun.Cache_index.t` | `('a, 'b, 'd, 'e) cursor -> 'd t -> 'e t` |
| `t` | no array: integers and cases | `('a, 'b, 'd, 'e) cursor -> t -> t` |

Either parameter of `('a, 'd) t` may be anonymous: `('a, _) t` holds no
fixed array, `(_, 'd) t` no payload. A fixed array lies on the type's set
parameter; a tensor type on any other set, such as `Nx.host Nx.bool_t`, is
an error at the part.

## Generated values

- `walk` for a type named `t`, and `walk_name` for a type `name`, with the
  types above.
- In an interface, `[@@deriving ptree]` declares the same values.

No description is generated, since a description is at one dtype and one
set, which its first use fixes. Make it where it is used:
`Nx.Ptree.instantiate (module M)` for `('a, 'd) t`, and `Nx.Ptree.v M.walk`
for `'d t` and `t`. `Nx.Ptree.place` and `Nx.Ptree.cast` take the module of
a type of two parameters.

## How a part is walked

Records walk their fields in declaration order under the fields' names.
Variants report the constructor's name as their case, then walk their
arguments: one argument at the constructor's path, several at indices `0`,
`1`, ..., an inline record's fields by name.

| Part's type | Walk |
|---|---|
| the payload `'a` | `Walk.leaf` |
| `('v, 's, 'd) Nx.t`, `'d Nx.float32_t` and the other aliases | `Walk.tensor` |
| `'d Nx.Rng.key` | `Nx.Rng.walk` |
| `'d Nx.packed` | `Walk.tensor` of the packed array |
| `ty option`, `ty list` | `Walk.option`, `Walk.list` |
| `ty array` | length reported with `Walk.int`, then each element at its index |
| a tuple | each component with `Walk.index` |
| `(args) M.t`, `(args) M.name`, `M.t` | `M.walk`, `M.walk_name`, `M.walk` |
| `(args) name` of the declaration group or defined before it | `walk_name` |
| `(p, 'd) M.t` where `p` mentions a parameter but is not `'a` | `Walk.under M.walk w`, `w` the walk of `p` |
| `int`, `bool` under `[@ptree.int]` | `Walk.int` |

The nested rule walks an optimizer state over a model,
`(('a, 'd) Mlp.t, 'd) Vega.Adam_state.t`, as
`under Vega.Adam_state.walk Mlp.walk`, and a frozen layer,
`('d Nx.float32_t, 'd) Kaun.Linear.t`, as `under Kaun.Linear.walk tensor`.

The deriver names each walk by convention and constrains it to the part's
type, so OCaml reports a mismatch at the part: `Unbound value M.walk` for a
module without one, a type error for a structure applied in another order
or at a closed payload such as `(int, 'd) Kaun.Linear.t`.

## Attributes

An attribute goes after a record field, after a constructor's single
argument (`Frozen of Nx.host Nx.float32_t [@ptree.skip]`), or on a type to
annotate a nested part: `(int [@ptree.int]) list`. A part takes at most one.
On a constructor with no argument or several, or on the type declaration, it
is an error.

- `[@ptree.int]` reports the part's integers and bools. A compiled program is
  cached per reported value: mark an integer that changes what a program
  computes, such as a window or a block size.
- `[@ptree.skip]` leaves the part out of the walk; the rebuilt value keeps it.
  It is for data no program on the set reads, such as a name or a host
  array, and its type mentions neither parameter.
- `[@ptree.walk f]` walks the part with `f`, of type
  `('a, 'b, 'd, 'e) cursor -> ty -> ty'`. A walk that moves values between
  sets rejects an `f` at one set, such as `Nx.Ptree.Walk.structure s`: walk
  the part with its structure's walk or with `Walk.under`.

An `int` or `bool` without an attribute is an error. Reporting it would compile
a program per value for data no program reads, and leaving it out would freeze
data a program reads into its first trace; the attribute says which it is.

## Errors

The deriver reports each unsupported part at its source location: scalar
data without an attribute, a fixed array off the set parameter, the set
parameter alone, `Nx.Placement.t` and other `Nx` types without a walk,
functions, objects, polymorphic variants, first-class modules, GADT
constructors, abstract, extensible and private types, types with more than
two parameters, and a parameter inside a skipped part. For a field
`mask : Nx.host Nx.bool_t` of `('a, 'd) t`:

```
Error: ppx_ptree: field [mask] has type Nx.host Nx.bool_t, an array on
Nx.host; a fixed array lies on the type's set parameter 'd: write 'd there,
or leave the part out with [@ptree.skip] if no program on the set reads it
```

## Example

The [linear-regression example](examples/01-rune-linear-regression/) derives
a parameter record and trains it with `Rune.grad` under `Rune.jit`, donating
the parameters at each call:

```sh
dune exec packages/ppx_ptree/examples/01-rune-linear-regression/main.exe
```

## License

ISC License. See [LICENSE](../../LICENSE) for details.
