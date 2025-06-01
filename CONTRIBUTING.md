# Contributing to Raven

## Documentation Style

### Overview

This guide establishes documentation conventions for the raven. We follow the Unix philosophy: terse, precise, no fluff. Document contracts and invariants, not implementation details.

### General Principles

1. **Be imperative and active** - "Creates tensor" not "This function creates a tensor"
2. **Document invariants, not implementation** - What must be true, not how it works
3. **Mention performance only when surprising** - O(1) views vs O(n) copies
4. **No redundant information** - If it's obvious from the type, don't repeat it

### Documentation Template

```ocaml
val zeros : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [zeros dtype shape] creates zero-filled tensor.   (* <-- function application pattern *)

    Extended description if needed. State invariants.  (* <-- optional extended info *)

    @raise Exception_name if [condition]               (* <-- exceptions *)

    Example creating a 2x3 matrix of zeros:            (* <-- example with description *)
    {[
      let t = Nx.zeros Nx.float32 [|2; 3|] in
      Nx.to_array t = [|0.; 0.; 0.; 0.; 0.; 0.|]
    ]} *)
```

### Formatting Conventions

#### Code References
- Use `[code]` for inline code: parameter names, function names, expressions
- Use `{[ ... ]}` for code blocks
- No backticks - this is odoc, not Markdown

#### First Line
Always start with: `[function_name arg1 arg2] does X`
Not: "Creates a tensor with..." or "This function..."

#### Mathematical Notation
- Use ASCII: `a * b`, not `a × b`
- Use `x^2` or `x ** 2` for powers
- Use `[start, stop)` for half-open intervals

### What to Document

✓ **Invariants and preconditions**: "Length of [data] must equal product of [shape]."  
✓ **Surprising performance**: "Returns view if possible (O(1)), otherwise copies (O(n))."  
✓ **Shape transformations**: "Result has shape [|m; n|] where m = length of [a]."

✗ **Not**: obvious information, implementation details, or redundant parameter descriptions

### Code Examples

Must be valid, compilable OCaml:
- Use qualified names (`Nx.function` not `open Nx`)
- Show expected results with `=`
- Each example in its own `{[ ... ]}` block with a description before it
- Self-contained (independently executable)

### Examples

#### Function with Constraints
```ocaml
val arange : ('a, 'b) dtype -> int -> int -> int -> ('a, 'b) t
(** [arange dtype start stop step] generates values from [start] to [stop).

    Step must be non-zero. Result length is [(stop - start) / step] rounded
    toward zero.

    @raise Failure if [step = 0]

    Generating even numbers from 0 to 10:
    {[
      let t1 = Nx.arange Nx.int32 0 10 2 in
      Nx.to_array t1 = [|0l; 2l; 4l; 6l; 8l|]
    ]} *)
```

#### Function with Multiple Behaviors
```ocaml
val dot : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t
(** [dot a b] computes generalized dot product.

    For 1-D tensors, returns inner product (scalar). For 2-D, performs
    matrix multiplication. Otherwise, contracts last axis of [a] with
    second-last of [b].

    @raise Invalid_argument if contraction axes have different sizes

    Computing inner product of two vectors:
    {[
      let v1 = Nx.of_array Nx.float32 [|1.; 2.|] in
      let v2 = Nx.of_array Nx.float32 [|3.; 4.|] in
      let scalar = Nx.dot v1 v2 in
      Nx.to_scalar scalar = 11.
    ]} *)
```

#### Optional Parameters
```ocaml
val sum : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [sum ?axes ?keepdims t] sums elements along specified axes.

    Default sums all axes. If [keepdims] is true, retains reduced
    dimensions with size 1.

    @raise Invalid_argument if any axis is out of bounds

    Summing all elements:
    {[
      let t = Nx.of_array Nx.float32 ~shape:[|2; 2|] [|1.; 2.; 3.; 4.|] in
      Nx.to_scalar (Nx.sum t) = 10.
    ]}

    Summing along rows (axis 0):
    {[
      let t = Nx.of_array Nx.float32 ~shape:[|2; 2|] [|1.; 2.; 3.; 4.|] in
      let sum_axis0 = Nx.sum ~axes:[|0|] t in
      Nx.to_array sum_axis0 = [|4.; 6.|]
    ]} *)
```

### Special Documentation Cases

**Broadcasting**: Always explain compatibility rules
```ocaml
(** [add t1 t2] computes element-wise sum with broadcasting.

    Shapes must be broadcast-compatible: each dimension must be equal
    or one of them must be 1. *)
```

**Memory behavior**: Be explicit about views vs copies
```ocaml
(** [transpose t] returns view with swapped axes (no copy). *)
(** [flatten t] returns new 1-D tensor (always copies). *)
(** [reshape shape t] returns view if possible, otherwise copies. *)
```

**Complex shapes**: Use examples to clarify
```ocaml
(** [stack axis tensors] stacks along new axis at position [axis].

    All tensors must have identical shape. Result has rank + 1.

    Stacking two 2x2 matrices along a new first axis:
    {[
      let t1 = Nx.of_array Nx.int32 ~shape:[|2; 2|] [|1l; 2l; 3l; 4l|] in
      let t2 = Nx.of_array Nx.int32 ~shape:[|2; 2|] [|5l; 6l; 7l; 8l|] in
      let stacked = Nx.stack ~axis:0 [t1; t2] in
      Nx.shape stacked = [|2; 2; 2|] &&
      Nx.to_array stacked = [|1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l|]
    ]} *)
```

### Module-level Documentation

```ocaml
(** N-dimensional array operations.

    This module provides NumPy-style tensor operations for OCaml.
    Tensors are immutable views over mutable buffers, supporting
    broadcasting, slicing, and efficient memory layout transformations.

    {1 Creating Tensors}

    Use {!create}, {!zeros}, {!ones}, or {!arange} to construct tensors... *)
```

Remember: If the Unix manual wouldn't say it, neither should we.
