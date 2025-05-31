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
val function_name : type_signature
(** [function_name arg1 arg2] does X.

    Extended description if needed. State invariants and constraints naturally
    in prose. Mention performance characteristics only if surprising.

    @raise Exception_name if [condition]

    {[
      let result = function_name value1 value2
      (* result = expected_output *)
    ]} *)
```

### Formatting Conventions

#### Code References
- Use `[code]` for inline code: parameter names, function names, expressions
- Use `{[ ... ]}` for code blocks
- No backticks - this is odoc, not Markdown

#### First Line
Always start with the function application pattern:
```ocaml
(** [create dtype shape data] creates a tensor from array [data]. *)
```

Not:
```ocaml
(** Creates a tensor with the given dtype and shape. *)
```

#### Mathematical Notation
- Use ASCII: `a * b`, not `a × b`
- Use `x^2` or `x ** 2` for powers
- Use `[start, stop)` for half-open intervals

### Content Guidelines

#### What to Include

✓ **Invariants and preconditions**
```ocaml
(** Length of [data] must equal product of [shape]. *)
```

✓ **Non-obvious performance characteristics**
```ocaml
(** Returns view if possible (O(1)), otherwise copies (O(n)). *)
```

✓ **Shape transformations**
```ocaml
(** Result has shape [|m; n|] where m = length of [a], n = length of [b]. *)
```

#### What to Exclude

✗ **Redundant parameter descriptions**
```ocaml
(* Bad: *)
(** @param dtype the data type of the tensor *)
(* Good: include constraints in prose if needed *)
```

✗ **Obvious information**
```ocaml
(* Bad: *)
(** Allocates a new tensor. *)  (* Obviously true for most operations *)
```

✗ **Implementation details**
```ocaml
(* Bad: *)
(** Internally uses Gaussian elimination with partial pivoting. *)
(* Good: *)
(** Computes matrix inverse. *)
```

### Code Examples

All code examples in documentation must be valid OCaml that can be compiled and executed. This allows for future integration with mdx testing.

#### Requirements for Code Examples

1. **Complete and valid syntax** - Examples must parse and typecheck
2. **Use qualified names** - Use `Nx.function` instead of `open Nx`
3. **Show actual values** - Use `=` to show expected results
4. **Proper type annotations** - Include when necessary for clarity
5. **Separate code blocks** - Each example should be in its own `{[ ... ]}` block with a description before it
6. **Self-contained examples** - Each block should be independently executable

### Examples

#### Simple Function
```ocaml
val zeros : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [zeros dtype shape] creates zero-filled tensor.

    Example creating a 2x3 matrix of zeros:
    {[
      let t = Nx.zeros Nx.float32 [|2; 3|] in
      Nx.to_array t = [|0.; 0.; 0.; 0.; 0.; 0.|]
    ]} *)
```

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
    ]}

    Counting down from 5 to 1:
    {[
      let t2 = Nx.arange Nx.int32 5 0 (-1) in
      Nx.to_array t2 = [|5l; 4l; 3l; 2l; 1l|]
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
    ]}

    Matrix multiplication of 2x2 matrices:
    {[
      let m1 = Nx.of_array Nx.float32 ~shape:[|2; 2|] [|1.; 2.; 3.; 4.|] in
      let m2 = Nx.of_array Nx.float32 ~shape:[|2; 2|] [|5.; 6.; 7.; 8.|] in
      let result = Nx.dot m1 m2 in
      Nx.to_array result = [|19.; 22.; 43.; 50.|]
    ]} *)
```

#### Optional Parameters
```ocaml
val sum : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [sum ?axes ?keepdims t] sums elements along specified axes.

    Default sums all axes. If [keepdims] is true, retains reduced
    dimensions with size 1.

    @raise Invalid_argument if any axis is out of bounds

    Summing all elements in a 2x2 matrix:
    {[
      let t = Nx.of_array Nx.float32 ~shape:[|2; 2|] [|1.; 2.; 3.; 4.|] in
      Nx.to_scalar (Nx.sum t) = 10.
    ]}

    Summing along rows (axis 0):
    {[
      let t = Nx.of_array Nx.float32 ~shape:[|2; 2|] [|1.; 2.; 3.; 4.|] in
      let sum_axis0 = Nx.sum ~axes:[|0|] t in
      Nx.to_array sum_axis0 = [|4.; 6.|]
    ]}

    Summing along columns (axis 1) while keeping dimensions:
    {[
      let t = Nx.of_array Nx.float32 ~shape:[|2; 2|] [|1.; 2.; 3.; 4.|] in
      let sum_keepdims = Nx.sum ~axes:[|1|] ~keepdims:true t in
      Nx.shape sum_keepdims = [|2; 1|] &&
      Nx.to_array sum_keepdims = [|3.; 7.|]
    ]} *)
```

### Special Cases

#### Broadcasting Functions
Always explain broadcasting behavior:
```ocaml
(** [add t1 t2] computes element-wise sum with broadcasting.

    Shapes must be broadcast-compatible: each dimension must be equal
    or one of them must be 1. *)
```

#### View vs Copy Operations
Be explicit about memory behavior:
```ocaml
(** [transpose t] returns view with swapped axes (no copy). *)
(** [flatten t] returns new 1-D tensor (always copies). *)
(** [reshape shape t] returns view if possible, otherwise copies. *)
```

#### Complex Shape Rules
Use examples to clarify:
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

### Checklist

Before committing documentation:

- [ ] First line uses `[function_name args]` pattern
- [ ] Uses active voice ("creates" not "is created")
- [ ] Documents invariants, not implementation
- [ ] Includes `@raise` for all exceptions
- [ ] Examples show actual usage with output
- [ ] Examples are valid OCaml code with proper module opens
- [ ] Examples use `=` to show expected results
- [ ] No redundant parameter lists
- [ ] Performance noted only if surprising
- [ ] Code properly formatted with `[...]` and `{[...]}`

### Anti-patterns to Avoid

```ocaml
(* Too verbose: *)
(** [add t1 t2] performs element-wise addition.

    This function takes two tensors t1 and t2 as input and returns
    a new tensor containing their element-wise sum. The tensors must
    have compatible shapes for broadcasting.

    @param t1 first tensor
    @param t2 second tensor
    @return new tensor with the sum *)

(* Better: *)
(** [add t1 t2] computes element-wise sum with broadcasting.

    @raise Invalid_argument if shapes incompatible *)
```

```ocaml
(* Invalid OCaml in examples: *)
(** {[
      zeros float32 [|2;3|]  (* Missing module qualification and syntax error *)
      (* [[0.;0.;0.];[0.;0.;0.]] *)
    ]} *)

(* Valid OCaml: *)
(** {[
      let t = Nx.zeros Nx.float32 [|2; 3|] in
      Nx.to_array t = [|0.; 0.; 0.; 0.; 0.; 0.|]
    ]} *)
```

Remember: If the Unix manual wouldn't say it, neither should we.
