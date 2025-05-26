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

### Examples

#### Simple Function
```ocaml
val zeros : ('a, 'b) dtype -> int array -> ('a, 'b) t
(** [zeros dtype shape] creates zero-filled tensor.

    {[
      zeros float32 [|2;3|]
      (* [[0.;0.;0.];[0.;0.;0.]] *)
    ]} *)
```

#### Function with Constraints
```ocaml
val arange : ('a, 'b) dtype -> int -> int -> int -> ('a, 'b) t
(** [arange dtype start stop step] generates values from [start] to [stop).

    Step must be non-zero. Result length is [(stop - start) / step] rounded
    toward zero.

    @raise Failure if [step = 0]

    {[
      arange int32 0 10 2 = [|0;2;4;6;8|]
      arange int32 5 0 (-1) = [|5;4;3;2;1|]
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

    {[
      dot [|1.;2.|] [|3.;4.|] = 11.
      dot [[1.;2.];[3.;4.]] [[5.;6.];[7.;8.]] = [[19.;22.];[43.;50.]]
    ]} *)
```

#### Optional Parameters
```ocaml
val sum : ?axes:int array -> ?keepdims:bool -> ('a, 'b) t -> ('a, 'b) t
(** [sum ?axes ?keepdims t] sums elements along specified axes.

    Default sums all axes. If [keepdims] is true, retains reduced
    dimensions with size 1.

    @raise Invalid_argument if any axis is out of bounds

    {[
      sum [[1.;2.];[3.;4.]] = 10.
      sum ~axes:[|0|] [[1.;2.];[3.;4.]] = [|4.;6.|]
      sum ~axes:[|1|] ~keepdims:true [[1.;2.]] = [[3.]]
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

    {[
      stack ~axis:0 [[1;2];[3;4]] [[5;6];[7;8]]
      (* [[[1;2];[3;4]];[[5;6];[7;8]]] shape [|2;2;2|] *)
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

Remember: If the Unix manual wouldn't say it, neither should we.
