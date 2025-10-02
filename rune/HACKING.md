# Rune Developer Guide

## Architecture

Rune provides JAX-like automatic differentiation and JIT compilation for OCaml. It builds on Nx tensors with effect-based AD and multi-backend execution.

### Core Components

- **[lib/autodiff.ml](lib/autodiff.ml)**: Effect-based reverse-mode and forward-mode AD
- **[lib/tensor.ml](lib/tensor.ml)**: Tensor operations wrapping Nx with AD support
- **[lib/jit.ml](lib/jit.ml)**: JIT compilation via effect handlers
- **[lib/vmap.ml](lib/vmap.ml)**: Vectorization via effect handlers
- **[lib/rng.ml](lib/rng.mli)**: Random number generation
- **[lib-jit/](lib-jit/)**: LLVM-based JIT backend

### Key Design Principles

1. **Pure functional API**: All operations are pure; state managed via effects
2. **Effect handlers**: AD, JIT, and vmap implemented as algebraic effects
3. **Interop with Nx**: Seamless conversion between Rune and Nx tensors
4. **Backend agnostic**: Same code runs on CPU, CUDA, Metal

## Automatic Differentiation

### Reverse-Mode AD

Implemented via effect handlers that build a computation tape:

```ocaml
(* Compute gradient of f at x *)
let grad f x =
  (* Effect handler tracks operations, builds backward pass *)
  ...

(* Get both value and gradient *)
let value_and_grad f x = ...
```

**How it works:**

1. Forward pass: Execute function, recording operations in `t_with_grad` nodes
2. Backward pass: Traverse tape in reverse, accumulate gradients via chain rule
3. Return accumulated gradient for inputs

### Forward-Mode AD

For computing Jacobian-vector products:

```ocaml
(* Compute J*v where J is Jacobian of f *)
let jvp f ~primals ~tangents = ...
```

Uses dual numbers storing `(primal, tangent)` pairs.

### Derivative Rules

Each backend operation has a derivative defined in [autodiff.ml](lib/autodiff.ml):

```ocaml
(* Example: derivative of log2 *)
let deriv_log2 x =
  (* d/dx log2(x) = 1 / (x * ln(2)) *)
  T.div (T.ones_like x) (T.mul x (T.full ... ln2))
```

**Adding new operations:**
1. Define derivative in `autodiff.ml`
2. Hook into effect handler in reverse/forward mode
3. Test with `gradcheck` (finite differences validation)

## JIT Compilation

JIT compiles computation graphs to optimized code:

```ocaml
(* JIT-compile a function *)
let f_jit = jit (fun x -> sum (mul x x))

(* Executes compiled code on subsequent calls *)
let result = f_jit x
```

**Implementation:**
- Effect handler captures operations instead of executing
- Build computation graph
- Lower to LLVM IR (see [lib-jit/](lib-jit/))
- Compile and cache native code
- Execute compiled function

## Development Workflow

### Building and Testing

```bash
# Build rune
dune build rune/

# Run all tests
dune build rune/test/test_rune.exe && _build/default/rune/test/test_rune.exe

# Run AD tests
_build/default/rune/test/test_rune.exe test "Autodiff"

# Run JIT tests (if LLVM available)
_build/default/rune/test/jit/test_llvm.exe test "basic operations"
```

### Testing Conventions

**Gradient checking:**

```ocaml
(* Validate gradients against finite differences *)
let test_grad_add () =
  let f x = add x (full Float32 [|2; 3|] 2.0) in
  let x = rand Float32 [|2; 3|] in
  gradcheck f x  (* Raises if gradient incorrect *)
```

**AD tests:**

```ocaml
(* Test reverse-mode AD *)
let test_reverse_mode () =
  let f x = sum (mul x x) in
  let x = create Float32 [|3|] [|1.; 2.; 3.|] in
  let dx = grad f x in
  (* Gradient should be 2*x *)
  check_tensor dx [|2.; 4.; 6.|]
```

### Debugging AD

1. Use `value_and_grad` to inspect both forward and backward
2. Print intermediate gradients during backward pass
3. Validate with `gradcheck` (finite differences)
4. Check derivative definitions in [autodiff.ml](lib/autodiff.ml)

Common issues:
- Missing derivative for operation → add to effect handler
- Shape mismatch in backward pass → check broadcasting in derivative
- Wrong accumulation → verify chain rule application

## Effect System

Rune uses OCaml 5 effects for composable transformations.

### Effect Types

```ocaml
(* Reverse-mode AD effect *)
effect Grad : ('a, 'b) t -> ('a, 'b) t_with_grad

(* Forward-mode AD effect *)
effect Jvp : ('a, 'b) t -> ('a, 'b) dual

(* JIT compilation effect *)
effect Jit : ('a, 'b) operation -> ('a, 'b) t

(* Vectorization effect *)
effect Vmap : ...
```

### Effect Handlers

Each transformation installs an effect handler:

```ocaml
(* grad handler builds reverse-mode tape *)
let grad f x =
  match f x with
  | result -> backward_pass result
  | effect (Grad t) k ->
      (* Create t_with_grad node, continue *)
      ...
```

Handlers are **composable**: `grad (jit f)` or `jit (grad f)` both work (with different semantics).

## Adding Features

### New Tensor Operations

1. Add operation to [tensor.ml](lib/tensor.ml):
```ocaml
let new_op x =
  (* Call Nx backend *)
  of_nx (Nx.new_op (to_nx x))
```

2. Add derivative to [autodiff.ml](lib/autodiff.ml):
```ocaml
let deriv_new_op output input =
  (* Return gradient w.r.t. input *)
  ...
```

3. Hook into effect handlers (reverse and forward modes)

4. Test with gradcheck:
```ocaml
let test_grad_new_op () =
  let f x = new_op x in
  gradcheck f (rand Float32 [|10|])
```

### New Derivatives

Derivative of `f(x)` w.r.t. `x` given gradient of loss w.r.t. `f(x)`:

```ocaml
(* grad_output: gradient of loss w.r.t. f(x) *)
(* input: x *)
(* output: f(x) *)
let deriv_f output input grad_output =
  (* Return: grad_output * (df/dx) *)
  ...
```

Chain rule: `dL/dx = dL/df * df/dx`

### JIT Backend Operations

Add LLVM lowering in [lib-jit/](lib-jit/):

1. Define IR builder for operation
2. Handle in computation graph traversal
3. Test with small example

## Common Pitfalls

### Effect Handler Scope

Effects must be handled before returning to user code:

```ocaml
(* Wrong: effect escapes handler *)
let f x = grad (fun y -> y) x

(* Correct: handler installed by grad *)
let g x = grad (fun y -> sum (mul y y)) x
```

### Tensor Lifetimes

Don't mix Rune tensors from different effect scopes:

```ocaml
(* Wrong: x1 from different grad scope *)
let x1 = grad (fun x -> x) input
let x2 = grad (fun x -> add x x1) input  (* Error! *)

(* Correct: recreate or pass as closure *)
```

### Higher-Order Derivatives

Compose `grad`:

```ocaml
(* Second derivative *)
let grad2 f = grad (grad f)

(* Hessian-vector product *)
let hvp f x v = grad (fun x -> sum (mul (grad f x) v)) x
```

### Performance

- JIT compile hot paths: `let f_jit = jit f`
- Avoid recompilation: cache JIT'd functions
- Profile before optimizing: check if JIT overhead worth it

## Code Style

- **Pure functions**: All operations pure, no side effects
- **Effect types**: Explicit effect type signatures
- **Error messages**: `"function_name: error description"`
- **Tests**: Use `gradcheck` for all differentiable operations
- **Documentation**: Explain effect handler behavior

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- [nx/HACKING.md](../nx/HACKING.md): Nx internals
- JAX documentation for API inspiration
