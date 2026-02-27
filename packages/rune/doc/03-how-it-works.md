# How It Works

This page explains how Rune implements automatic differentiation using OCaml 5 effect handlers. Understanding the mechanism is not required for using Rune, but it helps when debugging unexpected behavior or reasoning about performance.

## The Core Idea

When you call `Nx.add x y`, the operation raises an OCaml 5 effect before performing the actual computation. Normally, no handler is installed, so the effect is unhandled and falls through to the default C backend, which executes the operation directly.

Rune's transformations work by installing effect handlers that intercept these operations. Each transformation uses the intercepted operations differently:

- **Reverse-mode AD** records operations on a tape during the forward pass, then propagates gradients backward.
- **Forward-mode AD** propagates tangent vectors alongside primal values during a single forward pass.
- **vmap** unbatches inputs, runs the function on slices, and rebatches outputs.
- **debug** prints each operation and its arguments.

## Effect-Based Architecture

Every Nx tensor operation raises an effect. For example, `Nx.add` raises an `E_add` effect, `Nx.mul` raises `E_mul`, and so on. Each effect carries the input tensors and an output buffer.

```
User code: Nx.add x y
     │
     ├─ No handler installed → C backend executes directly
     │
     └─ Handler installed (e.g., by Rune.grad) → handler intercepts,
        records the operation, then continues execution
```

This design has a key property: **user code does not change**. You write functions using `Nx.add`, `Nx.mul`, `Nx.sin`, etc. and Rune transforms them by handling their effects differently. There is no special tensor type, no computation graph builder, and no tracing step.

## Reverse-Mode AD (grad)

When you call `Rune.grad f x`, Rune:

1. **Installs an effect handler** that intercepts every Nx operation.
2. **Runs `f x` under that handler**. As each operation executes, the handler records it on a tape (a list of operations with their inputs and outputs).
3. **Seeds the output** with a cotangent of 1.0 (since `f` returns a scalar).
4. **Walks the tape backward**, computing the gradient contribution of each operation using the chain rule.

The backward rules are the standard VJP (vector-Jacobian product) rules. For example:
- `add`: gradients flow through to both inputs unchanged
- `mul`: gradient of `a * b` w.r.t. `a` is `grad_out * b`
- `sin`: gradient is `grad_out * cos(x)`

Because the tape is walked as the continuation stack unwinds, this happens automatically — there is no separate "backward pass" function to call.

### Higher-order derivatives

Since `grad f` returns a regular OCaml function, calling `grad (grad f)` works naturally: the outer `grad` installs a handler, and when the inner `grad` runs its forward-backward pass, those operations are themselves intercepted and recorded by the outer handler.

## Forward-Mode AD (jvp)

Forward-mode AD is simpler than reverse-mode. When you call `Rune.jvp f x v`:

1. **Installs an effect handler** that maintains a tangent value alongside each tensor.
2. **Seeds the input** `x` with tangent `v`.
3. **Runs `f x`**. At each operation, the handler computes both the primal result and the tangent using the JVP rule for that operation.

For example, for `z = x * y`:
- Primal: `z = x * y`
- Tangent: `dz = dx * y + x * dy`

The result is `(f x, J_f(x) · v)` — the function value and the directional derivative in direction `v`.

## vmap

When you call `Rune.vmap f x`:

1. **Determines the batch size** from the mapped axis of `x`.
2. **Slices the input** along the batch axis.
3. **Runs `f` on each slice**, intercepting effects to track which operations happen.
4. **Stacks the outputs** along the specified output axis.

The handler ensures that operations inside `f` see unbatched tensors, while the overall result is properly batched.

## Composability

Because each transformation is just an effect handler, they compose naturally:

- `grad (grad f)` — nested handlers for higher-order derivatives
- `vmap (grad f)` — per-example gradients
- `debug (grad f)` — trace the backward pass

The OCaml effect system handles the nesting: each handler only intercepts unhandled effects, and re-raises operations it doesn't care about to the next handler in the stack.

## Implications for Users

**No graph construction step.** Unlike frameworks that build a computation graph and then execute it, Rune runs eagerly. Every operation happens immediately, and transformations work by intercepting these operations as they execute.

**Control flow works naturally.** Because Rune transforms ordinary OCaml functions, `if`, `for`, `while`, `match`, recursion, and higher-order functions all work as expected. There is no restriction to a "graph-compatible" subset of the language.

**Side effects in differentiated functions.** Printing, logging, and other side effects inside a function passed to `grad` will execute during the forward pass. The backward gradient propagation does not re-execute the function — it uses the recorded tape.

**Performance.** The effect handler adds overhead per-operation compared to raw Nx calls. For typical ML workloads where operations are large (e.g., matrix multiplications), this overhead is negligible. For workloads with many small operations, the overhead may be more noticeable.
