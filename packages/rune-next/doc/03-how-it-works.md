# How It Works

This page explains how rune-next implements its transformations with OCaml 5 effect handlers. Understanding the mechanism is not required for using the library, but it helps when debugging unexpected behavior or reasoning about performance.

## The Core Idea

Every Nx tensor operation raises an OCaml 5 effect before performing the actual computation. Normally no handler is installed, so the effect falls through to the default backend, which executes the operation directly.

Rune-next's transformations are effect handlers. Each one intercepts the operations a function performs and uses them differently:

- **Reverse mode** records pull thunks on a tape during the forward pass, then runs them backward.
- **Forward mode** propagates tangents alongside primal values in a single pass, with no tape.
- **vmap** presents batched tensors to the function as if they were unbatched, translating each primitive to its batched form.
- **with_debug** logs each operation and its output shape.

```
User code: Nx.add x y
     │
     ├─ no handler installed → backend executes directly
     │
     └─ handler installed (grad, jvp, vmap, ...) → handler intercepts,
        applies its treatment, re-performs the op in the enclosing context
```

The key property: **user code does not change**. You write functions with `Nx.add`, `Nx.matmul`, `Nx.sin`, and rune-next transforms them by handling their effects. There is no special tensor type, no graph builder, and no tracing step — and because handlers re-perform operations in the *enclosing* context, nesting one transformation inside another just works.

## Reverse Mode: the Tape

`grad (module P) f params` proceeds in two passes.

**Forward pass.** The handler marks the tensor leaves of `params` as *tracked* and runs `f params`. Every intercepted operation computes its primal result by re-performing the operation in the enclosing context; if any input is tracked, the output is marked tracked too and a *pull thunk* is recorded on the tape. A pull thunk knows how to map the operation's output cotangent to contributions on its inputs — the standard VJP rules:

- `add`: the cotangent flows to both inputs unchanged (reduced over broadcast axes);
- `mul`: the cotangent of `a * b` with respect to `a` is `cotangent * b`;
- `sin`: the cotangent is `cotangent * cos x`.

Operations whose inputs are all untracked are constants with respect to the differentiated inputs and are recorded nowhere — which is why closures over data cost nothing.

**Backward pass.** The output cotangent is seeded (with `1` for `grad`, or your explicit cotangent for `vjp`) and the pull thunks run in reverse order, accumulating cotangents keyed by tensor identity. Finally the accumulated cotangents of the parameter leaves are read back through the structure's `map`, producing a gradient with the parameters' own type.

Tensors are keyed by physical identity: every Nx operation allocates a fresh tensor, so a tensor value identifies a node of the computation graph. This is also why in-place mutation (`set_slice`, `assign`, ...) raises during differentiation — mutating a tracked tensor would corrupt the correspondence.

### Higher-order derivatives

Pull thunks execute ordinary Nx operations, so an enclosing transformation intercepts *them* too: an outer `grad` differentiates the backward pass of an inner `grad`, an outer `vmap` batches a pullback. Higher-order derivatives and compositions like `hvp` (forward over reverse) fall out of this with no dedicated machinery.

## Forward Mode: No Tape

`jvp` is simpler. Tangents propagate eagerly: the handler keeps a store mapping tensors to their tangents, seeds the parameter leaves with your tangents, and at each intercepted operation computes the output tangent immediately from the input tangents — `d(a * b) = da * b + a * db` — alongside the primal. There is no second pass. A tensor absent from the store is a constant with zero tangent.

Tangent arithmetic is re-performed in the enclosing context too, so forward-over-reverse (`hvp`), reverse-over-forward, and nested `jvp` all compose.

## vmap: a Virtual View

The mapped function is written for unbatched values. Under the `vmap` handler, every tensor is either *batched* — it physically carries the batch dimension, canonically at axis 0 — or a constant of the map. Two mechanisms keep this transparent:

- **Shape queries are intercepted.** For batched tensors the handler answers with the unbatched remainder of the shape, so the function (and the Nx frontend itself) makes exactly the decisions of the unbatched program — broadcasting, promotion, reshapes.
- **Each primitive is translated to its batched form.** Shape parameters gain a leading batch entry, axis parameters shift by one, and constants meeting batched operands are lifted with a broadcast view.

Operations whose operands are all constants fall through unintercepted, and a result that does not depend on the mapped inputs is broadcast along the batch axis. Nested `vmap`s stack: each handler owns its batched set and batch size, and the translations one level emits are re-translated by the level above.

Two consequences documented in [Transformations](02-transformations/) follow directly from this design. Reading a batched tensor's *value* inside the mapped function raises — there is one physical tensor for all lanes, not one value per lane — which is why a `cond` predicate cannot depend on mapped inputs. And implicit RNG draws identical values in every lane, because the RNG key is a constant of the map.

## Custom Rules and remat

`custom_vjp` and `custom_jvp` communicate with the ambient handlers through their own effects. Dispatch is by handler stacking: the innermost transformation that understands the effect applies the rule; enclosing transformations see the forward computation itself. A differentiation of the wrong mode raises — a custom VJP is not forward-differentiable, and vice versa. When no transformation is in scope, the plain forward function runs at the call site.

`remat` is a small application of this machinery: it wraps a function in a `custom_vjp` whose backward rule re-runs the function under a fresh tape and pulls the cotangent back through it. Nothing from the wrapped function's first execution is retained; the recomputation happens exactly when the backward pass needs it.

## detach and no_grad

The differentiation handlers share a tracing gate. `no_grad f` turns interception off for the extent of `f`, in both reverse and forward mode; `detach t` copies `t` with the gate closed, so the copy enters subsequent computations as an untracked constant. Neither erases anything from an existing tape — they prevent recording in the first place.

## The Failure Model

Every Nx effect constructor is matched explicitly by each engine. Operations without a rule fall into three deliberate categories:

- **Zero derivative** (comparisons, bitwise and integer ops, rounding, `argmax`/`argsort`, RNG, tensor creation): these fall through untracked, which yields the correct zero gradient.
- **No rule implemented** (`svd`, `eig`, `eigh`, `rfft`, `irfft`, `psum`, `mod` in reverse mode; the `fft` family and decompositions under `vmap`): these raise when an input is tracked, instead of silently producing a zero gradient. `detach` the input if differentiation should not flow through it.
- **In-place mutation** (`assign`, `set_slice`, `blit`): always raises during differentiation.

The intent is that rune-next never returns a wrong gradient quietly.

## Implications for Users

**No graph construction step.** Everything runs eagerly. Every operation happens immediately, and transformations intercept operations as they execute. `if`, `match`, `for`, recursion, and higher-order functions all work inside differentiated code — there is no "graph-compatible" subset of the language. (The `scan`/`cond`/`while_loop` combinators exist for a future `jit` that could stage them; they are not required today.)

**Side effects run in the forward pass.** Printing or logging inside a differentiated function executes during the forward pass. The backward pass runs the recorded pull thunks; it does not re-execute your function — unless you asked for that with `remat`.

**Per-operation overhead.** Handling an effect costs more than a raw Nx call. For workloads dominated by large operations (matrix multiplications), the overhead is negligible; for many tiny operations it is more visible.
