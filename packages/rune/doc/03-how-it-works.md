# How It Works

This page explains how rune implements its transformations with OCaml 5 effect handlers. Understanding the mechanism is not required for using the library, but it helps when debugging unexpected behavior or reasoning about performance.

## The Core Idea

Every Nx tensor operation is a value of one type, `Nx.Op.t` — `Binary (Add, x, y)`, `Reduce (Sum, axes, x)`, and so on. Rune's transformations are interpreters of those operations: functions from an operation to its result, each installed with `Nx.Op.intercept` for the extent of the function it transforms. An operation performed inside that extent is delivered to the interpreter as one OCaml 5 effect. While no interpreter is installed anywhere, Nx performs no effect and computes each operation directly.

Each interpreter matches every kind of operation, and the compiler checks that none is missing. They use them differently:

- **Reverse mode** records pull thunks on a tape during the forward pass, then runs them backward.
- **Forward mode** propagates tangents alongside primal values in a single pass, with no tape.
- **vmap** presents batched tensors to the function as if they were unbatched, translating each primitive to its batched form.

```
User code: Nx.add x y
     │
     ├─ no interpreter installed → Nx computes directly
     │
     └─ interpreter installed (grad, jvp, vmap, ...) → it receives the op,
        applies its treatment, evaluates the op in the enclosing context
```

The key property: **user code does not change**. You write functions with `Nx.add`, `Nx.matmul`, `Nx.sin`, and rune transforms them by interpreting their operations. There is no special tensor type, no graph builder, and no tracing step — and because interpreters evaluate operations in the *enclosing* context, nesting one transformation inside another just works.

## Reverse Mode: the Tape

`grad p f params` proceeds in two passes.

**Forward pass.** The handler replaces each tensor of `params` by a fresh alias, marks the aliases as *tracked* and runs `f` on them. The alias makes a tensor that `f` captures a constant even when it is also a tensor of `params`. Every intercepted operation computes its primal result by evaluating the operation in the enclosing context; if any input is tracked, the output is marked tracked too and a *pull thunk* is recorded on the tape. A pull thunk knows how to map the operation's output cotangent to contributions on its inputs — the standard VJP rules:

- `add`: the cotangent flows to both inputs unchanged (reduced over broadcast axes);
- `mul`: the cotangent of `a * b` with respect to `a` is `cotangent * b`;
- `sin`: the cotangent is `cotangent * cos x`.

Operations whose inputs are all untracked are constants with respect to the differentiated inputs and are recorded nowhere — which is why closures over data cost nothing.

**Backward pass.** The output cotangent is seeded (with `1` for `grad`, or your explicit cotangent for `vjp`) and the pull thunks run in reverse order, accumulating cotangents keyed by tensor identity. Finally the accumulated cotangents of the parameters are read back through the structure's walk, producing a gradient with the parameters' own type.

Tensors are keyed by physical identity: every Nx operation allocates a fresh tensor, so a tensor value identifies a node of the computation graph. Nx tensors are values (nothing writes into one after it exists), which is what keeps that correspondence sound.

### Higher-order derivatives

Pull thunks execute ordinary Nx operations, so an enclosing transformation intercepts *them* too: an outer `grad` differentiates the backward pass of an inner `grad`, an outer `vmap` batches a pullback. Higher-order derivatives and compositions like a Hessian-vector product (forward over reverse) fall out of this with no dedicated machinery.

## Forward Mode: No Tape

`jvp` is simpler. Tangents propagate eagerly: the handler keeps a store mapping tensors to their tangents, seeds the parameter leaves with your tangents, and at each intercepted operation computes the output tangent immediately from the input tangents — `d(a * b) = da * b + a * db` — alongside the primal. There is no second pass. A tensor absent from the store is a constant with zero tangent.

Tangent arithmetic runs in the enclosing context too, so forward-over-reverse, reverse-over-forward, and nested `jvp` all compose.

## Complex Tensors

A complex tensor is two real numbers per element. A function on complex tensors is therefore a function on twice as many real numbers, and its derivative at a point is a real linear map on those components. Rune packs a pair of components `(re, im)` as the complex number `re + i·im` and measures these vectors with the real inner product

```
<u, v> = Re(sum(conj(u) * v))
```

which is the ordinary dot product of the two real vectors.

A **tangent** is a displacement in this packing: moving `z` by `dre` in the real component and `dim` in the imaginary one is the tangent `dre + i·dim`. `jvp` takes and returns tangents, and what it returns is the directional derivative: displace the input by `h` times the tangent, and both components of the output move by `h` times the result.

A **gradient** is the vector of partial derivatives in the same packing. For a real-valued objective `L`, the gradient with respect to `z` is

```
dL/dre + i · dL/dim
```

It is the direction in which `L` grows fastest, and `<g, v>` is the change in `L` along a tangent `v`, so `z - lr * g` descends. `vjp` is the adjoint of `jvp` under the inner product: the cotangent `w` of an output pulls back to the gradient of the real objective `<w, f z>`. For a complex-differentiable `f` with derivative `f'`, the pullback is `conj(f' z) * w`.

### Inside the tape

The tape holds the conjugates of gradients, `dL/dre - i · dL/dim`, because they pair with a tangent by plain multiplication:

```
Re((dL/dre - i·dL/dim) * (dre + i·dim)) = dL/dre · dre + dL/dim · dim
```

Under that pairing the transpose of "multiply by the complex number `c`" is again "multiply by `c`". So on the tape a rule that multiplies the cotangent by a derivative and conjugates nothing is correct for every operation that has a complex derivative. `mul` pulls back as `cotangent * b`, `exp` as `cotangent * exp z`, `matmul` through an ordinary transpose, `fft` through `fft` itself: every real derivative formula carries over to complex unchanged. `grad`, `vjp` and `vjp_fun` conjugate the cotangents you give on the way in and the gradients they return on the way out, and a `custom_vjp` rule's `bwd` is called across the same boundary, so it sees gradients too. On real dtypes the conjugation is the identity.

Three kinds of rule need a conjugation of their own on the tape:

- **Real-valued operations.** `abs z` is the modulus, and its differential mixes the two components instead of scaling by one complex number. It pulls back through `conj(sign z)` and keeps only the real part of the cotangent, because a real-valued output cannot move in the imaginary direction. In forward mode it produces a real tangent.
- **Operations that turn with `z`.** `sign z = z / |z|` moves only along the unit circle: its derivative along `v` is `i · s · Im(conj(s) · v) / |z|`, with `s = sign z`.
- **Factorisations whose complex form conjugates.** A Hermitian Cholesky factor satisfies `H = L Lᴴ`, a unitary `Q` satisfies `Qᴴ Q = I`, and a transposed triangular solve reads `Aᴴ`. Their rules work on the conjugate cotangent, where a product transposes to its conjugate transpose, and conjugate the result back.

Rules are checked against a finite-difference oracle in `packages/rune/test/test_complex.ml`. It perturbs each component of each input separately, assembles the real Jacobian, and compares both engines against it, so it measures the operation rather than trusting another rule. A new rule reachable on a complex dtype belongs there.

## vmap: Lanes

The mapped function is written for unbatched values. Under `vmap`, every tensor is either a *lane* or a constant of the map. A lane is a traced value that holds its batched tensor, which physically carries the batch dimension at axis 0. Two properties keep this transparent:

- **A lane has the unbatched shape.** Its shape is the batched tensor's without the batch axis, and its placement the batched tensor's without that axis, so the function (and the Nx frontend itself) makes exactly the decisions of the unbatched program — broadcasting, promotion, reshapes. Reading a shape or a placement reads a field: nothing is intercepted.
- **Each operation on a lane is translated to its batched form.** Shape parameters gain a leading batch entry, axis parameters shift by one, and constants meeting lanes are lifted with a broadcast view.

Operations whose operands are all constants are evaluated as they are, and a result that does not depend on the mapped inputs is broadcast along the batch axis. Nested `vmap`s stack: each map owns its lanes and batch size, and the translations one level emits, over the enclosing map's lanes, are translated again by the level above. A lane of a map over an axis split across devices is a copy on each of them.

Two consequences documented in [Transformations](02-transformations.md) follow directly from this design. Reading a lane's *value* inside the mapped function raises — there is one physical tensor for all lanes, not one value per lane — which is why a branch's predicate cannot depend on mapped inputs. And implicit RNG draws identical values in every lane, because the RNG key is a constant of the map.

## Custom Rules and remat

`custom_vjp` and `custom_jvp` communicate with the ambient handlers through their own effects. Dispatch is by handler stacking: the innermost transformation that understands the effect applies the rule; enclosing transformations see the forward computation itself. A differentiation of the wrong mode runs the call's function when its result holds no tensor, and raises otherwise — a custom VJP with a tensor result is not forward-differentiable, and vice versa. `vmap` passes a custom call on as the call of its batched functions, so the differentiation outside the map applies the rule to the map's lanes. When no transformation is in scope, the plain forward function runs at the call site.

`remat` has an effect of its own, and every handler passes it on to the enclosing context with the function wrapped in itself, so that no transformation misses the tensors the function captures. Reverse mode learns, from the wrapped run, whether the result depends on a tracked tensor; if it does, it keeps the function's arguments and records a pull thunk that runs the function again under a fresh tape linked to its own and pulls the cotangent back through it. A tracked tensor the function captures becomes a leaf of the linked tape, and its cotangent goes back to the enclosing tape. Forward mode passes on the remat of the function's jvp, whose results include the tangents; `vmap` passes on the function batched. Nothing from the wrapped function's first execution is retained; the recomputation happens exactly when the backward pass needs it.

Under `jit`, a recomputation traced from the same arguments would be the same graph nodes as the forward pass, so the compiled program would keep the forward's intermediates until the backward pass read them. The recomputation therefore reads its arguments through a barrier: the arguments are materialised in the forward pass, and the backward pass reads their storage after the cotangents of the function's result, materialised too. The recomputation is then a computation of its own that runs only once those cotangents exist.

## detach

`detach t` copies `t` with the reverse- and forward-mode interpreters around it paused: they pass the copy on as it is, so it enters subsequent computations as an untracked constant. It erases nothing from an existing tape; it prevents recording in the first place.

## The Failure Model

Every Nx operation is matched explicitly by each engine, and the compiler refuses an engine that misses one. Operations without a rule fall into two deliberate categories:

- **Zero derivative** (comparisons, bitwise and integer ops, rounding, `argmax`/`argsort`, RNG, reads): these are evaluated untracked, which yields the correct zero gradient.
- **No rule implemented** (`svd`, `eig`, `eigh`, `Rune.lanes`, `mod` in reverse mode; the decompositions under `vmap`): these raise when an input is tracked, instead of silently producing a zero gradient. `detach` the input if differentiation should not flow through it.

The intent is that rune never returns a wrong gradient quietly.

## Implications for Users

**No graph construction step.** Everything runs eagerly. Every operation happens immediately, and transformations intercept operations as they execute. `if`, `match`, `for`, recursion, and higher-order functions all work inside differentiated code — there is no "graph-compatible" subset of the language. (The `scan` combinator gives a loop a structure `jit` can see: it compiles `scan` as a loop instead of unrolling it. It is never required.)

**Side effects run in the forward pass.** Printing or logging inside a differentiated function executes during the forward pass. The backward pass runs the recorded pull thunks; it does not re-execute your function — unless you asked for that with `remat`.

**Per-operation overhead.** Interpreting an operation costs more than a raw Nx call, and while any interpreter is installed, on any domain, every Nx operation performs an effect to find one. For workloads dominated by large operations (matrix multiplications), the overhead is negligible; for many tiny operations it is more visible.
