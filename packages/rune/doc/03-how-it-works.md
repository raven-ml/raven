# How It Works

This page explains how rune implements its transformations. Understanding the mechanism is not required for using the library, but it helps when debugging unexpected behavior or reasoning about performance.

## The Core Idea

Every Nx tensor operation is a value of one type, `Nx.Op.t`: `Binary (Add, x, y)`, `Reduce (Sum, axes, x)`, and so on. Rune's transformations are interpreters of those operations, functions from an operation to its result, each installed with `Nx.Op.intercept` for the extent of the function it transforms. While no interpreter is installed anywhere, Nx computes each operation directly.

There are four interpreters, and each matches every kind of operation, which the compiler checks:

- **Forward mode** (`jvp`) carries a tangent beside each value it differentiates. It holds the one derivative rule of every operation.
- **The recorder** (reverse mode) runs forward mode with symbolic tangents, records the linear operations those rules emit, and replays their transposes.
- **vmap** presents batched tensors to the function as if they were unbatched, translating each operation to its batched form.
- **The compiled call** (`jit`) turns each operation into a node of a program (see [Compilation](05-compilation.md)).

```
User code: Nx.add x y
     │
     ├─ no interpreter installed → Nx computes directly
     │
     └─ interpreter installed (grad, jvp, vmap, jit) → it receives the op,
        applies its rule, and evaluates what the rule issues in the
        enclosing context
```

User code does not change: you write functions with `Nx.add`, `Nx.matmul` and `Nx.sin`, and rune transforms them by interpreting their operations. There is no special tensor type and no graph builder.

## Values Name Their Transformation

An application of a transformation, an *installation*, makes traced values whose payload names it: forward mode makes *duals* (a primal and a tangent), the recorder makes *slots*, and `vmap` makes *lanes*. An interpreter applies its rule to an operation when one of the operands is its own value, and passes the operation outward otherwise. That ownership test is the whole of nesting: no levels are compared and nothing is looked up by identity. In

```ocaml
let () =
  let one = Nx.scalar Nx.float32 1.0 in
  let v =
    Rune.grad' (fun x -> Nx.mul x (Rune.grad' (fun y -> Nx.add x y) one)) one
  in
  Printf.printf "%g\n" (Nx.item [] v) (* 1 *)
```

the inner `grad'` owns `y` only. It passes `x + 1` on to the outer one, which differentiates it, and the inner gradient is a constant to the outer, so the result is `1`.

A value with a zero tangent is no dual: absence is the symbolic zero, so constants cost nothing. A dual, slot or lane that outlives its installation (kept in a reference, used on another domain, held by a closure that runs later) has no owner, and evaluating it raises. That is the "values stay inside" rule of [Transformations](02-transformations.md).

## Forward Mode

`jvp` makes each float or complex leaf of the parameters a dual of a fresh installation and runs `f`. A rule unwraps its own duals, evaluates the operation on the primals in the enclosing context, and computes the tangent: `d(a * b) = da * b + a * db`. It builds no term for an operand without a tangent, so a coefficient that is infinite where its operand is constant (the `log a` of `a ** b` at `a = 0`) never meets a zero. There is no second pass, and a tangent dies with its dual, so a fold holds the tangents of one step at a time.

## Reverse Mode: Linearize, Then Transpose

`grad` installs a recorder and, inside it, forward mode whose tangents are the recorder's slots. Each forward rule runs outside its interpreter, so the tangent computations it issues reach the recorder:

- an operation with a slot among its operands is recorded, and its result is a fresh slot;
- an operation with no slot (a primal, a coefficient such as `cos x`) is computed.

Every recorded operation must be linear in its slots, which the recorder checks against its transpose table. The pullback seeds the result's slots with the cotangents, walks the record backwards applying each operation's transpose, and returns the parameters' cotangents in their structure. A recorded entry is an operation or a *linear call*, a map rune states whole with its own pullback: a `custom_vjp` rule, `lanes`, a remat, or a staged scan.

Each derivative is stated once, as a forward rule, and reverse mode is its transpose. The pullback's arithmetic runs in the caller's interpretation, so an outer `jvp` differentiates it and an outer `vmap` batches it: higher-order derivatives and compositions such as a Hessian-vector product need no dedicated machinery. The pullback runs no part of `f` again, except the functions `f` passes to `remat`. When `vjp` runs outside every transformation it may be applied any number of times, from any domain; under another transformation it is transformed with it.

## Complex Tensors

A complex tensor is two real numbers per element. A function on complex tensors is therefore a function on twice as many real numbers, and its derivative at a point is a real linear map on those components. Rune packs a pair of components `(re, im)` as the complex number `re + i·im` and measures these vectors with the real inner product

```
<u, v> = Re(sum(conj(u) * v))
```

which is the ordinary dot product of the two real vectors.

A **tangent** is a displacement in this packing: moving `z` by `dre` in the real component and `dim` in the imaginary one is the tangent `dre + i·dim`. `jvp` takes and returns tangents, and what it returns is the directional derivative.

A **gradient** is the vector of partial derivatives in the same packing. For a real-valued objective `L`, the gradient with respect to `z` is `dL/dre + i · dL/dim`: the direction in which `L` grows fastest, so `z - lr * g` descends. `vjp`'s pullback is the adjoint of `jvp` under the inner product: the cotangent `w` of an output pulls back to the gradient of the real objective `<w, f z>`. For a complex-differentiable `f` with derivative `f'`, the pullback is `conj(f' z) * w`.

The recorder transposes under the bilinear pairing `Re(sum(c * v))`, conjugating the cotangents on the way in and the gradients on the way out. Under that pairing the transpose of "multiply by `c`" is again "multiply by `c`", so no transpose conjugates, except the triangular solve's, whose `transpose` flag is the conjugate transpose. Operations that are only real-linear, such as `abs z` and `sign z` of a complex `z`, state their forward rule in real-linear form. Each rule is checked against a central difference on complex operands, and each transpose by the adjoint identity, in the rules' suite.

## vmap: Lanes

The mapped function is written for unbatched values. Under `vmap`, every tensor is either a *lane* or a constant of the map. A lane is a traced value that holds its batched tensor, which carries the batch dimension at axis 0. Two properties keep this transparent:

- **A lane has the unbatched shape.** Its shape is the batched tensor's without the batch axis, and its placement the batched tensor's without that axis, so the function (and the Nx frontend itself) makes exactly the decisions of the unbatched program: broadcasting, promotion, reshapes.
- **Each operation on a lane is translated to its batched form.** Shape parameters gain a leading batch entry, axis parameters shift by one, a factorisation takes the batch as a leading axis, and constants meeting lanes are lifted with a broadcast view.

A result that does not depend on the mapped inputs is broadcast along the batch axis. Nested maps stack: each map owns its lanes, and the operations one map emits over the enclosing map's lanes are translated again by the enclosing one.

Two consequences documented in [Transformations](02-transformations.md) follow from this design. Reading a lane's value inside the mapped function raises, since there is one tensor for all lanes, which is why a branch's predicate cannot depend on mapped inputs. And implicit RNG draws identical values in every lane, because the RNG key is a constant of the map.

## Rune's Own Constructs

Besides Nx's operations, rune has a few constructs of its own: `scan`, `remat`, custom rules, `lanes`, `lane_index`, totals' additions and `detach`. Each is performed as an effect, and every installation answers each construct explicitly, or passes it outward. A construct no installation takes has a default computed where it is performed.

- **`scan`** is offered to the installations as a request to stage the loop. Each installation between the scan and a compiled call passes the scan on, transformed: forward mode adds the tangents to the carry, `vmap` the lanes. Only a compiled call stages it. A scan nothing stages is declined back to `Rune.scan`, which runs the loop where it is written, inside every interpreter, total and RNG scope around it. Under reverse mode the primal scan runs first, and the recorder then records one linear call whose pullback reruns each step at its carry, in reverse.
- **Custom rules.** The innermost differentiation that owns an argument of a custom call evaluates the rule on the arguments' primals. For `custom_jvp` it offers the call outward, so an enclosing differentiation applies the same rule at its own level, takes the answer as the result's primal, and applies the tangent map to its tangents; under reverse mode the map's operations are recorded and transposed. For `custom_vjp` the recorder records the rule's pullback as a linear call, and forward mode raises when the result holds a tensor. A rule that uses a value its own differentiation tracks other than through its arguments raises.
- **`remat`**: under reverse mode, it runs its function under a child installation that records onto a scratch recorder it then drops. Each application of the pullback reruns the function at the kept arguments, transposes that fresh recording, and returns the cotangents of the arguments and of the tensors the function captured. The rerun runs inside a scope that discards totals' additions, so an addition counts once. Forward mode passes the remat on as the remat of the function's `jvp`, over primals and tangents together, and `vmap` as the remat of the batched function.
- **`detach x`** strips the derivative each installation around the call owns and passes the rest outward. It copies nothing.
- **Totals.** A `Rune.Total.collect` scope answers additions to its total, and threads its sum through scans and remats as one more carry or result. `vmap` sums its lanes' additions before passing them on.

## The Failure Model

Every Nx operation is matched explicitly by each interpreter, and the compiler refuses one that misses one. Operations fall into three groups:

- **Zero derivative** (comparisons, bitwise and integer ops, rounding, `argmax`/`argsort`, RNG, reads): the result is a plain value, which is the correct zero derivative.
- **A forward rule, a transpose where the operation is linear, and a batching rule**: every other operation.
- **No definition**: the tangent of a complete SVD of a non-square matrix or a complete QR factorisation of a tall one raises when an input is tracked, instead of silently producing a wrong derivative. `detach` the input if differentiation should not flow through it.

The recorder also refuses an operation that is not linear in a slot, which only a `custom_jvp` tangent map can issue. One case passes silently: under reverse mode, a plain value a tangent map selects with `Nx.where`, concatenates, scatters or writes beside a tangent is taken as zero, so a map must give such a value only as a tangent's zero fill.

## Implications for Users

**No graph construction step.** Everything runs eagerly. `if`, `match`, `for`, recursion, and higher-order functions all work inside differentiated code. There is no graph-compatible subset of the language. The `scan` combinator gives a loop a structure `jit` can see, so it compiles as a loop instead of unrolling; it is never required.

**Coefficients are computed in the forward pass.** Reverse mode computes each rule's coefficients (`cos x` for `sin x`) as the function runs, and records only the linear operations. The backward pass replays the record; it does not re-execute your function, unless you asked for that with `remat`. Printing inside a differentiated function happens once, in the forward pass.

**Per-operation overhead.** Interpreting an operation costs more than a raw Nx call, and under `grad` each operation reaches two interpreters. For workloads dominated by large operations (matrix multiplications), the overhead is negligible; for many tiny operations it is visible, and `jit` removes it.
