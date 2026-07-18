# Transformations

Rune provides functional transformations over ordinary OCaml functions of Nx tensors. This guide covers every transformation available.

## Parameter Structures

Every structured transformation takes a first-class module implementing `Nx.Ptree.S` — three one-line traversals over the structure's tensor leaves. A single tensor is itself a one-leaf structure:

```ocaml
module Vec = struct
  type t = Nx.float32_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) v = f v
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) v = f v
end
```

For functions of a single tensor, the primed variants (`grad'`, `vjp'`, `jvp'`, `vmap'`, `hvp'`) skip the module argument entirely; they are used below wherever the structure does not matter. The `2`-suffixed variants (`vjp2`, `jvp2`, `vmap2`) take a second module for functions returning a *structure* rather than one tensor.

## Reverse-Mode AD

Reverse mode (backpropagation) computes gradients for all inputs in one backward pass — the right tool when a scalar objective depends on many parameters.

### grad

`grad (module P) f params` is the gradient of the scalar-valued `f` at `params`, with the same structure and leaf dtypes as `params`:

```ocaml
let () =
  let f v = Nx.sum (Nx.mul v v) in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  Printf.printf "%s\n" (Nx.to_string (Rune.grad (module Vec) f x))
  (* gradient: [2. 4. 6.] *)
```

`f params` must be a scalar (a tensor with exactly one element); use `vjp` for non-scalar outputs. Integer or boolean leaves raise — hold non-differentiable data in the closure or the auxiliary output.

### value_and_grad

Computes the value and the gradient in a single forward and backward pass:

```ocaml
let () =
  let f v = Nx.mean (Nx.mul v v) in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let value, gradient = Rune.value_and_grad (module Vec) f x in
  Printf.printf "f(x) = %.4f\n" (Nx.item [] value);
  Printf.printf "%s\n" (Nx.to_string gradient)
```

### value_and_grad_aux

When the objective returns auxiliary data alongside the loss — predictions, metrics, updated state — the `_aux` variant threads it through undifferentiated:

```ocaml
let () =
  let f v =
    let pred = Nx.mul v v in
    (Nx.mean pred, pred) (* pred is auxiliary *)
  in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let loss, gradient, pred = Rune.value_and_grad_aux (module Vec) f x in
  ignore (loss, gradient, pred)
```

### vjp

Vector-Jacobian product: the function need not return a scalar — you provide the cotangent to pull back. `vjp'` is the single-tensor variant:

```ocaml
let () =
  let f v = Nx.mul v v in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let ct = Nx.ones Nx.float32 [| 3 |] in
  let y, g = Rune.vjp' f x ct in
  Printf.printf "%s\n" (Nx.to_string y); (* [1. 4. 9.] *)
  Printf.printf "%s\n" (Nx.to_string g) (* [2. 4. 6.] *)
```

The cotangent must have the output's shape and dtype. For a function returning a whole structure, `vjp2 (module P) (module Q) f params cotangents` takes one cotangent per output leaf.

### vjp_fun

`vjp_fun` returns the output and a reusable pullback. Each call to the pullback runs one backward pass over the recorded computation without re-running `f` — useful for pulling back several cotangents:

```ocaml
let () =
  let f v = Nx.mul v v in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let y, pullback = Rune.vjp_fun' f x in
  let g1 = pullback (Nx.ones Nx.float32 [| 3 |]) in
  let g2 = pullback (Nx.create Nx.float32 [| 3 |] [| 0.; 1.; 0. |]) in
  ignore (y, g1, g2)
```

Calling the pullback under another transformation (for example `vmap'`) transforms the backward pass — that is how `jacrev'` vectorizes its rows.

## Forward-Mode AD

Forward mode propagates a tangent alongside the value in a single forward pass, with no tape. It is the right tool when inputs are few relative to outputs.

### jvp

Jacobian-vector product. The tangents mirror the parameters (same structure, each leaf its parameter leaf's shape); the output may have any shape:

```ocaml
let () =
  let f v = Nx.mul v v in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let v = Nx.ones Nx.float32 [| 3 |] in
  let y, tangent = Rune.jvp' f x v in
  Printf.printf "%s\n" (Nx.to_string y); (* [1. 4. 9.] — primal *)
  Printf.printf "%s\n" (Nx.to_string tangent)
  (* [2. 4. 6.] — directional derivative *)
```

`jvp_aux` carries auxiliary outputs; `jvp2` handles functions returning a structure (one tangent per output leaf).

### Choosing Between Forward and Reverse Mode

- **Reverse mode** (`grad`, `vjp`): one backward pass gives gradients for all inputs. Best when outputs ≪ inputs — the typical ML case of a scalar loss over many parameters.
- **Forward mode** (`jvp`): one forward pass gives one directional derivative. Best when inputs ≪ outputs, and as the outer layer of forward-over-reverse compositions (see `hvp` below).

## Vectorizing Maps

### vmap

`vmap` lifts a function written for one example to batched inputs. The mapped function observes each leaf without its mapped axis; its result gains a batch axis at `out_axis` (default `0`):

```ocaml
let () =
  (* f is written for a single vector of shape [5]. *)
  let f v = Nx.sum (Nx.mul v v) in
  let batch = Nx.ones Nx.float32 [| 10; 5 |] in
  let results = Rune.vmap' f batch in
  Format.printf "results has shape %a@." Nx.pp_shape (Nx.shape results)
  (* [10] — one scalar per example *)
```

Values the function closes over are constants of the map. For structures, `in_axes` pairs one entry per leaf in traversal order: `Some i` maps axis `i` (negative counts from the end), `None` passes the leaf whole as a constant:

```ocaml
type pair = { x : Nx.float32_t; y : Nx.float32_t }

module Pair = struct
  type t = pair

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { x; y } =
    { x = f x; y = f y }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { x = f a.x b.x; y = f a.y b.y }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { x; y } =
    f x;
    f y
end

let () =
  let pairs =
    {
      x = Nx.create Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int);
      y = Nx.create Nx.float32 [| 3 |] [| 1.; 0.; -1. |];
    }
  in
  (* Map over rows of x; y is passed whole to every lane. *)
  let dots =
    Rune.vmap ~in_axes:[ Some 0; None ]
      (module Pair)
      (fun p -> Nx.dot p.x p.y)
      pairs
  in
  Printf.printf "%s\n" (Nx.to_string dots)
```

`vmap2` is the variant for mapped functions returning a structure: every output leaf gains the batch axis.

**Note.** Implicit random number generation (`Nx.rand` and friends) inside the mapped function draws *identical* values for every lane — the RNG key is a constant of the map. Thread distinct randomness in as mapped inputs instead. Reading a batched tensor's value inside the mapped function raises.

### Per-Sample Gradients

Transformations nest freely, and `vmap` of `grad` is the canonical composition: write the loss for one example, differentiate it, map the differentiated function over the batch. Each gradient leaf gains a leading batch axis:

```ocaml
type params = { w : Nx.float32_t; b : Nx.float32_t }

module Params = struct
  type t = params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b } =
    { w = f w; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b } =
    f w;
    f b
end

let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let n, d = (8, 3) in
  let params =
    { w = Nx.randn Nx.float32 [| d |]; b = Nx.randn Nx.float32 [||] }
  in
  let batch =
    { x = Nx.randn Nx.float32 [| n; d |]; y = Nx.randn Nx.float32 [| n |] }
  in
  (* Squared error of a linear model on a single example. *)
  let loss ex p = Nx.square (Nx.sub (Nx.add (Nx.dot ex.x p.w) p.b) ex.y) in
  (* grad gives the per-example gradient; vmap2 maps it over the batch. *)
  let per_sample =
    Rune.vmap2
      (module Pair)
      (module Params)
      (fun ex -> Rune.grad (module Params) (loss ex) params)
      batch
  in
  Format.printf "per-sample dw: %a@." Nx.pp_shape (Nx.shape per_sample.w);
  Format.printf "per-sample db: %a@." Nx.pp_shape (Nx.shape per_sample.b)
  (* dw is [8; 3], db is [8]: one gradient per example. *)
```

The parameters are closed over, so they are constants of the map and gradients are taken with respect to them. Per-sample gradient norms — for gradient clipping in DP-SGD, for example — are one `Nx.sum` away. The full program, including a check against the explicit loop, is [`examples/02-per-sample-grads`](https://github.com/raven-ml/raven/tree/main/packages/rune/examples/02-per-sample-grads).

## Jacobians and Hessians

For whole derivative matrices, `jacfwd'` computes the Jacobian column by column in forward mode (prefer it when the input is smaller than the output) and `jacrev'` row by row in reverse mode (prefer it when the output is smaller). Both have shape `shape (f x) @ shape x`.

`hessian'` is forward-over-reverse; `hvp` computes Hessian-vector products without materializing the Hessian. Newton's method on the Rosenbrock function:

```ocaml
let rosenbrock x =
  let x0 = Nx.slice [ Nx.I 0 ] x and x1 = Nx.slice [ Nx.I 1 ] x in
  let a = Nx.square (Nx.sub_s x0 1.0) in
  let b = Nx.square (Nx.sub x1 (Nx.square x0)) in
  Nx.add a (Nx.mul_s b 100.0)

let () =
  let x = ref (Nx.create Nx.float64 [| 2 |] [| -1.2; 1.0 |]) in
  for _ = 1 to 8 do
    let g = Rune.grad' rosenbrock !x in
    let h = Rune.hessian' rosenbrock !x in
    x := Nx.sub !x (Nx.solve h g)
  done;
  Printf.printf "minimum at %s\n" (Nx.to_string !x)
  (* converges to (1, 1) *)
```

`hvp' f x v` equals `(hessian' f x) @ v` but never forms the matrix:

```ocaml
let () =
  let x = Nx.create Nx.float64 [| 2 |] [| -1.2; 1.0 |] in
  let v = Nx.create Nx.float64 [| 2 |] [| 0.5; -1.0 |] in
  let hv = Rune.hvp' rosenbrock x v in
  let hv' = Nx.matmul (Rune.hessian' rosenbrock x) v in
  Printf.printf "hvp:         %s\n" (Nx.to_string hv);
  Printf.printf "hessian @ v: %s\n" (Nx.to_string hv')
```

`hvp` (unprimed) does the same for any parameter structure. Under the hood it is `jvp` of `grad` — forward-over-reverse — one more instance of transformations composing.

## Gradient Checkpointing

`remat (module P) f params` is `f params` recomputed during the backward pass instead of having its intermediates retained by the tape. Gradients are unchanged; memory is traded for compute:

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let expensive v = Nx.mean (Nx.square (Nx.sin v)) in
  let g_plain = Rune.grad' expensive x in
  let g_remat =
    Rune.grad' (fun v -> Rune.remat (module Vec) expensive v) x
  in
  Printf.printf "%s\n" (Nx.to_string g_plain);
  Printf.printf "%s\n" (Nx.to_string g_remat) (* identical *)
```

Wrap the memory-heavy sub-computation (a transformer block, say), not the whole objective. `remat` is a `custom_vjp` rule underneath, so differentiating it in forward mode raises.

## Custom Differentiation Rules

When you know a better rule than the composition of primitive rules — cheaper, more stable, or for a function whose interior should not be differentiated — override it.

### custom_vjp

`custom_vjp (module P) ~fwd ~bwd params` computes `fst (fwd params)`; under the innermost reverse-mode transformation, `bwd residual cotangent` supplies the parameter gradients instead of differentiating `fwd`'s interior. The residual is whatever `fwd` returned alongside its result:

```ocaml
let () =
  (* f(x) = x², with a hand-written backward rule. *)
  let f x =
    Rune.custom_vjp
      (module Vec)
      ~fwd:(fun x -> (Nx.square x, x)) (* save x as the residual *)
      ~bwd:(fun x ct -> Nx.mul ct (Nx.mul_s x 2.0)) (* ct · 2x *)
      x
  in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  Printf.printf "%s\n"
    (Nx.to_string (Rune.grad' (fun v -> Nx.sum (f v)) x))
  (* [2. 4. 6.] — from the custom rule *)
```

Each gradient leaf must match its parameter leaf's shape and dtype. Enclosing transformations (an outer `grad`, a `vmap`) see the forward computation itself; only the innermost reverse-mode transformation applies the rule. Differentiating a `custom_vjp` call in forward mode raises — define a `custom_jvp` rule for that.

### custom_jvp

The forward-mode counterpart: `jvp params tangents` provides both the result and its tangent:

```ocaml
let () =
  let f x =
    Rune.custom_jvp
      (module Vec)
      ~f:Nx.square
      ~jvp:(fun x dx -> (Nx.square x, Nx.mul_s (Nx.mul x dx) 2.0))
      x
  in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let v = Nx.ones Nx.float32 [| 3 |] in
  let _, tangent = Rune.jvp' f x v in
  Printf.printf "%s\n" (Nx.to_string tangent)
  (* [2. 4. 6.] — from the custom rule *)
```

With no transformation in scope, both constructs just run the plain forward function.

## Gradient Checking

`check_grads` compares the reverse-mode gradient of a scalar objective against central-difference directional derivatives along deterministic directions:

```ocaml
module Vec64 = struct
  type t = Nx.float64_t

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) v = f v
  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) = f
  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) v = f v
end

let () =
  let x = Nx.create Nx.float64 [| 2 |] [| -1.2; 1.0 |] in
  match Rune.check_grads (module Vec64) rosenbrock x with
  | Ok () -> print_endline "reverse mode agrees with finite differences"
  | Error msg -> print_endline msg
```

The check is directional, not per-element: it validates gradients cheaply rather than exhaustively. Use float64 parameters for reliable results; float32 may need a looser `~tol` (the default is `1e-2` relative, with `~eps` the finite-difference step, default `1e-4`).

## Control Flow

Ordinary OCaml control flow — `if`, `match`, loops, recursion — works inside every transformation, because rune runs eagerly and intercepts operations as they execute. The `scan`, `cond`, and `while_loop` combinators exist for a different reason: their signatures are staging-ready, so code written with them differentiates and vectorizes today, and a future `jit` can stage them as structured control flow instead of unrolled traces.

### scan

`scan (module C) ~f ~init xs` folds `f` over slices of `xs` along axis 0; `f carry x` returns the next carry and a per-step output. The result is the final carry and the outputs stacked along a new axis 0:

```ocaml
let () =
  let xs = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let final, partials =
    Rune.scan
      (module Vec)
      ~f:(fun c x ->
        let c = Nx.add c x in
        (c, c))
      ~init:(Nx.scalar Nx.float32 0.0) xs
  in
  Printf.printf "final: %s\n" (Nx.to_string final);
  Printf.printf "%s\n" (Nx.to_string partials)
  (* the running sums [1. 3. 6. 10.] *)
```

Differentiating traces every step.

### cond and while_loop

`cond pred ~then_ ~else_` runs one branch according to the scalar boolean `pred`; `while_loop (module C) ~cond ~body init` iterates `body` on the carry while `cond` holds:

```ocaml
let () =
  let branch x =
    Rune.cond
      (Nx.greater (Nx.sum x) (Nx.scalar Nx.float32 0.0))
      ~then_:(fun () -> Nx.sum (Nx.mul x x))
      ~else_:(fun () -> Nx.sum x)
  in
  Printf.printf "then: %.2f\n"
    (Nx.item [] (branch (Nx.create Nx.float32 [| 2 |] [| 0.5; 2.0 |])));

  (* Double the carry until its sum exceeds 10. *)
  let y =
    Rune.while_loop
      (module Vec)
      ~cond:(fun c -> Nx.less (Nx.sum c) (Nx.scalar Nx.float32 10.0))
      ~body:(fun c -> Nx.mul_s c 2.0)
      (Nx.create Nx.float32 [| 2 |] [| 1.0; 0.5 |])
  in
  Printf.printf "%s\n" (Nx.to_string y) (* [8. 4.] *)
```

Differentiation traces the branch or iterations actually taken. Reading the predicate concretizes it: inside `vmap`, a predicate that depends on the mapped inputs raises, since the lanes could diverge.

## Debugging

`with_debug` runs a thunk and logs each tensor operation it performs — the operation name and output shape — to a formatter (`Format.err_formatter` by default). Run it outermost to also observe the operations other transformations emit:

```ocaml
let () =
  let f x = Nx.add (Nx.mul x x) (Nx.sin x) in
  let x = Nx.scalar Nx.float32 2.0 in
  Rune.with_debug (fun () -> ignore (Rune.grad' f x))
  (* logs the forward operations, then the backward-pass operations *)
```

## Autodiff Control

`detach t` is a copy of `t` through which gradients do not flow; `no_grad f` runs `f` with gradient tracking disabled entirely. Use them for baselines, targets, constants — and as the escape hatch for operations without gradient rules (see below).

## Limitations

Rune fails loudly rather than returning wrong gradients:

- **Ops without differentiation rules raise.** Reverse mode has no rule for `svd`, `eig`, `eigh`, `rfft`, `irfft`, `psum`, and `mod`; forward mode additionally lacks `qr`. Differentiating through them raises `Invalid_argument` — `detach` the input if gradients should not flow through. (`cholesky` and reverse-mode `qr` are supported.)
- **`vmap` has no rule for `fft`-family and decomposition ops** (`fft`, `ifft`, `rfft`, `irfft`, `cholesky`, `qr`, `svd`, `eig`, `eigh`) over batched inputs.
- **In-place mutation** (`set_item`, `set_slice`, `blit`, `assign`) raises during differentiation; write the update functionally.
- **No JIT yet.** Everything runs eagerly; `scan`/`cond`/`while_loop` are designed so a future `jit` can stage them without unrolling.

## Summary

| Transform | Purpose | When to use |
|-----------|---------|-------------|
| `grad` / `grad'` | Gradient of scalar objective | Training loss → parameter gradients |
| `value_and_grad` | Value + gradient together | Avoid a duplicate forward pass |
| `value_and_grad_aux` | ... plus auxiliary data | Thread state/metrics out of the objective |
| `vjp` / `vjp_fun` | Vector-Jacobian product | Non-scalar outputs, reusable pullbacks |
| `jvp` | Jacobian-vector product | Few inputs, many outputs |
| `vmap` / `vmap2` | Vectorize over a batch axis | Per-example computation |
| `jacfwd'` / `jacrev'` / `hessian'` | Whole derivative matrices | Small problems, second-order methods |
| `hvp` | Matrix-free Hessian-vector product | Large second-order computations |
| `remat` | Recompute in the backward pass | Memory-bound backward passes |
| `custom_vjp` / `custom_jvp` | User-defined rules | Stability, speed, opaque interiors |
| `scan` / `cond` / `while_loop` | Staging-ready control flow | Future-proof loops and branches |
| `check_grads` | Verify gradients | Testing custom rules and models |
| `detach` / `no_grad` | Stop gradient flow | Baselines, targets, unruled ops |
| `with_debug` | Log every operation | Understanding and debugging |
