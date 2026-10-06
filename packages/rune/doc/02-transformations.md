# Transformations

Rune provides functional transformations over ordinary OCaml functions of Nx tensors. This guide covers every transformation available.

## Parameter Structures

A transformation takes the structure of each value whose tensors it enumerates, an `'s Nx.Ptree.t`, and captures everything else. `grad`, `vjp`, `jvp` and their kin take the structure of the value they differentiate and, where they rebuild one, of the result; `vmap` and `remat` take the signature of the function they transform. `Nx.Ptree.tensor` is the structure of one tensor, and `Nx.Ptree.instantiate` makes one from a record's `walk` (see [Getting Started](01-getting-started.md)).

For functions of a single tensor, the primed variants (`grad'`, `vjp'`, `jvp'`, `vmap'`) take no structure; they are used below wherever the structure does not matter.

A transformation tracks each tensor of its arguments at its position. A tensor behind two positions is two arguments, each with its own derivative, and a tensor the function captures is a constant even when it is also the argument: `Rune.grad' (fun x -> Nx.sum (Nx.mul x w)) w` is `w`. To tie two uses of a weight, pass it once and use it twice.

### Values stay inside

A value computed inside a transformed function leaves it through the function's result. Kept in a reference and read after the transformation returns, used on another domain, or held by a closure that runs later, it has no bytes, and using it raises:

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let seen = ref None in
  let _ =
    Rune.grad'
      (fun x ->
        let h = Nx.tanh x in
        seen := Some h;
        Nx.sum h)
      x
  in
  match Nx.to_array (Option.get !seen) with
  | _ -> print_endline "read"
  | exception Invalid_argument _ -> print_endline "the value stayed inside"
```

Inside the function `Nx.item` and `Nx.print` read values, so an OCaml `if` on a value works under differentiation. To get a value out, return it: `value_and_grad_aux` returns an objective's auxiliary values, and `vjp` and `jvp` return whatever structure the function's result has.

## Reverse-Mode AD

Reverse mode (backpropagation) computes gradients for all inputs in one backward pass — the right tool when a scalar objective depends on many parameters.

### grad

`grad p f params` is the gradient of the scalar-valued `f` at `params`, with the same structure and leaf dtypes as `params`:

```ocaml
let () =
  let f v = Nx.sum (Nx.mul v v) in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  Printf.printf "%s\n" (Nx.to_string (Rune.grad Nx.Ptree.tensor f x))
  (* gradient: [2. 4. 6.] *)
```

`f params` must be a scalar (a tensor with exactly one element); use `vjp` for non-scalar outputs. A structure may hold integer or boolean tensors, such as an RNG key or a step counter: they are carried with a zero gradient. `grad'` of an integer tensor raises.

### value_and_grad

Computes the value and the gradient in a single forward and backward pass:

```ocaml
let () =
  let f v = Nx.mean (Nx.mul v v) in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let value, gradient = Rune.value_and_grad Nx.Ptree.tensor f x in
  Printf.printf "f(x) = %.4f\n" (Nx.item [] value);
  Printf.printf "%s\n" (Nx.to_string gradient)
```

### value_and_grad_aux

When the objective returns auxiliary data alongside the loss (predictions, metrics, updated state), the `_aux` variant returns it beside the gradient. It takes the auxiliary value's structure, which is how it leaves the differentiation as plain values:

```ocaml
let () =
  let f v =
    let pred = Nx.mul v v in
    (Nx.mean pred, pred) (* pred is auxiliary *)
  in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let loss, gradient, pred =
    Rune.value_and_grad_aux Nx.Ptree.tensor Nx.Ptree.tensor f x
  in
  ignore (loss, gradient, pred)
```

### vjp

Vector-Jacobian product: the function need not return a scalar. `vjp` returns the result and its pullback, which maps a cotangent of the result to the gradient. `vjp'` is the single-tensor variant:

```ocaml
let () =
  let f v = Nx.mul v v in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let y, pullback = Rune.vjp' f x in
  let g = pullback (Nx.ones Nx.float32 [| 3 |]) in
  Printf.printf "%s\n" (Nx.to_string y); (* [1. 4. 9.] *)
  Printf.printf "%s\n" (Nx.to_string g) (* [2. 4. 6.] *)
```

The cotangent must have the output's shape and dtype. `vjp p q f params` takes the structures of the parameters and of the result, so a function returning a structure takes one cotangent per tensor of its result:

```ocaml
let () =
  let f v = (Nx.mul v v, Nx.sum v) in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let _, pullback = Rune.vjp Nx.Ptree.tensor Nx.Ptree.(pair tensor tensor) f x in
  let g = pullback (Nx.ones Nx.float32 [| 3 |], Nx.scalar Nx.float32 1.0) in
  Printf.printf "%s\n" (Nx.to_string g) (* [3. 5. 7.] *)
```

The pullback runs no part of `f` again: where `f` passes a function to `remat`, it recomputes that function's operations from a record of them. When `vjp` runs outside every transformation, the pullback may be applied to any number of cotangents, from any domain. Under another transformation it is transformed with it: under `vmap'` the backward pass is batched, which is how `jacrev'` computes its rows.

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

`jvp p q f params tangents` takes the structures of the parameters and of the result, and returns one tangent per tensor of the result. A tensor returned beside the result is part of it: give `q` a structure that holds it, and its tangent comes back with the rest. A value that is not a tensor, such as a count, is computed from the result after `jvp` returns.

### Choosing Between Forward and Reverse Mode

- **Reverse mode** (`grad`, `vjp`): one backward pass gives gradients for all inputs. Best when outputs ≪ inputs — the typical ML case of a scalar loss over many parameters.
- **Forward mode** (`jvp`): one forward pass gives one directional derivative. Best when inputs ≪ outputs, and as the outer layer of forward-over-reverse compositions (see Hessian-vector products below).

## Vectorizing Maps

### vmap

`vmap` lifts a function written for one example to batched inputs. It maps axis 0 of every tensor of the function's arguments; the mapped function sees each without that axis, and every tensor of its result gains a batch axis 0. `vmap'` is the form for a function of one tensor:

```ocaml
let () =
  (* f is written for a single vector of shape [5]. *)
  let f v = Nx.sum (Nx.mul v v) in
  let batch = Nx.ones Nx.float32 [| 10; 5 |] in
  let results = Rune.vmap' f batch in
  Format.printf "results has shape %a@." Nx.pp_shape (Nx.shape results)
  (* [10] — one scalar per example *)
```

`vmap` takes the signature of the function it maps: one structure per argument, built with `@->`, and the structure of the result, with `returns`. A value the function captures is a constant of the map, passed whole to every lane, and another axis is mapped by moving it to the front with `Nx.moveaxis`, a view:

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 4; 3 |] (Array.init 12 float_of_int) in
  let y = Nx.create Nx.float32 [| 3 |] [| 1.; 0.; -1. |] in
  (* Map over rows of x; y is captured, whole in every lane. *)
  let dots =
    Rune.vmap Nx.Ptree.(tensor @-> returns tensor) (fun x -> Nx.dot x y) x
  in
  Printf.printf "%s\n" (Nx.to_string dots);
  (* Map over the columns of x and over y together. *)
  let scaled =
    Rune.vmap
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (fun column k -> Nx.mul column k)
      (Nx.moveaxis 1 0 x) y
  in
  Format.printf "scaled has shape %a@." Nx.pp_shape (Nx.shape scaled)
  (* [3; 4] — one column per lane *)
```

**Note.** Implicit random number generation (`Nx.rand` and friends) inside the mapped function draws *identical* values for every lane — the RNG key is a constant of the map. Thread distinct randomness in as a mapped input instead: map over a batch of keys from `Nx.Rng.split_batch`, walked with `Nx.Rng.ptree`, and each lane sees its own key. Reading a batched tensor's value inside the mapped function raises.

### Per-Sample Gradients

Transformations nest freely, and `vmap` of `grad` is the canonical composition: write the loss for one example, differentiate it, map the differentiated function over the batch. Each gradient leaf gains a leading batch axis:

```ocaml
type 'a params = { w : 'a; b : 'a }

module Params = struct
  type 'a t = 'a params

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    { w; b }
end

let params_ptree = Nx.Ptree.instantiate (module Params)

let () =
  Nx.Rng.with_key (Nx.Rng.key 0) @@ fun () ->
  let n, d = (8, 3) in
  let params =
    { w = Nx.randn Nx.float32 [| d |]; b = Nx.randn Nx.float32 [||] }
  in
  let xs = Nx.randn Nx.float32 [| n; d |] and ys = Nx.randn Nx.float32 [| n |] in
  (* Squared error of a linear model on a single example. *)
  let loss x y p = Nx.square (Nx.sub (Nx.add (Nx.dot x p.w) p.b) y) in
  (* grad gives the per-example gradient; vmap maps it over the batch. *)
  let per_sample =
    Rune.vmap
      Nx.Ptree.(tensor @-> tensor @-> returns params_ptree)
      (fun x y -> Rune.grad params_ptree (loss x y) params)
      xs ys
  in
  Format.printf "per-sample dw: %a@." Nx.pp_shape (Nx.shape per_sample.w);
  Format.printf "per-sample db: %a@." Nx.pp_shape (Nx.shape per_sample.b)
  (* dw is [8; 3], db is [8]: one gradient per example. *)
```

The parameters are closed over, so they are constants of the map and gradients are taken with respect to them. Per-sample gradient norms — for gradient clipping in DP-SGD, for example — are one `Nx.sum` away. The full program, including a check against the explicit loop, is [`examples/02-per-sample-grads`](https://github.com/raven-ml/raven/tree/main/packages/rune/examples/02-per-sample-grads).

## Jacobians and Hessians

For whole derivative matrices, `jacfwd'` computes the Jacobian column by column in forward mode (prefer it when the input is smaller than the output) and `jacrev'` row by row in reverse mode (prefer it when the output is smaller). Both have shape `shape (f x) @ shape x`.

A Hessian is the Jacobian of the gradient, `jacfwd' (grad' f) x`: forward mode over reverse mode. Newton's method on the Rosenbrock function:

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
    let h = Rune.jacfwd' (Rune.grad' rosenbrock) !x in
    x := Nx.sub !x (Nx.solve h g)
  done;
  Printf.printf "minimum at %s\n" (Nx.to_string !x)
  (* converges to (1, 1) *)
```

A Hessian-vector product is the tangent of the gradient along `v`, `snd (jvp' (grad' f) x v)`, which never forms the matrix:

```ocaml
let () =
  let x = Nx.create Nx.float64 [| 2 |] [| -1.2; 1.0 |] in
  let v = Nx.create Nx.float64 [| 2 |] [| 0.5; -1.0 |] in
  let hv = snd (Rune.jvp' (Rune.grad' rosenbrock) x v) in
  let hv' = Nx.matmul (Rune.jacfwd' (Rune.grad' rosenbrock) x) v in
  Printf.printf "hvp:         %s\n" (Nx.to_string hv);
  Printf.printf "hessian @ v: %s\n" (Nx.to_string hv')
```

For any parameter structure `p` it is `snd (Rune.jvp p p (Rune.grad p f) params v)`.

## Gradient Checkpointing

`remat s f` is `f` recomputed during the backward pass instead of having its intermediates retained by the tape. `s` is `f`'s signature, as for `vmap`, so a curried layer function is rematerialized as it is. Gradients are unchanged; memory is traded for compute:

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let expensive v = Nx.mean (Nx.square (Nx.sin v)) in
  let g_plain = Rune.grad' expensive x in
  let g_remat =
    Rune.grad' (Rune.remat Nx.Ptree.(tensor @-> returns tensor) expensive) x
  in
  Printf.printf "%s\n" (Nx.to_string g_plain);
  Printf.printf "%s\n" (Nx.to_string g_remat) (* identical *)
```

Wrap the memory-heavy sub-computation (a transformer block, say), not the whole objective. The block may close over its weights: every transformation sees `remat s f` as it sees `f`, so gradients and tangents reach the tensors it captures. Under `jit`, the recomputation runs in the backward pass, once the block's output cotangent exists, so the compiled program holds one block's intermediates at a time. A compiled `scan` already recomputes each step in its backward loop, so a `scan` step gains nothing from `remat`. Forward mode keeps no intermediates: under `jvp`, `remat s f` computes what `f` does.

## Custom Differentiation Rules

A custom rule gives a function the derivative a transformation would compute for it, in the form of that transformation's answer. The rule receives the arguments' values and returns the result with its derivative. It must not use a value its own differentiation tracks other than through its arguments: pass such a value as an argument.

### custom_jvp

`custom_jvp p q rule args` is `fst (rule args)`, whose derivative is the tangent map `snd (rule args)`: a function from the arguments' tangents to the result's, linear in them. One rule serves both modes, at every order. Forward mode applies the map, reverse mode transposes the operations the map performs, and every differentiation around the call differentiates the map's code for the second-order terms:

```ocaml
let stable x =
  Nx.add (Nx.maximum_s x 0.) (Nx.log (Nx.add_s (Nx.exp (Nx.neg (Nx.abs x))) 1.))

let softplus =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      (stable x, fun dx -> Nx.mul (Nx.sigmoid x) dx))

let () =
  let x = Nx.create Nx.float32 [| 3 |] [| -1.; 0.; 1. |] in
  let g = Rune.grad' (fun x -> Nx.sum (softplus x)) x in
  let _, t = Rune.jvp' softplus x (Nx.ones Nx.float32 [| 3 |]) in
  Printf.printf "%s\n%s\n" (Nx.to_string g) (Nx.to_string t)
  (* both are sigmoid x *)
```

Under reverse mode the map's tangents have no values, so the map must be linear: an operation that is not linear in a tangent (`Nx.exp dx`), an offset added to a tangent, or a read of a tangent's value raises, naming the entry point that differentiates. A value the map selects with `Nx.where`, concatenates, scatters or writes beside a tangent is taken as zero, so give such a value only as a tangent's zero fill. When the result holds no tensor, reverse mode does not apply the map: a rule with a unit result observes its arguments' tangents in forward mode and is inert under `grad`.

### custom_vjp

A backward rule that is not the transpose of a forward one is a `custom_vjp`. Its rule returns the result and the pullback, which maps the result's cotangents to the arguments' gradients as `vjp`'s pullback does:

```ocaml
let clip_grad c =
  Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      (x, fun g -> Nx.clamp ~min:(-.c) ~max:c g))

let () =
  let x = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let g = Rune.grad' (fun x -> Nx.sum (Nx.mul_s (clip_grad 1.0 x) 5.0)) x in
  Printf.printf "%s\n" (Nx.to_string g) (* [1. 1. 1.], clipped from 5 *)
```

The pullback receives zeros for a tensor of the result nothing used, and its result is checked against the arguments' structure. Differentiations around the call differentiate the rule's code and the pullback's. Forward mode has no rule to apply: `jvp` of a `custom_vjp` whose result holds a tensor raises.

`vmap` batches both kinds of rule, and the gradient of an argument the map does not batch is summed over its lanes. With no differentiation around the call, a rule only computes its result.

## Totals and Lanes

A total is a write-only sum that code anywhere inside a function adds to and that the caller reads when the function returns:

```ocaml
let () =
  let total = Rune.Total.make () in
  let cell h x =
    let h = Nx.tanh (Nx.add h x) in
    Rune.Total.add total (Nx.sum x);
    (h, h)
  in
  let xs = Nx.create Nx.float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let _, sum =
    Rune.Total.collect total ~zero:(Nx.zeros Nx.float32 [||]) (fun () ->
        Rune.scan' ~f:cell ~init:(Nx.zeros Nx.float32 [| 2 |]) xs)
  in
  Printf.printf "%s\n" (Nx.to_string sum)
  (* 21 — the rows' sums, added at every step of the scan *)
```

`Rune.Total.collect t ~zero f` runs `f` and returns its result with `zero` plus everything `f` added to `t`. Nothing reads a total before its `collect` returns, so an addition never changes a value the function computes, and with no `collect` open an addition does nothing. An addition counts once per execution of the code that makes it, whatever lies between it and the scope: `vmap` adds the sum of its lanes' additions, the replays of a backward pass add nothing, and a `scan` that `jit` stages carries the sum out of its loop, so the loop stays one loop and a replay computes the total again. A `jit` inside a scope compiles the function that also returns the sum of its additions, which the scope adds.

`Rune.axis ()` names a map. `Rune.vmap ~axis:a` (or `Rune.vmap' ~axis:a`) gives its map the name `a`, and inside it `Rune.lanes a x` is every lane's `x` stacked on a new leading axis, as data: the same value in every lane, whether `x` differs across the lanes or every lane shares it:

```ocaml
let () =
  let a = Rune.axis () in
  let dirs = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0. |] in
  let wide =
    Rune.vmap ~axis:a Nx.Ptree.(tensor @-> returns tensor)
      (fun _ -> Rune.lanes a (Nx.ones Nx.float32 [| 3 |]))
      dirs
  in
  Format.printf "wide: %a@." Nx.pp_shape (Nx.shape wide)
  (* wide: [2,2,3] — each of the 2 lanes holds the gather of 2 rows *)
```

The map named `a` answers the call itself, so the gathered value is a constant of that map, and enclosing transformations see the gather. Every other map passes the call on and keeps its own lanes in front of the gathered axis, and with no map named `a` around the call, `lanes a x` is one lane. `Rune.lane_index ?axis ()` addresses maps alike: the map named `axis`, or the innermost anonymous map when `axis` is absent, answers with each lane's index, and `Nx.Rng.fold_in_tensor k (Rune.lane_index ())` gives each lane its own key. `lanes a` is linear: `jvp` gathers the tangent with the primal, and `grad` gives each lane its row of the summed cotangent.

Together they let a forward-mode rule use every lane at once. Under `vmap ~axis:a` over tangent directions around `jvp`, a `custom_jvp` with a unit result, called anywhere in a model, has a tangent map that can gather its argument's tangents across the directions with `lanes a` and add a quantity built from all of them to a total that the caller collects inside that map. The rule receives the argument's value too, so the quantity may depend on it. The same model runs under `grad`, where the map is not applied and the call does nothing.

## Gradient Checking

`check_grads` compares the reverse-mode gradient of a scalar objective against central-difference directional derivatives along deterministic directions:

```ocaml
let () =
  let x = Nx.create Nx.float64 [| 2 |] [| -1.2; 1.0 |] in
  match Rune.check_grads Nx.Ptree.tensor rosenbrock x with
  | Ok () -> print_endline "reverse mode agrees with finite differences"
  | Error msg -> print_endline msg
```

The check is directional, not per-element: it validates gradients cheaply rather than exhaustively. Use float64 parameters for reliable results; float32 may need a looser `~tol` (the default is `1e-2` relative, with `~eps` the finite-difference step, default `1e-4`).

## Control Flow

Ordinary OCaml control flow — `if`, `match`, loops, recursion — works inside every transformation, because rune runs eagerly and intercepts operations as they execute. The `scan` combinator exists for a different reason: it gives a loop a structure the compiler can see. Under `Rune.jit` a `scan` compiles its fold step once and runs it as a loop in the compiled program, so a recurrence's compile time is independent of its sequence length.

### scan

`scan' ~f ~init xs` folds `f` over the slices of `xs` along axis 0; `f carry x` returns the next carry and a per-step output. The result is the final carry and the outputs stacked along a new axis 0:

```ocaml
let () =
  let xs = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let final, partials =
    Rune.scan'
      ~f:(fun c x ->
        let c = Nx.add c x in
        (c, c))
      ~init:(Nx.scalar Nx.float32 0.0) xs
  in
  Printf.printf "final: %s\n" (Nx.to_string final);
  Printf.printf "%s\n" (Nx.to_string partials)
  (* the running sums [1. 3. 6. 10.] *)
```

`scan c x y ~f ~init xs` is the same fold over structures: one structure each for the carry, the rows and the outputs, in the order of `f`'s type. Every tensor of `xs` has the same leading length, step `i` receives row `i` of every tensor, and every tensor of the outputs is stacked. A fold with nothing to emit passes `Nx.Ptree.unit` for the outputs:

```ocaml
let () =
  let xs = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let (sum, squares), () =
    Rune.scan
      Nx.Ptree.(pair tensor tensor)
      Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun (sum, squares) x ->
        ((Nx.add sum x, Nx.add squares (Nx.mul x x)), ()))
      ~init:(Nx.scalar Nx.float32 0.0, Nx.scalar Nx.float32 0.0)
      xs
  in
  Printf.printf "sum %s, squares %s\n" (Nx.to_string sum)
    (Nx.to_string squares)
```

The carry the step returns must have the visits of the one it received, and each step's outputs those of the first step's: a list keeps its length and an option its presence. `scan` raises otherwise, naming the first path where they differ.

Under `jit` the fold step compiles once and runs as a loop, and `grad` through a jitted scan compiles a reversed loop that replays a record of the step's operations at each step's carry, so the compiled program's size does not depend on the number of steps. `jvp`, `vmap` and `grad` of a scan compile as one loop too. The loop reads row `i` of each leaf of `xs` in place, so data that differs per step belongs in `xs`: a model of stacked layers passes its layer weights, stacked along a leading axis, as rows. Reading them instead from a captured stack with `Nx.D` at a step counter is a gather, and differentiating a captured tensor accumulates a cotangent of its full size on every step, where the cotangent of `xs` is stacked like the outputs, row `i` coming from step `i`.

A compiled function writes the loop out step by step instead when the carry changes its shapes across steps, when the step runs on the host or on devices of two kinds, and inside a `custom_jvp` tangent map under reverse mode. Everywhere outside `jit` the scan is its loop, run where it is written, inside every transformation and `Rune.Total.collect` around it.

Step `i` draws from a key scope of its own, rooted at `Nx.Rng.fold_in k i`, where `k` is one key the scan takes from the scope around at its call and computes at its first draw. The steps draw apart, and they draw the same values eagerly, compiled, batched and transformed. Every scan shifts the draws after it by one key, as a draw does.

### Branches and loops on values

A branch on a value is OCaml's `if` on `Nx.item`, and a loop whose length depends on a value is recursion. Differentiation follows the path taken:

```ocaml
let () =
  let branch x =
    if Nx.item [] (Nx.greater (Nx.sum x) (Nx.scalar Nx.float32 0.0)) then
      Nx.sum (Nx.mul x x)
    else Nx.sum x
  in
  let x = Nx.create Nx.float32 [| 2 |] [| 0.5; 2.0 |] in
  Printf.printf "%s\n" (Nx.to_string (Rune.grad' branch x)) (* [1. 4.] *);

  (* Double the carry until its sum exceeds 10. *)
  let rec double c =
    if Nx.item [] (Nx.less (Nx.sum c) (Nx.scalar Nx.float32 10.0)) then
      double (Nx.mul_s c 2.0)
    else c
  in
  Printf.printf "%s\n"
    (Nx.to_string (double (Nx.create Nx.float32 [| 2 |] [| 1.0; 0.5 |])))
  (* [8. 4.] *)
```

Reading a predicate concretizes it: inside `vmap`, a predicate that depends on the mapped inputs raises, since the lanes could diverge, and inside `jit` one that depends on the arguments raises `Rune.Jit_error`. `Nx.where` selects element by element everywhere.

## Debugging

An interpreter installed with `Nx.Op.intercept` sees every operation a thunk performs. One that prints each operation and evaluates it unchanged is an operation log; installed outermost, it also sees the operations other transformations emit:

```ocaml
let () =
  let f x = Nx.add (Nx.mul x x) (Nx.sin x) in
  let x = Nx.scalar Nx.float32 2.0 in
  let log =
    {
      Nx.Op.run =
        (fun op ->
          Format.eprintf "%a@." Nx.Op.pp op;
          Nx.Op.eval op);
      claims = (fun _ -> true);
    }
  in
  ignore (Nx.Op.intercept log (fun () -> Rune.grad' f x))
  (* logs the forward operations, then the backward-pass operations *)
```

## Autodiff Control

`detach t` is `t` with a zero derivative under every differentiation around the call. It copies nothing: outside every transformation, `detach t` is `t` itself. Use it for baselines, targets and running statistics, and for the input of an operation whose derivative has no definition there. Code outside the function a transformation receives is never differentiated, so evaluating a model needs nothing.

## Limitations

Rune raises where a derivative has no definition, and refuses a tangent map that is not linear:

- **Derivatives with no definition raise.** Every operation has a forward rule, a transpose where it is linear, and a batching rule. The tangent of a complete SVD of a non-square matrix and of a complete QR factorisation of a tall one has no definition and raises `Invalid_argument`, and the vector tangents of repeated eigenvalues or singular values are non-finite. `detach` the input where differentiation should not flow.
- **A tangent map must be linear** under reverse mode (see Custom Differentiation Rules).
- **`Rune.jit` rejects branches on traced values**; a scalar the compiled program would need to branch on cannot be read at trace time.

## Summary

| Transform | Purpose | When to use |
|-----------|---------|-------------|
| `grad` / `grad'` | Gradient of scalar objective | Training loss → parameter gradients |
| `value_and_grad` | Value + gradient together | Avoid a duplicate forward pass |
| `value_and_grad_aux` | ... plus auxiliary data | Return state or metrics from the objective |
| `vjp` | Result and pullback | Non-scalar outputs, several cotangents |
| `jvp` | Jacobian-vector product | Few inputs, many outputs |
| `vmap` / `vmap'` | Vectorize over axis 0 | Per-example computation |
| `jacfwd'` / `jacrev'` | Whole derivative matrices | Small problems, Hessians as `jacfwd' (grad' f)` |
| `remat` | Recompute in the backward pass | Memory-bound backward passes |
| `custom_jvp` / `custom_vjp` | User-defined rules | Stability, speed, opaque interiors |
| `scan` | Structured loops | Recurrences that compile as loops under `jit` |
| `check_grads` | Verify gradients | Testing custom rules and models |
| `detach` | Stop gradient flow | Baselines, targets, unruled ops |
