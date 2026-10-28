# Maps

A map applies a function once per lane: lane `k` receives row `k` of each
array of the arguments, and the lanes' results are stacked along a new
axis 0. The function runs once, over values that stand for every lane. The
Maps section of [`rune.mli`](../lib/rune.mli) states the rules.

## A function of one example, over a batch

`vmap'` maps a function of one array. Here the function normalises a vector
to unit length, and the map applies it to each row of a batch:

```ocaml
let batch = Nx.create Nx.float64 [| 2; 2 |] [| 3.; 4.; 0.; 2. |]

let unit_rows =
  Rune.vmap' (fun v -> Nx.Infix.(v / Nx.sqrt (Nx.sum (v * v)))) batch
```

```ocaml
# let () = Nx.print unit_rows;;
float64 [2; 2]
[[0.6, 0.8],
 [0, 1]]
```

The rows `[3; 4]` and `[0; 2]` have norms `5` and `2`, so the lanes return
`[0.6; 0.8]` and `[0; 1]`, stacked into a `[2; 2]` array.

## Per-example gradients

A map of a gradient gives one gradient per example. The weights `w` are
captured, the same in every lane; each lane differentiates its own
example's loss:

```ocaml
let w = Nx.create Nx.float64 [| 2 |] [| 1.; 2. |]
let xs = Nx.create Nx.float64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |]

let per_example =
  Rune.vmap'
    (fun x -> Rune.grad' (fun w -> Nx.square (Nx.sum Nx.Infix.(w * x))) w)
    xs
```

```ocaml
# let () = Nx.print per_example;;
float64 [3; 2]
[[2, 0],
 [0, 4],
 [6, 6]]
```

The loss of example `x` is `(w · x)²`, whose gradient is `2 (w · x) x`. The
dot products are `1`, `2` and `3`, so the rows are `[2; 0]`, `[0; 4]` and
`[6; 6]`.

## Reading across lanes

A map's lanes are named by an axis. Inside the map of that name, `lanes`
gathers every lane's value and `lane_index` gives the calling lane's
position:

```ocaml
let a = Rune.axis ()

let centred =
  Rune.vmap ~axis:a
    Nx.Ptree.(tensor @-> returns tensor)
    (fun v -> Nx.Infix.(v - Nx.mean ~axes:[ 0 ] (Rune.lanes a v)))
    batch

let positions =
  Rune.vmap ~axis:a
    Nx.Ptree.(tensor @-> returns tensor)
    (fun v -> Rune.lane_index a v)
    batch
```

```ocaml
# let () = Nx.print centred; Nx.print positions;;
float64 [2; 2]
[[1.5, 1],
 [-1.5, -1]]
int32 [2] [0, 1]
```

The mean of the two rows is `[1.5; 3]`, which each lane subtracts from its
row. The lanes' positions are `0` and `1`. A map applied without `~axis`
has a fresh name that no code can read (`axis`, `lanes` and `lane_index` in
the Maps section).

## Jacobians

`jacfwd'` computes every column of a Jacobian in one forward pass, a map of
`jvp'` over the unit tangents; `jacrev'` every row in one backward pass, a
map of the pullback over the unit cotangents (the Functions of one array
section of [`rune.mli`](../lib/rune.mli)):

```ocaml
let h x = Nx.Infix.(x * x * Nx.sum x)
let p = Nx.create Nx.float64 [| 2 |] [| 1.; 2. |]
let forward = Rune.jacfwd' h p
let backward = Rune.jacrev' h p
```

```ocaml
# let () = Nx.print forward; Nx.print backward;;
float64 [2; 2]
[[7, 1],
 [4, 16]]
float64 [2; 2]
[[7, 1],
 [4, 16]]
```

`hᵢ = xᵢ² s` with `s = Σ x`, so `∂hᵢ/∂xⱼ` is `xᵢ²`, plus `2 xᵢ s`
when `i = j`. At
`[1; 2]`, `s = 3` and both give `[[7; 1]; [4; 16]]`.

## Branches per lane

A lane's values have no elements to read, so an OCaml `if` cannot take one
branch per lane. `cond` and `switch` give each lane its own branch's result (the
Branches section of [`rune.mli`](../lib/rune.mli)).

```ocaml
let signs = Nx.create Nx.float64 [| 3 |] [| -2.; 0.5; 4. |]

let safe_log =
  Rune.vmap'
    (fun x -> Rune.cond' (Nx.greater_s x 0.) (fun () -> Nx.log x) (fun () -> x))
    signs
```

```ocaml
# let () = Nx.print safe_log;;
float64 [3] [-2, -0.6931471805599453, 1.3862943611198906]
```

The first lane is not positive and keeps `-2`; the others take `log`.
