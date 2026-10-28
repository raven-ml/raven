# `06-random-numbers`

Random numbers from keys: distributions, sampling, and shuffling. Every draw
is a function of a key. A sampler given `~key` draws from that key; one called
without it takes the next key of the enclosing `Rng.with_key` scope, so a
scope opened with `Rng.with_key (Rng.key 42)` gives the same numbers on every
run. Outside any scope, keys are seeded from the system's entropy.

```bash
dune exec packages/nx/examples/06-random-numbers/main.exe
```

## What You'll Learn

- Generating uniform, normal, and integer distributions
- Running a Monte Carlo simulation to estimate pi
- Creating synthetic training data with controlled noise
- Verifying reproducibility with `Rng.with_key`
- Drawing from explicit keys made with `Rng.split`
- Shuffling arrays with `Rng.permutation` and `take`

## Key Functions

Every sampler lives in `Nx.Rng` and takes an optional `?key`.

| Function                          | Purpose                                        |
| --------------------------------- | ---------------------------------------------- |
| `Rng.with_key k f`                | Run `f` in a scope rooted at `k`               |
| `Rng.key seed`                    | The key for a seed                             |
| `Rng.split ~n k`                  | `n` independent keys derived from `k`          |
| `Rng.uniform dtype shape`         | Uniform values in [0, 1)                       |
| `Rng.normal dtype shape`          | Standard normal (mean 0, variance 1)           |
| `Rng.randint ?low ~high shape`    | Uniform int32 values in [low, high)            |
| `Rng.permutation n`               | A random ordering of `0`, …, `n - 1`           |
| `Rng.poisson rates`               | One Poisson count per element of `rates`       |

## Output Walkthrough

### Monte Carlo pi estimation

Drop random points in a unit square. The fraction inside the unit circle
approximates pi/4, and `count` counts the points where the mask holds:

```ocaml
let xs = Nx.Rng.uniform Nx.float64 [| n |] in
let ys = Nx.Rng.uniform Nx.float64 [| n |] in
let inside = Nx.(less_s ((xs * xs) + (ys * ys)) 1.0) in
let count = Int64.to_float (Nx.item [] (Nx.count inside)) in
let pi_est = count *. 4.0 /. Float.of_int n
```

```
Monte Carlo pi (100000 points): 3.1321  (actual: 3.1416)
```

The estimate differs from run to run, since these draws come from no scope.

### Reproducibility

The same seed always produces the same numbers:

```ocaml
let draw () = Nx.Rng.normal Nx.float64 [| 3 |] in
let a = Nx.Rng.with_key (Nx.Rng.key 99) draw in
let b = Nx.Rng.with_key (Nx.Rng.key 99) draw in
(* Identical? true *)
```

### Explicit keys

A key is spent once used. `Rng.split` derives independent keys from one, and
each draw names its key:

```ocaml
let keys = Nx.Rng.split ~n:2 (Nx.Rng.key 7) in
let c = Nx.Rng.normal ~key:keys.(0) Nx.float64 [| 3 |] in
let d = Nx.Rng.normal ~key:keys.(1) Nx.float64 [| 3 |] in
```

### Shuffling

A shuffle gathers the rows at a random permutation of their positions:

```ocaml
let shuffled = Nx.take (Nx.Rng.permutation 8) data in
```

```
Original:  int32 [8] [0, 1, 2, 3, 4, 5, 6, 7]
Shuffled:  int32 [8] [2, 6, 3, 7, 1, 5, 4, 0]
```

## Try It

1. Roll 1000 dice with `Rng.randint` and compute the mean: it should approach
   the expected value of 3.5.
2. Increase the Monte Carlo sample count to 1,000,000 and observe how the pi
   estimate improves.
3. Generate two clusters of 2D points (one centered at origin, one at (3, 3))
   using `Rng.normal` with offsets.

## Next Steps

Continue to [07-linear-algebra](../07-linear-algebra/) to learn matrix
operations, decompositions, and solving linear systems.
