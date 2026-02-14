# `06-random-numbers`

Deterministic randomness with splittable keys — generate distributions, sample,
and shuffle. Every result is reproducible: same seed, same numbers.

```bash
dune exec nx/examples/06-random-numbers/main.exe
```

## What You'll Learn

- Creating deterministic random keys with `Rng.key`
- Splitting keys into independent streams with `Rng.split`
- Generating uniform, normal, and integer distributions
- Running a Monte Carlo simulation to estimate pi
- Creating synthetic training data with controlled noise
- Verifying reproducibility: same key always gives the same result
- Shuffling arrays with `Rng.shuffle`

## Key Functions

| Function                                 | Purpose                                           |
| ---------------------------------------- | ------------------------------------------------- |
| `Rng.key seed`                           | Create a deterministic random key from an integer |
| `Rng.split ~n key`                       | Split one key into `n` independent subkeys        |
| `Rng.uniform ~key dtype shape`           | Uniform random values in [0, 1)                   |
| `Rng.normal ~key dtype shape`            | Standard normal distribution (mean=0, std=1)      |
| `Rng.randint ~key ~high dtype shape low` | Random integers in [low, high)                    |
| `Rng.shuffle ~key t`                     | Randomly permute array elements                   |
| `rand dtype ~key shape`                  | Shorthand for uniform random values               |

## Output Walkthrough

### Key-based randomness

Keys are explicit — no hidden global state. Split one key into independent
subkeys for different experiments:

```ocaml
let key = Rng.key 42 in
let keys = Rng.split ~n:4 key in
```

### Monte Carlo pi estimation

Drop random points in a unit square. The fraction inside the unit circle
approximates pi/4:

```ocaml
let xs = rand float64 ~key:mc_keys.(0) [| n |] in
let ys = rand float64 ~key:mc_keys.(1) [| n |] in
let inside = less_s ((xs * xs) + (ys * ys)) 1.0 in
let pi_est = item [] (sum (cast Float64 inside)) *. 4.0 /. Float.of_int n
```

```
Monte Carlo pi (100000 points): 3.1420  (actual: 3.1416)
```

### Reproducibility

Same key always produces the same numbers:

```ocaml
let a = Rng.normal ~key:(Rng.key 99) Float64 [| 3 |] in
let b = Rng.normal ~key:(Rng.key 99) Float64 [| 3 |] in
(* Identical? true *)
```

## Try It

1. Roll 1000 dice with `Rng.randint` and compute the mean — it should
   approach the theoretical expected value of 3.5.
2. Increase the Monte Carlo sample count to 1,000,000 and observe how the pi
   estimate improves.
3. Generate two clusters of 2D points (one centered at origin, one at (3, 3))
   using `Rng.normal` with different keys and offsets.

## Next Steps

Continue to [07-linear-algebra](../07-linear-algebra/) to learn matrix
operations, decompositions, and solving linear systems.
