# `06-random-numbers`

Implicit RNG with reproducible scopes — generate distributions, sample,
and shuffle. Wrap code in `Rng.run ~seed` for deterministic results.

```bash
dune exec nx/examples/06-random-numbers/main.exe
```

## What You'll Learn

- Generating uniform, normal, and integer distributions
- Running a Monte Carlo simulation to estimate pi
- Creating synthetic training data with controlled noise
- Verifying reproducibility with `Rng.run ~seed`
- Shuffling arrays with `Rng.shuffle`

## Key Functions

| Function                                 | Purpose                                           |
| ---------------------------------------- | ------------------------------------------------- |
| `Rng.run ~seed f`                        | Execute `f` in a deterministic RNG scope          |
| `Rng.uniform dtype shape`                | Uniform random values in [0, 1)                   |
| `Rng.normal dtype shape`                 | Standard normal distribution (mean=0, std=1)      |
| `Rng.randint ~high dtype shape low`      | Random integers in [low, high)                    |
| `Rng.shuffle t`                          | Randomly permute array elements                   |
| `rand dtype shape`                       | Shorthand for uniform random values               |

## Output Walkthrough

### Monte Carlo pi estimation

Drop random points in a unit square. The fraction inside the unit circle
approximates pi/4:

```ocaml
let xs = rand float64 [| n |] in
let ys = rand float64 [| n |] in
let inside = less_s ((xs * xs) + (ys * ys)) 1.0 in
let pi_est = item [] (sum (cast Float64 inside)) *. 4.0 /. Float.of_int n
```

```
Monte Carlo pi (100000 points): 3.1420  (actual: 3.1416)
```

### Reproducibility

Same seed always produces the same numbers:

```ocaml
let a = Rng.run ~seed:99 (fun () -> Rng.normal Float64 [| 3 |]) in
let b = Rng.run ~seed:99 (fun () -> Rng.normal Float64 [| 3 |]) in
(* Identical? true *)
```

## Try It

1. Roll 1000 dice with `Rng.randint` and compute the mean — it should
   approach the theoretical expected value of 3.5.
2. Increase the Monte Carlo sample count to 1,000,000 and observe how the pi
   estimate improves.
3. Generate two clusters of 2D points (one centered at origin, one at (3, 3))
   using `Rng.normal` with offsets.

## Next Steps

Continue to [07-linear-algebra](../07-linear-algebra/) to learn matrix
operations, decompositions, and solving linear systems.
