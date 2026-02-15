# `05-reductions-and-statistics`

Summarize data with reductions — means, variances, and aggregations along any
axis. This example analyzes daily temperature readings across four cities.

```bash
dune exec nx/examples/05-reductions-and-statistics/main.exe
```

## What You'll Learn

- Reducing along specific axes with `mean`, `std`, `sum`
- Finding extremes and their positions with `min`, `max`, `argmax`
- Computing running totals with `cumsum`
- Preserving dimensions for broadcasting with `keepdims`
- Detecting outliers using z-score normalization
- Testing conditions with `all` and `any`

## Key Functions

| Function                      | Purpose                                     |
| ----------------------------- | ------------------------------------------- |
| `mean ~axes t`                | Average values along specified axes         |
| `std ~axes t`                 | Standard deviation along axes               |
| `min t` / `max t`             | Global minimum / maximum                    |
| `min ~axes t` / `max ~axes t` | Per-axis minimum / maximum                  |
| `argmax ~axis t`              | Index of the maximum along an axis          |
| `cumsum ~axis t`              | Cumulative sum along an axis                |
| `all t` / `any t`             | Test if all / any elements are true         |
| `greater_s t s`               | Element-wise `t > s` returning a bool array |
| `less_s t s`                  | Element-wise `t < s` returning a bool array |

## Output Walkthrough

The dataset is a 4×7 matrix — 4 cities, 7 days of temperature readings:

```ocaml
let city_means = mean ~axes:[ 1 ] temps in
```

```
City averages:
  Paris       mean=22.9  std=2.3
  Cairo       mean=32.0  std=2.1
  Helsinki    mean=-5.6  std=2.6
  London      mean=14.9  std=1.3
```

### Axis semantics

- `~axes:[1]` reduces across columns (days) → one value per city
- `~axes:[0]` reduces across rows (cities) → one value per day
- No axis → reduces everything to a scalar

### Outlier detection with z-scores

Using `keepdims:true` to broadcast the mean and std against the original data:

```ocaml
let mu = mean ~axes:[ 1 ] ~keepdims:true temps in
let sigma = std ~axes:[ 1 ] ~keepdims:true temps in
let z_scores = (temps - mu) / sigma in
let outlier_mask = greater_s (abs z_scores) 1.5
```

### Condition testing

```ocaml
let all_above_zero = all (greater_s temps 0.0) in    (* false — Helsinki *)
let any_below_neg5 = any (less_s temps (-5.0)) in     (* true  — Helsinki *)
```

## Try It

1. Compute the daily average across all cities with `mean ~axes:[0]` and find
   which day was warmest on average.
2. Use `cumsum ~axis:1` on the full temperature matrix to see running totals
   per city.
3. Find the day with the smallest temperature range across cities using
   `max ~axes:[0]` minus `min ~axes:[0]`.

## Next Steps

Continue to [06-random-numbers](../06-random-numbers/) to generate synthetic
data with controlled, reproducible distributions.
