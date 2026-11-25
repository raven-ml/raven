# `02-infix-and-arithmetic`

Element-wise math with operators: inside `Nx.Infix.( … )`, array code reads like
algebra.

```bash
dune exec packages/nx/examples/02-infix-and-arithmetic/main.exe
```

## What You'll Learn

- Writing array math with `Nx`'s operators inside a local open `Nx.Infix.( … )`
- Scalar arithmetic: `*$`, `+$`, `-$`, `/$` (array op scalar)
- Element-wise operations: `*`, `/`, `+`, `-` (array op array)
- Math functions: `abs`, `sqrt`, `square`, `exp`, `sign`
- Clamping values with `clamp ~min ~max`
- Min-max normalization for scaling data to [0, 1]

## Operators and Local Opens

`Nx.Infix` defines `+`, `-`, `*`, `/` and `**` on arrays, their scalar forms
such as `+$`, and the comparisons that return a boolean array, such as `<`
and `=$`. Inside `Nx.Infix.( … )` they apply to arrays; outside, OCaml's
integer operators are unchanged, and `open Nx` leaves them alone too.
`Nx.(Infix.( … ))` reaches `Nx`'s functions as well. Inside such an open,
integer arithmetic is `Stdlib.( + )` (the Operators section of
[`nx.mli`](../../lib/nx.mli)).

## Key Functions

| Function / Operator | Purpose                            |
| ------------------- | ---------------------------------- |
| `a +$ s`            | Add scalar to all elements         |
| `a -$ s`            | Subtract scalar from all elements  |
| `a *$ s`            | Multiply all elements by scalar    |
| `a /$ s`            | Divide all elements by scalar      |
| `a + b`             | Element-wise addition              |
| `a - b`             | Element-wise subtraction           |
| `a * b`             | Element-wise multiplication        |
| `a / b`             | Element-wise division              |
| `abs a`             | Absolute value of each element     |
| `sqrt a`            | Square root of each element        |
| `square a`          | Square each element                |
| `exp a`             | e^x for each element               |
| `sign a`            | Sign of each element (-1, 0, or 1) |
| `clamp ~min ~max a` | Clip values to [min, max]          |
| `min a` / `max a`   | Minimum / maximum element          |

## Output Walkthrough

When you run this example, you'll see various arithmetic operations applied to
arrays:

```
Celsius:    float64 [5] [0, 20, 37, 100, -40]
Fahrenheit: float64 [5] [32, 68, 98.6, 212, -40]
```

Temperature conversion uses scalar arithmetic, `*$` for multiplication and
`+$` for addition:

```ocaml
let fahrenheit = Nx.Infix.((celsius *$ 1.8) +$ 32.0) in
```

This reads like the formula `F = C × 1.8 + 32` and operates on the whole
array. A float prints to at most 8 significant digits, so 37 × 1.8 + 32,
which float64 holds as 98.60000000000001, prints as 98.6.

BMI calculation demonstrates element-wise array operations:

```
Heights (m): float64 [4] [1.65, 1.8, 1.72, 1.55]
Weights (kg): float64 [4] [68, 90, 75, 52]
BMI:          float64 [4] [24.977043, 27.777778, 25.351541, 21.644121]
```

The formula `BMI = weight / height²` becomes:

```ocaml
let bmi = Nx.Infix.(weight_kg / (height_m * height_m)) in
```

Both `*` and `/` work element by element, computing BMI for every individual.

Min-max normalization scales exam scores to the range [0, 1]:

```
Raw scores:  float64 [6] [72, 85, 60, 93, 78, 55]
Normalized:  float64 [6] [0.44736842, 0.78947368, 0.13157895, 1, 0.60526316, 0]
```

The implementation takes the minimum and maximum, 0-d arrays that broadcast
against the scores, then applies the normalization formula:

```ocaml
let lo = Nx.min scores in
let hi = Nx.max scores in
let normalized = Nx.Infix.((scores - lo) / (hi - lo)) in
```

Math functions apply element-wise to transform arrays:

```
x:       float64 [5] [-2, -1, 0, 1, 2]
abs(x):  float64 [5] [2, 1, 0, 1, 2]
x²:      float64 [5] [4, 1, 0, 1, 4]
√|x|:    float64 [5] [1.4142136, 1, 0, 1, 1.4142136]
sign(x): float64 [5] [-1, -1, 0, 1, 1]
```

`clamp` restricts values to a valid range, as for sensor data or inputs that
must stay within bounds:

```
Sensor readings: float64 [6] [-5, 12, 105, 42, -1, 99]
Clamped [0,100]: float64 [6] [0, 12, 100, 42, 0, 99]
```

## Try It

1. Create an array of angles in degrees and convert to radians using `*$`.
2. Build two 1D arrays and compute their element-wise difference, then use
   `abs` to get absolute differences.
3. Build data with `create`, then normalize it to [-1, 1] instead of [0, 1].

## Next Steps

Continue to [03-indexing-and-slicing](../03-indexing-and-slicing/) to learn how
to extract and modify specific regions of arrays.
