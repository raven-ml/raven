# `02-infix-and-arithmetic`

Element-wise math with operators — the Infix module makes array code read like
algebra.

```bash
dune exec nx/examples/02-infix-and-arithmetic/main.exe
```

## What You'll Learn

- Using the `Infix` module for clean operator-based math
- Scalar arithmetic: `*$`, `+$`, `-$`, `/$` (array op scalar)
- Element-wise operations: `*`, `/`, `+`, `-` (array op array)
- Math functions: `abs`, `sqrt`, `square`, `exp`, `sign`
- Clamping values with `clamp ~min ~max`
- Min-max normalization for scaling data to [0, 1]

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
Celsius:    [0, 20, 37, 100, -40]
Fahrenheit: [32, 68, 98.6, 212, -40]
```

Temperature conversion uses scalar arithmetic — `*$` for multiplication and
`+$` for addition:

```ocaml
let fahrenheit = celsius *$ 1.8 +$ 32.0 in
```

This reads just like the formula `F = C × 1.8 + 32`, but operates on the entire
array at once.

BMI calculation demonstrates element-wise array operations:

```
Heights (m): [1.65, 1.8, 1.72, 1.55]
Weights (kg): [68, 90, 75, 52]
BMI:          [24.977, 27.778, 25.351, 21.639]
```

The formula `BMI = weight / height²` becomes:

```ocaml
let bmi = weight_kg / (height_m * height_m) in
```

Both `*` and `/` work element-by-element, computing BMI for all individuals
simultaneously.

Min-max normalization scales exam scores to the range [0, 1]:

```
Raw scores:  [72, 85, 60, 93, 78, 55]
Normalized:  [0.447, 0.789, 0.132, 1, 0.605, 0]
```

The implementation extracts minimum and maximum values, then applies the
normalization formula:

```ocaml
let lo = min scores in
let hi = max scores in
let normalized = (scores - lo) / (hi - lo) in
```

Math functions apply element-wise to transform arrays:

```
x:       [-2, -1, 0, 1, 2]
abs(x):  [2, 1, 0, 1, 2]
x²:      [4, 1, 0, 1, 4]
√|x|:    [1.414, 1, 0, 1, 1.414]
exp(x):  [0.135, 0.368, 1, 2.718, 7.389]
sign(x): [-1, -1, 0, 1, 1]
```

`clamp` restricts values to a valid range — useful for sensor data or ensuring
inputs stay within bounds:

```
Sensor readings: [-5, 12, 105, 42, -1, 99]
Clamped [0,100]: [0, 12, 100, 42, 0, 99]
```

## Try It

1. Create an array of angles in degrees and convert to radians using `*$`.
2. Build two 1D arrays and compute their element-wise difference, then use
   `abs` to get absolute differences.
3. Generate random-looking data with `create`, then normalize it to [-1, 1]
   instead of [0, 1].

## Next Steps

Continue to [03-indexing-and-slicing](../03-indexing-and-slicing/) to learn how
to extract and modify specific regions of arrays.
