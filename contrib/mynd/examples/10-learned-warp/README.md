# `10-learned-warp`

Sampling and transforms have derivatives in their positions and
parameters: a mynd function is differentiable where its result is a float
that depends continuously on a float argument. This example finds the
angle between the grey photograph and a turned copy by gradient descent
through sampling.

```bash
dune exec ./main.exe
```

- `Mynd.Sample.resize` shrinks the grey photograph to 75 × 113, and
  `Mynd.Sample.at` with Keys' cubic reads it at its centres turned by an
  angle `a` (`Mynd.Transform.rotation`, `Mynd.Transform.apply`). The
  target is the image read at `a = 0.2`.
- The loss is the mean squared difference of the middle of the two
  images. `Rune.value_and_grad'` gives the loss and its derivative in
  `a`, through the transform's sines and cosines and the sampling
  weights.
- Each step moves `a` by `-0.3` times the derivative. Near the answer
  the loss's curvature in `a` is about 3.8, so each step multiplies the
  angle's error by about `1 - 0.3 × 3.8 = -0.15`; a factor above
  `2 / 3.8`, about 0.53, would make the error grow.

It prints the loss and the angle at the start of steps 1, 10, 20 and 30,
and the angle found, 0.2000. It prints no derivative: near the answer
its sign is rounding's.

Margins: the first loss lies `4.8e-5` from a change of its last digit;
from step 10 the loss is under `7e-7`. The angle at step 10 lies
`4.8e-5` from a change of its last digit, where float32 and float64
runs differ by `5e-8`, and from step 20 the angle is within `2e-8` of
0.2.
