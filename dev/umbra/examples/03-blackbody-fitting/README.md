# `03-blackbody-fitting`

Fits the effective temperature and luminosity normalization of a star from
synthetic UGRIZ broadband photometry using gradient descent on a blackbody
model.

```bash
dune exec dev/umbra/examples/03-blackbody-fitting/main.exe
```

## What You'll Learn

- Using physical constants (`Const.h_si`, `Const.k_b_si`, `Const.c`)
- Building a differentiable Planck function from Nx tensor operations
- Parameterizing in log-space for numerical stability
- Fitting chi-squared with Rune autodiff and Vega's Adam optimizer

## Key Functions

| Function              | Purpose                                            |
| --------------------- | -------------------------------------------------- |
| `Const.h_si`          | Planck constant (J s)                              |
| `Const.k_b_si`        | Boltzmann constant (J/K)                           |
| `Const.c`             | Speed of light (typed velocity)                    |
| `Unit.to_float`       | Extract scalar SI value from a typed constant      |
| `Rune.value_and_grads`| Compute loss and gradients in one pass             |
| `Vega.adam`           | Adam optimizer                                     |
| `Vega.step`           | Apply one optimization step                        |

## How It Works

The Planck spectral radiance B(lambda, T) = 2hc^2 / lambda^5 / (exp(hc /
lambda k T) - 1) is implemented entirely with Nx tensor operations. Since Rune
can differentiate any Nx computation, gradients of chi-squared with respect to
log(T) and log(A) are computed automatically.

The optimizer starts from T=5000 K and converges toward the true temperature
of 5800 K (Sun-like star). Log-space parameterization ensures positivity and
improves gradient conditioning.

## Try It

1. Change the true temperature to 10000 K (A-type star) and observe how the
   SED shape changes.
2. Add a third parameter for a dust extinction term.
3. Replace the central-wavelength approximation with proper filter integration
   using `Photometry.ab_mag` (see example 05).

## Next Steps

Continue to [04-extinction-and-magnitudes](../04-extinction-and-magnitudes/) to
learn about dust extinction, K-corrections, and magnitude systems.
