# `09-gravitational-lensing`

Fits gravitational lens parameters (lens center and Einstein radius) from
observed image positions of a quadruply-imaged quasar. The point-mass lens
equation is expressed as Nx tensor operations, making the model fully
differentiable through Rune for gradient-based fitting with Adam.

```bash
cd dev/umbra
dune exec --root . examples/09-gravitational-lensing/main.exe
```

## What You'll Learn

- Expressing the gravitational lens equation as differentiable tensor operations
- Minimizing source-plane variance to fit lens parameters
- Fitting physical parameters (lens position, Einstein radius) via Adam optimizer
- Using autodiff gradients with a physics-based loss function

## Key Functions

| Function                | Purpose                                                |
| ----------------------- | ------------------------------------------------------ |
| `Nx.square`             | Squared distances for radial computation               |
| `Nx.sqrt`               | Radial distance from lens center                       |
| `Nx.mean`               | Mean source position across images                     |
| `Rune.value_and_grads`  | Compute loss and gradients for all lens parameters      |
| `Vega.adam`             | Adam optimizer for parameter fitting                   |
| `Vega.step`             | Apply one optimization update                          |

## How It Works

A point-mass gravitational lens deflects light according to the lens equation:
beta = theta - theta_E^2 * theta_hat / |theta|, where beta is the true source
position, theta is the observed image position, and theta_E is the Einstein
radius. If the lens model is correct, all observed images should map back to the
same source position in the source plane.

The example generates synthetic image positions for a quadruply-imaged quasar
with known lens parameters (x_L=0.1, y_L=-0.05, theta_E=1.0) plus small noise.
The loss function maps each image back to the source plane using the current
lens parameters and computes the variance of the inferred source positions. A
correct lens model yields zero variance.

Starting from an initial guess of (x_L=0, y_L=0, theta_E=0.5), the Adam
optimizer runs for 500 steps. Rune differentiates through the entire lens
equation -- including the division by |theta|, the square root, and the
mean/variance -- to provide exact gradients that drive convergence to the true
parameters.

## Try It

1. Increase the noise level from 0.005 to 0.05 and observe how parameter
   uncertainties grow.
2. Add a shear term (gamma_1, gamma_2) to the lens model for external
   tidal perturbation.
3. Replace the point-mass with a singular isothermal sphere (SIS) profile
   where the deflection is constant: alpha = theta_E * theta_hat.

## Next Steps

Continue to [10-uncertainty-propagation](../10-uncertainty-propagation/) to
learn how to automatically propagate parameter uncertainties through
cosmological distance calculations using exact AD Jacobians.
