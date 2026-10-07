# Norn Examples

Each example teaches one part of norn on a small problem and runs in seconds.
Start with `01-distributions` and work through them in order.

| Example | Concept |
|---------|---------|
| [`01-distributions`](./01-distributions/) | Distributions |
| [`02-bijectors`](./02-bijectors/) | Bijectors |
| [`03-nuts`](./03-nuts/) | The No-U-Turn sampler |
| [`04-models`](./04-models/) | Models as generative functions |
| [`05-hmc-and-ensemble`](./05-hmc-and-ensemble/) | HMC and ensemble sampling |
| [`06-diagnostics`](./06-diagnostics/) | Diagnostics and findings |
| [`07-evidence`](./07-evidence/) | Evidence |
| [`08-laplace`](./08-laplace/) | Gaussian approximations |
| [`09-calibration`](./09-calibration/) | Simulation-based calibration |

Run one from the repository root:

```bash
dune exec packages/norn/examples/01-distributions/main.exe
```
