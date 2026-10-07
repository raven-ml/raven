# `11-derivatives`

Every ymir function is a formula of nx operations, so rune differentiates and
compiles it. This example takes gradients of a distance modulus in cosmological
parameters and of an aperture sum in the aperture's centre and radius.

```bash
dune exec contrib/ymir/examples/11-derivatives/main.exe
```

## What You'll Learn

- Differentiating a cosmology function with `Rune.grad'`
- Compiling it with `Rune.jit'`
- Differentiating an aperture sum in the region's centre and radius

## Key Functions

| Function                | Purpose                                           |
| ----------------------- | ------------------------------------------------- |
| `Rune.grad' f x`        | The gradient of a scalar function of a tensor     |
| `Rune.jit' f`           | `f` compiled                                      |
| `Observation.integrate` | Differentiable in the region's placement and size |
