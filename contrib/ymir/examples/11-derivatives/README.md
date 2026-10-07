# `11-derivatives`

Every ymir function is a formula of nx operations, so rune differentiates and
compiles it. This example takes gradients of a distance modulus in every field
of a cosmology and of an aperture sum in the aperture's centre and radius.

```bash
dune exec contrib/ymir/examples/11-derivatives/main.exe
```

## What You'll Learn

- Differentiating a cosmology in every field with `Rune.grad`
- Reading a gradient's field per the unit the cosmology holds it in
- Taking a derivative along one field with `Rune.jvp`
- Compiling a function of a cosmology with `Rune.jit`
- Differentiating an aperture sum in the region's centre and radius

## Key Functions

| Function                                | Purpose                                                           |
| --------------------------------------- | ----------------------------------------------------------------- |
| `Rune.grad p f x`                       | A value of `x`'s structure: each field's derivative, per its unit |
| `Rune.jvp p q f x t`                    | The derivative of `f` along the tangent `t`                       |
| `Rune.jit Nx.Ptree.(p @-> returns q) f` | `f` compiled                                                      |
| `Rune.grad' f x`                        | The gradient of a scalar function of a tensor                     |
| `Observation.integrate`                 | Differentiable in the region's placement and size                 |
