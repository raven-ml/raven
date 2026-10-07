# `11-compile-and-batch`

Every jera program runs unchanged under rune. This example differentiates an ODE
solve and an integral, compiles a solve, and maps a system solve over lanes with
a status each.

```bash
dune exec packages/jera/examples/11-compile-and-batch/main.exe
```

## What You'll Learn

- `Rune.grad'` through `Ode.solve` and `Quad.adaptive`
- `Rune.jit` of a solve
- `Rune.vmap` returning a `Solution.t` with `Solution.ptree`

## Key Functions

| Function           | Purpose                |
| ------------------ | ---------------------- |
| `Rune.grad' f x`   | Gradient               |
| `Rune.jit s f`     | Compile                |
| `Rune.vmap s f`    | Map over lanes         |
| `Solution.ptree s` | A solution's structure |
