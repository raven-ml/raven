# vega

Optimizers as values. An optimizer steps a whole parameter structure (an
`Nx.Ptree`); its state has the parameters' shape and is a structure itself,
so a training step is a pure function of parameters and state that
checkpoints by path and compiles under `Rune.jit` as one program.

- vega depends on nx only. It takes gradients as values; users apply rune.
- A leaf whose dtype is not a float passes through unchanged.
- A new optimizer composes from existing transformations where it can.
