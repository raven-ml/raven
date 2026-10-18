# kaun

Layers, initializers, losses, metrics and data batching as records and
functions over `Nx.Ptree` structures. There is no layer object and no
trainer: a user writes the model's record and the step.

- kaun depends on nx only. Users apply rune; kaun never calls it.
- No models in the library and no `kaun.models`. Models live in
  `examples/`. A test defines its small model inline; a bench builds its
  stack from layers.
- `kaun.hf` reads Hugging Face formats. Runs on a real model's weights live
  outside `runtest` and say why.
- Examples stay generic.
