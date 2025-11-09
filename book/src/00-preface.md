# Preface

This book is a tour of modern machine learning through the lens of the Raven ecosystem.  
Raven is a monorepo of OCaml libraries—Nx for tensors, Rune for automatic differentiation, Kaun for neural networks, Talon for data, Saga for text, Sowilo for vision, Fehu for reinforcement learning, and Quill for notebooks.  
Together they aim to make OCaml a first-class language for research and production systems.

We assume you already write software professionally and have passing familiarity with machine learning.  
Our goal is to show how Raven turns OCaml into a practical platform for deep learning, transformers, multimodal models, and reinforcement learning—without switching languages when it is time to ship.

Each chapter mixes narrative, runnable OCaml snippets, and Raven idioms:

- **Narrative** sections explain ideas and design choices.  
- ```ocaml``` blocks contain code that you can execute with `mdx` or in Quill notebooks.  
- **Callouts** highlight performance advice, API gotchas, or upcoming roadmap features.

Before diving in, install the Raven toolchain:

```bash
opam switch create raven ocaml-base-compiler.5.1.0
opam switch set raven
opam install dune mdx
```

We'll introduce the rest of the dependencies as soon as we need them.  
With your environment ready, let’s explore why Raven was built and what makes it compelling for contemporary ML workloads.

