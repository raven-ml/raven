(lang dune 3.21)

(name raven)

(source
 (github raven-ml/raven))

(authors "Thibaut Mattio <thibaut.mattio@gmail.com>")

(maintainers "Thibaut Mattio <thibaut.mattio@gmail.com>")

(license ISC)

(documentation "https://raven-ml.dev/docs/")

(bug_reports "https://github.com/raven-ml/raven/issues")

(using mdx 0.4)

(using directory-targets 0.1)

(version 1.0.0~alpha3)

(implicit_transitive_deps false)

(generate_opam_files true)

(opam_file_location inside_opam_directory)

(pin
 (url "git+https://github.com/invariant-hq/thumper.git")
 (package
  (name thumper)))

(package
 (name nx)
 (dir packages/nx)
 (synopsis "N-dimensional arrays for OCaml")
 (description
  "Nx provides n-dimensional arrays with NumPy-like semantics and OCaml's type safety. 19 data types, broadcasting, slicing, linear algebra, FFT, and I/O. The numerical foundation for the Raven ecosystem.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (dune-configurator :build)
  (conf-pkg-config :build)
  ; camlzip
  (conf-zlib :build)
  logs
  ; tests
  (windtrap :with-test)
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (numerical-computation tensor-library machine-learning)))

(package
 (name brot)
 (dir packages/brot)
 (synopsis "Tokenization for OCaml")
 (description
  "Fast, HuggingFace-compatible tokenization for language models. BPE, WordPiece, Unigram, word-level, and character-level algorithms with composable pipelines and training from scratch.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  re
  jsont
  bytesrw
  (uunf
   (>= 15.1.0))
  uucp
  (windtrap :with-test)
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (tokenization bpe wordpiece subword-tokenization language-models)))

(package
 (name talon)
 (dir packages/talon)
 (synopsis "Dataframes for OCaml")
 (description
  "Fast and elegant dataframes with type-safe operations. Heterogeneous columns, applicative row operations, vectorized aggregations, and CSV I/O, built on Nx.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (windtrap :with-test)
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (dataframe data-manipulation data-science tabular-data)))

(package
 (name rune)
 (dir packages/rune)
 (synopsis "Functional transformations for Nx arrays")
 (description
  "Automatic differentiation and vectorizing maps for Nx tensors. Reverse-mode AD (grad, vjp), forward-mode AD (jvp), vmap, and gradient checking, built on OCaml 5 effect handlers.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (dune-configurator :build)
  (nx
   (= :version))
  (tolk
   (= :version))
  (windtrap :with-test)
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (automatic-differentiation machine-learning deep-learning optimization)))

(package
 (name tolk)
 (dir packages/tolk)
 (synopsis "A minimal ML compiler for GPU tensor computation")
 (description
  "Tolk is a minimal, readable ML compiler for GPU tensor computation in the Raven ecosystem.")
 (depends
  (ocaml
   (>= 5.2))
  dune
  (windtrap :with-test)
  (thumper :with-test))
 (tags
  (compiler gpu tensor-computation)))

(package
 (name norn)
 (dir packages/norn)
 (synopsis "MCMC sampling for OCaml")
 (description
  "Markov chain Monte Carlo samplers with automatic gradients via Rune. Hamiltonian Monte Carlo with dual-averaging step-size adaptation.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (rune
   (= :version))
  (windtrap :with-test)
  (thumper :with-test))
 (tags
  (mcmc sampling bayesian machine-learning)))

(package
 (name vega)
 (dir packages/vega)
 (synopsis "Per-parameter gradient-based optimizers for OCaml")
 (description
  "Typed, per-parameter optimizer primitives: Adam, AdamW, SGD, RMSprop, Adagrad, and learning-rate schedules. Built on Nx with no autodiff dependency.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (windtrap :with-test)
  (thumper :with-test))
 (tags
  (optimization machine-learning gradient-descent)))

(package
 (name kaun)
 (dir packages/kaun)
 (synopsis "Neural networks for OCaml")
 (description
  "Composable layers, parameter trees, optimizers, training loops, data pipelines, and HuggingFace Hub integration. Built on Rune.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (rune
   (= :version))
  (vega
   (= :version))
  (nx
   (= :version))
  jsont
  bytesrw
  (windtrap :with-test)
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (neural-networks machine-learning deep-learning)))

(package
 (name munin)
 (dir packages/munin)
 (synopsis "Local experiment tracking for Raven")
 (description
  "Local-first experiment tracking with append-only event logs, versioned artifacts, a terminal dashboard, and a CLI. The core library (munin) provides Session, Run, Store, and Artifact modules. The TUI library (munin.tui) provides a Mosaic-based dashboard.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  jsont
  bytesrw
  sha
  cmdliner
  mosaic
  (dune-configurator :build)
  matrix
  (windtrap :with-test))
 (tags
  (experiment-tracking machine-learning monitoring)))

(package
 (name sowilo)
 (dir packages/sowilo)
 (synopsis "Differentiable computer vision for OCaml")
 (description
  "Image processing operations expressed as Nx tensor computations. Geometric transforms, spatial filters, edge detection, morphological operations, and color space conversions, all compatible with Rune.grad and Rune.vmap.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (windtrap :with-test)
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (computer-vision image-processing feature-detection machine-learning)))

(package
 (name fehu)
 (dir packages/fehu)
 (synopsis "Reinforcement learning for OCaml")
 (description
  "Type-safe RL environments, observation/action spaces, vectorized environments, trajectory collection, replay buffers, and generalized advantage estimation. Built on Nx.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (windtrap :with-test)
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (reinforcement-learning machine-learning environments)))

(package
 (name hugin)
 (dir packages/hugin)
 (synopsis "Declarative plotting and visualization for OCaml")
 (description "Composable, beautiful-by-default plotting built on Nx.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (dune-configurator :build)
  (conf-sdl2 :build)
  (conf-cairo :build)
  (nx
   (= :version))
  (windtrap :with-test)
  (mdx :with-test))
 (tags
  (visualization plotting charts data-science graphics)))

(package
 (name quill)
 (dir packages/quill)
 (synopsis "Interactive REPL and markdown notebooks")
 (description
  "Quill is a REPL and notebook environment for OCaml. Interactive toplevel with syntax highlighting, completion, and history. Markdown notebooks with a terminal UI, web frontend, batch evaluation, and watch mode.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  cmarkit
  cmdliner
  bytesrw
  jsont
  mosaic
  (windtrap :with-test)
  (mdx :with-test))
 (tags
  (repl toplevel notebooks interactive-computing literate-programming)))

(package
 (name raven)
 (allow_empty)
 (dir packages/raven)
 (synopsis "Modern scientific computing for OCaml")
 (description
  "Raven is an ecosystem of composable libraries for numerical computing in OCaml. Tensors, automatic differentiation, neural networks, dataframes, plotting, tokenization, computer vision, reinforcement learning, and interactive notebooks.")
 (depends
  (nx
   (= :version))
  (tolk
   (= :version))
  (brot
   (= :version))
  (talon
   (= :version))
  (rune
   (= :version))
  (vega
   (= :version))
  (kaun
   (= :version))
  (munin
   (= :version))
  (sowilo
   (= :version))
  (fehu
   (= :version))
  (hugin
   (= :version))
  (quill
   (= :version)))
 (tags
  (machine-learning data-science numerical-computation)))
