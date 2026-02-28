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

(version 1.0.0~alpha2)

(implicit_transitive_deps false)

(generate_opam_files true)
(opam_file_location inside_opam_directory)

(pin
 (url git+https://github.com/tmattio/mosaic)
 (package
  (name matrix)))

(pin
 (url git+https://github.com/tmattio/mosaic)
 (package
  (name toffee)))

(pin
 (url git+https://github.com/tmattio/mosaic)
 (package
  (name mosaic)))

(pin
 (url git+https://github.com/tmattio/mosaic)
 (package
  (name tree-sitter)))

(package
 (name nx)
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
  (mdx :with-test))
 (tags
  (numerical-computation tensor-library machine-learning)))

(package
 (name brot)
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
  uunf
  uucp
  (windtrap :with-test))
 (tags
  (tokenization bpe wordpiece subword-tokenization language-models)))

(package
 (name talon)
 (synopsis "Dataframes for OCaml")
 (description
  "Fast and elegant dataframes with type-safe operations. Heterogeneous columns, applicative row operations, vectorized aggregations, and CSV I/O, built on Nx.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  re
  (windtrap :with-test))
 (tags
  (dataframe data-manipulation data-science tabular-data)))

(package
 (name rune)
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
  (windtrap :with-test))
 (tags
  (automatic-differentiation machine-learning deep-learning optimization)))

(package
 (name kaun)
 (synopsis "Neural networks for OCaml")
 (description
  "Composable layers, parameter trees, optimizers, training loops, data pipelines, and HuggingFace Hub integration. Built on Rune.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (rune
   (= :version))
  (nx
   (= :version))
  jsont
  (windtrap :with-test))
 (tags
  (neural-networks machine-learning deep-learning)))

(package
 (name kaun-board)
 (synopsis "Training dashboard and logging for Raven")
 (description
  "Lightweight training logger and terminal dashboard for monitoring runs. The core library (kaun-board) provides a Log API for writing JSONL events and a reader for consuming them. The TUI library (kaun-board.tui) provides a Mosaic-based dashboard.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  jsont
  bytesrw
  cmdliner
  mosaic
  (dune-configurator :build)
  matrix)
 (tags
  (training-dashboard monitoring logging machine-learning)))

(package
 (name sowilo)
 (synopsis "Differentiable computer vision for OCaml")
 (description
  "Image processing operations expressed as Nx tensor computations. Geometric transforms, spatial filters, edge detection, morphological operations, and color space conversions, all compatible with Rune.grad and Rune.vmap.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (windtrap :with-test))
 (tags
  (computer-vision image-processing feature-detection machine-learning)))

(package
 (name fehu)
 (synopsis "Reinforcement learning for OCaml")
 (description
  "Type-safe RL environments, observation/action spaces, vectorized environments, trajectory collection, replay buffers, and generalized advantage estimation. Built on Nx.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (windtrap :with-test))
 (tags
  (reinforcement-learning machine-learning environments)))

(package
 (name hugin)
 (synopsis "Plotting for OCaml")
 (description
  "Publication-quality 2D and 3D plots using Cairo rendering. Line plots, scatter, bar charts, contour plots, and image display from Nx tensors.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (dune-configurator :build)
  conf-sdl2
  cairo2
  (nx
   (= :version))
  (windtrap :with-test))
 (tags
  (visualization plotting charts data-science graphics)))

(package
 (name quill)
 (synopsis "Notebooks as markdown files")
 (description
  "Quill turns markdown files into interactive OCaml notebooks. Terminal UI for cell navigation and execution, batch evaluation with watch mode, and in-place output serialization. All Raven packages are pre-loaded.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  cmarkit
  cmdliner
  (windtrap :with-test)
  (mdx :with-test))
 (tags
  (notebooks interactive-computing literate-programming)))

(package
 (name raven)
 (allow_empty)
 (synopsis "Modern scientific computing for OCaml")
 (description
  "Raven is an ecosystem of composable libraries for numerical computing in OCaml. Tensors, automatic differentiation, neural networks, dataframes, plotting, tokenization, computer vision, reinforcement learning, and interactive notebooks.")
 (depends
  (nx
   (= :version))
  (brot
   (= :version))
  (talon
   (= :version))
  (rune
   (= :version))
  (kaun
   (= :version))
  (kaun-board
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
