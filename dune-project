(lang dune 3.19)

(name raven)

(source
 (github raven-ml/raven))

(authors "Thibaut Mattio <thibaut.mattio@gmail.com>")

(maintainers "Thibaut Mattio <thibaut.mattio@gmail.com>")

(license ISC)

(documentation "https://raven-ml.dev/docs/")

(bug_reports "https://github.com/raven-ml/raven/issues")

(using mdx 0.4)

(version 1.0.0~alpha2)

(implicit_transitive_deps false)

(generate_opam_files true)

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

(pin
 (url git+https://github.com/invariant-hq/windtrap)
 (package
  (name windtrap)))

(package
 (name nx)
 (synopsis "High-performance N-dimensional array library for OCaml")
 (description
  "Nx is the core component of the Raven ecosystem providing efficient numerical computation with multi-device support. It offers NumPy-like functionality with the benefits of OCaml's type system.")
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
 (synopsis "Fast tokenization library for OCaml")
 (description
  "Brot tokenizes text for language models. It provides configurable pipelines with normalization, pre-tokenization, vocabulary-based encoding (BPE, WordPiece, Unigram), post-processing, and decoding. Compatible with HuggingFace tokenizer formats.")
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
 (synopsis "A dataframe library for OCaml")
 (description
  "Talon provides efficient tabular data manipulation with heterogeneous column types, inspired by pandas and polars.")
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
 (synopsis "Automatic differentiation for OCaml")
 (description
  "Rune provides automatic differentiation capabilities for the Raven ecosystem. It enables gradient-based optimization and supports functional transformations like grad, value_and_grad, and vmap, making it the foundation for deep learning in OCaml.")
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
 (synopsis "Flax-inspired neural network library for OCaml")
 (description
  "Kaun brings modern deep learning to OCaml with a flexible, type-safe API for building and training neural networks. It leverages Rune for automatic differentiation and computation graph optimization while maintaining OCaml's functional programming advantages.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  logs
  mosaic
  jsont
  bytesrw
  (domainslib
   (>= 0.5.0))
  (brot
   (= :version))
  (rune
   (= :version))
  (nx
   (= :version))
  (windtrap :with-test))
 (tags
  (neural-networks machine-learning deep-learning)))

(package
 (name sowilo)
 (synopsis "Computer vision extensions for Rune")
 (description
  "Computer vision operations and algorithms built on top of the Rune library. Provides image processing, feature extraction, and other vision-related functionality.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (rune
   (= :version))
  (windtrap :with-test))
 (tags
  (computer-vision image-processing feature-detection machine-learning)))

(package
 (name fehu)
 (synopsis "Reinforcement learning framework for OCaml")
 (description
  "Fehu is a reinforcement learning framework built on Raven's ecosystem, providing environments, algorithms, and training utilities")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (rune
   (= :version))
  (kaun
   (= :version))
  jsont
  bytesrw
  (windtrap :with-test))
 (tags
  (reinforcement-learning machine-learning ai environments agents)))

(package
 (name hugin)
 (synopsis "Visualization library for OCaml")
 (description
  "Hugin is a powerful visualization library for OCaml that produces publication-quality plots and charts. It integrates with the Raven ecosystem to provide visualization of Nx data.")
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
 (synopsis "Interactive notebooks for OCaml")
 (description
  "Quill turns markdown files into interactive OCaml notebooks. It provides a terminal UI for cell navigation and execution, batch evaluation with watch mode, and in-place output serialization. All Raven packages are pre-loaded.")
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
 (synopsis "Meta package for the Raven ML ecosystem")
 (description
  "Raven is a comprehensive machine learning ecosystem for OCaml. This meta package installs all Raven components including Nx (tensors), Hugin (plotting), Rune (autodiff), Kaun (neural networks), and Sowilo (computer vision).")
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
