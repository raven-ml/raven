(lang dune 3.19)

(name raven)

(source
 (github raven-ml/raven))

(authors "Thibaut Mattio <thibaut.mattio@gmail.com>")

(maintainers "Thibaut Mattio <thibaut.mattio@gmail.com>")

(license ISC)

(documentation "https://raven-ml.dev/docs/")

(bug_reports "https://github.com/raven-ml/raven/issues")

(using directory-targets 0.1)

(using dune_site 0.1)

(using ctypes 0.3)

(using mdx 0.4)

(version 1.0.0~alpha2)

(implicit_transitive_deps false)

(generate_opam_files true)

; (pin
;  (url git+https://github.com/PataphysicalSociety/soupault#5.0.0)
;  (package
;   (name soupault)))

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
 (url git+https://github.com/invariant-hq/windtrap)
 (package
  (name windtrap)))

; (package
;  (name dev)
;  (allow_empty)
;  (depends odoc odoc-driver soupault lambdasoup landmarks landmarks-ppx))

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
  stdlib-shims
  ; tests
  (windtrap :with-test)
  (mdx :with-test))
 (tags
  (numerical-computation tensor-library machine-learning)))

(package
 (name nx-datasets)
 (synopsis "Common datasets for machine learning")
 (description
  "A collection of common datasets for machine learning tasks, including image classification, regression, and more. This package provides easy access to popular datasets in a format compatible with Nx.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  ocurl
  csv
  logs
  (nx
   (= :version))
  (windtrap :with-test))
 (tags
  (datasets machine-learning data-science image-classification regression)))

(package
 (name saga)
 (synopsis "Text processing and NLP extensions for Nx")
 (description
  "Text processing library that extends Nx with natural language processing capabilities. Provides tokenization, encoding, and text manipulation functionality compatible with the Nx tensor library.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  re
  uucp
  (yojson
   (>= 2.0.0))
  (windtrap :with-test))
 (tags
  (text-processing nlp natural-language-processing tokenization)))

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
  (yojson
   (>= 2.0.0))
  csv
  (windtrap :with-test))
 (tags
  (dataframe data-manipulation data-science tabular-data)))

(package
 (name rune)
 (synopsis "Automatic differentiation and JIT compilation for OCaml")
 (description
  "Rune provides automatic differentiation capabilities and experimental JIT compilation for the Raven ecosystem. It enables gradient-based optimization and supports functional transformations like grad, value_and_grad, and vmap, making it the foundation for deep learning in OCaml.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (dune-configurator :build)
  (conf-llvm-static
   (>= 19))
  (nx
   (= :version))
  (ctypes
   (>= 0.21.0))
  (ctypes-foreign
   (>= 0.21.0))
  ; rune.metal - only on macOS
  (integers
   (= :os macos))
  (windtrap :with-test))
 (tags
  (automatic-differentiation
   jit-compilation
   machine-learning
   deep-learning
   optimization)))

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
  (yojson
   (>= 2.0.0))
  (domainslib
   (>= 0.5.0))
  (saga
   (= :version))
  (rune
   (= :version))
  (nx-datasets
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
  (yojson
   (>= 2.0.0))
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
  ; for hugin.top
  (base64
   (>= 3.1.0))
  (windtrap :with-test))
 (tags
  (visualization plotting charts data-science graphics)))

(package
 (name raven)
 (allow_empty)
 (synopsis "Meta package for the Raven ML ecosystem")
 (description
  "Raven is a comprehensive machine learning ecosystem for OCaml. This meta package installs all Raven components including Nx (tensors), Hugin (plotting), Rune (autodiff), Kaun (neural networks), and Sowilo (computer vision).")
 (depends
  (nx
   (= :version))
  (nx-datasets
   (= :version))
  (saga
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
  ; for raven book
  (mdx :with-test))
 (tags
  (machine-learning data-science numerical-computation)))
