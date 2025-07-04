(lang dune 3.19)

(name raven)

(source
 (github raven-ml/raven))

(authors "Thibaut Mattio <thibaut.mattio@gmail.com>")

(maintainers "Thibaut Mattio <thibaut.mattio@gmail.com>")

(license ISC)

(using directory-targets 0.1)

(using dune_site 0.1)

(using ctypes 0.3)

(using mdx 0.4)

(version 1.0.0~alpha0)

(implicit_transitive_deps false)

(generate_opam_files true)

(pin
 (url git+https://github.com/PataphysicalSociety/soupault#5.0.0)
 (package
  (name soupault)))

(package
 (name dev)
 (allow_empty)
 (depends odoc odoc-driver soupault lambdasoup landmarks landmarks-ppx))

(package
 (name nx)
 (synopsis "High-performance N-dimensional array library for OCaml")
 (description
  "Nx is the core component of the Raven ecosystem providing efficient numerical computation with multi-device support. It offers NumPy-like functionality with the benefits of OCaml's type system.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  ; nx.io
  camlzip
  (dune-configurator :build)
  ; nx.metal - only on macOS
  (ctypes
   (and
    (>= 0.21.0)
    (= :os macos)))
  (ctypes-foreign
   (and
    (>= 0.21.0)
    (= :os macos)))
  (integers
   (= :os macos))
  ; tests
  (alcotest :with-test)
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
  camlzip
  ocurl
  csv
  logs
  (nx
   (= :version))
  (alcotest :with-test))
 (tags
  (datasets machine-learning data-science image-classification regression)))

(package
 (name nx-text)
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
  uutf
  uucp
  (alcotest :with-test))
 (tags
  (text-processing nlp natural-language-processing tokenization)))

(package
 (name rune)
 (synopsis "Automatic differentiation and JIT compilation for OCaml")
 (description
  "Rune provides automatic differentiation capabilities and experimental JIT compilation for the Raven ecosystem. It enables gradient-based optimization and supports functional transformations like grad, value_and_grad, and vmap, making it the foundation for deep learning in OCaml.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (ctypes
   (>= 0.21.0))
  ; rune.jit.metal
  (ctypes-foreign
   (and
    (>= 0.21.0)
    (= :os macos)))
  (alcotest :with-test)))

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
  (rune
   (= :version))
  (nx-datasets
   (= :version))
  (alcotest :with-test))
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
  (alcotest :with-test))
 (tags
  (computer-vision image-processing feature-detection machine-learning)))

(package
 (name quill)
 (allow_empty)
 (synopsis "Interactive notebook for OCaml data science")
 (description
  "Quill is an interactive notebook application for data exploration, prototyping, and knowledge sharing in OCaml. It provides a Jupyter-like experience with rich visualization and documentation capabilities.")
 (sites
  (lib toplevel_libs))
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (dune-site
   (>= 3.19.0))
  cmdliner
  wasm_of_ocaml-compiler
  js_of_ocaml-toplevel
  (dream
   (>= 1.0.0~alpha8))
  ppx_deriving_yojson
  crunch
  cmarkit
  vdom
  brr
  base64
  (nx
   (= :version))
  (nx-datasets
   (= :version))
  (nx-text
   (= :version))
  (rune
   (= :version))
  (kaun
   (= :version))
  (sowilo
   (= :version))
  (hugin
   (= :version))
  (alcotest :with-test))
 (tags
  (notebook interactive-computing data-science literate-programming)))

(package
 (name hugin)
 (synopsis "Visualization library for OCaml")
 (description
  "Hugin is a powerful visualization library for OCaml that produces publication-quality plots and charts. It integrates with the Raven ecosystem to provide visualization of Nx data.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  conf-sdl2
  cairo2
  (nx
   (= :version))
  (alcotest :with-test))
 (tags
  (visualization plotting charts data-science graphics)))

(package
 (name xla)
 (synopsis "OCaml bindings to XLA (Accelerated Linear Algebra)")
 (description
  "OCaml bindings to Google's XLA compiler for machine learning computations. XLA optimizes linear algebra computations for various hardware accelerators including GPUs and TPUs.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  ctypes
  ctypes-foreign
  (dune-configurator :build)
  (alcotest :with-test))
 (tags
  (xla compiler machine-learning accelerated-computing)))

(package
 (name raven)
 (allow_empty)
 (synopsis "Meta package for the Raven ML ecosystem")
 (description
  "Raven is a comprehensive machine learning ecosystem for OCaml. This meta package installs all Raven components including Nx (tensors), Hugin (plotting), Quill (notebooks), Rune (autodiff), Kaun (neural networks), and Sowilo (computer vision).")
 (depends
  (nx
   (= :version))
  (nx-datasets
   (= :version))
  (nx-text
   (= :version))
  (rune
   (= :version))
  (kaun
   (= :version))
  (sowilo
   (= :version))
  (quill
   (= :version))
  (hugin
   (= :version)))
 (tags
  (machine-learning data-science numerical-computation)))
