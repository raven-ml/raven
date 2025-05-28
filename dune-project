(lang dune 3.17)

(name raven)

(source
 (github raven-ml/raven))

(authors "Thibaut Mattio <thibaut.mattio@gmail.com>")

(maintainers "Thibaut Mattio <thibaut.mattio@gmail.com>")

(license ISC)

(using directory-targets 0.1)

(using dune_site 0.1)

(using ctypes 0.3)

(version 0.1.0)

(pin
 (url git+https://github.com/PataphysicalSociety/soupault#5.0.0)
 (package
  (name soupault)))

; (package
;  (name dev)
;  (allow_empty)
;  (depends odoc odoc-driver soupault))

(package
 (name nx)
 (synopsis "High-performance N-dimensional array library for OCaml")
 (description
  "Nx is the core component of the Raven ecosystem providing efficient numerical computation with multi-device support. It offers NumPy-like functionality with the benefits of OCaml's type system.")
 (depends
  (ocaml
   (>= 5.0.0))
  dune
  ; nx.io
  camlzip
  (alcotest :with-test))
 (tags
  (numerical-computation tensor-library machine-learning)))

(package
 (name nx-datasets)
 (synopsis "Common datasets for machine learning")
 (description
  "A collection of common datasets for machine learning tasks, including image classification, regression, and more. This package provides easy access to popular datasets in a format compatible with Nx.")
 (depends
  (ocaml
   (>= 5.0.0))
  dune
  camlzip
  ; Currently fails with dune pkg - update when fixed:
  ; This rule defines a directory target "default/.pkg/ocurl/target" that
  ; matches the requested path "default/.pkg/ocurl/target/lib/curl/curl.cma" but
  ; the rule's action didn't produce it
  (ocurl
   (< 0.10.0))
  csv
  (nx
   (= :version)))
 (tags
  (datasets machine-learning data-science image-classification regression)))

(package
 (name rune)
 (synopsis "")
 (description "")
 (depends
  (ocaml
   (>= 5.0.0))
  dune
  (nx
   (= :version))
  ctypes
  (alcotest :with-test)))

(package
 (name sowilo)
 (synopsis "Computer vision extensions for Rune")
 (description
  "Computer vision operations and algorithms built on top of the Rune library. Provides image processing, feature extraction, and other vision-related functionality.")
 (depends
  (ocaml
   (>= 4.14.0))
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
   (>= 4.14.0))
  dune
  dune-site
  cmdliner
  wasm_of_ocaml-compiler
  js_of_ocaml-toplevel
  dream
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
  (hugin
   (= :version))
  (rune
   (= :version))
  (sowilo
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
   (>= 4.14.0))
  dune
  (nx
   (= :version))
  cairo2
  (ctypes
   (>= 0.21.0))
  (ctypes-foreign
   (>= 0.21.0))
  (alcotest :with-test))
 (tags
  (visualization plotting charts data-science graphics)))
