(lang dune 3.20)

(name nx-oxcaml)

(generate_opam_files true)

(using oxcaml 0.1)

(source
 (github raven-ml/raven))

(authors "Thibaut Mattio")

(maintainers "Thibaut Mattio <thibaut.mattio@gmail.com>")

(license ISC)

(pin
 (url ../../)
 (package
  (name nx)))

(package
 (name nx-oxcaml)
 (synopsis "High-performance Nx backend using OxCaml's unboxed types")
 (description
  "An experimental backend for Nx that leverages OxCaml's unboxed types for improved performance.")
 (depends
  (ocaml-variants
   (= 5.2.0+ox))
  dune
  nx))
