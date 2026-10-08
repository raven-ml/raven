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

(pin
 (url
  "git+https://github.com/invariant-hq/windtrap.git#7be03df651610949a87ed3bb797984015e9691b9")
 (package
  (name windtrap))
 (package
  (name ppx_windtrap)))

(package
 (name compress)
 (dir otherlibs/compress)
 (synopsis "Deflate, gzip, zlib, Snappy, LZ4 and Zstandard codecs")
 (description
  "Compression codecs with no dependencies: deflate and its zlib and gzip framings, Snappy blocks, and LZ4 and Zstandard decompression. Strings compress and decompress whole; streams go through incremental encoders and decoders in bounded memory; data held in memory, such as a page of a mapped Parquet file, decompresses directly between byte arrays.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  ; tests
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (ppx_windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (thumper :with-test))
 (tags
  (compression deflate gzip zlib snappy lz4 zstd)))

(package
 (name nx)
 (dir packages/nx)
 (license ISC)
 (synopsis "N-dimensional arrays for OCaml")
 (description
  "Nx provides n-dimensional arrays with NumPy-like semantics and OCaml's type safety. 19 data types, broadcasting, slicing, linear algebra, FFT, and I/O. The numerical foundation for the Raven ecosystem.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (compress
   (= :version))
  ; tests
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (ppx_windtrap
   (and
    :with-test
    (>= 0.2.0)))
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
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test)
  ; examples
  (nx
   (and
    (= :version)
    :with-dev-setup)))
 (tags
  (tokenization bpe wordpiece subword-tokenization language-models)))

(package
 (name talon)
 (dir packages/talon)
 (synopsis "Dataframes for OCaml")
 (description
  "Typed columns and expressions whose shape is in their type, queries that fail on their schema before reading data, and CSV and Parquet files read and written in batches, built on Nx.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (compress
   (= :version))
  bytesrw
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (ppx_windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (dataframe data-manipulation data-science tabular-data)))

(package
 (name rune)
 (dir packages/rune)
 (synopsis "Functional transformations for Nx arrays")
 (description
  "Functional transformations for Nx arrays: typed autodiff (grad/vjp/jvp), vmap, and more, built on OCaml effect handlers. Transformations operate over any user-defined parameter structure through the Nx.Ptree.S interface, with full static typing and mixed-dtype parameters in a single pass.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (nx
   (= :version))
  (tolk
   (= :version))
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (ppx_windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (automatic-differentiation machine-learning deep-learning optimization)))

(package
 (name ppx_ptree)
 (dir packages/ppx_ptree)
 (synopsis "Derive the walk of a Raven structure")
 (description
  "Ppx_ptree derives the walk function of Nx.Ptree.S from a type definition: records, variants, tuples, options, lists, tensors of a fixed type and nested structures, with errors located at the source.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (ppxlib
   (>= 0.38.0))
  (nx
   (and
    (= :version)
    :with-test))
  (rune
   (and
    (= :version)
    :with-test))
  (vega
   (and
    (= :version)
    :with-test))
  (kaun
   (and
    (= :version)
    :with-test))
  (windtrap
   (and
    :with-test
    (>= 0.2.0))))
 (tags
  (ppx parameter-tree automatic-differentiation machine-learning)))

(package
 (name kaun)
 (dir packages/kaun)
 (synopsis "Neural networks for OCaml")
 (description
  "Neural networks over Nx, trained with Rune. Models are typed parameter structures with pure apply functions; training steps compose value_and_grad with structural optimizers; there is no opaque trainer or layer type.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (nx
   (= :version))
  logs
  jsont
  bytesrw
  ; rune and vega are not library dependencies: users compose them with
  ; kaun in their own code, as kaun's tests, benches and docs do.
  (rune
   (and
    (= :version)
    :with-test))
  (vega
   (and
    (= :version)
    :with-test))
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test)
  ; examples
  (brot
   (and
    (= :version)
    :with-dev-setup)))
 (tags
  (machine-learning deep-learning neural-networks)))

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
  (nx
   (= :version))
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (ppx_windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (compiler gpu tensor-computation)))

(package
 (name vega)
 (dir packages/vega)
 (synopsis "Gradient-based optimizers for OCaml")
 (description
  "Gradient-descent optimizers over parameter structures (SGD, Adam, AdamW, RAdam, LAMB, LARS, RMSprop, Adagrad, Adan, Lion, Adafactor, L-BFGS), whose states are structures too, and learning-rate schedules. Built on Nx with no autodiff dependency.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  (nx
   (= :version))
  (rune
   (and
    (= :version)
    :with-test))
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (optimization machine-learning gradient-descent)))

(package
 (name jera)
 (dir packages/jera)
 (synopsis "Numerical methods for OCaml")
 (description
  "Numerical methods over Nx tensors: linear and nonlinear systems, minimisation with bounds, quadrature, splines and Chebyshev fits, ordinary, stiff, differential-algebraic, delay and stochastic differential equations, and symplectic splittings. Every solve meets a stated tolerance and reports a status per lane, and Rune differentiates each answer through its own equation.")
 (depends
  (nx
   (= :version))
  (rune
   (= :version))
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (numerical-methods linear-algebra optimization quadrature interpolation differential-equations)))

(package
 (name norn)
 (dir packages/norn)
 (synopsis "Probabilistic inference for OCaml")
 (description
  "Bayesian inference over the user's own structures: distributions, bijectors, a model language of generative functions, NUTS, HMC and ensemble samplers batched over chains, nested sampling and tempered SMC for the evidence, Gaussian approximations, draws and diagnostics. Built on Nx, Rune and Jera.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (rune
   (= :version))
  (jera
   (= :version))
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test))
 (tags
  (bayesian inference mcmc probabilistic-programming)))

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
  cmdliner
  mosaic
  matrix
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  ; examples
  (nx
   (and
    (= :version)
    :with-dev-setup))
  (rune
   (and
    (= :version)
    :with-dev-setup))
  (vega
   (and
    (= :version)
    :with-dev-setup))
  (kaun
   (and
    (= :version)
    :with-dev-setup)))
 (tags
  (experiment-tracking machine-learning monitoring)))

(package
 (name hugin)
 (dir packages/hugin)
 (synopsis "Declarative plotting and visualization for OCaml")
 (description "Composable, beautiful-by-default plotting built on Nx.")
 (depends
  (ocaml
   (>= 5.2.0))
  dune
  (nx
   (= :version))
  (compress
   (= :version))
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (ppx_windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test)
  (thumper :with-test))
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
  matrix
  toffee
  tree-sitter
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (mdx :with-test))
 (tags
  (repl toplevel notebooks interactive-computing literate-programming)))

(package
 (name rig)
 (dir dev/rig)
 (license
  "ISC AND MIT AND NCSA AND Apache-2.0 WITH LLVM-exception AND LicenseRef-amdgpu-firmware")
 (synopsis "Compute devices for OCaml")
 (description
  "Rig drives the host's CPU, Metal, CUDA, AMD and NVIDIA GPUs, files and remote machines as devices: their memory, the order of work across them, and the work itself. AMD and NVIDIA GPUs run through their kernel drivers or directly over PCI.")
 (depends
  (ocaml
   (>= 5.5.0))
  dune
  ; tests
  (windtrap
   (and
    :with-test
    (>= 0.2.0)))
  (thumper :with-test))
 (tags
  (gpu device driver metal cuda amd nvidia)))

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
  (jera
   (= :version))
  (norn
   (= :version))
  (kaun
   (= :version))
  (munin
   (= :version))
  (hugin
   (= :version))
  (quill
   (= :version)))
 (tags
  (machine-learning data-science numerical-computation)))
