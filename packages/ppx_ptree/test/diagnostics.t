  $ ./pp.exe -impl cases/variant.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: variants are not supported in version 1"]
  $ ./pp.exe -impl cases/variant.ml 2>/dev/null | grep '^ *let map'
  [1]

  $ ./pp.exe -impl cases/metadata.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: metadata type [string] must be annotated [@ptree.ignore]"]

  $ ./pp.exe -impl cases/unknown_qualified.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: cannot infer a traversal for qualified type [Rune.Rng.key]; annotate it [@ptree.leaf], [@ptree.ignore], or [@ptree.using M]"]

  $ ./pp.exe -impl cases/conflicting_attributes.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: a type position may have only one of [@ptree.leaf], [@ptree.ignore], and [@ptree.using]"]

  $ ./pp.exe -impl cases/two_primaries.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: a declaration group may contain only one primary type named [t] or [params]"]
        "ppx_ptree: a declaration group may contain only one primary type named [t] or [params]"]

The short `[@ignore]` spelling is not consumed by this PPX; the real metadata
diagnostic remains, proving that the attribute namespace is exact.

  $ ./pp.exe -impl cases/short_attribute.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: metadata type [string] must be annotated [@ptree.ignore]"]

  $ ./pp.exe -impl cases/arrow.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: function types are not parameter-tree shapes; annotate metadata with [@ptree.ignore]"]

  $ ./pp.exe -impl cases/polymorphic_variant.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: polymorphic variants are not supported in version 1"]

  $ ./pp.exe -impl cases/abstract.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: an abstract implementation has no traversable shape"]

  $ ./pp.exe -impl cases/bad_using.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: [@ptree.using] expects a module path, for example [@ptree.using Params]"]

  $ ./pp.exe -impl cases/functor_path.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: cannot infer a traversal for qualified type [F(X).state]; annotate it [@ptree.leaf], [@ptree.ignore], or [@ptree.using M]"]

Every rejected core-type form has an explicit diagnostic.

  $ ./pp.exe -impl cases/any.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: wildcard type [_] has no derivable parameter-tree shape"]
  $ ./pp.exe -impl cases/type_variable.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: type variable ['a] is not known to be a tensor leaf; add [@ptree.leaf], [@ptree.ignore], or [@ptree.using M]"]
  $ ./pp.exe -impl cases/object.ml 2>&1 | grep 'ppx_ptree:'
    struct [%%ocaml.error "ppx_ptree: object types are not supported"] end
  $ ./pp.exe -impl cases/class.ml 2>&1 | grep 'ppx_ptree:'
  include struct [%%ocaml.error "ppx_ptree: class types are not supported"] end
  $ ./pp.exe -impl cases/polymorphic_field.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: explicitly polymorphic field types are not supported"]
  $ ./pp.exe -impl cases/package.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: first-class module types are not supported"]
  $ ./pp.exe -impl cases/extension.ml 2>&1 | grep 'ppx_ptree:'
    struct [%%ocaml.error "ppx_ptree: extension types are not supported"] end
  $ ./pp.exe -impl cases/local_open.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: locally opened types are not supported"]

Unsupported standard containers are rejected unless the whole value is ignored.

  $ ./pp.exe -impl cases/ref.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: container [ref] is not supported; use an explicit parameter-tree module or [@ptree.ignore]"]
  $ ./pp.exe -impl cases/lazy.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: container [lazy_t] is not supported; use an explicit parameter-tree module or [@ptree.ignore]"]
  $ ./pp.exe -impl cases/result.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: container [result] is not supported; use an explicit parameter-tree module or [@ptree.ignore]"]

Every rejected declaration form has an explicit diagnostic.

  $ ./pp.exe -impl cases/extensible.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: extensible variants are not supported"]
  $ ./pp.exe -impl cases/private.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: private type implementations cannot be derived"]
  $ ./pp.exe -impl cases/reexport_record.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: representation re-exports with records are not supported"]
  $ ./pp.exe -impl cases/reexport_variant.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: variants are not supported in version 1"]
  $ ./pp.exe -impl cases/gadt.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: variants are not supported in version 1"]
  $ ./pp.exe -impl cases/inline_record.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: variants are not supported in version 1"]
  $ ./pp.exe -impl cases/anonymous_parameter.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: anonymous type parameters are not supported"]

The exact attribute namespace and constructor arities are validated.

  $ ./pp.exe -impl cases/short_leaf.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: metadata type [string] must be annotated [@ptree.ignore]"]
  $ ./pp.exe -impl cases/short_using.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: cannot infer a traversal for qualified type [Custom.tree]; annotate it [@ptree.leaf], [@ptree.ignore], or [@ptree.using M]"]
  $ ./pp.exe -impl cases/duplicate_attribute.ml 2>&1 | grep 'Duplicated attribute'
  include struct [%%ocaml.error "Duplicated attribute"] end[@@ocaml.doc
  $ ./pp.exe -impl cases/using_functor.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: [@ptree.using] expects a module path, for example [@ptree.using Params]"]
  $ ./pp.exe -impl cases/bad_tensor_arity.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: tensor type Nx.t must have two type arguments"]
  $ ./pp.exe -impl cases/bad_alias_arity.ml 2>&1 | grep 'ppx_ptree:'
      [%%ocaml.error "ppx_ptree: Nx tensor aliases do not take type arguments"]
  $ ./pp.exe -impl cases/bad_container_arity.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: container types must have exactly one argument"]
  $ ./pp.exe -impl cases/dtype_metadata.ml 2>&1 | grep 'ppx_ptree:'
        "ppx_ptree: [Nx.dtype] is metadata and must be annotated [@ptree.ignore]"]

Expansion exposes only functions, with one primary name and suffixed helpers.

  $ ./pp.exe -impl cases/expansion.ml 2>/dev/null | grep '^ *let \(map\|map2\|iter\)' | awk '{ print $1, $2 }'
  let map_helper
  let map
  let map2_helper
  let map2
  let iter_helper
  let iter
