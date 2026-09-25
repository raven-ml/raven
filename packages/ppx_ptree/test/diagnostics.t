Every rejected form has a located diagnostic that names the part and says what
to write instead.

  $ for f in cases/*.ml; do echo "== $f"; ./pp.exe -impl $f 2>&1 | grep -o 'ppx_ptree: [^"]*\|Duplicated attribute\|Attribute payload[^"]*'; done
  == cases/abstract.ml
  ppx_ptree: abstract type [t] has no parts to walk
  == cases/alias.ml
  ppx_ptree: Aliases [ty as 'a] have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/any.ml
  ppx_ptree: the wildcard _ at field [value] has no walk
  == cases/applied_parameter.ml
  ppx_ptree: field [l] has type 'a option Linear.t; a derived walk applies a structure to the parameter alone, or a module's [t] to a type without the parameter
  == cases/arrow.ml
  ppx_ptree: Functions have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/bad_alias_arity.ml
  ppx_ptree: Nx.float32_t takes no type argument
  == cases/bad_container_arity.ml
  ppx_ptree: option takes one type argument
  == cases/bad_tensor_arity.ml
  ppx_ptree: Nx.t takes two type arguments
  == cases/bool.ml
  ppx_ptree: field [causal] is a bool; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
  == cases/class.ml
  ppx_ptree: Class types have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/conflicting_attributes.ml
  ppx_ptree: a part takes one of [@ptree.int], [@ptree.skip] and [@ptree.walk]
  ppx_ptree: field [count] is an int; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
  == cases/constraints.ml
  ppx_ptree: type [t] has constraints, which a derived walk does not support
  == cases/dtype.ml
  ppx_ptree: field [dtype] is a dtype, which is data; leave it out with [@ptree.skip]
  == cases/duplicate_attribute.ml
  Duplicated attribute
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == cases/existential.ml
  ppx_ptree: constructor [Pack] has a GADT type, which a derived walk cannot rebuild at another parameter
  ppx_ptree: type variable 'b at argument 1 of [Pack] is not the structure's parameter
  == cases/expansion.ml
  == cases/extensible.ml
  ppx_ptree: extensible variant [t] has no derived walk
  == cases/extension.ml
  ppx_ptree: Extension nodes have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/fixed_data.ml
  ppx_ptree: int has no structure at one type to nest; walk the part with [@ptree.walk f]
  == cases/functor_path.ml
  ppx_ptree: functor applications in type paths have no walk
  == cases/gadt.ml
  ppx_ptree: constructor [Value] has a GADT type, which a derived walk cannot rebuild at another parameter
  == cases/int.ml
  ppx_ptree: field [count] is an int; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
  == cases/int_on_string.ml
  ppx_ptree: [@ptree.int] on field [name], whose type string has no int or bool
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == cases/lazy.ml
  ppx_ptree: field [value] has type Nx.float32_t lazy_t, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/local_fixed.ml
  ppx_ptree: [pair] is applied to Nx.float32_t; a type of this declaration is walked at the parameter only
  == cases/local_open.ml
  ppx_ptree: Locally opened types have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/metadata.ml
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == cases/object.ml
  ppx_ptree: Object types have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/package.ml
  ppx_ptree: First-class modules have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/polymorphic_field.ml
  ppx_ptree: Polymorphic types have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/polymorphic_variant.ml
  ppx_ptree: Polymorphic variants have no derived walk; walk the part with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/private.ml
  ppx_ptree: private type [t] cannot be rebuilt by a derived walk
  == cases/ref.ml
  ppx_ptree: field [value] has type Nx.float32_t ref, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/result.ml
  ppx_ptree: field [value] has type (Nx.float32_t, string) result, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == cases/short_attribute.ml
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == cases/short_int.ml
  ppx_ptree: field [count] is an int; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
  == cases/short_walk.ml
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == cases/skip_parameter.ml
  ppx_ptree: [@ptree.skip] copies field [w], whose type 'a mentions the parameter
  == cases/tensor_parameter.ml
  ppx_ptree: field [w] is a tensor of type (float, 'a) Nx.t, which mentions the parameter; the parameter is a position of its own, walked as a leaf
  == cases/two_parameters.ml
  ppx_ptree: [t] has 2 type parameters; a structure has one, the positions of its tensors
  == cases/type_variable.ml
  ppx_ptree: type variable 'b at field [v] is not the structure's parameter
  == cases/walk_payload.ml
  ppx_ptree: [@ptree.walk] takes the walk as an expression, as in [@ptree.walk M.walk]

A declaration group derives one walk per type, and a structure for each type
without a parameter.

  $ ./pp.exe -impl cases/expansion.ml 2>/dev/null | grep -oE '^    (let( rec)?|and) (walk|ptree)(_[a-z]+)?\b' | sed 's/^ *//'
  let rec walk_helper
  and walk
  and walk_state
  let ptree_state
