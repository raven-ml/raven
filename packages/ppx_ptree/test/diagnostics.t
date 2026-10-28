Every rejected form has a located diagnostic that names the part and says what
to write instead.

  $ for f in support/cases/*.ml; do echo "== $f"; ./pp.exe -impl $f 2>&1 | grep -o 'ppx_ptree: [^"]*\|Duplicated attribute\|Attribute payload[^"]*'; done
  == support/cases/abstract.ml
  ppx_ptree: abstract type [t] has no parts to walk
  == support/cases/alias.ml
  ppx_ptree: field [value] is an alias [ty as 'a], which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/any.ml
  ppx_ptree: field [value] is the wildcard _, which has no walk
  == support/cases/arrow.ml
  ppx_ptree: field [callback] is a function, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/bad_alias_arity.ml
  ppx_ptree: field [value] has type Nx.float32_t; Nx.float32_t takes one type argument, its set
  == support/cases/bad_container_arity.ml
  ppx_ptree: field [value] has type option; option takes one type argument
  == support/cases/bad_tensor_arity.ml
  ppx_ptree: field [value] has type (float, Nx.float32_elt) Nx.t; Nx.t takes three type arguments: the element's OCaml type, its format and the set
  == support/cases/bool.ml
  ppx_ptree: field [causal] is a bool; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
  == support/cases/class.ml
  ppx_ptree: field [value] is a class type, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/conflicting_attributes.ml
  ppx_ptree: a part takes one of [@ptree.int], [@ptree.skip] and [@ptree.walk]
  == support/cases/constraints.ml
  ppx_ptree: type [t] has constraints, which a derived walk does not support
  == support/cases/constructor_attribute.ml
  ppx_ptree: [@ptree.int] on constructor [C], which has 2 arguments; put it on an argument's type, as in [C of (int [@ptree.int]) * ...]
  == support/cases/declaration_attribute.ml
  ppx_ptree: [@@ptree.skip] on type [t]: put the attribute on a field or a part
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == support/cases/dtype.ml
  ppx_ptree: field [dtype] has type (float, Nx.float32_elt) Nx.dtype; a dtype is data; leave it out with [@ptree.skip]
  == support/cases/duplicate_attribute.ml
  Duplicated attribute
  == support/cases/existential.ml
  ppx_ptree: constructor [Pack] has a GADT type, which a derived walk cannot rebuild at other parameters
  == support/cases/expansion.ml
  == support/cases/extensible.ml
  ppx_ptree: extensible variant [t] has no derived walk
  == support/cases/extension.ml
  ppx_ptree: field [value] is an extension node, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/functor_path.ml
  ppx_ptree: field [nested] has type 'a F(X).t; functor applications in type paths have no walk
  == support/cases/gadt.ml
  ppx_ptree: constructor [Value] has a GADT type, which a derived walk cannot rebuild at other parameters
  == support/cases/host_tensor.ml
  ppx_ptree: field [mask] has type Nx.host Nx.bool_t, an array on Nx.host; a fixed array lies on the type's set parameter 'd: write 'd there, or leave the part out with [@ptree.skip] if no program on the set reads it
  == support/cases/int.ml
  ppx_ptree: field [count] is an int; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
  == support/cases/int_on_string.ml
  ppx_ptree: [@ptree.int] on field [name], whose type string has no int or bool
  == support/cases/lazy.ml
  ppx_ptree: field [value] has type 'd Nx.float32_t lazy_t; lazy_t has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/local_open.ml
  ppx_ptree: field [value] is a locally opened type, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/metadata.ml
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == support/cases/missing_alias.ml
  ppx_ptree: field [scale] has type 'd Nx.float8_e4m3_t; Nx.float8_e4m3_t is not a tensor alias ppx_ptree knows: ('v, 's, 'd) Nx.t and aliases such as 'd Nx.float32_t
  == support/cases/nx_index.ml
  ppx_ptree: field [at] has type 'd Nx.index; Nx.index has no walk; walk it with [@ptree.walk f], or leave it out with [@ptree.skip] if its type mentions no parameter
  == support/cases/object.ml
  ppx_ptree: field [value] is an object type, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/one_parameter.ml
  ppx_ptree: field [w] is the set parameter 'a, which holds no value; an array on the set has a tensor type, as 'd Nx.float32_t. A type of one parameter holds fixed arrays only; for positions of a payload write ('a, _) t
  == support/cases/package.ml
  ppx_ptree: field [value] is a first-class module, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/payload_set.ml
  ppx_ptree: field [w] has type 'a Nx.float32_t, whose set is the payload parameter 'a; a position of the payload is 'a alone, and a fixed array lies on the type's set parameter, as in (_, 'd) t and 'd Nx.float32_t
  == support/cases/placement.ml
  ppx_ptree: field [at] has type 'd Nx.Placement.t; Nx.Placement.t has no walk: it says where arrays lie and holds none. Pass it beside the structure, or leave it out with [@ptree.skip] if its type mentions no parameter
  == support/cases/polymorphic_field.ml
  ppx_ptree: field [value] is a polymorphic type, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/polymorphic_variant.ml
  ppx_ptree: field [choice] is a polymorphic variant, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/private.ml
  ppx_ptree: private type [t] cannot be rebuilt by a derived walk
  == support/cases/ref.ml
  ppx_ptree: field [value] has type 'd Nx.float32_t ref; ref has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/result.ml
  ppx_ptree: field [value] has type ('d Nx.float32_t, string) result; result has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
  == support/cases/set_part.ml
  ppx_ptree: field [set] is the set parameter 'd, which holds no value; an array on the set has a tensor type, as 'd Nx.float32_t
  == support/cases/short_attribute.ml
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == support/cases/short_int.ml
  ppx_ptree: field [count] is an int; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
  == support/cases/short_walk.ml
  ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
  == support/cases/skip_parameter.ml
  ppx_ptree: [@ptree.skip] copies field [w], whose type 'a mentions a parameter
  == support/cases/tensor_parameter.ml
  ppx_ptree: field [w] has type (float, 'a, 'd) Nx.t; a tensor's element types cannot mention a parameter; a position of the payload is the payload parameter alone, walked as a leaf
  == support/cases/three_parameters.ml
  ppx_ptree: [t] has 3 type parameters; a structure has at most two, its payload then its set, as ('a, 'd) t
  == support/cases/type_variable.ml
  ppx_ptree: type variable 'b at field [v] is not a parameter of the type
  == support/cases/unnamed_set.ml
  ppx_ptree: field [steps] has type Nx.host Nx.int32_t, an array on Nx.host; a fixed array lies on the type's set parameter: name the set, as in (_, 'd) t and 'd Nx.float32_t, or leave the part out with [@ptree.skip] if no program on the set reads it
  == support/cases/walk_payload.ml
  ppx_ptree: [@ptree.walk] takes the walk as an expression, as in [@ptree.walk M.walk]

A declaration group derives one walk per type, and no description.

  $ ./pp.exe -impl support/cases/expansion.ml 2>/dev/null | grep -oE '^    (let( rec)?|and) (walk|ptree)(_[a-z]+)?\b' | sed 's/^ *//'
  let rec walk_helper
  and walk
  and walk_state
