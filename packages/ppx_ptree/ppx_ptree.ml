(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ppxlib
module B = Ast_builder.Default

(* Attributes *)

let int_core = Attribute.declare_flag "@ptree.int" Attribute.Context.core_type

let int_label =
  Attribute.declare_flag "@ptree.int" Attribute.Context.label_declaration

let skip_core = Attribute.declare_flag "@ptree.skip" Attribute.Context.core_type

let skip_label =
  Attribute.declare_flag "@ptree.skip" Attribute.Context.label_declaration

let walk_core =
  Attribute.declare "@ptree.walk" Attribute.Context.core_type Ast_pattern.__
    Fun.id

let walk_label =
  Attribute.declare "@ptree.walk" Attribute.Context.label_declaration
    Ast_pattern.__ Fun.id

type annotation = Int | Skip | Walk of expression

(* Errors are collected and emitted as located error nodes, so one expansion
   reports all of them. *)

type env = {
  param : string option;
  locals : string list;
  mutable errors : Location.Error.t list;
}

let error env ~loc fmt =
  Format.kasprintf
    (fun msg ->
      env.errors <-
        Location.Error.make ~loc ("ppx_ptree: " ^ msg) ~sub:[] :: env.errors)
    fmt

let catch env f =
  try f ()
  with exn -> (
    match Location.Error.of_exn exn with
    | Some e ->
        env.errors <- e :: env.errors;
        None
    | None -> raise exn)

let annotation env ~loc ~int ~skip ~walk node =
  let flag a = catch env (fun () -> Some (Attribute.has_flag a node)) in
  let found =
    List.concat
      [
        (if flag int = Some true then [ Int ] else []);
        (if flag skip = Some true then [ Skip ] else []);
        (match catch env (fun () -> Attribute.get walk node) with
        | Some (PStr [ { pstr_desc = Pstr_eval (e, _); _ } ]) -> [ Walk e ]
        | Some _ ->
            error env ~loc
              "[@ptree.walk] takes the walk as an expression, as in \
               [@ptree.walk M.walk]";
            []
        | None -> []);
      ]
  in
  match found with
  | [] -> None
  | [ a ] -> Some a
  | _ ->
      error env ~loc
        "a part takes one of [@ptree.int], [@ptree.skip] and [@ptree.walk]";
      None

let core_annotation env ty =
  annotation env ~loc:ty.ptyp_loc ~int:int_core ~skip:skip_core ~walk:walk_core
    ty

let label_annotation env ld =
  annotation env ~loc:ld.pld_loc ~int:int_label ~skip:skip_label
    ~walk:walk_label ld

(* Names *)

let walk_name = function "t" -> "walk" | n -> "walk_" ^ n
let ptree_name = function "t" -> "ptree" | n -> "ptree_" ^ n
let cursor = "ptree__c"
let value = "ptree__x"
let var i = "ptree__" ^ string_of_int i
let ident ~loc name = B.pexp_ident ~loc { loc; txt = Longident.parse name }
let walk_fn ~loc name = ident ~loc ("Nx.Ptree.Walk." ^ name)
let ptree_fn ~loc name = ident ~loc ("Nx.Ptree." ^ name)

let in_module ~loc m name =
  B.pexp_ident ~loc { loc; txt = Longident.Ldot (m, name) }

let show ty = Format.asprintf "%a" Pprintast.core_type ty

(* Types *)

let nx_aliases =
  [
    "float16_t";
    "float32_t";
    "float64_t";
    "bfloat16_t";
    "float8_e4m3_t";
    "float8_e5m2_t";
    "int4_t";
    "uint4_t";
    "int8_t";
    "uint8_t";
    "int16_t";
    "uint16_t";
    "int32_t";
    "uint32_t";
    "int64_t";
    "uint64_t";
    "complex64_t";
    "complex128_t";
    "bool_t";
  ]

let data =
  [
    "float";
    "string";
    "char";
    "unit";
    "bytes";
    "int32";
    "int64";
    "nativeint";
    "exn";
  ]

let containers = [ "ref"; "lazy_t"; "result"; "format4"; "format6" ]

let is_tensor lid args =
  match (lid, args) with
  | Longident.Ldot (Lident "Nx", "t"), _ -> true
  | Ldot (Ldot (Lident "Nx", "Rng"), "key"), _ -> true
  | (Ldot (Lident "Nx", n) | Lident n), _ -> List.mem n nx_aliases
  | _ -> false

let rec mentions env ty =
  match ty.ptyp_desc with
  | Ptyp_var v -> env.param = Some v
  | Ptyp_arrow (_, a, b) -> mentions env a || mentions env b
  | Ptyp_tuple l | Ptyp_constr (_, l) | Ptyp_class (_, l) ->
      List.exists (mentions env) l
  | Ptyp_alias (t, _) | Ptyp_poly (_, t) | Ptyp_open (_, t) -> mentions env t
  | Ptyp_object (fields, _) ->
      List.exists
        (fun f ->
          match f.pof_desc with Otag (_, t) | Oinherit t -> mentions env t)
        fields
  | Ptyp_variant (rows, _, _) ->
      List.exists
        (fun r ->
          match r.prf_desc with
          | Rtag (_, _, l) -> List.exists (mentions env) l
          | Rinherit t -> mentions env t)
        rows
  | Ptyp_package (_, l) -> List.exists (fun (_, t) -> mentions env t) l
  | Ptyp_any | Ptyp_extension _ -> false

let is_param env ty =
  match ty.ptyp_desc with Ptyp_var v -> env.param = Some v | _ -> false

let unsupported env ~loc what =
  error env ~loc
    "%s have no derived walk; walk the part with [@ptree.walk f] or leave it \
     out with [@ptree.skip]"
    what

(* [lambda ~loc pat body] is [fun ptree__c pat -> body]. *)
let lambda ~loc pat body =
  B.pexp_fun ~loc Nolabel None (B.pvar ~loc cursor)
    (B.pexp_fun ~loc Nolabel None pat body)

let rec sequence ~loc lets result =
  match lets with
  | [] -> result
  | (name, e) :: rest ->
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:(B.pvar ~loc name) ~expr:e ]
        (sequence ~loc rest result)

let check_skip env ~loc part ty =
  if mentions env ty then
    error env ~loc
      "[@ptree.skip] copies %s, whose type %s mentions the parameter" part
      (show ty)

let rec has_int ty =
  match ty.ptyp_desc with
  | Ptyp_constr ({ txt = Lident ("int" | "bool"); _ }, []) -> true
  | Ptyp_constr (_, l) | Ptyp_tuple l -> List.exists has_int l
  | _ -> false

let check_int env ~loc part ty =
  if not (has_int ty) then
    error env ~loc "[@ptree.int] on %s, whose type %s has no int or bool" part
      (show ty)

(* [walker env ~ints part ty] is an expression of type [('a, 'b)
   Nx.Ptree.Walk.cursor -> ty -> ty'] that walks a value of [ty], where [ty'] is
   [ty] with the parameter replaced by ['b]. [part] names the part in messages.
   [ints] is [true] under [@ptree.int]. *)
let rec walker env ~ints part ty =
  let loc = ty.ptyp_loc in
  match core_annotation env ty with
  | Some (Walk e) -> e
  | Some Skip ->
      check_skip env ~loc part ty;
      let x = B.pvar ~loc value in
      B.pexp_fun ~loc Nolabel None (B.ppat_any ~loc)
        (B.pexp_fun ~loc Nolabel None x (ident ~loc value))
  | Some Int ->
      check_int env ~loc part ty;
      shape env ~ints:true part ty
  | None -> shape env ~ints part ty

and shape env ~ints part ty =
  let loc = ty.ptyp_loc in
  match ty.ptyp_desc with
  | Ptyp_var v when env.param = Some v -> walk_fn ~loc "leaf"
  | Ptyp_var v ->
      error env ~loc "type variable '%s at %s is not the structure's parameter"
        v part;
      walk_fn ~loc "leaf"
  | Ptyp_tuple tys ->
      let lets =
        List.mapi
          (fun i ty ->
            let loc = ty.ptyp_loc in
            ( var i,
              B.eapply ~loc (walk_fn ~loc "index")
                [
                  ident ~loc cursor;
                  B.eint ~loc i;
                  walker env ~ints part ty;
                  ident ~loc (var i);
                ] ))
          tys
      in
      let names = List.mapi (fun i _ -> var i) tys in
      lambda ~loc
        (B.ppat_tuple ~loc (List.map (B.pvar ~loc) names))
        (sequence ~loc lets (B.pexp_tuple ~loc (List.map (ident ~loc) names)))
  | Ptyp_constr ({ txt = lid; loc = lid_loc }, args) ->
      constr env ~ints part ty lid lid_loc args
  | Ptyp_any ->
      error env ~loc "the wildcard _ at %s has no walk" part;
      walk_fn ~loc "leaf"
  | Ptyp_arrow _ ->
      unsupported env ~loc "Functions";
      walk_fn ~loc "leaf"
  | Ptyp_object _ ->
      unsupported env ~loc "Object types";
      walk_fn ~loc "leaf"
  | Ptyp_class _ ->
      unsupported env ~loc "Class types";
      walk_fn ~loc "leaf"
  | Ptyp_variant _ ->
      unsupported env ~loc "Polymorphic variants";
      walk_fn ~loc "leaf"
  | Ptyp_package _ ->
      unsupported env ~loc "First-class modules";
      walk_fn ~loc "leaf"
  | Ptyp_poly _ ->
      unsupported env ~loc "Polymorphic types";
      walk_fn ~loc "leaf"
  | Ptyp_alias _ ->
      unsupported env ~loc "Aliases [ty as 'a]";
      walk_fn ~loc "leaf"
  | Ptyp_open _ ->
      unsupported env ~loc "Locally opened types";
      walk_fn ~loc "leaf"
  | Ptyp_extension _ ->
      unsupported env ~loc "Extension nodes";
      walk_fn ~loc "leaf"

and constr env ~ints part ty lid lid_loc args =
  let loc = ty.ptyp_loc in
  let leaf () = walk_fn ~loc "leaf" in
  let rec applies = function
    | Longident.Lident _ -> false
    | Ldot (m, _) -> applies m
    | Lapply _ -> true
  in
  match (lid, args) with
  | lid, _ when applies lid ->
      error env ~loc:lid_loc "functor applications in type paths have no walk";
      leaf ()
  | Longident.Ldot (Lident "Nx", "dtype"), _ ->
      error env ~loc
        "%s is a dtype, which is data; leave it out with [@ptree.skip]" part;
      leaf ()
  | lid, args when is_tensor lid args ->
      (match (lid, args) with
      | Ldot (Lident "Nx", "t"), [ _; _ ] -> ()
      | Ldot (Lident "Nx", "t"), _ ->
          error env ~loc "Nx.t takes two type arguments"
      | _, [] -> ()
      | lid, _ ->
          error env ~loc "%s takes no type argument" (Longident.name lid));
      if mentions env ty then
        error env ~loc
          "%s is a tensor of type %s, which mentions the parameter; the \
           parameter is a position of its own, walked as a leaf"
          part (show ty);
      walk_fn ~loc "tensor"
  | Lident (("option" | "list" | "array") as c), args -> (
      match args with
      | [ elt ] -> (
          let w = walker env ~ints part elt in
          match c with
          | "array" -> array ~loc w
          | c -> B.eapply ~loc (walk_fn ~loc c) [ w ])
      | _ ->
          error env ~loc "%s takes one type argument" c;
          leaf ())
  | Lident "int", [] ->
      if not ints then
        error env ~loc
          "%s is an int; report it with [@ptree.int] if a compiled program \
           depends on it, or leave it out with [@ptree.skip]"
          part;
      walk_fn ~loc "int"
  | Lident "bool", [] ->
      if not ints then
        error env ~loc
          "%s is a bool; report it with [@ptree.int] if a compiled program \
           depends on it, or leave it out with [@ptree.skip]"
          part;
      let b = B.pvar ~loc value in
      lambda ~loc b
        (B.eapply ~loc
           (ident ~loc "Stdlib.( <> )")
           [
             B.eapply ~loc (walk_fn ~loc "int")
               [
                 ident ~loc cursor;
                 B.eapply ~loc
                   (ident ~loc "Stdlib.Bool.to_int")
                   [ ident ~loc value ];
               ];
             B.eint ~loc 0;
           ])
  | Lident n, _ when List.mem n containers ->
      error env ~loc
        "%s has type %s, which has no derived walk; walk it with [@ptree.walk \
         f] or leave it out with [@ptree.skip]"
        part (show ty);
      leaf ()
  | Lident n, _ when List.mem n data ->
      error env ~loc
        "%s is a %s, which has no walk; leave it out with [@ptree.skip], or \
         hold data a compiled program depends on in a tensor"
        part n;
      leaf ()
  | Lident n, [] when List.mem n env.locals -> ident ~loc (walk_name n)
  | Lident n, [ a ] when List.mem n env.locals && is_param env a ->
      ident ~loc (walk_name n)
  | Lident n, _ when List.mem n env.locals ->
      error env ~loc
        "[%s] is applied to %s; a type of this declaration is walked at the \
         parameter only"
        n
        (String.concat ", " (List.map show args));
      leaf ()
  | Lident n, [] ->
      B.eapply ~loc (walk_fn ~loc "structure") [ ident ~loc (ptree_name n) ]
  | Lident n, [ a ] when is_param env a -> ident ~loc (walk_name n)
  | Ldot (m, n), [] ->
      B.eapply ~loc (walk_fn ~loc "structure")
        [ in_module ~loc m (ptree_name n) ]
  | Ldot (m, n), [ a ] when is_param env a -> in_module ~loc m (walk_name n)
  | Ldot (m, "t"), [ a ] when not (mentions env a) ->
      B.eapply ~loc (walk_fn ~loc "structure")
        [
          B.eapply ~loc (ptree_fn ~loc "nest")
            [
              B.pexp_pack ~loc (B.pmod_ident ~loc { loc; txt = m }); fixed env a;
            ];
        ]
  | _ ->
      error env ~loc
        "%s has type %s; a derived walk applies a structure to the parameter \
         alone, or a module's [t] to a type without the parameter"
        part (show ty);
      leaf ()

(* [fixed env ty] is the structure at one type of [ty], a type without the
   parameter, for [Walk.structure]. *)
and fixed env ty =
  let loc = ty.ptyp_loc in
  match ty.ptyp_desc with
  | Ptyp_constr ({ txt = lid; _ }, args) when is_tensor lid args ->
      ptree_fn ~loc "tensor"
  | Ptyp_constr ({ txt = Lident "unit"; _ }, []) -> ptree_fn ~loc "unit"
  | Ptyp_constr ({ txt = Lident (("option" | "list") as c); _ }, [ a ]) ->
      B.eapply ~loc (ptree_fn ~loc c) [ fixed env a ]
  | Ptyp_tuple [ a; b ] ->
      B.eapply ~loc (ptree_fn ~loc "pair") [ fixed env a; fixed env b ]
  | Ptyp_constr ({ txt = Ldot (m, n); _ }, []) ->
      in_module ~loc m (ptree_name n)
  | Ptyp_constr ({ txt = Ldot (m, "t"); _ }, [ a ]) ->
      B.eapply ~loc (ptree_fn ~loc "nest")
        [ B.pexp_pack ~loc (B.pmod_ident ~loc { loc; txt = m }); fixed env a ]
  | _ ->
      error env ~loc
        "%s has no structure at one type to nest; walk the part with \
         [@ptree.walk f]"
        (show ty);
      ptree_fn ~loc "tensor"

(* [Walk] has no array walker: an array reports its length with [int] and walks
   its elements at their indices. *)
and array ~loc w =
  let i = "ptree__i" and e = "ptree__e" in
  lambda ~loc (B.pvar ~loc value)
    (B.pexp_sequence ~loc
       (B.pexp_apply ~loc
          (ident ~loc "Stdlib.ignore")
          [
            ( Nolabel,
              B.eapply ~loc (walk_fn ~loc "int")
                [
                  ident ~loc cursor;
                  B.eapply ~loc
                    (ident ~loc "Stdlib.Array.length")
                    [ ident ~loc value ];
                ] );
          ])
       (B.eapply ~loc
          (ident ~loc "Stdlib.Array.mapi")
          [
            B.pexp_fun ~loc Nolabel None (B.pvar ~loc i)
              (B.pexp_fun ~loc Nolabel None (B.pvar ~loc e)
                 (B.eapply ~loc (walk_fn ~loc "index")
                    [ ident ~loc cursor; ident ~loc i; w; ident ~loc e ]));
            ident ~loc value;
          ]))

(* Declarations *)

let field_part name = Printf.sprintf "field [%s]" name

let record env ~loc lds ~get ~build =
  let lets =
    List.map
      (fun ld ->
        let name = ld.pld_name.txt and ty = ld.pld_type in
        let loc = ld.pld_loc in
        let part = field_part name in
        let current = get ~loc name in
        let e =
          match label_annotation env ld with
          | Some Skip ->
              check_skip env ~loc part ty;
              current
          | annot ->
              let w =
                match annot with
                | Some (Walk e) -> e
                | Some Int ->
                    check_int env ~loc part ty;
                    walker env ~ints:true part ty
                | Some Skip | None -> walker env ~ints:false part ty
              in
              B.eapply ~loc (walk_fn ~loc "field")
                [ ident ~loc cursor; B.estring ~loc name; w; current ]
        in
        ("ptree__f_" ^ name, e))
      lds
  in
  let fields =
    List.map
      (fun ld ->
        let loc = ld.pld_loc in
        ( { loc; txt = Longident.Lident ld.pld_name.txt },
          ident ~loc ("ptree__f_" ^ ld.pld_name.txt) ))
      lds
  in
  sequence ~loc lets (build (B.pexp_record ~loc fields None))

let variant env ~loc cds =
  let case cd =
    let loc = cd.pcd_loc and name = cd.pcd_name.txt in
    if cd.pcd_res <> None || cd.pcd_vars <> [] then
      error env ~loc
        "constructor [%s] has a GADT type, which a derived walk cannot rebuild \
         at another parameter"
        name;
    let tag =
      B.eapply ~loc (walk_fn ~loc "case")
        [ ident ~loc cursor; B.estring ~loc name ]
    in
    let lid = { loc; txt = Longident.Lident name } in
    let pat, body =
      match cd.pcd_args with
      | Pcstr_tuple [] -> (None, B.pexp_construct ~loc lid None)
      | Pcstr_tuple [ ty ] ->
          let part = Printf.sprintf "the argument of [%s]" name in
          ( Some (B.pvar ~loc (var 0)),
            sequence ~loc
              [
                ( var 0,
                  B.eapply ~loc
                    (walker env ~ints:false part ty)
                    [ ident ~loc cursor; ident ~loc (var 0) ] );
              ]
              (B.pexp_construct ~loc lid (Some (ident ~loc (var 0)))) )
      | Pcstr_tuple tys ->
          let names = List.mapi (fun i _ -> var i) tys in
          let lets =
            List.mapi
              (fun i ty ->
                let part = Printf.sprintf "argument %d of [%s]" (i + 1) name in
                ( var i,
                  B.eapply ~loc (walk_fn ~loc "index")
                    [
                      ident ~loc cursor;
                      B.eint ~loc i;
                      walker env ~ints:false part ty;
                      ident ~loc (var i);
                    ] ))
              tys
          in
          ( Some (B.ppat_tuple ~loc (List.map (B.pvar ~loc) names)),
            sequence ~loc lets
              (B.pexp_construct ~loc lid
                 (Some (B.pexp_tuple ~loc (List.map (ident ~loc) names)))) )
      | Pcstr_record lds ->
          ( Some (B.pvar ~loc value),
            record env ~loc lds
              ~get:(fun ~loc f ->
                B.pexp_field ~loc (ident ~loc value)
                  { loc; txt = Longident.Lident f })
              ~build:(fun r -> B.pexp_construct ~loc lid (Some r)) )
    in
    B.case
      ~lhs:(B.ppat_construct ~loc lid pat)
      ~guard:None
      ~rhs:(B.pexp_sequence ~loc tag body)
  in
  B.pexp_function_cases ~loc (List.map case cds)

(* [refers names e] is [true] iff [e] mentions one of [names] unqualified. *)
let refers names e =
  let found = ref false in
  object
    inherit Ast_traverse.iter as super

    method! expression e =
      match e.pexp_desc with
      | Pexp_ident { txt = Lident n; _ } when List.mem n names -> found := true
      | _ -> super#expression e
  end
    #expression
    e;
  !found

(* [body env td] is the walk of [td] after its cursor: a function of the value
   being walked. *)
let body env td =
  let loc = td.ptype_loc in
  let fun_value e = B.pexp_fun ~loc Nolabel None (B.pvar ~loc value) e in
  match (td.ptype_kind, td.ptype_manifest) with
  | Ptype_record lds, _ ->
      Some
        (fun_value
           (record env ~loc lds
              ~get:(fun ~loc f ->
                B.pexp_field ~loc (ident ~loc value)
                  { loc; txt = Longident.Lident f })
              ~build:Fun.id))
  | Ptype_variant cds, _ -> Some (variant env ~loc cds)
  | Ptype_abstract, Some ty ->
      let part = Printf.sprintf "type [%s]" td.ptype_name.txt in
      Some
        (fun_value
           (B.eapply ~loc
              (walker env ~ints:false part ty)
              [ ident ~loc cursor; ident ~loc value ]))
  | Ptype_abstract, None ->
      error env ~loc "abstract type [%s] has no parts to walk" td.ptype_name.txt;
      None
  | Ptype_open, _ ->
      error env ~loc "extensible variant [%s] has no derived walk"
        td.ptype_name.txt;
      None

(* A type has at most one parameter: the positions a walk turns from ['a] into
   ['b]. *)
let parameter env td =
  match td.ptype_params with
  | [] -> `None
  | [ ({ ptyp_desc = Ptyp_var v; _ }, _) ] -> `Named v
  | [ ({ ptyp_desc = Ptyp_any; _ }, _) ] -> `Anonymous
  | params ->
      error env ~loc:td.ptype_loc
        "[%s] has %d type parameters; a structure has one, the positions of \
         its tensors"
        td.ptype_name.txt (List.length params);
      `Anonymous

let walk_type ~loc td =
  let name = td.ptype_name.txt in
  let lid = { loc; txt = Longident.Lident name } in
  let cursor =
    B.ptyp_constr ~loc
      { loc; txt = Longident.parse "Nx.Ptree.Walk.cursor" }
      [ B.ptyp_var ~loc "a"; B.ptyp_var ~loc "b" ]
  in
  let self v =
    match td.ptype_params with
    | [] -> B.ptyp_constr ~loc lid []
    | _ -> B.ptyp_constr ~loc lid [ B.ptyp_var ~loc v ]
  in
  B.ptyp_arrow ~loc Nolabel cursor
    (B.ptyp_arrow ~loc Nolabel (self "a") (self "b"))

let ptree_type ~loc td =
  B.ptyp_constr ~loc
    { loc; txt = Longident.parse "Nx.Ptree.t" }
    [ B.ptyp_constr ~loc { loc; txt = Longident.Lident td.ptype_name.txt } [] ]

let ptree_binding ~loc td =
  let name = td.ptype_name.txt in
  let structure =
    B.pmod_structure ~loc
      [
        B.pstr_type ~loc Nonrecursive
          [
            B.type_declaration ~loc ~name:{ loc; txt = "t" }
              ~params:[ (B.ptyp_any ~loc, (NoVariance, NoInjectivity)) ]
              ~cstrs:[] ~kind:Ptype_abstract ~private_:Public
              ~manifest:
                (Some
                   (B.ptyp_constr ~loc { loc; txt = Longident.Lident name } []));
          ];
        B.pstr_value ~loc Nonrecursive
          [
            B.value_binding ~loc ~pat:(B.pvar ~loc "walk")
              ~expr:(ident ~loc (walk_name name));
          ];
      ]
  in
  B.value_binding ~loc
    ~pat:
      (B.ppat_constraint ~loc
         (B.pvar ~loc (ptree_name name))
         (ptree_type ~loc td))
    ~expr:
      (B.eapply ~loc
         (ptree_fn ~loc "instantiate")
         [ B.pexp_pack ~loc structure ])

let errors extension env =
  List.rev_map
    (fun e ->
      let loc = Location.Error.get_location e in
      extension ~loc (Location.Error.to_extension e) [])
    env.errors

let generate_impl ~ctxt (rec_flag, tds) =
  let loc = Expansion_context.Deriver.derived_item_loc ctxt in
  let locals =
    match really_recursive rec_flag tds with
    | Recursive -> List.map (fun td -> td.ptype_name.txt) tds
    | Nonrecursive -> []
  in
  let bindings =
    List.filter_map
      (fun td ->
        let env = { param = None; locals; errors = [] } in
        let param =
          match parameter env td with `Named v -> Some v | _ -> None
        in
        let env = { env with param } in
        let bad_parameters = env.errors <> [] in
        if td.ptype_private = Private then
          error env ~loc:td.ptype_loc
            "private type [%s] cannot be rebuilt by a derived walk"
            td.ptype_name.txt;
        if td.ptype_cstrs <> [] then
          error env ~loc:td.ptype_loc
            "type [%s] has constraints, which a derived walk does not support"
            td.ptype_name.txt;
        let body = if bad_parameters then None else body env td in
        match (env.errors, body) with
        | [], Some body ->
            let loc = td.ptype_loc in
            let c =
              if refers [ cursor ] body then B.pvar ~loc cursor
              else B.ppat_any ~loc
            in
            let body = B.pexp_fun ~loc Nolabel None c body in
            let typ =
              B.ptyp_poly ~loc
                [ { loc; txt = "a" }; { loc; txt = "b" } ]
                (walk_type ~loc td)
            in
            Some
              (Ok
                 ( td,
                   B.value_binding ~loc
                     ~pat:
                       (B.ppat_constraint ~loc
                          (B.pvar ~loc (walk_name td.ptype_name.txt))
                          typ)
                     ~expr:body ))
        | _ -> Some (Error env))
      tds
  in
  match
    List.filter_map (function Error e -> Some e | Ok _ -> None) bindings
  with
  | _ :: _ as failed -> List.concat_map (errors B.pstr_extension) failed
  | [] ->
      let walks =
        List.filter_map (function Ok b -> Some b | Error _ -> None) bindings
      in
      let recursive =
        List.exists
          (fun (_, vb) -> refers (List.map walk_name locals) vb.pvb_expr)
          walks
      in
      let values =
        B.pstr_value ~loc
          (if recursive then Recursive else Nonrecursive)
          (List.map snd walks)
      in
      let ptrees =
        List.filter_map
          (fun (td, _) ->
            match td.ptype_params with
            | [] ->
                Some (B.pstr_value ~loc Nonrecursive [ ptree_binding ~loc td ])
            | _ -> None)
          walks
      in
      values :: ptrees

let generate_intf ~ctxt:_ (_, tds) =
  List.concat_map
    (fun td ->
      let env = { param = None; locals = []; errors = [] } in
      ignore (parameter env td);
      match env.errors with
      | _ :: _ -> errors B.psig_extension env
      | [] ->
          let loc = td.ptype_loc in
          let walk =
            B.psig_value ~loc
              (B.value_description ~loc
                 ~name:{ loc; txt = walk_name td.ptype_name.txt }
                 ~type_:(walk_type ~loc td) ~prim:[])
          in
          if td.ptype_params = [] then
            [
              walk;
              B.psig_value ~loc
                (B.value_description ~loc
                   ~name:{ loc; txt = ptree_name td.ptype_name.txt }
                   ~type_:(ptree_type ~loc td) ~prim:[]);
            ]
          else [ walk ])
    tds

let () =
  Deriving.add "ptree"
    ~str_type_decl:(Deriving.Generator.V2.make_noarg generate_impl)
    ~sig_type_decl:(Deriving.Generator.V2.make_noarg generate_intf)
  |> Deriving.ignore
