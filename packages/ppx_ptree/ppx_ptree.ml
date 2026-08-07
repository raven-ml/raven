(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ppxlib
module B = Ast_builder.Default
module Int_set = Set.Make (Int)
module String_map = Map.Make (String)
module String_set = Set.Make (String)

let leaf_core = Attribute.declare_flag "@ptree.leaf" Attribute.Context.core_type

let leaf_label =
  Attribute.declare_flag "@ptree.leaf" Attribute.Context.label_declaration

let ignore_core =
  Attribute.declare_flag "@ptree.ignore" Attribute.Context.core_type

let ignore_label =
  Attribute.declare_flag "@ptree.ignore" Attribute.Context.label_declaration

let using_core =
  Attribute.declare "@ptree.using" Attribute.Context.core_type
    Ast_pattern.(single_expr_payload __)
    Fun.id

let using_label =
  Attribute.declare "@ptree.using" Attribute.Context.label_declaration
    Ast_pattern.(single_expr_payload __)
    Fun.id

type names = { map : string; map2 : string; iter : string }

type shape = { desc : shape_desc; loc : Location.t }

and shape_desc =
  | Leaf
  | Ignored
  | Tuple of shape list
  | Option of shape
  | List of shape
  | Array of shape
  | Local of string
  | Using of Longident.t

type body = Alias of shape | Record of (label_declaration * shape) list

type declaration = {
  type_decl : type_declaration;
  body : body option;
  dependencies : String_set.t;
}

type annotation =
  | Leaf_annotation
  | Ignore_annotation
  | Using_annotation of Longident.t

type validation = { mutable errors_rev : Location.Error.t list }

let shape ~loc desc = { desc; loc }

let names_for_type = function
  | "t" | "params" -> { map = "map"; map2 = "map2"; iter = "iter" }
  | name ->
      { map = "map_" ^ name; map2 = "map2_" ^ name; iter = "iter_" ^ name }

let add_error validation ~loc fmt =
  Format.kasprintf
    (fun message ->
      validation.errors_rev <-
        Location.Error.make ~loc message ~sub:[] :: validation.errors_rev)
    fmt

let get_flag validation attribute node =
  try Attribute.has_flag attribute node
  with exception_ -> (
    match Location.Error.of_exn exception_ with
    | Some error ->
        validation.errors_rev <- error :: validation.errors_rev;
        false
    | None -> Stdlib.raise exception_)

let get_attribute validation attribute node =
  try Attribute.get attribute node
  with exception_ -> (
    match Location.Error.of_exn exception_ with
    | Some error ->
        validation.errors_rev <- error :: validation.errors_rev;
        None
    | None -> Stdlib.raise exception_)

let rec longident_parts = function
  | Longident.Lident name -> Some [ name ]
  | Ldot (parent, name) ->
      Option.map (fun parts -> parts @ [ name ]) (longident_parts parent)
  | Lapply _ -> None

let longident_of_parts = function
  | [] -> invalid_arg "Ppx_ptree.longident_of_parts: empty path"
  | first :: rest ->
      List.fold_left
        (fun path name -> Longident.Ldot (path, name))
        (Longident.Lident first) rest

let module_path_of_expression validation expression =
  let loc = expression.pexp_loc in
  let path =
    match expression.pexp_desc with
    | Pexp_construct (path, None) | Pexp_ident path -> Some path.txt
    | _ -> None
  in
  match path with
  | Some path when Option.is_some (longident_parts path) -> Some path
  | Some _ ->
      add_error validation ~loc
        "ppx_ptree: [@ptree.using] does not accept functor application paths";
      None
  | None ->
      add_error validation ~loc
        "ppx_ptree: [@ptree.using] expects a module path, for example \
         [@ptree.using Params]";
      None

let annotation_at_core validation core_type =
  let annotations = ref [] in
  if get_flag validation leaf_core core_type then
    annotations := Leaf_annotation :: !annotations;
  if get_flag validation ignore_core core_type then
    annotations := Ignore_annotation :: !annotations;
  Option.iter
    (fun expression ->
      Option.iter
        (fun path -> annotations := Using_annotation path :: !annotations)
        (module_path_of_expression validation expression))
    (get_attribute validation using_core core_type);
  !annotations

let annotation_at_label validation label =
  let annotations = ref [] in
  if get_flag validation leaf_label label then
    annotations := Leaf_annotation :: !annotations;
  if get_flag validation ignore_label label then
    annotations := Ignore_annotation :: !annotations;
  Option.iter
    (fun expression ->
      Option.iter
        (fun path -> annotations := Using_annotation path :: !annotations)
        (module_path_of_expression validation expression))
    (get_attribute validation using_label label);
  !annotations

let select_annotation validation ~loc annotations =
  match annotations with
  | [] -> None
  | [ annotation ] -> Some annotation
  | _ ->
      add_error validation ~loc
        "ppx_ptree: a type position may have only one of [@ptree.leaf], \
         [@ptree.ignore], and [@ptree.using]";
      None

let nx_aliases =
  String_set.of_list
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

let metadata_types =
  String_set.of_list
    [
      "unit";
      "bool";
      "char";
      "string";
      "bytes";
      "int";
      "int32";
      "int64";
      "nativeint";
      "float";
    ]

let unsupported_containers = String_set.of_list [ "lazy_t"; "ref"; "result" ]

let is_path path expected =
  match longident_parts path with
  | Some parts -> parts = expected
  | None -> false

let is_nx_alias path =
  match longident_parts path with
  | Some [ name ] -> String_set.mem name nx_aliases
  | Some [ "Nx"; name ] -> String_set.mem name nx_aliases
  (* An RNG key is a tensor, and it belongs in a parameter structure whenever a
     jitted step draws from it. Its name does not end in [t], so the qualified
     fallback below cannot recognise it. *)
  | Some [ "Rng"; "key" ] | Some [ "Nx"; "Rng"; "key" ] -> true
  | _ -> false

let container_kind path =
  match longident_parts path with
  | Some [ "option" ]
  | Some [ "Option"; "t" ]
  | Some [ "Stdlib"; "Option"; "t" ] ->
      Some `Option
  | Some [ "list" ] | Some [ "List"; "t" ] | Some [ "Stdlib"; "List"; "t" ] ->
      Some `List
  | Some [ "array" ] | Some [ "Array"; "t" ] | Some [ "Stdlib"; "Array"; "t" ]
    ->
      Some `Array
  | _ -> None

let local_name path =
  match path with
  | Longident.Lident name -> Some name
  | Ldot _ | Lapply _ -> None

let external_primary path =
  match longident_parts path with
  | Some parts -> (
      match List.rev parts with
      | ("t" | "params") :: reversed_module ->
          Some (longident_of_parts (List.rev reversed_module))
      | _ -> None)
  | None -> None

let rec classify validation ?label core_type =
  let label_annotations =
    match label with
    | None -> []
    | Some label -> annotation_at_label validation label
  in
  let annotations =
    label_annotations @ annotation_at_core validation core_type
  in
  let error_loc =
    match label with None -> core_type.ptyp_loc | Some label -> label.pld_loc
  in
  match select_annotation validation ~loc:error_loc annotations with
  | Some Leaf_annotation -> shape ~loc:core_type.ptyp_loc Leaf
  | Some Ignore_annotation -> shape ~loc:core_type.ptyp_loc Ignored
  | Some (Using_annotation path) -> shape ~loc:core_type.ptyp_loc (Using path)
  | None -> classify_unannotated validation core_type

and classify_unannotated validation core_type =
  let loc = core_type.ptyp_loc in
  match core_type.ptyp_desc with
  | Ptyp_tuple elements ->
      shape ~loc (Tuple (List.map (classify validation) elements))
  | Ptyp_constr (path, arguments) ->
      classify_constructor validation ~loc path.txt arguments
  | Ptyp_alias (inner, _) -> classify validation inner
  | Ptyp_any ->
      add_error validation ~loc
        "ppx_ptree: wildcard type [_] has no derivable parameter-tree shape";
      shape ~loc Ignored
  | Ptyp_var name ->
      add_error validation ~loc
        "ppx_ptree: type variable ['%s] is not known to be a tensor leaf; add \
         [@ptree.leaf], [@ptree.ignore], or [@ptree.using M]"
        name;
      shape ~loc Ignored
  | Ptyp_arrow _ ->
      add_error validation ~loc
        "ppx_ptree: function types are not parameter-tree shapes; annotate \
         metadata with [@ptree.ignore]";
      shape ~loc Ignored
  | Ptyp_object _ ->
      add_error validation ~loc "ppx_ptree: object types are not supported";
      shape ~loc Ignored
  | Ptyp_class _ ->
      add_error validation ~loc "ppx_ptree: class types are not supported";
      shape ~loc Ignored
  | Ptyp_variant _ ->
      add_error validation ~loc
        "ppx_ptree: polymorphic variants are not supported in version 1";
      shape ~loc Ignored
  | Ptyp_poly _ ->
      add_error validation ~loc
        "ppx_ptree: explicitly polymorphic field types are not supported";
      shape ~loc Ignored
  | Ptyp_package _ ->
      add_error validation ~loc
        "ppx_ptree: first-class module types are not supported";
      shape ~loc Ignored
  | Ptyp_extension _ ->
      add_error validation ~loc "ppx_ptree: extension types are not supported";
      shape ~loc Ignored
  | Ptyp_open _ ->
      add_error validation ~loc
        "ppx_ptree: locally opened types are not supported";
      shape ~loc Ignored

and classify_constructor validation ~loc path arguments =
  match container_kind path with
  | Some kind -> (
      match arguments with
      | [ argument ] -> (
          let child = classify validation argument in
          match kind with
          | `Option -> shape ~loc (Option child)
          | `List -> shape ~loc (List child)
          | `Array -> shape ~loc (Array child))
      | _ ->
          add_error validation ~loc
            "ppx_ptree: container types must have exactly one argument";
          shape ~loc Ignored)
  | None when is_path path [ "Nx"; "t" ] || is_path path [ "Nx_effect"; "t" ] ->
      if List.length arguments <> 2 then
        add_error validation ~loc
          "ppx_ptree: tensor type %a must have two type arguments"
          Pprintast.longident path;
      shape ~loc Leaf
  | None when is_nx_alias path ->
      if arguments <> [] then
        add_error validation ~loc
          "ppx_ptree: Nx tensor aliases do not take type arguments";
      shape ~loc Leaf
  | None -> (
      match longident_parts path with
      | Some [ name ] when String_set.mem name metadata_types ->
          add_error validation ~loc
            "ppx_ptree: metadata type [%s] must be annotated [@ptree.ignore]"
            name;
          shape ~loc Ignored
      | Some [ name ] when String_set.mem name unsupported_containers ->
          add_error validation ~loc
            "ppx_ptree: container [%s] is not supported; use an explicit \
             parameter-tree module or [@ptree.ignore]"
            name;
          shape ~loc Ignored
      | Some [ "Nx"; "dtype" ] ->
          add_error validation ~loc
            "ppx_ptree: [Nx.dtype] is metadata and must be annotated \
             [@ptree.ignore]";
          shape ~loc Ignored
      | _ -> (
          match local_name path with
          | Some name -> shape ~loc (Local name)
          | None -> (
              match external_primary path with
              | Some module_path -> shape ~loc (Using module_path)
              | None ->
                  add_error validation ~loc
                    "ppx_ptree: cannot infer a traversal for qualified type \
                     [%a]; annotate it [@ptree.leaf], [@ptree.ignore], or \
                     [@ptree.using M]"
                    Pprintast.longident path;
                  shape ~loc Ignored)))

let rec dependencies shape =
  match shape.desc with
  | Leaf | Ignored | Using _ -> String_set.empty
  | Local name -> String_set.singleton name
  | Tuple shapes ->
      List.fold_left
        (fun result shape -> String_set.union result (dependencies shape))
        String_set.empty shapes
  | Option shape | List shape | Array shape -> dependencies shape

let validate_type_parameters validation type_decl =
  List.iter
    (fun (parameter, _) ->
      match parameter.ptyp_desc with
      | Ptyp_var _ -> ()
      | _ ->
          add_error validation ~loc:parameter.ptyp_loc
            "ppx_ptree: anonymous type parameters are not supported")
    type_decl.ptype_params

let validate_primary_owner validation type_declarations =
  let primaries =
    List.filter
      (fun type_decl ->
        type_decl.ptype_name.txt = "t" || type_decl.ptype_name.txt = "params")
      type_declarations
  in
  match primaries with
  | [] | [ _ ] -> ()
  | _ ->
      List.iter
        (fun type_decl ->
          add_error validation ~loc:type_decl.ptype_name.loc
            "ppx_ptree: a declaration group may contain only one primary type \
             named [t] or [params]")
        primaries

let validate_declaration ~signature validation type_decl =
  validate_type_parameters validation type_decl;
  if (not signature) && type_decl.ptype_private = Private then
    add_error validation ~loc:type_decl.ptype_name.loc
      "ppx_ptree: private type implementations cannot be derived";
  let body =
    match (type_decl.ptype_kind, type_decl.ptype_manifest) with
    | Ptype_record labels, None ->
        Some
          (Record
             (List.map
                (fun label ->
                  (label, classify validation ~label label.pld_type))
                labels))
    | Ptype_abstract, Some manifest ->
        Some (Alias (classify validation manifest))
    | Ptype_abstract, None when signature -> None
    | Ptype_abstract, None ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: an abstract implementation has no traversable shape";
        None
    | Ptype_record _, Some _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: representation re-exports with records are not supported";
        None
    | Ptype_variant _, _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: variants are not supported in version 1";
        None
    | Ptype_open, _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: extensible variants are not supported";
        None
  in
  let dependencies =
    match body with
    | None -> String_set.empty
    | Some (Alias shape) -> dependencies shape
    | Some (Record fields) ->
        List.fold_left
          (fun result (_, shape) ->
            String_set.union result (dependencies shape))
          String_set.empty fields
  in
  { type_decl; body; dependencies }

let validate ~signature type_declarations =
  let validation = { errors_rev = [] } in
  validate_primary_owner validation type_declarations;
  let declarations =
    List.map (validate_declaration ~signature validation) type_declarations
  in
  let errors = List.rev validation.errors_rev in
  (declarations, errors)

let deduplicate_errors errors =
  let seen = Hashtbl.create (List.length errors) in
  List.filter
    (fun error ->
      let location = Location.Error.get_location error in
      let key =
        ( location.loc_start.pos_fname,
          location.loc_start.pos_cnum,
          location.loc_end.pos_cnum,
          Location.Error.message error )
      in
      if Hashtbl.mem seen key then false
      else (
        Hashtbl.add seen key ();
        true))
    errors

let structure_errors errors =
  List.map
    (fun error ->
      let loc = Location.Error.get_location error in
      B.pstr_extension ~loc (Location.Error.to_extension error) [])
    (deduplicate_errors errors)

let signature_errors errors =
  List.map
    (fun error ->
      let loc = Location.Error.get_location error in
      B.psig_extension ~loc (Location.Error.to_extension error) [])
    (deduplicate_errors errors)

let located_lid ~loc path = { txt = path; loc }
let lident ~loc name = located_lid ~loc (Longident.Lident name)
let append_lid path name = Longident.Ldot (path, name)
let stdlib_operator name = Longident.Ldot (Lident "Stdlib", name)
let ident ~loc path = B.pexp_ident ~loc (located_lid ~loc path)
let evar ~loc name = B.evar ~loc name
let pvar ~loc name = B.pvar ~loc name

let apply ~loc function_ arguments =
  B.pexp_apply ~loc function_
    (List.map (fun expression -> (Nolabel, expression)) arguments)

let call ~loc path arguments = apply ~loc (ident ~loc path) arguments

let let_one ~loc name expression body =
  B.pexp_let ~loc Nonrecursive
    [ B.value_binding ~loc ~pat:(pvar ~loc name) ~expr:expression ]
    body

let lets ~loc bindings body =
  List.fold_right
    (fun (name, expression) body -> let_one ~loc name expression body)
    bindings body

let lets_located bindings body =
  List.fold_right
    (fun (loc, name, expression) body -> let_one ~loc name expression body)
    bindings body

let construct ~loc name argument =
  B.pexp_construct ~loc (lident ~loc name) argument

let construct_pattern ~loc name argument =
  B.ppat_construct ~loc (lident ~loc name) argument

let invalid_argument ~loc message =
  call ~loc
    (Longident.Ldot (Lident "Stdlib", "invalid_arg"))
    [ B.estring ~loc message ]

let callback_type ~loc operation variable_a variable_b =
  let variable name = B.ptyp_var ~loc name in
  let leaf =
    B.ptyp_constr ~loc
      (located_lid ~loc (Longident.Ldot (Lident "Nx", "t")))
      [ variable variable_a; variable variable_b ]
  in
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  let body =
    match operation with
    | `Map -> arrow leaf leaf
    | `Map2 -> arrow leaf (arrow leaf leaf)
    | `Iter -> arrow leaf (B.ptyp_constr ~loc (lident ~loc "unit") [])
  in
  B.ptyp_poly ~loc [ { txt = variable_a; loc }; { txt = variable_b; loc } ] body

let used_type_variables type_decl =
  List.fold_left
    (fun used (parameter, _) ->
      match parameter.ptyp_desc with
      | Ptyp_var name -> String_set.add name used
      | _ -> used)
    String_set.empty type_decl.ptype_params

let fresh_callback_variables type_decl =
  let used = used_type_variables type_decl in
  let rec choose prefix index =
    let candidate =
      if index = 0 then prefix else prefix ^ string_of_int index
    in
    if String_set.mem candidate used then choose prefix (index + 1)
    else candidate
  in
  let first = choose "ptree_a" 0 in
  let second =
    let used = String_set.add first used in
    let rec choose_second index =
      let candidate =
        if index = 0 then "ptree_b" else "ptree_b" ^ string_of_int index
      in
      if String_set.mem candidate used then choose_second (index + 1)
      else candidate
    in
    choose_second 0
  in
  (first, second)

let declared_type type_decl =
  B.ptyp_constr ~loc:type_decl.ptype_name.loc
    (lident ~loc:type_decl.ptype_name.loc type_decl.ptype_name.txt)
    (List.map fst type_decl.ptype_params)

let constrained_parameter ~loc name type_ =
  B.ppat_constraint ~loc (pvar ~loc name) type_

let function_expression ~loc ~callback_type ~input_types body =
  let callback = constrained_parameter ~loc "f" callback_type in
  let inputs =
    List.mapi
      (fun index type_ ->
        constrained_parameter ~loc (if index = 0 then "x" else "y") type_)
      input_types
  in
  List.fold_right
    (fun pattern body -> B.pexp_fun ~loc Nolabel None pattern body)
    (callback :: inputs) body

let field ~loc expression label =
  B.pexp_field ~loc expression
    (located_lid ~loc (Longident.Lident label.pld_name.txt))

let tuple_bindings ~loc expression count =
  let names = List.init count (fun _ -> gen_symbol ~prefix:"ptree_tuple" ()) in
  let pattern = B.ppat_tuple ~loc (List.map (pvar ~loc) names) in
  (names, pattern, expression)

let rec map_shape callback shape expression =
  let loc = shape.loc in
  match shape.desc with
  | Leaf -> call ~loc (Lident callback) [ expression ]
  | Ignored -> expression
  | Local name ->
      call ~loc (Lident (names_for_type name).map)
        [ evar ~loc callback; expression ]
  | Using module_path ->
      call ~loc
        (append_lid module_path "map")
        [ evar ~loc callback; expression ]
  | Tuple shapes ->
      let names, pattern, tuple =
        tuple_bindings ~loc expression (List.length shapes)
      in
      let mapped_names =
        List.map (fun _ -> gen_symbol ~prefix:"ptree_mapped" ()) shapes
      in
      let mapped =
        List.map2
          (fun shape name ->
            map_shape callback shape (evar ~loc:shape.loc name))
          shapes names
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr:tuple ]
        (lets ~loc
           (List.combine mapped_names mapped)
           (B.pexp_tuple ~loc (List.map (evar ~loc) mapped_names)))
  | Option shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      B.pexp_match ~loc expression
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc value)))
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some (map_shape callback shape (evar ~loc:shape.loc value))));
        ]
  | List shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let mapper =
        B.pexp_fun ~loc Nolabel None (pvar ~loc value)
          (map_shape callback shape (evar ~loc:shape.loc value))
      in
      call ~loc (Longident.parse "Stdlib.List.map") [ mapper; expression ]
  | Array shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let mapper =
        B.pexp_fun ~loc Nolabel None (pvar ~loc value)
          (map_shape callback shape (evar ~loc:shape.loc value))
      in
      call ~loc (Longident.parse "Stdlib.Array.map") [ mapper; expression ]

let path_name = function [] -> "<root>" | parts -> String.concat "." parts

let mismatch_message function_path kind path =
  Format.sprintf "%s: %s mismatch at %s" function_path kind (path_name path)

let rec map2_shape ~function_path ~path callback shape left right =
  let loc = shape.loc in
  match shape.desc with
  | Leaf -> call ~loc (Lident callback) [ left; right ]
  | Ignored -> left
  | Local name ->
      call ~loc (Lident (names_for_type name).map2)
        [ evar ~loc callback; left; right ]
  | Using module_path ->
      call ~loc
        (append_lid module_path "map2")
        [ evar ~loc callback; left; right ]
  | Tuple shapes ->
      let left_names, left_pattern, left_tuple =
        tuple_bindings ~loc left (List.length shapes)
      in
      let right_names, right_pattern, right_tuple =
        tuple_bindings ~loc right (List.length shapes)
      in
      let mapped_names =
        List.map (fun _ -> gen_symbol ~prefix:"ptree_mapped" ()) shapes
      in
      let mapped =
        List.mapi
          (fun index shape ->
            map2_shape ~function_path
              ~path:(path @ [ string_of_int index ])
              callback shape
              (evar ~loc:shape.loc (List.nth left_names index))
              (evar ~loc:shape.loc (List.nth right_names index)))
          shapes
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:left_pattern ~expr:left_tuple ]
        (B.pexp_let ~loc Nonrecursive
           [ B.value_binding ~loc ~pat:right_pattern ~expr:right_tuple ]
           (lets ~loc
              (List.combine mapped_names mapped)
              (B.pexp_tuple ~loc (List.map (evar ~loc) mapped_names))))
  | Option shape ->
      let left_value = gen_symbol ~prefix:"ptree_left" () in
      let right_value = gen_symbol ~prefix:"ptree_right" () in
      let pair = B.pexp_tuple ~loc [ left; right ] in
      B.pexp_match ~loc pair
        [
          B.case
            ~lhs:
              (B.ppat_tuple ~loc
                 [
                   construct_pattern ~loc "None" None;
                   construct_pattern ~loc "None" None;
                 ])
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:
              (B.ppat_tuple ~loc
                 [
                   construct_pattern ~loc "Some" (Some (pvar ~loc left_value));
                   construct_pattern ~loc "Some" (Some (pvar ~loc right_value));
                 ])
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some
                    (map2_shape ~function_path ~path callback shape
                       (evar ~loc:shape.loc left_value)
                       (evar ~loc:shape.loc right_value))));
          B.case ~lhs:(B.ppat_any ~loc) ~guard:None
            ~rhs:
              (invalid_argument ~loc
                 (mismatch_message function_path "option constructor" path));
        ]
  | List shape -> map2_list ~loc ~function_path ~path callback shape left right
  | Array shape ->
      map2_array ~loc ~function_path ~path callback shape left right

and map2_list ~loc ~function_path ~path callback shape left right =
  let left_name = gen_symbol ~prefix:"ptree_left" () in
  let right_name = gen_symbol ~prefix:"ptree_right" () in
  let left_length = gen_symbol ~prefix:"ptree_left_length" () in
  let right_length = gen_symbol ~prefix:"ptree_right_length" () in
  let left_value = gen_symbol ~prefix:"ptree_left_value" () in
  let right_value = gen_symbol ~prefix:"ptree_right_value" () in
  let mapper =
    B.pexp_fun ~loc Nolabel None (pvar ~loc left_value)
      (B.pexp_fun ~loc Nolabel None (pvar ~loc right_value)
         (map2_shape ~function_path ~path callback shape
            (evar ~loc:shape.loc left_value)
            (evar ~loc:shape.loc right_value)))
  in
  lets ~loc
    [
      (left_name, left);
      (right_name, right);
      ( left_length,
        call ~loc (Longident.parse "Stdlib.List.length") [ evar ~loc left_name ]
      );
      ( right_length,
        call ~loc
          (Longident.parse "Stdlib.List.length")
          [ evar ~loc right_name ] );
    ]
    (B.pexp_ifthenelse ~loc
       (call ~loc (stdlib_operator "<>")
          [ evar ~loc left_length; evar ~loc right_length ])
       (invalid_argument ~loc
          (mismatch_message function_path "list length" path))
       (Some
          (call ~loc
             (Longident.parse "Stdlib.List.map2")
             [ mapper; evar ~loc left_name; evar ~loc right_name ])))

and map2_array ~loc ~function_path ~path callback shape left right =
  let left_name = gen_symbol ~prefix:"ptree_left" () in
  let right_name = gen_symbol ~prefix:"ptree_right" () in
  let left_length = gen_symbol ~prefix:"ptree_left_length" () in
  let right_length = gen_symbol ~prefix:"ptree_right_length" () in
  let index = gen_symbol ~prefix:"ptree_index" () in
  let value_at array =
    call ~loc
      (Longident.parse "Stdlib.Array.unsafe_get")
      [ evar ~loc array; evar ~loc index ]
  in
  let array_initializer =
    B.pexp_fun ~loc Nolabel None (pvar ~loc index)
      (map2_shape ~function_path ~path callback shape (value_at left_name)
         (value_at right_name))
  in
  lets ~loc
    [
      (left_name, left);
      (right_name, right);
      ( left_length,
        call ~loc
          (Longident.parse "Stdlib.Array.length")
          [ evar ~loc left_name ] );
      ( right_length,
        call ~loc
          (Longident.parse "Stdlib.Array.length")
          [ evar ~loc right_name ] );
    ]
    (B.pexp_ifthenelse ~loc
       (call ~loc (stdlib_operator "<>")
          [ evar ~loc left_length; evar ~loc right_length ])
       (invalid_argument ~loc
          (mismatch_message function_path "array length" path))
       (Some
          (call ~loc
             (Longident.parse "Stdlib.Array.init")
             [ evar ~loc left_length; array_initializer ])))

let rec iter_shape callback shape expression =
  let loc = shape.loc in
  match shape.desc with
  | Leaf -> call ~loc (Lident callback) [ expression ]
  | Ignored -> B.eunit ~loc
  | Local name ->
      call ~loc (Lident (names_for_type name).iter)
        [ evar ~loc callback; expression ]
  | Using module_path ->
      call ~loc
        (append_lid module_path "iter")
        [ evar ~loc callback; expression ]
  | Tuple shapes ->
      let names, pattern, tuple =
        tuple_bindings ~loc expression (List.length shapes)
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr:tuple ]
        (B.esequence ~loc
           (List.map2
              (fun shape name ->
                iter_shape callback shape (evar ~loc:shape.loc name))
              shapes names))
  | Option shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      B.pexp_match ~loc expression
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None ~rhs:(B.eunit ~loc);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc value)))
            ~guard:None
            ~rhs:(iter_shape callback shape (evar ~loc:shape.loc value));
        ]
  | List shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let iterator =
        B.pexp_fun ~loc Nolabel None (pvar ~loc value)
          (iter_shape callback shape (evar ~loc:shape.loc value))
      in
      call ~loc (Longident.parse "Stdlib.List.iter") [ iterator; expression ]
  | Array shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let iterator =
        B.pexp_fun ~loc Nolabel None (pvar ~loc value)
          (iter_shape callback shape (evar ~loc:shape.loc value))
      in
      call ~loc (Longident.parse "Stdlib.Array.iter") [ iterator; expression ]

let rec uses_callback shape =
  match shape.desc with
  | Leaf | Local _ | Using _ -> true
  | Ignored -> false
  | Tuple shapes -> List.exists uses_callback shapes
  | Option shape | List shape | Array shape -> uses_callback shape

let body_uses_callback = function
  | Alias shape -> uses_callback shape
  | Record fields -> List.exists (fun (_, shape) -> uses_callback shape) fields

let record_map ~loc callback fields input =
  let bindings =
    List.map
      (fun (label, shape) ->
        let field_loc = label.pld_loc in
        let name = gen_symbol ~prefix:("ptree_" ^ label.pld_name.txt) () in
        ( field_loc,
          name,
          map_shape callback shape (field ~loc:field_loc input label),
          label ))
      fields
  in
  lets_located
    (List.map
       (fun (field_loc, name, expression, _) -> (field_loc, name, expression))
       bindings)
    (B.pexp_record ~loc
       (List.map
          (fun (field_loc, name, _, label) ->
            ( lident ~loc:label.pld_name.loc label.pld_name.txt,
              evar ~loc:field_loc name ))
          bindings)
       None)

let record_map2 ~loc ~function_path callback fields left right =
  let bindings =
    List.map
      (fun (label, shape) ->
        let field_loc = label.pld_loc in
        let name = gen_symbol ~prefix:("ptree_" ^ label.pld_name.txt) () in
        let path = [ label.pld_name.txt ] in
        ( field_loc,
          name,
          map2_shape ~function_path ~path callback shape
            (field ~loc:field_loc left label)
            (field ~loc:field_loc right label),
          label ))
      fields
  in
  lets_located
    (List.map
       (fun (field_loc, name, expression, _) -> (field_loc, name, expression))
       bindings)
    (B.pexp_record ~loc
       (List.map
          (fun (field_loc, name, _, label) ->
            ( lident ~loc:label.pld_name.loc label.pld_name.txt,
              evar ~loc:field_loc name ))
          bindings)
       None)

let record_iter ~loc callback fields input =
  B.esequence ~loc
    (List.map
       (fun (label, shape) ->
         iter_shape callback shape (field ~loc:label.pld_loc input label))
       fields)

let operation_body ~loc ~function_path operation body =
  let input = evar ~loc "x" in
  let second = evar ~loc "y" in
  let expression =
    match (operation, body) with
    | `Map, Alias shape -> map_shape "f" shape input
    | `Map, Record fields -> record_map ~loc "f" fields input
    | `Map2, Alias shape ->
        map2_shape ~function_path ~path:[] "f" shape input second
    | `Map2, Record fields ->
        record_map2 ~loc ~function_path "f" fields input second
    | `Iter, Alias shape -> iter_shape "f" shape input
    | `Iter, Record fields -> record_iter ~loc "f" fields input
  in
  if body_uses_callback body then expression
  else
    let expression =
      match operation with
      | `Map | `Iter -> expression
      | `Map2 ->
          B.pexp_sequence ~loc
            (call ~loc (Longident.parse "Stdlib.ignore") [ evar ~loc "y" ])
            expression
    in
    B.pexp_sequence ~loc
      (call ~loc (Longident.parse "Stdlib.ignore") [ evar ~loc "f" ])
      expression

let operation_name operation names =
  match operation with
  | `Map -> names.map
  | `Map2 -> names.map2
  | `Iter -> names.iter

let make_binding ~module_path operation declaration =
  let type_decl = declaration.type_decl in
  let loc = type_decl.ptype_loc in
  let names = names_for_type type_decl.ptype_name.txt in
  let name = operation_name operation names in
  let function_path =
    if module_path = "" then name else module_path ^ "." ^ name
  in
  let variable_a, variable_b = fresh_callback_variables type_decl in
  let callback_type = callback_type ~loc operation variable_a variable_b in
  let type_ = declared_type type_decl in
  let body =
    match declaration.body with
    | Some body -> operation_body ~loc ~function_path operation body
    | None -> assert false
  in
  let input_types =
    match operation with `Map | `Iter -> [ type_ ] | `Map2 -> [ type_; type_ ]
  in
  B.value_binding ~loc ~pat:(pvar ~loc name)
    ~expr:(function_expression ~loc ~callback_type ~input_types body)

let strongly_connected_components declarations =
  let declarations_by_name =
    List.fold_left
      (fun result declaration ->
        String_map.add declaration.type_decl.ptype_name.txt declaration result)
      String_map.empty declarations
  in
  let index = ref 0 in
  let indices = Hashtbl.create (List.length declarations) in
  let lowlinks = Hashtbl.create (List.length declarations) in
  let on_stack = Hashtbl.create (List.length declarations) in
  let stack = Stack.create () in
  let components = ref [] in
  let rec visit name =
    Hashtbl.add indices name !index;
    Hashtbl.add lowlinks name !index;
    incr index;
    Stack.push name stack;
    Hashtbl.replace on_stack name true;
    let declaration = String_map.find name declarations_by_name in
    String_set.iter
      (fun dependency ->
        if String_map.mem dependency declarations_by_name then
          if not (Hashtbl.mem indices dependency) then (
            visit dependency;
            Hashtbl.replace lowlinks name
              (min
                 (Hashtbl.find lowlinks name)
                 (Hashtbl.find lowlinks dependency)))
          else if
            Option.value (Hashtbl.find_opt on_stack dependency) ~default:false
          then
            Hashtbl.replace lowlinks name
              (min
                 (Hashtbl.find lowlinks name)
                 (Hashtbl.find indices dependency)))
      declaration.dependencies;
    if Hashtbl.find lowlinks name = Hashtbl.find indices name then (
      let component = ref [] in
      let finished = ref false in
      while not !finished do
        let member = Stack.pop stack in
        Hashtbl.replace on_stack member false;
        component := member :: !component;
        finished := String.equal member name
      done;
      components := !component :: !components)
  in
  List.iter
    (fun declaration ->
      let name = declaration.type_decl.ptype_name.txt in
      if not (Hashtbl.mem indices name) then visit name)
    declarations;
  let source_index = Hashtbl.create (List.length declarations) in
  List.iteri
    (fun index declaration ->
      Hashtbl.add source_index declaration.type_decl.ptype_name.txt index)
    declarations;
  List.map
    (List.sort (fun left right ->
         Int.compare
           (Hashtbl.find source_index left)
           (Hashtbl.find source_index right)))
    (List.rev !components)

let ordered_components declarations =
  let components = strongly_connected_components declarations in
  let component_of_name = Hashtbl.create (List.length declarations) in
  List.iteri
    (fun index component ->
      List.iter (fun name -> Hashtbl.add component_of_name name index) component)
    components;
  let declarations_by_name =
    List.fold_left
      (fun result declaration ->
        String_map.add declaration.type_decl.ptype_name.txt declaration result)
      String_map.empty declarations
  in
  let component_dependencies index component =
    List.fold_left
      (fun result name ->
        let declaration = String_map.find name declarations_by_name in
        String_set.fold
          (fun dependency result ->
            match Hashtbl.find_opt component_of_name dependency with
            | Some dependency_index when dependency_index <> index ->
                Int_set.add dependency_index result
            | _ -> result)
          declaration.dependencies result)
      Int_set.empty component
  in
  let dependencies =
    List.mapi
      (fun index component -> component_dependencies index component)
      components
  in
  let rec emit emitted pending result =
    match pending with
    | [] -> List.rev result
    | _ ->
        let ready, waiting =
          List.partition
            (fun index -> Int_set.subset (List.nth dependencies index) emitted)
            pending
        in
        if ready = [] then assert false;
        let emitted =
          List.fold_left
            (fun result index -> Int_set.add index result)
            emitted ready
        in
        emit emitted waiting (List.rev_append ready result)
  in
  let indices = List.init (List.length components) Fun.id in
  let order = emit Int_set.empty indices [] in
  List.map
    (fun index ->
      List.map
        (fun name -> String_map.find name declarations_by_name)
        (List.nth components index))
    order

let component_rec_flag component =
  match component with
  | [] -> assert false
  | [ declaration ] ->
      if
        String_set.mem declaration.type_decl.ptype_name.txt
          declaration.dependencies
      then Recursive
      else Nonrecursive
  | _ -> Recursive

let generate_operation ~module_path operation components =
  List.map
    (fun component ->
      let loc = (List.hd component).type_decl.ptype_loc in
      B.pstr_value ~loc
        (component_rec_flag component)
        (List.map (make_binding ~module_path operation) component))
    components

let structure_generator ~ctxt (_, type_declarations) =
  let declarations, errors = validate ~signature:false type_declarations in
  if errors <> [] then structure_errors errors
  else
    let module_path =
      Expansion_context.Deriver.code_path ctxt |> Code_path.fully_qualified_path
    in
    let components = ordered_components declarations in
    List.concat_map
      (fun operation -> generate_operation ~module_path operation components)
      [ `Map; `Map2; `Iter ]

let signature_type operation type_decl =
  let loc = type_decl.ptype_loc in
  let variable_a, variable_b = fresh_callback_variables type_decl in
  let callback = callback_type ~loc operation variable_a variable_b in
  let type_ = declared_type type_decl in
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  match operation with
  | `Map -> arrow callback (arrow type_ type_)
  | `Map2 -> arrow callback (arrow type_ (arrow type_ type_))
  | `Iter ->
      arrow callback (arrow type_ (B.ptyp_constr ~loc (lident ~loc "unit") []))

let signature_generator ~ctxt (_, type_declarations) =
  let declarations, errors = validate ~signature:true type_declarations in
  if errors <> [] then signature_errors errors
  else
    let add_values declaration =
      let type_decl = declaration.type_decl in
      let names = names_for_type type_decl.ptype_name.txt in
      List.map
        (fun operation ->
          let name = operation_name operation names in
          let loc = type_decl.ptype_name.loc in
          B.psig_value ~loc
            (B.value_description ~loc ~name:{ txt = name; loc }
               ~type_:(signature_type operation type_decl)
               ~prim:[]))
        [ `Map; `Map2; `Iter ]
    in
    Stdlib.ignore ctxt;
    List.concat_map add_values declarations

(* Uniform mode: rank-1 traversals over a single payload type parameter.

   A declaration group derives uniform traversals when every type has exactly
   one named type parameter and at least one declaration mentions its parameter
   outside tensor leaf types, at a position not annotated with a ptree
   attribute. Everything else derives the classic rank-2 traversals above. *)

type uniform_names = {
  umap : string;
  umap2 : string;
  uiter : string;
  ufold : string;
  ufold2 : string;
  unames : string;
}

type ushape_desc =
  | UPayload
  | UStatic
  | UTuple of ushape list
  | UOption of ushape
  | UList of ushape
  | UArray of ushape
  | ULocal of string
  | UUsing of Longident.t

and ushape = { udesc : ushape_desc; uloc : Location.t }

type ubody = UAlias of ushape | URecord of (label_declaration * ushape) list

type udeclaration = {
  utype_decl : type_declaration;
  ubody : ubody option;
  udependencies : String_set.t;
}

let ushape ~loc desc = { udesc = desc; uloc = loc }

let uniform_names_for_type = function
  | "t" | "params" ->
      {
        umap = "map";
        umap2 = "map2";
        uiter = "iter";
        ufold = "fold";
        ufold2 = "fold2";
        unames = "names";
      }
  | name ->
      {
        umap = "map_" ^ name;
        umap2 = "map2_" ^ name;
        uiter = "iter_" ^ name;
        ufold = "fold_" ^ name;
        ufold2 = "fold2_" ^ name;
        unames = "names_" ^ name;
      }

let uoperation_name operation names =
  match operation with
  | `Umap -> names.umap
  | `Umap2 -> names.umap2
  | `Uiter -> names.uiter
  | `Ufold -> names.ufold
  | `Ufold2 -> names.ufold2
  | `Unames -> names.unames

(* Mode dispatch *)

let single_parameter type_decl =
  match type_decl.ptype_params with
  | [ ({ ptyp_desc = Ptyp_var name; _ }, _) ] -> Some name
  | _ -> None

let is_tensor_path path =
  is_nx_alias path
  || is_path path [ "Nx"; "t" ]
  || is_path path [ "Nx_effect"; "t" ]

let is_ptree_attribute name =
  String.equal name "ptree.leaf"
  || String.equal name "ptree.ignore"
  || String.equal name "ptree.using"

let core_has_ptree_attribute core_type =
  List.exists
    (fun attribute -> is_ptree_attribute attribute.attr_name.txt)
    core_type.ptyp_attributes

let label_has_ptree_attribute label =
  List.exists
    (fun attribute -> is_ptree_attribute attribute.attr_name.txt)
    label.pld_attributes

(* [occurs_bare parameter core_type] holds when the parameter appears outside
   tensor leaf types and outside ptree-annotated positions; annotated positions
   keep their rank-2 meaning. [occurs ~respect_attributes:false] detects the
   parameter even under annotations, to report meaningless annotations. *)
let rec occurs ~respect_attributes parameter core_type =
  ((not respect_attributes) || not (core_has_ptree_attribute core_type))
  &&
  match core_type.ptyp_desc with
  | Ptyp_var name -> String.equal name parameter
  | Ptyp_constr (path, _) when is_tensor_path path.txt -> false
  | Ptyp_constr (_, arguments) ->
      List.exists (occurs ~respect_attributes parameter) arguments
  | Ptyp_tuple elements ->
      List.exists (occurs ~respect_attributes parameter) elements
  | Ptyp_alias (inner, _) -> occurs ~respect_attributes parameter inner
  | Ptyp_arrow (_, left, right) ->
      occurs ~respect_attributes parameter left
      || occurs ~respect_attributes parameter right
  | _ -> false

let occurs_bare = occurs ~respect_attributes:true

let declaration_occurs_bare type_decl =
  match single_parameter type_decl with
  | None -> false
  | Some parameter -> (
      match (type_decl.ptype_kind, type_decl.ptype_manifest) with
      | Ptype_record labels, None ->
          List.exists
            (fun label ->
              (not (label_has_ptree_attribute label))
              && occurs_bare parameter label.pld_type)
            labels
      | Ptype_abstract, Some manifest -> occurs_bare parameter manifest
      | _ -> false)

let uniform_group type_declarations =
  List.for_all
    (fun type_decl -> Option.is_some (single_parameter type_decl))
    type_declarations
  && List.exists declaration_occurs_bare type_declarations

(* Classification *)

let rec uconsume_attributes validation core_type =
  Stdlib.ignore (annotation_at_core validation core_type : annotation list);
  match core_type.ptyp_desc with
  | Ptyp_tuple elements -> List.iter (uconsume_attributes validation) elements
  | Ptyp_constr (_, arguments) ->
      List.iter (uconsume_attributes validation) arguments
  | Ptyp_alias (inner, _) -> uconsume_attributes validation inner
  | Ptyp_arrow (_, left, right) ->
      uconsume_attributes validation left;
      uconsume_attributes validation right
  | _ -> ()

let rec uclassify validation parameter core_type =
  let loc = core_type.ptyp_loc in
  match core_type.ptyp_desc with
  | Ptyp_constr (path, _) when is_tensor_path path.txt ->
      uconsume_attributes validation core_type;
      ushape ~loc UStatic
  | _ when not (occurs_bare parameter core_type) ->
      if occurs ~respect_attributes:false parameter core_type then
        add_error validation ~loc
          "ppx_ptree: ptree attributes do not apply to payload positions of \
           uniform (parameterised) types; the payload parameter determines the \
           traversal";
      uconsume_attributes validation core_type;
      ushape ~loc UStatic
  | _ ->
      Stdlib.ignore (annotation_at_core validation core_type : annotation list);
      uclassify_bare validation parameter core_type

and uclassify_bare validation parameter core_type =
  let loc = core_type.ptyp_loc in
  match core_type.ptyp_desc with
  | Ptyp_var _ -> ushape ~loc UPayload
  | Ptyp_constr (path, arguments) when container_kind path.txt <> None -> (
      match (container_kind path.txt, arguments) with
      | Some `Option, [ argument ] ->
          ushape ~loc (UOption (uclassify validation parameter argument))
      | Some `List, [ argument ] ->
          ushape ~loc (UList (uclassify validation parameter argument))
      | Some `Array, [ argument ] ->
          ushape ~loc (UArray (uclassify validation parameter argument))
      | _ ->
          add_error validation ~loc
            "ppx_ptree: container types must have exactly one argument";
          ushape ~loc UStatic)
  | Ptyp_constr (path, [ { ptyp_desc = Ptyp_var _; _ } ]) -> (
      match local_name path.txt with
      | Some name -> ushape ~loc (ULocal name)
      | None -> (
          match external_primary path.txt with
          | Some module_path -> ushape ~loc (UUsing module_path)
          | None ->
              add_error validation ~loc
                "ppx_ptree: cannot delegate a uniform traversal to [%a]; only \
                 locally declared types and qualified [M.t] or [M.params] \
                 types can be applied to the payload parameter"
                Pprintast.longident path.txt;
              ushape ~loc UStatic))
  | Ptyp_constr (path, _) ->
      add_error validation ~loc
        "ppx_ptree: cannot derive a uniform traversal through [%a]; the \
         payload parameter may appear directly, under tuples, [option], \
         [list], [array], or as [M.t] applied to the parameter alone"
        Pprintast.longident path.txt;
      ushape ~loc UStatic
  | Ptyp_tuple elements ->
      ushape ~loc (UTuple (List.map (uclassify validation parameter) elements))
  | Ptyp_alias (inner, _) -> uclassify validation parameter inner
  | Ptyp_arrow _ ->
      add_error validation ~loc
        "ppx_ptree: function types cannot carry the payload parameter";
      ushape ~loc UStatic
  | _ ->
      add_error validation ~loc
        "ppx_ptree: this type cannot carry the payload parameter";
      ushape ~loc UStatic

let rec udependencies shape =
  match shape.udesc with
  | UPayload | UStatic | UUsing _ -> String_set.empty
  | ULocal name -> String_set.singleton name
  | UTuple shapes ->
      List.fold_left
        (fun result shape -> String_set.union result (udependencies shape))
        String_set.empty shapes
  | UOption shape | UList shape | UArray shape -> udependencies shape

let rec ushape_has_payload shape =
  match shape.udesc with
  | UPayload | ULocal _ | UUsing _ -> true
  | UStatic -> false
  | UTuple shapes -> List.exists ushape_has_payload shapes
  | UOption shape | UList shape | UArray shape -> ushape_has_payload shape

let ubody_has_payload = function
  | UAlias shape -> ushape_has_payload shape
  | URecord fields ->
      List.exists (fun (_, shape) -> ushape_has_payload shape) fields

let uvalidate_declaration ~signature validation type_decl =
  let parameter =
    match single_parameter type_decl with
    | Some parameter -> parameter
    | None ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: every type in a uniform declaration group must have \
           exactly one type parameter";
        "_"
  in
  if (not signature) && type_decl.ptype_private = Private then
    add_error validation ~loc:type_decl.ptype_name.loc
      "ppx_ptree: private type implementations cannot be derived";
  let classify_label label =
    if
      annotation_at_label validation label <> []
      && occurs_bare parameter label.pld_type
    then
      add_error validation ~loc:label.pld_loc
        "ppx_ptree: ptree attributes do not apply to payload positions of \
         uniform (parameterised) types; the payload parameter determines the \
         traversal";
    (label, uclassify validation parameter label.pld_type)
  in
  let body =
    match (type_decl.ptype_kind, type_decl.ptype_manifest) with
    | Ptype_record labels, None ->
        Some (URecord (List.map classify_label labels))
    | Ptype_abstract, Some manifest ->
        Some (UAlias (uclassify validation parameter manifest))
    | Ptype_abstract, None when signature -> None
    | Ptype_abstract, None ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: an abstract implementation has no traversable shape";
        None
    | Ptype_record _, Some _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: representation re-exports with records are not supported";
        None
    | Ptype_variant _, _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: variants are not supported in version 1";
        None
    | Ptype_open, _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: extensible variants are not supported";
        None
  in
  (match body with
  | Some body when not (ubody_has_payload body) ->
      add_error validation ~loc:type_decl.ptype_name.loc
        "ppx_ptree: the type parameter ['%s] of [%s] never occurs in a payload \
         position"
        parameter type_decl.ptype_name.txt
  | _ -> ());
  let dependencies =
    match body with
    | None -> String_set.empty
    | Some (UAlias shape) -> udependencies shape
    | Some (URecord fields) ->
        List.fold_left
          (fun result (_, shape) ->
            String_set.union result (udependencies shape))
          String_set.empty fields
  in
  { utype_decl = type_decl; ubody = body; udependencies = dependencies }

let uvalidate ~signature type_declarations =
  let validation = { errors_rev = [] } in
  validate_primary_owner validation type_declarations;
  let declarations =
    List.map (uvalidate_declaration ~signature validation) type_declarations
  in
  (declarations, List.rev validation.errors_rev)

(* Uniform declarations reuse the rank-2 component ordering by wrapping. *)

let udecl_to_decl udecl =
  {
    type_decl = udecl.utype_decl;
    body = None;
    dependencies = udecl.udependencies;
  }

let uordered_components declarations =
  let components = ordered_components (List.map udecl_to_decl declarations) in
  List.map
    (List.map (fun declaration ->
         List.find
           (fun udecl ->
             String.equal udecl.utype_decl.ptype_name.txt
               declaration.type_decl.ptype_name.txt)
           declarations))
    components

let ucomponent_rec_flag component =
  match component with
  | [] -> assert false
  | [ udecl ] ->
      if String_set.mem udecl.utype_decl.ptype_name.txt udecl.udependencies then
        Recursive
      else Nonrecursive
  | _ -> Recursive

(* Leaf paths. Record fields and tuple positions are known while expanding, so a
   path stays a literal until a container index or a delegated sub-path makes it
   depend on the value. Segments are joined with ["."], matching the checkpoint
   naming convention; the root is the empty path. *)

type upath = Uroot | Uliteral of string | Ucomputed of expression

let concat ~loc left right = call ~loc (stdlib_operator "^") [ left; right ]

let ujoin_literal ~loc path segment =
  match path with
  | Uroot -> Uliteral segment
  | Uliteral prefix -> Uliteral (prefix ^ "." ^ segment)
  | Ucomputed prefix ->
      Ucomputed (concat ~loc prefix (B.estring ~loc ("." ^ segment)))

let ujoin_computed ~loc path segment =
  match path with
  | Uroot -> Ucomputed segment
  | Uliteral prefix ->
      Ucomputed (concat ~loc (B.estring ~loc (prefix ^ ".")) segment)
  | Ucomputed prefix ->
      Ucomputed (concat ~loc (concat ~loc prefix (B.estring ~loc ".")) segment)

let upath_expr ~loc path =
  match path with
  | Uroot -> B.estring ~loc ""
  | Uliteral path -> B.estring ~loc path
  | Ucomputed path -> path

(* A delegated traversal numbers its leaves from its own root, so a payload
   sitting at that root hands back the empty path. Joining it unconditionally
   would leave a trailing separator. *)
let ujoin_delegated ~loc path segment =
  let guarded prefix =
    B.pexp_ifthenelse ~loc
      (call ~loc
         (Longident.parse "Stdlib.String.equal")
         [ segment; B.estring ~loc "" ])
      (upath_expr ~loc prefix)
      (Some (upath_expr ~loc (ujoin_computed ~loc prefix segment)))
  in
  match path with
  | Uroot -> segment
  | Uliteral _ -> guarded path
  | Ucomputed prefix ->
      let name = gen_symbol ~prefix:"ptree_prefix" () in
      let_one ~loc name prefix (guarded (Ucomputed (evar ~loc name)))

let uindex_path ~loc path index =
  ujoin_computed ~loc path
    (call ~loc (Longident.parse "Stdlib.string_of_int") [ evar ~loc index ])

let umismatch ~loc ~function_path ~path kind =
  let message = Format.sprintf "%s: %s mismatch at " function_path kind in
  match path with
  | Uroot -> invalid_argument ~loc (message ^ "<root>")
  | Uliteral path -> invalid_argument ~loc (message ^ path)
  | Ucomputed path ->
      call ~loc
        (Longident.Ldot (Lident "Stdlib", "invalid_arg"))
        [ concat ~loc (B.estring ~loc message) path ]

(* map *)

let rec umap_expr fvar shape expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> call ~loc (Lident fvar) [ expr ]
  | UStatic -> expr
  | ULocal name ->
      call ~loc (Lident (uniform_names_for_type name).umap)
        [ evar ~loc fvar; expr ]
  | UUsing module_path ->
      call ~loc (append_lid module_path "map") [ evar ~loc fvar; expr ]
  | UTuple shapes ->
      let names, pattern, tuple =
        tuple_bindings ~loc expr (List.length shapes)
      in
      let mapped_names =
        List.map (fun _ -> gen_symbol ~prefix:"ptree_mapped" ()) shapes
      in
      let mapped =
        List.map2
          (fun shape name -> umap_expr fvar shape (evar ~loc:shape.uloc name))
          shapes names
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr:tuple ]
        (lets ~loc
           (List.combine mapped_names mapped)
           (B.pexp_tuple ~loc (List.map (evar ~loc) mapped_names)))
  | UOption shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      B.pexp_match ~loc expr
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc value)))
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some (umap_expr fvar shape (evar ~loc:shape.uloc value))));
        ]
  | UList shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let mapper =
        B.pexp_fun ~loc Nolabel None (pvar ~loc value)
          (umap_expr fvar shape (evar ~loc:shape.uloc value))
      in
      call ~loc (Longident.parse "Stdlib.List.map") [ mapper; expr ]
  | UArray shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let mapper =
        B.pexp_fun ~loc Nolabel None (pvar ~loc value)
          (umap_expr fvar shape (evar ~loc:shape.uloc value))
      in
      call ~loc (Longident.parse "Stdlib.Array.map") [ mapper; expr ]

(* map2 *)

let rec umap2_expr ~function_path ~path fvar shape left right =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> call ~loc (Lident fvar) [ left; right ]
  | UStatic -> left
  | ULocal name ->
      call ~loc (Lident (uniform_names_for_type name).umap2)
        [ evar ~loc fvar; left; right ]
  | UUsing module_path ->
      call ~loc (append_lid module_path "map2") [ evar ~loc fvar; left; right ]
  | UTuple shapes ->
      let left_names, left_pattern, left_tuple =
        tuple_bindings ~loc left (List.length shapes)
      in
      let right_names, right_pattern, right_tuple =
        tuple_bindings ~loc right (List.length shapes)
      in
      let mapped_names =
        List.map (fun _ -> gen_symbol ~prefix:"ptree_mapped" ()) shapes
      in
      let mapped =
        List.mapi
          (fun index shape ->
            umap2_expr ~function_path
              ~path:(ujoin_literal ~loc path (string_of_int index))
              fvar shape
              (evar ~loc:shape.uloc (List.nth left_names index))
              (evar ~loc:shape.uloc (List.nth right_names index)))
          shapes
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:left_pattern ~expr:left_tuple ]
        (B.pexp_let ~loc Nonrecursive
           [ B.value_binding ~loc ~pat:right_pattern ~expr:right_tuple ]
           (lets ~loc
              (List.combine mapped_names mapped)
              (B.pexp_tuple ~loc (List.map (evar ~loc) mapped_names))))
  | UOption shape ->
      let left_value = gen_symbol ~prefix:"ptree_left" () in
      let right_value = gen_symbol ~prefix:"ptree_right" () in
      B.pexp_match ~loc
        (B.pexp_tuple ~loc [ left; right ])
        [
          B.case
            ~lhs:
              (B.ppat_tuple ~loc
                 [
                   construct_pattern ~loc "None" None;
                   construct_pattern ~loc "None" None;
                 ])
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:
              (B.ppat_tuple ~loc
                 [
                   construct_pattern ~loc "Some" (Some (pvar ~loc left_value));
                   construct_pattern ~loc "Some" (Some (pvar ~loc right_value));
                 ])
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some
                    (umap2_expr ~function_path ~path fvar shape
                       (evar ~loc:shape.uloc left_value)
                       (evar ~loc:shape.uloc right_value))));
          B.case ~lhs:(B.ppat_any ~loc) ~guard:None
            ~rhs:(umismatch ~loc ~function_path ~path "option constructor");
        ]
  | UList shape ->
      let left_name = gen_symbol ~prefix:"ptree_left" () in
      let right_name = gen_symbol ~prefix:"ptree_right" () in
      let left_value = gen_symbol ~prefix:"ptree_left_value" () in
      let right_value = gen_symbol ~prefix:"ptree_right_value" () in
      let mapper =
        B.pexp_fun ~loc Nolabel None (pvar ~loc left_value)
          (B.pexp_fun ~loc Nolabel None (pvar ~loc right_value)
             (umap2_expr ~function_path ~path fvar shape
                (evar ~loc:shape.uloc left_value)
                (evar ~loc:shape.uloc right_value)))
      in
      lets ~loc
        [ (left_name, left); (right_name, right) ]
        (B.pexp_ifthenelse ~loc
           (call ~loc (stdlib_operator "<>")
              [
                call ~loc
                  (Longident.parse "Stdlib.List.length")
                  [ evar ~loc left_name ];
                call ~loc
                  (Longident.parse "Stdlib.List.length")
                  [ evar ~loc right_name ];
              ])
           (umismatch ~loc ~function_path ~path "list length")
           (Some
              (call ~loc
                 (Longident.parse "Stdlib.List.map2")
                 [ mapper; evar ~loc left_name; evar ~loc right_name ])))
  | UArray shape ->
      let left_name = gen_symbol ~prefix:"ptree_left" () in
      let right_name = gen_symbol ~prefix:"ptree_right" () in
      let length_name = gen_symbol ~prefix:"ptree_length" () in
      let index = gen_symbol ~prefix:"ptree_index" () in
      let value_at array =
        call ~loc
          (Longident.parse "Stdlib.Array.unsafe_get")
          [ evar ~loc array; evar ~loc index ]
      in
      let array_initializer =
        B.pexp_fun ~loc Nolabel None (pvar ~loc index)
          (umap2_expr ~function_path ~path fvar shape (value_at left_name)
             (value_at right_name))
      in
      lets ~loc
        [
          (left_name, left);
          (right_name, right);
          ( length_name,
            call ~loc
              (Longident.parse "Stdlib.Array.length")
              [ evar ~loc left_name ] );
        ]
        (B.pexp_ifthenelse ~loc
           (call ~loc (stdlib_operator "<>")
              [
                evar ~loc length_name;
                call ~loc
                  (Longident.parse "Stdlib.Array.length")
                  [ evar ~loc right_name ];
              ])
           (umismatch ~loc ~function_path ~path "array length")
           (Some
              (call ~loc
                 (Longident.parse "Stdlib.Array.init")
                 [ evar ~loc length_name; array_initializer ])))

(* iter *)

let rec uiter_expr fvar shape expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> call ~loc (Lident fvar) [ expr ]
  | UStatic -> B.eunit ~loc
  | ULocal name ->
      call ~loc (Lident (uniform_names_for_type name).uiter)
        [ evar ~loc fvar; expr ]
  | UUsing module_path ->
      call ~loc (append_lid module_path "iter") [ evar ~loc fvar; expr ]
  | UTuple shapes ->
      let names, pattern, tuple =
        tuple_bindings ~loc expr (List.length shapes)
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr:tuple ]
        (B.esequence ~loc
           (List.map2
              (fun shape name ->
                uiter_expr fvar shape (evar ~loc:shape.uloc name))
              shapes names))
  | UOption shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      B.pexp_match ~loc expr
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None ~rhs:(B.eunit ~loc);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc value)))
            ~guard:None
            ~rhs:(uiter_expr fvar shape (evar ~loc:shape.uloc value));
        ]
  | UList shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let iterator =
        B.pexp_fun ~loc Nolabel None (pvar ~loc value)
          (uiter_expr fvar shape (evar ~loc:shape.uloc value))
      in
      call ~loc (Longident.parse "Stdlib.List.iter") [ iterator; expr ]
  | UArray shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let iterator =
        B.pexp_fun ~loc Nolabel None (pvar ~loc value)
          (uiter_expr fvar shape (evar ~loc:shape.uloc value))
      in
      call ~loc (Longident.parse "Stdlib.Array.iter") [ iterator; expr ]

(* fold: the callback receives the leaf path, the accumulator, then the leaf.
   Delegation re-roots the sub-structure's paths under the current path. *)

let udelegate_callback ~loc ~path fvar arity =
  let sub_path = gen_symbol ~prefix:"ptree_path" () in
  let rest = List.init arity (fun _ -> gen_symbol ~prefix:"ptree_arg" ()) in
  B.pexp_fun ~loc Nolabel None (pvar ~loc sub_path)
    (List.fold_right
       (fun name body -> B.pexp_fun ~loc Nolabel None (pvar ~loc name) body)
       rest
       (call ~loc (Lident fvar)
          (ujoin_delegated ~loc path (evar ~loc sub_path)
          :: List.map (evar ~loc) rest)))

let rec ufold_expr ~path fvar shape expr acc_expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> call ~loc (Lident fvar) [ upath_expr ~loc path; acc_expr; expr ]
  | UStatic -> acc_expr
  | ULocal _ | UUsing _ ->
      let target =
        match shape.udesc with
        | ULocal name -> Longident.Lident (uniform_names_for_type name).ufold
        | UUsing module_path -> append_lid module_path "fold"
        | _ -> assert false
      in
      call ~loc target [ udelegate_callback ~loc ~path fvar 2; acc_expr; expr ]
  | UTuple shapes ->
      let names, pattern, tuple =
        tuple_bindings ~loc expr (List.length shapes)
      in
      let _, threaded =
        List.fold_left2
          (fun (index, acc) shape name ->
            ( index + 1,
              ufold_expr
                ~path:(ujoin_literal ~loc path (string_of_int index))
                fvar shape (evar ~loc name) acc ))
          (0, acc_expr) shapes names
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr:tuple ]
        threaded
  | UOption shape ->
      let acc_name = gen_symbol ~prefix:"ptree_acc" () in
      let value = gen_symbol ~prefix:"ptree_value" () in
      let_one ~loc acc_name acc_expr
        (B.pexp_match ~loc expr
           [
             B.case
               ~lhs:(construct_pattern ~loc "None" None)
               ~guard:None ~rhs:(evar ~loc acc_name);
             B.case
               ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc value)))
               ~guard:None
               ~rhs:
                 (ufold_expr ~path fvar shape
                    (evar ~loc:shape.uloc value)
                    (evar ~loc acc_name));
           ])
  | UList shape ->
      let acc_name = gen_symbol ~prefix:"ptree_acc" () in
      let index = gen_symbol ~prefix:"ptree_index" () in
      let value = gen_symbol ~prefix:"ptree_value" () in
      let pair = gen_symbol ~prefix:"ptree_pair" () in
      let child_path = uindex_path ~loc path index in
      let inner_body =
        B.pexp_let ~loc Nonrecursive
          [
            B.value_binding ~loc
              ~pat:(B.ppat_tuple ~loc [ pvar ~loc index; pvar ~loc value ])
              ~expr:(evar ~loc pair);
          ]
          (ufold_expr ~path:child_path fvar shape (evar ~loc value)
             (evar ~loc acc_name))
      in
      call ~loc
        (Longident.parse "Stdlib.List.fold_left")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc acc_name)
            (B.pexp_fun ~loc Nolabel None (pvar ~loc pair) inner_body);
          acc_expr;
          call ~loc
            (Longident.parse "Stdlib.List.mapi")
            [
              B.pexp_fun ~loc Nolabel None (pvar ~loc index)
                (B.pexp_fun ~loc Nolabel None (pvar ~loc value)
                   (B.pexp_tuple ~loc [ evar ~loc index; evar ~loc value ]));
              expr;
            ];
        ]
  | UArray shape ->
      let acc_name = gen_symbol ~prefix:"ptree_acc" () in
      let index = gen_symbol ~prefix:"ptree_index" () in
      let value = gen_symbol ~prefix:"ptree_value" () in
      let pair = gen_symbol ~prefix:"ptree_pair" () in
      let child_path = uindex_path ~loc path index in
      let inner_body =
        B.pexp_let ~loc Nonrecursive
          [
            B.value_binding ~loc
              ~pat:(B.ppat_tuple ~loc [ pvar ~loc index; pvar ~loc value ])
              ~expr:(evar ~loc pair);
          ]
          (ufold_expr ~path:child_path fvar shape (evar ~loc value)
             (evar ~loc acc_name))
      in
      call ~loc
        (Longident.parse "Stdlib.Array.fold_left")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc acc_name)
            (B.pexp_fun ~loc Nolabel None (pvar ~loc pair) inner_body);
          acc_expr;
          call ~loc
            (Longident.parse "Stdlib.Array.mapi")
            [
              B.pexp_fun ~loc Nolabel None (pvar ~loc index)
                (B.pexp_fun ~loc Nolabel None (pvar ~loc value)
                   (B.pexp_tuple ~loc [ evar ~loc index; evar ~loc value ]));
              expr;
            ];
        ]

(* fold2 *)

let rec ufold2_expr ~function_path ~path fvar shape left right acc_expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload ->
      call ~loc (Lident fvar) [ upath_expr ~loc path; acc_expr; left; right ]
  | UStatic -> acc_expr
  | ULocal _ | UUsing _ ->
      let target =
        match shape.udesc with
        | ULocal name -> Longident.Lident (uniform_names_for_type name).ufold2
        | UUsing module_path -> append_lid module_path "fold2"
        | _ -> assert false
      in
      call ~loc target
        [ udelegate_callback ~loc ~path fvar 3; acc_expr; left; right ]
  | UTuple shapes ->
      let left_names, left_pattern, left_tuple =
        tuple_bindings ~loc left (List.length shapes)
      in
      let right_names, right_pattern, right_tuple =
        tuple_bindings ~loc right (List.length shapes)
      in
      let _, threaded =
        List.fold_left2
          (fun (index, acc) shape (left_name, right_name) ->
            ( index + 1,
              ufold2_expr ~function_path
                ~path:(ujoin_literal ~loc path (string_of_int index))
                fvar shape (evar ~loc left_name) (evar ~loc right_name) acc ))
          (0, acc_expr) shapes
          (List.combine left_names right_names)
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:left_pattern ~expr:left_tuple ]
        (B.pexp_let ~loc Nonrecursive
           [ B.value_binding ~loc ~pat:right_pattern ~expr:right_tuple ]
           threaded)
  | UOption shape ->
      let acc_name = gen_symbol ~prefix:"ptree_acc" () in
      let left_value = gen_symbol ~prefix:"ptree_left" () in
      let right_value = gen_symbol ~prefix:"ptree_right" () in
      let_one ~loc acc_name acc_expr
        (B.pexp_match ~loc
           (B.pexp_tuple ~loc [ left; right ])
           [
             B.case
               ~lhs:
                 (B.ppat_tuple ~loc
                    [
                      construct_pattern ~loc "None" None;
                      construct_pattern ~loc "None" None;
                    ])
               ~guard:None ~rhs:(evar ~loc acc_name);
             B.case
               ~lhs:
                 (B.ppat_tuple ~loc
                    [
                      construct_pattern ~loc "Some"
                        (Some (pvar ~loc left_value));
                      construct_pattern ~loc "Some"
                        (Some (pvar ~loc right_value));
                    ])
               ~guard:None
               ~rhs:
                 (ufold2_expr ~function_path ~path fvar shape
                    (evar ~loc:shape.uloc left_value)
                    (evar ~loc:shape.uloc right_value)
                    (evar ~loc acc_name));
             B.case ~lhs:(B.ppat_any ~loc) ~guard:None
               ~rhs:(umismatch ~loc ~function_path ~path "option constructor");
           ])
  | UList shape ->
      let left_name = gen_symbol ~prefix:"ptree_left" () in
      let right_name = gen_symbol ~prefix:"ptree_right" () in
      let acc_name = gen_symbol ~prefix:"ptree_acc" () in
      let index = gen_symbol ~prefix:"ptree_index" () in
      let left_value = gen_symbol ~prefix:"ptree_left_value" () in
      let right_value = gen_symbol ~prefix:"ptree_right_value" () in
      let pair = gen_symbol ~prefix:"ptree_pair" () in
      let child_path = uindex_path ~loc path index in
      let inner_body =
        B.pexp_let ~loc Nonrecursive
          [
            B.value_binding ~loc
              ~pat:(B.ppat_tuple ~loc [ pvar ~loc index; pvar ~loc left_value ])
              ~expr:(evar ~loc pair);
          ]
          (ufold2_expr ~function_path ~path:child_path fvar shape
             (evar ~loc left_value) (evar ~loc right_value) (evar ~loc acc_name))
      in
      lets ~loc
        [ (left_name, left); (right_name, right) ]
        (B.pexp_ifthenelse ~loc
           (call ~loc (stdlib_operator "<>")
              [
                call ~loc
                  (Longident.parse "Stdlib.List.length")
                  [ evar ~loc left_name ];
                call ~loc
                  (Longident.parse "Stdlib.List.length")
                  [ evar ~loc right_name ];
              ])
           (umismatch ~loc ~function_path ~path "list length")
           (Some
              (call ~loc
                 (Longident.parse "Stdlib.List.fold_left2")
                 [
                   B.pexp_fun ~loc Nolabel None (pvar ~loc acc_name)
                     (B.pexp_fun ~loc Nolabel None (pvar ~loc pair)
                        (B.pexp_fun ~loc Nolabel None (pvar ~loc right_value)
                           inner_body));
                   acc_expr;
                   call ~loc
                     (Longident.parse "Stdlib.List.mapi")
                     [
                       B.pexp_fun ~loc Nolabel None (pvar ~loc index)
                         (B.pexp_fun ~loc Nolabel None (pvar ~loc left_value)
                            (B.pexp_tuple ~loc
                               [ evar ~loc index; evar ~loc left_value ]));
                       evar ~loc left_name;
                     ];
                   evar ~loc right_name;
                 ])))
  | UArray shape ->
      let left_name = gen_symbol ~prefix:"ptree_left" () in
      let right_name = gen_symbol ~prefix:"ptree_right" () in
      let length_name = gen_symbol ~prefix:"ptree_length" () in
      let acc_name = gen_symbol ~prefix:"ptree_acc" () in
      let index = gen_symbol ~prefix:"ptree_index" () in
      let child_path = uindex_path ~loc path index in
      let value_at array =
        call ~loc
          (Longident.parse "Stdlib.Array.unsafe_get")
          [ evar ~loc array; evar ~loc index ]
      in
      lets ~loc
        [
          (left_name, left);
          (right_name, right);
          ( length_name,
            call ~loc
              (Longident.parse "Stdlib.Array.length")
              [ evar ~loc left_name ] );
        ]
        (B.pexp_ifthenelse ~loc
           (call ~loc (stdlib_operator "<>")
              [
                evar ~loc length_name;
                call ~loc
                  (Longident.parse "Stdlib.Array.length")
                  [ evar ~loc right_name ];
              ])
           (umismatch ~loc ~function_path ~path "array length")
           (Some
              (call ~loc
                 (Longident.parse "Stdlib.Array.fold_left")
                 [
                   B.pexp_fun ~loc Nolabel None (pvar ~loc acc_name)
                     (B.pexp_fun ~loc Nolabel None (pvar ~loc index)
                        (ufold2_expr ~function_path ~path:child_path fvar shape
                           (value_at left_name) (value_at right_name)
                           (evar ~loc acc_name)));
                   acc_expr;
                   call ~loc
                     (Longident.parse "Stdlib.Array.init")
                     [
                       evar ~loc length_name;
                       ident ~loc (Longident.parse "Stdlib.Fun.id");
                     ];
                 ])))

(* names: a path-labelled map. Payload positions become their path; static
   positions are copied from the input. *)

let rec unames_expr ~path shape expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> upath_expr ~loc path
  | UStatic -> expr
  | ULocal _ | UUsing _ ->
      let map_target, names_target =
        match shape.udesc with
        | ULocal name ->
            let names = uniform_names_for_type name in
            (Longident.Lident names.umap, Longident.Lident names.unames)
        | UUsing module_path ->
            (append_lid module_path "map", append_lid module_path "names")
        | _ -> assert false
      in
      let sub_path = gen_symbol ~prefix:"ptree_path" () in
      call ~loc map_target
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc sub_path)
            (ujoin_delegated ~loc path (evar ~loc sub_path));
          call ~loc names_target [ expr ];
        ]
  | UTuple shapes ->
      let names =
        List.map (fun _ -> gen_symbol ~prefix:"ptree_tuple" ()) shapes
      in
      let pattern =
        B.ppat_tuple ~loc
          (List.map2
             (fun shape name ->
               match shape.udesc with
               | UPayload -> B.ppat_any ~loc
               | _ -> pvar ~loc name)
             shapes names)
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr ]
        (B.pexp_tuple ~loc
           (List.mapi
              (fun index shape ->
                unames_expr
                  ~path:(ujoin_literal ~loc path (string_of_int index))
                  shape
                  (evar ~loc (List.nth names index)))
              shapes))
  | UOption shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      let value_pattern =
        match shape.udesc with
        | UPayload -> B.ppat_any ~loc
        | _ -> pvar ~loc value
      in
      B.pexp_match ~loc expr
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some value_pattern))
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some (unames_expr ~path shape (evar ~loc:shape.uloc value))));
        ]
  | UList shape ->
      let index = gen_symbol ~prefix:"ptree_index" () in
      let value = gen_symbol ~prefix:"ptree_value" () in
      let value_pattern =
        match shape.udesc with
        | UPayload -> B.ppat_any ~loc
        | _ -> pvar ~loc value
      in
      call ~loc
        (Longident.parse "Stdlib.List.mapi")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc index)
            (B.pexp_fun ~loc Nolabel None value_pattern
               (unames_expr
                  ~path:(uindex_path ~loc path index)
                  shape (evar ~loc value)));
          expr;
        ]
  | UArray shape ->
      let index = gen_symbol ~prefix:"ptree_index" () in
      let value = gen_symbol ~prefix:"ptree_value" () in
      let value_pattern =
        match shape.udesc with
        | UPayload -> B.ppat_any ~loc
        | _ -> pvar ~loc value
      in
      call ~loc
        (Longident.parse "Stdlib.Array.mapi")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc index)
            (B.pexp_fun ~loc Nolabel None value_pattern
               (unames_expr
                  ~path:(uindex_path ~loc path index)
                  shape (evar ~loc value)));
          expr;
        ]

(* Operation bodies over a declaration body. The tree inputs are [x] (and [y]
   for binary operations), the accumulator is [acc], the callback is [f]. *)

let urecord ~loc bindings =
  lets_located
    (List.map
       (fun (field_loc, name, expression, _) -> (field_loc, name, expression))
       bindings)
    (B.pexp_record ~loc
       (List.map
          (fun (_, name, _, label) ->
            (lident ~loc:label.pld_name.loc label.pld_name.txt, evar ~loc name))
          bindings)
       None)

let unames_uses_input = function
  | UAlias { udesc = UPayload; _ } -> false
  | UAlias _ -> true
  | URecord fields ->
      List.exists (fun (_, shape) -> shape.udesc <> UPayload) fields

let uoperation_body ~loc ~function_path operation ubody =
  let input = evar ~loc "x" in
  let second = evar ~loc "y" in
  let accumulator = evar ~loc "acc" in
  let field_path label = Uliteral label.pld_name.txt in
  match (operation, ubody) with
  | `Umap, UAlias shape -> umap_expr "f" shape input
  | `Umap, URecord fields ->
      urecord ~loc
        (List.map
           (fun (label, shape) ->
             ( label.pld_loc,
               gen_symbol ~prefix:("ptree_" ^ label.pld_name.txt) (),
               umap_expr "f" shape (field ~loc:label.pld_loc input label),
               label ))
           fields)
  | `Umap2, UAlias shape ->
      umap2_expr ~function_path ~path:Uroot "f" shape input second
  | `Umap2, URecord fields ->
      urecord ~loc
        (List.map
           (fun (label, shape) ->
             ( label.pld_loc,
               gen_symbol ~prefix:("ptree_" ^ label.pld_name.txt) (),
               umap2_expr ~function_path ~path:(field_path label) "f" shape
                 (field ~loc:label.pld_loc input label)
                 (field ~loc:label.pld_loc second label),
               label ))
           fields)
  | `Uiter, UAlias shape -> uiter_expr "f" shape input
  | `Uiter, URecord fields ->
      B.esequence ~loc
        (List.map
           (fun (label, shape) ->
             uiter_expr "f" shape (field ~loc:label.pld_loc input label))
           fields)
  | `Ufold, UAlias shape -> ufold_expr ~path:Uroot "f" shape input accumulator
  | `Ufold, URecord fields ->
      List.fold_left
        (fun acc (label, shape) ->
          ufold_expr ~path:(field_path label) "f" shape
            (field ~loc:label.pld_loc input label)
            acc)
        accumulator fields
  | `Ufold2, UAlias shape ->
      ufold2_expr ~function_path ~path:Uroot "f" shape input second accumulator
  | `Ufold2, URecord fields ->
      List.fold_left
        (fun acc (label, shape) ->
          ufold2_expr ~function_path ~path:(field_path label) "f" shape
            (field ~loc:label.pld_loc input label)
            (field ~loc:label.pld_loc second label)
            acc)
        accumulator fields
  | `Unames, UAlias shape ->
      let body = unames_expr ~path:Uroot shape input in
      if unames_uses_input ubody then body
      else
        B.pexp_sequence ~loc
          (call ~loc (Longident.parse "Stdlib.ignore") [ input ])
          body
  | `Unames, URecord fields ->
      let body =
        B.pexp_record ~loc
          (List.map
             (fun (label, shape) ->
               ( lident ~loc:label.pld_name.loc label.pld_name.txt,
                 unames_expr ~path:(field_path label) shape
                   (field ~loc:label.pld_loc input label) ))
             fields)
          None
      in
      if unames_uses_input ubody then body
      else
        B.pexp_sequence ~loc
          (call ~loc (Longident.parse "Stdlib.ignore") [ input ])
          body

let ufresh_variables type_decl =
  let used = used_type_variables type_decl in
  let choose base =
    let rec loop candidate index =
      if String_set.mem candidate used then
        loop (base ^ string_of_int index) (index + 1)
      else candidate
    in
    loop base 0
  in
  (choose "a", choose "b", choose "c", choose "acc")

let umake_binding ~module_path operation udecl =
  let type_decl = udecl.utype_decl in
  let loc = type_decl.ptype_loc in
  let names = uniform_names_for_type type_decl.ptype_name.txt in
  let name = uoperation_name operation names in
  let function_path =
    if module_path = "" then name else module_path ^ "." ^ name
  in
  let variable_a, variable_b, variable_c, variable_acc =
    ufresh_variables type_decl
  in
  let applied variable =
    B.ptyp_constr ~loc
      (lident ~loc type_decl.ptype_name.txt)
      [ B.ptyp_var ~loc variable ]
  in
  let variable name = B.ptyp_var ~loc name in
  let string_type = B.ptyp_constr ~loc (lident ~loc "string") [] in
  let unit_type = B.ptyp_constr ~loc (lident ~loc "unit") [] in
  let accumulator_type = variable variable_acc in
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  let body =
    match udecl.ubody with
    | Some ubody -> uoperation_body ~loc ~function_path operation ubody
    | None -> assert false
  in
  let functions parameters body =
    List.fold_right
      (fun (name, type_) body ->
        B.pexp_fun ~loc Nolabel None
          (constrained_parameter ~loc name type_)
          body)
      parameters body
  in
  let constrained body type_ = B.pexp_constraint ~loc body type_ in
  let expr =
    match operation with
    | `Umap ->
        functions
          [
            ("f", arrow (variable variable_a) (variable variable_b));
            ("x", applied variable_a);
          ]
          (constrained body (applied variable_b))
    | `Umap2 ->
        functions
          [
            ( "f",
              arrow (variable variable_a)
                (arrow (variable variable_b) (variable variable_c)) );
            ("x", applied variable_a);
            ("y", applied variable_b);
          ]
          (constrained body (applied variable_c))
    | `Uiter ->
        functions
          [
            ("f", arrow (variable variable_a) unit_type);
            ("x", applied variable_a);
          ]
          body
    | `Ufold ->
        functions
          [
            ( "f",
              arrow string_type
                (arrow accumulator_type
                   (arrow (variable variable_a) accumulator_type)) );
            ("acc", accumulator_type);
            ("x", applied variable_a);
          ]
          body
    | `Ufold2 ->
        functions
          [
            ( "f",
              arrow string_type
                (arrow accumulator_type
                   (arrow (variable variable_a)
                      (arrow (variable variable_b) accumulator_type))) );
            ("acc", accumulator_type);
            ("x", applied variable_a);
            ("y", applied variable_b);
          ]
          body
    | `Unames ->
        functions
          [ ("x", applied variable_a) ]
          (constrained body
             (B.ptyp_constr ~loc
                (lident ~loc type_decl.ptype_name.txt)
                [ string_type ]))
  in
  B.value_binding ~loc ~pat:(pvar ~loc name) ~expr

let uniform_operations = [ `Umap; `Umap2; `Uiter; `Ufold; `Ufold2; `Unames ]

let ugenerate_operation ~module_path operation components =
  List.map
    (fun component ->
      let loc = (List.hd component).utype_decl.ptype_loc in
      B.pstr_value ~loc
        (ucomponent_rec_flag component)
        (List.map (umake_binding ~module_path operation) component))
    components

let uniform_structure ~ctxt type_declarations =
  let declarations, errors = uvalidate ~signature:false type_declarations in
  if errors <> [] then structure_errors errors
  else
    let module_path =
      Expansion_context.Deriver.code_path ctxt |> Code_path.fully_qualified_path
    in
    let components = uordered_components declarations in
    List.concat_map
      (fun operation -> ugenerate_operation ~module_path operation components)
      uniform_operations

let usignature_type operation type_decl =
  let loc = type_decl.ptype_loc in
  let variable_a, variable_b, variable_c, variable_acc =
    ufresh_variables type_decl
  in
  let applied variable =
    B.ptyp_constr ~loc
      (lident ~loc type_decl.ptype_name.txt)
      [ B.ptyp_var ~loc variable ]
  in
  let variable name = B.ptyp_var ~loc name in
  let string_type = B.ptyp_constr ~loc (lident ~loc "string") [] in
  let unit_type = B.ptyp_constr ~loc (lident ~loc "unit") [] in
  let accumulator_type = variable variable_acc in
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  match operation with
  | `Umap ->
      arrow
        (arrow (variable variable_a) (variable variable_b))
        (arrow (applied variable_a) (applied variable_b))
  | `Umap2 ->
      arrow
        (arrow (variable variable_a)
           (arrow (variable variable_b) (variable variable_c)))
        (arrow (applied variable_a)
           (arrow (applied variable_b) (applied variable_c)))
  | `Uiter ->
      arrow
        (arrow (variable variable_a) unit_type)
        (arrow (applied variable_a) unit_type)
  | `Ufold ->
      arrow
        (arrow string_type
           (arrow accumulator_type
              (arrow (variable variable_a) accumulator_type)))
        (arrow accumulator_type (arrow (applied variable_a) accumulator_type))
  | `Ufold2 ->
      arrow
        (arrow string_type
           (arrow accumulator_type
              (arrow (variable variable_a)
                 (arrow (variable variable_b) accumulator_type))))
        (arrow accumulator_type
           (arrow (applied variable_a)
              (arrow (applied variable_b) accumulator_type)))
  | `Unames ->
      arrow (applied variable_a)
        (B.ptyp_constr ~loc
           (lident ~loc type_decl.ptype_name.txt)
           [ string_type ])

let uniform_signature ~ctxt type_declarations =
  let declarations, errors = uvalidate ~signature:true type_declarations in
  if errors <> [] then signature_errors errors
  else
    let add_values udecl =
      let type_decl = udecl.utype_decl in
      let names = uniform_names_for_type type_decl.ptype_name.txt in
      List.map
        (fun operation ->
          let name = uoperation_name operation names in
          let loc = type_decl.ptype_name.loc in
          B.psig_value ~loc
            (B.value_description ~loc ~name:{ txt = name; loc }
               ~type_:(usignature_type operation type_decl)
               ~prim:[]))
        uniform_operations
    in
    Stdlib.ignore ctxt;
    List.concat_map add_values declarations

(* Mirror mode: [@@deriving ptree ~mirror] on a concrete type additionally
   generates [module Uniform] (the payload-generic mirror of the record, with
   all uniform traversals), [to_uniform] packing tensor leaves as
   [Nx.Ptree.tensor], and — when every leaf dtype is statically known —
   [of_uniform] unpacking them back with dtype checks. *)

let strip_ptree_attributes =
  object
    inherit Ast_traverse.map

    method! attributes attributes =
      List.filter
        (fun attribute -> not (is_ptree_attribute attribute.attr_name.txt))
        attributes
  end

let rec shape_has_local shape =
  match shape.desc with
  | Local _ -> true
  | Leaf | Ignored | Using _ -> false
  | Tuple shapes -> List.exists shape_has_local shapes
  | Option shape | List shape | Array shape -> shape_has_local shape

let body_has_local = function
  | Alias shape -> shape_has_local shape
  | Record fields ->
      List.exists (fun (_, shape) -> shape_has_local shape) fields

let body_has_leaf =
  let rec shape_has_leaf shape =
    match shape.desc with
    | Leaf | Using _ -> true
    | Ignored | Local _ -> false
    | Tuple shapes -> List.exists shape_has_leaf shapes
    | Option shape | List shape | Array shape -> shape_has_leaf shape
  in
  function
  | Alias shape -> shape_has_leaf shape
  | Record fields -> List.exists (fun (_, shape) -> shape_has_leaf shape) fields

let rec unwrap_core_type core_type =
  match core_type.ptyp_desc with
  | Ptyp_alias (inner, _) -> unwrap_core_type inner
  | _ -> core_type

let container_argument core_type =
  match (unwrap_core_type core_type).ptyp_desc with
  | Ptyp_constr (_, [ argument ]) -> argument
  | _ -> assert false

let tuple_arguments core_type =
  match (unwrap_core_type core_type).ptyp_desc with
  | Ptyp_tuple elements -> elements
  | _ -> assert false

(* The mirror type of a position: payloads become ['m], delegations become ['m
   M.Uniform.t], static positions keep their original type. *)
let rec mirror_type shape core_type =
  let loc = shape.loc in
  match shape.desc with
  | Leaf -> B.ptyp_var ~loc "m"
  | Ignored -> strip_ptree_attributes#core_type core_type
  | Using module_path ->
      B.ptyp_constr ~loc
        (located_lid ~loc (append_lid (append_lid module_path "Uniform") "t"))
        [ B.ptyp_var ~loc "m" ]
  | Local _ -> assert false
  | Tuple shapes ->
      B.ptyp_tuple ~loc
        (List.map2 mirror_type shapes (tuple_arguments core_type))
  | Option shape ->
      B.ptyp_constr ~loc (lident ~loc "option")
        [ mirror_type shape (container_argument core_type) ]
  | List shape ->
      B.ptyp_constr ~loc (lident ~loc "list")
        [ mirror_type shape (container_argument core_type) ]
  | Array shape ->
      B.ptyp_constr ~loc (lident ~loc "array")
        [ mirror_type shape (container_argument core_type) ]

(* Dtype recovery for [of_uniform]. *)

let dtype_name_of_nx_name = function
  | "float16" | "float16_elt" -> Some "float16"
  | "float32" | "float32_elt" -> Some "float32"
  | "float64" | "float64_elt" -> Some "float64"
  | "bfloat16" | "bfloat16_elt" -> Some "bfloat16"
  | "float8_e4m3" | "float8_e4m3_elt" -> Some "float8_e4m3"
  | "float8_e5m2" | "float8_e5m2_elt" -> Some "float8_e5m2"
  | "int4" | "int4_elt" -> Some "int4"
  | "uint4" | "uint4_elt" -> Some "uint4"
  | "int8" | "int8_elt" -> Some "int8"
  | "uint8" | "uint8_elt" -> Some "uint8"
  | "int16" | "int16_elt" -> Some "int16"
  | "uint16" | "uint16_elt" -> Some "uint16"
  | "int32" | "int32_elt" -> Some "int32"
  | "uint32" | "uint32_elt" -> Some "uint32"
  | "int64" | "int64_elt" -> Some "int64"
  | "uint64" | "uint64_elt" -> Some "uint64"
  | "complex64" | "complex32_elt" -> Some "complex64"
  | "complex128" | "complex64_elt" -> Some "complex128"
  | "bool" | "bool_elt" -> Some "bool"
  | _ -> None

let chop_suffix name suffix =
  let name_length = String.length name in
  let suffix_length = String.length suffix in
  if
    name_length > suffix_length
    && String.equal
         (String.sub name (name_length - suffix_length) suffix_length)
         suffix
  then Some (String.sub name 0 (name_length - suffix_length))
  else None

let dtype_expr_of_elt_type ~loc core_type =
  match (unwrap_core_type core_type).ptyp_desc with
  | Ptyp_constr (path, []) -> (
      match longident_parts path.txt with
      | Some [ name ] | Some [ "Nx"; name ] -> (
          match dtype_name_of_nx_name name with
          | Some dtype_name ->
              Some (ident ~loc (Longident.Ldot (Lident "Nx", dtype_name)))
          | None -> None)
      | _ -> None)
  | _ -> None

let dtype_expr_of_core_type ~loc core_type =
  match (unwrap_core_type core_type).ptyp_desc with
  | Ptyp_constr (path, []) -> (
      match longident_parts path.txt with
      | Some [ name ] | Some [ "Nx"; name ] -> (
          match chop_suffix name "_t" with
          | Some base -> (
              match dtype_name_of_nx_name base with
              | Some dtype_name ->
                  Some (ident ~loc (Longident.Ldot (Lident "Nx", dtype_name)))
              | None -> None)
          | None -> None)
      | _ -> None)
  | Ptyp_constr (path, [ _scalar; elt_type ])
    when is_path path.txt [ "Nx"; "t" ] || is_path path.txt [ "Nx_effect"; "t" ]
    ->
      dtype_expr_of_elt_type ~loc elt_type
  | _ -> None

let rec mirror_dtypes_known shape core_type =
  match shape.desc with
  | Leaf -> Option.is_some (dtype_expr_of_core_type ~loc:shape.loc core_type)
  | Ignored | Using _ -> true
  | Local _ -> assert false
  | Tuple shapes ->
      List.for_all2 mirror_dtypes_known shapes (tuple_arguments core_type)
  | Option shape | List shape | Array shape ->
      mirror_dtypes_known shape (container_argument core_type)

(* to_uniform / of_uniform *)

let pack_leaf ~loc expr =
  B.pexp_construct ~loc
    (located_lid ~loc
       (Longident.Ldot (Longident.Ldot (Lident "Nx", "Ptree"), "P")))
    (Some expr)

let unpack_leaf ~loc ~path ~dtype expr =
  B.pexp_apply ~loc
    (ident ~loc (Longident.parse "Nx.Ptree.unpack"))
    [ (Labelled "at", upath_expr ~loc path); (Nolabel, dtype); (Nolabel, expr) ]

let rec to_uniform_shape shape expr =
  let loc = shape.loc in
  match shape.desc with
  | Leaf -> pack_leaf ~loc expr
  | Ignored -> expr
  | Using module_path ->
      call ~loc (append_lid module_path "to_uniform") [ expr ]
  | Local _ -> assert false
  | Tuple shapes ->
      let names, pattern, tuple =
        tuple_bindings ~loc expr (List.length shapes)
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr:tuple ]
        (B.pexp_tuple ~loc
           (List.map2
              (fun shape name -> to_uniform_shape shape (evar ~loc name))
              shapes names))
  | Option shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      B.pexp_match ~loc expr
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc value)))
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some (to_uniform_shape shape (evar ~loc value))));
        ]
  | List shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      call ~loc
        (Longident.parse "Stdlib.List.map")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc value)
            (to_uniform_shape shape (evar ~loc value));
          expr;
        ]
  | Array shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      call ~loc
        (Longident.parse "Stdlib.Array.map")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc value)
            (to_uniform_shape shape (evar ~loc value));
          expr;
        ]

let rec of_uniform_shape ~path shape core_type expr =
  let loc = shape.loc in
  match shape.desc with
  | Leaf ->
      let dtype = Option.get (dtype_expr_of_core_type ~loc core_type) in
      unpack_leaf ~loc ~path ~dtype expr
  | Ignored -> expr
  | Using module_path ->
      call ~loc (append_lid module_path "of_uniform") [ expr ]
  | Local _ -> assert false
  | Tuple shapes ->
      let arguments = tuple_arguments core_type in
      let names, pattern, tuple =
        tuple_bindings ~loc expr (List.length shapes)
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr:tuple ]
        (B.pexp_tuple ~loc
           (List.mapi
              (fun index shape ->
                of_uniform_shape
                  ~path:(ujoin_literal ~loc path (string_of_int index))
                  shape (List.nth arguments index)
                  (evar ~loc (List.nth names index)))
              shapes))
  | Option shape ->
      let value = gen_symbol ~prefix:"ptree_value" () in
      B.pexp_match ~loc expr
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc value)))
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some
                    (of_uniform_shape ~path shape
                       (container_argument core_type)
                       (evar ~loc value))));
        ]
  | List shape ->
      let index = gen_symbol ~prefix:"ptree_index" () in
      let value = gen_symbol ~prefix:"ptree_value" () in
      call ~loc
        (Longident.parse "Stdlib.List.mapi")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc index)
            (B.pexp_fun ~loc Nolabel None (pvar ~loc value)
               (of_uniform_shape
                  ~path:(uindex_path ~loc path index)
                  shape
                  (container_argument core_type)
                  (evar ~loc value)));
          expr;
        ]
  | Array shape ->
      let index = gen_symbol ~prefix:"ptree_index" () in
      let value = gen_symbol ~prefix:"ptree_value" () in
      call ~loc
        (Longident.parse "Stdlib.Array.mapi")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc index)
            (B.pexp_fun ~loc Nolabel None (pvar ~loc value)
               (of_uniform_shape
                  ~path:(uindex_path ~loc path index)
                  shape
                  (container_argument core_type)
                  (evar ~loc value)));
          expr;
        ]

(* Synthesising the mirror declaration and its uniform traversals *)

let mirror_shapes declaration =
  let rec convert shape =
    let desc =
      match shape.desc with
      | Leaf -> UPayload
      | Ignored -> UStatic
      | Using module_path -> UUsing (append_lid module_path "Uniform")
      | Local _ -> assert false
      | Tuple shapes -> UTuple (List.map convert shapes)
      | Option shape -> UOption (convert shape)
      | List shape -> UList (convert shape)
      | Array shape -> UArray (convert shape)
    in
    { udesc = desc; uloc = shape.loc }
  in
  match declaration.body with
  | Some (Alias shape) -> `Alias (convert shape)
  | Some (Record fields) ->
      `Record (List.map (fun (label, shape) -> (label, convert shape)) fields)
  | None -> assert false

let synthesise_mirror_decl ~loc declaration =
  let params = [ (B.ptyp_var ~loc "m", (NoVariance, NoInjectivity)) ] in
  match declaration.body with
  | Some (Alias shape) ->
      let manifest =
        mirror_type shape (Option.get declaration.type_decl.ptype_manifest)
      in
      {
        ptype_name = { txt = "t"; loc };
        ptype_params = params;
        ptype_cstrs = [];
        ptype_kind = Ptype_abstract;
        ptype_private = Public;
        ptype_manifest = Some manifest;
        ptype_attributes = [];
        ptype_loc = loc;
      }
  | Some (Record fields) ->
      let labels =
        List.map
          (fun (label, shape) ->
            {
              label with
              pld_type = mirror_type shape label.pld_type;
              pld_attributes = [];
            })
          fields
      in
      {
        ptype_name = { txt = "t"; loc };
        ptype_params = params;
        ptype_cstrs = [];
        ptype_kind = Ptype_record labels;
        ptype_private = Public;
        ptype_manifest = None;
        ptype_attributes = [];
        ptype_loc = loc;
      }
  | None -> assert false

let mirror_udeclaration ~loc declaration =
  let synth = synthesise_mirror_decl ~loc declaration in
  let ubody =
    match (mirror_shapes declaration, synth.ptype_kind) with
    | `Alias ushape, _ -> UAlias ushape
    | `Record fields, Ptype_record synth_labels ->
        URecord
          (List.map2
             (fun (_, ushape) synth_label -> (synth_label, ushape))
             fields synth_labels)
    | `Record _, _ -> assert false
  in
  { utype_decl = synth; ubody = Some ubody; udependencies = String_set.empty }

let mirror_check declarations =
  match declarations with
  | [ declaration ] -> (
      match declaration.body with
      | None ->
          Error
            [
              Location.Error.make ~loc:declaration.type_decl.ptype_name.loc
                "ppx_ptree: mirror mode requires the type's definition; \
                 abstract types cannot be mirrored"
                ~sub:[];
            ]
      | Some body ->
          if body_has_local body then
            Error
              [
                Location.Error.make ~loc:declaration.type_decl.ptype_name.loc
                  "ppx_ptree: mirror mode does not support locally declared or \
                   recursive types; move the sub-structure to its own module \
                   deriving [ptree ~mirror]"
                  ~sub:[];
              ]
          else if not (body_has_leaf body) then
            Error
              [
                Location.Error.make ~loc:declaration.type_decl.ptype_name.loc
                  "ppx_ptree: mirror mode requires at least one tensor leaf"
                  ~sub:[];
              ]
          else Ok declaration)
  | _ ->
      let loc =
        match declarations with
        | declaration :: _ -> declaration.type_decl.ptype_name.loc
        | [] -> Location.none
      in
      Error
        [
          Location.Error.make ~loc
            "ppx_ptree: mirror mode supports a single type declaration" ~sub:[];
        ]

let mirror_declaration_dtypes_known declaration =
  match declaration.body with
  | Some (Alias shape) ->
      mirror_dtypes_known shape
        (Option.get declaration.type_decl.ptype_manifest)
  | Some (Record fields) ->
      List.for_all
        (fun (label, shape) -> mirror_dtypes_known shape label.pld_type)
        fields
  | None -> false

let tensor_uniform_type ~loc =
  B.ptyp_constr ~loc
    (located_lid ~loc (Longident.Ldot (Lident "Uniform", "t")))
    [
      B.ptyp_constr ~loc
        (located_lid ~loc (Longident.parse "Nx.Ptree.tensor"))
        [];
    ]

let mirror_module_items ~module_path declaration =
  let loc = declaration.type_decl.ptype_loc in
  let udecl = mirror_udeclaration ~loc declaration in
  let type_item = B.pstr_type ~loc Nonrecursive [ udecl.utype_decl ] in
  let module_path =
    if module_path = "" then "Uniform" else module_path ^ ".Uniform"
  in
  let traversals =
    List.map
      (fun operation ->
        B.pstr_value ~loc Nonrecursive
          [ umake_binding ~module_path operation udecl ])
      uniform_operations
  in
  type_item :: traversals

let mirror_to_uniform ~loc declaration =
  let body =
    match declaration.body with
    | Some (Alias shape) -> to_uniform_shape shape (evar ~loc "x")
    | Some (Record fields) ->
        B.pexp_record ~loc
          (List.map
             (fun (label, shape) ->
               ( located_lid ~loc:label.pld_name.loc
                   (Longident.Ldot (Lident "Uniform", label.pld_name.txt)),
                 to_uniform_shape shape
                   (field ~loc:label.pld_loc (evar ~loc "x") label) ))
             fields)
          None
    | None -> assert false
  in
  B.pstr_value ~loc Nonrecursive
    [
      B.value_binding ~loc ~pat:(pvar ~loc "to_uniform")
        ~expr:
          (B.pexp_fun ~loc Nolabel None
             (constrained_parameter ~loc "x"
                (declared_type declaration.type_decl))
             (B.pexp_constraint ~loc body (tensor_uniform_type ~loc)));
    ]

let mirror_of_uniform ~loc declaration =
  let body =
    match declaration.body with
    | Some (Alias shape) ->
        of_uniform_shape ~path:Uroot shape
          (Option.get declaration.type_decl.ptype_manifest)
          (evar ~loc "u")
    | Some (Record fields) ->
        B.pexp_record ~loc
          (List.map
             (fun (label, shape) ->
               ( lident ~loc:label.pld_name.loc label.pld_name.txt,
                 of_uniform_shape ~path:(Uliteral label.pld_name.txt) shape
                   label.pld_type
                   (B.pexp_field ~loc:label.pld_loc (evar ~loc "u")
                      (located_lid ~loc:label.pld_name.loc
                         (Longident.Ldot (Lident "Uniform", label.pld_name.txt))))
               ))
             fields)
          None
    | None -> assert false
  in
  B.pstr_value ~loc Nonrecursive
    [
      B.value_binding ~loc ~pat:(pvar ~loc "of_uniform")
        ~expr:
          (B.pexp_fun ~loc Nolabel None
             (constrained_parameter ~loc "u" (tensor_uniform_type ~loc))
             (B.pexp_constraint ~loc body (declared_type declaration.type_decl)));
    ]

let mirror_structure ~ctxt type_declarations =
  let declarations, errors = validate ~signature:false type_declarations in
  if errors <> [] then []
  else
    match mirror_check declarations with
    | Error errors -> structure_errors errors
    | Ok declaration ->
        let loc = declaration.type_decl.ptype_loc in
        let module_path =
          Expansion_context.Deriver.code_path ctxt
          |> Code_path.fully_qualified_path
        in
        let module_item =
          B.pstr_module ~loc
            (B.module_binding ~loc
               ~name:{ txt = Some "Uniform"; loc }
               ~expr:
                 (B.pmod_structure ~loc
                    (mirror_module_items ~module_path declaration)))
        in
        let conversions =
          mirror_to_uniform ~loc declaration
          ::
          (if mirror_declaration_dtypes_known declaration then
             [ mirror_of_uniform ~loc declaration ]
           else [])
        in
        module_item :: conversions

let mirror_signature ~ctxt type_declarations =
  let declarations, errors = validate ~signature:true type_declarations in
  if errors <> [] then []
  else
    match mirror_check declarations with
    | Error errors -> signature_errors errors
    | Ok declaration ->
        Stdlib.ignore ctxt;
        let loc = declaration.type_decl.ptype_loc in
        let udecl = mirror_udeclaration ~loc declaration in
        let type_item = B.psig_type ~loc Nonrecursive [ udecl.utype_decl ] in
        let traversal_values =
          List.map
            (fun operation ->
              let name =
                uoperation_name operation (uniform_names_for_type "t")
              in
              B.psig_value ~loc
                (B.value_description ~loc ~name:{ txt = name; loc }
                   ~type_:(usignature_type operation udecl.utype_decl)
                   ~prim:[]))
            uniform_operations
        in
        let module_item =
          B.psig_module ~loc
            (B.module_declaration ~loc
               ~name:{ txt = Some "Uniform"; loc }
               ~type_:(B.pmty_signature ~loc (type_item :: traversal_values)))
        in
        let to_uniform_item =
          B.psig_value ~loc
            (B.value_description ~loc
               ~name:{ txt = "to_uniform"; loc }
               ~type_:
                 (B.ptyp_arrow ~loc Nolabel
                    (declared_type declaration.type_decl)
                    (tensor_uniform_type ~loc))
               ~prim:[])
        in
        let of_uniform_items =
          if mirror_declaration_dtypes_known declaration then
            [
              B.psig_value ~loc
                (B.value_description ~loc
                   ~name:{ txt = "of_uniform"; loc }
                   ~type_:
                     (B.ptyp_arrow ~loc Nolabel (tensor_uniform_type ~loc)
                        (declared_type declaration.type_decl))
                   ~prim:[]);
            ]
          else []
        in
        module_item :: to_uniform_item :: of_uniform_items

(* Deriver registration: one deriver, dispatching on the shape of the
   declaration group. *)

let mirror_on_uniform_error type_declarations =
  let loc =
    match type_declarations with
    | type_decl :: _ -> type_decl.ptype_name.loc
    | [] -> Location.none
  in
  Location.Error.make ~loc
    "ppx_ptree: [~mirror] applies to concrete types; uniform (parameterised) \
     types are their own mirror"
    ~sub:[]

let ptree_structure ~ctxt (recursive, type_declarations) mirror =
  if uniform_group type_declarations then
    if mirror then
      structure_errors [ mirror_on_uniform_error type_declarations ]
    else uniform_structure ~ctxt type_declarations
  else
    let base = structure_generator ~ctxt (recursive, type_declarations) in
    if mirror then base @ mirror_structure ~ctxt type_declarations else base

let ptree_signature ~ctxt (recursive, type_declarations) mirror =
  if uniform_group type_declarations then
    if mirror then
      signature_errors [ mirror_on_uniform_error type_declarations ]
    else uniform_signature ~ctxt type_declarations
  else
    let base = signature_generator ~ctxt (recursive, type_declarations) in
    if mirror then base @ mirror_signature ~ctxt type_declarations else base

let () =
  Deriving.add "ptree"
    ~str_type_decl:
      (Deriving.Generator.V2.make ~unused_code_warnings:true
         Deriving.Args.(empty +> flag "mirror")
         ptree_structure)
    ~sig_type_decl:
      (Deriving.Generator.V2.make ~unused_code_warnings:true
         Deriving.Args.(empty +> flag "mirror")
         ptree_signature)
  |> Deriving.ignore
