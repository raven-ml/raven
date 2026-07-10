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
       (call ~loc (Longident.Lident "<>")
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
       (call ~loc (Longident.Lident "<>")
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

let () =
  Deriving.add "ptree"
    ~str_type_decl:
      (Deriving.Generator.V2.make_noarg ~unused_code_warnings:true
         structure_generator)
    ~sig_type_decl:
      (Deriving.Generator.V2.make_noarg ~unused_code_warnings:true
         signature_generator)
  |> Deriving.ignore
