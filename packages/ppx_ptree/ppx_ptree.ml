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

(* ——————————————————————————————— *)
(*  Gtree (uniform homomorphic)   *)
(* ——————————————————————————————— *)

type unames = {
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

let unames_for_type = function
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

(* Occurrence analysis: does the payload parameter occur at this position? *)
let rec uclassify_inner validation param_name core_type =
  let loc = core_type.ptyp_loc in
  match core_type.ptyp_desc with
  | Ptyp_var name when name = param_name -> ushape ~loc UPayload
  | Ptyp_var _ -> ushape ~loc UStatic
  | Ptyp_constr (path, args) when container_kind path.txt <> None ->
      uclassify_container validation param_name loc path.txt args
  | Ptyp_constr (path, _)
    when is_nx_alias path.txt
         || is_path path.txt [ "Nx"; "t" ]
         || is_path path.txt [ "Nx_effect"; "t" ] ->
      ushape ~loc UStatic
  | Ptyp_constr (path, [ arg ]) when Option.is_some (local_name path.txt) -> (
      let child = uclassify_inner validation param_name arg in
      if child.udesc = UStatic then ushape ~loc UStatic
      else
        match local_name path.txt with
        | Some name -> ushape ~loc (ULocal name)
        | None -> assert false)
  | Ptyp_constr (path, _) when Option.is_some (external_primary path.txt) ->
      ushape ~loc (UUsing (Option.get (external_primary path.txt)))
  | Ptyp_constr (path, _) ->
      add_error validation ~loc
        "ppx_ptree: cannot infer a gtree role for [%a]; the type is applied to \
         a type parameter but is not a known container or tensor. Annotate \
         with [@gtree.ignore] or use a type that does not mention the payload \
         parameter"
        Pprintast.longident path.txt;
      ushape ~loc UStatic
  | Ptyp_tuple args ->
      ushape ~loc
        (UTuple (List.map (uclassify_inner validation param_name) args))
  | Ptyp_alias (inner, _) -> uclassify_inner validation param_name inner
  | Ptyp_any ->
      add_error validation ~loc
        "ppx_ptree: wildcard type [_] has no derivable gtree shape";
      ushape ~loc UStatic
  | Ptyp_arrow _ ->
      add_error validation ~loc
        "ppx_ptree: function types are not gtree shapes; annotate with \
         [@ptree.ignore] or [@gtree.ignore]";
      ushape ~loc UStatic
  | Ptyp_object _ ->
      add_error validation ~loc "ppx_ptree: object types are not supported";
      ushape ~loc UStatic
  | Ptyp_class _ ->
      add_error validation ~loc "ppx_ptree: class types are not supported";
      ushape ~loc UStatic
  | Ptyp_variant _ ->
      add_error validation ~loc
        "ppx_ptree: polymorphic variants are not supported in gtree";
      ushape ~loc UStatic
  | Ptyp_poly _ ->
      add_error validation ~loc
        "ppx_ptree: explicitly polymorphic field types are not supported";
      ushape ~loc UStatic
  | Ptyp_package _ ->
      add_error validation ~loc
        "ppx_ptree: first-class module types are not supported";
      ushape ~loc UStatic
  | Ptyp_extension _ ->
      add_error validation ~loc "ppx_ptree: extension types are not supported";
      ushape ~loc UStatic
  | Ptyp_open _ ->
      add_error validation ~loc
        "ppx_ptree: locally opened types are not supported";
      ushape ~loc UStatic

and uclassify_container validation param_name loc path args =
  match (container_kind path, args) with
  | Some `Option, [ arg ] ->
      ushape ~loc (UOption (uclassify_inner validation param_name arg))
  | Some `List, [ arg ] ->
      ushape ~loc (UList (uclassify_inner validation param_name arg))
  | Some `Array, [ arg ] ->
      ushape ~loc (UArray (uclassify_inner validation param_name arg))
  | Some _, _ ->
      add_error validation ~loc
        "ppx_ptree: container types must have exactly one argument";
      ushape ~loc UStatic
  | None, _ -> assert false

let rec udependencies shape =
  match shape.udesc with
  | UPayload | UStatic | UUsing _ -> String_set.empty
  | ULocal name -> String_set.singleton name
  | UTuple shapes ->
      List.fold_left
        (fun result s -> String_set.union result (udependencies s))
        String_set.empty shapes
  | UOption shape | UList shape | UArray shape -> udependencies shape

let uvalidate_declaration ~signature validation type_decl param_name =
  let body =
    match (type_decl.ptype_kind, type_decl.ptype_manifest) with
    | Ptype_record labels, None ->
        Some
          (URecord
             (List.map
                (fun label ->
                  (label, uclassify_inner validation param_name label.pld_type))
                labels))
    | Ptype_abstract, Some manifest ->
        Some (UAlias (uclassify_inner validation param_name manifest))
    | Ptype_abstract, None when signature -> None
    | Ptype_abstract, None ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: an abstract implementation has no derivable gtree shape";
        None
    | Ptype_record _, Some _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: representation re-exports with records are not supported";
        None
    | Ptype_variant _, _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: variant gtree types are not supported in version 1";
        None
    | Ptype_open, _ ->
        add_error validation ~loc:type_decl.ptype_name.loc
          "ppx_ptree: extensible variants are not supported";
        None
  in
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

let ufind_payload_param validation type_decls =
  match type_decls with
  | [] -> None
  | first :: _ -> (
      let params = first.ptype_params in
      match params with
      | [ (param, _) ] -> (
          match param.ptyp_desc with
          | Ptyp_var name -> Some name
          | _ ->
              add_error validation ~loc:param.ptyp_loc
                "ppx_ptree: gtree type parameter must be a type variable";
              None)
      | [] -> None
      | _ ->
          add_error validation ~loc:first.ptype_name.loc
            "ppx_ptree: gtree types support exactly one type parameter";
          None)

let uhas_payload shape =
  let rec loop s =
    match s.udesc with
    | UPayload -> true
    | UStatic -> false
    | UTuple shapes -> List.exists loop shapes
    | UOption shape | UList shape | UArray shape -> loop shape
    | ULocal _ | UUsing _ ->
        (* A delegation implies the payload is present in sub-module. Treat as
           payload at this level — the sub-module check is a compile error. *)
        true
  in
  loop shape

let uvalidate ~signature type_declarations =
  let validation = { errors_rev = [] } in
  (* Only process if we find a type with a payload param. *)
  let payload_param = ufind_payload_param validation type_declarations in
  let declarations =
    match payload_param with
    | None -> []
    | Some param_name ->
        List.map
          (fun td ->
            let decl =
              uvalidate_declaration ~signature validation td param_name
            in
            (* Check that at least one Payload position exists. *)
            (match decl.ubody with
            | Some (UAlias s) ->
                if not (uhas_payload s) then
                  add_error validation ~loc:td.ptype_name.loc
                    "ppx_ptree: gtree type parameter [%s] does not occur in a \
                     payload position; it only appears inside tensor leaf \
                     types. Use [@@deriving ptree, gtree] for a mirror view \
                     instead"
                    param_name
            | Some (URecord fields) ->
                if not (List.exists (fun (_, s) -> uhas_payload s) fields) then
                  add_error validation ~loc:td.ptype_name.loc
                    "ppx_ptree: gtree type parameter [%s] does not occur in \
                     any field as a payload; it only appears inside tensor \
                     leaf types. Use [@@deriving ptree, gtree] for a mirror \
                     view instead"
                    param_name
            | None -> ());
            decl)
          type_declarations
  in
  let errors = List.rev validation.errors_rev in
  (declarations, errors)

(* Wrap udeclarations as declarations for the SCC ordering machinery. *)
let udecl_to_decl udecl =
  {
    type_decl = udecl.utype_decl;
    body = None;
    dependencies = udecl.udependencies;
  }

let uordered_components declarations =
  let wrapped = List.map udecl_to_decl declarations in
  let components = strongly_connected_components wrapped in
  let result = ref [] in
  List.iter
    (fun comp_names ->
      let comp_decls =
        List.map
          (fun name ->
            List.find
              (fun udecl -> udecl.utype_decl.ptype_name.txt = name)
              declarations)
          comp_names
      in
      result := comp_decls :: !result)
    components;
  List.rev !result

let ucomponent_rec_flag component =
  match component with
  | [] -> assert false
  | [ udecl ] ->
      if String_set.mem udecl.utype_decl.ptype_name.txt udecl.udependencies then
        Recursive
      else Nonrecursive
  | _ -> Recursive

(* ——— Codegen ——— *)

let payload_callback_shape ~loc var_a var_b =
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  let type_ = B.ptyp_var ~loc var_a in
  let result = B.ptyp_var ~loc var_b in
  arrow type_ result

let payload_callback2_shape ~loc var_a var_b var_c =
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  let type_a = B.ptyp_var ~loc var_a in
  let type_b = B.ptyp_var ~loc var_b in
  let type_c = B.ptyp_var ~loc var_c in
  arrow type_a (arrow type_b type_c)

let callback_with_path_shape ~loc callback_path callback_payload acc_var =
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  let string_type = B.ptyp_constr ~loc (lident ~loc "string") [] in
  arrow string_type (arrow callback_payload (arrow acc_var acc_var))

let ufunction_expression ~loc ~callback_type ~input_types body =
  let callback = constrained_parameter ~loc "f" callback_type in
  let inputs =
    List.mapi
      (fun index type_ ->
        let name =
          match index with
          | 0 -> "x"
          | 1 -> "y"
          | 2 -> "z"
          | _ -> Printf.sprintf "p%d" index
        in
        constrained_parameter ~loc name type_)
      input_types
  in
  List.fold_right
    (fun pattern body -> B.pexp_fun ~loc Nolabel None pattern body)
    (callback :: inputs) body

let unames_target_type ~loc type_ =
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  let string_type = B.ptyp_constr ~loc (lident ~loc "string") [] in
  arrow type_ string_type

(* — map — rank-1, much simpler than ptree map *)
let rec umap_expr fvar shape expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> call ~loc (Lident fvar) [ expr ]
  | UStatic -> expr
  | ULocal name ->
      call ~loc
        (Longident.Ldot (Lident (names_for_type name).map, fvar))
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

(* — map2 — *)
let rec umap2_expr ~function_path fvar shape left right =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> call ~loc (Lident fvar) [ left; right ]
  | UStatic -> left
  | ULocal name ->
      call ~loc
        (Longident.Ldot (Lident (names_for_type name).map2, fvar))
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
            umap2_expr ~function_path fvar shape
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
                    (umap2_expr ~function_path fvar shape
                       (evar ~loc:shape.uloc left_value)
                       (evar ~loc:shape.uloc right_value))));
          B.case ~lhs:(B.ppat_any ~loc) ~guard:None
            ~rhs:
              (invalid_argument ~loc
                 (mismatch_message function_path "option constructor" []));
        ]
  | UList shape -> umap2_list ~loc ~function_path fvar shape left right
  | UArray shape -> umap2_array ~loc ~function_path fvar shape left right

and umap2_list ~loc ~function_path fvar shape left right =
  let left_name = gen_symbol ~prefix:"ptree_left" () in
  let right_name = gen_symbol ~prefix:"ptree_right" () in
  let left_length = gen_symbol ~prefix:"ptree_left_length" () in
  let right_length = gen_symbol ~prefix:"ptree_right_length" () in
  let left_value = gen_symbol ~prefix:"ptree_left_value" () in
  let right_value = gen_symbol ~prefix:"ptree_right_value" () in
  let mapper =
    B.pexp_fun ~loc Nolabel None (pvar ~loc left_value)
      (B.pexp_fun ~loc Nolabel None (pvar ~loc right_value)
         (umap2_expr ~function_path fvar shape
            (evar ~loc:shape.uloc left_value)
            (evar ~loc:shape.uloc right_value)))
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
       (invalid_argument ~loc (mismatch_message function_path "list length" []))
       (Some
          (call ~loc
             (Longident.parse "Stdlib.List.map2")
             [ mapper; evar ~loc left_name; evar ~loc right_name ])))

and umap2_array ~loc ~function_path fvar shape left right =
  let left_name = gen_symbol ~prefix:"ptree_left" () in
  let right_name = gen_symbol ~prefix:"ptree_right" () in
  let left_length = gen_symbol ~prefix:"ptree_left_length" () in
  let right_length = gen_symbol ~prefix:"ptree_right_length" () in
  let index = gen_symbol ~prefix:"ptree_index" () in
  let value_at arr =
    call ~loc
      (Longident.parse "Stdlib.Array.unsafe_get")
      [ evar ~loc arr; evar ~loc index ]
  in
  let array_initializer =
    B.pexp_fun ~loc Nolabel None (pvar ~loc index)
      (umap2_expr ~function_path fvar shape (value_at left_name)
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
          (mismatch_message function_path "array length" []))
       (Some
          (call ~loc
             (Longident.parse "Stdlib.Array.init")
             [ evar ~loc left_length; array_initializer ])))

(* — iter — *)
let rec uiter_expr fvar shape expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> call ~loc (Lident fvar) [ expr ]
  | UStatic -> B.eunit ~loc
  | ULocal name ->
      call ~loc
        (Longident.Ldot (Lident (names_for_type name).iter, fvar))
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

(* — fold (with paths) —

   ~path: an OCaml expression of type string evaluating to the path segment for
   this position (e.g. the field name, or "prefix." ^ idx for list items). At
   the top level this is the empty string literal "". *)

let rec ufold_expr ~path fvar shape expr acc_expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload ->
      (* f path acc expr *)
      call ~loc (Lident fvar) [ path; acc_expr; expr ]
  | UStatic -> acc_expr
  | ULocal name ->
      let fold_name = (unames_for_type name).ufold in
      let sub_path = gen_symbol ~prefix:"u_p" () in
      let sub_acc = gen_symbol ~prefix:"u_a" () in
      let sub_x = gen_symbol ~prefix:"u_x" () in
      (* name.fold (fun p a x -> f (path ^ "." ^ p) a x) acc expr *)
      let callback =
        B.pexp_fun ~loc Nolabel None (pvar ~loc sub_path)
          (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_acc)
             (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_x)
                (call ~loc (Lident fvar)
                   [
                     call ~loc (Longident.Lident "^")
                       [
                         call ~loc (Longident.Lident "^")
                           [ path; B.estring ~loc "." ];
                         evar ~loc sub_path;
                       ];
                     evar ~loc sub_acc;
                     evar ~loc sub_x;
                   ])))
      in
      call ~loc
        (Longident.Ldot (Lident name, fold_name))
        [ callback; acc_expr; expr ]
  | UUsing module_path ->
      let sub_path = gen_symbol ~prefix:"u_p" () in
      let sub_acc = gen_symbol ~prefix:"u_a" () in
      let sub_x = gen_symbol ~prefix:"u_x" () in
      let callback =
        B.pexp_fun ~loc Nolabel None (pvar ~loc sub_path)
          (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_acc)
             (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_x)
                (call ~loc (Lident fvar)
                   [
                     call ~loc (Longident.Lident "^")
                       [
                         call ~loc (Longident.Lident "^")
                           [ path; B.estring ~loc "." ];
                         evar ~loc sub_path;
                       ];
                     evar ~loc sub_acc;
                     evar ~loc sub_x;
                   ])))
      in
      call ~loc (append_lid module_path "fold") [ callback; acc_expr; expr ]
  | UTuple shapes ->
      let names, pattern, tuple =
        tuple_bindings ~loc expr (List.length shapes)
      in
      (* let (x0, x1, ...) = expr in f (path ^ ".0") x0 (f (path ^ ".1") x1 (...
         acc)) *)
      let rec thread idx acc shapes names =
        match (shapes, names) with
        | [], [] -> acc
        | shape :: rest_shapes, name :: rest_names ->
            let child_path =
              call ~loc (Longident.Lident "^")
                [
                  call ~loc (Longident.Lident "^") [ path; B.estring ~loc "." ];
                  B.estring ~loc (string_of_int idx);
                ]
            in
            ufold_expr ~path:child_path fvar shape (evar ~loc name)
              (thread (idx + 1) acc rest_shapes rest_names)
        | _ -> assert false
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr:tuple ]
        (thread 0 acc_expr shapes names)
  | UOption shape ->
      let value = gen_symbol ~prefix:"u_opt" () in
      B.pexp_match ~loc expr
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None ~rhs:acc_expr;
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc value)))
            ~guard:None
            ~rhs:
              (ufold_expr ~path fvar shape
                 (evar ~loc:shape.uloc value)
                 acc_expr);
        ]
  | UList shape ->
      let acc_p = gen_symbol ~prefix:"u_a" () in
      let idx_v = gen_symbol ~prefix:"u_i" () in
      let val_v = gen_symbol ~prefix:"u_v" () in
      let pair_v = gen_symbol ~prefix:"u_p" () in
      let child_path =
        call ~loc (Longident.Lident "^")
          [
            call ~loc (Longident.Lident "^") [ path; B.estring ~loc "." ];
            call ~loc
              (Longident.parse "Stdlib.string_of_int")
              [ evar ~loc idx_v ];
          ]
      in
      let inner_body =
        B.pexp_let ~loc Nonrecursive
          [
            B.value_binding ~loc
              ~pat:(B.ppat_tuple ~loc [ pvar ~loc idx_v; pvar ~loc val_v ])
              ~expr:(evar ~loc pair_v);
          ]
          (ufold_expr ~path:child_path fvar shape (evar ~loc val_v)
             (evar ~loc acc_p))
      in
      call ~loc
        (Longident.parse "Stdlib.List.fold_left")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc acc_p)
            (B.pexp_fun ~loc Nolabel None (pvar ~loc pair_v) inner_body);
          acc_expr;
          call ~loc
            (Longident.parse "Stdlib.List.mapi")
            [
              B.pexp_fun ~loc Nolabel None (pvar ~loc idx_v)
                (B.pexp_fun ~loc Nolabel None (pvar ~loc val_v)
                   (B.pexp_tuple ~loc [ evar ~loc idx_v; evar ~loc val_v ]));
              expr;
            ];
        ]
  | UArray shape ->
      let acc_p = gen_symbol ~prefix:"u_a" () in
      let idx_v = gen_symbol ~prefix:"u_i" () in
      let val_v = gen_symbol ~prefix:"u_v" () in
      let pair_v = gen_symbol ~prefix:"u_p" () in
      let child_path =
        call ~loc (Longident.Lident "^")
          [
            call ~loc (Longident.Lident "^") [ path; B.estring ~loc "." ];
            call ~loc
              (Longident.parse "Stdlib.string_of_int")
              [ evar ~loc idx_v ];
          ]
      in
      let inner_body =
        B.pexp_let ~loc Nonrecursive
          [
            B.value_binding ~loc
              ~pat:(B.ppat_tuple ~loc [ pvar ~loc idx_v; pvar ~loc val_v ])
              ~expr:(evar ~loc pair_v);
          ]
          (ufold_expr ~path:child_path fvar shape (evar ~loc val_v)
             (evar ~loc acc_p))
      in
      call ~loc
        (Longident.parse "Stdlib.Array.fold_left")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc acc_p)
            (B.pexp_fun ~loc Nolabel None (pvar ~loc pair_v) inner_body);
          acc_expr;
          call ~loc
            (Longident.parse "Stdlib.Array.mapi")
            [
              B.pexp_fun ~loc Nolabel None (pvar ~loc idx_v)
                (B.pexp_fun ~loc Nolabel None (pvar ~loc val_v)
                   (B.pexp_tuple ~loc [ evar ~loc idx_v; evar ~loc val_v ]));
              expr;
            ];
        ]

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

(* ——————————————————————————————— *)
(*  Gtree — fold2, names, &        *)
(*  structure / signature           *)
(*  generators and deriver          *)
(* ——————————————————————————————— *)

(* For fold2, the path threading is identical to fold; we just pass two value
   expressions (left and right) to the callback, include structural equality
   checks for containers, and skip static fields. *)

let rec ufold2_expr ~path fvar shape left right acc_expr =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload -> call ~loc (Lident fvar) [ path; acc_expr; left; right ]
  | UStatic -> acc_expr
  | ULocal name ->
      let fold2_name = (unames_for_type name).ufold2 in
      let sub_p = gen_symbol ~prefix:"u2_p" () in
      let sub_a = gen_symbol ~prefix:"u2_a" () in
      let sub_x = gen_symbol ~prefix:"u2_x" () in
      let sub_y = gen_symbol ~prefix:"u2_y" () in
      let callback =
        B.pexp_fun ~loc Nolabel None (pvar ~loc sub_p)
          (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_a)
             (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_x)
                (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_y)
                   (call ~loc (Lident fvar)
                      [
                        call ~loc (Longident.Lident "^")
                          [
                            call ~loc (Longident.Lident "^")
                              [ path; B.estring ~loc "." ];
                            evar ~loc sub_p;
                          ];
                        evar ~loc sub_a;
                        evar ~loc sub_x;
                        evar ~loc sub_y;
                      ]))))
      in
      call ~loc
        (Longident.Ldot (Lident name, fold2_name))
        [ callback; acc_expr; left; right ]
  | UUsing module_path ->
      let sub_p = gen_symbol ~prefix:"u2_p" () in
      let sub_a = gen_symbol ~prefix:"u2_a" () in
      let sub_x = gen_symbol ~prefix:"u2_x" () in
      let sub_y = gen_symbol ~prefix:"u2_y" () in
      let callback =
        B.pexp_fun ~loc Nolabel None (pvar ~loc sub_p)
          (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_a)
             (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_x)
                (B.pexp_fun ~loc Nolabel None (pvar ~loc sub_y)
                   (call ~loc (Lident fvar)
                      [
                        call ~loc (Longident.Lident "^")
                          [
                            call ~loc (Longident.Lident "^")
                              [ path; B.estring ~loc "." ];
                            evar ~loc sub_p;
                          ];
                        evar ~loc sub_a;
                        evar ~loc sub_x;
                        evar ~loc sub_y;
                      ]))))
      in
      call ~loc
        (append_lid module_path "fold2")
        [ callback; acc_expr; left; right ]
  | UTuple shapes ->
      let left_names, left_pattern, left_tuple =
        tuple_bindings ~loc left (List.length shapes)
      in
      let right_names, right_pattern, right_tuple =
        tuple_bindings ~loc right (List.length shapes)
      in
      let rec thread idx acc shapes lnames rnames =
        match (shapes, lnames, rnames) with
        | [], [], [] -> acc
        | shape :: rest_shapes, lname :: rest_lnames, rname :: rest_rnames ->
            let child_path =
              call ~loc (Longident.Lident "^")
                [
                  call ~loc (Longident.Lident "^") [ path; B.estring ~loc "." ];
                  B.estring ~loc (string_of_int idx);
                ]
            in
            ufold2_expr ~path:child_path fvar shape (evar ~loc lname)
              (evar ~loc rname)
              (thread (idx + 1) acc rest_shapes rest_lnames rest_rnames)
        | _ -> assert false
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:left_pattern ~expr:left_tuple ]
        (B.pexp_let ~loc Nonrecursive
           [ B.value_binding ~loc ~pat:right_pattern ~expr:right_tuple ]
           (thread 0 acc_expr shapes left_names right_names))
  | UOption shape ->
      let lv = gen_symbol ~prefix:"u2_l" () in
      let rv = gen_symbol ~prefix:"u2_r" () in
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
            ~guard:None ~rhs:acc_expr;
          B.case
            ~lhs:
              (B.ppat_tuple ~loc
                 [
                   construct_pattern ~loc "Some" (Some (pvar ~loc lv));
                   construct_pattern ~loc "Some" (Some (pvar ~loc rv));
                 ])
            ~guard:None
            ~rhs:
              (ufold2_expr ~path fvar shape (evar ~loc lv) (evar ~loc rv)
                 acc_expr);
          B.case ~lhs:(B.ppat_any ~loc) ~guard:None
            ~rhs:(invalid_argument ~loc "ufold2: option constructor mismatch");
        ]
  | UList shape ->
      let acc_p = gen_symbol ~prefix:"u2_a" () in
      let idx_v = gen_symbol ~prefix:"u2_i" () in
      let val_v = gen_symbol ~prefix:"u2_v" () in
      let val_w = gen_symbol ~prefix:"u2_w" () in
      let pair_v = gen_symbol ~prefix:"u2_p" () in
      let child_path =
        call ~loc (Longident.Lident "^")
          [
            call ~loc (Longident.Lident "^") [ path; B.estring ~loc "." ];
            call ~loc
              (Longident.parse "Stdlib.string_of_int")
              [ evar ~loc idx_v ];
          ]
      in
      let inner_body =
        B.pexp_let ~loc Nonrecursive
          [
            B.value_binding ~loc
              ~pat:(B.ppat_tuple ~loc [ pvar ~loc idx_v; pvar ~loc val_v ])
              ~expr:(evar ~loc pair_v);
          ]
          (ufold2_expr ~path:child_path fvar shape (evar ~loc val_v)
             (evar ~loc val_w) (evar ~loc acc_p))
      in
      call ~loc
        (Longident.parse "Stdlib.List.fold_left2")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc acc_p)
            (B.pexp_fun ~loc Nolabel None (pvar ~loc pair_v)
               (B.pexp_fun ~loc Nolabel None (pvar ~loc val_w) inner_body));
          acc_expr;
          call ~loc
            (Longident.parse "Stdlib.List.mapi")
            [
              B.pexp_fun ~loc Nolabel None (pvar ~loc idx_v)
                (B.pexp_fun ~loc Nolabel None (pvar ~loc val_v)
                   (B.pexp_tuple ~loc [ evar ~loc idx_v; evar ~loc val_v ]));
              left;
            ];
          right;
        ]
  | UArray shape ->
      let idx_v = gen_symbol ~prefix:"u2_i" () in
      let val_v = gen_symbol ~prefix:"u2_v" () in
      let val_w = gen_symbol ~prefix:"u2_w" () in
      let len = gen_symbol ~prefix:"u2_n" () in
      let child_path =
        call ~loc (Longident.Lident "^")
          [
            call ~loc (Longident.Lident "^") [ path; B.estring ~loc "." ];
            call ~loc
              (Longident.parse "Stdlib.string_of_int")
              [ evar ~loc idx_v ];
          ]
      in
      (* Ensure equal length, then fold_left over zipped pairs. *)
      lets ~loc
        [ (len, call ~loc (Longident.parse "Stdlib.Array.length") [ left ]) ]
        (B.pexp_ifthenelse ~loc
           (call ~loc (Longident.Lident "<>")
              [
                evar ~loc len;
                call ~loc (Longident.parse "Stdlib.Array.length") [ right ];
              ])
           (invalid_argument ~loc "ufold2: array length mismatch")
           (Some
              (call ~loc
                 (Longident.parse "Stdlib.Array.fold_left")
                 [
                   B.pexp_fun ~loc Nolabel None (pvar ~loc idx_v)
                     (B.pexp_fun ~loc Nolabel None (pvar ~loc val_v)
                        (B.pexp_fun ~loc Nolabel None (pvar ~loc val_w)
                           (ufold2_expr ~path:child_path fvar shape
                              (evar ~loc val_v) (evar ~loc val_w)
                              (evar ~loc idx_v))));
                   acc_expr;
                   call ~loc
                     (Longident.parse "Stdlib.Array.map2")
                     [
                       B.pexp_fun ~loc Nolabel None (pvar ~loc val_v)
                         (B.pexp_fun ~loc Nolabel None (pvar ~loc val_w)
                            (evar ~loc val_w));
                       left;
                       right;
                     ];
                 ])))

(* — names — *)

let module_path_string path =
  match longident_parts path with
  | Some parts -> String.concat "." parts
  | None -> ""

let rec unames_expr shape =
  let loc = shape.uloc in
  match shape.udesc with
  | UPayload ->
      (* names is a string tree; leaves are the field path. This is wrapped at
         the record level — individual payload positions are just the empty
         string placeholder. *)
      B.estring ~loc ""
  | UStatic ->
      (* Static fields are omitted from names (they have no path). *)
      B.estring ~loc ""
  | ULocal name ->
      let local_names = (unames_for_type name).unames in
      let sub_n = gen_symbol ~prefix:"u_n" () in
      call ~loc
        (Longident.Ldot (Lident name, "map"))
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc sub_n)
            (call ~loc (Longident.Lident "^")
               [
                 call ~loc (Longident.Lident "^")
                   [ B.estring ~loc ""; B.estring ~loc "." ];
                 evar ~loc sub_n;
               ]);
          evar ~loc local_names;
        ]
  | UUsing module_path ->
      let sub_n = gen_symbol ~prefix:"u_n" () in
      call ~loc
        (append_lid module_path "map")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc sub_n)
            (call ~loc (Longident.Lident "^")
               [
                 call ~loc (Longident.Lident "^")
                   [ B.estring ~loc ""; B.estring ~loc "." ];
                 evar ~loc sub_n;
               ]);
          evar ~loc (module_path_string module_path ^ ".names");
        ]
  | UTuple shapes -> B.pexp_tuple ~loc (List.map unames_expr shapes)
  | UOption _ | UList _ | UArray _ ->
      (* names for containers is not statically determinable; skip *)
      B.estring ~loc ""

(* Helper to extract module path as string for names construction *)
let module_path_string path =
  match longident_parts path with
  | Some parts -> String.concat "." parts
  | None -> ""

(* — Record / alias codegen for each operation — *)

let uoperation_body ~loc operation ubody =
  let input = evar ~loc "x" in
  match (operation, ubody) with
  | `Umap, UAlias shape -> umap_expr "f" shape input
  | `Umap, URecord fields ->
      let bindings =
        List.map
          (fun (label, shape) ->
            let name = gen_symbol ~prefix:("u_" ^ label.pld_name.txt) () in
            ( label.pld_loc,
              name,
              umap_expr "f" shape (field ~loc:label.pld_loc input label),
              label ))
          fields
      in
      lets_located
        (List.map (fun (loc, n, e, _) -> (loc, n, e)) bindings)
        (B.pexp_record ~loc
           (List.map
              (fun (_, n, _, label) ->
                (lident ~loc:label.pld_name.loc label.pld_name.txt, evar ~loc n))
              bindings)
           None)
  | `Umap2, UAlias shape ->
      umap2_expr ~function_path:"" "f" shape input (evar ~loc "y")
  | `Umap2, URecord fields ->
      let left = evar ~loc "x" in
      let right = evar ~loc "y" in
      let bindings =
        List.map
          (fun (label, shape) ->
            let name = gen_symbol ~prefix:("u2_" ^ label.pld_name.txt) () in
            ( label.pld_loc,
              name,
              umap2_expr ~function_path:"" "f" shape
                (field ~loc:label.pld_loc left label)
                (field ~loc:label.pld_loc right label),
              label ))
          fields
      in
      lets_located
        (List.map (fun (loc, n, e, _) -> (loc, n, e)) bindings)
        (B.pexp_record ~loc
           (List.map
              (fun (_, n, _, label) ->
                (lident ~loc:label.pld_name.loc label.pld_name.txt, evar ~loc n))
              bindings)
           None)
  | `Uiter, UAlias shape -> uiter_expr "f" shape input
  | `Uiter, URecord fields ->
      B.esequence ~loc
        (List.map
           (fun (label, shape) ->
             uiter_expr "f" shape (field ~loc:label.pld_loc input label))
           fields)
  | `Ufold, UAlias shape ->
      ufold_expr ~path:(B.estring ~loc "") "f" shape (evar ~loc "y")
        (evar ~loc "x")
  | `Ufold, URecord fields ->
      List.fold_left
        (fun acc_expr (label, shape) ->
          ufold_expr
            ~path:(B.estring ~loc label.pld_name.txt)
            "f" shape
            (field ~loc:label.pld_loc (evar ~loc "y") label)
            acc_expr)
        (evar ~loc "x") fields
  | `Ufold2, UAlias shape ->
      ufold2_expr ~path:(B.estring ~loc "") "f" shape (evar ~loc "y")
        (evar ~loc "z") (evar ~loc "x")
  | `Ufold2, URecord fields ->
      let left = evar ~loc "y" in
      let right = evar ~loc "z" in
      List.fold_left
        (fun acc_expr (label, shape) ->
          ufold2_expr
            ~path:(B.estring ~loc label.pld_name.txt)
            "f" shape
            (field ~loc:label.pld_loc left label)
            (field ~loc:label.pld_loc right label)
            acc_expr)
        (evar ~loc "x") fields
  | `Unames, UAlias shape -> unames_expr shape
  | `Unames, URecord fields ->
      B.pexp_record ~loc
        (List.map
           (fun (label, shape) ->
             ( lident ~loc:label.pld_name.loc label.pld_name.txt,
               unames_expr shape ))
           fields)
        None

(* FIXME: fvar in map2/map2 refers to "f" hardcoded above but in the
   callback-type builders we need the *actual* parameter. The code above assumes
   the callback is named "f" which matches ufunction_expression below. *)

let rec ubody_uses_callback = function
  | UAlias shape -> ushape_has_payload shape
  | URecord fields -> List.exists (fun (_, s) -> ushape_has_payload s) fields

and ushape_has_payload shape =
  match shape.udesc with
  | UPayload -> true
  | UStatic -> false
  | UTuple shapes -> List.exists ushape_has_payload shapes
  | UOption s | UList s | UArray s -> ushape_has_payload s
  | ULocal _ | UUsing _ -> true

let ucallback_type ~loc operation var_a var_b var_c =
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  let string_ty = B.ptyp_constr ~loc (lident ~loc "string") [] in
  let acc_ty = B.ptyp_var ~loc "acc" in
  let var_ty name = B.ptyp_var ~loc name in
  match operation with
  | `Umap -> arrow (var_ty var_a) (var_ty var_b)
  | `Umap2 -> arrow (var_ty var_a) (arrow (var_ty var_b) (var_ty var_c))
  | `Uiter -> arrow (var_ty var_a) (B.ptyp_constr ~loc (lident ~loc "unit") [])
  | `Ufold -> arrow string_ty (arrow acc_ty (arrow (var_ty var_a) acc_ty))
  | `Ufold2 ->
      arrow string_ty
        (arrow acc_ty (arrow (var_ty var_a) (arrow (var_ty var_b) acc_ty)))
  | `Unames ->
      (* names : unit -> string t, i.e., a constant *)
      arrow
        (B.ptyp_constr ~loc (lident ~loc "unit") [])
        (B.ptyp_constr ~loc (lident ~loc "string") [])

(** arity of the function value after the callback: map=1, map2=2, iter=1,
    fold=2 (tree + acc), fold2=3 (tree + tree + acc), names=1 *)
let uoperation_arity = function
  | `Umap | `Uiter | `Unames -> 1
  | `Umap2 -> 2
  | `Ufold -> 2
  | `Ufold2 -> 3

let umake_binding ~module_path operation udecl =
  let type_decl = udecl.utype_decl in
  let loc = type_decl.ptype_loc in
  let names = unames_for_type type_decl.ptype_name.txt in
  let name = uoperation_name operation names in
  let var_a = "a" in
  let var_b = "b" in
  let var_c = "c" in
  let cb_ty = ucallback_type ~loc operation var_a var_b var_c in
  let type_ = declared_type type_decl in
  let body_expr =
    match udecl.ubody with
    | Some ubody -> uoperation_body ~loc operation ubody
    | None -> assert false
  in
  let input_types =
    match operation with
    | `Unames -> [ type_ ]
    | `Ufold -> [ B.ptyp_var ~loc "acc"; type_ ]
    | `Ufold2 -> [ B.ptyp_var ~loc "acc"; type_; type_ ]
    | _ -> List.init (uoperation_arity operation) (fun _ -> type_)
  in
  let fun_expr =
    ufunction_expression ~loc ~callback_type:cb_ty ~input_types body_expr
  in
  let pat =
    if operation = `Unames then
      (* names takes no callback argument; it's a constant. *)
      pvar ~loc name
    else pvar ~loc name
  in
  B.value_binding ~loc ~pat ~expr:fun_expr

let ucomponent_rec_flag component =
  match component with
  | [] -> assert false
  | [ udecl ] ->
      if String_set.mem udecl.utype_decl.ptype_name.txt udecl.udependencies then
        Recursive
      else Nonrecursive
  | _ -> Recursive

let ugenerate_operation ~module_path operation components =
  List.map
    (fun component ->
      let loc = (List.hd component).utype_decl.ptype_loc in
      B.pstr_value ~loc
        (ucomponent_rec_flag component)
        (List.map (umake_binding ~module_path operation) component))
    components

let ustructure_has_payload_param type_decls =
  match type_decls with
  | decl :: _ -> (
      match decl.ptype_params with
      | [ (p, _) ] -> (
          match p.ptyp_desc with Ptyp_var _ -> true | _ -> false)
      | _ -> false)
  | [] -> false

(* ——— Mirror mode ——— *)

(* Translate a static ptree [shape] into a parameterised ushape. Every [Leaf]
   becomes the payload param ['a]; [Ignored] stays static; [Using M] becomes
   [UUsing M] (Gtree.t delegation convention). *)

let rec ushape_of_shape shape =
  let loc = shape.loc in
  match shape.desc with
  | Leaf -> ushape ~loc UPayload
  | Ignored -> ushape ~loc UStatic
  | Local name -> ushape ~loc (ULocal name)
  | Using m -> ushape ~loc (UUsing (append_lid m "Gtree"))
  | Tuple shapes -> ushape ~loc (UTuple (List.map ushape_of_shape shapes))
  | Option s -> ushape ~loc (UOption (ushape_of_shape s))
  | List s -> ushape ~loc (UList (ushape_of_shape s))
  | Array s -> ushape ~loc (UArray (ushape_of_shape s))

(* Build the core_type for a mirror type position. UPayload → 'a, UStatic → unit
   (placeholder, replaced at record level), ULocal/UUsing → 'a M.t or 'a
   M.Gtree.t, containers → recurse. *)

let rec build_mirror_type ~loc shape =
  match shape.udesc with
  | UPayload -> B.ptyp_var ~loc "m"
  | UStatic -> B.ptyp_constr ~loc (lident ~loc "unit") []
  | ULocal name ->
      B.ptyp_constr ~loc
        (located_lid ~loc
           (Longident.Ldot (Longident.Ldot (Lident name, "Gtree"), "t")))
        [ B.ptyp_var ~loc "m" ]
  | UUsing path ->
      B.ptyp_constr ~loc
        (located_lid ~loc (append_lid path "t"))
        [ B.ptyp_var ~loc "m" ]
  | UTuple shapes ->
      B.ptyp_tuple ~loc (List.map (build_mirror_type ~loc) shapes)
  | UOption s ->
      B.ptyp_constr ~loc (lident ~loc "option") [ build_mirror_type ~loc s ]
  | UList s ->
      B.ptyp_constr ~loc (lident ~loc "list") [ build_mirror_type ~loc s ]
  | UArray s ->
      B.ptyp_constr ~loc (lident ~loc "array") [ build_mirror_type ~loc s ]

(* Build a synthetic ['a t] type declaration. *)

let synth_mirror_type_decl ~loc name fields : type_declaration =
  {
    ptype_name = { txt = name; loc };
    ptype_params = [ (B.ptyp_var ~loc "m", (Covariant, NoInjectivity)) ];
    ptype_cstrs = [];
    ptype_kind = Ptype_record fields;
    ptype_private = Public;
    ptype_manifest = None;
    ptype_attributes = [];
    ptype_loc = loc;
  }

let synth_mirror_alias_decl ~loc name alias_type : type_declaration =
  {
    ptype_name = { txt = name; loc };
    ptype_params = [ (B.ptyp_var ~loc "m", (Covariant, NoInjectivity)) ];
    ptype_cstrs = [];
    ptype_kind = Ptype_abstract;
    ptype_private = Public;
    ptype_manifest = Some alias_type;
    ptype_attributes = [];
    ptype_loc = loc;
  }

(* Generate traversal bindings from a synthetic udeclaration. *)

let mirror_traversals ~module_path udecl : structure =
  let type_decl = udecl.utype_decl in
  let loc = type_decl.ptype_loc in
  List.map
    (fun op ->
      let vb = umake_binding ~module_path op udecl in
      B.pstr_value ~loc Nonrecursive [ vb ])
    [ `Umap; `Umap2; `Uiter; `Ufold; `Ufold2; `Unames ]

(* Generate [module Gtree = struct ... end]. *)

let generate_gtree_module ~module_path static_decls : structure =
  let type_decl = (List.hd static_decls).type_decl in
  let loc = type_decl.ptype_loc in
  let body = (List.hd static_decls).body in
  let module_items : structure =
    match body with
    | Some (Alias s) ->
        let us = ushape_of_shape s in
        let alias_ty = build_mirror_type ~loc us in
        let synth = synth_mirror_alias_decl ~loc "t" alias_ty in
        let type_item = B.pstr_type ~loc Nonrecursive [ synth ] in
        let udecl =
          {
            utype_decl = synth;
            ubody = Some (UAlias us);
            udependencies = String_set.empty;
          }
        in
        type_item :: mirror_traversals ~module_path udecl
    | Some (Record fields) ->
        let ufields =
          List.map (fun (lbl, sh) -> (lbl, ushape_of_shape sh)) fields
        in
        let synth_labels =
          List.map
            (fun (lbl, us) -> { lbl with pld_type = build_mirror_type ~loc us })
            ufields
        in
        let synth = synth_mirror_type_decl ~loc "t" synth_labels in
        let type_item = B.pstr_type ~loc Nonrecursive [ synth ] in
        let udecl =
          {
            utype_decl = synth;
            ubody = Some (URecord ufields);
            udependencies = String_set.empty;
          }
        in
        type_item :: mirror_traversals ~module_path udecl
    | None -> []
  in
  let mod_name = { txt = Some "Gtree"; loc } in
  let mod_expr = B.pmod_structure ~loc module_items in
  let mb = B.module_binding ~loc ~name:mod_name ~expr:mod_expr in
  [ B.pstr_module ~loc mb ]

(* ——— dtype extraction for mirror-mode of_gtree ——— *)

(* Maps an Nx type alias base name (without _t suffix) or element type name to
   the corresponding dtype value name in the Nx module. *)
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
  let name_len = String.length name in
  let suffix_len = String.length suffix in
  if
    name_len > suffix_len
    && String.equal (String.sub name (name_len - suffix_len) suffix_len) suffix
  then Some (String.sub name 0 (name_len - suffix_len))
  else None

(** [dtype_expr_of_core_type ~loc core_type] returns an expression that
    evaluates to the {!Nx_core.Dtype.t} for the Nx tensor type represented by
    [core_type], or [None] if the dtype cannot be determined. *)
let rec dtype_expr_of_core_type ~loc core_type =
  match core_type.ptyp_desc with
  | Ptyp_constr (path, []) -> (
      match longident_parts path.txt with
      | Some [ name ] | Some [ "Nx"; name ] -> (
          match chop_suffix name "_t" with
          | Some base -> (
              match dtype_name_of_nx_name base with
              | Some dname ->
                  Some (ident ~loc (Longident.Ldot (Lident "Nx", dname)))
              | None -> None)
          | None -> None)
      | _ -> None)
  | Ptyp_constr (path, [ _scalar; elt_type ])
    when is_path path.txt [ "Nx"; "t" ] || is_path path.txt [ "Nx_effect"; "t" ]
    ->
      dtype_expr_of_elt_type ~loc elt_type
  | Ptyp_alias (inner, _) -> dtype_expr_of_core_type ~loc inner
  | _ -> None

(** [dtype_expr_of_elt_type ~loc core_type] extracts a dtype expression from an
    element type like [float32_elt] or [complex32_elt]. *)
and dtype_expr_of_elt_type ~loc core_type =
  match core_type.ptyp_desc with
  | Ptyp_constr (path, []) -> (
      match longident_parts path.txt with
      | Some [ name ] | Some [ "Nx"; name ] -> (
          match dtype_name_of_nx_name name with
          | Some dname ->
              Some (ident ~loc (Longident.Ldot (Lident "Nx", dname)))
          | None -> None)
      | _ -> None)
  | Ptyp_alias (inner, _) -> dtype_expr_of_elt_type ~loc inner
  | _ -> None

(* Converters — pack/unpack tensor leaves through the existential. *)

let pack_leaf ~loc expr =
  (* Nx.Ptree.P expr *)
  B.pexp_construct ~loc
    (located_lid ~loc
       (Longident.Ldot (Longident.Ldot (Lident "Nx", "Ptree"), "P")))
    (Some expr)

let unpack_leaf ~loc ~dtype expr =
  call ~loc (Longident.parse "Nx.Ptree.unpack") [ dtype; expr ]

(* Recursively convert a static expression through a ptree [shape] into a gtree
   mirror expression. *)
let rec to_gtree_shape ~loc shape expr =
  match shape.desc with
  | Leaf -> pack_leaf ~loc expr
  | Ignored -> expr
  | Using m -> call ~loc (append_lid m "to_gtree") [ expr ]
  | Local _ -> expr
  | Tuple shapes ->
      let count = List.length shapes in
      let names = List.init count (fun _ -> gen_symbol ~prefix:"ptree_tg" ()) in
      let pattern = B.ppat_tuple ~loc (List.map (pvar ~loc) names) in
      let converted =
        List.map2
          (fun shape name ->
            to_gtree_shape ~loc:shape.loc shape (evar ~loc:shape.loc name))
          shapes names
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr ]
        (B.pexp_tuple ~loc converted)
  | Option shape ->
      let v = gen_symbol ~prefix:"ptree_tg" () in
      B.pexp_match ~loc expr
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc v)))
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some
                    (to_gtree_shape ~loc:shape.loc shape (evar ~loc:shape.loc v))));
        ]
  | List shape ->
      let v = gen_symbol ~prefix:"ptree_tg" () in
      call ~loc
        (Longident.parse "Stdlib.List.map")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc v)
            (to_gtree_shape ~loc:shape.loc shape (evar ~loc:shape.loc v));
          expr;
        ]
  | Array shape ->
      let v = gen_symbol ~prefix:"ptree_tg" () in
      call ~loc
        (Longident.parse "Stdlib.Array.map")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc v)
            (to_gtree_shape ~loc:shape.loc shape (evar ~loc:shape.loc v));
          expr;
        ]

(* Recursively convert a gtree mirror expression back through a ptree [shape]
   into a static expression. *)
let rec of_gtree_shape ~loc shape expr =
  match shape.desc with
  | Leaf -> expr
  | Ignored -> expr
  | Using m -> call ~loc (append_lid m "of_gtree") [ expr ]
  | Local _ -> expr
  | Tuple shapes ->
      let count = List.length shapes in
      let names = List.init count (fun _ -> gen_symbol ~prefix:"ptree_og" ()) in
      let pattern = B.ppat_tuple ~loc (List.map (pvar ~loc) names) in
      let converted =
        List.map2
          (fun shape name ->
            of_gtree_shape ~loc:shape.loc shape (evar ~loc:shape.loc name))
          shapes names
      in
      B.pexp_let ~loc Nonrecursive
        [ B.value_binding ~loc ~pat:pattern ~expr ]
        (B.pexp_tuple ~loc converted)
  | Option shape ->
      let v = gen_symbol ~prefix:"ptree_og" () in
      B.pexp_match ~loc expr
        [
          B.case
            ~lhs:(construct_pattern ~loc "None" None)
            ~guard:None
            ~rhs:(construct ~loc "None" None);
          B.case
            ~lhs:(construct_pattern ~loc "Some" (Some (pvar ~loc v)))
            ~guard:None
            ~rhs:
              (construct ~loc "Some"
                 (Some
                    (of_gtree_shape ~loc:shape.loc shape (evar ~loc:shape.loc v))));
        ]
  | List shape ->
      let v = gen_symbol ~prefix:"ptree_og" () in
      call ~loc
        (Longident.parse "Stdlib.List.map")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc v)
            (of_gtree_shape ~loc:shape.loc shape (evar ~loc:shape.loc v));
          expr;
        ]
  | Array shape ->
      let v = gen_symbol ~prefix:"ptree_og" () in
      call ~loc
        (Longident.parse "Stdlib.Array.map")
        [
          B.pexp_fun ~loc Nolabel None (pvar ~loc v)
            (of_gtree_shape ~loc:shape.loc shape (evar ~loc:shape.loc v));
          expr;
        ]

let generate_to_gtree ~module_path static_decls : structure =
  let type_decl = (List.hd static_decls).type_decl in
  let loc = type_decl.ptype_loc in
  let body = (List.hd static_decls).body in
  let rhs =
    match body with
    | Some (Alias shape) -> to_gtree_shape ~loc shape (evar ~loc "x")
    | Some (Record fields) ->
        let field_exprs =
          List.map
            (fun (lbl, sh) ->
              let access = field ~loc:lbl.pld_loc (evar ~loc "x") lbl in
              let expr =
                match sh.desc with
                | Leaf -> pack_leaf ~loc access
                | Ignored -> access
                | Using m -> call ~loc (append_lid m "to_gtree") [ access ]
                | Local _ -> access
                | Tuple _ | Option _ | List _ | Array _ ->
                    to_gtree_shape ~loc:sh.loc sh access
              in
              ( located_lid ~loc
                  (Longident.Ldot (Lident "Gtree", lbl.pld_name.txt)),
                expr ))
            fields
        in
        B.pexp_record ~loc
          (List.map (fun (lid, e) -> (lid, e)) field_exprs)
          None
    | None -> B.eunit ~loc
  in
  [
    B.pstr_value ~loc Nonrecursive
      [
        B.value_binding ~loc ~pat:(pvar ~loc "to_gtree")
          ~expr:(B.pexp_fun ~loc Nolabel None (pvar ~loc "x") rhs);
      ];
  ]

let gtree_field_access ~loc expr name =
  B.pexp_field ~loc expr
    (located_lid ~loc (Longident.Ldot (Lident "Gtree", name)))

let generate_of_gtree ~module_path static_decls : structure =
  let type_decl = (List.hd static_decls).type_decl in
  let loc = type_decl.ptype_loc in
  let body = (List.hd static_decls).body in
  let rhs =
    match body with
    | Some (Alias s) -> (
        match s.desc with
        | Leaf -> (
            match type_decl.ptype_manifest with
            | Some manifest -> (
                match dtype_expr_of_core_type ~loc manifest with
                | Some dtype -> unpack_leaf ~loc ~dtype (evar ~loc "u")
                | None -> evar ~loc "u")
            | None -> evar ~loc "u")
        | _ -> of_gtree_shape ~loc s (evar ~loc "u"))
    | Some (Record fields) ->
        let field_exprs =
          List.map
            (fun (lbl, sh) ->
              let access =
                gtree_field_access ~loc (evar ~loc "u") lbl.pld_name.txt
              in
              let expr =
                match sh.desc with
                | Leaf -> (
                    match dtype_expr_of_core_type ~loc lbl.pld_type with
                    | Some dtype -> unpack_leaf ~loc ~dtype access
                    | None -> access)
                | Ignored -> access
                | Using m -> call ~loc (append_lid m "of_gtree") [ access ]
                | Local _ -> access
                | Tuple _ | Option _ | List _ | Array _ ->
                    of_gtree_shape ~loc:sh.loc sh access
              in
              (lident ~loc:lbl.pld_name.loc lbl.pld_name.txt, expr))
            fields
        in
        B.pexp_record ~loc field_exprs None
    | None -> evar ~loc "u"
  in
  [
    B.pstr_value ~loc Nonrecursive
      [
        B.value_binding ~loc ~pat:(pvar ~loc "of_gtree")
          ~expr:(B.pexp_fun ~loc Nolabel None (pvar ~loc "u") rhs);
      ];
  ]

let generate_gtree_mirror ~ctxt type_declarations : structure =
  let ptree_decls, errors = validate ~signature:false type_declarations in
  if errors <> [] then structure_errors errors
  else if ptree_decls = [] then []
  else
    let module_path =
      Expansion_context.Deriver.code_path ctxt |> Code_path.fully_qualified_path
    in
    let mod_part = generate_gtree_module ~module_path ptree_decls in
    let to_part = generate_to_gtree ~module_path ptree_decls in
    let of_part = generate_of_gtree ~module_path ptree_decls in
    mod_part @ to_part @ of_part

let gtree_structure_generator ~ctxt (_, type_declarations) =
  let declarations, errors = uvalidate ~signature:false type_declarations in
  if errors <> [] then structure_errors errors
  else if declarations = [] then
    (* Mirror mode *)
    generate_gtree_mirror ~ctxt type_declarations
  else
    let module_path =
      Expansion_context.Deriver.code_path ctxt |> Code_path.fully_qualified_path
    in
    let components = uordered_components declarations in
    List.concat_map
      (fun operation -> ugenerate_operation ~module_path operation components)
      [ `Umap; `Umap2; `Uiter; `Ufold; `Ufold2; `Unames ]

let usignature_type operation type_decl =
  let loc = type_decl.ptype_loc in
  let var_a = "a" in
  let var_b = "b" in
  let var_c = "c" in
  let cb_ty = ucallback_type ~loc operation var_a var_b var_c in
  let type_ = declared_type type_decl in
  let arrow left right = B.ptyp_arrow ~loc Nolabel left right in
  let acc_ty = B.ptyp_var ~loc "acc" in
  match operation with
  | `Umap -> arrow cb_ty (arrow type_ (B.ptyp_var ~loc "b"))
  | `Umap2 -> arrow cb_ty (arrow type_ (arrow type_ (B.ptyp_var ~loc "c")))
  | `Uiter ->
      arrow cb_ty (arrow type_ (B.ptyp_constr ~loc (lident ~loc "unit") []))
  | `Ufold -> arrow cb_ty (arrow acc_ty (arrow type_ acc_ty))
  | `Ufold2 -> arrow cb_ty (arrow acc_ty (arrow type_ (arrow type_ acc_ty)))
  | `Unames ->
      arrow
        (B.ptyp_constr ~loc (lident ~loc "unit") [])
        (B.ptyp_constr ~loc (lident ~loc "string") [])

let gtree_signature_generator ~ctxt (_, type_declarations) =
  let declarations, errors = uvalidate ~signature:true type_declarations in
  if errors <> [] then signature_errors errors
  else if declarations = [] then []
  else
    let add_values udecl =
      let type_decl = udecl.utype_decl in
      let names = unames_for_type type_decl.ptype_name.txt in
      List.map
        (fun operation ->
          let name = uoperation_name operation names in
          let loc = type_decl.ptype_name.loc in
          B.psig_value ~loc
            (B.value_description ~loc ~name:{ txt = name; loc }
               ~type_:(usignature_type operation type_decl)
               ~prim:[]))
        [ `Umap; `Umap2; `Uiter; `Ufold; `Ufold2; `Unames ]
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

let () =
  Deriving.add "gtree"
    ~str_type_decl:
      (Deriving.Generator.V2.make_noarg ~unused_code_warnings:true
         gtree_structure_generator)
    ~sig_type_decl:
      (Deriving.Generator.V2.make_noarg ~unused_code_warnings:true
         gtree_signature_generator)
  |> Deriving.ignore
