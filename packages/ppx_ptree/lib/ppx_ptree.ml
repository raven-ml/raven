(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ppxlib
module B = Ast_builder.Default

(* Attributes *)

type 'a attributes = {
  int : 'a Attribute.flag;
  skip : 'a Attribute.flag;
  walk : ('a, payload) Attribute.t;
}

let attributes context =
  {
    int = Attribute.declare_flag "@ptree.int" context;
    skip = Attribute.declare_flag "@ptree.skip" context;
    walk = Attribute.declare "@ptree.walk" context Ast_pattern.__ Fun.id;
  }

let core_attributes = attributes Attribute.Context.core_type
let label_attributes = attributes Attribute.Context.label_declaration

let constructor_attributes =
  attributes Attribute.Context.constructor_declaration

let declaration_attributes = attributes Attribute.Context.type_declaration

(* [Invalid]: an attribute error was reported, and the part reports nothing
   else. *)
type annotation = Int | Skip | Walk of expression | Invalid

let attribute_label = function
  | Int -> "ptree.int"
  | Skip -> "ptree.skip"
  | Walk _ -> "ptree.walk"
  | Invalid -> "ptree"

(* Errors are collected and emitted as located error nodes, so one expansion
   reports all of them. *)

(* A declaration's parameters, read by position: [('a, 'd) t] has the payload
   ['a] and the set ['d], ['d t] the set alone, [t] neither. [None] is an
   anonymous or absent parameter. *)
type env = {
  payload : string option;
  set : string option;
  set_only : bool; (* The declaration has one parameter, its set. *)
  mutable errors : Location.Error.t list;
}

let error env ~loc fmt =
  Format.kasprintf
    (fun msg ->
      env.errors <-
        Location.Error.make ~loc ("ppx_ptree: " ^ msg) ~sub:[] :: env.errors)
    fmt

let annotation env ~loc attrs node =
  let failed = ref false in
  let catch f =
    try f ()
    with exn -> (
      match Location.Error.of_exn exn with
      | Some e ->
          env.errors <- e :: env.errors;
          failed := true;
          None
      | None -> raise exn)
  in
  let flag a = catch (fun () -> Some (Attribute.has_flag a node)) in
  let found =
    List.concat
      [
        (if flag attrs.int = Some true then [ Int ] else []);
        (if flag attrs.skip = Some true then [ Skip ] else []);
        (match catch (fun () -> Attribute.get attrs.walk node) with
        | Some (PStr [ { pstr_desc = Pstr_eval (e, _); _ } ]) -> [ Walk e ]
        | Some _ ->
            error env ~loc
              "[@ptree.walk] takes the walk as an expression, as in \
               [@ptree.walk M.walk]";
            failed := true;
            []
        | None -> []);
      ]
  in
  match found with
  | _ when !failed -> Some Invalid
  | [] -> None
  | [ a ] -> Some a
  | _ ->
      error env ~loc
        "a part takes one of [@ptree.int], [@ptree.skip] and [@ptree.walk]";
      Some Invalid

(* Names *)

let walk_name = function "t" -> "walk" | n -> "walk_" ^ n
let cursor = "ptree__c"
let value = "ptree__x"
let var i = "ptree__" ^ string_of_int i

(* The locally abstract types of a walk's input and output payloads and sets. *)
let type_a = "ptree_a"
let type_b = "ptree_b"
let type_d = "ptree_d"
let type_e = "ptree_e"
let ident ~loc name = B.pexp_ident ~loc { loc; txt = Longident.parse name }
let walk_fn ~loc name = ident ~loc ("Nx.Ptree.Walk." ^ name)

let in_module ~loc m name =
  B.pexp_ident ~loc { loc; txt = Longident.Ldot (m, name) }

let show ty = Format.asprintf "%a" Pprintast.core_type ty

(* Types *)

(* Nx's tensor aliases, each ['d name] for [(_, _, 'd) Nx.t]. *)
let nx_aliases =
  [
    "float64_t";
    "float32_t";
    "float16_t";
    "bfloat16_t";
    "float8_e4m3fn_t";
    "float8_e5m2_t";
    "float4_e2m1fn_t";
    "int64_t";
    "uint64_t";
    "int32_t";
    "uint32_t";
    "int16_t";
    "uint16_t";
    "int8_t";
    "uint8_t";
    "int4_t";
    "uint4_t";
    "complex128_t";
    "complex64_t";
    "bool_t";
    "bit_t";
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
let is_param env v = env.payload = Some v || env.set = Some v

let rec mentions env ty =
  match ty.ptyp_desc with
  | Ptyp_var v -> is_param env v
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

let is_payload env ty =
  match ty.ptyp_desc with Ptyp_var v -> env.payload = Some v | _ -> false

let is_set env ty =
  match ty.ptyp_desc with Ptyp_var v -> env.set = Some v | _ -> false

(* [at env ~payload ~set ty] is [ty] without attributes, the payload parameter
   replaced by the type [payload] and the set parameter by the type [set]. *)
let at env ~payload ~set ty =
  object
    inherit Ast_traverse.map as super

    method! core_type ty =
      let ty = super#core_type { ty with ptyp_attributes = [] } in
      let named x =
        B.ptyp_constr ~loc:ty.ptyp_loc
          { loc = ty.ptyp_loc; txt = Longident.Lident x }
          []
      in
      match ty.ptyp_desc with
      | Ptyp_var v when env.payload = Some v -> named payload
      | Ptyp_var v when env.set = Some v -> named set
      | _ -> ty
  end
    #core_type
    ty

let cursor_type ~loc a b d e =
  B.ptyp_constr ~loc
    { loc; txt = Longident.parse "Nx.Ptree.Walk.cursor" }
    [ a; b; d; e ]

(* [walk_of env ty e] is [e] constrained to walk [ty], at [e]'s location, so a
   mistyped walk is reported where it is written. *)
let walk_of env ty e =
  let loc = e.pexp_loc in
  let abstract x = B.ptyp_constr ~loc { loc; txt = Longident.Lident x } [] in
  B.pexp_constraint ~loc e
    (B.ptyp_arrow ~loc Nolabel
       (cursor_type ~loc (abstract type_a) (abstract type_b) (abstract type_d)
          (abstract type_e))
       (B.ptyp_arrow ~loc Nolabel
          (at env ~payload:type_a ~set:type_d ty)
          (at env ~payload:type_b ~set:type_e ty)))

let unsupported env ~loc part what =
  error env ~loc
    "%s is %s, which has no derived walk; walk it with [@ptree.walk f] or \
     leave it out with [@ptree.skip]"
    part what

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
    error env ~loc "[@ptree.skip] copies %s, whose type %s mentions a parameter"
      part (show ty)

let rec has_int ty =
  match ty.ptyp_desc with
  | Ptyp_constr ({ txt = Lident ("int" | "bool"); _ }, []) -> true
  | Ptyp_constr (_, l) | Ptyp_tuple l -> List.exists has_int l
  | _ -> false

let check_int env ~loc part ty =
  let ok = has_int ty in
  if not ok then
    error env ~loc "[@ptree.int] on %s, whose type %s has no int or bool" part
      (show ty);
  ok

let placeholder ~loc = walk_fn ~loc "leaf"

(* [set_error env ~loc part ty set] reports that the fixed array [ty] of [part]
   lies on [set], which is not the declaration's set parameter. *)
let set_error env ~loc part ty set =
  match (set.ptyp_desc, env.set) with
  | Ptyp_var v, _ when env.payload = Some v ->
      error env ~loc
        "%s has type %s, whose set is the payload parameter '%s; a position of \
         the payload is '%s alone, and a fixed array lies on the type's set \
         parameter, as in (_, 'd) t and 'd Nx.float32_t"
        part (show ty) v v
  | _, Some d ->
      error env ~loc
        "%s has type %s, an array on %s; a fixed array lies on the type's set \
         parameter '%s: write '%s there, or leave the part out with \
         [@ptree.skip] if no program on the set reads it"
        part (show ty) (show set) d d
  | _, None ->
      error env ~loc
        "%s has type %s, an array on %s; a fixed array lies on the type's set \
         parameter: name the set, as in (_, 'd) t and 'd Nx.float32_t, or \
         leave the part out with [@ptree.skip] if no program on the set reads \
         it"
        part (show ty) (show set)

(* [walker env ~ints part ty] is an expression of type [('a, 'b, 'd, 'e)
   Nx.Ptree.Walk.cursor -> ty -> ty'] that walks a value of [ty], where [ty'] is
   [ty] with the payload parameter replaced by ['b] and the set parameter by
   ['e]. [part] names the part in messages. [ints] is [true] under
   [@ptree.int]. *)
let rec walker env ~ints part ty =
  let loc = ty.ptyp_loc in
  annotated env ~ints part ty (annotation env ~loc core_attributes ty)
    ~default:(fun ~ints -> shape env ~ints part ty)

(* [annotated env ~ints part ty annot ~default] is the walker of the part [part]
   of type [ty] under the attribute [annot], and [default ~ints] without one. *)
and annotated env ~ints part ty annot ~default =
  let loc = ty.ptyp_loc in
  match annot with
  | Some (Walk e) -> walk_of env ty e
  | Some Skip ->
      check_skip env ~loc part ty;
      B.pexp_fun ~loc Nolabel None (B.ppat_any ~loc)
        (B.pexp_fun ~loc Nolabel None (B.pvar ~loc value) (ident ~loc value))
  | Some Int ->
      if check_int env ~loc part ty then default ~ints:true
      else placeholder ~loc
  | Some Invalid -> placeholder ~loc
  | None -> default ~ints

and shape env ~ints part ty =
  let loc = ty.ptyp_loc in
  match ty.ptyp_desc with
  | Ptyp_var v when env.payload = Some v -> walk_fn ~loc "leaf"
  | Ptyp_var v when env.set = Some v ->
      error env ~loc
        "%s is the set parameter '%s, which holds no value; an array on the \
         set has a tensor type, as 'd Nx.float32_t%s"
        part v
        (if env.set_only then
           ". A type of one parameter holds fixed arrays only; for positions \
            of a payload write ('a, _) t"
         else "");
      placeholder ~loc
  | Ptyp_var v ->
      error env ~loc "type variable '%s at %s is not a parameter of the type" v
        part;
      placeholder ~loc
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
  | Ptyp_constr ({ txt = lid; _ }, args) -> constr env ~ints part ty lid args
  | Ptyp_any ->
      error env ~loc "%s is the wildcard _, which has no walk" part;
      placeholder ~loc
  | Ptyp_arrow _ ->
      unsupported env ~loc part "a function";
      placeholder ~loc
  | Ptyp_object _ ->
      unsupported env ~loc part "an object type";
      placeholder ~loc
  | Ptyp_class _ ->
      unsupported env ~loc part "a class type";
      placeholder ~loc
  | Ptyp_variant _ ->
      unsupported env ~loc part "a polymorphic variant";
      placeholder ~loc
  | Ptyp_package _ ->
      unsupported env ~loc part "a first-class module";
      placeholder ~loc
  | Ptyp_poly _ ->
      unsupported env ~loc part "a polymorphic type";
      placeholder ~loc
  | Ptyp_alias _ ->
      unsupported env ~loc part "an alias [ty as 'a]";
      placeholder ~loc
  | Ptyp_open _ ->
      unsupported env ~loc part "a locally opened type";
      placeholder ~loc
  | Ptyp_extension _ ->
      unsupported env ~loc part "an extension node";
      placeholder ~loc

and constr env ~ints part ty lid args =
  let loc = ty.ptyp_loc in
  let refuse fmt =
    Format.kasprintf
      (fun why ->
        error env ~loc "%s has type %s; %s" part (show ty) why;
        placeholder ~loc)
      fmt
  in
  (* A fixed array on [set]: [w] if [set] is the set parameter. *)
  let on set w =
    if is_set env set then w
    else (
      set_error env ~loc part ty set;
      placeholder ~loc)
  in
  let rec applies = function
    | Longident.Lident _ -> false
    | Ldot (m, _) -> applies m
    | Lapply _ -> true
  in
  match (lid, args) with
  | lid, _ when applies lid ->
      refuse "functor applications in type paths have no walk"
  | Longident.Ldot (Lident "Nx", "t"), [ v; s; set ] ->
      if mentions env v || mentions env s then
        refuse
          "a tensor's element types cannot mention a parameter; a position of \
           the payload is the payload parameter alone, walked as a leaf"
      else on set (walk_fn ~loc "tensor")
  | Ldot (Lident "Nx", "t"), _ ->
      refuse
        "%s takes three type arguments: the element's OCaml type, its format \
         and the set"
        (Longident.name lid)
  | (Ldot (Lident "Nx", n) | Lident n), args when List.mem n nx_aliases -> (
      match args with
      | [ set ] -> on set (walk_fn ~loc "tensor")
      | _ -> refuse "%s takes one type argument, its set" (Longident.name lid))
  | (Ldot (Ldot (Lident "Nx", "Rng"), "key") | Ldot (Lident "Rng", "key")), args
    -> (
      match args with
      | [ set ] -> on set (ident ~loc "Nx.Rng.walk")
      | _ -> refuse "%s takes one type argument, its set" (Longident.name lid))
  | Ldot (Lident "Nx", "packed"), args -> (
      match args with
      | [ set ] -> on set (packed ~loc)
      | _ -> refuse "Nx.packed takes one type argument, its set")
  | (Ldot (Lident "Nx", "dtype") | Ldot (Ldot (Lident "Nx", "Dtype"), "t")), _
    ->
      refuse "a dtype is data; leave it out with [@ptree.skip]"
  | ( ( Ldot (Lident "Nx", "devices")
      | Ldot (Ldot (Lident "Nx", ("Placement" | "Mesh")), "t") ),
      _ ) ->
      refuse
        "%s has no walk: it says where arrays lie and holds none. Pass it \
         beside the structure, or leave it out with [@ptree.skip] if its type \
         mentions no parameter"
        (Longident.name lid)
  | Ldot (Lident "Nx", n), _
    when String.length n > 2 && String.ends_with ~suffix:"_t" n ->
      refuse
        "Nx.%s is not a tensor alias ppx_ptree knows: ('v, 's, 'd) Nx.t and \
         aliases such as 'd Nx.float32_t"
        n
  | Ldot (Lident "Nx", n), _ ->
      refuse
        "Nx.%s has no walk; walk it with [@ptree.walk f], or leave it out with \
         [@ptree.skip] if its type mentions no parameter"
        n
  | Lident (("option" | "list" | "array") as c), args -> (
      match args with
      | [ elt ] -> (
          let w = walker env ~ints part elt in
          match c with
          | "array" -> array ~loc w
          | c -> B.eapply ~loc (walk_fn ~loc c) [ w ])
      | _ -> refuse "%s takes one type argument" c)
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
      refuse
        "%s has no derived walk; walk it with [@ptree.walk f] or leave it out \
         with [@ptree.skip]"
        n
  | Lident n, _ when List.mem n data ->
      error env ~loc
        "%s is a %s, which has no walk; leave it out with [@ptree.skip], or \
         hold data a compiled program depends on in a tensor"
        part n;
      placeholder ~loc
  | Lident n, args ->
      structure env ~ints part ty (ident ~loc (walk_name n)) args
  | Ldot (m, n), args ->
      structure env ~ints part ty (in_module ~loc m (walk_name n)) args
  | Lapply _, _ -> refuse "functor applications in type paths have no walk"

(* [structure env ~ints part ty w args] is the walk [w] of a structure applied
   to [args], constrained to walk [ty]. A payload argument that mentions a
   parameter without being the payload parameter is a structure nested at the
   positions, walked by [Walk.under]: [(('a, 'd) V.t, 'd) U.t] is [under U.walk
   V.walk]. *)
and structure env ~ints part ty w args =
  let loc = ty.ptyp_loc in
  let w =
    match args with
    | [ p; _ ] when mentions env p && not (is_payload env p) ->
        B.eapply ~loc (walk_fn ~loc "under") [ w; walker env ~ints part p ]
    | _ -> w
  in
  walk_of env ty w

(* [fun c (Nx.P x) -> Nx.P (Walk.tensor c x)]: a packed array is a fixed array
   of its own dtype. *)
and packed ~loc =
  let p = { loc; txt = Longident.parse "Nx.P" } in
  lambda ~loc
    (B.ppat_construct ~loc p (Some (B.pvar ~loc value)))
    (B.pexp_construct ~loc p
       (Some
          (B.eapply ~loc (walk_fn ~loc "tensor")
             [ ident ~loc cursor; ident ~loc value ])))

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
          match annotation env ~loc label_attributes ld with
          | Some Skip ->
              check_skip env ~loc part ty;
              current
          | annot ->
              let w =
                annotated env ~ints:false part ty annot ~default:(fun ~ints ->
                    walker env ~ints part ty)
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
    let gadt = cd.pcd_res <> None || cd.pcd_vars <> [] in
    if gadt then
      error env ~loc
        "constructor [%s] has a GADT type, which a derived walk cannot rebuild \
         at other parameters"
        name;
    let annot = annotation env ~loc constructor_attributes cd in
    let misplaced =
      match (annot, cd.pcd_args) with
      | (None | Some Invalid), _ | Some _, Pcstr_tuple [ _ ] -> false
      | Some a, args ->
          error env ~loc
            "[@%s] on constructor [%s], which has %s; put it on an argument's \
             type, as in [%s of (int [@%s]) * ...]"
            (attribute_label a) name
            (match args with
            | Pcstr_record _ -> "a record argument"
            | Pcstr_tuple [] -> "no argument"
            | Pcstr_tuple l -> Printf.sprintf "%d arguments" (List.length l))
            name (attribute_label a);
          true
    in
    let tag =
      B.eapply ~loc (walk_fn ~loc "case")
        [ ident ~loc cursor; B.estring ~loc name ]
    in
    let lid = { loc; txt = Longident.Lident name } in
    let pat, body =
      match cd.pcd_args with
      | _ when gadt || misplaced -> (None, B.pexp_construct ~loc lid None)
      | Pcstr_tuple [] -> (None, B.pexp_construct ~loc lid None)
      | Pcstr_tuple [ ty ] ->
          let part = Printf.sprintf "the argument of [%s]" name in
          let walked =
            match annot with
            | Some Skip ->
                check_skip env ~loc part ty;
                ident ~loc (var 0)
            | annot ->
                B.eapply ~loc
                  (annotated env ~ints:false part ty annot
                     ~default:(fun ~ints -> walker env ~ints part ty))
                  [ ident ~loc cursor; ident ~loc (var 0) ]
          in
          ( Some (B.pvar ~loc (var 0)),
            sequence ~loc
              [ (var 0, walked) ]
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

(* A type has at most two parameters, read by position: the payload, whose
   positions a walk turns from ['a] into ['b], then the set of its fixed arrays,
   which it turns from ['d] into ['e]. One parameter is the set. *)
let parameters td =
  let env = { payload = None; set = None; set_only = false; errors = [] } in
  let name (ty, _) =
    match ty.ptyp_desc with Ptyp_var v -> Some v | _ -> None
  in
  match td.ptype_params with
  | [] -> env
  | [ s ] -> { env with set = name s; set_only = true }
  | [ p; s ] -> { env with payload = name p; set = name s }
  | params ->
      error env ~loc:td.ptype_loc
        "[%s] has %d type parameters; a structure has at most two, its payload \
         then its set, as ('a, 'd) t"
        td.ptype_name.txt (List.length params);
      env

(* An attribute on the declaration itself belongs to one of its parts. *)
let check_declaration env td =
  match annotation env ~loc:td.ptype_loc declaration_attributes td with
  | None | Some Invalid -> ()
  | Some a ->
      error env ~loc:td.ptype_loc
        "[@@@@%s] on type [%s]: put the attribute on a field or a part"
        (attribute_label a) td.ptype_name.txt

(* [walk_type ~loc td a b d e] is [(a, b, d, e) Nx.Ptree.Walk.cursor -> (a, d) t
   -> (b, e) t], with [d t -> e t] for a type of one parameter and [t -> t] for
   one of none. *)
let walk_type ~loc td a b d e =
  let lid = { loc; txt = Longident.Lident td.ptype_name.txt } in
  let self x y =
    match td.ptype_params with
    | [] -> B.ptyp_constr ~loc lid []
    | [ _ ] -> B.ptyp_constr ~loc lid [ y ]
    | _ -> B.ptyp_constr ~loc lid [ x; y ]
  in
  B.ptyp_arrow ~loc Nolabel (cursor_type ~loc a b d e)
    (B.ptyp_arrow ~loc Nolabel (self a d) (self b e))

(* [let walk : 'ptree_a 'ptree_b 'ptree_d 'ptree_e. ... = fun (type ptree_a
   ptree_b ptree_d ptree_e) -> (body : ...)], so that a mistyped part is
   reported at the part and a nested type may recur at other arguments. *)
let walk_binding ~loc td body =
  let names = [ type_a; type_b; type_d; type_e ] in
  let var x = B.ptyp_var ~loc x in
  let abstract x = B.ptyp_constr ~loc { loc; txt = Longident.Lident x } [] in
  let poly =
    B.ptyp_poly ~loc
      (List.map (fun txt -> { loc; txt }) names)
      (walk_type ~loc td (var type_a) (var type_b) (var type_d) (var type_e))
  in
  let constrained =
    B.pexp_constraint ~loc body
      (walk_type ~loc td (abstract type_a) (abstract type_b) (abstract type_d)
         (abstract type_e))
  in
  let expr =
    List.fold_right
      (fun txt e -> B.pexp_newtype ~loc { loc; txt } e)
      names constrained
  in
  B.value_binding ~loc
    ~pat:
      (B.ppat_constraint ~loc (B.pvar ~loc (walk_name td.ptype_name.txt)) poly)
    ~expr

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
    List.map
      (fun td ->
        let env = parameters td in
        let bad_parameters = env.errors <> [] in
        check_declaration env td;
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
            Ok (walk_binding ~loc td body)
        | _ -> Error env)
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
          (fun vb -> refers (List.map walk_name locals) vb.pvb_expr)
          walks
      in
      [
        B.pstr_value ~loc (if recursive then Recursive else Nonrecursive) walks;
      ]

let generate_intf ~ctxt:_ (_, tds) =
  List.concat_map
    (fun td ->
      let env = parameters td in
      check_declaration env td;
      match env.errors with
      | _ :: _ -> errors B.psig_extension env
      | [] ->
          let loc = td.ptype_loc in
          let v x = B.ptyp_var ~loc x in
          [
            B.psig_value ~loc
              (B.value_description ~loc
                 ~name:{ loc; txt = walk_name td.ptype_name.txt }
                 ~type_:(walk_type ~loc td (v "a") (v "b") (v "d") (v "e"))
                 ~prim:[]);
          ])
    tds

let () =
  Deriving.add "ptree"
    ~str_type_decl:(Deriving.Generator.V2.make_noarg generate_impl)
    ~sig_type_decl:(Deriving.Generator.V2.make_noarg generate_intf)
  |> Deriving.ignore
