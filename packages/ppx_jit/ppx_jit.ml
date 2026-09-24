(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ppxlib
module B = Ast_builder.Default

(* [let[@jit] f (a : A.t) (b : B.t) : D.t * E.t = body] expands to two generated
   modules and a wrapper: an input module packing the arguments into a record,
   an output module packing the result into a one-field record, and [f] as a
   partial application of [Rune.jit2] wrapped to recover the original calling
   convention. See [test/expansion_cases/simple.expected] for the full
   expansion.

   The traversals are not emitted through a [@@deriving ptree] attribute: the
   whole-structure pass of [ppx_jit] runs after the context-free pass that
   expands derivers, so the attribute would never be seen. [Ppx_ptree] exposes
   its structure generator and [ppx_jit] calls it directly. *)

let jit_attribute =
  Attribute.declare "jit" Attribute.Context.Value_binding
    Ast_pattern.(pstr __)
    Fun.id

(* {1 Payload} *)

type options = {
  device : expression option;
  beam : expression option;
  beam_parallel : expression option;
}

let no_options = { device = None; beam = None; beam_parallel = None }

let parse_options ~loc payload =
  let malformed ~loc =
    Location.raise_errorf ~loc
      "ppx_jit: expected a record payload, e.g. [@jit { device = \"NV\"; beam \
       = 8 }]"
  in
  match (payload : structure) with
  | [] -> no_options
  | [
   {
     pstr_desc = Pstr_eval ({ pexp_desc = Pexp_record (fields, None); _ }, _);
     _;
   };
  ] ->
      List.fold_left
        (fun opts ({ txt = lid; loc = field_loc }, expr) ->
          let set field =
            match field with
            | Some _ ->
                Location.raise_errorf ~loc:field_loc "ppx_jit: duplicate option"
            | None -> Some expr
          in
          match lid with
          | Longident.Lident "device" -> { opts with device = set opts.device }
          | Longident.Lident "beam" -> { opts with beam = set opts.beam }
          | Longident.Lident "beam_parallel" ->
              { opts with beam_parallel = set opts.beam_parallel }
          | _ ->
              Location.raise_errorf ~loc:field_loc
                "ppx_jit: unknown option; expected one of: device, beam, \
                 beam_parallel")
        no_options fields
  | [ { pstr_loc; _ } ] -> malformed ~loc:pstr_loc
  | _ -> malformed ~loc

(* {1 Arguments} *)

(* The original [function_param] is kept so the wrapper can reuse it verbatim,
   preserving labels, optional defaults and annotations. *)
type arg = {
  param : function_param;
  name : string;
  typ : core_type;
  loc : Location.t;
}

let extract_arg param =
  match param.pparam_desc with
  | Pparam_val
      ( _,
        _,
        {
          ppat_desc =
            Ppat_constraint
              ({ ppat_desc = Ppat_var { txt = name; _ }; ppat_loc; _ }, typ);
          _;
        } ) ->
      { param; name; typ; loc = ppat_loc }
  | Pparam_val (_, _, { ppat_desc = Ppat_var { txt = name; _ }; ppat_loc; _ })
    ->
      Location.raise_errorf ~loc:ppat_loc
        "ppx_jit: argument `%s` needs a type annotation selecting its Ptree \
         module, e.g. (%s : Params.t)"
        name name
  | Pparam_val (_, _, pat) ->
      Location.raise_errorf ~loc:pat.ppat_loc
        "ppx_jit: arguments of a [@jit] function must be annotated variables, \
         e.g. (a : A.t)"
  | Pparam_newtype { loc; _ } ->
      Location.raise_errorf ~loc
        "ppx_jit: locally abstract types are not supported in [@jit] functions"

(* {1 Expansion} *)

let lident ~loc txt = { txt = Longident.Lident txt; loc }
let ldot ~loc a b = { txt = Longident.Ldot (Lident a, b); loc }

let expand_vb ~base_ctxt ~modules payload vb =
  let loc = vb.pvb_loc in
  let options = parse_options ~loc payload in
  let name =
    match vb.pvb_pat.ppat_desc with
    | Ppat_var { txt; _ } -> txt
    | _ ->
        Location.raise_errorf ~loc:vb.pvb_pat.ppat_loc
          "ppx_jit: [@jit] must be attached to a function name: let[@jit] f (a \
           : A.t) : B.t = ..."
  in
  let params, return_type, body =
    match vb.pvb_expr.pexp_desc with
    | Pexp_function (params, Some (Pconstraint return_type), Pfunction_body body)
      ->
        (params, return_type, body)
    | Pexp_function
        ( params,
          None,
          Pfunction_body { pexp_desc = Pexp_constraint (body, return_type); _ }
        ) ->
        (params, return_type, body)
    | Pexp_function (_, Some (Pcoerce _), _) ->
        Location.raise_errorf ~loc
          "ppx_jit: coercion annotations are not supported; use a plain return \
           type annotation"
    | Pexp_function (_, _, Pfunction_cases _) ->
        Location.raise_errorf ~loc
          "ppx_jit: [function] expressions are not supported; use [fun] with \
           an annotated return type"
    | Pexp_function (_, None, _) ->
        Location.raise_errorf ~loc
          "ppx_jit: missing return type annotation: let[@jit] f (a : A.t) : \
           B.t = ..."
    | _ ->
        Location.raise_errorf ~loc
          "ppx_jit: [@jit] must be attached to a function: let[@jit] f (a : \
           A.t) : B.t = ..."
  in
  let args = List.map extract_arg params in
  let rec check_duplicates seen args =
    match args with
    | [] -> ()
    | a :: rest ->
        if List.mem a.name seen then
          Location.raise_errorf ~loc:a.loc "ppx_jit: duplicate argument `%s`"
            a.name
        else check_duplicates (a.name :: seen) rest
  in
  check_duplicates [] args;
  let in_module_name = "Ppx_jit_in_" ^ name in
  let out_module_name = "Ppx_jit_out_" ^ name in
  let in_decl =
    B.type_declaration ~loc ~name:{ txt = "t"; loc } ~params:[] ~cstrs:[]
      ~kind:
        (Ptype_record
           (List.map
              (fun a ->
                B.label_declaration ~loc:a.loc
                  ~name:{ txt = a.name; loc = a.loc }
                  ~mutable_:Immutable ~type_:a.typ)
              args))
      ~private_:Public ~manifest:None
  in
  let out_decl =
    B.type_declaration ~loc ~name:{ txt = "t"; loc } ~params:[] ~cstrs:[]
      ~kind:
        (Ptype_record
           [
             B.label_declaration ~loc ~name:{ txt = "result"; loc }
               ~mutable_:Immutable ~type_:return_type;
           ])
      ~private_:Public ~manifest:None
  in
  let deriver_ctxt module_name =
    let base =
      List.fold_left
        (fun ctxt module_name ->
          Expansion_context.Base.enter_module ~loc module_name ctxt)
        base_ctxt
        (modules @ [ module_name ])
    in
    Expansion_context.Deriver.make ~derived_item_loc:loc ~inline:false ~base ()
  in
  let module_item module_name decl =
    let traversals =
      Ppx_ptree.ptree_structure ~ctxt:(deriver_ctxt module_name)
        (Nonrecursive, [ decl ]) false
    in
    B.pstr_module ~loc
      (B.module_binding ~loc
         ~name:{ txt = Some module_name; loc }
         ~expr:
           (B.pmod_structure ~loc
              (B.pstr_type ~loc Nonrecursive [ decl ] :: traversals)))
  in
  let record_pat =
    B.ppat_record ~loc
      (List.map
         (fun a ->
           ( ldot ~loc in_module_name a.name,
             B.ppat_var ~loc:a.loc { txt = a.name; loc = a.loc } ))
         args)
      Closed
  in
  let traced =
    B.pexp_fun ~loc Nolabel None record_pat
      (B.pexp_record ~loc [ (ldot ~loc out_module_name "result", body) ] None)
  in
  let optional_labelled label value =
    match value with Some expr -> [ (Labelled label, expr) ] | None -> []
  in
  let impl =
    B.pexp_apply ~loc
      (B.pexp_ident ~loc (ldot ~loc "Rune" "jit2"))
      (optional_labelled "device" options.device
      @ optional_labelled "beam" options.beam
      @ optional_labelled "beam_parallel" options.beam_parallel
      @ [
          ( Nolabel,
            B.pexp_pack ~loc (B.pmod_ident ~loc (lident ~loc in_module_name)) );
          ( Nolabel,
            B.pexp_pack ~loc (B.pmod_ident ~loc (lident ~loc out_module_name))
          );
          (Nolabel, traced);
        ])
  in
  let impl_name = "ppx_jit_impl" in
  let call =
    B.pexp_apply ~loc
      (B.pexp_ident ~loc (lident ~loc impl_name))
      [
        ( Nolabel,
          B.pexp_record ~loc
            (List.map
               (fun a ->
                 ( ldot ~loc in_module_name a.name,
                   B.pexp_ident ~loc:a.loc (lident ~loc:a.loc a.name) ))
               args)
            None );
      ]
  in
  let wrapper_body =
    B.pexp_field ~loc call (ldot ~loc out_module_name "result")
  in
  let wrapper =
    B.pexp_function ~loc
      (List.map (fun a -> a.param) args)
      None (Pfunction_body wrapper_body)
  in
  let binding =
    B.value_binding ~loc
      ~pat:(B.ppat_var ~loc { txt = name; loc = vb.pvb_pat.ppat_loc })
      ~expr:
        (B.pexp_let ~loc Nonrecursive
           [
             B.value_binding ~loc
               ~pat:(B.ppat_var ~loc { txt = impl_name; loc })
               ~expr:impl;
           ]
           wrapper)
  in
  let binding =
    {
      binding with
      pvb_attributes =
        List.filter
          (fun (a : attribute) -> a.attr_name.txt <> "jit")
          vb.pvb_attributes;
    }
  in
  [
    module_item in_module_name in_decl;
    module_item out_module_name out_decl;
    B.pstr_value ~loc Nonrecursive [ binding ];
  ]

let has_jit_attribute vb =
  List.exists (fun (a : attribute) -> a.attr_name.txt = "jit") vb.pvb_attributes

let expand_item ~base_ctxt ~modules item =
  match item.pstr_desc with
  | Pstr_value (rec_flag, vbs) when List.exists has_jit_attribute vbs -> (
      match (rec_flag, vbs) with
      | _, [] -> [ item ]
      | Recursive, { pvb_loc; _ } :: _ ->
          Location.raise_errorf ~loc:pvb_loc
            "ppx_jit: [@jit] does not support recursive functions"
      | Nonrecursive, [ vb ] -> (
          match Attribute.get jit_attribute vb with
          | Some payload -> expand_vb ~base_ctxt ~modules payload vb
          | None -> [ item ])
      | Nonrecursive, { pvb_loc; _ } :: _ ->
          Location.raise_errorf ~loc:pvb_loc
            "ppx_jit: [@jit] does not support [let ... and ...] groups")
  | _ -> [ item ]

class mapper base_ctxt =
  object (self)
    inherit Ast_traverse.map as super
    val mutable modules = []

    method! module_binding mb =
      let saved = modules in
      (match mb.pmb_name.txt with
      | Some name -> modules <- modules @ [ name ]
      | None -> ());
      Fun.protect
        ~finally:(fun () -> modules <- saved)
        (fun () -> super#module_binding mb)

    method! structure str =
      let str = super#structure str in
      List.concat_map (fun item -> self#expand_item item) str

    method private expand_item item = expand_item ~base_ctxt ~modules item

    method! expression expr =
      (match expr.pexp_desc with
      | Pexp_let (_, vbs, _) when List.exists has_jit_attribute vbs ->
          Location.raise_errorf ~loc:expr.pexp_loc
            "ppx_jit: [@jit] is only supported at module level; move the \
             binding out of the expression"
      | _ -> ());
      super#expression expr
  end

let () =
  Driver.V2.register_transformation "ppx_jit" ~impl:(fun ctxt str ->
      (new mapper ctxt)#structure str)
