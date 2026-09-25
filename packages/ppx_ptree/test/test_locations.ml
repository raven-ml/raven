(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generated code carries the locations of the source it comes from, so the
   compiler reports a mistyped part at the part, and every diagnostic of the
   deriver points at the offending source. *)

open Ppxlib
open Windtrap

let source =
  {|module Nested = struct
  type 'a t = 'a
end

type 'a t = {
  weight : 'a;
  optional : 'a option;
  nested : 'a Nested.t;
}
[@@deriving ptree]
|}

let parse ~filename source =
  let lexbuf = Lexing.from_string source in
  lexbuf.lex_curr_p <-
    { pos_fname = filename; pos_lnum = 1; pos_bol = 0; pos_cnum = 0 };
  Parse.implementation lexbuf

let pattern_name pattern =
  match pattern.ppat_desc with
  | Ppat_var name -> Some name.txt
  | Ppat_constraint ({ ppat_desc = Ppat_var name; _ }, _) -> Some name.txt
  | _ -> None

let rec bindings structure =
  List.concat_map
    (fun item ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) -> bindings
      | Pstr_include
          { pincl_mod = { pmod_desc = Pmod_structure included; _ }; _ } ->
          bindings included
      | _ -> [])
    structure

let find_binding name structure =
  List.find
    (fun binding -> pattern_name binding.pvb_pat = Some name)
    (bindings structure)

let line location = location.loc_start.pos_lnum

class locations =
  object
    inherit Ast_traverse.iter as super
    val mutable fields_rev = []
    val mutable walks_rev = []
    method fields = List.rev fields_rev
    method walks = List.rev walks_rev

    method! expression expression =
      (match expression.pexp_desc with
      | Pexp_apply
          ( {
              pexp_desc =
                Pexp_ident
                  {
                    txt =
                      Ldot (Ldot (Ldot (Lident "Nx", "Ptree"), "Walk"), "field");
                    _;
                  };
              _;
            },
            [
              _;
              (_, { pexp_desc = Pexp_constant (Pconst_string (name, _, _)); _ });
              (_, walk);
              _;
            ] ) ->
          fields_rev <- (name, line expression.pexp_loc) :: fields_rev;
          walks_rev <- (name, line walk.pexp_loc) :: walks_rev
      | _ -> ());
      super#expression expression
  end

let test_generated_locations () =
  let expanded = Driver.map_structure (parse ~filename:"locations.ml" source) in
  let walk = find_binding "walk" expanded in
  equal ~msg:"the walk is at the type declaration" int 5
    (line walk.pvb_pat.ppat_loc);
  let locations = new locations in
  locations#expression walk.pvb_expr;
  equal ~msg:"each field's walk is at the field"
    (list (pair string int))
    [ ("weight", 6); ("optional", 7); ("nested", 8) ]
    locations#fields;
  equal ~msg:"each part's walker is at the part's type"
    (list (pair string int))
    [ ("weight", 6); ("optional", 7); ("nested", 8) ]
    locations#walks

class error_locations =
  object
    inherit Ast_traverse.iter as super
    val mutable locations_rev = []
    method locations = List.rev locations_rev

    method! structure_item item =
      (match item.pstr_desc with
      | Pstr_extension (({ txt = "ocaml.error"; loc }, _), _) ->
          locations_rev <- loc :: locations_rev
      | _ -> ());
      super#structure_item item
  end

let read_file path = In_channel.with_open_bin path In_channel.input_all

let cases_dir =
  match Sys.getenv_opt "PPX_PTREE_CASE" with
  | Some path -> Filename.dirname path
  | None -> Filename.concat (Filename.dirname __FILE__) "cases"

let diagnostic_lines path =
  let source = read_file path in
  let expanded = Driver.map_structure (parse ~filename:path source) in
  let errors = new error_locations in
  errors#structure expanded;
  List.map
    (fun location ->
      is_false
        ~msg:(path ^ ": diagnostic location is not ghost")
        location.loc_ghost;
      equal
        ~msg:(path ^ ": diagnostic filename")
        string path location.loc_start.pos_fname;
      line location)
    errors#locations

let test_diagnostic_locations () =
  let cases =
    [
      ("abstract.ml", [ 1 ]);
      ("alias.ml", [ 1 ]);
      ("any.ml", [ 1 ]);
      ("applied_parameter.ml", [ 1 ]);
      ("arrow.ml", [ 1 ]);
      ("bad_alias_arity.ml", [ 1 ]);
      ("bad_container_arity.ml", [ 1 ]);
      ("bad_tensor_arity.ml", [ 1 ]);
      ("bool.ml", [ 1 ]);
      ("class.ml", [ 1 ]);
      ("conflicting_attributes.ml", [ 1; 1 ]);
      ("constraints.ml", [ 1 ]);
      ("dtype.ml", [ 1 ]);
      ("duplicate_attribute.ml", [ 1; 1 ]);
      ("existential.ml", [ 1; 1 ]);
      ("extensible.ml", [ 1 ]);
      ("extension.ml", [ 1 ]);
      ("fixed_data.ml", [ 1 ]);
      ("functor_path.ml", [ 1 ]);
      ("gadt.ml", [ 1 ]);
      ("int.ml", [ 1 ]);
      ("int_on_string.ml", [ 1; 1 ]);
      ("lazy.ml", [ 1 ]);
      ("local_fixed.ml", [ 2 ]);
      ("local_open.ml", [ 1 ]);
      ("metadata.ml", [ 1 ]);
      ("object.ml", [ 1 ]);
      ("package.ml", [ 3 ]);
      ("polymorphic_field.ml", [ 1 ]);
      ("polymorphic_variant.ml", [ 1 ]);
      ("private.ml", [ 1 ]);
      ("ref.ml", [ 1 ]);
      ("result.ml", [ 1 ]);
      ("short_attribute.ml", [ 1 ]);
      ("short_int.ml", [ 1 ]);
      ("short_walk.ml", [ 1 ]);
      ("skip_parameter.ml", [ 1 ]);
      ("tensor_parameter.ml", [ 1 ]);
      ("two_parameters.ml", [ 1 ]);
      ("type_variable.ml", [ 1 ]);
      ("walk_payload.ml", [ 1 ]);
    ]
  in
  List.iter
    (fun (name, expected) ->
      let path = Filename.concat cases_dir name in
      equal
        ~msg:(path ^ ": diagnostic lines")
        (list int) expected (diagnostic_lines path))
    cases

let () =
  run "ppx_ptree locations"
    [
      test "keeps source locations in generated code" test_generated_locations;
      test "reports rejected forms at their source" test_diagnostic_locations;
    ]
