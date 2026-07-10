(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Ppxlib
open Windtrap

let source =
  {|module Nested = struct
  type t = Nx.float32_t
  let map f x = f x
  let map2 f x y = f x y
  let iter f x = f x
end

type t = {
  weight : Nx.float32_t;
  optional : Nx.float32_t option;
  nested : Nested.t;
}
[@@deriving ptree]
|}

let parse ~filename source =
  let lexbuf = Lexing.from_string source in
  lexbuf.lex_curr_p <-
    { pos_fname = filename; pos_lnum = 1; pos_bol = 0; pos_cnum = 0 };
  Parse.implementation lexbuf

let pattern_name pattern =
  match pattern.ppat_desc with Ppat_var name -> Some name.txt | _ -> None

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
    val mutable callback_calls_rev = []
    val mutable delegated_rev = []
    val mutable invalid_arguments_rev = []
    val mutable field_accesses_rev = []
    val mutable reconstructed_fields_rev = []
    method callback_calls = List.rev callback_calls_rev
    method delegated = List.rev delegated_rev
    method invalid_arguments = List.rev invalid_arguments_rev
    method field_accesses = List.rev field_accesses_rev
    method reconstructed_fields = List.rev reconstructed_fields_rev

    method! expression expression =
      (match expression.pexp_desc with
      | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident "f"; _ }; _ }, _) ->
          callback_calls_rev <- line expression.pexp_loc :: callback_calls_rev
      | Pexp_ident { txt = Ldot (Lident "Nested", "map"); _ } ->
          delegated_rev <- line expression.pexp_loc :: delegated_rev
      | Pexp_apply
          ( {
              pexp_desc =
                Pexp_ident { txt = Ldot (Lident "Stdlib", "invalid_arg"); _ };
              _;
            },
            _ ) ->
          invalid_arguments_rev <-
            line expression.pexp_loc :: invalid_arguments_rev
      | Pexp_field (_, { txt = Lident name; _ }) ->
          field_accesses_rev <-
            (name, line expression.pexp_loc) :: field_accesses_rev
      | Pexp_record (fields, None) ->
          List.iter
            (fun (field, value) ->
              match field.txt with
              | Lident name ->
                  reconstructed_fields_rev <-
                    (name, line value.pexp_loc) :: reconstructed_fields_rev
              | Ldot _ | Lapply _ -> ())
            fields
      | _ -> ());
      super#expression expression
  end

let inspect expression =
  let locations = new locations in
  locations#expression expression;
  locations

let test_generated_locations () =
  let expanded = Driver.map_structure (parse ~filename:"locations.ml" source) in
  let map = find_binding "map" expanded in
  equal ~msg:"generated binding name uses the type declaration" int 8
    (line map.pvb_pat.ppat_loc);
  let map_locations = inspect map.pvb_expr in
  equal ~msg:"leaf calls use their core-type locations" (list int) [ 9; 10 ]
    map_locations#callback_calls;
  equal ~msg:"delegation uses the delegated core-type location" (list int)
    [ 11 ] map_locations#delegated;
  equal ~msg:"field accesses use field declaration locations"
    (list (pair string int))
    [ ("weight", 9); ("optional", 10); ("nested", 11) ]
    map_locations#field_accesses;
  equal ~msg:"record reconstruction uses field declaration locations"
    (list (pair string int))
    [ ("weight", 9); ("optional", 10); ("nested", 11) ]
    map_locations#reconstructed_fields;
  let map2 = find_binding "map2" expanded in
  let map2_locations = inspect map2.pvb_expr in
  equal ~msg:"container mismatch uses the container core-type location"
    (list int) [ 10 ] map2_locations#invalid_arguments

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
      ("anonymous_parameter.ml", [ 1 ]);
      ("any.ml", [ 1 ]);
      ("arrow.ml", [ 1 ]);
      ("bad_alias_arity.ml", [ 1 ]);
      ("bad_container_arity.ml", [ 1 ]);
      ("bad_tensor_arity.ml", [ 1 ]);
      ("bad_using.ml", [ 1 ]);
      ("class.ml", [ 1 ]);
      ("conflicting_attributes.ml", [ 1 ]);
      ("dtype_metadata.ml", [ 1 ]);
      ("duplicate_attribute.ml", [ 1 ]);
      ("extensible.ml", [ 1 ]);
      ("extension.ml", [ 1 ]);
      ("functor_path.ml", [ 1 ]);
      ("gadt.ml", [ 1 ]);
      ("inline_record.ml", [ 1 ]);
      ("lazy.ml", [ 1 ]);
      ("local_open.ml", [ 1 ]);
      ("metadata.ml", [ 1 ]);
      ("object.ml", [ 1 ]);
      ("package.ml", [ 3 ]);
      ("polymorphic_field.ml", [ 1 ]);
      ("polymorphic_variant.ml", [ 1 ]);
      ("private.ml", [ 1 ]);
      ("reexport_record.ml", [ 1 ]);
      ("reexport_variant.ml", [ 1 ]);
      ("ref.ml", [ 1 ]);
      ("result.ml", [ 1 ]);
      ("short_attribute.ml", [ 1 ]);
      ("short_leaf.ml", [ 1 ]);
      ("short_using.ml", [ 1 ]);
      ("two_primaries.ml", [ 1; 2 ]);
      ("type_variable.ml", [ 1 ]);
      ("unknown_qualified.ml", [ 1 ]);
      ("using_functor.ml", [ 1 ]);
      ("variant.ml", [ 1 ]);
    ]
  in
  List.iter
    (fun (name, expected) ->
      let path = Filename.concat cases_dir name in
      equal
        ~msg:(path ^ ": diagnostic lines")
        (list int) expected (diagnostic_lines path))
    cases

let tests =
  [
    test "preserves source locations in generated nodes"
      test_generated_locations;
    test "reports rejected forms at their source locations"
      test_diagnostic_locations;
  ]

let () = run "ppx_ptree locations" tests
