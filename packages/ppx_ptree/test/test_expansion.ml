(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What [@@deriving ptree] expands to, seen from the source: the values it
   binds, the locations it gives them, and every error it reports, printed as
   OCaml prints it. *)

open Ppxlib
open Windtrap

let parse_impl ~filename source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf filename;
  Parse.implementation lexbuf

let parse_intf ~filename source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf filename;
  Parse.interface lexbuf

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

let rec values signature =
  List.concat_map
    (fun item ->
      match item.psig_desc with
      | Psig_value v -> [ v ]
      | Psig_include { pincl_mod = { pmty_desc = Pmty_signature s; _ }; _ } ->
          values s
      | _ -> [])
    signature

let find_binding name structure =
  List.find
    (fun binding -> pattern_name binding.pvb_pat = Some name)
    (bindings structure)

(* Generated values *)

let group_source =
  {|type ('a, 'd) tree = Leaf of 'a | Node of ('a, 'd) tree list

and ('a, 'd) t = {
  root : ('a, 'd) tree;
  frozen : ('d Nx.float32_t, 'd) tree;
  window : int option; [@ptree.int]
}

and 'd state = { steps : 'd Nx.int32_t; index : 'd Cache_index.t }
and shape = { depth : int [@ptree.int] } [@@deriving ptree]
|}

let test_group_names () =
  let expanded =
    Driver.map_structure (parse_impl ~filename:"x.ml" group_source)
  in
  equal (list string)
    [ "walk_tree"; "walk"; "walk_state"; "walk_shape" ]
    (List.filter_map (fun b -> pattern_name b.pvb_pat) (bindings expanded))

let interface =
  {|type ('a, 'd) t = { weight : 'a } [@@deriving ptree]
type 'd keys = 'd Nx.Rng.key list [@@deriving ptree]
type flags = { causal : bool [@ptree.int] } [@@deriving ptree]
|}

let test_interface_names () =
  let expanded =
    Driver.map_signature (parse_intf ~filename:"x.mli" interface)
  in
  equal (list string)
    [ "walk"; "walk_keys"; "walk_flags" ]
    (List.map (fun v -> v.pval_name.txt) (values expanded))

(* Every form of part, in a structure and an interface, for the ghost locations
   below. *)
let ghost_source =
  {|module Nested = struct
  type ('a, 'd) t = 'a
end

type ('a, 'd) t = {
  weight : 'a;
  pair : 'a * 'a option;
  items : 'a array;
  nested : (('a, 'd) Nested.t, 'd) Nested.t;
  causal : bool; [@ptree.int]
  name : string; [@ptree.skip]
  custom : 'a; [@ptree.walk fun c x -> Nx.Ptree.Walk.leaf c x]
  scale : 'd Nx.float32_t;
}
[@@deriving ptree]

type ('a, 'd) u =
  | Empty
  | One of 'a [@ptree.walk Nx.Ptree.Walk.leaf]
  | Two of 'a * ('a, 'd) t
  | Named of { first : 'a; key : 'd Nx.Rng.key; packed : 'd Nx.packed }
[@@deriving ptree]

type 'd keys = 'd Nx.Rng.key list [@@deriving ptree]
|}

(* The [[@ptree.walk]] payloads of the source. *)
let payloads structure =
  let found = ref [] in
  object
    inherit Ast_traverse.iter as super

    method! attribute a =
      (match (a.attr_name.txt, a.attr_payload) with
      | "ptree.walk", PStr [ { pstr_desc = Pstr_eval (e, _); _ } ] ->
          found := e :: !found
      | _ -> ());
      super#attribute a
  end
    #structure
    structure;
  List.rev !found

(* [real] collects the non-ghost locations outside the user's payloads, and
   [kept] the payloads found unchanged, locations included. *)
class real_locations user =
  object
    inherit Ast_traverse.iter as super
    val mutable real_rev = []
    val mutable kept_rev = []
    method real = List.rev real_rev
    method kept = List.rev kept_rev
    method! location loc = if not loc.loc_ghost then real_rev <- loc :: real_rev

    method! expression e =
      if List.mem e user then kept_rev <- e :: kept_rev else super#expression e
  end

let show_loc loc =
  Printf.sprintf "%d:%d-%d:%d" loc.loc_start.pos_lnum
    (loc.loc_start.pos_cnum - loc.loc_start.pos_bol)
    loc.loc_end.pos_lnum
    (loc.loc_end.pos_cnum - loc.loc_end.pos_bol)

(* A payload's text and its location, which [List.mem] in [real_locations]
   compares with every location inside it. *)
let show_payload e =
  Format.asprintf "%a at %s" Pprintast.expression e (show_loc e.pexp_loc)

let test_ghost_impl () =
  let source = parse_impl ~filename:"ghost.ml" ghost_source in
  let user = payloads source in
  equal ~msg:"the source's [@ptree.walk] payloads" (list string)
    [
      "fun c x -> Nx.Ptree.Walk.leaf c x at 12:28-12:61";
      "Nx.Ptree.Walk.leaf at 19:27-19:45";
    ]
    (List.map show_payload user);
  let expanded = Driver.map_structure source in
  let locations = new real_locations user in
  List.iter
    (fun name -> locations#value_binding (find_binding name expanded))
    [ "walk"; "walk_u"; "walk_keys" ];
  equal ~msg:"non-ghost locations in the walks, outside the payloads"
    (list string) []
    (List.map show_loc locations#real);
  equal ~msg:"each payload is in a walk once, with its own locations"
    (list string)
    (List.map show_payload user)
    (List.map show_payload locations#kept)

let test_ghost_intf () =
  let expanded =
    Driver.map_signature (parse_intf ~filename:"ghost.mli" interface)
  in
  let locations = new real_locations [] in
  List.iter locations#value_description (values expanded);
  equal (list string) [] (List.map show_loc locations#real)

(* Errors *)

let errors =
  object
    inherit [Location.Error.t list] Ast_traverse.fold as super

    method! extension ext acc =
      let acc =
        match Location.Error.of_extension ext with
        | Some e -> e :: acc
        | None -> acc
      in
      super#extension ext acc
  end

let location =
  Testable.structural ~pp:(fun ppf l -> Format.pp_print_string ppf (show_loc l))

(* [report source] is what OCaml prints for each error that deriving [source],
   the file [x.ml], reports: the location, the source line and the message. Each
   error lies in the source, so an editor shows it there. *)
let report source =
  let lexbuf = Lexing.from_string source in
  Lexing.set_filename lexbuf "x.ml";
  Astlib.Location.set_input_name "x.ml";
  Astlib.Location.set_input_lexbuf (Some lexbuf);
  let expanded = Driver.map_structure (Parse.implementation lexbuf) in
  let buffer = Buffer.create 256 in
  let ppf = Format.formatter_of_buffer buffer in
  List.iter
    (fun e ->
      let loc = Location.Error.get_location e in
      satisfies ~claim:"a location in the source, not a ghost" location
        (fun l -> not l.loc_ghost)
        loc;
      try Location.Error.raise e with exn -> Location.report_exception ppf exn)
    (List.rev (errors#structure expanded []));
  Format.pp_print_flush ppf ();
  Buffer.contents buffer

let reports name rows =
  cases name
    ~id:(fun (id, _, _) -> id)
    rows
    (fun (_, source, expected) -> expect (report source) expected)

let () =
  exit
    (run "ppx_ptree expansion"
       [
         group "declarations"
           [
             reports
               "a fixed array off the set parameter is refused at the part"
               [
                 ( "on the host",
                   {|type ('a, 'd) t = { w : 'a; mask : Nx.host Nx.bool_t; key : Nx.host Nx.Rng.key; p : Nx.host Nx.packed } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 35-52:
                     1 | type ('a, 'd) t = { w : 'a; mask : Nx.host Nx.bool_t; key : Nx.host Nx.Rng.key; p : Nx.host Nx.packed } [@@deriving ptree]
                                                            ^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [mask] has type Nx.host Nx.bool_t, an array on Nx.host; a fixed array lies on the type's set parameter 'd: write 'd there, or leave the part out with [@ptree.skip] if no program on the set reads it

                     File "x.ml", line 1, characters 60-78:
                     1 | type ('a, 'd) t = { w : 'a; mask : Nx.host Nx.bool_t; key : Nx.host Nx.Rng.key; p : Nx.host Nx.packed } [@@deriving ptree]
                                                                                     ^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [key] has type Nx.host Nx.Rng.key, an array on Nx.host; a fixed array lies on the type's set parameter 'd: write 'd there, or leave the part out with [@ptree.skip] if no program on the set reads it

                     File "x.ml", line 1, characters 84-101:
                     1 | type ('a, 'd) t = { w : 'a; mask : Nx.host Nx.bool_t; key : Nx.host Nx.Rng.key; p : Nx.host Nx.packed } [@@deriving ptree]
                                                                                                             ^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [p] has type Nx.host Nx.packed, an array on Nx.host; a fixed array lies on the type's set parameter 'd: write 'd there, or leave the part out with [@ptree.skip] if no program on the set reads it
                     |}
                 );
                 ( "on the host, the set unnamed",
                   {|type ('a, _) t = { w : 'a; steps : Nx.host Nx.int32_t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 35-53:
                     1 | type ('a, _) t = { w : 'a; steps : Nx.host Nx.int32_t } [@@deriving ptree]
                                                            ^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [steps] has type Nx.host Nx.int32_t, an array on Nx.host; a fixed array lies on the type's set parameter: name the set, as in (_, 'd) t and 'd Nx.float32_t, or leave the part out with [@ptree.skip] if no program on the set reads it
                     |}
                 );
                 ( "on the payload parameter",
                   {|type ('a, 'd) t = { w : 'a Nx.float32_t; p : 'a Nx.packed } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 24-39:
                     1 | type ('a, 'd) t = { w : 'a Nx.float32_t; p : 'a Nx.packed } [@@deriving ptree]
                                                 ^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [w] has type 'a Nx.float32_t, whose set is the payload parameter 'a; a position of the payload is 'a alone, and a fixed array lies on the type's set parameter, as in (_, 'd) t and 'd Nx.float32_t

                     File "x.ml", line 1, characters 45-57:
                     1 | type ('a, 'd) t = { w : 'a Nx.float32_t; p : 'a Nx.packed } [@@deriving ptree]
                                                                      ^^^^^^^^^^^^
                     Error: ppx_ptree: field [p] has type 'a Nx.packed, whose set is the payload parameter 'a; a position of the payload is 'a alone, and a fixed array lies on the type's set parameter, as in (_, 'd) t and 'd Nx.float32_t
                     |}
                 );
               ];
             reports "more than two parameters are refused"
               [
                 ( "three",
                   {|type ('a, 'd, 'e) t = { value : 'a } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 0-55:
                     1 | type ('a, 'd, 'e) t = { value : 'a } [@@deriving ptree]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: [t] has 3 type parameters; a structure has at most two, its payload then its set, as ('a, 'd) t
                     |}
                 );
               ];
           ];
         group "generated values"
           [
             test "a declaration group binds one walk per type and nothing else"
               test_group_names;
             test "an interface declares one walk per type" test_interface_names;
             test
               "a structure's walks have ghost locations, except in \
                [@ptree.walk] payloads"
               test_ghost_impl;
             test "an interface's walks have ghost locations" test_ghost_intf;
           ];
         group "how a part is walked"
           [
             reports
               "a structure applied in another order is refused at the part"
               [
                 ( "('d, 'a) M.t",
                   {|type ('a, 'd) t = { w : 'a; l : ('d, 'a) Linear.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 33-35:
                     1 | type ('a, 'd) t = { w : 'a; l : ('d, 'a) Linear.t } [@@deriving ptree]
                                                          ^^
                     Error: ppx_ptree: field [l] is the set parameter 'd, which holds no value; an array on the set has a tensor type, as 'd Nx.float32_t
                     |}
                 );
               ];
           ];
         group "attributes"
           [
             reports "a part takes one attribute"
               [
                 ( "two of them",
                   {|type t = { count : int [@ptree.int] [@ptree.skip] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 11-49:
                     1 | type t = { count : int [@ptree.int] [@ptree.skip] } [@@deriving ptree]
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: a part takes one of [@ptree.int], [@ptree.skip] and [@ptree.walk]
                     |}
                 );
                 ( "one twice",
                   {|type t = { name : string [@ptree.skip] [@ptree.skip] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 41-51:
                     1 | type t = { name : string [@ptree.skip] [@ptree.skip] } [@@deriving ptree]
                                                                  ^^^^^^^^^^
                     Error: Duplicated attribute
                     |}
                 );
               ];
             reports
               "an attribute on a constructor without a single argument, or on \
                the declaration, is refused"
               [
                 ( "constructors",
                   {|type t = A [@ptree.int] | B of { x : int [@ptree.int] } [@ptree.skip] | C of int * string [@ptree.int] [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 9-23:
                     1 | type t = A [@ptree.int] | B of { x : int [@ptree.int] } [@ptree.skip] | C of int * string [@ptree.int] [@@deriving ptree]
                                  ^^^^^^^^^^^^^^
                     Error: ppx_ptree: [@ptree.int] on constructor [A], which has no argument; put it on an argument's type, as in [A of (int [@ptree.int]) * ...]

                     File "x.ml", line 1, characters 24-69:
                     1 | type t = A [@ptree.int] | B of { x : int [@ptree.int] } [@ptree.skip] | C of int * string [@ptree.int] [@@deriving ptree]
                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: [@ptree.skip] on constructor [B], which has a record argument; put it on an argument's type, as in [B of (int [@ptree.skip]) * ...]

                     File "x.ml", line 1, characters 70-102:
                     1 | type t = A [@ptree.int] | B of { x : int [@ptree.int] } [@ptree.skip] | C of int * string [@ptree.int] [@@deriving ptree]
                                                                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: [@ptree.int] on constructor [C], which has 2 arguments; put it on an argument's type, as in [C of (int [@ptree.int]) * ...]
                     |}
                 );
                 ( "the declaration",
                   {|type t = { name : string } [@@ptree.skip] [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 0-60:
                     1 | type t = { name : string } [@@ptree.skip] [@@deriving ptree]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: [@@ptree.skip] on type [t]: put the attribute on a field or a part

                     File "x.ml", line 1, characters 18-24:
                     1 | type t = { name : string } [@@ptree.skip] [@@deriving ptree]
                                           ^^^^^^
                     Error: ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
                     |}
                 );
               ];
             reports
               "an attribute in the arguments of a part walked whole is refused"
               [
                 ( "an element type",
                   {|type 'd t = { x : ((float[@ptree.int]), Nx.float32_elt, 'd) Nx.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 20-25:
                     1 | type 'd t = { x : ((float[@ptree.int]), Nx.float32_elt, 'd) Nx.t } [@@deriving ptree]
                                             ^^^^^
                     Error: ppx_ptree: [@ptree.int] on float, an argument of (float, Nx.float32_elt, 'd) Nx.t in field [x], which Nx.Ptree.Walk.tensor walks whole; remove it, or walk the whole part with ((float, Nx.float32_elt, 'd) Nx.t [@ptree.walk g])
                     |}
                 );
                 ( "a tensor alias's set",
                   {|type 'd t = { x : ('d [@ptree.int]) Nx.float32_t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-21:
                     1 | type 'd t = { x : ('d [@ptree.int]) Nx.float32_t } [@@deriving ptree]
                                            ^^
                     Error: ppx_ptree: [@ptree.int] on 'd, an argument of 'd Nx.float32_t in field [x], which Nx.Ptree.Walk.tensor walks whole; remove it, or walk the whole part with ('d Nx.float32_t [@ptree.walk g])
                     |}
                 );
                 ( "a key's set",
                   {|type 'd t = ('d[@ptree.walk f]) Nx.Rng.key list [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 13-15:
                     1 | type 'd t = ('d[@ptree.walk f]) Nx.Rng.key list [@@deriving ptree]
                                      ^^
                     Error: ppx_ptree: [@ptree.walk] on 'd, an argument of 'd Nx.Rng.key in type [t], which Nx.Rng.walk walks whole; remove it, or walk the whole part with ('d Nx.Rng.key [@ptree.walk g])
                     |}
                 );
                 ( "a packed array's set",
                   {|type 'd t = { p : ('d[@ptree.skip]) Nx.packed } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-21:
                     1 | type 'd t = { p : ('d[@ptree.skip]) Nx.packed } [@@deriving ptree]
                                            ^^
                     Error: ppx_ptree: [@ptree.skip] on 'd, an argument of 'd Nx.packed in field [p], which Nx.Ptree.Walk.tensor walks whole; remove it, or walk the whole part with ('d Nx.packed [@ptree.walk g])
                     |}
                 );
                 ( "a structure's set",
                   {|type ('a, 'd) t = { q : ('a, ('d[@ptree.skip])) Foo.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 30-32:
                     1 | type ('a, 'd) t = { q : ('a, ('d[@ptree.skip])) Foo.t } [@@deriving ptree]
                                                       ^^
                     Error: ppx_ptree: [@ptree.skip] on 'd, an argument of ('a, 'd) Foo.t in field [q], which Foo.walk walks whole; remove it, or walk the whole part with (('a, 'd) Foo.t [@ptree.walk g])
                     |}
                 );
                 ( "a nested structure's set",
                   {|type ('a, 'd) t = { s : (('a, 'd) V.t, ('d[@ptree.skip])) U.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 40-42:
                     1 | type ('a, 'd) t = { s : (('a, 'd) V.t, ('d[@ptree.skip])) U.t } [@@deriving ptree]
                                                                 ^^
                     Error: ppx_ptree: [@ptree.skip] on 'd, an argument of (('a, 'd) V.t, 'd) U.t in field [s], which U.walk walks whole; remove it, or walk the whole part with ((('a, 'd) V.t, 'd) U.t [@ptree.walk g])
                     |}
                 );
               ];
             reports
               "[@ptree.walk] on a structure's payload argument points to \
                Walk.under"
               [
                 ( "('a [@ptree.walk f], 'd) M.t",
                   {|type ('a, 'd) t = { p : (('a[@ptree.walk f]), 'd) Foo.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 26-28:
                     1 | type ('a, 'd) t = { p : (('a[@ptree.walk f]), 'd) Foo.t } [@@deriving ptree]
                                                   ^^
                     Error: ppx_ptree: [@ptree.walk] on 'a, the payload argument of ('a, 'd) Foo.t in field [p], which Foo.walk walks; walk its positions with (('a, 'd) Foo.t [@ptree.walk Nx.Ptree.Walk.under Foo.walk f])
                     |}
                 );
               ];
             reports "[@ptree.int] on a part with no int or bool is refused"
               [
                 ( "a string",
                   {|type t = { name : string [@ptree.int] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 18-24:
                     1 | type t = { name : string [@ptree.int] } [@@deriving ptree]
                                           ^^^^^^
                     Error: ppx_ptree: [@ptree.int] on field [name], whose type string has no int or bool
                     |}
                 );
               ];
             reports "[@ptree.walk] takes an expression"
               [
                 ( "none",
                   {|type 'd t = { value : 'd Nx.float32_t [@ptree.walk] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 14-51:
                     1 | type 'd t = { value : 'd Nx.float32_t [@ptree.walk] } [@@deriving ptree]
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: [@ptree.walk] takes the walk as an expression, as in [@ptree.walk M.walk]
                     |}
                 );
               ];
             reports "a short name is not the deriver's attribute"
               [
                 ( "[@skip]",
                   {|type t = { name : string [@skip] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 18-24:
                     1 | type t = { name : string [@skip] } [@@deriving ptree]
                                           ^^^^^^
                     Error: ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
                     |}
                 );
                 ( "[@int]",
                   {|type t = { count : int [@int] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-22:
                     1 | type t = { count : int [@int] } [@@deriving ptree]
                                            ^^^
                     Error: ppx_ptree: field [count] is an int; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "[@walk f]",
                   {|type t = { name : string [@walk f] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 18-24:
                     1 | type t = { name : string [@walk f] } [@@deriving ptree]
                                           ^^^^^^
                     Error: ppx_ptree: field [name] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
                     |}
                 );
               ];
           ];
         group "errors"
           [
             reports
               "an int or bool without an attribute, other scalar data and a \
                dtype are refused at the part"
               [
                 ( "an int",
                   {|type ('a, _) t = { w : 'a; count : int } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 35-38:
                     1 | type ('a, _) t = { w : 'a; count : int } [@@deriving ptree]
                                                            ^^^
                     Error: ppx_ptree: field [count] is an int; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "a bool",
                   {|type ('a, _) t = { w : 'a; causal : bool } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 36-40:
                     1 | type ('a, _) t = { w : 'a; causal : bool } [@@deriving ptree]
                                                             ^^^^
                     Error: ppx_ptree: field [causal] is a bool; report it with [@ptree.int] if a compiled program depends on it, or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "scalar data",
                   {|type t = {
  a : float;
  b : string;
  c : char;
  d : unit;
  e : bytes;
  f : int32;
  g : int64;
  h : nativeint;
  i : exn;
}
[@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 2, characters 6-11:
                     2 |   a : float;
                               ^^^^^
                     Error: ppx_ptree: field [a] is a float, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor

                     File "x.ml", line 3, characters 6-12:
                     3 |   b : string;
                               ^^^^^^
                     Error: ppx_ptree: field [b] is a string, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor

                     File "x.ml", line 4, characters 6-10:
                     4 |   c : char;
                               ^^^^
                     Error: ppx_ptree: field [c] is a char, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor

                     File "x.ml", line 5, characters 6-10:
                     5 |   d : unit;
                               ^^^^
                     Error: ppx_ptree: field [d] is a unit, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor

                     File "x.ml", line 6, characters 6-11:
                     6 |   e : bytes;
                               ^^^^^
                     Error: ppx_ptree: field [e] is a bytes, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor

                     File "x.ml", line 7, characters 6-11:
                     7 |   f : int32;
                               ^^^^^
                     Error: ppx_ptree: field [f] is a int32, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor

                     File "x.ml", line 8, characters 6-11:
                     8 |   g : int64;
                               ^^^^^
                     Error: ppx_ptree: field [g] is a int64, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor

                     File "x.ml", line 9, characters 6-15:
                     9 |   h : nativeint;
                               ^^^^^^^^^
                     Error: ppx_ptree: field [h] is a nativeint, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor

                     File "x.ml", line 10, characters 6-9:
                     10 |   i : exn;
                                ^^^
                     Error: ppx_ptree: field [i] is a exn, which has no walk; leave it out with [@ptree.skip], or hold data a compiled program depends on in a tensor
                     |}
                 );
                 ( "a dtype",
                   {|type t = { dtype : (float, Nx.float32_elt) Nx.dtype; dt : Nx.Dtype.float32_elt Nx.Dtype.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-51:
                     1 | type t = { dtype : (float, Nx.float32_elt) Nx.dtype; dt : Nx.Dtype.float32_elt Nx.Dtype.t } [@@deriving ptree]
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [dtype] has type (float, Nx.float32_elt) Nx.dtype; a dtype is data; leave it out with [@ptree.skip]

                     File "x.ml", line 1, characters 58-89:
                     1 | type t = { dtype : (float, Nx.float32_elt) Nx.dtype; dt : Nx.Dtype.float32_elt Nx.Dtype.t } [@@deriving ptree]
                                                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [dt] has type Nx.Dtype.float32_elt Nx.Dtype.t; a dtype is data; leave it out with [@ptree.skip]
                     |}
                 );
               ];
             reports
               "the set parameter alone, and a parameter in a tensor's element \
                types, are refused at the part"
               [
                 ( "the set parameter",
                   {|type ('a, 'd) t = { w : 'a; set : 'd } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 34-36:
                     1 | type ('a, 'd) t = { w : 'a; set : 'd } [@@deriving ptree]
                                                           ^^
                     Error: ppx_ptree: field [set] is the set parameter 'd, which holds no value; an array on the set has a tensor type, as 'd Nx.float32_t
                     |}
                 );
                 ( "the one parameter",
                   {|type 'a t = { w : 'a } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 18-20:
                     1 | type 'a t = { w : 'a } [@@deriving ptree]
                                           ^^
                     Error: ppx_ptree: field [w] is the set parameter 'a, which holds no value; an array on the set has a tensor type, as 'd Nx.float32_t. A type of one parameter holds fixed arrays only; for positions of a payload write ('a, _) t
                     |}
                 );
                 ( "an element type",
                   {|type ('a, 'd) t = { w : (float, 'a, 'd) Nx.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 24-44:
                     1 | type ('a, 'd) t = { w : (float, 'a, 'd) Nx.t } [@@deriving ptree]
                                                 ^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [w] has type (float, 'a, 'd) Nx.t; a tensor's element types cannot mention a parameter; a position of the payload is the payload parameter alone, walked as a leaf
                     |}
                 );
               ];
             reports "Nx types without a walk are refused at the part"
               [
                 ( "placements and meshes",
                   {|type ('a, 'd) t = { w : 'a; at : 'd Nx.Placement.t; mesh : 'd Nx.Mesh.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 33-50:
                     1 | type ('a, 'd) t = { w : 'a; at : 'd Nx.Placement.t; mesh : 'd Nx.Mesh.t } [@@deriving ptree]
                                                          ^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [at] has type 'd Nx.Placement.t; Nx.Placement.t has no walk: it says where arrays lie and holds none. Pass it beside the structure, or leave it out with [@ptree.skip] if its type mentions no parameter

                     File "x.ml", line 1, characters 59-71:
                     1 | type ('a, 'd) t = { w : 'a; at : 'd Nx.Placement.t; mesh : 'd Nx.Mesh.t } [@@deriving ptree]
                                                                                    ^^^^^^^^^^^^
                     Error: ppx_ptree: field [mesh] has type 'd Nx.Mesh.t; Nx.Mesh.t has no walk: it says where arrays lie and holds none. Pass it beside the structure, or leave it out with [@ptree.skip] if its type mentions no parameter
                     |}
                 );
                 ( "an index",
                   {|type 'd t = { at : 'd Nx.index } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-30:
                     1 | type 'd t = { at : 'd Nx.index } [@@deriving ptree]
                                            ^^^^^^^^^^^
                     Error: ppx_ptree: field [at] has type 'd Nx.index; Nx.index has no walk; walk it with [@ptree.walk f], or leave it out with [@ptree.skip] if its type mentions no parameter
                     |}
                 );
                 ( "a name ending in _t",
                   {|type 'd t = { scale : 'd Nx.float8_e4m3_t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-41:
                     1 | type 'd t = { scale : 'd Nx.float8_e4m3_t } [@@deriving ptree]
                                               ^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [scale] has type 'd Nx.float8_e4m3_t; Nx.float8_e4m3_t is not a tensor alias ppx_ptree knows: ('v, 's, 'd) Nx.t and aliases such as 'd Nx.float32_t
                     |}
                 );
               ];
             reports
               "a form with no derived walk is refused without [@ptree.walk] \
                or [@ptree.skip]"
               [
                 ( "ref",
                   {|type 'd t = { value : 'd Nx.float32_t ref } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-41:
                     1 | type 'd t = { value : 'd Nx.float32_t ref } [@@deriving ptree]
                                               ^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] has type 'd Nx.float32_t ref; ref has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "lazy_t",
                   {|type 'd t = { value : 'd Nx.float32_t lazy_t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-44:
                     1 | type 'd t = { value : 'd Nx.float32_t lazy_t } [@@deriving ptree]
                                               ^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] has type 'd Nx.float32_t lazy_t; lazy_t has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "result",
                   {|type 'd t = { value : ('d Nx.float32_t, string) result } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-54:
                     1 | type 'd t = { value : ('d Nx.float32_t, string) result } [@@deriving ptree]
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] has type ('d Nx.float32_t, string) result; result has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "format strings",
                   {|type t = { f : (int, unit, string, string) format4; g : (int, unit, string, string, string, string) format6 } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 15-50:
                     1 | type t = { f : (int, unit, string, string) format4; g : (int, unit, string, string, string, string) format6 } [@@deriving ptree]
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [f] has type (int, unit, string, string) format4; format4 has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]

                     File "x.ml", line 1, characters 56-107:
                     1 | type t = { f : (int, unit, string, string) format4; g : (int, unit, string, string, string, string) format6 } [@@deriving ptree]
                                                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [g] has type (int, unit, string, string, string, string) format6; format6 has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "a function",
                   {|type t = { callback : int -> int } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-32:
                     1 | type t = { callback : int -> int } [@@deriving ptree]
                                               ^^^^^^^^^^
                     Error: ppx_ptree: field [callback] is a function, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "an object",
                   {|type t = { value : < get : int > } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-32:
                     1 | type t = { value : < get : int > } [@@deriving ptree]
                                            ^^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] is an object type, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "a class",
                   {|type t = { value : #container } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-29:
                     1 | type t = { value : #container } [@@deriving ptree]
                                            ^^^^^^^^^^
                     Error: ppx_ptree: field [value] is a class type, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "a polymorphic variant",
                   {|type 'd t = { choice : [ `A of 'd Nx.float32_t ] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 23-48:
                     1 | type 'd t = { choice : [ `A of 'd Nx.float32_t ] } [@@deriving ptree]
                                                ^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [choice] is a polymorphic variant, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "a first-class module",
                   {|type t = { value : (module S) } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-29:
                     1 | type t = { value : (module S) } [@@deriving ptree]
                                            ^^^^^^^^^^
                     Error: ppx_ptree: field [value] is a first-class module, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "a polymorphic type",
                   {|type t = { value : 'a. 'a list } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-30:
                     1 | type t = { value : 'a. 'a list } [@@deriving ptree]
                                            ^^^^^^^^^^^
                     Error: ppx_ptree: field [value] is a polymorphic type, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "an alias",
                   {|type 'd t = { value : 'd Nx.float32_t as 'x } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-43:
                     1 | type 'd t = { value : 'd Nx.float32_t as 'x } [@@deriving ptree]
                                               ^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] is an alias [ty as 'a], which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "a local open",
                   {|type 'd t = { value : Shapes.('d Nx.float32_t) } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-46:
                     1 | type 'd t = { value : Shapes.('d Nx.float32_t) } [@@deriving ptree]
                                               ^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] is a locally opened type, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "an extension node",
                   {|type t = { value : [%unknown_type] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-34:
                     1 | type t = { value : [%unknown_type] } [@@deriving ptree]
                                            ^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] is an extension node, which has no derived walk; walk it with [@ptree.walk f] or leave it out with [@ptree.skip]
                     |}
                 );
                 ( "_",
                   {|type t = { value : _ } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-20:
                     1 | type t = { value : _ } [@@deriving ptree]
                                            ^
                     Error: ppx_ptree: field [value] is the wildcard _, which has no walk
                     |}
                 );
               ];
             reports "GADT and existential constructors are refused"
               [
                 ( "a GADT",
                   {|type ('a, 'd) t = Value : 'd Nx.float32_t -> ('a, 'd) t [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 18-55:
                     1 | type ('a, 'd) t = Value : 'd Nx.float32_t -> ('a, 'd) t [@@deriving ptree]
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: constructor [Value] has a GADT type, which a derived walk cannot rebuild at other parameters
                     |}
                 );
                 ( "an existential",
                   {|type ('a, 'd) t = Pack : 'b * 'a -> ('a, 'd) t [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 18-46:
                     1 | type ('a, 'd) t = Pack : 'b * 'a -> ('a, 'd) t [@@deriving ptree]
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: constructor [Pack] has a GADT type, which a derived walk cannot rebuild at other parameters
                     |}
                 );
               ];
             reports
               "abstract, extensible, private and constrained types are refused"
               [
                 ( "abstract",
                   {|type t [@@deriving ptree]|},
                   __POS_OF__
                     {|
                   File "x.ml", line 1, characters 0-25:
                   1 | type t [@@deriving ptree]
                       ^^^^^^^^^^^^^^^^^^^^^^^^^
                   Error: ppx_ptree: abstract type [t] has no parts to walk
                   |}
                 );
                 ( "extensible",
                   {|type t = .. [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 0-30:
                     1 | type t = .. [@@deriving ptree]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: extensible variant [t] has no derived walk
                     |}
                 );
                 ( "private",
                   {|type 'd t = private { value : 'd Nx.float32_t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 0-66:
                     1 | type 'd t = private { value : 'd Nx.float32_t } [@@deriving ptree]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: private type [t] cannot be rebuilt by a derived walk
                     |}
                 );
                 ( "constrained",
                   {|type ('a, _) t = { w : 'a } constraint 'a = int [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 0-66:
                     1 | type ('a, _) t = { w : 'a } constraint 'a = int [@@deriving ptree]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: type [t] has constraints, which a derived walk does not support
                     |}
                 );
               ];
             reports
               "a parameter in a skipped part, and a type variable that is not \
                a parameter, are refused"
               [
                 ( "a skipped parameter",
                   {|type ('a, _) t = { w : 'a [@ptree.skip] } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-39:
                     1 | type ('a, _) t = { w : 'a [@ptree.skip] } [@@deriving ptree]
                                            ^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: [@ptree.skip] copies field [w], whose type 'a mentions a parameter
                     |}
                 );
                 ( "a stray variable",
                   {|type ('a, _) t = { w : 'a; v : 'b } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 31-33:
                     1 | type ('a, _) t = { w : 'a; v : 'b } [@@deriving ptree]
                                                        ^^
                     Error: ppx_ptree: type variable 'b at field [v] is not a parameter of the type
                     |}
                 );
               ];
             reports
               "a type of the wrong arity, or a functor application, is \
                refused at the part"
               [
                 ( "a tensor alias",
                   {|type 'd t = { value : Nx.float32_t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-34:
                     1 | type 'd t = { value : Nx.float32_t } [@@deriving ptree]
                                               ^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] has type Nx.float32_t; Nx.float32_t takes one type argument, its set
                     |}
                 );
                 ( "Nx.t",
                   {|type 'd t = { value : (float, Nx.float32_elt) Nx.t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 22-50:
                     1 | type 'd t = { value : (float, Nx.float32_elt) Nx.t } [@@deriving ptree]
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Error: ppx_ptree: field [value] has type (float, Nx.float32_elt) Nx.t; Nx.t takes three type arguments: the element's OCaml type, its format and the set
                     |}
                 );
                 ( "option",
                   {|type t = { value : option } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 19-25:
                     1 | type t = { value : option } [@@deriving ptree]
                                            ^^^^^^
                     Error: ppx_ptree: field [value] has type option; option takes one type argument
                     |}
                 );
                 ( "a functor application",
                   {|type 'a t = { nested : 'a F(X).t } [@@deriving ptree]|},
                   __POS_OF__
                     {|
                     File "x.ml", line 1, characters 23-32:
                     1 | type 'a t = { nested : 'a F(X).t } [@@deriving ptree]
                                                ^^^^^^^^^
                     Error: ppx_ptree: field [nested] has type 'a F(X).t; functor applications in type paths have no walk
                     |}
                 );
               ];
           ];
       ])
