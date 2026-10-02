(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_kit

let locale = Testable.make ~pp:Locale.pp ~equal:Locale.equal
let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
let months l = List.init 12 (fun i -> Locale.month l (i + 1))
let names s = Array.of_list (String.split_on_char ' ' s)
let english = names "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec"

let french =
  names "janv. févr. mars avr. mai juin juil. août sept. oct. nov. déc."

let with_month base i name =
  let m = Array.copy base in
  m.(i) <- name;
  m

(* Locales *)

let defaults =
  group "defaults"
    [
      test "default is v ()" (fun () ->
          equal locale (Locale.v ()) Locale.default);
      test "the decimal separator is a full stop" (fun () ->
          equal string "." (Locale.decimal Locale.default));
      test "the group separator is a comma" (fun () ->
          equal string "," (Locale.group Locale.default));
      test "groups have three digits" (fun () ->
          equal (list int) [ 3 ] (Locale.grouping Locale.default));
      test "the minus sign is U+2212" (fun () ->
          equal string "\u{2212}" (Locale.minus Locale.default));
      test "months have their English short names" (fun () ->
          equal (list string) (Array.to_list english) (months Locale.default));
    ]

let given =
  group "given strings"
    [
      test "are kept" (fun () ->
          let l =
            Locale.v ~decimal:"," ~group:"\u{202F}" ~grouping:[ 3; 2 ]
              ~minus:"-" ~months:french ()
          in
          equal string "," (Locale.decimal l);
          equal string "\u{202F}" (Locale.group l);
          equal (list int) [ 3; 2 ] (Locale.grouping l);
          equal string "-" (Locale.minus l);
          equal (list string) (Array.to_list french) (months l));
      test "an empty group separator is allowed" (fun () ->
          equal string "" (Locale.group (Locale.v ~group:"" ())));
      test "the months array is copied" (fun () ->
          let m = Array.copy french in
          let l = Locale.v ~months:m () in
          m.(0) <- "x";
          equal string "janv." (Locale.month l 1));
      cases
        ~name:(fun (g, _) -> String.concat ";" (List.map string_of_int g))
        "grouping drops the repeats of its last size"
        [
          ([ 3; 3 ], [ 3 ]); ([ 3; 2; 2; 2 ], [ 3; 2 ]); ([ 2; 3; 3 ], [ 2; 3 ]);
        ]
        (fun (g, kept) ->
          equal (list int) kept (Locale.grouping (Locale.v ~grouping:g ())));
    ]

let errors =
  let bad = "\xff" in
  group "errors"
    [
      cases ~name:fst "v raises Invalid_argument on"
        [
          ("an empty grouping", fun () -> Locale.v ~grouping:[] ());
          ("a group size of 0", fun () -> Locale.v ~grouping:[ 3; 0 ] ());
          ("a negative group size", fun () -> Locale.v ~grouping:[ -1 ] ());
          ( "eleven months",
            fun () -> Locale.v ~months:(Array.sub french 0 11) () );
          ( "thirteen months",
            fun () -> Locale.v ~months:(Array.append french [| "x" |]) () );
          ("an empty decimal separator", fun () -> Locale.v ~decimal:"" ());
          ("an empty minus sign", fun () -> Locale.v ~minus:"" ());
          ( "an empty month name",
            fun () -> Locale.v ~months:(with_month french 5 "") () );
          ( "a group separator equal to the decimal",
            fun () -> Locale.v ~group:"." () );
          ( "a decimal separator equal to the group",
            fun () -> Locale.v ~decimal:"," () );
          ("an invalid UTF-8 decimal", fun () -> Locale.v ~decimal:bad ());
          ("an invalid UTF-8 group", fun () -> Locale.v ~group:bad ());
          ("an invalid UTF-8 minus", fun () -> Locale.v ~minus:bad ());
          ( "an invalid UTF-8 month",
            fun () -> Locale.v ~months:(with_month french 11 bad) () );
        ]
        (fun (_, f) -> invalid f);
      cases ~name:string_of_int "month raises outside [1;12]" [ 0; 13; -1 ]
        (fun m -> invalid (fun () -> Locale.month Locale.default m));
    ]

(* Comparing *)

let gen_locale =
  let open Gen in
  let+ decimal = of_list [ "."; "," ]
  and+ group = of_list [ "."; ","; "\u{202F}"; "" ]
  and+ grouping = of_list [ [ 3 ]; [ 3; 2 ]; [ 4 ] ]
  and+ minus = of_list [ "-"; "\u{2212}" ]
  and+ months = of_list [ english; french; with_month english 4 "May." ] in
  let group = if group = decimal then "" else group in
  Locale.v ~decimal ~group ~grouping ~minus ~months ()

let gen_locale = Gen.with_pp Locale.pp gen_locale

let comparing =
  group "equal"
    [
      prop "is an equivalence"
        (Gen.pair gen_locale gen_locale)
        (Law.equivalence locale);
      test "tells a month name apart" (fun () ->
          not_equal locale Locale.default
            (Locale.v ~months:(with_month english 4 "May.") ()));
      test "tells a grouping apart" (fun () ->
          not_equal locale Locale.default (Locale.v ~grouping:[ 3; 2 ] ()));
      test "a repeated last size is the same grouping" (fun () ->
          equal locale Locale.default (Locale.v ~grouping:[ 3; 3 ] ()));
    ]

let () = exit (run "Locale" [ defaults; given; errors; comparing ])
