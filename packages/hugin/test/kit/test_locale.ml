(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit

let locale = Testable.make ~pp:Locale.pp ~equal:Locale.equal
let invalid f = raises_match (Exn.invalid_arg ?sub:None) f
let months l = List.init 12 (fun i -> Locale.month l (i + 1))
let names s = Array.of_list (String.split_on_char ' ' s)
let english = names "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec"

let french =
  names "janv. févr. mars avr. mai juin juil. août sept. oct. nov. déc."

let with_month base i name =
  let m = Array.copy base in
  m.(i) <- name;
  m

(* What a locale holds, read through its accessors. *)
let strings l =
  ( (Locale.decimal l, Locale.group l, Locale.grouping l),
    (Locale.minus l, months l) )

let strings_w =
  pair (triple string string (list int)) (pair string (list string))

(* The arguments of [v], each from a short list, with groupings that end in no
   repeat of their last size. *)
let gen_args =
  let open Gen in
  let pp ppf (decimal, group, grouping, minus, months) =
    Format.fprintf ppf "~decimal:%S ~group:%S ~grouping:[%s] ~minus:%S %s"
      decimal group
      (String.concat "; " (List.map string_of_int grouping))
      minus
      (if months == english then "English" else "French")
  in
  with_pp pp
    (let+ decimal = of_list [ "."; ","; "\u{066B}" ]
     and+ group = of_list [ "."; ","; "\u{202F}"; "" ]
     and+ grouping = of_list [ [ 3 ]; [ 3; 2 ]; [ 4 ]; [ 1; 2; 3 ] ]
     and+ minus = of_list [ "-"; "\u{2212}" ]
     and+ months = of_list [ english; french ] in
     let group = if group = decimal then "" else group in
     (decimal, group, grouping, minus, months))

let gen_locale =
  Gen.with_pp Locale.pp
    (Gen.map
       (fun (decimal, group, grouping, minus, months) ->
         Locale.v ~decimal ~group ~grouping ~minus ~months ())
       gen_args)

(* Locales *)

let v =
  group "v"
    [
      test "decimal defaults to a full stop" (fun () ->
          equal string "." (Locale.decimal (Locale.v ())));
      test "group defaults to a comma" (fun () ->
          equal string "," (Locale.group (Locale.v ())));
      test "grouping defaults to [3]" (fun () ->
          equal (list int) [ 3 ] (Locale.grouping (Locale.v ())));
      test "minus defaults to U+2212" (fun () ->
          equal string "\u{2212}" (Locale.minus (Locale.v ())));
      test "months default to their English short names" (fun () ->
          equal (list string) (Array.to_list english) (months (Locale.v ())));
      test "takes an empty group separator" (fun () ->
          equal string "" (Locale.group (Locale.v ~group:"" ())));
      test "copies the months array" (fun () ->
          let m = Array.copy french in
          let l = Locale.v ~months:m () in
          m.(0) <- "x";
          equal string "janv." (Locale.month l 1));
      cases "raises on" ~id:fst
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
          ("a decimal of byte 0xFF", fun () -> Locale.v ~decimal:"\xff" ());
          ("a group of byte 0xFF", fun () -> Locale.v ~group:"\xff" ());
          ("a minus of byte 0xFF", fun () -> Locale.v ~minus:"\xff" ());
          ( "a month of byte 0xFF",
            fun () -> Locale.v ~months:(with_month french 11 "\xff") () );
          ( "a decimal cut inside a sequence",
            fun () -> Locale.v ~decimal:"\xc3" () );
          ( "a decimal in an overlong encoding",
            fun () -> Locale.v ~decimal:"\xc0\xae" () );
          ( "a decimal that encodes a surrogate",
            fun () -> Locale.v ~decimal:"\xed\xa0\x80" () );
        ]
        (fun (_, f) -> invalid f);
    ]

let default =
  group "default"
    [ test "is v ()" (fun () -> equal locale (Locale.v ()) Locale.default) ]

(* Accessors *)

let accessors =
  group "accessors"
    [
      prop "return what v is given" gen_args
        (fun (decimal, group, grouping, minus, months) ->
          let l = Locale.v ~decimal ~group ~grouping ~minus ~months () in
          equal strings_w
            ((decimal, group, grouping), (minus, Array.to_list months))
            (strings l));
      cases "grouping drops the repeats of its last size"
        ~id:(fun (g, _) -> String.concat ";" (List.map string_of_int g))
        [
          ([ 3; 3 ], [ 3 ]);
          ([ 3; 2; 2; 2 ], [ 3; 2 ]);
          ([ 2; 3; 3 ], [ 2; 3 ]);
          ([ 3; 3; 2 ], [ 3; 3; 2 ]);
        ]
        (fun (g, kept) ->
          equal (list int) kept (Locale.grouping (Locale.v ~grouping:g ())));
      cases "month raises outside [1;12]" ~id:string_of_int
        [ 0; 13; -1; min_int; max_int ] (fun m ->
          invalid (fun () -> Locale.month Locale.default m));
    ]

(* Comparing *)

let comparing =
  group "equal"
    [
      (* Equal pairs come at p of about 1/8, and each field apart alone rarely:
         the examples reach both. *)
      prop "holds iff the accessors agree"
        ~examples:
          [
            (Locale.default, Locale.v ~grouping:[ 3; 3 ] ());
            (Locale.v ~group:"" (), Locale.v ~group:"" ~decimal:"," ());
            (Locale.default, Locale.v ~group:"" ());
            (Locale.default, Locale.v ~grouping:[ 3; 2 ] ());
            (Locale.default, Locale.v ~minus:"-" ());
            (Locale.default, Locale.v ~months:(with_month english 4 "May.") ());
          ]
        (Gen.pair gen_locale gen_locale)
        (fun (l, l') ->
          let same = strings l = strings l' in
          cover "equal accessors" same;
          cover "other accessors" (not same);
          if same then equal locale l l' else not_equal locale l l');
    ]

let () = exit (run "Locale" [ v; default; accessors; comparing ])
