(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit

let dash = Testable.make ~pp:Dash.pp ~equal:Dash.equal
let lengths = list float_exact
let invalid f = raises_match (Exn.invalid_arg ?sub:None) f

let pp_lengths =
  Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_float

(* Lengths that make a pattern: empty, or with a positive sum. *)
let gen_lengths =
  Gen.list ~size:(Gen.int_range 0 5)
    (Gen.frequency
       [
         (4, Gen.float_range 0. 8.);
         (1, Gen.of_list ~pp:Format.pp_print_float [ 0.; 1.; 4. ]);
       ])
  |> Gen.map (fun l -> if List.fold_left ( +. ) 0. l = 0. then [] else l)
  |> Gen.with_pp pp_lengths

(* Patterns *)

let patterns =
  group "patterns"
    [
      cases "v raises on lengths that make no pattern"
        ~id:(fun (n, _) -> n)
        [
          ("a negative length", [ 4.; -1. ]);
          ("nan", [ Float.nan ]);
          ("infinity", [ 1.; Float.infinity ]);
          ("neg_infinity", [ Float.neg_infinity; 2. ]);
          ("a zero length alone", [ 0. ]);
          ("zero lengths", [ 0.; 0.; 0. ]);
        ]
        (fun (_, l) -> invalid (fun () -> Dash.v l));
      test "v names a bad length as negative or not finite" (fun () ->
          raises
            (Invalid_argument "Dash.v: the length -1 is negative or not finite")
            (fun () -> Dash.v [ 4.; -1. ]));
      test "solid is v []" (fun () -> equal dash (Dash.v []) Dash.solid);
      cases "the presets have the lengths they state" ~id:fst
        [
          ("dashed", (Dash.dashed, [ 4.; 2. ]));
          ("dotted", (Dash.dotted, [ 1.; 2. ]));
          ("dash_dot", (Dash.dash_dot, [ 4.; 2.; 1.; 2. ]));
        ]
        (fun (_, (d, l)) -> equal dash (Dash.v l) d);
      test "all is solid, dashed, dotted and dash_dot in order" (fun () ->
          equal (list dash)
            [ Dash.solid; Dash.dashed; Dash.dotted; Dash.dash_dot ]
            Dash.all);
    ]

(* Observing *)

let observing =
  group "lengths"
    [
      prop "is what v is given"
        ~examples:[ [ 0.; 2. ]; [ 3. ]; [ 1.; 0. ] ]
        gen_lengths
        (fun l ->
          cover "a zero length" (List.mem 0. l);
          cover "an odd number of lengths" (List.length l mod 2 = 1);
          equal lengths l (Dash.lengths (Dash.v l)));
      test "is [] for solid" (fun () ->
          equal lengths [] (Dash.lengths Dash.solid));
    ]

(* Comparing and formatting *)

let comparing =
  group "equal"
    [
      (* Equal pairs come from the generator's repeats at p of about 1/8: the
         example reaches them. *)
      prop "holds iff the lengths are equal by Float.equal"
        ~examples:
          [
            ([ 4.; 2. ], [ 4.; 2. ]);
            ([ 4.; 2. ], [ 4.; 2.; 4.; 2. ]);
            ([], [ 1.; 2. ]);
          ]
        (Gen.pair gen_lengths gen_lengths)
        (fun (l, l') ->
          let same = List.equal Float.equal l l' in
          cover "equal lengths" same;
          cover "other lengths" (not same);
          if same then equal dash (Dash.v l) (Dash.v l')
          else not_equal dash (Dash.v l) (Dash.v l'));
    ]

let formatting =
  group "pp"
    [
      test "prints solid or the lengths" (fun () ->
          expect
            (Format.asprintf "%a|%a|%a" Dash.pp Dash.solid Dash.pp Dash.dash_dot
               Dash.pp (Dash.v [ 0.5 ]))
          @@ __POS_OF__ {| solid|4 2 1 2|0.5 |});
    ]

let () = exit (run "Dash" [ patterns; observing; comparing; formatting ])
