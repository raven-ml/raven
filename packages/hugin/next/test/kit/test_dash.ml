(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_kit

let dash = Testable.make ~pp:Dash.pp ~equal:Dash.equal
let lengths = list float_exact
let invalid f = raises_match (Exn.invalid_arg ?substring:None) f

let pp_lengths =
  Format.pp_print_list ~pp_sep:Format.pp_print_space Format.pp_print_float

let gen_lengths =
  Gen.list ~size:(Gen.int_range 0 5)
    (Gen.frequency
       [
         (4, Gen.float_range 0. 8.);
         (1, Gen.of_list ~pp:Format.pp_print_float [ 0.; 1.; 4. ]);
       ])
  |> Gen.map (fun l -> if List.fold_left ( +. ) 0. l = 0. then [] else l)
  |> Gen.with_pp pp_lengths

let patterns =
  group "patterns"
    [
      cases "v refuses lengths that make no pattern"
        ~name:(fun (n, _) -> n)
        [
          ("a negative length", [ 4.; -1. ]);
          ("nan", [ Float.nan ]);
          ("infinity", [ 1.; Float.infinity ]);
          ("a zero length alone", [ 0. ]);
          ("zero lengths", [ 0.; 0.; 0. ]);
        ]
        (fun (_, l) -> invalid (fun () -> Dash.v l));
      cases "v takes every pattern with a positive length"
        ~name:(fun (n, _) -> n)
        [
          ("dots", [ 0.; 2. ]);
          ("an odd pattern", [ 3. ]);
          ("a zero gap", [ 1.; 0. ]);
        ]
        (fun (_, l) -> equal lengths l (Dash.lengths (Dash.v l)));
      test "v of no length is solid" (fun () ->
          equal dash Dash.solid (Dash.v []));
      test "the presets have the lengths they state" (fun () ->
          equal (list lengths)
            [ []; [ 4.; 2. ]; [ 1.; 2. ]; [ 4.; 2.; 1.; 2. ] ]
            (List.map Dash.lengths
               [ Dash.solid; Dash.dashed; Dash.dotted; Dash.dash_dot ]));
      test "all is the presets in order" (fun () ->
          equal (list dash)
            [ Dash.solid; Dash.dashed; Dash.dotted; Dash.dash_dot ]
            Dash.all);
      prop "lengths reads back what v is given" gen_lengths (fun l ->
          equal lengths l (Dash.lengths (Dash.v l)));
    ]

let comparing =
  group "comparing"
    [
      prop "equal is an equivalence"
        (let d = Gen.map Dash.v gen_lengths |> Gen.with_pp Dash.pp in
         Gen.pair d d)
        (Law.equivalence dash);
      test "patterns of other lengths differ" (fun () ->
          not_equal dash Dash.dashed (Dash.v [ 4.; 2.; 4.; 2. ]);
          not_equal dash Dash.solid Dash.dotted);
      test "pp prints solid or the lengths" (fun () ->
          expect
            (Format.asprintf "%a|%a|%a" Dash.pp Dash.solid Dash.pp Dash.dash_dot
               Dash.pp (Dash.v [ 0.5 ]))
          @@ __POS_OF__ {| solid|4 2 1 2|0.5 |});
    ]

let () = exit (run "Dash" [ patterns; comparing ])
