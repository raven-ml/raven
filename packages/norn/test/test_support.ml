(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Norn.Support

let to_string s = Format.asprintf "%a" S.pp s
let support = Testable.make ~pp:S.pp ~equal:S.equal

(* Every constructor, bounds that differ, and the floats [Float.equal] reads as
   equal though they differ: NaN, and [0.] against [-0.]. *)
let supports =
  S.
    [
      Real;
      Greater 0.;
      Greater (-0.);
      Greater (-1.5);
      Greater Float.nan;
      Less 0.;
      Less 2.5;
      Interval (0., 1.);
      Interval (0., 2.);
      Simplex 3;
      Simplex 4;
      Ordered;
      Correlation_cholesky 4;
      Sum_to_zero;
      Integers_from 0;
      Integer_interval (0, 1);
      Integer_interval (0, 10);
      Boolean;
    ]

(* Structural comparison is [Float.equal] on floats. *)
let equal_ =
  group "equal"
    [
      test "is constructor and argument equality, floats by Float.equal"
        (fun () ->
          List.iter
            (fun s ->
              List.iter
                (fun s' ->
                  satisfies
                    ~msg:(to_string s ^ " against " ^ to_string s')
                    ~claim:"S.equal agrees with compare = 0"
                    (pair support support)
                    (fun (s, s') -> S.equal s s' = (compare s s' = 0))
                    (s, s'))
                supports)
            supports);
    ]

let pp =
  group "pp"
    [
      cases "formats a support as a set" ~id:fst
        [
          ("(-inf, inf)", S.Real);
          ("(0, inf)", S.Greater 0.);
          ("(-1.5, inf)", S.Greater (-1.5));
          ("(-inf, 0)", S.Less 0.);
          ("(-inf, 2.5)", S.Less 2.5);
          ("(0, 1)", S.Interval (0., 1.));
          ("{0, 1, 2, ...}", S.Integers_from 0);
          ("{0, 1}", S.Integer_interval (0, 1));
          ("{0, 1, 2}", S.Integer_interval (0, 2));
          ("{0, 1, ..., 10}", S.Integer_interval (0, 10));
          ("{false, true}", S.Boolean);
          ("simplex of 3", S.Simplex 3);
        ]
        (fun (expected, s) -> equal string expected (to_string s));
      test "an empty integer interval is {}" (fun () ->
          equal (list string) [ "{}"; "{}" ]
            [
              to_string (S.Integer_interval (5, 4));
              to_string (S.Integer_interval (5, 3));
            ]);
      prop "a bound reads back as itself" Gen.float
        (Law.round_trip float_exact string
           (fun x -> to_string (S.Greater x))
           (fun s -> float_of_string (String.sub s 1 (String.index s ',' - 1))));
      (* The phrases of the other supports, which norn.mli leaves to pp. *)
      test "the other supports are phrases" (fun () ->
          expect
            (String.concat "\n"
               (List.map to_string
                  S.[ Ordered; Correlation_cholesky 4; Sum_to_zero ]))
          @@ __POS_OF__
               {|
                 ordered vectors
                 Cholesky factors of 4 x 4 correlations
                 vectors summing to zero
               |});
    ]

let () = exit (run "Norn.Support" [ equal_; pp ])
