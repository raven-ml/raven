(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Norn.Support

let to_string s = Format.asprintf "%a" S.pp s

let all =
  S.
    [
      Real;
      Greater 0.;
      Greater (-1.5);
      Interval (0., 1.);
      Simplex 3;
      Ordered;
      Correlation_cholesky 4;
      Sum_to_zero;
      Integers_from 0;
      Integer_interval (0, 1);
      Integer_interval (0, 10);
      Boolean;
    ]

let printing =
  cases ~name:fst "pp"
    [
      ("(-inf, inf)", S.Real);
      ("(0, inf)", S.Greater 0.);
      ("(-1.5, inf)", S.Greater (-1.5));
      ("(0, 1)", S.Interval (0., 1.));
      ("{0, 1, 2, ...}", S.Integers_from 0);
      ("{0, 1}", S.Integer_interval (0, 1));
      ("{0, 1, 2}", S.Integer_interval (0, 2));
      ("{0, 1, ..., 10}", S.Integer_interval (0, 10));
      ("{false, true}", S.Boolean);
      ("simplex of 3", S.Simplex 3);
    ]
    (fun (expected, s) -> equal string expected (to_string s))

let equality =
  group "equal"
    [
      test "each support equals itself and no other" (fun () ->
          List.iteri
            (fun i s ->
              List.iteri
                (fun j s' ->
                  equal
                    ~msg:(to_string s ^ " vs " ^ to_string s')
                    bool (i = j) (S.equal s s'))
                all)
            all);
      test "bounds are compared" (fun () ->
          equal bool false (S.equal (S.Interval (0., 1.)) (S.Interval (0., 2.))));
    ]

let () = exit (run "Norn.Support" [ printing; equality ])
