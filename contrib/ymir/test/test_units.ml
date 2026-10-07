(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Astronomy's units, against their definitions: the IAU 2015 parsec, 648000/π
   au, and the Julian year, 365.25 days of 86400 s. *)

open Windtrap
open Ymir

let unit = Testable.make ~pp:Unit.pp ~equal:Unit.equal

let definitions =
  group "definitions"
    [
      test "the parsec subtends one arcsecond per astronomical unit" (fun () ->
          equal unit
            Unit.(astronomical_unit / arcsecond)
            Unit.(Units.parsec / radian));
      test "the parsec in metres, rounded once" (fun () ->
          (* IAU 2015 Resolution B2: 3.08567758149136727e16 m. *)
          equal float_exact 3.0856775814913673e16
            (Unit.ratio Nx.float64 Units.parsec Unit.metre));
      test "the Julian year is 31557600 s" (fun () ->
          equal unit Unit.(int 31557600 * second) Units.julian_year);
      test "the Julian year is 365.25 days" (fun () ->
          equal float_exact 365.25
            (Unit.ratio Nx.float64 Units.julian_year Unit.day));
    ]

let () = exit (run "Units" [ definitions ])
