(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Time scales: instants convert between UTC, TAI, TT and TDB as SOFA's
   published values and ERFA's conversions state, leap seconds included, and
   round-trip within the stated bounds. *)

open Windtrap
open Ymir
module Reference = Ymir_test.Times_reference

let day_s = 86400.
let seconds x = Quantity.v Unit.second x
let in_seconds q = Quantity.value Unit.second q
let scalar x = Nx.scalar Nx.float64 x
let days n = Nx.scalar Nx.int64 (Int64.of_int n)
let at s day second = Time.v s ~day:(days day) ~second:(seconds (scalar second))
let utc_day = Time.of_date Time.utc

(* [instant t] is [t]'s one instant as (MJD, seconds since its 0h). *)
let instant t =
  ( Int64.to_int (Nx.item [] (Time.day t)),
    Nx.item [] (in_seconds (Time.second t)) )

(* [apart a b] is the seconds from the instant [b] to the instant [a]. *)
let apart (d, s) (d', s') = (float_of_int (d - d') *. day_s) +. (s -. s')

let close ~tol expected t =
  at_most float_exact ~than:tol (Float.abs (apart (instant t) expected))

(* [residual xs] is the sum of [xs] to about 1e-30 of its terms: each sum's
   error is carried exactly by [two_sum]. *)
let residual xs =
  let h, l =
    List.fold_left
      (fun (h, l) x ->
        let s, e = Ymir_test.Frames.two_sum h x in
        (s, l +. e))
      (0., 0.) xs
  in
  h +. l

let ulp x = Float.succ (Float.abs x) -. Float.abs x

let fraction t = Nx.item [] (snd (Time.jd t))
let to_tai t = Time.convert Time.tai t
let to_tt t = Time.convert Time.tt t
let to_tdb t = Time.convert Time.tdb t
let to_utc t = Time.convert Time.utc t

(* The last day of each leap second, and TAI - UTC that day, from IERS
   Bulletin C: a leap second ends the day before each change. *)
let leap_days =
  List.map
    (fun (start, after) -> (start - 1, after - 1))
    [
      (41499, 11); (41683, 12); (42048, 13); (42413, 14); (42778, 15);
      (43144, 16); (43509, 17); (43874, 18); (44239, 19); (44786, 20);
      (45151, 21); (45516, 22); (46247, 23); (47161, 24); (47892, 25);
      (48257, 26); (48804, 27); (49169, 28); (49534, 29); (50083, 30);
      (50630, 31); (51179, 32); (53736, 33); (54832, 34); (56109, 35);
      (57204, 36); (57754, 37);
    ]

let utc_start = 37300
let leap_seconds = 41317

(* Half an ulp of the seconds of a day, the most a sum of them rounds. *)
let half_ulp = 0x1p-37

let pre_1972_tol = 4e-11

(* SOFA's published values *)

(* The expected values of the IAU SOFA library's test program t_sofa_c.c,
   release 2023-10-11, and of its "Time Scale and Calendar Tools" cookbook. A
   day's fraction in SOFA's two-part Julian dates reads as 86400 of its
   seconds on these dates, which end with no leap second. *)

let sofa_day = 1e-12 *. day_s

let sofa =
  group "SOFA's published values"
    [
      cases "iauDat: TAI - UTC at 0h" ~id:(fun ((y, m, d), _) ->
          Printf.sprintf "%d-%02d-%02d" y m d)
        [ ((2003, 6, 1), 32.); ((2008, 1, 17), 33.); ((2017, 9, 1), 37.) ]
        (fun (date, delta) ->
          let t = utc_day date (0, 0, 0.) in
          equal float_exact delta (snd (instant (to_tai t))));
      test "iauUtctai: TAI 0.8924826384444444444 within 1e-12 day" (fun () ->
          let t = at Time.utc 53750 (0.892100694 *. day_s) in
          equal (float 1e-12) 0.8924826384444444444 (fraction (to_tai t)));
      test "iauTaiutc: UTC 0.8921006945555555556 within 1e-12 day" (fun () ->
          let t = at Time.tai 53750 (0.892482639 *. day_s) in
          equal (float sofa_day) (0.8921006945555555556 *. day_s)
            (snd (instant (to_utc t))));
      test "iauTaitt: TT 0.892855139 within 1e-12 day" (fun () ->
          let t = Time.of_jd Time.tai (scalar 2453750.5) (scalar 0.892482639) in
          equal (float 1e-12) 0.892855139 (fraction (to_tt t)));
      test "iauTttai: TAI 0.892110139 within 1e-12 day" (fun () ->
          let t = Time.of_jd Time.tt (scalar 2453750.5) (scalar 0.892482639) in
          equal (float 1e-12) 0.892110139 (fraction (to_tai t)));
      test "iauCal2jd: 2003-06-01 is MJD 52791" (fun () ->
          equal int 52791 (fst (instant (utc_day (2003, 6, 1) (0, 0, 0.)))));
      test "iauJd2cal: JD 2400000.5 + 50123.9999 is 1996-02-10, 0.9999 day"
        (fun () ->
          let t = Time.of_jd Time.tt (scalar 2400000.5) (scalar 50123.9999) in
          let date, _ = Time.date t in
          equal (triple int int int) (1996, 2, 10) date;
          equal (float 1e-7) 0.9999 (fraction t));
      test "iauDtf2d: UTC 1994-06-30 23:59:60.13599 is quasi-JD 2449534.49999"
        (fun () ->
          let t = utc_day (1994, 6, 30) (23, 59, 60.13599) in
          let d, s = instant t in
          let quasi = 2400000.5 +. float_of_int d +. (s /. (day_s +. 1.)) in
          equal (float 1e-6) 2449534.49999 quasi);
      test "iauD2dtf: UTC quasi-JD 2400000.5 + 49533.99999 is 23:59:60.13599"
        (fun () ->
          let t = at Time.utc 49533 (0.99999 *. (day_s +. 1.)) in
          let date, (h, mi, s) = Time.date t in
          equal (triple int int int) (1994, 6, 30) date;
          equal (pair int int) (23, 59) (h, mi);
          equal (float 5e-6) 60.13599 s);
      test "cookbook: TAI 2009-01-01 00:00:33.7 is UTC 2008-12-31 23:59:60.7"
        (fun () ->
          let t = to_utc (Time.of_date Time.tai (2009, 1, 1) (0, 0, 33.7)) in
          let date, (h, mi, s) = Time.date t in
          equal (triple int int int) (2008, 12, 31) date;
          equal (pair int int) (23, 59) (h, mi);
          equal (float 5e-4) 60.7 s);
      test
        "cookbook: UTC 2006-01-15 21:24:37.5 is TAI 21:25:10.5, TT \
         21:25:42.684"
        (fun () ->
          let t = utc_day (2006, 1, 15) (21, 24, 37.5) in
          let clock t = Time.date t in
          equal (pair (triple int int int) (triple int int (float 5e-7)))
            ((2006, 1, 15), (21, 25, 10.5))
            (clock (to_tai t));
          equal (pair (triple int int int) (triple int int (float 5e-7)))
            ((2006, 1, 15), (21, 25, 42.684))
            (clock (to_tt t)));
      test "cookbook: its topocentric TDB 21:25:42.684373 is within 2.1 us"
        (fun () ->
          let t = to_tdb (utc_day (2006, 1, 15) (21, 24, 37.5)) in
          let _, (h, mi, s) = Time.date t in
          equal (pair int int) (21, 25) (h, mi);
          equal (float (2.1e-6 +. 5e-7)) 42.684373 s);
    ]

(* ERFA's conversions *)

(* ERFA's two-part Julian date of a day's fraction resolves 2.2e-16 days,
   1.9e-11 s: its UTC is a quasi-JD of one such rounding, and each step to
   TAI, TT and TDB rounds again. *)
let erfa_tol = 6e-11

let erfa =
  let id ((y, m, d), (h, mi, s), _, _, _) =
    Printf.sprintf "%d-%02d-%02d %02d:%02d:%.6f" y m d h mi s
  in
  group "ERFA's conversions"
    [
      cases "UTC to TAI is eraUtctai's" ~id Reference.utc
        (fun (date, clock, tai, _, _) ->
          close ~tol:erfa_tol tai (to_tai (utc_day date clock)));
      cases "UTC to TT is eraTaitt's of eraUtctai's" ~id Reference.utc
        (fun (date, clock, _, tt, _) ->
          close ~tol:erfa_tol tt (to_tt (utc_day date clock)));
      cases "UTC to TDB is eraTttdb's with eraDtdb's TDB - TT at TDB" ~id
        Reference.utc (fun (date, clock, _, _, tdb) ->
          close ~tol:erfa_tol tdb (to_tdb (utc_day date clock)));
      cases "TDB - TT is eraDtdb's at the geocentre within 1e-15 s"
        ~id:(fun (d, _) -> string_of_int d)
        Reference.dtdb
        (fun (d, dtdb) ->
          let t = at Time.tt d 0.5 in
          equal (float 1e-15) dtdb (apart (instant (to_tdb t)) (d, 0.5)));
    ]


(* Leap seconds and UTC's span *)

let leaps =
  group "Leap seconds"
    [
      cases "a leap second's TAI is its second after 86400 plus TAI - UTC"
        ~id:(fun (d, _) -> string_of_int d)
        leap_days
        (fun (d, delta) ->
          let t = at Time.utc d 86400.5 in
          equal (pair int float_exact)
            (d + 1, 0.5 +. float_of_int delta)
            (instant (to_tai t)));
      cases "TAI in a leap second is UTC 23:59:60"
        ~id:(fun (d, _) -> string_of_int d)
        leap_days
        (fun (d, delta) ->
          let t = at Time.tai (d + 1) (float_of_int delta +. 0.25) in
          equal (pair int float_exact) (d, 86400.25) (instant (to_utc t)));
      test "2016-12-31 23:59:60.5 UTC is 2017-01-01 00:00:36.5 TAI" (fun () ->
          let t = to_tai (utc_day (2016, 12, 31) (23, 59, 60.5)) in
          equal
            (pair (triple int int int) (triple int int float_exact))
            ((2017, 1, 1), (0, 0, 36.5))
            (Time.date t));
      test "23:59:60 on a day without a leap second raises" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"past the end of the utc day")
            (fun () -> utc_day (2016, 12, 30) (23, 59, 60.)));
      test "a leap second ends at 61 s" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"86401 s long") (fun () ->
              utc_day (2016, 12, 31) (23, 59, 61.)));
      test "v refuses a second past a UTC day's end" (fun () ->
          let d = Nx.create Nx.int64 [| 2 |] [| 57753L; 57752L |] in
          let s = Nx.create Nx.float64 [| 2 |] [| 86400.5; 86400.5 |] in
          raises
            (Invalid_argument
               "Time.v: second 86400.5 at [1] is outside [0, 86400) of utc \
                day 57752")
            (fun () -> Time.v Time.utc ~day:d ~second:(seconds s)));
      test "1961-07-31 lasts 86399.95 UTC seconds" (fun () ->
          ignore (utc_day (1961, 7, 31) (23, 59, 59.9));
          raises_match (Exn.invalid_arg ~sub:"86399.95 s long") (fun () ->
              utc_day (1961, 7, 31) (23, 59, 59.95)));
      test "UTC begins on 1961-01-01" (fun () ->
          ignore (utc_day (1961, 1, 1) (0, 0, 0.));
          raises_match (Exn.invalid_arg ~sub:"precedes UTC") (fun () ->
              utc_day (1960, 12, 31) (23, 59, 59.)));
      test "TAI before UTC's start raises in convert" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"precedes UTC") (fun () ->
              to_utc (Time.of_date Time.tai (1961, 1, 1) (0, 0, 1.))));
      test "UTC's first instant converts to TDB and back" (fun () ->
          let t = utc_day (1961, 1, 1) (0, 0, 0.) in
          close ~tol:pre_1972_tol (utc_start, 0.) (to_utc (to_tdb t)));
      test "a TAI within 4e-11 s before UTC's start is UTC's first instant"
        (fun () ->
          let t = at Time.tai utc_start (1.422818 -. 3e-11) in
          equal (pair int float_exact) (utc_start, 0.) (instant (to_utc t)));
      test "TAI at UTC's start converts" (fun () ->
          let t = Time.of_date Time.tai (1961, 1, 1) (0, 0, 1.422818) in
          equal (pair int (float 1e-12)) (utc_start, 0.)
            (instant (to_utc t)));
      test "the table is known until Bulletin C 72's expiry, 2027-06-28"
        (fun () ->
          equal
            (pair (triple int int int) (triple int int float_exact))
            ((2027, 6, 28), (0, 0, 0.))
            (Time.date Time.leap_seconds_known_until));
      test "past the table's expiry, TAI - UTC stays its last value" (fun () ->
          let d, _ = instant Time.leap_seconds_known_until in
          let t = to_tai (at Time.utc (d + 1000) 0.) in
          equal (pair int float_exact) (d + 1000, 37.) (instant t));
    ]

(* Interval ends *)

(* SOFA's convention maps the last instants of a day that ends in a jump
   before 1972 onto the next day's first, or leaves a gap before them: by
   jump × rate / 86400 SI seconds, under 3.3e-9 s. *)
let overlap (_, length, _, rate, _) =
  Float.abs ((length -. day_s) *. rate /. day_s)

let jump_tol = 3.3e-9
let pre_1972 (d, _, _, _, _) = d < leap_seconds
let end_id (d, _, _, _, _) = string_of_int d

(* [utc_distance (d, length) (d', s') s] is the UTC seconds from [s] of day
   [d], [length] long, to [s'] of day [d'], the same day or the next. *)
let utc_distance (d, length) (d', s') s =
  if d' = d then Float.abs (s' -. s)
  else if d' = d + 1 then length -. s +. s'
  else Float.infinity

let ends =
  group "The ends of TAI - UTC's intervals"
    [
      cases "each jump's overlap is under 3.3e-9 s" ~id:end_id Reference.ends
        (fun e -> less float_exact ~than:jump_tol (overlap e));
      cases "a UTC day's last nanoseconds come back within the overlap"
        ~id:(fun ((d, _, _, _, _), k) -> Printf.sprintf "%d - %g s" d k)
        (List.concat_map
           (fun e -> List.map (fun k -> (e, k)) [ 1e-9; 3e-9; 1e-11 ])
           (List.filter pre_1972 Reference.ends))
        (fun (((d, length, _, _, _) as e), k) ->
          let s = length -. k in
          let back = instant (to_utc (to_tai (at Time.utc d s))) in
          let d', s' = back in
          if d' = d then less float_exact ~than:length s';
          at_most float_exact
            ~than:(overlap e +. pre_1972_tol)
            (utc_distance (d, length) back s));
      cases "a TAI just before a UTC 0h is a UTC instant of a day"
        ~id:(fun ((d, _, _, _, _), k) -> Printf.sprintf "%d - %g s" d k)
        (List.concat_map
           (fun e -> List.map (fun k -> (e, k)) [ 1e-12; 1e-9 ])
           Reference.ends)
        (fun (((d, length, _, _, next) as e), k) ->
          let tai = (d + 1, next -. k) in
          let t = to_utc (at Time.tai (fst tai) (snd tai)) in
          let d', s' = instant t in
          if d' = d then less float_exact ~than:length s'
          else equal int (d + 1) d';
          let tol =
            if pre_1972 e then overlap e +. pre_1972_tol else half_ulp
          in
          close ~tol tai (to_tai t));
    ]

(* Round trips *)

let mjd_last = 70000

(* UTC instants: in a leap second, before 1972, and from 1972, each drawn
   often. They begin a day after UTC does: a round trip through TT may round
   UTC's first instant to a TAI just before it, which has no UTC. *)
let utc_instants =
  let open Gen in
  frequency
    [
      ( 1,
        let+ d, _ = of_list leap_days and+ u = float_range 0. 1. in
        (d, Float.min (86400. +. u) (Float.pred 86401.)) );
      ( 1,
        let+ d = int_range (utc_start + 1) (leap_seconds - 1)
        and+ s = float_range 0. 86399.8 in
        (d, s) );
      ( 2,
        let+ d = int_range leap_seconds mjd_last
        and+ s = float_range 0. 86400. in
        (d, Float.min s (Float.pred 86400.)) );
    ]

(* Julian dates as two exact parts: whole days d and fractions k / 2^31 of the
   first part, which is near 2.4e6 days, and k / 2^40 of the second. *)
let splits =
  let open Gen in
  quad (int_range 2_300_000 2_500_000)
    (int_range 0 ((1 lsl 31) - 1))
    (int_range (-3000) 3000)
    (int_range 0 ((1 lsl 40) - 1))

let uniform_instants =
  let open Gen in
  let+ d = int_range (-1_000_000) 1_000_000
  and+ s = float_range 0. 86400. in
  (d, Float.min s (Float.pred 86400.))

let cover_utc (d, s) =
  cover ~share:0.15 "in a leap second" (s >= 86400.);
  cover ~share:0.15 "before 1972" (d < leap_seconds);
  cover ~share:0.15 "from 1972" (d >= leap_seconds && s < 86400.)

let round_trips =
  group "Round trips"
    [
      prop "UTC to TAI and back is the instant within half an ulp, before \
            1972 within 4e-11 s"
        utc_instants (fun (d, s) ->
          cover_utc (d, s);
          let tol = if d < leap_seconds then pre_1972_tol else half_ulp in
          close ~tol (d, s) (to_utc (to_tai (at Time.utc d s))));
      prop "TAI to UTC and back is the instant within half an ulp from 1972"
        uniform_instants (fun (d, s) ->
          let d = leap_seconds + 1 + (abs d mod (mjd_last - leap_seconds)) in
          close ~tol:half_ulp (d, s) (to_tai (to_utc (at Time.tai d s))));
      prop "a TAI second from 1972 is UTC's plus whole seconds, bit for bit"
        uniform_instants (fun (d, s) ->
          let d = leap_seconds + 1 + (abs d mod (mjd_last - leap_seconds)) in
          let s = Float.floor (s *. 1024.) /. 1024. in
          equal (pair int float_exact) (d, s)
            (instant (to_tai (to_utc (at Time.tai d s)))));
      prop "TAI to TT and back is within an ulp" uniform_instants
        (fun (d, s) ->
          close ~tol:(2. *. half_ulp) (d, s)
            (to_tai (to_tt (at Time.tai d s))));
      prop "TT to TDB and back is within an ulp" uniform_instants
        (fun (d, s) ->
          close ~tol:(2. *. half_ulp) (d, s)
            (to_tt (to_tdb (at Time.tt d s))));
      prop "UTC to TDB and back passes through TAI and TT" utc_instants
        (fun (d, s) ->
          let tol = if d < leap_seconds then pre_1972_tol else 6. *. half_ulp in
          close ~tol (d, s) (to_utc (to_tdb (at Time.utc d s))));
    ]

(* Batches *)

let batches =
  group "A batch converts as its instants"
    [
      prop "UTC to TDB of a batch is each instant's, bit for bit"
        (Gen.list ~size:(Gen.int_range 1 12) utc_instants)
        (fun xs ->
          let n = List.length xs in
          let day =
            Nx.create Nx.int64 [| n |]
              (Array.of_list (List.map (fun (d, _) -> Int64.of_int d) xs))
          in
          let second =
            Nx.create Nx.float64 [| n |] (Array.of_list (List.map snd xs))
          in
          let batch = to_tdb (Time.v Time.utc ~day ~second:(seconds second)) in
          let each =
            List.map (fun (d, s) -> instant (to_tdb (at Time.utc d s))) xs
          in
          let days = Array.map Int64.to_int (Nx.to_array (Time.day batch)) in
          let secs = Nx.to_array (in_seconds (Time.second batch)) in
          equal (list (pair int float_exact)) each
            (List.init n (fun i -> (days.(i), secs.(i)))));
      test "days and seconds broadcast" (fun () ->
          let day = Nx.create Nx.int64 [| 2; 1 |] [| 51544L; 51545L |] in
          let second = Nx.create Nx.float64 [| 3 |] [| 0.; 1.; 2. |] in
          let t = Time.v Time.tt ~day ~second:(seconds second) in
          equal (array int) [| 2; 3 |] (Nx.shape (in_seconds (Time.second t)));
          equal (array int) [| 2; 3 |] (Nx.shape (Time.day t)));
      test "a NaN second is a NaN instant through every conversion" (fun () ->
          let t = at Time.utc 57000 Float.nan in
          let back = to_utc (to_tdb t) in
          satisfies float_exact ~claim:"NaN" Float.is_nan (snd (instant back)));
    ]

(* Calendar and Julian dates *)

let dates =
  let open Gen in
  let* y, m = pair (int_range (-5000) 5000) (int_range 1 12) in
  let last = match m with 2 -> 28 | 4 | 6 | 9 | 11 -> 30 | _ -> 31 in
  let+ d = int_range 1 last
  and+ h = int_range 0 23
  and+ mi = int_range 0 59
  and+ s = float_range 0. 60. in
  ((y, m, d), (h, mi, Float.min s (Float.pred 60.)))

let calendar =
  group "Calendar and Julian dates"
    [
      test "MJD 0 began 1858-11-17" (fun () ->
          let t = Time.of_date Time.tt (1858, 11, 17) (0, 0, 0.) in
          equal int 0 (fst (instant t)));
      test "1994 April 1 commenced at MJD 49443, JD 2449443.5" (fun () ->
          let t = Time.of_date Time.tt (1994, 4, 1) (0, 0, 0.) in
          equal int 49443 (fst (instant t));
          equal float_exact 2449443.5 (Nx.item [] (fst (Time.jd t))));
      test "JD 0 is noon of -4713-11-24" (fun () ->
          let t = Time.of_date Time.tt (-4713, 11, 24) (12, 0, 0.) in
          let j1, j2 = Time.jd t in
          equal float_exact 0. (Nx.item [] j1 +. Nx.item [] j2));
      prop "of_date of date is the instant, bit for bit" dates
        (fun (date, clock) ->
          let t = Time.of_date Time.tai date clock in
          let date', clock' = Time.date t in
          equal (pair int float_exact) (instant t)
            (instant (Time.of_date Time.tai date' clock')));
      prop "of_date is the date's 0h and the clock's seconds within half an ulp"
        dates (fun (date, (h, mi, sec)) ->
          let t = Time.of_date Time.tai date (h, mi, sec) in
          let d0 = fst (instant (Time.of_date Time.tai date (0, 0, 0.))) in
          let total = (3600. *. float_of_int h) +. (60. *. float_of_int mi) in
          at_most float_exact ~than:half_ulp
            (Float.abs (residual [ apart (instant t) (d0, total); -.sec ])));
      test "a clock that rounds to the day's end is the next day's 0h"
        (fun () ->
          let t = Time.of_date Time.tai (0, 1, 1) (23, 59, Float.pred 60.) in
          let next = Time.of_date Time.tai (0, 1, 2) (0, 0, 0.) in
          equal (pair int float_exact) (instant next) (instant t));
      prop "of_jd of jd is the instant within an ulp" uniform_instants
        (fun (d, s) ->
          let j1, j2 = Time.jd (at Time.tdb d s) in
          close ~tol:(2. *. half_ulp) (d, s) (Time.of_jd Time.tdb j1 j2));
      prop "of_jd is within 2e-11 s of j1 + j2, either way round" splits
        (fun (d1, k1, d2, k2) ->
          (* j1 = d1 + k1 / 2^31 and j2 = d2 + k2 / 2^40 are exact, and so is
             MJD = d1 + d2 - 2400000 days plus n / 2^40 s. *)
          let j1 = float_of_int d1 +. Float.ldexp (float_of_int k1) (-31)
          and j2 = float_of_int d2 +. Float.ldexp (float_of_int k2) (-40) in
          let n = ((k1 * 512) - (1 lsl 39) + k2) * 86400 in
          let days = d1 + d2 - 2400000 in
          let error t =
            let d, s = instant t in
            let units = Float.round (Float.ldexp s 40) in
            let whole = (d - days) * 86400 * (1 lsl 40) in
            let e = whole + int_of_float units - n in
            Float.abs (Float.ldexp (float_of_int e) (-40))
          in
          at_most float_exact ~than:2e-11
            (error (Time.of_jd Time.tt (scalar j1) (scalar j2)));
          at_most float_exact ~than:2e-11
            (error (Time.of_jd Time.tt (scalar j2) (scalar j1))));
      test "v refuses an infinite second" (fun () ->
          raises (Invalid_argument "Time.v: second is infinite") (fun () ->
              at Time.tai 0 Float.infinity));
      test "v refuses a second that is not a time" (fun () ->
          raises_match Exn.(invalid_arg ~sub:"") (fun () ->
              Time.v Time.tai ~day:(days 0)
                ~second:(Quantity.v Unit.metre (scalar 1.))));
      test "v refuses a day past 2^53 days" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"more than 2^53 days") (fun () ->
              Time.v Time.tai
                ~day:(Nx.scalar Nx.int64 0x0020_0000_0000_0001L)
                ~second:(seconds (scalar 0.))));
      test "of_date refuses a date past 2^53 days" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"more than 2^53 days") (fun () ->
              Time.of_date Time.tt (25_000_000_000_000, 1, 1) (0, 0, 0.));
          raises_match (Exn.invalid_arg ~sub:"more than 2^53 days") (fun () ->
              Time.of_date Time.tt (max_int, 1, 1) (0, 0, 0.)));
      test "of_date refuses 2023-02-29" (fun () ->
          raises
            (Invalid_argument "Time.of_date: 2023-02-29 is not a date")
            (fun () -> Time.of_date Time.tt (2023, 2, 29) (0, 0, 0.)));
      test "of_date refuses 60 s outside a day's last minute" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"not a time of day") (fun () ->
              Time.of_date Time.tt (2016, 12, 31) (23, 58, 60.)));
      test "of_jd refuses an infinite part" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"not finite") (fun () ->
              Time.of_jd Time.tt (scalar Float.infinity) (scalar 0.)));
      test "date refuses a batch" (fun () ->
          let t =
            Time.v Time.tt ~day:(days 0)
              ~second:(seconds (Nx.create Nx.float64 [| 2 |] [| 0.; 1. |]))
          in
          raises_match (Exn.invalid_arg ~sub:"single instant") (fun () ->
              Time.date t));
    ]

(* Arithmetic *)

let spans =
  Gen.frequency
    [
      (2, Gen.float_range (-1e9) 1e9);
      (1, Gen.float_range (-1e-3) 1e-3);
      (1, Gen.map (fun x -> -.day_s +. x) (Gen.float_range (-1e-3) 1e-3));
    ]


let arithmetic =
  group "Arithmetic on uniform scales"
    [
      prop "add is the instant span later within 7.3e-12 s"
        (Gen.pair uniform_instants spans)
        (fun ((d, s), span) ->
          let t = Time.add (at Time.tt d s) (seconds (scalar span)) in
          let d', s' = instant t in
          let days = float_of_int (d' - d) *. day_s in
          at_most float_exact ~than:half_ulp
            (Float.abs (residual [ days; s'; -.s; -.span ])));
      prop "diff is a - b within half an ulp and 7.3e-12 s"
        (Gen.pair uniform_instants uniform_instants)
        (fun ((da, sa), (db, sb)) ->
          let r = Time.diff (at Time.tai da sa) (at Time.tai db sb) in
          let r = Nx.item [] (in_seconds r) in
          let days = float_of_int (da - db) *. day_s in
          at_most float_exact
            ~than:((ulp r /. 2.) +. 7.3e-12)
            (Float.abs (residual [ r; -.days; -.sa; sb ])));
      test "add of a small negative span rounds once" (fun () ->
          let s = 80477.79928499414 and span = -6.300034147017414e-05 in
          let d', s' =
            instant (Time.add (at Time.tt 0 s) (seconds (scalar span)))
          in
          let days = float_of_int d' *. day_s in
          at_most float_exact ~than:half_ulp
            (Float.abs (residual [ days; s'; -.s; -.span ])));
      test "add carries into the next day" (fun () ->
          let t = Time.add (at Time.tai 51544 86399.5) (seconds (scalar 1.)) in
          equal (pair int float_exact) (51545, 0.5) (instant t));
      test "add moves back across days" (fun () ->
          let back = seconds (scalar (-.day_s *. 3.)) in
          let t = Time.add (at Time.tai 51544 0.25) back in
          equal (pair int float_exact) (51541, 0.25) (instant t));
      test "diff counts whole days exactly" (fun () ->
          let a = at Time.tdb 60000 12.5 and b = at Time.tdb 10000 2.5 in
          equal float_exact ((50000. *. day_s) +. 10.)
            (Nx.item [] (in_seconds (Time.diff a b))));
      test "add refuses a span past 2^53 days" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"more than 2^53 days") (fun () ->
              Time.add (at Time.tt 0 0.) (seconds (scalar 1e300))));
      test "add refuses an infinite span" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"not finite") (fun () ->
              Time.add (at Time.tt 0 0.) (seconds (scalar Float.infinity))));
    ]

(* Derivatives and compilation *)

let tdb_second x =
  let t = Time.v Time.tt ~day:(days 60000) ~second:(seconds x) in
  in_seconds (Time.second (to_tdb t))

let derivatives =
  group "Derivatives and compilation"
    [
      test "d TDB / d TT is 1 within 4e-10" (fun () ->
          let g = Rune.grad' (fun x -> Nx.sum (tdb_second x)) (scalar 43200.) in
          equal (float 4e-10) 1. (Nx.item [] g));
      test "UTC to TDB compiles to the eager result, bit for bit" (fun () ->
          let second = Nx.linspace Nx.float64 0. 86399. 16 in
          let f x =
            in_seconds
              (Time.second
                 (to_tdb
                    (Time.v Time.utc ~day:(days 57753) ~second:(seconds x))))
          in
          equal (array float_exact) (Nx.to_array (f second))
            (Nx.to_array (Rune.jit' f second)));
    ]

(* Structure *)

let structure =
  group "Structure"
    [
      test "visits" (fun () ->
          List.iter
            (fun v -> Format.printf "%a@." Nx.Ptree.pp_visit v)
            (Nx.Ptree.visits (Time.ptree ()) (at Time.tdb 1 2.));
          expect (output ())
          @@ __POS_OF__
               {|
            the root: case "tdb"
            day: a leaf
            second: a leaf
            |});
      test "name is the scale's name" (fun () ->
          equal (list string) [ "utc"; "tai"; "tt"; "tdb" ]
            [
              Time.name Time.utc;
              Time.name Time.tai;
              Time.name Time.tt;
              Time.name Time.tdb;
            ]);
    ]

let () =
  exit
    (run "Time"
       [ sofa; erfa; leaps; ends; round_trips; batches; calendar; arithmetic;
         derivatives; structure ])
