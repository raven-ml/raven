(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_kit

let instant =
  Testable.with_compare Time.compare
    (Testable.make ~pp:Time.pp ~equal:Time.equal)

let interval = Testable.make ~pp:Time.pp_interval ~equal:Time.equal_interval
let date = triple int int int
let date_time = pair date date
let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
let ns k = Time.nanoseconds k
let at ?tz_offset_s d tm = Time.of_date_time ?tz_offset_s (d, tm)
let utc s = Time.v S (Int64.of_int s)

(* [with_ns t k] is [t] moved by [k >= 0] nanoseconds. *)
let with_ns t k = if k = 0 then t else Time.add (ns k) 1 t

(* Generators *)

let gen_sec =
  Gen.frequency
    [
      (4, Gen.map Int64.of_int (Gen.int_range (-10_000_000_000) 10_000_000_000));
      (2, Gen.int64);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf n -> Format.fprintf ppf "%LdL" n)
          [
            Int64.min_int;
            Int64.succ Int64.min_int;
            Int64.pred Int64.max_int;
            Int64.max_int;
            0L;
            -1L;
          ] );
    ]

let gen_nsec =
  Gen.frequency
    [ (1, Gen.of_list [ 0; 1; 999_999_999 ]); (3, Gen.int_range 0 999_999_999) ]

let gen_instant =
  Gen.with_pp Time.pp
    (Gen.map
       (fun (s, n) ->
         let t = Time.v S s in
         if Int64.equal s Int64.max_int then t else with_ns t n)
       (Gen.pair gen_sec gen_nsec))

let gen_tz =
  Gen.frequency
    [
      (2, Gen.of_list [ 0; 3600; -18000; 19800; 86399; -86399 ]);
      (1, Gen.int_range (-86399) 86399);
    ]

let gen_interval =
  Gen.with_pp Time.pp_interval
    (Gen.one_of
       [
         Gen.map ns (Gen.int_range 1 2_000_000_000);
         Gen.map Time.seconds (Gen.of_list [ 1; 5; 15; 30; 7 ]);
         Gen.map Time.minutes (Gen.of_list [ 1; 5; 15; 30 ]);
         Gen.map Time.hours (Gen.of_list [ 1; 3; 6; 12; 5 ]);
         Gen.map Time.days (Gen.of_list [ 1; 2; 3 ]);
         Gen.map Time.weeks (Gen.of_list [ 1; 2 ]);
         Gen.map Time.months (Gen.of_list [ 1; 2; 3; 6; 7 ]);
         Gen.map Time.years (Gen.of_list [ 1; 2; 5; 10; 100; 1000 ]);
       ])

(* Instants whose boundaries are representable: within a few centuries. *)
let gen_near =
  Gen.with_pp Time.pp
    (Gen.map
       (fun (s, n) -> with_ns (Time.v S (Int64.of_int s)) n)
       (Gen.pair (Gen.int_range (-20_000_000_000) 20_000_000_000) gen_nsec))

(* Instants *)

let instants =
  group "instants"
    [
      test "a millisecond before the epoch decomposes with a positive nsec"
        (fun () ->
          let t = Time.v Ms (-1L) in
          equal int64 (-1L) t.Time.sec;
          equal int 999_000_000 t.Time.nsec);
      test "counts of different resolutions are one instant" (fun () ->
          equal instant (Time.v S 1L) (Time.v Ms 1000L);
          equal instant (Time.v Us (-1_500_000L)) (Time.v Ns (-1_500_000_000L)));
      test "the epoch is the zero count" (fun () ->
          equal instant (Time.v Ns 0L) Time.epoch);
      test "to_int64 floors" (fun () ->
          equal (option int64) (Some (-1L)) (Time.to_int64 S (Time.v Ms (-1L)));
          equal (option int64) (Some 1L) (Time.to_int64 Ms (Time.v Us 1999L)));
      test "to_int64 is None when the count overflows" (fun () ->
          equal (option int64) None
            (Time.to_int64 Ns (Time.v S 10_000_000_000L));
          equal (option int64) None (Time.to_int64 Ms (Time.v S Int64.min_int)));
      test "to_int64 reaches the extremes of nanoseconds" (fun () ->
          equal (option int64) (Some Int64.max_int)
            (Time.to_int64 Ns (Time.v Ns Int64.max_int));
          equal (option int64) (Some Int64.min_int)
            (Time.to_int64 Ns (Time.v Ns Int64.min_int)));
      prop "to_int64 r (v r n) is n"
        (Gen.pair (Gen.of_list [ Time.S; Ms; Us; Ns ]) gen_sec)
        (fun (r, n) ->
          equal (option int64) (Some n) (Time.to_int64 r (Time.v r n)));
      prop "to_int64 is the greatest count not after the instant"
        (Gen.pair (Gen.of_list [ Time.Ms; Us; Ns ]) gen_near)
        (fun (r, t) ->
          let n =
            match Time.to_int64 r t with Some n -> n | None -> reject ()
          in
          at_most instant ~than:t (Time.v r n);
          less instant ~than:(Time.v r (Int64.succ n)) t);
      prop "compare is a total order"
        (Gen.triple gen_instant gen_instant gen_instant)
        (Law.order instant);
    ]

(* Civil dates and times *)

let known =
  [
    ((1970, 1, 1), (0, 0, 0), 0);
    ((1969, 12, 31), (0, 0, 0), -86_400);
    ((2000, 3, 1), (0, 0, 0), 951_868_800);
    ((2000, 2, 29), (12, 0, 0), 951_825_600);
    ((2026, 10, 1), (12, 30, 0), 1_790_857_800);
    ((1900, 1, 1), (0, 0, 0), -2_208_988_800);
    ((0, 1, 1), (0, 0, 0), -62_167_219_200);
    ((-1, 12, 31), (23, 59, 59), -62_167_219_201);
    ((2038, 1, 19), (3, 14, 8), 2_147_483_648);
  ]

let civil =
  group "civil"
    [
      cases
        ~name:(fun ((y, m, d), _, s) ->
          Printf.sprintf "%d-%02d-%02d is %d s" y m d s)
        "known instants" known
        (fun (d, tm, s) ->
          equal instant (utc s) (at d tm);
          equal date_time (d, tm) (Time.to_date_time (utc s)));
      test "an offset east of UTC is earlier in UTC" (fun () ->
          equal instant Time.epoch (at ~tz_offset_s:3600 (1970, 1, 1) (1, 0, 0));
          equal instant (utc 18_000)
            (Time.of_date ~tz_offset_s:(-18_000) (1970, 1, 1)));
      test "to_date_time reads local time before the epoch" (fun () ->
          equal date_time
            ((1969, 12, 31), (19, 0, 0))
            (Time.to_date_time ~tz_offset_s:(-18_000) Time.epoch));
      test "to_date_time drops nanoseconds" (fun () ->
          let t = Time.v Ms (-1L) in
          equal date_time ((1969, 12, 31), (23, 59, 59)) (Time.to_date_time t);
          not_equal instant t (Time.of_date_time (Time.to_date_time t)));
      test "of_date is the start of the day" (fun () ->
          equal instant (at (2026, 3, 3) (0, 0, 0)) (Time.of_date (2026, 3, 3)));
      test "the extremes of int64 seconds have dates" (fun () ->
          let check t =
            equal instant t (Time.of_date_time (Time.to_date_time t))
          in
          check (Time.v S Int64.max_int);
          check (Time.v S Int64.min_int);
          let (y, _, _), _ = Time.to_date_time (Time.v S Int64.max_int) in
          equal int 292_277_026_596 y);
      cases
        ~name:(fun (n, _) -> n)
        "of_date_time raises on"
        [
          ("month 0", fun () -> Time.of_date (2026, 0, 1));
          ("month 13", fun () -> Time.of_date (2026, 13, 1));
          ("day 0", fun () -> Time.of_date (2026, 1, 0));
          ("February 29 1900", fun () -> Time.of_date (1900, 2, 29));
          ("April 31", fun () -> Time.of_date (2026, 4, 31));
          ("hour 24", fun () -> at (2026, 1, 1) (24, 0, 0));
          ("minute 60", fun () -> at (2026, 1, 1) (0, 60, 0));
          ("second 60", fun () -> at (2026, 1, 1) (0, 0, 60));
          ("a negative second", fun () -> at (2026, 1, 1) (0, 0, -1));
          ( "an offset of a day",
            fun () -> Time.of_date ~tz_offset_s:86_400 (2026, 1, 1) );
          ( "an offset of minus a day",
            fun () -> Time.of_date ~tz_offset_s:(-86_400) (2026, 1, 1) );
          ( "a year past int64 seconds",
            fun () -> Time.of_date (292_277_026_597, 1, 1) );
          ( "a year before int64 seconds",
            fun () -> Time.of_date (-292_277_026_597, 1, 1) );
          ("a huge year", fun () -> Time.of_date (max_int, 1, 1));
          (* Years whose day count overflows an [int] to that of a representable
             day. *)
          ( "year 4116195793225948401",
            fun () -> Time.of_date (4116195793225948401, 1, 1) );
          ( "year -1111120336821728399",
            fun () -> Time.of_date (-1111120336821728399, 1, 1) );
        ]
        (fun (_, f) -> invalid f);
      test "to_date_time raises on an offset out of range" (fun () ->
          invalid (fun () -> Time.to_date_time ~tz_offset_s:90_000 Time.epoch));
      test "days from 1900 to 2100 follow each other" (fun () ->
          let y = ref 1900 and m = ref 1 and d = ref 1 in
          let expected = ref (-2_208_988_800) in
          while !y <= 2100 do
            let dt = ((!y, !m, !d), (0, 0, 0)) in
            equal
              ~msg:(Printf.sprintf "%d-%d-%d" !y !m !d)
              instant (utc !expected) (Time.of_date_time dt);
            equal date_time dt (Time.to_date_time (utc !expected));
            expected := !expected + 86_400;
            let leap = (!y mod 4 = 0 && !y mod 100 <> 0) || !y mod 400 = 0 in
            let len =
              [|
                31;
                (if leap then 29 else 28);
                31;
                30;
                31;
                30;
                31;
                31;
                30;
                31;
                30;
                31;
              |].(!m - 1)
            in
            if !d < len then incr d
            else begin
              d := 1;
              if !m < 12 then incr m
              else begin
                m := 1;
                incr y
              end
            end
          done);
      prop "to_date_time inverts of_date_time over the representable range"
        (Gen.pair gen_tz gen_instant) (fun (tz_offset_s, t) ->
          let whole = Time.v S t.Time.sec in
          equal instant whole
            (Time.of_date_time ~tz_offset_s (Time.to_date_time ~tz_offset_s t)));
      prop "of_date_time inverts to_date_time for years within a million"
        (Gen.triple gen_tz
           (Gen.triple
              (Gen.int_range (-1_000_000) 1_000_000)
              (Gen.int_range 1 12) (Gen.int_range 1 31))
           (Gen.triple (Gen.int_range 0 23) (Gen.int_range 0 59)
              (Gen.int_range 0 59)))
        (fun (tz_offset_s, ((y, m, d) as dt), tm) ->
          assume
            (d <= 28 || m <> 2
            || (d = 29 && y mod 4 = 0 && (y mod 100 <> 0 || y mod 400 = 0)));
          assume (d <= 30 || not (List.mem m [ 2; 4; 6; 9; 11 ]));
          let t = Time.of_date_time ~tz_offset_s (dt, tm) in
          equal date_time (dt, tm) (Time.to_date_time ~tz_offset_s t));
    ]

(* Intervals *)

let equalities =
  [
    ("60 seconds are a minute", Time.seconds 60, Time.minutes 1, true);
    ("24 hours are a day", Time.hours 24, Time.days 1, true);
    ("7 days are not a week", Time.days 7, Time.weeks 1, false);
    ("12 months are a year", Time.months 12, Time.years 1, true);
    ("24 months are 2 years", Time.months 24, Time.years 2, true);
    ("18 months are not a year and a half", Time.months 18, Time.years 1, false);
    ("10^9 nanoseconds are a second", ns 1_000_000_000, Time.seconds 1, true);
    ( "1000 microseconds are a millisecond",
      Time.microseconds 1000,
      Time.milliseconds 1,
      true );
    ( "1000 milliseconds are a second",
      Time.milliseconds 1000,
      Time.seconds 1,
      true );
    ("14 days are not 2 weeks", Time.days 14, Time.weeks 2, false);
    ("2 weeks are not a week", Time.weeks 2, Time.weeks 1, false);
  ]

let boundaries ?tz_offset_s i t t' =
  Array.to_list (Time.range ?tz_offset_s i t t')

let intervals =
  group "intervals"
    [
      cases
        ~name:(fun (n, _, _, _) -> n)
        "equal_interval" equalities
        (fun (_, i, i', eq) -> equal bool eq (Time.equal_interval i i'));
      prop "equal_interval is an equivalence"
        (Gen.pair gen_interval gen_interval)
        (Law.equivalence interval);
      cases ~name:fst "constructors raise on"
        [
          ("a stride of 0", fun () -> ignore (Time.seconds 0));
          ("a negative stride", fun () -> ignore (Time.months (-1)));
          ("a year stride of 0", fun () -> ignore (Time.years 0));
          ("a nanosecond stride of 0", fun () -> ignore (ns 0));
          ("53 376 days", fun () -> ignore (Time.days 53_376));
          ("7 626 weeks", fun () -> ignore (Time.weeks 7_626));
          ( "a microsecond stride past 2^62 ns",
            fun () -> ignore (Time.microseconds (max_int / 999)) );
        ]
        (fun (_, f) -> invalid f);
      test "constructors accept strides up to 2^62 ns" (fun () ->
          ignore (Time.days 53_375);
          ignore (Time.weeks 7_625);
          ignore (ns max_int));
      test "pp_interval names units" (fun () ->
          let s i = Format.asprintf "%a" Time.pp_interval i in
          equal (list string)
            [
              "3 months";
              "1 year";
              "1 minute";
              "2 weeks";
              "1 day";
              "1500 milliseconds";
              "7 nanoseconds";
            ]
            [
              s (Time.months 3);
              s (Time.months 12);
              s (Time.seconds 60);
              s (Time.weeks 2);
              s (Time.hours 24);
              s (Time.milliseconds 1500);
              s (ns 7);
            ]);
      test "quarters start in January, April, July and October" (fun () ->
          equal (list instant)
            (List.map (fun m -> Time.of_date (2026, m, 1)) [ 1; 4; 7; 10 ])
            (boundaries (Time.months 3)
               (Time.of_date (2026, 1, 1))
               (Time.of_date (2026, 12, 31))));
      test "six hours start at midnight local time" (fun () ->
          let tz = 19_800 in
          equal (list instant)
            (List.map
               (fun h -> at ~tz_offset_s:tz (2026, 5, 2) (h, 0, 0))
               [ 0; 6; 12; 18 ])
            (boundaries ~tz_offset_s:tz (Time.hours 6)
               (Time.of_date ~tz_offset_s:tz (2026, 5, 2))
               (at ~tz_offset_s:tz (2026, 5, 2) (23, 0, 0))));
      test "two days alternate evenly across month ends" (fun () ->
          equal (list instant)
            (List.map Time.of_date
               [ (2026, 1, 29); (2026, 1, 31); (2026, 2, 2) ])
            (boundaries (Time.days 2)
               (Time.of_date (2026, 1, 29))
               (Time.of_date (2026, 2, 3))));
      test "weeks start on Monday" (fun () ->
          equal instant
            (Time.of_date (2026, 9, 28))
            (Time.floor (Time.weeks 1) (at (2026, 10, 1) (12, 0, 0)));
          equal instant
            (Time.of_date (1969, 12, 29))
            (Time.floor (Time.weeks 2) Time.epoch));
      test "floor reads days at a negative offset before 1970" (fun () ->
          let tz = -18_000 in
          equal instant
            (Time.of_date ~tz_offset_s:tz (1969, 12, 30))
            (Time.floor ~tz_offset_s:tz (Time.days 1)
               (at (1969, 12, 31) (3, 0, 0))));
      test "years of a stride start on its multiples" (fun () ->
          equal instant
            (Time.of_date (2000, 1, 1))
            (Time.floor (Time.years 25) (Time.of_date (2024, 6, 1)));
          equal instant
            (Time.of_date (-100, 1, 1))
            (Time.floor (Time.years 100) (Time.of_date (-1, 6, 1)));
          equal instant
            (Time.of_date (2025, 1, 1))
            (Time.ceil (Time.years 25) (Time.of_date (2024, 6, 1))));
      test "a month after January 31 is the end of February" (fun () ->
          let t = with_ns (at (2026, 1, 31) (10, 20, 30)) 5 in
          equal instant
            (with_ns (at (2026, 2, 28) (10, 20, 30)) 5)
            (Time.add (Time.months 1) 1 t);
          equal instant
            (at (2024, 2, 29) (0, 0, 0))
            (Time.add (Time.months 1) 1 (Time.of_date (2024, 1, 31)));
          equal instant
            (Time.of_date (2025, 2, 28))
            (Time.add (Time.years 1) 1 (Time.of_date (2024, 2, 29))));
      test "add moves fixed intervals by their duration" (fun () ->
          equal instant (utc (-90)) (Time.add (Time.seconds 30) (-3) Time.epoch);
          equal instant (Time.v Ns (-1L)) (Time.add (ns 1) (-1) Time.epoch);
          equal instant
            (Time.v S 2_305_843_009_213_693_952L)
            (Time.add (Time.seconds 1) (1 lsl 61) Time.epoch);
          equal instant
            (with_ns (Time.v S 2_305_843_006_907_850_942L) 786_306_048)
            (Time.add (ns 999_999_999) (1 lsl 61) Time.epoch);
          equal instant
            (with_ns
               (Time.v S (-2_305_843_006_907_850_943L))
               (1_000_000_000 - 786_306_048))
            (Time.add (ns 999_999_999) (-(1 lsl 61)) Time.epoch));
      test "add of zero strides is the identity" (fun () ->
          equal instant (Time.v Ns 7L)
            (Time.add (Time.years 1_000_000_000_000) 0 (Time.v Ns 7L)));
      cases ~name:fst "raise when the result is not representable"
        [
          ( "add past int64 seconds",
            fun () ->
              ignore (Time.add (Time.seconds 1) 1 (Time.v S Int64.max_int)) );
          ( "add before int64 seconds",
            fun () -> ignore (Time.add (ns 1) (-1) (Time.v S Int64.min_int)) );
          ( "add of many years",
            fun () -> ignore (Time.add (Time.years 1) max_int Time.epoch) );
          ( "add of a huge year stride",
            fun () -> ignore (Time.add (Time.years max_int) 1 Time.epoch) );
          ( "add of many days",
            fun () -> ignore (Time.add (Time.days 53_375) max_int Time.epoch) );
          ( "floor below int64 seconds",
            fun () -> ignore (Time.floor (Time.days 1) (Time.v S Int64.min_int))
          );
          ( "ceil above int64 seconds",
            fun () -> ignore (Time.ceil (Time.years 1) (Time.v S Int64.max_int))
          );
          ( "ceil to a year whose day count overflows",
            fun () ->
              ignore (Time.ceil (Time.years 4116195793225948401) Time.epoch) );
          ( "ceil to a month whose day count overflows",
            fun () ->
              ignore (Time.ceil (Time.months 1515164095665993602) Time.epoch) );
        ]
        (fun (_, f) -> invalid f);
      test "range of one boundary is that boundary" (fun () ->
          equal (list instant) [ Time.epoch ]
            (boundaries (Time.days 1) Time.epoch Time.epoch));
      test "ceil skips a month start with nanoseconds" (fun () ->
          equal instant
            (Time.of_date (2026, 4, 1))
            (Time.ceil (Time.months 1) (with_ns (Time.of_date (2026, 3, 1)) 5));
          equal instant
            (Time.of_date (2026, 4, 1))
            (Time.ceil (Time.months 1) (at (2026, 3, 1) (0, 0, 1))));
      test "ceil to years skips a month start" (fun () ->
          equal instant
            (Time.of_date (2027, 1, 1))
            (Time.ceil (Time.years 1) (Time.of_date (2026, 3, 1))));
      test "nanoseconds carry into seconds" (fun () ->
          let t = Time.add (ns 1) 1 (Time.v Ns 999_999_999L) in
          equal int 0 t.Time.nsec;
          equal instant (Time.v S 1L) t);
      test "range is empty when the end is before the start" (fun () ->
          equal int 0
            (Array.length (Time.range (Time.days 1) (utc 86_400) Time.epoch)));
      test "range includes both ends when they are boundaries" (fun () ->
          equal (list instant)
            [ Time.epoch; utc 86_400 ]
            (boundaries (Time.days 1) Time.epoch (utc 86_400)));
      test "range stops at the last representable boundary" (fun () ->
          let last = Time.v S Int64.max_int in
          let r =
            Time.range (Time.seconds 1)
              (Time.v S (Int64.sub Int64.max_int 2L))
              last
          in
          equal int 3 (Array.length r);
          equal instant last r.(2));
    ]

let interval_laws =
  group "interval laws"
    [
      prop "floor i t <= t < add i 1 (floor i t)"
        (Gen.triple gen_tz gen_interval gen_near) (fun (tz_offset_s, i, t) ->
          let f = Time.floor ~tz_offset_s i t in
          at_most instant ~than:t f;
          less instant ~than:(Time.add ~tz_offset_s i 1 f) t);
      prop "ceil i t >= t > add i (-1) (ceil i t)"
        (Gen.triple gen_tz gen_interval gen_near) (fun (tz_offset_s, i, t) ->
          let c = Time.ceil ~tz_offset_s i t in
          at_least instant ~than:t c;
          greater instant ~than:(Time.add ~tz_offset_s i (-1) c) t);
      prop "floor is idempotent" (Gen.triple gen_tz gen_interval gen_near)
        (fun (tz_offset_s, i, t) ->
          let f = Time.floor ~tz_offset_s i in
          equal instant (f t) (f (f t)));
      prop "ceil and floor agree exactly on boundaries"
        (Gen.triple gen_tz gen_interval gen_near) (fun (tz_offset_s, i, t) ->
          let f = Time.floor ~tz_offset_s i t in
          equal instant f (Time.ceil ~tz_offset_s i f);
          equal bool (Time.equal f t)
            (Time.equal t (Time.ceil ~tz_offset_s i t)));
      prop "boundaries are closed under add"
        (Gen.quad gen_tz gen_interval gen_near (Gen.int_range (-50) 50))
        (fun (tz_offset_s, i, t, n) ->
          let b = Time.add ~tz_offset_s i n (Time.floor ~tz_offset_s i t) in
          equal instant b (Time.floor ~tz_offset_s i b));
      prop "range lists consecutive boundaries"
        (Gen.quad gen_tz gen_interval gen_near (Gen.int_range 1 20))
        (fun (tz_offset_s, i, t, n) ->
          let t' = Time.add ~tz_offset_s i n t in
          let r = Time.range ~tz_offset_s i t t' in
          at_least int ~than:n (Array.length r);
          equal instant (Time.ceil ~tz_offset_s i t) r.(0);
          for k = 1 to Array.length r - 1 do
            equal instant (Time.add ~tz_offset_s i 1 r.(k - 1)) r.(k)
          done;
          at_most instant ~than:t' r.(Array.length r - 1);
          less instant
            ~than:(Time.add ~tz_offset_s i 1 r.(Array.length r - 1))
            t');
      prop "add of fixed strides composes"
        (Gen.triple gen_instant
           (Gen.int_range (-1000) 1000)
           (Gen.int_range (-1000) 1000))
        (fun (t, a, b) ->
          let i = ns 999_999_937 in
          match (Time.add i (a + b) t, Time.add i b (Time.add i a t)) with
          | expected, actual -> equal instant expected actual
          | exception Invalid_argument _ -> reject ());
    ]

let printing =
  group "pp"
    [
      cases ~name:snd "writes RFC 3339 at UTC"
        [
          ( with_ns (at (2026, 10, 1) (12, 30, 0)) 250_000_000,
            "2026-10-01T12:30:00.25Z" );
          (Time.epoch, "1970-01-01T00:00:00Z");
          (Time.v Ns (-1L), "1969-12-31T23:59:59.999999999Z");
          (Time.of_date (-1, 1, 1), "-0001-01-01T00:00:00Z");
          (Time.of_date (12_345, 6, 7), "+12345-06-07T00:00:00Z");
          (Time.of_date (0, 1, 1), "0000-01-01T00:00:00Z");
          (Time.of_date (9999, 12, 31), "9999-12-31T00:00:00Z");
          (Time.of_date (10_000, 1, 1), "+10000-01-01T00:00:00Z");
          (Time.v Us 1L, "1970-01-01T00:00:00.000001Z");
        ]
        (fun (t, s) -> equal string s (Format.asprintf "%a" Time.pp t));
    ]

let () =
  exit (run "Time" [ instants; civil; intervals; interval_laws; printing ])
