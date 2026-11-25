(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_kit

let instant =
  Testable.with_compare Time.compare
    (Testable.make ~pp:Time.pp ~equal:Time.equal)

let interval = Testable.make ~pp:Time.pp_interval ~equal:Time.equal_interval
let fields = pair int64 int
let fields_of t = (t.Time.sec, t.Time.nsec)
let date = triple int int int
let date_time = pair date date
let invalid f = raises_match (Exn.invalid_arg ?sub:None) f
let ns k = Time.nanoseconds k
let at ?tz_offset_s d tm = Time.of_date_time ?tz_offset_s (d, tm)
let utc s = Time.v S (Int64.of_int s)
let ns_per_s = 1_000_000_000

(* [with_ns t k] is [t] moved by [k >= 0] nanoseconds. Instants past the range
   of an int64 count of nanoseconds have no other constructor. *)
let with_ns t k = if k = 0 then t else Time.add (ns k) 1 t

(* [fdiv a b] is the floor of [a / b], for [b > 0]. *)
let fdiv a b = if a >= 0 then a / b else -((b - 1 - a) / b)
let sign c = Int.compare c 0
let leap y = (y mod 4 = 0 && y mod 100 <> 0) || y mod 400 = 0

let month_length y = function
  | 2 -> if leap y then 29 else 28
  | 4 | 6 | 9 | 11 -> 30
  | _ -> 31

let next (y, m, d) =
  if d < month_length y m then (y, m, d + 1)
  else if m < 12 then (y, m + 1, 1)
  else (y + 1, 1, 1)

(* Generators *)

let resolutions = [ Time.S; Ms; Us; Ns ]

let units_per_s = function
  | Time.S -> 1
  | Ms -> 1_000
  | Us -> 1_000_000
  | Ns -> ns_per_s

let name_of_resolution = function
  | Time.S -> "S"
  | Ms -> "Ms"
  | Us -> "Us"
  | Ns -> "Ns"

let gen_resolution =
  Gen.of_list
    ~pp:(fun ppf r -> Format.pp_print_string ppf (name_of_resolution r))
    resolutions

(* Four resolutions drawn at 1/4 each: [(3/4)^100] is 3e-13. *)
let cover_resolution r =
  List.iter (fun r' -> cover (name_of_resolution r') (r = r')) resolutions

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

(* Instants within [s] seconds of the epoch. *)
let gen_within s =
  Gen.with_pp Time.pp
    (Gen.map
       (fun (s, n) -> with_ns (Time.v S (Int64.of_int s)) n)
       (Gen.pair (Gen.int_range (-s) s) gen_nsec))

(* Instants whose boundaries are representable: within a few centuries. *)
let gen_near = gen_within 20_000_000_000

let gen_tz =
  Gen.frequency
    [
      ( 2,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 3600; -18000; 19800; 86399; -86399 ] );
      (1, Gen.int_range (-86399) 86399);
    ]

(* The constructors of fixed durations up to days, with the nanoseconds of one
   unit. *)
let durations =
  [
    ("nanoseconds", Time.nanoseconds, 1);
    ("microseconds", Time.microseconds, 1_000);
    ("milliseconds", Time.milliseconds, 1_000_000);
    ("seconds", Time.seconds, ns_per_s);
    ("minutes", Time.minutes, 60 * ns_per_s);
    ("hours", Time.hours, 3600 * ns_per_s);
    ("days", Time.days, 86_400 * ns_per_s);
  ]

let fixed = durations @ [ ("weeks", Time.weeks, 7 * 86_400 * ns_per_s) ]
let pp_name ppf (name, _, _) = Format.pp_print_string ppf name

let cover_name names name =
  List.iter (fun (n, _, _) -> cover n (String.equal n name)) names

(* How the boundaries of a unit are indexed: whole units of [n] nanoseconds
   since 1970-01-01 local time, weeks since Monday 1969-12-29, [12 y + m - 1],
   or the year. *)
type index = Fixed of int | Weeks | Months | Years

(* Each unit with its index, its constructor and the strides drawn. A stride
   that divides no offset of whole half hours, as minutes 7, has boundaries that
   move with the offset. *)
let units =
  [
    ("nanoseconds", Fixed 1, ns, Gen.int_range 1 2_000_000_000);
    ("seconds", Fixed ns_per_s, Time.seconds, Gen.of_list [ 1; 5; 15; 30; 7 ]);
    ( "minutes",
      Fixed (60 * ns_per_s),
      Time.minutes,
      Gen.of_list [ 1; 5; 15; 30; 7 ] );
    ( "hours",
      Fixed (3600 * ns_per_s),
      Time.hours,
      Gen.of_list [ 1; 3; 6; 12; 5 ] );
    ("days", Fixed (86_400 * ns_per_s), Time.days, Gen.of_list [ 1; 2; 3 ]);
    ("weeks", Weeks, Time.weeks, Gen.of_list [ 1; 2 ]);
    ("months", Months, Time.months, Gen.of_list [ 1; 2; 3; 6; 7 ]);
    ("years", Years, Time.years, Gen.of_list [ 1; 2; 5; 10; 100; 1000 ]);
  ]

(* An interval with its unit's name, its index and its stride. *)
let gen_interval =
  Gen.with_pp
    (fun ppf (_, i) -> Time.pp_interval ppf i)
    Gen.(
      let* name, index, make, strides = of_list units in
      map (fun k -> ((name, index, k), make k)) strides)

let fmod a b = a - (b * fdiv a b)

(* [boundary tz (_, index, k) t] is [true] iff [t] starts a unit whose index at
   [tz] is a multiple of [k], by the rules of [Time.interval]. *)
let boundary tz (_, index, k) t =
  let local_s = Int64.to_int t.Time.sec + tz and nsec = t.Time.nsec in
  (* [shift_s] moves the origin of weeks back to Monday 1969-12-29. *)
  let fixed ~shift_s n =
    let period = k * n in
    if period mod ns_per_s = 0 then
      nsec = 0 && fmod (local_s + shift_s) (period / ns_per_s) = 0
    else
      (* [period] is below 2 * 10^9 ns, so the product fits an [int]. *)
      fmod ((fmod (local_s + shift_s) period * ns_per_s) + nsec) period = 0
  in
  let (y, m, d), (hh, mm, ss) = Time.to_date_time ~tz_offset_s:tz t in
  let starts_month = (d, hh, mm, ss, nsec) = (1, 0, 0, 0, 0) in
  match index with
  | Fixed n -> fixed ~shift_s:0 n
  | Weeks -> fixed ~shift_s:(3 * 86_400) (7 * 86_400 * ns_per_s)
  | Months -> starts_month && fmod ((12 * y) + m - 1) k = 0
  | Years -> starts_month && m = 1 && fmod y k = 0

(* Eight units drawn at 1/8 each, at an offset other than 0 with p of about 0.88
   (gen_tz draws 0 at about 1/9): [(1 - 0.11)^200] is 7e-11. *)
let unit_count = 200

let cover_unit name tz =
  List.iter
    (fun (u, _, _, _) ->
      cover (u ^ " at a nonzero offset") (String.equal u name && tz <> 0))
    units

(* Dates within a million years, many in leap years and in February, half of
   them on the last day of a month. *)
let gen_date =
  let open Gen in
  let pp ppf (y, m, d) = Format.fprintf ppf "(%d, %d, %d)" y m d in
  let year = int_range (-1_000_000) 1_000_000 in
  with_pp pp
    (let* y = frequency [ (1, year); (1, map (fun y -> 4 * (y / 4)) year) ] in
     let* m = frequency [ (1, constant 2); (2, int_range 1 12) ] in
     let n = month_length y m in
     let+ d = frequency [ (1, constant n); (1, int_range 1 n) ] in
     (y, m, d))

let gen_time =
  Gen.triple (Gen.int_range 0 23) (Gen.int_range 0 59) (Gen.int_range 0 59)

(* Instants *)

(* [decompose r n] is the instant [n] units of [r] after the epoch, as its whole
   seconds and the nanoseconds after them. *)
let decompose r n =
  let k = Int64.of_int (units_per_s r) in
  let q = Int64.div n k and rem = Int64.rem n k in
  let sec, rem =
    if rem < 0L then (Int64.pred q, Int64.add rem k) else (q, rem)
  in
  (sec, Int64.to_int rem * (ns_per_s / units_per_s r))

let v =
  group "v"
    [
      prop "is the instant n units after the epoch"
        ~examples:[ (Time.S, 1L); (Ms, 1000L) ]
        (Gen.pair gen_resolution gen_sec)
        (fun (r, n) ->
          cover_resolution r;
          equal fields (decompose r n) (fields_of (Time.v r n)));
      test "v Ms (-1L) is { sec = -1L; nsec = 999_000_000 }" (fun () ->
          equal fields (-1L, 999_000_000) (fields_of (Time.v Ms (-1L))));
    ]

let to_int64 =
  group "to_int64"
    [
      prop "inverts v"
        ~examples:[ (Time.Ns, Int64.max_int); (Ns, Int64.min_int) ]
        (Gen.pair gen_resolution gen_sec)
        (fun (r, n) ->
          cover_resolution r;
          Law.round_trip int64 instant (Time.v r)
            (fun t -> require_some (Time.to_int64 r t))
            n);
      (* Within 9 * 10^9 s of the epoch every count fits an int64. *)
      prop "is the greatest count not after the instant"
        ~examples:[ (Time.Ms, Time.v Us 1999L) ]
        (Gen.pair gen_resolution (gen_within 9_000_000_000))
        (fun (r, t) ->
          cover_resolution r;
          let n = require_some (Time.to_int64 r t) in
          at_most instant ~than:t (Time.v r n);
          less instant ~than:(Time.v r (Int64.succ n)) t);
      test "to_int64 S (v Ms (-1L)) is Some (-1L)" (fun () ->
          equal (option int64) (Some (-1L)) (Time.to_int64 S (Time.v Ms (-1L))));
      cases "is None when the count does not fit an int64" ~id:fst
        [
          ("nanoseconds of 10^10 s", (Time.Ns, Time.v S 10_000_000_000L));
          ("milliseconds of the least second", (Ms, Time.v S Int64.min_int));
          ( "nanoseconds 1 ns after the greatest count",
            (Ns, Time.add (ns 1) 1 (Time.v Ns Int64.max_int)) );
          ( "nanoseconds 1 ns before the least count",
            (Ns, Time.add (ns 1) (-1) (Time.v Ns Int64.min_int)) );
        ]
        (fun (_, (r, t)) -> equal (option int64) None (Time.to_int64 r t));
    ]

let epoch =
  group "epoch"
    [
      test "is 1970-01-01 00:00:00 UTC" (fun () ->
          equal instant (at (1970, 1, 1) (0, 0, 0)) Time.epoch);
      test "is the zero count" (fun () ->
          equal fields (0L, 0) (fields_of Time.epoch));
    ]

(* Civil dates and times *)

let tz_offset_s =
  let functions =
    [
      ( "of_date_time",
        fun tz -> ignore (at ~tz_offset_s:tz (1970, 1, 1) (0, 0, 0)) );
      ("of_date", fun tz -> ignore (Time.of_date ~tz_offset_s:tz (1970, 1, 1)));
      ( "to_date_time",
        fun tz -> ignore (Time.to_date_time ~tz_offset_s:tz Time.epoch) );
      ( "floor",
        fun tz -> ignore (Time.floor ~tz_offset_s:tz (Time.days 1) Time.epoch)
      );
      ( "ceil",
        fun tz -> ignore (Time.ceil ~tz_offset_s:tz (Time.days 1) Time.epoch) );
      ( "add",
        fun tz -> ignore (Time.add ~tz_offset_s:tz (Time.months 1) 1 Time.epoch)
      );
      ( "range",
        fun tz ->
          ignore
            (Time.range ~tz_offset_s:tz (Time.days 1) Time.epoch (utc 86_400))
      );
    ]
  in
  group "tz_offset_s"
    [
      cases "raises in every function outside ]-86 400;86 400["
        ~id:(fun ((name, _), tz) -> Printf.sprintf "%s at %d" name tz)
        (List.concat_map (fun f -> [ (f, 86_400); (f, -86_400) ]) functions)
        (fun ((_, f), tz) -> invalid (fun () -> f tz));
    ]

(* Seconds since the epoch as GNU date reads them: [date -u -d @s]. *)
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

let of_date_time =
  group "of_date_time"
    [
      cases "is the instant of a civil date and time at UTC"
        ~id:(fun ((y, m, d), (hh, mm, ss), _) ->
          Printf.sprintf "%d-%02d-%02d %02d:%02d:%02d" y m d hh mm ss)
        known
        (fun (d, tm, s) -> equal instant (utc s) (at d tm));
      prop "is the instant at UTC moved back by the offset"
        ~examples:
          [
            (3600, (1970, 1, 1), (1, 0, 0)); (-18_000, (1970, 1, 1), (0, 0, 0));
          ]
        (Gen.triple gen_tz gen_date gen_time)
        (fun (tz_offset_s, d, tm) ->
          let t = at d tm in
          equal fields
            (Int64.sub t.Time.sec (Int64.of_int tz_offset_s), 0)
            (fields_of (at ~tz_offset_s d tm)));
      prop "is read back by to_date_time" (Gen.triple gen_tz gen_date gen_time)
        (fun (tz_offset_s, d, tm) ->
          Law.round_trip date_time instant
            (Time.of_date_time ~tz_offset_s)
            (Time.to_date_time ~tz_offset_s)
            (d, tm));
      cases "raises on" ~id:fst
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
    ]

let of_date =
  group "of_date"
    [
      prop "is of_date_time at 00:00:00" (Gen.pair gen_tz gen_date)
        (fun (tz_offset_s, d) ->
          equal instant
            (at ~tz_offset_s d (0, 0, 0))
            (Time.of_date ~tz_offset_s d));
      (* February 29 at p of about 0.12 and December 31 at about 0.03 over 100
         cases: each is met by an example. *)
      prop "the day after a date starts 86 400 s later"
        ~examples:
          [ (1900, 2, 28); (2000, 2, 29); (2100, 2, 28); (1969, 12, 31) ]
        gen_date
        (fun d ->
          cover "February 29" (match d with _, 2, 29 -> true | _ -> false);
          cover "December 31" (match d with _, 12, 31 -> true | _ -> false);
          let start = Time.of_date d in
          equal instant
            (Time.v S (Int64.add start.Time.sec 86_400L))
            (Time.of_date (next d)));
    ]

let to_date_time =
  group "to_date_time"
    [
      test "drops the nanoseconds" (fun () ->
          equal date_time
            ((1969, 12, 31), (23, 59, 59))
            (Time.to_date_time (Time.v Ms (-1L))));
      test "reads local time at an offset west of UTC" (fun () ->
          equal date_time
            ((1969, 12, 31), (19, 0, 0))
            (Time.to_date_time ~tz_offset_s:(-18_000) Time.epoch));
      prop "is read back by of_date_time without the nanoseconds"
        ~examples:
          [
            (0, Time.epoch);
            (0, Time.v S Int64.max_int);
            (0, Time.v S Int64.min_int);
          ]
        (Gen.pair gen_tz gen_instant)
        (fun (tz_offset_s, t) ->
          cover "a whole second" (t.Time.nsec = 0);
          cover "a fraction of a second" (t.Time.nsec <> 0);
          equal fields (t.Time.sec, 0)
            (fields_of
               (Time.of_date_time ~tz_offset_s
                  (Time.to_date_time ~tz_offset_s t))));
    ]

(* Intervals *)

let boundaries ?tz_offset_s i t t' =
  Array.to_list (Time.range ?tz_offset_s i t t')

let intervals =
  group "interval"
    [
      test "months 3 start January, April, July and October" (fun () ->
          equal (list instant)
            (List.map (fun m -> Time.of_date (2026, m, 1)) [ 1; 4; 7; 10 ])
            (boundaries (Time.months 3)
               (Time.of_date (2026, 1, 1))
               (Time.of_date (2026, 12, 31))));
      test "hours 6 start at 00:00, 06:00, 12:00 and 18:00 local time"
        (fun () ->
          let tz = 19_800 in
          equal (list instant)
            (List.map
               (fun h -> at ~tz_offset_s:tz (2026, 5, 2) (h, 0, 0))
               [ 0; 6; 12; 18 ])
            (boundaries ~tz_offset_s:tz (Time.hours 6)
               (Time.of_date ~tz_offset_s:tz (2026, 5, 2))
               (at ~tz_offset_s:tz (2026, 5, 2) (23, 0, 0))));
      test "days 2 start every other day across the ends of months" (fun () ->
          equal (list instant)
            (List.map Time.of_date
               [ (2026, 1, 29); (2026, 1, 31); (2026, 2, 2) ])
            (boundaries (Time.days 2)
               (Time.of_date (2026, 1, 29))
               (Time.of_date (2026, 2, 3))));
      test "days count from midnight local time" (fun () ->
          let tz = -18_000 in
          equal instant
            (Time.of_date ~tz_offset_s:tz (1969, 12, 30))
            (Time.floor ~tz_offset_s:tz (Time.days 1)
               (at (1969, 12, 31) (3, 0, 0))));
      test "weeks start on Monday" (fun () ->
          equal instant
            (Time.of_date (2026, 9, 28))
            (Time.floor (Time.weeks 1) (at (2026, 10, 1) (12, 0, 0))));
      test "weeks count from Monday 1969-12-29" (fun () ->
          equal instant
            (Time.of_date (1969, 12, 29))
            (Time.floor (Time.weeks 2) Time.epoch));
      cases "years of a stride start on its multiples"
        ~id:(fun (k, (y, _, _), _) -> Printf.sprintf "years %d in %d" k y)
        [ (25, (2024, 6, 1), (2000, 1, 1)); (100, (-1, 6, 1), (-100, 1, 1)) ]
        (fun (k, d, start) ->
          equal instant (Time.of_date start)
            (Time.floor (Time.years k) (Time.of_date d)));
      (* Seven units drawn at 1/7 each: [(6/7)^150] is 1e-10. *)
      prop "a fixed unit up to days is nanoseconds of its duration" ~count:150
        (Gen.pair (Gen.of_list ~pp:pp_name durations) (Gen.int_range 1 1000))
        (fun ((name, f, unit), k) ->
          cover_name durations name;
          equal interval (ns (unit * k)) (f k));
      cases "constructors raise on a stride below 1" ~id:fst
        [
          ("nanoseconds 0", fun () -> ns 0);
          ("microseconds 0", fun () -> Time.microseconds 0);
          ("milliseconds 0", fun () -> Time.milliseconds 0);
          ("seconds 0", fun () -> Time.seconds 0);
          ("minutes 0", fun () -> Time.minutes 0);
          ("hours 0", fun () -> Time.hours 0);
          ("days 0", fun () -> Time.days 0);
          ("weeks 0", fun () -> Time.weeks 0);
          ("months 0", fun () -> Time.months 0);
          ("years 0", fun () -> Time.years 0);
          ("months -1", fun () -> Time.months (-1));
          ("seconds min_int", fun () -> Time.seconds min_int);
        ]
        (fun (_, f) -> invalid f);
      (* The largest stride of each unit within 2^62 ns is floor (2^62 / unit),
         as no unit but the nanosecond divides 2^62. The nanosecond's is
         max_int, 2^62 - 1, since 2^62 is not an [int]. *)
      cases "constructors take a stride of up to 2^62 ns" ~id:fst
        [
          ("nanoseconds", fun () -> ns 4_611_686_018_427_387_903);
          ("microseconds", fun () -> Time.microseconds 4_611_686_018_427_387);
          ("milliseconds", fun () -> Time.milliseconds 4_611_686_018_427);
          ("seconds", fun () -> Time.seconds 4_611_686_018);
          ("minutes", fun () -> Time.minutes 76_861_433);
          ("hours", fun () -> Time.hours 1_281_023);
          ("days", fun () -> Time.days 53_375);
          ("weeks", fun () -> Time.weeks 7_625);
        ]
        (fun (_, f) -> ignore (f ()));
      cases "constructors raise on a stride past 2^62 ns" ~id:fst
        [
          ("microseconds", fun () -> Time.microseconds 4_611_686_018_427_388);
          ("milliseconds", fun () -> Time.milliseconds 4_611_686_018_428);
          ("seconds", fun () -> Time.seconds 4_611_686_019);
          ("minutes", fun () -> Time.minutes 76_861_434);
          ("hours", fun () -> Time.hours 1_281_024);
          ("days", fun () -> Time.days 53_376);
          ("weeks", fun () -> Time.weeks 7_626);
        ]
        (fun (_, f) -> invalid f);
    ]

let floor =
  group "floor"
    [
      prop "is the latest boundary not after t" ~count:unit_count
        (Gen.triple gen_tz gen_interval gen_near)
        (fun (tz_offset_s, (u, i), t) ->
          let name, _, _ = u in
          cover_unit name tz_offset_s;
          let f = Time.floor ~tz_offset_s i t in
          satisfies instant ~claim:"a boundary" (boundary tz_offset_s u) f;
          at_most instant ~than:t f;
          less instant ~than:(Time.add ~tz_offset_s i 1 f) t);
      test "raises when the boundary is not representable" (fun () ->
          invalid (fun () -> Time.floor (Time.days 1) (Time.v S Int64.min_int)));
    ]

let ceil =
  let month = ("months", Months, 1) in
  group "ceil"
    [
      prop "is the earliest boundary not before t" ~count:unit_count
        ~examples:
          [
            (0, (month, Time.months 1), Time.of_date (2026, 3, 1));
            (0, (month, Time.months 1), with_ns (Time.of_date (2026, 3, 1)) 5);
            (0, (month, Time.months 1), at (2026, 3, 1) (0, 0, 1));
            (0, (("years", Years, 1), Time.years 1), Time.of_date (2026, 3, 1));
          ]
        (Gen.triple gen_tz gen_interval gen_near)
        (fun (tz_offset_s, (u, i), t) ->
          let name, _, _ = u in
          cover_unit name tz_offset_s;
          cover "t is a boundary" (boundary tz_offset_s u t);
          let c = Time.ceil ~tz_offset_s i t in
          satisfies instant ~claim:"a boundary" (boundary tz_offset_s u) c;
          at_least instant ~than:t c;
          greater instant ~than:(Time.add ~tz_offset_s i (-1) c) t);
      cases "raises when the boundary is not representable" ~id:fst
        [
          ( "years 1 of the greatest instant",
            fun () -> Time.ceil (Time.years 1) (Time.v S Int64.max_int) );
          (* Strides whose day count overflows an [int]. *)
          ( "years 4116195793225948401",
            fun () -> Time.ceil (Time.years 4116195793225948401) Time.epoch );
          ( "months 1515164095665993602",
            fun () -> Time.ceil (Time.months 1515164095665993602) Time.epoch );
        ]
        (fun (_, f) -> invalid f);
    ]

(* [moved t d n] is [t] moved by [n] times [d] nanoseconds, as its whole seconds
   and the nanoseconds after them, for [|n| <= 1000] and [d] up to 2^62. *)
let moved t d n =
  let ns = t.Time.nsec + (n * (d mod ns_per_s)) in
  let s = (n * (d / ns_per_s)) + fdiv ns ns_per_s in
  (Int64.add t.Time.sec (Int64.of_int s), ns - (ns_per_s * fdiv ns ns_per_s))

let add =
  group "add"
    [
      (* Eight units drawn at 1/8 each: [(7/8)^200] is 3e-12. *)
      prop "moves t by a fixed duration up to weeks" ~count:unit_count
        ~examples:[ (Time.v Ns 999_999_999L, (List.hd fixed, 1), 1) ]
        (Gen.triple gen_near
           (Gen.pair (Gen.of_list ~pp:pp_name fixed) (Gen.int_range 1 1000))
           (Gen.int_range (-1000) 1000))
        (fun (t, ((name, f, unit), k), n) ->
          cover_name fixed name;
          let carry = t.Time.nsec + (n * (unit * k mod ns_per_s)) in
          cover "a carry across seconds" (carry < 0 || carry >= ns_per_s);
          equal fields (moved t (unit * k) n) (fields_of (Time.add (f k) n t)));
      (* [2^61 (10^9 - 1)] ns is 2 305 843 006 907 850 942 s and 786 306 048
         ns. *)
      cases "moves by 2^61 strides of a fixed unit"
        ~id:(fun (name, _, _) -> name)
        [
          ( "2^61 seconds",
            Time.v S 2_305_843_009_213_693_952L,
            fun () -> Time.add (Time.seconds 1) (1 lsl 61) Time.epoch );
          ( "2^61 strides of 10^9 - 1 ns",
            with_ns (Time.v S 2_305_843_006_907_850_942L) 786_306_048,
            fun () -> Time.add (ns 999_999_999) (1 lsl 61) Time.epoch );
          ( "-2^61 strides of 10^9 - 1 ns",
            with_ns
              (Time.v S (-2_305_843_006_907_850_943L))
              (1_000_000_000 - 786_306_048),
            fun () -> Time.add (ns 999_999_999) (-(1 lsl 61)) Time.epoch );
        ]
        (fun (_, expected, f) -> equal instant expected (f ()));
      test "reaches the last nanosecond of the greatest second" (fun () ->
          equal fields
            (Int64.max_int, 999_999_999)
            (fields_of (Time.add (ns 999_999_999) 1 (Time.v S Int64.max_int))));
      (* The day clamps at p of about 0.02: the examples reach it. *)
      prop "moves the civil date by months and keeps the time of day"
        ~examples:
          [
            (0, ((1, 1), 1), with_ns (at (2026, 1, 31) (10, 20, 30)) 5);
            (0, ((1, 1), 1), Time.of_date (2024, 1, 31));
            (0, ((1, 12), 1), Time.of_date (2024, 2, 29));
          ]
        (Gen.triple gen_tz
           (Gen.pair
              (Gen.pair (Gen.int_range 1 13)
                 (Gen.of_list ~pp:Format.pp_print_int [ 1; 12 ]))
              (Gen.int_range (-50) 50))
           gen_near)
        (fun (tz_offset_s, ((k, months), n), t) ->
          (* [months] is the months of a unit: 1 for months, 12 for years. *)
          let i = if months = 1 then Time.months k else Time.years k in
          let (y, m, d), tm = Time.to_date_time ~tz_offset_s t in
          let j = (12 * y) + m - 1 + (k * months * n) in
          let y' = fdiv j 12 and m' = j - (12 * fdiv j 12) + 1 in
          let d' = Int.min d (month_length y' m') in
          cover "years" (months = 12);
          cover "the day clamps" (d' < d);
          equal fields
            ((at ~tz_offset_s (y', m', d') tm).Time.sec, t.Time.nsec)
            (fields_of (Time.add ~tz_offset_s i n t)));
      test "of no stride is t" (fun () ->
          equal instant (Time.v Ns 7L)
            (Time.add (Time.years 1_000_000_000_000) 0 (Time.v Ns 7L)));
      prop "keeps the boundaries" ~count:unit_count
        (Gen.quad gen_tz gen_interval gen_near (Gen.int_range (-50) 50))
        (fun (tz_offset_s, (u, i), t, n) ->
          let name, _, _ = u in
          cover_unit name tz_offset_s;
          let b = Time.add ~tz_offset_s i n (Time.floor ~tz_offset_s i t) in
          satisfies instant ~claim:"a boundary" (boundary tz_offset_s u) b);
      prop "composes on boundaries" ~count:unit_count
        (Gen.triple gen_tz gen_interval
           (Gen.triple gen_near
              (Gen.int_range (-1000) 1000)
              (Gen.int_range (-1000) 1000)))
        (fun (tz_offset_s, (u, i), (t, a, b)) ->
          let name, _, _ = u in
          cover_unit name tz_offset_s;
          let add n t = Time.add ~tz_offset_s i n t in
          let t = Time.floor ~tz_offset_s i t in
          let sum = add (a + b) t in
          satisfies instant ~claim:"a boundary" (boundary tz_offset_s u) sum;
          equal instant sum (add b (add a t)));
      cases "raises when the result is not representable" ~id:fst
        [
          ( "past int64 seconds",
            fun () -> Time.add (Time.seconds 1) 1 (Time.v S Int64.max_int) );
          ( "1 ns past the greatest instant",
            fun () ->
              Time.add (ns 1) 1
                (Time.add (ns 999_999_999) 1 (Time.v S Int64.max_int)) );
          ( "before int64 seconds",
            fun () -> Time.add (ns 1) (-1) (Time.v S Int64.min_int) );
          ("many years", fun () -> Time.add (Time.years 1) max_int Time.epoch);
          ( "a huge year stride",
            fun () -> Time.add (Time.years max_int) 1 Time.epoch );
          ("many days", fun () -> Time.add (Time.days 53_375) max_int Time.epoch);
        ]
        (fun (_, f) -> invalid f);
    ]

let range =
  group "range"
    [
      prop "is the boundaries in [t;t'] in order" ~count:unit_count
        (Gen.quad gen_tz gen_interval gen_near (Gen.int_range 1 20))
        (fun (tz_offset_s, ((name, _, _), i), t, n) ->
          cover_unit name tz_offset_s;
          let t' = Time.add ~tz_offset_s i n t in
          let rec from b =
            if Time.compare b t' > 0 then []
            else b :: from (Time.add ~tz_offset_s i 1 b)
          in
          equal (list instant)
            (from (Time.ceil ~tz_offset_s i t))
            (boundaries ~tz_offset_s i t t'));
      test "of one boundary is that boundary" (fun () ->
          equal (list instant) [ Time.epoch ]
            (boundaries (Time.days 1) Time.epoch Time.epoch));
      test "includes both ends when they are boundaries" (fun () ->
          equal (list instant)
            [ Time.epoch; utc 86_400 ]
            (boundaries (Time.days 1) Time.epoch (utc 86_400)));
      test "is empty when t' is before t" (fun () ->
          equal (list instant) []
            (boundaries (Time.days 1) (utc 86_400) Time.epoch));
      test "stops at the last representable boundary" (fun () ->
          let last = Int64.max_int in
          equal (list instant)
            (List.map (fun d -> Time.v S (Int64.sub last d)) [ 2L; 1L; 0L ])
            (boundaries (Time.seconds 1)
               (Time.v S (Int64.sub last 2L))
               (Time.v S last)));
    ]

let equal_interval =
  group "equal_interval"
    [
      cases "holds for intervals with the same boundaries" ~id:fst
        [
          ("seconds 60 and minutes 1", (Time.seconds 60, Time.minutes 1));
          ("hours 24 and days 1", (Time.hours 24, Time.days 1));
          ("months 12 and years 1", (Time.months 12, Time.years 1));
          ("months 24 and years 2", (Time.months 24, Time.years 2));
        ]
        (fun (_, (i, i')) -> equal interval i i');
      cases "fails for intervals with other boundaries" ~id:fst
        [
          ("days 7 and weeks 1", (Time.days 7, Time.weeks 1));
          ("days 14 and weeks 2", (Time.days 14, Time.weeks 2));
          ("weeks 2 and weeks 1", (Time.weeks 2, Time.weeks 1));
          ("months 18 and years 1", (Time.months 18, Time.years 1));
        ]
        (fun (_, (i, i')) -> not_equal interval i i');
      prop "is an equivalence"
        (let i = Gen.with_pp Time.pp_interval (Gen.map snd gen_interval) in
         Gen.pair i i)
        (Law.equivalence interval);
    ]

let pp_interval =
  group "pp_interval"
    [
      test "writes months 3 as 3 months" (fun () ->
          equal string "3 months"
            (Format.asprintf "%a" Time.pp_interval (Time.months 3)));
    ]

(* Comparing and formatting *)

let compare =
  group "compare"
    [
      (* Equal pairs come at p of about 1/8: the example reaches them. *)
      prop "orders instants by their seconds, then nanoseconds"
        ~examples:[ (Time.epoch, Time.v Ms 0L) ]
        (Gen.pair gen_instant gen_instant)
        (fun (t, t') ->
          cover "the same instant" (fields_of t = fields_of t');
          equal int
            (sign (Stdlib.compare (fields_of t) (fields_of t')))
            (sign (Time.compare t t')));
      prop "is a total order compatible with equal"
        (Gen.triple gen_instant gen_instant gen_instant)
        (Law.order instant);
    ]

let pp =
  group "pp"
    [
      (* The dates of the extremes are Hinnant's [civil_from_days] of their day
         counts, computed in Python's integers. *)
      cases "writes RFC 3339 at UTC" ~id:snd
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
          (Time.v S Int64.max_int, "+292277026596-12-04T15:30:07Z");
          (Time.v S Int64.min_int, "-292277022657-01-27T08:29:52Z");
        ]
        (fun (t, s) -> equal string s (Format.asprintf "%a" Time.pp t));
    ]

let () =
  exit
    (run "Time"
       [
         v;
         to_int64;
         epoch;
         tz_offset_s;
         of_date_time;
         of_date;
         to_date_time;
         intervals;
         floor;
         ceil;
         add;
         range;
         equal_interval;
         pp_interval;
         compare;
         pp;
       ])
